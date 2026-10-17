"""Experimental, fully local Ollama video observer. Never calls a cloud model."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener
import uuid

from analyzers import Analyzer, AnalysisError, validate_result


ROOT = Path(__file__).resolve().parent
CODES = {
    "personal_items": "Личные вещи в кассовой зоне",
    "cash_storage": "Возможное нарушение хранения наличных",
    "receive_counting": "Вопрос к пересчёту при приёме наличных",
    "issue_counting": "Вопрос к пересчёту при выдаче наличных",
    "cash_document": "Вопрос к оформлению кассового документа",
    "receipt": "Вопрос к выдаче подтверждающего документа",
    "other": "Действие требует дополнительной проверки",
}

OBSERVATION_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "maxLength": 150},
        "events": {
            "type": "array",
            "maxItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "frame_index": {"type": "integer"},
                    "observation": {"type": "string", "maxLength": 150},
                    "reason": {"type": "string", "maxLength": 150},
                    "rule_id": {"type": "string", "enum": ["", "nb231_12", "nb231_20", "nb231_19_2", "nb231_19_5", "nb231_19_6", "nb231_31_4", "nb231_31_5", "nb231_31_6"]},
                    "certainty": {"type": "string", "enum": ["clear", "uncertain"]},
                },
                "required": ["frame_index", "observation", "reason", "rule_id", "certainty"],
                "additionalProperties": False,
            },
        },
        "limitations": {"type": "string", "maxLength": 150},
    },
    "required": ["summary", "events", "limitations"],
    "additionalProperties": False,
}


class OllamaVideoAnalyzer(Analyzer):
    connected = True
    experimental = True

    def __init__(self, *, model=None, endpoint=None):
        self.model = model or os.environ.get("KFM_OLLAMA_MODEL", "qwen3-vl:2b-instruct")
        self.endpoint = (endpoint or os.environ.get("KFM_OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
        parsed = urlsplit(self.endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Only a local Ollama endpoint is allowed")
        if ":cloud" in self.model or self.model.endswith("-cloud"):
            raise ValueError("Cloud models are not allowed")
        self.name = f"{self.model} · экспериментальный анализ"
        self.ffmpeg = Path(os.environ.get("KFM_FFMPEG", ROOT / ".tools" / "ffmpeg.exe"))
        self.opener = build_opener(ProxyHandler({}))
        rules_file = ROOT / "rules.json"
        self.rules = json.loads(rules_file.read_text(encoding="utf-8")) if rules_file.exists() else []
        self.by_id = {rule["id"]: rule for rule in self.rules}
        self.scratch = ROOT / "data" / "analysis"
        self.scratch.mkdir(parents=True, exist_ok=True)

    def request(self, path, body=None, timeout=10):
        request = Request(self.endpoint + path, None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8"), {"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=timeout) as response:
                payload = response.read(4 * 1024 * 1024)
            return json.loads(payload)
        except HTTPError as exc:
            # Never include a prompt or a model's scene description in a log.
            raise AnalysisError("local_model_error", f"Локальная модель вернула HTTP {exc.code}.") from None
        except (OSError, ValueError) as exc:
            raise AnalysisError("local_model_unavailable", "Локальная модель недоступна или превысила время ожидания.") from exc

    def ready(self):
        try:
            tags = self.request("/api/tags")
            return any(x.get("name") == self.model for x in tags.get("models", [])) and self.ffmpeg.is_file()
        except AnalysisError:
            return False

    def extract_frames(self, path: Path, timestamps: list[float], folder: Path, edge=448):
        if not self.ffmpeg.is_file():
            raise AnalysisError("decoder_missing", "Не найден локальный FFmpeg для чтения видео.")
        frames = []
        for index, stamp in enumerate(timestamps):
            destination = folder / f"frame-{index:03d}.jpg"
            command = [str(self.ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-protocol_whitelist", "file,pipe", "-ss", f"{stamp:.3f}", "-i", str(path), "-frames:v", "1", "-vf", f"scale={edge}:{edge}:force_original_aspect_ratio=decrease", "-q:v", "3", "-threads", "1", "-y", str(destination)]
            result = subprocess.run(command, capture_output=True, timeout=25, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode or not destination.exists():
                raise AnalysisError("video_decode_error", "Не удалось извлечь кадры из видео. Проверьте формат записи.")
            frames.append(destination)
        return frames

    def observe(self, path: Path, timestamps: list[float], *, folder=None, timeout=120):
        own_folder = folder is None
        folder = Path(folder) if folder else self.scratch / uuid.uuid4().hex
        folder.mkdir(parents=True, exist_ok=True)
        try:
            frames = self.extract_frames(path, timestamps, folder)
            rules = "\n".join(f"{r['id']}: {r['condition']} Exception/limit: {r['limitation']}" for r in self.rules)
            instructions = (
                "You inspect staged bank cashier training videos. The images are consecutive sampled frames, not the complete recording. "
                "Describe only directly visible actions. Do not identify faces or people, do not read personal data. "
                "Do not infer transaction amounts, hidden accounting, authorization, intent or theft. "
                "Absence in sampled frames does NOT prove that counting, signatures or a receipt never happened. "
                "A cash counting machine is permitted; sealed intact banknote bundles have exceptions to recounting. "
                "People near a cashier are not automatically unauthorized. A telephone or personal bag may be relevant only if clearly visible in the cashier zone. "
                "Return JSON matching the schema. summary, observation, reason and limitations must be in Russian. "
                "An event needs a precise visible observation. certainty=clear only if both the action and applicable condition are visible. "
                "For unclear legal applicability use certainty=uncertain and rule_id=''. If nothing relevant is visible use events=[]. "
                "Do not invent documents or paragraph numbers. Valid rule IDs and conditions:\n" + rules +
                "\nframe_index is ZERO-BASED and must identify the best image showing the action. "
                "Select rule_id only from the schema's IDs; use an empty string if no rule applies. "
                "Keep the response VERY SHORT: at most 1 event; summary, observation, reason and limitations each at most 12 Russian words."
            )
            started = time.monotonic()
            result = {"summary": "", "events": [], "limitations": ""}
            summaries, limitations = [], []
            # Separate image embeddings have a minimum token cost. Two images
            # per request keep this within the 4K context on a 4 GB GPU.
            for offset in range(0, len(frames), 2):
                remaining = timeout - (time.monotonic() - started)
                if remaining < 3:
                    raise AnalysisError("analysis_timeout", "Не удалось завершить все группы кадров за отведённое время.")
                group = frames[offset:offset+2]
                stamps = timestamps[offset:offset+2]
                stamp_text = ", ".join(f"image {i} = {t:.2f}s" for i, t in enumerate(stamps))
                prompt = instructions + "\nImage order and timestamps: " + stamp_text
                response = self.request("/api/chat", {
                    "model": self.model, "stream": False, "think": False, "format": OBSERVATION_SCHEMA,
                    "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(frame.read_bytes()).decode("ascii") for frame in group]}],
                    "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 700},
                    "keep_alive": "10m",
                }, timeout=remaining)
                metrics = {key: response.get(key) for key in ("done_reason", "prompt_eval_count", "eval_count", "total_duration")}
                metrics["content_length"] = len(response.get("message", {}).get("content", ""))
                metrics_path = ROOT / ".tools" / "model_metrics.jsonl"
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with metrics_path.open("a", encoding="utf-8") as metrics_file:
                    metrics_file.write(json.dumps(metrics) + "\n")
                try:
                    part = json.loads(response.get("message", {}).get("content", ""))
                    if not isinstance(part.get("events"), list) or not isinstance(part.get("summary"), str):
                        raise ValueError()
                except (ValueError, TypeError, AttributeError):
                    raise AnalysisError("invalid_observation", "Локальная модель не сформировала законченный ответ.") from None
                summaries.append(part["summary"])
                limitations.append(str(part.get("limitations", "")))
                for event in part["events"]:
                    if not (isinstance(event, dict) and isinstance(event.get("frame_index"), int) and not isinstance(event.get("frame_index"), bool) and 0 <= event["frame_index"] < len(group)):
                        raise AnalysisError("invalid_observation", "Модель указала кадр, которого нет в проверяемом фрагменте.")
                    # Derive the category from the verified rule instead of
                    # asking a small model to repeat that mapping correctly.
                    rule = self.by_id.get(event.get("rule_id"))
                    event["code"] = rule["code"] if rule else "other"
                    event["frame_index"] += offset
                    result["events"].append(event)
            result["summary"] = " ".join(dict.fromkeys(summaries))
            result["limitations"] = " ".join(dict.fromkeys(value for value in limitations if value))
            result["timestamps"] = timestamps
            result["elapsed_sec"] = round(time.monotonic() - started, 2)
            result["frames"] = [str(frame.resolve()) for frame in frames]
            return result
        finally:
            # Caller-owned dataset evidence stays local for its report.
            if own_folder and folder.resolve().is_relative_to(self.scratch.resolve()):
                shutil.rmtree(folder)

    def analyze(self, video_path: Path, metadata: dict, progress) -> dict:
        if not self.ready():
            raise AnalysisError("local_model_unavailable", "Не запущена локальная модель или отсутствует декодер видео.")
        duration = metadata.get("duration_sec")
        if not duration or duration <= 0:
            raise AnalysisError("unknown_video_duration", "Не удалось определить длительность видео.")
        started = time.monotonic()
        deadline = started + 255
        window = 12.0
        starts = []
        start = 0
        while start < duration:
            starts.append(start)
            if start + window >= duration:
                break
            start += 10
        events = []
        uncertain = 0
        work = self.scratch / uuid.uuid4().hex
        work.mkdir(parents=True)
        try:
            for index, start in enumerate(starts):
                remaining = deadline - time.monotonic()
                if remaining < 15:
                    raise AnalysisError("analysis_timeout", "Не все фрагменты обработаны за отведённое время. Неполная проверка не считается отсутствием нарушений.")
                end = min(duration - 0.15, start + window)
                if end < start:
                    continue
                timestamps = sorted({round(start + (end - start) * fraction, 3) for fraction in (0, .25, .5, .75, 1)})
                progress(10 + int(index / len(starts) * 75), f"Анализ фрагмента {index + 1} из {len(starts)}")
                folder = work / f"window-{index:03d}"
                observed = self.observe(video_path, timestamps, folder=folder, timeout=max(10, remaining - 3))
                for event in observed["events"][:8]:
                    code = event.get("code")
                    rule = self.by_id.get(event.get("rule_id"))
                    if code not in CODES or event.get("certainty") != "clear" or not rule or rule.get("code") != code:
                        uncertain += 1
                        continue
                    frame_index = event.get("frame_index")
                    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or not 0 <= frame_index < len(timestamps):
                        continue
                    stamp = timestamps[frame_index]
                    if any(old["type"] == CODES[code] and abs(old["timecode_sec"] - stamp) < 12 for old in events):
                        continue
                    image = Path(observed["frames"][frame_index]).read_bytes()
                    events.append({
                        "type": CODES[code], "timecode_sec": stamp,
                        "description": f"Наблюдение: {event.get('observation', '')} Возможный признак: {event.get('reason', '')} Основание для проверки: {rule['title']}, {rule['section']}. Вывод экспериментальной модели требует проверки человеком.",
                        "evidence_frames": [{"timestamp_sec": stamp, "image_base64": base64.b64encode(image).decode("ascii")}],
                        "rule": {"title": rule["title"], "section": rule["section"], "url": rule["url"]},
                    })
            progress(95, "Подготовка кадров и отчёта")
            if not events and uncertain:
                raise AnalysisError("uncertain_observations", "Модель заметила неоднозначные действия, но не смогла надёжно связать их с проверенными правилами. Вердикт об отсутствии нарушений не сформирован.")
            return validate_result({"verdict": "violation" if events else "no_violation", "violations": events}, duration)
        finally:
            if work.resolve().is_relative_to(self.scratch.resolve()):
                shutil.rmtree(work)
