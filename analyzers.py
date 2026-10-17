"""The single integration point between KFM and a future local video model."""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


class AnalysisError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class Analyzer:
    name = "Модель не подключена"
    connected = False

    def analyze(self, video_path: Path, metadata: dict, progress) -> dict:
        raise AnalysisError(
            "model_not_configured",
            "Видео загружено. Подключите локальную модель, чтобы получить результат анализа.",
        )


class LocalHTTPAnalyzer(Analyzer):
    """A model wrapper receives a local path and returns the documented result.

    This is intentionally not a direct Ollama integration: a vision model still
    needs video decoding, temporal reasoning, rules and evidence extraction.
    Those responsibilities belong to the local model wrapper, not the browser.
    """
    connected = True

    def __init__(self, endpoint: str, name: str, rules: list):
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username
            or parsed.password
            or parsed.fragment
        ):
            raise ValueError("KFM_ANALYZER_URL must be a local http:// loopback URL.")
        self.endpoint = endpoint
        self.name = name or "Локальная модель"
        self.rules = rules

    def analyze(self, video_path: Path, metadata: dict, progress) -> dict:
        progress(25, "Анализ видеозаписи")
        payload = json.dumps({
            "video_path": str(video_path.resolve()),
            "metadata": metadata,
            "rules": self.rules,
        }, ensure_ascii=False).encode("utf-8")
        request = Request(self.endpoint, payload, {"Content-Type": "application/json"})
        try:
            with build_opener(ProxyHandler({})).open(request, timeout=250) as response:
                raw = response.read(16 * 1024 * 1024 + 1)
            if len(raw) > 16 * 1024 * 1024:
                raise ValueError("Response is too large")
            result = json.loads(raw)
        except Exception as exc:
            raise AnalysisError(
                "analyzer_unavailable",
                "Локальный анализатор не вернул результат. Проверьте его запуск и повторите проверку.",
            ) from exc
        progress(90, "Подготовка доказательств")
        return validate_result(result, metadata.get("duration_sec"))


def number(value, name: str, duration=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    if duration and value > duration + 0.5:
        raise ValueError(f"{name} is outside the video")
    return float(value)


def validate_result(result: dict, duration=None) -> dict:
    """Never silently turn a malformed model response into a clean verdict."""
    try:
        if not isinstance(result, dict):
            raise ValueError("Expected an object")
        verdict = result.get("verdict")
        events = result.get("violations")
        if verdict not in {"violation", "no_violation"}:
            raise ValueError("Invalid verdict")
        if not isinstance(events, list) or len(events) > 100:
            raise ValueError("Invalid violations")
        if (verdict == "no_violation" and events) or (verdict == "violation" and not events):
            raise ValueError("Verdict and violations disagree")
        normalized = []
        for event in events:
            if not isinstance(event, dict):
                raise ValueError("Invalid event")
            title, description = event.get("type"), event.get("description")
            if not isinstance(title, str) or not title.strip() or len(title) > 300:
                raise ValueError("Invalid event type")
            if not isinstance(description, str) or not description.strip() or len(description) > 8000:
                raise ValueError("Invalid description")
            frames = event.get("evidence_frames")
            if not isinstance(frames, list) or not 1 <= len(frames) <= 3:
                raise ValueError("Expected 1–3 JPEG evidence frames")
            clean_frames = []
            for frame in frames:
                stamp = number(frame.get("timestamp_sec"), "timestamp_sec", duration)
                data = frame.get("image_base64")
                if not isinstance(data, str) or len(data) > 8 * 1024 * 1024:
                    raise ValueError("Invalid image")
                decoded = base64.b64decode(data, validate=True)
                if not decoded.startswith(b"\xff\xd8\xff") or not decoded.endswith(b"\xff\xd9"):
                    raise ValueError("Evidence must be JPEG")
                clean_frames.append({"timestamp_sec": stamp, "image_base64": data})
            clean = {
                "type": title.strip(),
                "timecode_sec": number(event.get("timecode_sec"), "timecode_sec", duration),
                "description": description.strip(),
                "evidence_frames": clean_frames,
            }
            rule = event.get("rule")
            if rule is not None:
                if not isinstance(rule, dict):
                    raise ValueError("Invalid rule")
                url = rule.get("url")
                if url and (not isinstance(url, str) or urlsplit(url).scheme not in {"http", "https"}):
                    raise ValueError("Invalid rule URL")
                clean["rule"] = {
                    "title": str(rule.get("title", ""))[:1000],
                    "section": str(rule.get("section", ""))[:500],
                    "url": url,
                }
            normalized.append(clean)
        normalized.sort(key=lambda x: x["timecode_sec"])
        return {"verdict": verdict, "violations": normalized}
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        raise AnalysisError(
            "invalid_model_result",
            "Ответ модели не соответствует формату KFM. Результат не опубликован; проверьте адаптер модели.",
        ) from exc


def analyze_demo(scenario: str, asset_dir: Path, progress) -> dict:
    for percent, label in [
        (15, "Подготовка учебного сценария"),
        (40, "Демонстрация поиска событий"),
        (70, "Подготовка учебных кадров"),
        (95, "Сборка демонстрационного отчёта"),
    ]:
        progress(percent, label)
        time.sleep(0.6)
    if scenario == "clean":
        return {"verdict": "no_violation", "violations": []}

    def frame(name, stamp):
        return {
            "timestamp_sec": stamp,
            "image_base64": base64.b64encode((asset_dir / name).read_bytes()).decode("ascii"),
        }

    return {
        "verdict": "violation",
        "violations": [
            {
                "type": "Наличные без наблюдения",
                "timecode_sec": 12.4,
                "description": "Учебная сцена: сотрудник отошёл от стола, на котором остались банкноты. Это иллюстрация интерфейса, а не результат анализа видеозаписи.",
                "evidence_frames": [frame("unattended.jpg", 12.4)],
                "rule": {"title": "Демонстрационное правило", "section": "В рабочем режиме здесь будет проверенный нормативный источник.", "url": None},
            },
            {
                "type": "Доступ к рабочей зоне",
                "timecode_sec": 27.8,
                "description": "Учебная сцена: второй человек оказался со стороны рабочего места кассира. Правомерность доступа в реальной записи требует проверки контекста и применимого правила.",
                "evidence_frames": [frame("access.jpg", 27.8)],
                "rule": {"title": "Демонстрационное правило", "section": "Пример категории события без юридического заключения.", "url": None},
            },
        ],
    }
