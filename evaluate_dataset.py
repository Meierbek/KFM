"""Reproducible local evaluation. Stdout contains aggregate metrics only.

Source labels and reference timestamps are evaluation data, never inference
inputs. Detailed responses (including any evidence images) stay on local disk.
The held-out portion is a small split of the supplied dataset, not evidence of
generalization to other branches, cameras, or new violation categories.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import sys
import time
import uuid

from analyzers import AnalysisError, validate_result
from server import read_metadata


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 2
DEFAULT_SEED = 20261017
ABSTENTION_CODES = {
    "uncertain_observations", "insufficient_evidence", "uncertain_result",
    "cannot_determine", "low_confidence", "insufficient_visibility",
}
SOURCE_FILES = (
    "evaluate_dataset.py", "analyzers.py", "local_model.py", "refined_model.py",
    "visual_sampling.py", "server.py", "rules.json", "visual_baseline.py", "guided_model.py", "object_model.py", "yolo_candidates.py", "sequence_model.py", "large_model.py",
    ".models/visual-baseline-v1/config.json", ".models/visual-baseline-v1/weights.npz",
    ".models/dinov2-small/encoder.json",
    ".models/yolov8n/yolov8n.pt",
)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def parse_label(value):
    """A missing timestamp does not imply a negative class."""
    text = " ".join(str(value).strip().casefold().split())
    mapping = {"нарушение": True, "без нарушения": False}
    if text not in mapping:
        raise ValueError("Unknown reference label; evaluation stopped")
    return mapping[text]


def parse_timestamp(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value) and value >= 0:
            return float(value)
        raise ValueError("Invalid reference timestamp")
    text = str(value).strip().casefold().replace(",", ".")
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?", text):
        parts = [float(piece) for piece in text.split(":")]
        if any(part >= 60 for part in parts[1:]):
            raise ValueError("Invalid reference timestamp")
        return sum(part * 60 ** index for index, part in enumerate(reversed(parts)))
    minutes = re.search(r"(\d+(?:\.\d+)?)\s*мин", text)
    seconds = re.search(r"(\d+(?:\.\d+)?)\s*сек", text)
    if not minutes and not seconds:
        raise ValueError("Unrecognized reference timestamp")
    return (float(minutes.group(1)) * 60 if minutes else 0.0) + (float(seconds.group(1)) if seconds else 0.0)


def load_dataset(dataset):
    import openpyxl
    dataset = Path(dataset).resolve()
    workbook_path = dataset / "open_labels.xlsx"
    workbook = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
    records = []
    seen = set()
    try:
        rows = iter(workbook.active.values)
        next(rows)
        for row in rows:
            if not row or all(item is None for item in row):
                continue
            if len(row) < 3:
                raise ValueError("Reference worksheet requires three columns")
            name, reference_time, label = row[:3]
            stem = str(name).strip()
            filename = stem if Path(stem).suffix.casefold() in {".mov", ".mp4"} else stem + ".mov"
            if Path(filename).name != filename or "/" in filename or "\\" in filename:
                raise ValueError("Reference filename is not a basename")
            if filename.casefold() in seen:
                raise ValueError("Duplicate reference filename")
            seen.add(filename.casefold())
            path = dataset / "videos" / filename
            if not path.is_file():
                raise ValueError("Referenced video does not exist")
            records.append({"filename": filename, "expected_violation": parse_label(label),
                            "reference_time_sec": parse_timestamp(reference_time),
                            "video_sha256": file_hash(path), "size_bytes": path.stat().st_size})
    finally:
        workbook.close()
    if not records:
        raise ValueError("No evaluation samples")
    records.sort(key=lambda item: item["filename"])
    return records, file_hash(workbook_path)


def build_split(records, *, seed=DEFAULT_SEED, holdout_size=14, forced_dev=()):
    """Fixed stratification; never uses duration, model verdicts, or accuracy."""
    if not 0 < holdout_size < len(records):
        raise ValueError("Holdout must contain between one and N-1 examples")
    if len({row["filename"] for row in records}) != len(records):
        raise ValueError("Duplicate split filename")
    forced_dev = set(forced_dev)
    known = {item["filename"] for item in records}
    forced_dev &= known
    if len(records) - len(forced_dev) < holdout_size:
        raise ValueError("Too many previously evaluated samples for this holdout size")
    rng = random.Random(seed)
    groups = {label: sorted([dict(item) for item in records if item["expected_violation"] is label],
                            key=lambda item: item["filename"]) for label in (False, True)}
    if not all(groups.values()):
        raise ValueError("Stratified evaluation requires both classes")
    targets = {label: int(holdout_size * len(group) / len(records)) for label, group in groups.items()}
    remainder_order = sorted(groups, key=lambda label: (-(holdout_size * len(groups[label]) / len(records) - targets[label]), label))
    for label in remainder_order[:holdout_size - sum(targets.values())]:
        targets[label] += 1
    selected = set()
    for label, group in groups.items():
        candidates = [item for item in group if item["filename"] not in forced_dev]
        if len(candidates) < targets[label]:
            raise ValueError("Previously evaluated examples prevent the requested stratification")
        rng.shuffle(candidates)
        selected.update(item["filename"] for item in candidates[:targets[label]])
    shuffled = sorted([dict(item) for item in records], key=lambda item: item["filename"])
    rng.shuffle(shuffled)
    for index, item in enumerate(shuffled, 1):
        item.update(sample_id=f"sample-{index:03d}", split="holdout" if item["filename"] in selected else "dev",
                    previously_evaluated=item["filename"] in forced_dev)
    return shuffled


def get_manifest(dataset, path, *, seed, holdout_size):
    records, workbook_digest = load_dataset(dataset)
    fingerprint = canonical_hash({"records": records, "workbook_sha256": workbook_digest})
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("schema_version") != SCHEMA_VERSION
                or manifest.get("dataset_fingerprint") != fingerprint
                or manifest.get("seed") != seed):
            raise ValueError("Saved split differs from the dataset or requested configuration; do not silently redraw it")
        if canonical_hash(manifest["samples"]) != manifest.get("split_fingerprint"):
            raise ValueError("Saved split manifest was modified")
        if len({row["video_sha256"] for row in manifest["samples"]}) != len(manifest["samples"]):
            raise ValueError("Evaluation split must contain unique video content")
        return manifest
    previous_file = ROOT / "local_review" / "benchmark.json"
    previous = json.loads(previous_file.read_text(encoding="utf-8")) if previous_file.exists() else []
    forced_dev = {item["filename"] for item in previous if isinstance(item, dict) and "filename" in item}
    samples = deduplicate_split(build_split(records, seed=seed, holdout_size=holdout_size, forced_dev=forced_dev))
    manifest = {
        "schema_version": SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed, "holdout_size": sum(row["split"] == "holdout" for row in samples),
        "dataset_fingerprint": fingerprint, "workbook_sha256": workbook_digest,
        "split_fingerprint": canonical_hash(samples), "samples": samples,
        "selection": "Seeded stratification by reference class only; prior benchmark examples forced into dev.",
        "limitation": "All source videos have previously received guided automatic review. Holdout means excluded from new tuning, not pristine external test data.",
    }
    write_json(path, manifest)
    return manifest


def deduplicate_split(samples):
    """One content hash per unit; prior dev exposure always takes precedence."""
    groups = {}
    for sample in samples:
        groups.setdefault(sample["video_sha256"], []).append(sample)
    result = []
    for group in groups.values():
        if len({row["expected_violation"] for row in group}) != 1:
            raise ValueError("Identical videos have conflicting labels")
        exposed = [row for row in group if row["split"] == "dev"]
        representative = dict((exposed or group)[0])
        representative["split"] = "dev" if exposed else "holdout"
        representative["reference_times_sec"] = sorted({row["reference_time_sec"] for row in group if row.get("reference_time_sec") is not None})
        representative["aliases"] = sorted({row["filename"] for row in group})
        representative["previously_evaluated"] = any(row.get("previously_evaluated", False) for row in group)
        result.append(representative)
    return result


def repair_split_manifest(source, destination):
    """Keep the original audit trail; never move an exposed video into holdout."""
    if Path(destination).exists():
        raise ValueError("Corrected manifest already exists")
    old = json.loads(Path(source).read_text(encoding="utf-8"))
    if canonical_hash(old["samples"]) != old["split_fingerprint"]:
        raise ValueError("Original manifest integrity failure")
    samples = deduplicate_split(old["samples"])
    corrected = {**old, "schema_version": SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                 "samples": samples, "split_fingerprint": canonical_hash(samples),
                 "holdout_size": sum(row["split"] == "holdout" for row in samples),
                 "source_split_fingerprint": old["split_fingerprint"],
                 "selection": "Original fixed split deduplicated by SHA-256. Any previously-dev content stays dev. Alias timestamps merged. No model output used.",
                 "duplicate_files_removed": len(old["samples"])-len(samples)}
    write_json(destination, corrected)
    return corrected


def safe_metadata(metadata):
    clean = {}
    for key in ("duration_sec", "width", "height"):
        value = metadata.get(key)
        clean[key] = value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None
    return clean


@contextmanager
def neutral_video(source, scratch_root):
    """Copy to a random directory: no filename or annotation reaches inference."""
    root = Path(scratch_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory = root / uuid.uuid4().hex
    directory.mkdir()
    try:
        target = directory / "input.mov"
        # A copy prevents an accidental analyzer write from changing the dataset.
        shutil.copyfile(source, target)
        yield target
    finally:
        if directory.resolve().parent != root:
            raise RuntimeError("Invalid temporary evaluation directory")
        shutil.rmtree(directory)


def safe_error_code(exc):
    code = getattr(exc, "code", "analyzer_error")
    return code if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,79}", code) else "analyzer_error"


def evaluate_one(analyzer, source, scratch_root):
    """This function deliberately has no label, timestamp or source ID argument."""
    started = time.monotonic()
    try:
        with neutral_video(source, scratch_root) as input_path:
            metadata = safe_metadata(read_metadata(input_path))
            # An adapter must not leak local scene text via diagnostic output.
            with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink), redirect_stderr(sink):
                raw = analyzer.analyze(input_path, metadata, lambda *args: None)
                result = validate_result(raw, metadata.get("duration_sec"))
        record = {"status": "done", "predicted_violation": result["verdict"] == "violation",
                  "result": result, "duration_sec": metadata.get("duration_sec")}
    except Exception as exc:
        code = safe_error_code(exc)
        record = {"status": "abstained" if code in ABSTENTION_CODES else "failed", "error_code": code}
    diagnostics = getattr(analyzer, "last_diagnostics", {})
    if isinstance(diagnostics, dict):
        record["diagnostics"] = {
            key: value for key, value in diagnostics.items()
            if key in {"frames", "windows", "candidates", "accepted", "rejected", "unresolved", "calls", "elapsed_sec", "candidate_frames", "proposal_windows", "detector_seconds", "object_proposals", "object_frames", "object_seconds"}
            and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        }
    record["elapsed_sec"] = round(time.monotonic() - started, 3)
    return record


def ratio(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else None


def harmonic(precision, recall):
    if precision is None or recall is None:
        return None
    return round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0


def summarize(records, *, total_selected=None, time_tolerance=5.0):
    counts = {key: 0 for key in ("tp", "tn", "fp", "fn", "unresolved_positive", "unresolved_negative")}
    status = Counter()
    timed_positive = temporal_hits = 0
    for row in records:
        expected = row["expected_violation"]
        if type(expected) is not bool:
            raise ValueError("Expected class must be a boolean")
        state = row["status"]
        status[state] += 1
        if state == "done":
            predicted = row["predicted_violation"]
            if type(predicted) is not bool:
                raise ValueError("Predicted class must be a boolean")
            counts["tp" if expected and predicted else "fn" if expected else "fp" if predicted else "tn"] += 1
        else:
            counts["unresolved_positive" if expected else "unresolved_negative"] += 1
        references = row.get("reference_times_sec") or ([row["reference_time_sec"]] if row.get("reference_time_sec") is not None else [])
        if expected and references:
            timed_positive += 1
            if state == "done" and row["predicted_violation"]:
                stamps = [event.get("timecode_sec") for event in row.get("result", {}).get("violations", [])]
                temporal_hits += any(isinstance(stamp, (int, float)) and not isinstance(stamp, bool)
                                     and math.isfinite(stamp) and any(abs(stamp-reference) <= time_tolerance for reference in references) for stamp in stamps)
    tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
    completed = tp + tn + fp + fn
    positive = tp + fn + counts["unresolved_positive"]
    negative = tn + fp + counts["unresolved_negative"]
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, positive)
    specificity = ratio(tn, negative)
    conditional_recall = ratio(tp, tp + fn)
    conditional_specificity = ratio(tn, tn + fp)
    timings = sorted(row["elapsed_sec"] for row in records if isinstance(row.get("elapsed_sec"), (int, float)) and math.isfinite(row["elapsed_sec"]))
    diagnostic_totals = {key: sum(row.get("diagnostics", {}).get(key, 0) for row in records)
                         for key in ("frames", "windows", "candidates", "accepted", "rejected", "calls")}
    return {
        "selected": len(records) if total_selected is None else total_selected,
        "evaluated": len(records), "reference_positive": positive, "reference_negative": negative,
        "completed": completed, "abstained": status["abstained"], "failed": status["failed"],
        "confusion_matrix": counts, "coverage": ratio(completed, len(records)),
        "end_to_end_accuracy": ratio(tp + tn, len(records)),
        "precision": precision, "recall": recall, "f1": ratio(2 * tp, 2 * tp + fp + fn + counts["unresolved_positive"]),
        "specificity": specificity,
        "balanced_accuracy": round((recall + specificity) / 2, 6) if recall is not None and specificity is not None else None,
        "resolved_only": {
            "accuracy": ratio(tp + tn, completed), "precision": precision,
            "recall": conditional_recall, "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "balanced_accuracy": round((conditional_recall + conditional_specificity) / 2, 6)
            if conditional_recall is not None and conditional_specificity is not None else None,
        },
        "timecode": {"tolerance_sec": time_tolerance, "annotated_positive": timed_positive,
                     "matched_positive": temporal_hits, "recall": ratio(temporal_hits, timed_positive),
                     "note": "Matches one supplied reference timestamp per positive clip; event completeness and legal correctness are not scored."},
        "timing": {"total_sec": round(sum(timings), 3),
                   "mean_sec": round(statistics.mean(timings), 3) if timings else None,
                   "median_sec": round(statistics.median(timings), 3) if timings else None,
                   "p95_sec": timings[max(0, math.ceil(len(timings) * .95) - 1)] if timings else None,
                   "max_sec": max(timings) if timings else None,
                   "over_300_sec": sum(value > 300 for value in timings)},
        "error_counts": dict(sorted(Counter(row.get("error_code", "unknown") for row in records if row["status"] != "done").items())),
        "diagnostic_totals": diagnostic_totals,
    }


def fingerprint_config(analyzer, args, manifest):
    source_hashes = {name: file_hash(ROOT / name) if (ROOT / name).exists() else None for name in SOURCE_FILES}
    runtime = {"model": getattr(analyzer, "model", args.model), "python": sys.version.split()[0]}
    request = getattr(analyzer, "request", None)
    if request:
        try:
            tags = request("/api/tags")
            matching = [item for item in tags.get("models", []) if item.get("name") == runtime["model"]]
            runtime["model_digest"] = matching[0].get("digest") if matching else None
            runtime["ollama_version"] = request("/api/version").get("version")
        except Exception:
            runtime["model_digest"] = None
    # Explicit allowlist: never save proxy credentials or unrelated environment.
    env = {key: value for key, value in os.environ.items() if key.startswith("KFM_REFINED_")
           and not any(piece in key for piece in ("KEY", "TOKEN", "PASSWORD", "SECRET", "URL"))}
    config = {"schema_version": SCHEMA_VERSION, "analyzer": args.analyzer, "runtime": runtime,
              "source_sha256": source_hashes, "environment": dict(sorted(env.items())),
              "dataset_fingerprint": manifest["dataset_fingerprint"],
              "split_fingerprint": manifest["split_fingerprint"], "split": args.split,
              "seed": manifest["seed"], "limit": args.limit, "time_tolerance_sec": args.time_tolerance,
              "split_selection": manifest.get("selection"), "unique_content_units": len(manifest["samples"])}
    config["fingerprint"] = canonical_hash(config)
    return config


def percent(value):
    return "не определено" if value is None else f"{value * 100:.1f}%"


def write_report(output, config, metrics):
    report = {"config": config, "metrics": metrics, "updated_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "summary.json", report)
    cm = metrics["confusion_matrix"]
    lines = [
        "# KFM — локальная проверка качества", "",
        f"Модель: `{config['runtime']['model']}`. Анализатор: `{config['analyzer']}`. Выборка: `{config['split']}`.", "",
        f"Обработано {metrics['evaluated']} из {metrics['selected']} выбранных видео; нарушений по исходной разметке: {metrics['reference_positive']}, без нарушений: {metrics['reference_negative']}.", "",
        "| Показатель | Результат |", "|---|---:|",
        f"| Доля видео с окончательным ответом | {percent(metrics['coverage'])} |",
        f"| Правильный ответ среди всех обработанных видео | {percent(metrics['end_to_end_accuracy'])} |",
        f"| Precision: доля верных тревог | {percent(metrics['precision'])} |",
        f"| Recall: найденные положительные видео, включая отказы в знаменателе | {percent(metrics['recall'])} |",
        f"| F1 | {percent(metrics['f1'])} |",
        f"| Сбалансированная точность, включая отказы | {percent(metrics['balanced_accuracy'])} |",
        f"| Отказы из-за неопределённости | {metrics['abstained']} |",
        f"| Технические ошибки | {metrics['failed']} |", "",
        "| Исходная разметка | Ответ «нарушение» | Ответ «без нарушения» | Нет окончательного ответа |",
        "|---|---:|---:|---:|",
        f"| Нарушение | {cm['tp']} | {cm['fn']} | {cm['unresolved_positive']} |",
        f"| Без нарушения | {cm['fp']} | {cm['tn']} | {cm['unresolved_negative']} |", "",
        f"Таймкод совпал с опорной отметкой с допуском ±{metrics['timecode']['tolerance_sec']:g} с в {metrics['timecode']['matched_positive']} из {metrics['timecode']['annotated_positive']} положительных видео с отметкой. Это не проверка полноты всех событий и не проверка правильности ссылки на правило.", "",
        f"Время на видео: медиана {metrics['timing']['median_sec']} с, 95-й процентиль {metrics['timing']['p95_sec']} с, максимум {metrics['timing']['max_sec']} с. Более 300 с: {metrics['timing']['over_300_sec']}.", "",
        "Метки, исходные таймкоды и исходные имена файлов не передавались анализатору. Для каждого вызова создавалась отдельная локальная копия с нейтральным именем input.mov. Результаты прежних разборов не использовались для предсказаний.", "",
        f"Разбиение первоначально зафиксировано с seed={config['seed']}; дубликаты объединены по SHA-256, ранее использованное в dev содержимое оставлено в dev. Всего уникальных записей: {config['unique_content_units']}. Все исходные видео уже проходили предварительный автоматический разбор, поэтому holdout означает исключение из текущей настройки, а не совершенно новый внешний набор. Выборка мала и не доказывает качество на других камерах, отделениях или неизвестных типах нарушения. Оценки на dev и после повторной настройки по holdout не являются независимой итоговой проверкой.", "",
        "Ограниченный прогон первых примеров dev служит только проверкой работоспособности; по нему нельзя оценивать итоговую точность." if config["limit"] else "Прогон не ограничен первыми примерами; полнота завершения указана выше.", "",
        "Неопределённые ответы и ошибки показаны отдельно; они не превращены в ответ «без нарушения». В основной полноте и сбалансированной точности они считаются неуспешными решениями. Метрики только среди полученных ответов приведены отдельно в summary.json. Если знаменатель равен нулю, метрика не определена.", "",
        f"Отпечаток конфигурации и кода: `{config['fingerprint']}`.",
        f"Отпечаток разбиения: `{config['split_fingerprint']}`.", "",
        "Подробные ответы и кадры находятся только в локальной папке samples. Не передавайте их облачным сервисам.", "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def make_analyzer(kind, model):
    if kind == "large":
        from large_model import LargeVideoAnalyzer
        return LargeVideoAnalyzer(model=model)
    if kind == "sequence":
        from sequence_model import SequenceVideoAnalyzer
        return SequenceVideoAnalyzer(model=model)
    if kind == "objects":
        from object_model import ObjectAssistedAnalyzer
        return ObjectAssistedAnalyzer(model=model)
    if kind == "legacy":
        from local_model import OllamaVideoAnalyzer
        return OllamaVideoAnalyzer(model=model)
    if kind == "guided":
        from guided_model import GuidedVideoAnalyzer
        return GuidedVideoAnalyzer(model=model)
    from refined_model import RefinedVideoAnalyzer
    return RefinedVideoAnalyzer(model=model)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT.parent / "open")
    parser.add_argument("--manifest", type=Path, default=ROOT / "local_review" / "evaluation_split_deduplicated.json")
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--analyzer", choices=("legacy", "refined", "guided", "objects", "sequence", "large"), default="refined")
    parser.add_argument("--model", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--holdout-size", type=int, default=14)
    parser.add_argument("--time-tolerance", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=0, help="First N samples in fixed randomized order; exploratory only")
    parser.add_argument("--prepare-only", action="store_true", help="Save and check split without loading or running a model")
    args = parser.parse_args(argv)
    if not math.isfinite(args.time_tolerance) or args.time_tolerance < 0 or args.limit < 0:
        parser.error("time tolerance and limit must be non-negative")
    if args.limit and args.split != "dev":
        parser.error("--limit is allowed only for exploratory dev runs; holdout must remain complete")
    manifest = get_manifest(args.dataset, args.manifest, seed=args.seed, holdout_size=args.holdout_size)
    samples = [row for row in manifest["samples"] if args.split == "all" or row["split"] == args.split]
    if args.limit:
        samples = samples[:args.limit]
    if args.prepare_only:
        print(json.dumps({"prepared": True, "split": args.split, "selected": len(samples),
                          "positive": sum(row["expected_violation"] for row in samples),
                          "negative": sum(not row["expected_violation"] for row in samples),
                          "previously_evaluated": sum(row["previously_evaluated"] for row in samples),
                          "split_fingerprint": manifest["split_fingerprint"]}), flush=True)
        return 0
    analyzer = make_analyzer(args.analyzer, args.model)
    if hasattr(analyzer, "ready") and not analyzer.ready():
        raise AnalysisError("model_not_ready", "Local model is not ready")
    config = fingerprint_config(analyzer, args, manifest)
    output = args.output.resolve()
    run_file = output / "run.json"
    if run_file.exists():
        previous = json.loads(run_file.read_text(encoding="utf-8"))
        if not args.resume:
            raise ValueError("Run exists; use --resume or a new output directory")
        if previous.get("fingerprint") != config["fingerprint"]:
            raise ValueError("Cannot resume a different model, code, dataset, or sampling configuration")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not an empty evaluation run")
    write_json(run_file, config)
    snapshot = output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name, digest in config["source_sha256"].items():
        source = ROOT / name
        if digest and file_hash(source) == digest:
            (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, snapshot / name)
    records = []
    for index, reference in enumerate(samples, 1):
        record_path = output / "samples" / (reference["sample_id"] + ".json")
        if args.resume and record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if (record.get("run_fingerprint") != config["fingerprint"]
                    or record.get("sample_id") != reference["sample_id"]
                    or record.get("video_sha256") != reference["video_sha256"]):
                raise ValueError("Saved sample does not match this evaluation run")
        else:
            outcome = evaluate_one(analyzer, args.dataset / "videos" / reference["filename"], output / ".inputs")
            record = {**reference, **outcome, "run_fingerprint": config["fingerprint"]}
            write_json(record_path, record)
        records.append(record)
        metrics = summarize(records, total_selected=len(samples), time_tolerance=args.time_tolerance)
        write_report(output, config, metrics)
        print(json.dumps({"processed": index, "total": len(samples), "completed": metrics["completed"],
                          "abstained": metrics["abstained"], "failed": metrics["failed"],
                          "elapsed_sec": record["elapsed_sec"]}), flush=True)
    print(json.dumps({"complete": True, "metrics": summarize(records, total_selected=len(samples), time_tolerance=args.time_tolerance)}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Never emit exception messages: an adapter might include scene text.
        print(json.dumps({"evaluation_error": safe_error_code(exc), "exception_type": type(exc).__name__}), file=sys.stderr, flush=True)
        raise SystemExit(1)
