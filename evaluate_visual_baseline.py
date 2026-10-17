"""Evaluate a frozen local candidate detector without exposing labels to inference.

The existing fixed holdout is the default. Dev scores are explicitly marked as
fitted-set measurements, never independent accuracy. Only aggregate progress is
printed; detailed numeric outputs stay in local JSON files, without any images.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch

from analyzers import AnalysisError
from evaluate_dataset import (canonical_hash, file_hash, get_manifest, neutral_video,
                              percent, safe_error_code, safe_metadata, summarize, write_json)
from server import read_metadata
import visual_baseline
from visual_baseline import (DEFAULT_ENCODER, DEFAULT_MODEL, ROOT, VisualBaseline,
                             remaining, select_candidates)


SCHEMA_VERSION = 1
TIME_TOLERANCE = 5.0
CANDIDATE_LIMIT = 5
CANDIDATE_SEPARATION = 6.0
DEADLINE_SEC = 300.0
SOURCE_FILES = ("evaluate_visual_baseline.py", "visual_baseline.py", "evaluate_dataset.py", "server.py")


def split_integrity(manifest, selected_split):
    """Byte-identical clips must not cross the training/evaluation boundary."""
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("No fixed evaluation samples")
    if canonical_hash(samples) != manifest.get("split_fingerprint"):
        raise ValueError("Split manifest integrity failure")
    by_hash = defaultdict(list)
    ids = set()
    for row in samples:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not re.fullmatch(r"sample-[0-9]+", sample_id) or sample_id in ids:
            raise ValueError("Invalid or duplicate sample identifier")
        ids.add(sample_id)
        if row.get("split") not in {"dev", "holdout"} or type(row.get("expected_violation")) is not bool:
            raise ValueError("Invalid split or reference class")
        digest = row.get("video_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Invalid video checksum")
        by_hash[digest].append(row)
    for group in by_hash.values():
        if len({row["split"] for row in group}) != 1:
            raise ValueError("Byte-identical content crosses dev and holdout")
        if len({row["expected_violation"] for row in group}) != 1:
            raise ValueError("Byte-identical content has conflicting reference classes")
    selected = [row for row in samples if row["split"] == selected_split]
    if not selected:
        raise ValueError("Requested fixed split is empty")
    counts = {
        "dataset_files": len(samples), "dataset_unique_sha256": len(by_hash),
        "duplicate_content_groups": sum(len(group) > 1 for group in by_hash.values()),
        "cross_split_sha256_groups": 0,
        "selected_files": len(selected), "selected_unique_sha256": len({row["video_sha256"] for row in selected}),
        "selected_positive": sum(row["expected_violation"] for row in selected),
        "selected_negative": sum(not row["expected_violation"] for row in selected),
    }
    return selected, counts


def package_versions():
    values = {"python": sys.version.split()[0]}
    for package in ("numpy", "torch", "transformers", "Pillow"):
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = None
    return values


def fingerprint_config(model, args, manifest, uniqueness):
    if model.config.get("train_split") != "dev" or model.config.get("split_fingerprint") != manifest["split_fingerprint"]:
        raise ValueError("Frozen model was not trained on this exact dev split")
    if not math.isfinite(model.threshold) or not 0 <= model.threshold <= 1:
        raise ValueError("Invalid frozen decision threshold")
    config = {
        "schema_version": SCHEMA_VERSION, "evaluator": "frozen_visual_candidate_baseline",
        "model_version": model.config["version"], "model_fingerprint": model.config.get("fingerprint"),
        "model_config_sha256": file_hash(model.directory / "config.json"),
        "model_weights_sha256": file_hash(model.directory / "weights.npz"),
        "encoder_fingerprint": model.encoder.fingerprint,
        "source_sha256": {name: file_hash(ROOT / name) for name in SOURCE_FILES},
        "runtime": package_versions(), "dataset_fingerprint": manifest["dataset_fingerprint"],
        "split_fingerprint": manifest["split_fingerprint"], "split": args.split,
        "seed": manifest["seed"], "content_uniqueness": uniqueness,
        "threshold": model.threshold, "video_decision": "max(window_scores) >= frozen_threshold",
        "candidate_limit": CANDIDATE_LIMIT, "candidate_separation_sec": CANDIDATE_SEPARATION,
        "time_tolerance_sec": TIME_TOLERANCE, "deadline_sec": DEADLINE_SEC,
        "feature_cache_policy": "fresh temporary cache for each evaluated video; decoding is included",
        "evaluation_role": "fitted_dev_diagnostic" if args.split == "dev" else "fixed_holdout",
        "limitation": "Candidate detection only; event type and legal correctness are not evaluated.",
    }
    config["fingerprint"] = canonical_hash(config)
    return config


def ensure_frozen(model, config):
    if (file_hash(model.directory / "config.json") != config["model_config_sha256"]
            or file_hash(model.directory / "weights.npz") != config["model_weights_sha256"]
            or any(file_hash(ROOT / name) != digest for name, digest in config["source_sha256"].items())):
        raise ValueError("Model or evaluation source changed during this run")


def evaluate_one(model, source, scratch_root):
    """Ground-truth class, timestamp and source ID are deliberately not arguments."""
    started = time.monotonic()
    deadline = started + DEADLINE_SEC
    metadata = {}
    try:
        with neutral_video(source, scratch_root) as neutral:
            metadata = safe_metadata(read_metadata(neutral))
            duration = metadata["duration_sec"]
            if duration is None or not 0 < duration <= 300:
                raise AnalysisError("unsupported_video_duration", "Unsupported local video duration")
            original_extract = visual_baseline.extract_embeddings

            def fresh_extract(*positional, **keywords):
                keywords["cache"] = neutral.parent / "fresh_features"
                return original_extract(*positional, **keywords)

            # Reuse the exact production score_windows implementation, while
            # ensuring cached training/evaluation features cannot inflate speed.
            with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink), redirect_stderr(sink), \
                    patch.object(visual_baseline, "extract_embeddings", new=fresh_extract):
                centers, scores = model.score_windows(neutral, duration, deadline=deadline)
            centers = [float(value) for value in centers]
            scores = [float(value) for value in scores]
            if (not centers or len(centers) != len(scores)
                    or any(not math.isfinite(value) or not 0 <= value < duration for value in centers)
                    or any(right <= left for left, right in zip(centers, centers[1:]))
                    or any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores)):
                raise AnalysisError("invalid_candidate_scores", "Invalid local candidate output")
            remaining(deadline)
            maximum = max(scores)
            candidates = select_candidates(centers, scores, model.threshold,
                                           separation=CANDIDATE_SEPARATION, limit=CANDIDATE_LIMIT)
            record = {"status": "done", "predicted_violation": maximum >= model.threshold,
                      "video_max_score": maximum, "window_centers_sec": centers,
                      "window_scores": scores, "candidates": candidates,
                      "duration_sec": duration, "diagnostics": {"frames": len(centers), "candidates": len(candidates)}}
        remaining(deadline)
    except Exception as exc:
        record = {"status": "failed", "error_code": safe_error_code(exc),
                  "duration_sec": metadata.get("duration_sec")}
    record["elapsed_sec"] = round(time.monotonic() - started, 3)
    return record


def metrics_for(records, total_selected):
    # Reuse the tested class/abstention accounting, adapting only numeric
    # candidate times. These internal result objects are not KFM legal reports.
    adapted = [{**row, "result": {"violations": [{"timecode_sec": item["timecode"]}
                                                  for item in row.get("candidates", [])]}}
               for row in records]
    metrics = summarize(adapted, total_selected=total_selected, time_tolerance=TIME_TOLERANCE)
    metrics["prediction_scope"] = "visual candidates, not legally established violations"
    metrics["timecode"]["candidate_limit"] = CANDIDATE_LIMIT
    metrics["timecode"]["candidate_separation_sec"] = CANDIDATE_SEPARATION
    metrics["timing"]["includes"] = "neutral copy, container metadata, fresh decoding, local encoding and scoring; initial encoder loading is charged to the first video"
    return metrics


def validate_saved_record(record, sample, config):
    if not isinstance(record, dict):
        raise ValueError("Malformed saved sample")
    checksum = record.get("record_fingerprint")
    unsigned = {key: value for key, value in record.items() if key != "record_fingerprint"}
    if checksum != canonical_hash(unsigned) or record.get("run_fingerprint") != config["fingerprint"]:
        raise ValueError("Saved sample integrity or run mismatch")
    for key in ("sample_id", "video_sha256", "expected_violation", "reference_time_sec", "reference_times_sec", "split"):
        if record.get(key) != sample.get(key):
            raise ValueError("Saved sample differs from fixed manifest")
    if record.get("status") not in {"done", "failed"}:
        raise ValueError("Unsupported saved sample status")
    if record["status"] == "done" and type(record.get("predicted_violation")) is not bool:
        raise ValueError("Malformed saved prediction")
    elapsed = record.get("elapsed_sec")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Malformed saved timing")


def write_reports(output, config, records, total):
    metrics = metrics_for(records, total)
    write_json(output / "summary.json", {"config": config, "metrics": metrics,
               "updated_utc": datetime.now(timezone.utc).isoformat()})
    write_json(output / "details.json", records)
    cm = metrics["confusion_matrix"]
    uniqueness = config["content_uniqueness"]
    role = ("Оценка на обучающей части dev. Модель обучалась на этих видео: результат завышен и не является независимой проверкой."
            if config["split"] == "dev" else
            "Оценка на зафиксированной части holdout, исключённой из обучения и подбора порога этой модели.")
    lines = [
        "# KFM — проверка визуального детектора кандидатов", "", role, "",
        "Детектор ищет подозрительные фрагменты. Он не определяет тип нарушения и не устанавливает юридическую применимость правила.", "",
        f"Обработано {metrics['evaluated']} из {metrics['selected']} выбранных файлов. В выборке {uniqueness['selected_unique_sha256']} уникальных записей по SHA-256; совпадающих файлов между dev и holdout нет.", "",
        f"Порог зафиксирован до запуска: {config['threshold']:.8f}. Ответ по видео положительный, если максимальная оценка его окон не ниже порога. Для таймкодов используется до пяти кандидатов с разносом не менее шести секунд.", "",
        "| Показатель | Результат |", "|---|---:|",
        f"| TP / TN / FP / FN | {cm['tp']} / {cm['tn']} / {cm['fp']} / {cm['fn']} |",
        f"| Доля завершённых ответов | {percent(metrics['coverage'])} |",
        f"| Технические ошибки | {metrics['failed']} |",
        f"| Верные ответы среди всех обработанных видео | {percent(metrics['end_to_end_accuracy'])} |",
        f"| Precision | {percent(metrics['precision'])} |",
        f"| Recall, с учётом ошибок обработки | {percent(metrics['recall'])} |",
        f"| F1, с учётом ошибок обработки | {percent(metrics['f1'])} |",
        f"| Сбалансированная точность, с учётом ошибок | {percent(metrics['balanced_accuracy'])} |",
        f"| Таймкод с допуском ±5 секунд | {metrics['timecode']['matched_positive']} / {metrics['timecode']['annotated_positive']} |", "",
        f"Не получен ответ для {cm['unresolved_positive']} положительных и {cm['unresolved_negative']} отрицательных видео. Ошибки не считаются ответом «нарушения нет». Условные метрики только завершённых ответов сохранены отдельно в summary.json.", "",
        f"Время на видео: медиана {metrics['timing']['median_sec']} с, 95-й процентиль {metrics['timing']['p95_sec']} с, максимум {metrics['timing']['max_sec']} с. Более 300 секунд: {metrics['timing']['over_300_sec']}.", "",
        "Замер включает создание нейтральной копии, чтение метаданных, извлечение кадров, локальное кодирование и расчёт оценок. Для каждого видео используется новый временный кэш признаков. Первая запись также включает загрузку кодировщика в память; проверка файлов весов при инициализации выполнена до поштучного замера.", "",
        "Метки, опорные таймкоды и исходные имена файлов не передаются модели. Она получает локальную копию input.mov и длительность. Все вычисления выполняются локально; подробный JSON содержит только числовые результаты и разметку, без изображений и описаний сцен.", "",
        "Размер выборки мал. Разбиение не подтверждает независимость камер и сеансов. Все исходные записи ранее проходили автоматический предварительный разбор; holdout означает исключение из текущего обучения, а не новый внешний набор. Дальнейший подбор параметров по этим результатам лишает их статуса независимой проверки.", "",
        "Совпадение таймкода с одной из опорных отметок не проверяет полноту событий или юридическую правильность результата. Опорные отметки совпадающих файлов объединены в манифесте. При нулевом знаменателе метрика не определена.", "",
        f"Отпечаток запуска: `{config['fingerprint']}`.",
        f"Отпечаток разбиения: `{config['split_fingerprint']}`.", "",
    ]
    temporary = output / "report.md.pending"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output / "report.md")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT.parent / "open")
    parser.add_argument("--manifest", type=Path, default=ROOT / "local_review" / "evaluation_split_deduplicated.json")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--split", choices=("holdout", "dev"), default="holdout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.manifest.is_file():
        raise ValueError("Prepare and freeze the split before baseline evaluation")
    saved_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest = get_manifest(args.dataset, args.manifest, seed=saved_manifest["seed"],
                            holdout_size=saved_manifest["holdout_size"])
    samples, uniqueness = split_integrity(manifest, args.split)
    # Constructor reads only frozen local artifacts. It cannot fit or tune.
    model = VisualBaseline(args.model, encoder_directory=args.encoder)
    config = fingerprint_config(model, args, manifest, uniqueness)
    output = args.output.resolve()
    run_file = output / "run.json"
    if run_file.exists():
        previous = json.loads(run_file.read_text(encoding="utf-8"))
        if not args.resume:
            raise ValueError("Run already exists; explicit matching resume is required")
        if (previous.get("fingerprint") != config["fingerprint"]
                or canonical_hash({key: value for key, value in previous.items() if key != "fingerprint"}) != config["fingerprint"]):
            raise ValueError("Cannot resume different weights, code, split, encoder, or policy")
    elif args.resume:
        raise ValueError("No existing run to resume")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not an empty evaluation run")
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_json(run_file, config)
    rows_dir = output / "samples"
    rows_dir.mkdir(exist_ok=True)
    expected_files = {sample["sample_id"] + ".json" for sample in samples}
    if any(path.name not in expected_files for path in rows_dir.glob("*.json")):
        raise ValueError("Run contains samples outside the selected split")
    records = []
    for index, sample in enumerate(samples, 1):
        ensure_frozen(model, config)
        saved = rows_dir / (sample["sample_id"] + ".json")
        if saved.exists():
            if not args.resume:
                raise ValueError("Existing sample requires explicit resume")
            record = json.loads(saved.read_text(encoding="utf-8"))
            validate_saved_record(record, sample, config)
        else:
            source = args.dataset / "videos" / sample["filename"]
            if file_hash(source) != sample["video_sha256"]:
                raise ValueError("Dataset video changed after split validation")
            # Attach labels only after inference finishes.
            record = evaluate_one(model, source, ROOT / "data" / "baseline_evaluation")
            record.update({key: sample.get(key) for key in ("sample_id", "video_sha256", "expected_violation", "reference_time_sec", "reference_times_sec", "split")})
            record["run_fingerprint"] = config["fingerprint"]
            record["record_fingerprint"] = canonical_hash(record)
            write_json(saved, record)
        records.append(record)
        metrics = write_reports(output, config, records, len(samples))
        print(json.dumps({"processed": index, "total": len(samples), "completed": metrics["completed"],
                          "failed": metrics["failed"], "elapsed_sec": record["elapsed_sec"]}), flush=True)
    ensure_frozen(model, config)
    print(json.dumps({"complete": True, "split": args.split, "metrics": metrics}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Never reveal arbitrary model/decoder exception text on the console.
        print(json.dumps({"evaluation_error": safe_error_code(exc), "exception_type": type(exc).__name__}), file=sys.stderr, flush=True)
        raise SystemExit(1)
