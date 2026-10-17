"""Train and tune a local visual candidate detector on fixed DEV videos only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

from visual_baseline import (ROOT, VERSION, DEFAULT_ENCODER, DEFAULT_MODEL, LocalEncoder,
                             canonical_hash, extract_embeddings, hash_file, select_candidates, temporal_features)

SEED = 20261017
CS = (.01, .1, 1.0, 10.0)


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def download_encoder(directory=DEFAULT_ENCODER):
    """Only this explicit setup command accesses the network; no media upload."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    endpoint = "https://huggingface.co/api/models/facebook/dinov2-small?blobs=true"
    with urlopen(Request(endpoint, headers={"User-Agent": "KFM-local-setup"}), timeout=60) as response:
        info = json.load(response)
    revision = info["sha"]
    siblings = {entry["rfilename"]: entry for entry in info["siblings"]}
    selected = ("config.json", "preprocessor_config.json", "model.safetensors")
    entries = {}
    for name in selected:
        target = directory / name
        expected = siblings[name].get("lfs", {}).get("sha256")
        if not target.exists() or (expected and hash_file(target) != expected):
            url = f"https://huggingface.co/facebook/dinov2-small/resolve/{revision}/{name}"
            pending = target.with_name(target.name + ".part")
            with urlopen(Request(url, headers={"User-Agent": "KFM-local-setup"}), timeout=120) as response, pending.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            actual = hash_file(pending)
            if expected and actual != expected:
                raise ValueError("Downloaded encoder checksum mismatch")
            pending.replace(target)
        entries[name] = {"sha256": hash_file(target), "size_bytes": target.stat().st_size}
    save_json(directory / "encoder.json", {"repository": "facebook/dinov2-small", "revision": revision,
                                           "source": "https://huggingface.co/facebook/dinov2-small", "license": "Apache-2.0",
                                           "files": entries})
    print(json.dumps({"encoder_downloaded": True, "bytes": sum(item["size_bytes"] for item in entries.values()), "revision": revision}), flush=True)


def dev_samples(manifest):
    samples = manifest["samples"]
    if canonical_hash(samples) != manifest["split_fingerprint"]:
        # The evaluation manifest uses ensure_ascii=False; names can be Unicode.
        import hashlib
        expected = hashlib.sha256(json.dumps(samples, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        if expected != manifest["split_fingerprint"]:
            raise ValueError("Split manifest integrity failure")
    result = [sample for sample in samples if sample["split"] == "dev"]
    if not result or any(sample["split"] != "dev" for sample in result):
        raise ValueError("Training is restricted to dev")
    return result


def training_rows(videos, indexes):
    """Unlabeled windows of positive videos are not mislabeled negative."""
    import numpy as np
    x, y, weights, groups = [], [], [], []
    class_counts = {label: sum(videos[index]["expected_violation"] == label for index in indexes) for label in (False, True)}
    if min(class_counts.values()) == 0:
        raise ValueError("Training fold needs both video classes")
    for index in indexes:
        video = videos[index]
        if video["expected_violation"]:
            references = video.get("reference_times_sec") or ([video["reference_time_sec"]] if video.get("reference_time_sec") is not None else [])
            if not references:
                raise ValueError("Positive dev sample lacks timestamp")
            distances = np.min(np.abs(video["centers"][:, None] - np.asarray(references)[None, :]), axis=1)
            selected = np.flatnonzero(distances <= 2.0 + 1e-6)
            if not len(selected):
                selected = np.unique([int(np.argmin(np.abs(video["centers"] - reference))) for reference in references])
        else:
            selected = np.arange(len(video["features"]))
        each_weight = 1.0 / (class_counts[video["expected_violation"]] * len(selected))
        for row in selected:
            x.append(video["features"][row])
            y.append(int(video["expected_violation"]))
            weights.append(each_weight)
            groups.append(video["sample_id"])
    weights = np.asarray(weights, dtype=np.float64)
    weights *= len(weights) / weights.sum()
    return np.asarray(x), np.asarray(y), weights, groups


def fit_preprocessing(x):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(x)
    normalized = scaler.transform(x)
    pca = PCA(n_components=min(32, len(x) - 2, x.shape[1]), svd_solver="randomized", random_state=SEED).fit(normalized)
    return scaler, pca, pca.transform(normalized)


def metric_set(labels, scores, threshold):
    tp = sum(bool(label) and score >= threshold for label, score in zip(labels, scores))
    tn = sum(not label and score < threshold for label, score in zip(labels, scores))
    fp = sum(not label and score >= threshold for label, score in zip(labels, scores))
    fn = sum(bool(label) and score < threshold for label, score in zip(labels, scores))
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "precision": tp / (tp + fp) if tp + fp else None,
            "recall": recall, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "balanced_accuracy": (recall + specificity) / 2, "accuracy": (tp + tn) / len(labels)}


def choose_threshold(labels, scores):
    distinct = sorted(set(float(score) for score in scores))
    thresholds = [0.0, 1.0] + [(left + right) / 2 for left, right in zip(distinct, distinct[1:])]
    candidates = [(threshold, metric_set(labels, scores, threshold)) for threshold in thresholds]
    return max(candidates, key=lambda item: (item[1]["balanced_accuracy"], item[1]["f1"], -item[1]["fp"], item[0]))


def make_folds(videos, count=5):
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold
    labels = np.asarray([int(video["expected_violation"]) for video in videos])
    groups = np.asarray([video["video_sha256"] for video in videos])
    label_sets = {group: {int(labels[index]) for index in range(len(videos)) if groups[index] == group} for group in set(groups)}
    if any(len(values) != 1 for values in label_sets.values()):
        raise ValueError("Identical video contents have conflicting labels")
    splits = min(count, sum(values == {1} for values in label_sets.values()), sum(values == {0} for values in label_sets.values()))
    if splits < 2:
        raise ValueError("Too few dev videos for grouped validation")
    result = list(StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=SEED).split(np.zeros(len(videos)), labels, groups))
    for train, test in result:
        if {videos[i]["video_sha256"] for i in train} & {videos[i]["video_sha256"] for i in test}:
            raise ValueError("Video overlap between train and validation")
    return result


def train(videos, encoder, manifest, output, report_dir):
    import numpy as np
    import sklearn
    from importlib.metadata import version as package_version
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    started = time.monotonic()
    folds = make_folds(videos)
    oof = {c: {} for c in CS}
    fold_records = []
    with threadpool_limits(limits=3):
        for fold_index, (train_indices, test_indices) in enumerate(folds):
            x, y, weights, groups = training_rows(videos, train_indices)
            scaler, pca, reduced = fit_preprocessing(x)
            transformed = {index: pca.transform(scaler.transform(videos[index]["features"])) for index in test_indices}
            fold_records.append({"fold": fold_index, "train_video_ids": [videos[i]["sample_id"] for i in train_indices],
                                 "validation_video_ids": [videos[i]["sample_id"] for i in test_indices],
                                 "train_video_hashes": [videos[i]["video_sha256"] for i in train_indices],
                                 "validation_video_hashes": [videos[i]["video_sha256"] for i in test_indices],
                                 "train_windows": len(x), "pca_dimensions": pca.n_components_})
            for c in CS:
                classifier = LogisticRegression(C=c, solver="liblinear", max_iter=2000, random_state=SEED).fit(reduced, y, sample_weight=weights)
                for index in test_indices:
                    oof[c][int(index)] = classifier.predict_proba(transformed[index])[:, 1]
            print(json.dumps({"validation_folds_completed": fold_index + 1, "total_folds": len(folds)}), flush=True)
        labels = [video["expected_violation"] for video in videos]
        choices = []
        for c in CS:
            video_scores = [float(oof[c][index].max()) for index in range(len(videos))]
            threshold, metrics = choose_threshold(labels, video_scores)
            choices.append({"C": c, "threshold": threshold, "metrics": metrics})
        selected = max(choices, key=lambda item: (item["metrics"]["balanced_accuracy"], item["metrics"]["f1"], -item["metrics"]["fp"], -item["C"]))
        chosen_c, threshold = selected["C"], selected["threshold"]
        x, y, weights, _ = training_rows(videos, range(len(videos)))
        scaler, pca, reduced = fit_preprocessing(x)
        classifier = LogisticRegression(C=chosen_c, solver="liblinear", max_iter=2000, random_state=SEED).fit(reduced, y, sample_weight=weights)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "weights.npz", scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, pca_mean=pca.mean_,
                        pca_components=pca.components_, coef=classifier.coef_, intercept=classifier.intercept_)
    source_hashes = {name: hash_file(ROOT / name) for name in ("visual_baseline.py", "train_visual_baseline.py")}
    config = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(), "encoder_fingerprint": encoder.fingerprint,
              "weights_sha256": hash_file(output / "weights.npz"), "threshold": threshold, "C": chosen_c,
              "window_span_sec": 8, "sample_interval_sec": 2, "positive_training_radius_sec": 2,
              "features": "L2-normalized CLS: mean, standard deviation, absolute last-first difference",
              "train_split": "dev", "training_videos": len(videos), "training_windows": len(x), "pca_dimensions": int(pca.n_components_),
              "split_fingerprint": manifest["split_fingerprint"], "source_sha256": source_hashes,
              "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "torch": encoder.torch.__version__,
                           "transformers": package_version("transformers"), "pillow": package_version("Pillow")},
              "purpose": "Candidate detection only; no violation type or legal conclusion"}
    config["fingerprint"] = canonical_hash(config)
    save_json(output / "config.json", config)
    details = []
    time_hits = positive_total = 0
    for index, video in enumerate(videos):
        scores = oof[chosen_c][index]
        predicted = float(scores.max()) >= threshold
        positive_total += bool(video["expected_violation"])
        candidates = select_candidates(video["centers"], scores, threshold)
        references = video.get("reference_times_sec") or ([video["reference_time_sec"]] if video.get("reference_time_sec") is not None else [])
        hit = bool(video["expected_violation"] and predicted and any(abs(candidate["timecode"] - reference) <= 5 for candidate in candidates for reference in references))
        time_hits += hit
        details.append({"sample_id": video["sample_id"], "expected_violation": video["expected_violation"],
                        "video_max_score": float(scores.max()), "predicted_violation": predicted, "timecode_hit_5s": hit,
                        "centers": video["centers"].tolist(), "window_scores": scores.tolist(), "candidates": candidates})
    summary = {"dev_videos": len(videos), "positive": sum(labels), "negative": len(labels) - sum(labels), "folds": len(folds),
               "selected_C": chosen_c, "selected_threshold": threshold, "out_of_fold_tuned": selected["metrics"],
               "timecode_hits_within_5s": time_hits, "positive_with_timestamp": positive_total,
               "training_seconds": round(time.monotonic() - started, 3), "holdout_evaluated": 0,
               "feature_extraction_seconds": round(sum(video.get("feature_seconds", 0) for video in videos), 3),
               "feature_frames": sum(len(video["centers"]) for video in videos),
               "limitation": "C and threshold selected from these same out-of-fold scores; estimate is optimistic. No independent holdout result yet.",
               "grouping_limitation": "Groups are unique video SHA-256 contents, not verified independent cameras or customer sessions; related recordings may be correlated.",
               "leakage_guards": ["Only fixed dev videos used for fitting and threshold selection", "Videos grouped by SHA-256 content between training and validation", "Exact duplicates collapsed before fitting, all positive reference times retained", "Scaler and PCA fitted on training folds only", "Positive windows only near supplied dev reference times", "Other positive-video windows excluded, never labeled negative", "Threshold optimized on video-max out-of-fold scores", "No filenames or annotation timestamps are inference inputs", "Frozen public encoder, CPU inference, no remote media processing"],
               "split_fingerprint": manifest["split_fingerprint"], "model_fingerprint": config["fingerprint"]}
    save_json(report_dir / "summary.json", summary)
    save_json(report_dir / "selection.json", choices)
    save_json(report_dir / "folds.json", fold_records)
    save_json(report_dir / "out_of_fold.json", details)
    m = selected["metrics"]
    precision_text = "не определено" if m["precision"] is None else f"{m['precision']:.1%}"
    report = f"""# KFM — обученный визуальный детектор кандидатов

Обучение выполнено локально на {len(videos)} видео dev. Holdout не использовался.

Проверка разделяет целые видео по SHA-256 содержимого между {len(folds)} блоками; кадры одного видео не попадают одновременно в обучение и проверку. Точные копии исключены, все положительные отметки времени объединены. Нормализация и PCA обучаются отдельно внутри каждого обучающего блока. Признаки извлекает замороженная DINOv2-small на CPU.

| Показатель на dev, вне обучающего блока | Результат |
|---|---:|
| Сбалансированная точность | {m['balanced_accuracy']:.1%} |
| Precision | {precision_text} |
| Recall | {m['recall']:.1%} |
| F1 | {m['f1']:.1%} |
| TP / TN / FP / FN | {m['tp']} / {m['tn']} / {m['fp']} / {m['fn']} |
| Таймкод найден с допуском ±5 секунд | {time_hits} / {positive_total} |

Порог {threshold:.6f} и C={chosen_c:g} выбраны по сбалансированной точности целых видео; F1 используется при равенстве. Эти параметры подбирались по тем же результатам вне обучающего блока, поэтому оценка оптимистична и требует отдельной проверки holdout. Это не доказательство работы на новых камерах или новых видах нарушения.

Детектор возвращает только время и оценку сходства. Он не определяет юридическое нарушение и не может сам обосновать ссылку на правило. Для будущего локального анализа используется predict_candidates(path, duration).

Ни реальные кадры, ни их описания не передавались облачным сервисам.
"""
    (report_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary), flush=True)


def compare_cached(videos, encoder, manifest, report_dir, budget_sec=240):
    """Bounded dev-only comparison; holdout is deliberately not an argument."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.svm import SVC
    from threadpoolctl import threadpool_limits
    started = time.monotonic()
    folds = make_folds(videos)
    labels = np.asarray([video["expected_violation"] for video in videos], dtype=bool)

    def l2(x):
        x = np.asarray(x, dtype=np.float64)
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)

    def video_vector(video):
        frames = l2(video["embeddings"])
        return np.concatenate((frames.mean(axis=0), frames.std(axis=0), frames.max(axis=0)))

    whole_video = l2([video_vector(video) for video in videos])
    windows = [l2(video["features"]) for video in videos]
    designs = []
    for scope in ("window", "video"):
        designs += [{"scope": scope, "estimator": "logistic", "C": c} for c in (1e-4, 1e-3, .01, .1, 1.0)]
        designs += [{"scope": scope, "estimator": "linear_svm", "C": c} for c in (.001, .01, .1, 1.0)]
        designs += [{"scope": scope, "estimator": "rbf_svm", "C": c} for c in (.01, .1, 1.0)]
        designs += [{"scope": scope, "estimator": "ridge", "alpha": alpha} for alpha in (.1, 1., 10., 100.)]

    def estimator(design):
        if design["estimator"] == "logistic":
            return LogisticRegression(C=design["C"], solver="liblinear", max_iter=2000, random_state=SEED)
        if design["estimator"] == "ridge":
            return RidgeClassifier(alpha=design["alpha"], solver="lsqr")
        return SVC(C=design["C"], kernel="rbf" if design["estimator"] == "rbf_svm" else "linear", gamma="scale", probability=False, random_state=SEED)

    def data(indices, scope):
        if scope == "window":
            x, y, weights, _ = training_rows(videos, indices)
            return l2(x), y, weights
        y = labels[indices].astype(int)
        counts = np.bincount(y, minlength=2)
        weights = np.asarray([len(y) / (2 * counts[value]) for value in y])
        return whole_video[indices], y, weights

    comparisons, stored_scores = [], {}
    with threadpool_limits(limits=3):
        for number, design in enumerate(designs):
            if time.monotonic() - started > budget_sec:
                break
            oof, oof_windows = np.full(len(videos), np.nan), {}
            for train_indices, test_indices in folds:
                x, y, weights = data(train_indices, design["scope"])
                fitted = estimator(design).fit(x, y, sample_weight=weights)
                for index in test_indices:
                    target = windows[index] if design["scope"] == "window" else whole_video[index:index + 1]
                    margins = fitted.decision_function(target)
                    scores = 1 / (1 + np.exp(-np.clip(margins, -60, 60)))
                    oof[int(index)] = float(scores.max())
                    oof_windows[int(index)] = scores
            threshold, metrics = choose_threshold(labels.tolist(), oof.tolist())
            row = {**design, "threshold": threshold, "tuned_metrics": metrics,
                   "fixed_threshold_metrics": metric_set(labels.tolist(), oof.tolist(), .5),
                   "video_auc": float(roc_auc_score(labels, oof)),
                   "score_range": [float(oof.min()), float(oof.max())]}
            if design["scope"] == "window":
                hits = 0
                for index, video in enumerate(videos):
                    references = video.get("reference_times_sec") or ([video["reference_time_sec"]] if video.get("reference_time_sec") is not None else [])
                    candidates = select_candidates(video["centers"], oof_windows[index], threshold)
                    hits += bool(video["expected_violation"] and any(abs(candidate["timecode"] - reference) <= 5 for candidate in candidates for reference in references))
                row["timecode_hits_within_5s"] = hits
            comparisons.append(row)
            stored_scores[number] = oof_windows
            print(json.dumps({"dev_comparisons_completed": len(comparisons), "planned": len(designs),
                              "balanced_accuracy": round(metrics["balanced_accuracy"], 4), "scope": design["scope"]}), flush=True)
        ranked = sorted(enumerate(comparisons), key=lambda item: (item[1]["tuned_metrics"]["balanced_accuracy"], item[1]["tuned_metrics"]["f1"], -item[1]["tuned_metrics"]["fp"]), reverse=True)
        best_window = next((item for item in ranked if item[1]["scope"] == "window"), None)
        best_video = next((item for item in ranked if item[1]["scope"] == "video"), None)
        exported = None
        if best_window and best_window[1]["tuned_metrics"]["balanced_accuracy"] >= .7 and best_window[1]["tuned_metrics"]["recall"] >= .7:
            number, selected = best_window
            x, y, weights = data(np.arange(len(videos)), "window")
            fitted = estimator(selected).fit(x, y, sample_weight=weights)
            arrays = {"intercept": fitted.intercept_}
            if selected["estimator"] == "rbf_svm":
                arrays.update(support_vectors=fitted.support_vectors_, dual_coef=fitted.dual_coef_, gamma=np.asarray([fitted._gamma]))
            else:
                arrays["coef"] = fitted.coef_
            destination = ROOT / ".models" / "visual-baseline-v2"
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination / "weights.npz", **arrays)
            config = {"version": VERSION, "model_revision": "v2", "preprocessing": "row_l2", "estimator": selected["estimator"],
                      "encoder_fingerprint": encoder.fingerprint, "weights_sha256": hash_file(destination / "weights.npz"),
                      "threshold": selected["threshold"], "window_span_sec": 8, "sample_interval_sec": 2,
                      "train_split": "dev", "training_videos": len(videos), "split_fingerprint": manifest["split_fingerprint"],
                      "selected_configuration": selected, "source_sha256": {name: hash_file(ROOT / name) for name in ("visual_baseline.py", "train_visual_baseline.py")},
                      "purpose": "Uncalibrated visual candidate ranking, not legal violation finding"}
            config["fingerprint"] = canonical_hash(config)
            save_json(destination / "config.json", config)
            exported = config["fingerprint"]
    summary = {"dev_videos": len(videos), "configurations_tested": len(comparisons), "elapsed_sec": round(time.monotonic() - started, 3),
               "best_window": best_window[1] if best_window else None, "best_video_comparator": best_video[1] if best_video else None,
               "exported_v2_fingerprint": exported, "holdout_evaluated": 0, "split_fingerprint": manifest["split_fingerprint"],
               "limitation": "Exploratory comparison and threshold selection reuse the same dev out-of-fold scores. Ranking is optimistic; fixed-threshold metrics expose calibration sensitivity. Video comparator provides no event localization."}
    save_json(report_dir / "comparison.json", comparisons)
    save_json(report_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-encoder", action="store_true")
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--manifest", type=Path, default=ROOT / "local_review" / "evaluation_split_deduplicated.json")
    parser.add_argument("--dataset", type=Path, default=ROOT.parent / "open")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=ROOT / "local_review" / "visual_baseline")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Bounded cached dev-only alternatives; preserves v1")
    args = parser.parse_args(argv)
    if args.download_encoder:
        download_encoder(args.encoder)
        return
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples = dev_samples(manifest)
    if len({sample["video_sha256"] for sample in samples}) != len(samples):
        raise ValueError("Training manifest must collapse exact duplicate contents first")
    encoder = LocalEncoder(args.encoder, threads=args.threads)
    from server import read_metadata
    videos = []
    begin = time.monotonic()
    for index, sample in enumerate(samples):
        feature_started = time.monotonic()
        path = args.dataset / "videos" / sample["filename"]
        if hash_file(path) != sample["video_sha256"]:
            raise ValueError("Dataset video changed after split creation")
        duration = read_metadata(path)["duration_sec"]
        embeddings, timestamps = extract_embeddings(path, duration, encoder)
        features, centers = temporal_features(embeddings, timestamps)
        videos.append({"sample_id": sample["sample_id"], "expected_violation": sample["expected_violation"],
                       "reference_time_sec": sample["reference_time_sec"], "features": features, "centers": centers,
                       "reference_times_sec": sample.get("reference_times_sec", []), "video_sha256": sample["video_sha256"],
                       "embeddings": embeddings,
                       "feature_seconds": time.monotonic() - feature_started})
        print(json.dumps({"dev_features_completed": index + 1, "total_dev": len(samples), "frames": len(timestamps),
                          "elapsed_sec": round(time.monotonic() - begin, 2)}), flush=True)
    if args.compare:
        compare_cached(videos, encoder, manifest, ROOT / "local_review" / "visual_baseline_tuning")
    elif not args.features_only:
        encoder.load()
        train(videos, encoder, manifest, args.output, args.report)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"baseline_error": type(exc).__name__}), file=sys.stderr, flush=True)
        raise SystemExit(1)
