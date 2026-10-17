"""Frozen local DINOv2 features plus a supervised candidate detector.

The numeric score is a learned similarity signal, not a legal finding. This
module never reads labels, reference timecodes, filenames, or prior verdicts.
All encoder inference is offline and CPU-only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from analyzers import AnalysisError

ROOT = Path(__file__).resolve().parent
DEPENDENCIES = ROOT / ".ml-deps"
if DEPENDENCIES.exists() and str(DEPENDENCIES) not in sys.path:
    sys.path.insert(0, str(DEPENDENCIES))
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "3")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

VERSION = "dino-temporal-v1"
DEFAULT_ENCODER = ROOT / ".models" / "dinov2-small"
DEFAULT_MODEL = ROOT / ".models" / "visual-baseline-v1"
SAMPLE_INTERVAL = 2.0
WINDOW_SPAN = 8.0


def remaining(deadline):
    if deadline is None:
        return 3600.0
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise AnalysisError("analysis_timeout", "Превышено время локального поиска подозрительных фрагментов.")
    return seconds


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sample_times(duration, interval=SAMPLE_INTERVAL):
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 < duration <= 300:
        raise ValueError("Video duration must be in (0,300]")
    if interval <= 0:
        raise ValueError("Invalid sampling interval")
    last = max(0.0, duration - min(.1, duration / 2))
    return [round(index * interval, 3) for index in range(int(last / interval) + 1)]


def temporal_features(embeddings, timestamps, span=WINDOW_SPAN):
    import numpy as np
    vectors = np.asarray(embeddings, dtype=np.float32)
    times = np.asarray(timestamps, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != len(times) or not len(times) or not np.isfinite(vectors).all() or not np.isfinite(times).all():
        raise ValueError("Invalid frame embeddings")
    if not np.all(times[1:] > times[:-1]):
        raise ValueError("Timestamps must be increasing")
    normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    features = []
    for center in times:
        window = normalized[np.abs(times - center) <= span / 2 + 1e-6]
        features.append(np.concatenate((window.mean(axis=0), window.std(axis=0), np.abs(window[-1] - window[0]))))
    return np.asarray(features, dtype=np.float32), times.copy()


class LocalEncoder:
    def __init__(self, directory=DEFAULT_ENCODER, *, threads=3, batch_size=6):
        self.directory = Path(directory)
        self.threads = max(1, min(4, int(threads)))
        self.batch_size = max(1, min(12, int(batch_size)))
        self.model = self.processor = self.torch = None
        manifest_file = self.directory / "encoder.json"
        if not manifest_file.is_file():
            raise ValueError("Download the local encoder before extracting features")
        self.manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
            if hash_file(self.directory / name) != self.manifest["files"][name]["sha256"]:
                raise ValueError("Local encoder checksum mismatch")
        self.fingerprint = canonical_hash({"encoder": self.manifest, "version": VERSION,
                                           "sampling_interval": SAMPLE_INTERVAL, "resize": "full-frame-letterbox-224"})

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModel
        torch.set_num_threads(self.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(self.directory, local_files_only=True, use_fast=False)
        self.model = AutoModel.from_pretrained(self.directory, local_files_only=True, trust_remote_code=False,
                                              use_safetensors=True).to("cpu").eval().requires_grad_(False)

    def encode(self, paths, *, deadline=None):
        import numpy as np
        from PIL import Image, ImageOps
        remaining(deadline)
        self.load()
        remaining(deadline)
        output = []
        for start in range(0, len(paths), self.batch_size):
            remaining(deadline)
            images = []
            for path in paths[start:start + self.batch_size]:
                with Image.open(path) as image:
                    images.append(ImageOps.pad(image.convert("RGB"), (224, 224), method=Image.Resampling.BICUBIC,
                                               color=(124, 116, 104)))
            inputs = self.processor(images=images, return_tensors="pt", do_resize=False, do_center_crop=False)
            with self.torch.inference_mode():
                vectors = self.model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
            output.append(vectors)
            remaining(deadline)
        return np.concatenate(output, axis=0).astype(np.float32)


def extract_embeddings(path, duration, encoder, *, cache=None, ffmpeg=None, deadline=None):
    """Cache contains only pretrained image embeddings, never labels/verdicts."""
    import numpy as np
    remaining(deadline)
    path = Path(path)
    cache_root = Path(cache or ROOT / "local_review" / "visual_features")
    cache_root.mkdir(parents=True, exist_ok=True)
    times = sample_times(duration)
    key = canonical_hash({"content": hash_file(path), "encoder": encoder.fingerprint, "timestamps": times})
    destination = cache_root / (key + ".npz")
    if destination.exists():
        with np.load(destination, allow_pickle=False) as stored:
            vectors, saved_times = stored["embeddings"], stored["timestamps"]
        if list(saved_times) == times and len(vectors) == len(times) and np.isfinite(vectors).all():
            remaining(deadline)
            return vectors, saved_times
        raise ValueError("Feature cache integrity failure")
    executable = Path(ffmpeg or ROOT / ".tools" / "ffmpeg.exe")
    scratch_root = ROOT / "data" / "visual_features"
    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch = scratch_root / uuid.uuid4().hex
    scratch.mkdir()
    try:
        paths = []
        for index, stamp in enumerate(times):
            remaining(deadline)
            frame = scratch / f"frame-{index:05d}.jpg"
            command = [str(executable), "-nostdin", "-hide_banner", "-loglevel", "error", "-protocol_whitelist", "file,pipe",
                       "-ss", str(stamp), "-i", str(path), "-frames:v", "1", "-vf", "scale=448:448:force_original_aspect_ratio=decrease",
                       "-q:v", "3", "-threads", "1", "-y", str(frame)]
            try:
                result = subprocess.run(command, capture_output=True, timeout=min(25, remaining(deadline)),
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except subprocess.TimeoutExpired:
                raise AnalysisError("analysis_timeout", "Превышено время чтения видеокадров.") from None
            if result.returncode or not frame.is_file():
                raise ValueError("Local frame decoding failed")
            paths.append(frame)
        vectors = encoder.encode(paths, deadline=deadline)
        remaining(deadline)
        pending = destination.with_suffix(".pending.npz")
        np.savez_compressed(pending, embeddings=vectors, timestamps=np.asarray(times, dtype=np.float64))
        pending.replace(destination)
        return vectors, np.asarray(times)
    finally:
        if scratch.resolve().parent != scratch_root.resolve():
            raise ValueError("Invalid scratch path")
        shutil.rmtree(scratch)


def linear_probabilities(features, weights):
    import numpy as np
    normalized = (np.asarray(features, dtype=np.float64) - weights["scaler_mean"]) / weights["scaler_scale"]
    reduced = (normalized - weights["pca_mean"]) @ weights["pca_components"].T
    margin = reduced @ weights["coef"].reshape(-1) + float(weights["intercept"].reshape(-1)[0])
    return 1 / (1 + np.exp(-np.clip(margin, -60, 60)))


def model_scores(features, weights, config):
    """Scores are ranking signals; SVM scores are not calibrated probabilities."""
    if config.get("preprocessing") != "row_l2":
        return linear_probabilities(features, weights)
    import numpy as np
    x = np.asarray(features, dtype=np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    if config["estimator"] == "rbf_svm":
        support = weights["support_vectors"]
        squared = np.maximum((x * x).sum(axis=1)[:, None] + (support * support).sum(axis=1)[None, :] - 2 * x @ support.T, 0)
        kernel = np.exp(-float(weights["gamma"].reshape(-1)[0]) * squared)
        margin = kernel @ weights["dual_coef"].reshape(-1) + float(weights["intercept"].reshape(-1)[0])
    else:
        margin = x @ weights["coef"].reshape(-1) + float(weights["intercept"].reshape(-1)[0])
    return 1 / (1 + np.exp(-np.clip(margin, -60, 60)))


def select_candidates(centers, scores, threshold, *, separation=6.0, limit=5):
    pairs = [(float(stamp), float(score)) for stamp, score in zip(centers, scores)
             if math.isfinite(float(score)) and float(score) >= threshold]
    chosen = []
    for stamp, score in sorted(pairs, key=lambda item: (-item[1], item[0])):
        if all(abs(stamp - old["timecode"]) >= separation for old in chosen):
            chosen.append({"timecode": round(stamp, 3), "score": round(score, 6)})
            if len(chosen) >= limit:
                break
    return sorted(chosen, key=lambda item: item["timecode"])


class VisualBaseline:
    """Candidate ranking only. It supplies no violation type or legal citation."""
    def __init__(self, directory=DEFAULT_MODEL, *, encoder_directory=DEFAULT_ENCODER):
        import numpy as np
        self.directory = Path(directory)
        self.config = json.loads((self.directory / "config.json").read_text(encoding="utf-8"))
        if self.config.get("version") != VERSION:
            raise ValueError("Unsupported visual baseline version")
        if self.config.get("window_span_sec") != WINDOW_SPAN or self.config.get("sample_interval_sec") != SAMPLE_INTERVAL:
            raise ValueError("Visual sampling configuration differs from training")
        with np.load(self.directory / "weights.npz", allow_pickle=False) as arrays:
            self.weights = {key: arrays[key].copy() for key in arrays.files}
        if hash_file(self.directory / "weights.npz") != self.config["weights_sha256"]:
            raise ValueError("Visual baseline weights changed")
        self.encoder = LocalEncoder(encoder_directory)
        if self.encoder.fingerprint != self.config["encoder_fingerprint"]:
            raise ValueError("Visual encoder differs from training")
        self.threshold = float(self.config["threshold"])

    def score_windows(self, path, duration, *, deadline=None):
        vectors, timestamps = extract_embeddings(path, duration, self.encoder, deadline=deadline)
        features, centers = temporal_features(vectors, timestamps)
        return centers, model_scores(features, self.weights, self.config)

    def predict_candidates(self, path, duration, *, deadline=None):
        centers, scores = self.score_windows(path, duration, deadline=deadline)
        return select_candidates(centers, scores, self.threshold)


_default_baseline = None


def predict_candidates(path, duration, *, deadline=None):
    global _default_baseline
    if _default_baseline is None:
        _default_baseline = VisualBaseline()
    return _default_baseline.predict_candidates(path, duration, deadline=deadline)
