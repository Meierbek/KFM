"""Offline CPU YOLOv8n object proposals, never violation or ownership verdicts.

The optional third-party detector/weights are AGPL-3.0 (see YOLO_NOTICE.md).
Input frames, detections and crops remain on disk on this computer.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
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

ROOT = Path(__file__).resolve().parent
for _folder in (ROOT / ".ml-deps", ROOT / ".yolo-deps"):
    if _folder.exists() and str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

MODEL_DIRECTORY = ROOT / ".models" / "yolov8n"
OFFICIAL_WEIGHT_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
CLASS_NAMES = {24: "backpack", 26: "handbag", 39: "bottle", 41: "cup", 67: "phone"}
VERSION = "yolov8n-object-proposals-v1"


def check_deadline(deadline):
    if deadline is None:
        return 25.0
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        from analyzers import AnalysisError
        raise AnalysisError("analysis_timeout", "Превышено время локального поиска объектов.")
    return min(25.0, remaining)


def sampling_times(duration, interval=4.0):
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 < duration <= 300:
        raise ValueError("Invalid video duration")
    if not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval < 1:
        raise ValueError("Invalid sample interval")
    times = [round(i * interval, 3) for i in range(int(max(0, duration - .1) / interval) + 1)]
    end = round(max(0, duration - min(.1, duration / 2)), 3)
    if end - times[-1] >= 1:
        times.append(end)
    return times


def clip_box(box, width, height):
    if len(box) != 4 or width <= 0 or height <= 0 or not all(math.isfinite(float(v)) for v in box):
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    bounds = [max(0, min(width, x1)), max(0, min(height, y1)), max(0, min(width, x2)), max(0, min(height, y2))]
    return bounds if bounds[2] - bounds[0] >= 2 and bounds[3] - bounds[1] >= 2 else None


def detections_to_proposals(rows, stamp, width, height, confidence=.3):
    """Rows are xyxy, confidence, COCO class ID; only five classes survive."""
    if not math.isfinite(float(stamp)) or stamp < 0 or not 0 <= confidence <= 1:
        raise ValueError("Invalid proposal threshold/time")
    result = []
    for row in rows:
        if len(row) < 6 or not all(math.isfinite(float(v)) for v in row[:6]):
            continue
        class_id, score = int(row[5]), float(row[4])
        if row[5] != class_id or class_id not in CLASS_NAMES or not confidence <= score <= 1:
            continue
        box = clip_box(row[:4], width, height)
        if box is None:
            continue
        result.append({"timecode": round(float(stamp), 3), "kind": CLASS_NAMES[class_id],
                       "score": round(score, 6), "bbox_xyxy": [round(v, 2) for v in box],
                       "bbox_normalized": [round(box[0] / width, 6), round(box[1] / height, 6),
                                           round(box[2] / width, 6), round(box[3] / height, 6)],
                       "frame_width": int(width), "frame_height": int(height), "source": VERSION})
    return result


def rank_proposals(proposals, *, limit=12, separation=6.0):
    """Keep high scores and avoid flooding the verifier with the same object."""
    if not isinstance(limit, int) or limit < 1 or not math.isfinite(separation) or separation < 0:
        raise ValueError("Invalid proposal selection")
    selected = []
    for proposal in sorted(proposals, key=lambda p: (-p["score"], p["timecode"], p["kind"])):
        def overlaps(old):
            a, b = proposal.get("bbox_normalized"), old.get("bbox_normalized")
            if a is None or b is None:
                return True
            overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
            return union > 0 and overlap / union >= .3
        if any(proposal["kind"] == old["kind"] and abs(proposal["timecode"] - old["timecode"]) < separation and overlaps(old) for old in selected):
            continue
        selected.append(proposal)
        if len(selected) >= limit:
            break
    return sorted(selected, key=lambda p: (p["timecode"], p["kind"]))


def save_context_crop(image, bbox, path, *, expansion=1.0, min_side=160):
    """Include context around a detection; preserve the full frame separately."""
    from PIL import ImageOps
    box = clip_box(bbox, *image.size)
    if box is None:
        raise ValueError("Invalid crop box")
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = max(min_side, (x2 - x1) * (1 + expansion)), max(min_side, (y2 - y1) * (1 + expansion))
    crop_box = (max(0, math.floor(cx - w / 2)), max(0, math.floor(cy - h / 2)),
                min(image.width, math.ceil(cx + w / 2)), min(image.height, math.ceil(cy + h / 2)))
    crop = image.convert("RGB").crop(crop_box)
    crop = ImageOps.contain(crop, (768, 768))
    crop.save(path, "JPEG", quality=92)
    return list(crop_box)


class LocalYoloCandidates:
    def __init__(self, directory=MODEL_DIRECTORY, *, confidence=.3, threads=2, image_size=960):
        self.directory = Path(directory).resolve()
        self.weights = self.directory / "yolov8n.pt"
        if not self.weights.is_file():
            raise ValueError("Local YOLO weights are not installed")
        actual = hashlib.sha256(self.weights.read_bytes()).hexdigest()
        if actual != OFFICIAL_WEIGHT_SHA256:
            raise ValueError("Local YOLO checkpoint hash differs from the approved official download")
        if not 0 <= confidence <= 1:
            raise ValueError("Invalid object confidence")
        self.confidence = float(confidence)
        if image_size not in (640, 960):
            raise ValueError("Supported detector image sizes are 640 and 960")
        self.image_size = int(image_size)
        self.threads = max(1, min(2, int(threads)))
        self.model = None
        self.last_diagnostics = {}

    def load(self):
        if self.model is not None:
            return
        # Set before importing: this suppresses DNS checks, autoinstall and telemetry.
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ["YOLO_AUTOINSTALL"] = "false"
        os.environ["YOLO_VERBOSE"] = "false"
        os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "data" / "yolo-settings")
        with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink), redirect_stderr(sink):
            import torch
            from ultralytics import YOLO, settings
            from ultralytics.utils import torch_utils
            settings.update({"sync": False, "hub": False, "wandb": False, "comet": False, "clearml": False})
            # Ultralytics select_device otherwise resets CPU thread count to 8.
            torch_utils.NUM_THREADS = self.threads
            torch.set_num_threads(self.threads)
            self.model = YOLO(str(self.weights), task="detect")
            self.model.to("cpu")

    def detect_image(self, image, stamp=0.0, *, deadline=None):
        check_deadline(deadline)
        self.load()
        check_deadline(deadline)
        with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink), redirect_stderr(sink):
            predictions = self.model.predict(image, device="cpu", imgsz=self.image_size, conf=self.confidence,
                                             classes=list(CLASS_NAMES), iou=.5, max_det=30,
                                             verbose=False, save=False, save_txt=False, save_crop=False,
                                             show=False, half=False, batch=1)
        check_deadline(deadline)
        rows = predictions[0].boxes.data.detach().cpu().tolist()
        return detections_to_proposals(rows, stamp, image.width, image.height, self.confidence)

    def propose_video(self, path, duration, *, output_dir=None, interval=4.0, limit=12, deadline=None):
        from PIL import Image
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError("A local video file is required")
        times = sampling_times(duration, interval)
        scratch_root = ROOT / "data" / "yolo-work"
        scratch_root.mkdir(parents=True, exist_ok=True)
        scratch = scratch_root / uuid.uuid4().hex
        scratch.mkdir()
        started = time.monotonic()
        self.last_diagnostics = {}
        try:
            proposals = []
            for index, stamp in enumerate(times):
                frame = scratch / f"frame-{index:04d}.jpg"
                command = [str(ROOT / ".tools" / "ffmpeg.exe"), "-nostdin", "-hide_banner", "-loglevel", "error",
                           "-protocol_whitelist", "file,pipe", "-ss", str(stamp), "-i", str(path),
                           "-frames:v", "1",
                           "-q:v", "2", "-threads", "1", "-y", str(frame)]
                decoded = subprocess.run(command, capture_output=True, timeout=check_deadline(deadline),
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if decoded.returncode or not frame.is_file():
                    raise ValueError("Local YOLO frame decoding failed")
                with Image.open(frame) as image:
                    found = self.detect_image(image.convert("RGB"), stamp, deadline=deadline)
                for item in found:
                    item["_frame"] = frame
                proposals.extend(found)
            selected = rank_proposals(proposals, limit=limit)
            destination = Path(output_dir).resolve() if output_dir is not None else None
            if destination:
                destination.mkdir(parents=True, exist_ok=True)
            for index, item in enumerate(selected):
                frame = item.pop("_frame")
                if destination:
                    # UUID filenames avoid overwriting another analysis in the same folder.
                    token = uuid.uuid4().hex
                    crop_path, frame_path = destination / f"{token}-crop.jpg", destination / f"{token}-frame.jpg"
                    with Image.open(frame) as image:
                        item["crop_bbox_xyxy"] = save_context_crop(image, item["bbox_xyxy"], crop_path)
                    shutil.copyfile(frame, frame_path)
                    item.update(crop_path=str(crop_path), frame_path=str(frame_path))
            self.last_diagnostics = {"frames": len(times), "candidates": len(selected), "raw_candidates": len(proposals),
                                     "elapsed_sec": round(time.monotonic() - started, 3)}
            return selected
        finally:
            if scratch.resolve().parent != scratch_root.resolve():
                raise RuntimeError("Invalid YOLO temporary path")
            shutil.rmtree(scratch)


def predict_candidates(path, duration, *, output_dir=None, deadline=None, **kwargs):
    return LocalYoloCandidates().propose_video(path, duration, output_dir=output_dir, deadline=deadline, **kwargs)
