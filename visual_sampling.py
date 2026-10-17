"""Local video sampling and contact sheets. Never sends media to a network."""
from __future__ import annotations

import math
from pathlib import Path
import subprocess
import time

from PIL import Image, ImageDraw, ImageFont, ImageOps

from analyzers import AnalysisError


def remaining(deadline, minimum=0.1):
    value = deadline - time.monotonic()
    if value < minimum:
        raise AnalysisError("analysis_timeout", "Не все фрагменты обработаны за отведённое время. Неполная проверка не считается отсутствием нарушений.")
    return value


def sample_windows(duration, span=24.0, count=9):
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise AnalysisError("unknown_video_duration", "Не удалось определить длительность видео.")
    if duration > 300:
        raise AnalysisError("video_too_long", "Экспериментальная версия поддерживает записи до пяти минут.")
    end = max(0, duration - min(0.1, duration / 2))
    starts = [0.0]
    while starts[-1] + span < end:
        starts.append(starts[-1] + span - 3)
    return [sorted({round(start + (min(end, start + span) - start) * i / (count - 1), 3) for i in range(count)}) for start in starts]


class LocalVideo:
    def __init__(self, path, ffmpeg, folder, deadline):
        self.path, self.ffmpeg, self.folder = Path(path), Path(ffmpeg), Path(folder)
        self.deadline = deadline
        self.folder.mkdir(parents=True, exist_ok=True)
        self.cache = {}

    def frame(self, stamp):
        stamp = round(float(stamp), 3)
        if stamp in self.cache:
            return self.cache[stamp]
        remaining(self.deadline)
        destination = self.folder / f"frame-{stamp:010.3f}.jpg"
        args = [str(self.ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-protocol_whitelist", "file,pipe", "-ss", f"{stamp:.3f}", "-i", str(self.path), "-frames:v", "1", "-vf", "scale=1280:1280:force_original_aspect_ratio=decrease", "-q:v", "2", "-threads", "1", "-y", str(destination)]
        try:
            response = subprocess.run(args, capture_output=True, timeout=min(20, remaining(self.deadline)), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            raise AnalysisError("video_decode_timeout", "Превышено время чтения видеокадра.") from None
        if response.returncode or not destination.is_file():
            raise AnalysisError("video_decode_error", "Не удалось прочитать кадр видеозаписи.")
        remaining(self.deadline)
        try:
            with Image.open(destination) as frame:
                frame.verify()
        except (OSError, ValueError):
            raise AnalysisError("video_decode_error", "Получен повреждённый видеокадр.") from None
        self.cache[stamp] = destination
        return destination

    def frames(self, stamps):
        return [self.frame(stamp) for stamp in stamps]


def _font(size):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def storyboard(paths, stamps, destination, columns=3, tile_width=448, hero=None):
    """Synthetic layout for local inference; evidence always uses originals."""
    if not paths or len(paths) != len(stamps):
        raise ValueError("A storyboard needs paired paths and timestamps")
    with Image.open(paths[0]) as first:
        ratio = first.height / first.width
    # Portrait cameras must not be squeezed into square cells: the empty
    # padding otherwise consumes most of the vision encoder's pixel budget.
    if ratio > 1:
        tile_width = min(tile_width, 320)
    tile_height = max(168, round(tile_width * ratio))
    label = 30
    width = columns * tile_width
    entries = list(range(len(paths)))
    left = 0
    hero_width = hero_height = 0
    if hero is not None:
        if not isinstance(hero, int) or not 0 <= hero < len(paths):
            raise ValueError("Invalid large-frame index")
        columns, tile_width = 2, 256 if ratio > 1 else 320
        tile_height = round(tile_width * ratio)
        hero_width = 720 if ratio > 1 else 800
        hero_height = round(hero_width * ratio)
        left = hero_width
        entries.remove(hero)
        width = hero_width + columns * tile_width
    height = max(hero_height + label, math.ceil(len(entries) / columns) * (tile_height + label))
    canvas = Image.new("RGB", (width, height), "#11151c")
    draw = ImageDraw.Draw(canvas)
    font = _font(22)
    if hero is not None:
        with Image.open(paths[hero]) as original:
            image = ImageOps.contain(original.convert("RGB"), (hero_width, hero_height), Image.Resampling.LANCZOS)
            canvas.paste(image, ((hero_width-image.width)//2, 0))
        draw.text((8, hero_height+2), f"{hero} | {stamps[hero]:.2f}s | LARGE VIEW", fill="white", font=font)
    for cell, index in enumerate(entries):
        path, stamp = paths[index], stamps[index]
        x, y = left + cell % columns * tile_width, cell // columns * (tile_height + label)
        with Image.open(path) as original:
            image = ImageOps.contain(original.convert("RGB"), (tile_width, tile_height), Image.Resampling.LANCZOS)
            canvas.paste(image, (x + (tile_width-image.width)//2, y + (tile_height-image.height)//2))
        draw.text((x+8, y+tile_height+2), f"{index} | {stamp:.2f}s", fill="white", font=font)
    canvas.save(destination, "JPEG", quality=91)
    return destination
