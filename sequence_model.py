"""Separate-frame local Gemma input; uses the same grounded verification gate.

Original decoded frames alone are used as evidence. Numbered, resized pictures
below are transient model inputs and never become evidence or leave localhost.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from refined_model import RefinedVideoAnalyzer


class SequenceVideoAnalyzer(RefinedVideoAnalyzer):
    version = "sequence-v1"

    def __init__(self, *, model=None, endpoint=None):
        super().__init__(model=model or "gemma3:4b-it-q4_K_M", endpoint=endpoint)
        self.name = f"{self.model} · последовательная проверка кадров"

    def visual_input(self, paths, stamps, destination, hero=None, overview=False):
        if not paths or len(paths) != len(stamps):
            raise ValueError("Separate frames need paired paths and timestamps")
        destination = Path(destination)
        output = []
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 21)
        except OSError:
            font = ImageFont.load_default(size=21)
        for index, (path, stamp) in enumerate(zip(paths, stamps)):
            with Image.open(path) as image:
                frame = ImageOps.contain(image.convert("RGB"), (768, 768))
                canvas = Image.new("RGB", (frame.width, frame.height + 32), "#111111")
                canvas.paste(frame, (0, 32))
                ImageDraw.Draw(canvas).text((8, 3), f"FRAME {index} | {stamp:.3f} s", font=font, fill="white")
                target = destination.with_name(f"{destination.stem}-{index}.jpg")
                canvas.save(target, "JPEG", quality=92)
                output.append(target)
        return output

    def infer(self, image, prompt, schema, deadline, tokens=260):
        prompt = ("Each numbered contextual image is ONE video frame. Read its printed FRAME index; numbered context frames are chronological. "
                  "A preceding detail crop, if explicitly described below, belongs to its stated frame and is not an additional moment. "
                  "Compare context frames as successive moments, not simultaneous different cameras. " + prompt)
        return super().infer(image, prompt, schema, deadline, tokens)

    def finalize_result(self, result):
        result["analysis"]["strategy"] = "separate_temporal_frames"
        return result
