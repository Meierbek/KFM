"""Synthetic geometry/selection tests; no private dataset or model calls."""
import math
from pathlib import Path
import shutil
import unittest
import uuid

from yolo_candidates import ROOT, clip_box, detections_to_proposals, rank_proposals, sampling_times, save_context_crop


class ObjectProposalTests(unittest.TestCase):
    def test_whitelist_and_invalid_predictions(self):
        rows = [[1, 2, 80, 90, .8, 67], [1, 2, 80, 90, .9, 0], [1, 2, 80, 90, .1, 41],
                [1, 2, 80, 90, math.nan, 67], [1, 2, 80, 90, .8, 67.5], [1, 2, 1, 90, .8, 67]]
        result = detections_to_proposals(rows, 12, 100, 100)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["kind"], "phone")
        self.assertNotIn("verdict", result[0])
        self.assertNotIn("rule_id", result[0])

    def test_sampling_independent_of_labels(self):
        self.assertEqual(sampling_times(9, 4), [0, 4, 8])
        self.assertEqual(sampling_times(.1), [0])
        for duration in (0, 301, float("nan"), True):
            with self.assertRaises(ValueError):
                sampling_times(duration)

    def test_box_clamping(self):
        self.assertEqual(clip_box([-2, -4, 120, 60], 100, 50), [0, 0, 100, 50])
        self.assertIsNone(clip_box([80, 0, 40, 50], 100, 50))

    def test_classwise_temporal_selection(self):
        proposals = [{"timecode": t, "kind": kind, "score": score} for t, kind, score in
                     [(1, "cup", .7), (3, "cup", .9), (4, "phone", .8), (11, "cup", .6)]]
        selected = rank_proposals(proposals, limit=3)
        self.assertEqual([(p["timecode"], p["kind"]) for p in selected], [(3, "cup"), (4, "phone"), (11, "cup")])

    def test_context_crop_stays_inside_frame(self):
        from PIL import Image
        folder = ROOT / ".qa" / ("yolo-test-" + uuid.uuid4().hex)
        folder.mkdir(parents=True)
        try:
            target = folder / "crop.jpg"
            box = save_context_crop(Image.new("RGB", (300, 200), "white"), [280, 180, 299, 199], target)
            self.assertTrue(0 <= box[0] < box[2] <= 300 and 0 <= box[1] < box[3] <= 200)
            with Image.open(target) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertGreaterEqual(image.width, 2)
        finally:
            self.assertEqual(folder.resolve().parent, (ROOT / ".qa").resolve())
            shutil.rmtree(folder)

    def test_same_class_different_objects_survive(self):
        proposals = [{"timecode": 1, "kind": "cup", "score": .9, "bbox_normalized": [0, 0, .1, .1]},
                     {"timecode": 1, "kind": "cup", "score": .8, "bbox_normalized": [.8, .8, 1, 1]}]
        self.assertEqual(len(rank_proposals(proposals)), 2)


if __name__ == "__main__":
    unittest.main()
