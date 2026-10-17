"""Tests use synthetic metadata and files; no real video inference."""
import io
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

from analyzers import AnalysisError
from evaluate_dataset import ROOT, build_split, deduplicate_split, evaluate_one, parse_label, parse_timestamp, safe_metadata, summarize


class EvaluationTests(unittest.TestCase):
    def test_identical_content_cannot_cross_splits_or_count_twice(self):
        rows = [
            {"sample_id": "a", "filename": "a.mov", "video_sha256": "same", "split": "holdout", "expected_violation": True, "reference_time_sec": 2},
            {"sample_id": "b", "filename": "b.mov", "video_sha256": "same", "split": "dev", "expected_violation": True, "reference_time_sec": 8},
            {"sample_id": "c", "filename": "c.mov", "video_sha256": "different", "split": "holdout", "expected_violation": False, "reference_time_sec": None},
        ]
        clean = deduplicate_split(rows)
        self.assertEqual(len(clean), 2)
        self.assertEqual(clean[0]["sample_id"], "b")
        self.assertEqual(clean[0]["split"], "dev")
        self.assertEqual(clean[0]["reference_times_sec"], [2, 8])
        self.assertEqual(clean[1]["split"], "holdout")
        rows[0]["expected_violation"] = False
        with self.assertRaises(ValueError):
            deduplicate_split(rows)

    def test_timestamp_does_not_define_class(self):
        self.assertTrue(parse_label("нарушение"))
        self.assertFalse(parse_label("без нарушения"))
        self.assertIsNone(parse_timestamp(None))
        self.assertEqual(parse_timestamp("1минута 8 секунда"), 68)
        self.assertEqual(parse_timestamp("00:00"), 0)
        with self.assertRaises(ValueError):
            parse_label(None)

    def test_split_is_reproducible_stratified_and_prior_examples_stay_in_dev(self):
        rows = [{"filename": f"clip-{i:02d}.mov", "expected_violation": i < 22} for i in range(42)]
        forced = {"clip-00.mov", "clip-01.mov", "clip-22.mov", "clip-23.mov", "clip-24.mov"}
        first = build_split(rows, forced_dev=forced)
        second = build_split(list(reversed(rows)), forced_dev=forced)
        self.assertEqual(first, second)
        holdout = [row for row in first if row["split"] == "holdout"]
        self.assertEqual(len(holdout), 14)
        self.assertEqual(sum(row["expected_violation"] for row in holdout), 7)
        self.assertFalse(forced & {row["filename"] for row in holdout})
        self.assertEqual(len({row["sample_id"] for row in first}), 42)
        self.assertEqual({row["filename"] for row in first}, {row["filename"] for row in rows})
        self.assertNotEqual(first, build_split(rows, seed=13, forced_dev=forced))

    def test_unresolved_samples_reduce_end_to_end_metrics(self):
        rows = [
            {"expected_violation": True, "predicted_violation": True, "status": "done", "reference_time_sec": 8, "result": {"violations": [{"timecode_sec": 12}]}},
            {"expected_violation": True, "predicted_violation": False, "status": "done"},
            {"expected_violation": False, "predicted_violation": True, "status": "done"},
            {"expected_violation": False, "predicted_violation": False, "status": "done"},
            {"expected_violation": True, "status": "abstained", "reference_time_sec": 6},
            {"expected_violation": False, "status": "failed"},
        ]
        result = summarize(rows)
        self.assertEqual(result["confusion_matrix"], {"tp": 1, "tn": 1, "fp": 1, "fn": 1, "unresolved_positive": 1, "unresolved_negative": 1})
        self.assertAlmostEqual(result["coverage"], 2 / 3, places=6)
        self.assertAlmostEqual(result["end_to_end_accuracy"], 1 / 3, places=6)
        self.assertAlmostEqual(result["recall"], 1 / 3, places=6)
        self.assertEqual(result["resolved_only"]["recall"], .5)
        self.assertEqual(result["precision"], .5)
        self.assertAlmostEqual(result["f1"], .4, places=5)
        self.assertAlmostEqual(result["balanced_accuracy"], 1 / 3, places=6)
        self.assertEqual(result["timecode"]["recall"], .5)

    def test_no_answers_cannot_produce_good_accuracy(self):
        rows = [{"expected_violation": True, "status": "failed"}, {"expected_violation": False, "status": "abstained"}]
        result = summarize(rows)
        self.assertEqual(result["coverage"], 0)
        self.assertEqual(result["end_to_end_accuracy"], 0)
        self.assertEqual(result["balanced_accuracy"], 0)
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["resolved_only"]["accuracy"])
        self.assertIsNone(summarize([])["end_to_end_accuracy"])

    def test_inference_receives_neutral_copy_and_allowlisted_metadata(self):
        scratch = ROOT / ".qa" / ("evaluation-test-" + uuid.uuid4().hex)
        scratch.mkdir(parents=True)
        source = scratch / "label-violation-at-68s.mov"
        source.write_bytes(b"synthetic-video-placeholder")
        seen = {}

        class FakeAnalyzer:
            def analyze(self, path, metadata, progress):
                seen.update(path=path, metadata=metadata, bytes=path.read_bytes())
                print("Synthetic private description must be suppressed")
                path.write_bytes(b"changed copy")
                return {"verdict": "no_violation", "violations": []}

        try:
            with patch("evaluate_dataset.read_metadata", return_value={"duration_sec": 12, "width": 1280, "height": 720, "label": True, "filename": source.name, "reference_time_sec": 68}), patch("sys.stdout", new_callable=io.StringIO) as stdout:
                result = evaluate_one(FakeAnalyzer(), source, scratch / "inputs")
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(result["status"], "done")
            self.assertEqual(seen["path"].name, "input.mov")
            self.assertNotIn(source.stem, str(seen["path"]))
            self.assertEqual(seen["metadata"], {"duration_sec": 12, "width": 1280, "height": 720})
            self.assertEqual(source.read_bytes(), b"synthetic-video-placeholder")
            self.assertFalse(seen["path"].exists())
        finally:
            self.assertEqual(scratch.resolve().parent, (ROOT / ".qa").resolve())
            shutil.rmtree(scratch)

    def test_uncertainty_is_separate_from_negative_prediction(self):
        class FakeAnalyzer:
            def analyze(self, *args):
                raise AnalysisError("uncertain_observations", "Synthetic details")
        with patch("evaluate_dataset.neutral_video") as neutral, patch("evaluate_dataset.read_metadata", return_value={}):
            neutral.return_value.__enter__.return_value = Path("input.mov")
            result = evaluate_one(FakeAnalyzer(), Path("ignored.mov"), ROOT / ".qa")
        self.assertEqual(result["status"], "abstained")
        self.assertNotIn("predicted_violation", result)
        self.assertNotIn("Synthetic", str(result))
        self.assertEqual(safe_metadata({"duration_sec": float("nan"), "width": True, "height": -2}), {"duration_sec": None, "width": None, "height": None})


if __name__ == "__main__":
    unittest.main()
