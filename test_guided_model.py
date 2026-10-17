"""Contract checks for the learned proposal / visual confirmation boundary."""
import time
import unittest
from unittest.mock import patch

from analyzers import AnalysisError
from guided_model import GuidedVideoAnalyzer


class FakeBaseline:
    threshold = .7
    config = {"fingerprint": "synthetic-model"}

    def score_windows(self, path, duration, *, deadline=None):
        self.received = (path, duration, deadline)
        return [0, 2, 8, 16, 24], [.9, .85, .8, .95, .92]


class GuidedTests(unittest.TestCase):
    def setUp(self):
        self.model = object.__new__(GuidedVideoAnalyzer)
        self.model.baseline = FakeBaseline()
        self.model.last_diagnostics = {}
        self.model.proposals = []

    def test_proposals_use_only_video_duration_and_preserve_deadline(self):
        deadline = time.monotonic()+60
        windows = self.model.prepare_windows("input.mov", 25, deadline)
        self.assertEqual(self.model.baseline.received, ("input.mov", 25, deadline))
        self.assertEqual(len(windows), 3)
        self.assertEqual([proposal["timecode"] for proposal in self.model.proposals], [0, 16, 24])
        self.assertTrue(all(0 <= stamp < 25 for window in windows for stamp in window))

    def test_positive_signal_without_confirmation_never_becomes_clean(self):
        self.model.proposals = [{"timecode": 2, "score": .9}]
        with self.assertRaises(AnalysisError) as error:
            self.model.finalize_result({"verdict": "no_violation", "violations": [], "analysis": {}})
        self.assertEqual(error.exception.code, "uncertain_observations")

    def test_no_signal_is_explicit_in_result_metadata(self):
        result = self.model.finalize_result({"verdict": "no_violation", "violations": [], "analysis": {}})
        self.assertFalse(result["analysis"]["detector_signal"])
        self.assertEqual(result["analysis"]["sampling_strategy"], "learned_candidates")
        self.assertEqual(result["analysis"]["detector_fingerprint"], "synthetic-model")

    def test_invalid_numeric_scores_are_not_clean_predictions(self):
        with patch.object(self.model.baseline, "score_windows", return_value=([1], [float("nan")])):
            with self.assertRaises(AnalysisError) as error:
                self.model.prepare_windows("input.mov", 10, time.monotonic()+60)
        self.assertEqual(error.exception.code, "invalid_candidate_scores")


if __name__ == "__main__":
    unittest.main()
