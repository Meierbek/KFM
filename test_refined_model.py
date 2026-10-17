"""Safety and grounding checks using synthetic pictures and mocked local inference.

These tests deliberately never read videos, labels, or generated observations.
They validate the adapter contract; they do not measure visual model accuracy.
"""
import base64
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
import unittest
from unittest.mock import patch
import uuid

from analyzers import AnalysisError
from local_model import ROOT
from refined_model import PREDICATES, RefinedVideoAnalyzer, gate
from visual_sampling import sample_windows


SYNTHETIC_FRAME = ROOT / "static" / "demo" / "normal.jpg"


def candidate(kind="personal_object", frame=0):
    return {"visibility": "adequate", "activities": [{"kind": kind, "frame": frame}]}


def checks(kind="personal_object", state="present", evidence=None):
    return {"checks": {key: state for key in PREDICATES[kind]}, "evidence": [0] if evidence is None else evidence}


class SyntheticVideo:
    """Imitates decoding without opening the submitted video path."""

    def __init__(self, path, ffmpeg, folder, deadline):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True)
        self.cache = {}

    def frames(self, stamps):
        output = []
        for stamp in stamps:
            if stamp not in self.cache:
                destination = self.folder / f"synthetic-{stamp:.3f}.jpg"
                shutil.copyfile(SYNTHETIC_FRAME, destination)
                self.cache[stamp] = destination
            output.append(self.cache[stamp])
        return output


def synthetic_board(paths, stamps, destination, **kwargs):
    # A separate byte sequence catches accidentally publishing a contact sheet
    # instead of an original decoded evidence frame.
    Path(destination).write_bytes(b"CONTACT-SHEET-NOT-AN-EVIDENCE-FRAME")
    return Path(destination)


class RefinedModelTests(unittest.TestCase):
    def setUp(self):
        self.folder = ROOT / ".qa" / ("refined-tests-" + uuid.uuid4().hex)
        self.folder.mkdir(parents=True)
        self.model = RefinedVideoAnalyzer(model="synthetic-local-model")
        self.model.scratch = self.folder
        self.model.last_diagnostics = {"calls": 0}

    def tearDown(self):
        self.assertTrue(self.folder.resolve().is_relative_to((ROOT / ".qa").resolve()))
        shutil.rmtree(self.folder)

    def assert_analysis_error(self, code, action):
        with self.assertRaises(AnalysisError) as caught:
            action()
        self.assertEqual(caught.exception.code, code)

    def infer_value(self, value, method="screen", kind="personal_object", count=3):
        with patch.object(self.model, "infer", return_value=deepcopy(value)):
            if method == "screen":
                return self.model.screen(SYNTHETIC_FRAME, count, time.monotonic() + 10)
            return self.model.verify(SYNTHETIC_FRAME, kind, count, time.monotonic() + 10)

    def analyze_values(self, values, duration=5):
        with patch.object(self.model, "ready", return_value=True), \
             patch("refined_model.LocalVideo", SyntheticVideo), \
             patch("refined_model.storyboard", synthetic_board), \
             patch.object(self.model, "infer", side_effect=deepcopy(values)):
            return self.model.analyze(Path("synthetic-input.mov"), {"duration_sec": duration}, lambda *_: None)

    def test_screen_rejects_missing_extra_keys_and_invalid_enums(self):
        invalid = [
            {"activities": []},
            {"visibility": "adequate", "activities": [], "verdict": "clean"},
            {"visibility": "clear", "activities": []},
            {"visibility": [], "activities": []},
            {"visibility": "adequate", "activities": "none"},
            candidate("invented_violation"),
            {"visibility": "adequate", "activities": [{"kind": "ordinary_service"}]},
            {"visibility": "adequate", "activities": [{"kind": "ordinary_service", "frame": 0, "rule": "invented"}]},
        ]
        for value in invalid:
            with self.subTest(value=value):
                self.assert_analysis_error("invalid_observation", lambda: self.infer_value(value))

    def test_screen_rejects_boolean_negative_and_out_of_range_frame(self):
        for frame in [True, False, -1, 3, 0.5, "0", None]:
            with self.subTest(frame=frame):
                self.assert_analysis_error("invalid_observation", lambda: self.infer_value(candidate(frame=frame)))

    def test_verifier_requires_exact_predicates_valid_states_and_evidence(self):
        base = checks()
        missing = deepcopy(base)
        missing["checks"].pop("employee_possession")
        extra = deepcopy(base)
        extra["checks"]["model_confidence"] = "present"
        bad_state = deepcopy(base)
        bad_state["checks"]["employee_possession"] = True
        for value in [missing, extra, bad_state, checks(evidence=[]), checks(evidence=[0, 1, 2, 0]), checks(evidence=[True]), checks(evidence=[3])]:
            with self.subTest(value=value):
                self.assert_analysis_error("invalid_observation", lambda: self.infer_value(value, "verify"))

    def test_tri_state_gate_accepts_only_all_required_conditions(self):
        for state, expected in [("present", "accepted"), ("absent", "rejected"), ("unknown", "unresolved")]:
            value = checks()
            value["checks"]["employee_possession"] = state
            with self.subTest(state=state):
                decision, _ = self.infer_value(value, "verify")
                self.assertEqual(decision, expected)
        self.assertEqual(gate({"one": "unknown", "two": "absent"}), "rejected")

    def test_temporal_claim_needs_two_distinct_evidence_frames(self):
        for kind in ["cashier_departure", "previous_cash"]:
            for evidence, expected in [([1], "unresolved"), ([1, 1], "unresolved"), ([2, 0], "accepted")]:
                with self.subTest(kind=kind, evidence=evidence):
                    decision, result = self.infer_value(checks(kind, evidence=evidence), "verify", kind)
                    self.assertEqual(decision, expected)
                    self.assertEqual(result["evidence"], sorted(set(evidence)))

    def test_truncated_model_output_never_becomes_a_result(self):
        payload = {"done": True, "done_reason": "length", "message": {"content": json.dumps(candidate())}}
        with patch.object(self.model, "request", return_value=payload):
            self.assert_analysis_error("truncated_observation", lambda: self.model.infer(SYNTHETIC_FRAME, "Synthetic", {}, time.monotonic() + 10))

    def test_malformed_response_envelope_is_controlled_analysis_error(self):
        for payload in [None, [], {"message": None}, {"message": []}, {"message": {"content": "not-json"}}]:
            with self.subTest(payload=payload), patch.object(self.model, "request", return_value=payload):
                self.assert_analysis_error("invalid_observation", lambda: self.model.infer(SYNTHETIC_FRAME, "Synthetic", {}, time.monotonic() + 10))

    def test_accepted_evidence_is_original_image_and_whitelisted_rule(self):
        result = self.analyze_values([candidate(), checks(evidence=[0, 2])])
        self.assertEqual(result["verdict"], "violation")
        event = result["violations"][0]
        self.assertEqual(event["rule"], {key: self.model.by_id["nb231_12"][key] for key in ("title", "section", "url")})
        originals = SYNTHETIC_FRAME.read_bytes()
        for frame in event["evidence_frames"]:
            self.assertEqual(base64.b64decode(frame["image_base64"]), originals)
            self.assertGreaterEqual(frame["timestamp_sec"], 0)
            self.assertLess(frame["timestamp_sec"], 5)
        self.assertIn(event["timecode_sec"], [frame["timestamp_sec"] for frame in event["evidence_frames"]])

    def test_visible_counterexample_rejects_candidate_without_false_positive(self):
        response = checks()
        response["checks"]["employee_possession"] = "absent"
        result = self.analyze_values([candidate(), response])
        self.assertEqual(result["verdict"], "no_violation")
        self.assertEqual(result["violations"], [])
        self.assertEqual(result["analysis"]["unresolved_candidates"], 0)

    def test_unresolved_candidate_cannot_become_clean(self):
        response = checks()
        response["checks"]["employee_possession"] = "unknown"
        self.assert_analysis_error("uncertain_observations", lambda: self.analyze_values([candidate(), response]))

    def test_confirmed_event_retains_explicit_unresolved_candidate_metadata(self):
        overview = {"visibility": "adequate", "activities": [{"kind": "personal_object", "frame": 0}, {"kind": "cashier_departure", "frame": 3}]}
        unknown = checks("cashier_departure", evidence=[0, 2])
        unknown["checks"]["service_unfinished"] = "unknown"
        result = self.analyze_values([overview, checks(), unknown])
        self.assertEqual(result["verdict"], "violation")
        self.assertEqual(len(result["violations"]), 1)
        self.assertEqual(result["analysis"]["unresolved_candidates"], 1)
        self.assertTrue(result["analysis"]["sampled"])
        self.assertTrue(result["analysis"]["coverage_complete"])
        self.assertIn("nb231_12", result["analysis"]["supported_rule_ids"])

    def test_later_failed_window_cannot_return_clean_or_early_candidate(self):
        for first in [candidate("ordinary_service"), candidate()]:
            with self.subTest(first=first):
                failure = AnalysisError("analysis_timeout", "Synthetic timeout")
                with patch.object(self.model, "ready", return_value=True), \
                     patch("refined_model.LocalVideo", SyntheticVideo), \
                     patch("refined_model.storyboard", synthetic_board), \
                     patch.object(self.model, "infer", side_effect=[deepcopy(first), failure]):
                    self.assert_analysis_error("analysis_timeout", lambda: self.model.analyze(Path("synthetic.mov"), {"duration_sec": 30}, lambda *_: None))

    def test_poor_visibility_prevents_completed_verdict_even_with_event(self):
        overview = candidate()
        overview["visibility"] = "poor"
        self.assert_analysis_error("insufficient_visibility", lambda: self.analyze_values([overview, checks()]))

    def test_tiny_video_still_executes_screening(self):
        result = self.analyze_values([{ "visibility": "adequate", "activities": []}], duration=0.001)
        self.assertEqual(result["verdict"], "no_violation")
        self.assertEqual(self.model.last_diagnostics["windows"], 1)
        self.assertGreater(self.model.last_diagnostics["frames"], 0)

    def test_sampler_reaches_start_and_tail_without_large_temporal_gaps(self):
        for duration in [0.001, 0.1, 0.15, 1, 24, 24.1, 49, 142.534, 300]:
            with self.subTest(duration=duration):
                windows = sample_windows(duration)
                stamps = sorted(set(stamp for window in windows for stamp in window))
                self.assertTrue(stamps)
                self.assertEqual(stamps[0], 0)
                self.assertGreaterEqual(stamps[-1], duration - 0.101)
                self.assertLessEqual(stamps[-1], duration)
                self.assertTrue(all(0 <= stamp <= duration for stamp in stamps))
                self.assertTrue(all(b - a <= 3.001 for a, b in zip(stamps, stamps[1:])))
                self.assertTrue(all(window == sorted(set(window)) for window in windows))

    def test_invalid_or_unsupported_duration_never_silently_returns_clean(self):
        for duration in [None, True, False, "2", 0, -1, float("nan"), float("inf"), 301]:
            with self.subTest(duration=duration):
                with self.assertRaises(AnalysisError):
                    sample_windows(duration)

    def test_refined_analyzer_inherits_local_only_endpoint_enforcement(self):
        for endpoint in ["https://example.com", "http://192.168.1.10:11434", "https://127.0.0.1:11434"]:
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                RefinedVideoAnalyzer(endpoint=endpoint)
        with self.assertRaises(ValueError):
            RefinedVideoAnalyzer(model="synthetic-cloud")


if __name__ == "__main__":
    unittest.main()
