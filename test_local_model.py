import base64
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from analyzers import AnalysisError
from local_model import OllamaVideoAnalyzer, ROOT


class LocalModelTests(unittest.TestCase):
    def test_structured_rule_determines_event_category(self):
        model = OllamaVideoAnalyzer()
        part = {"summary": "Synthetic scene", "limitations": "", "events": [{"frame_index": 0, "rule_id": "nb231_12", "certainty": "clear", "observation": "Synthetic object", "reason": "Synthetic condition"}]}
        with patch.object(model, "extract_frames", return_value=[ROOT / "static" / "demo" / "normal.jpg"]), patch.object(model, "request", return_value={"message": {"content": json.dumps(part)}}):
            result = model.observe(Path("synthetic.mov"), [0.1])
        self.assertEqual(result["events"][0]["code"], "personal_items")

    def test_invalid_frame_reference_cannot_disappear_into_clean_result(self):
        model = OllamaVideoAnalyzer()
        part = {"summary": "Synthetic scene", "limitations": "", "events": [{"frame_index": 9, "rule_id": "", "certainty": "uncertain"}]}
        with patch.object(model, "extract_frames", return_value=[ROOT / "static" / "demo" / "normal.jpg"]), patch.object(model, "request", return_value={"message": {"content": json.dumps(part)}}):
            with self.assertRaises(AnalysisError) as error:
                model.observe(Path("synthetic.mov"), [0.1])
        self.assertEqual(error.exception.code, "invalid_observation")

    def test_cloud_endpoint_and_model_are_rejected(self):
        with self.assertRaises(ValueError):
            OllamaVideoAnalyzer(endpoint="https://cloud.example")
        with self.assertRaises(ValueError):
            OllamaVideoAnalyzer(model="anything-cloud")

    def test_clean_result_requires_complete_pass(self):
        model = OllamaVideoAnalyzer()
        observation = {"events": [], "summary": "Synthetic", "timestamps": [0, 1], "frames": []}
        with patch.object(model, "ready", return_value=True), patch.object(model, "observe", return_value=observation):
            result = model.analyze(Path("synthetic.mov"), {"duration_sec": 5}, lambda *args: None)
        self.assertEqual(result["verdict"], "no_violation")
        self.assertEqual(result["violations"], [])

    def test_uncertainty_does_not_become_no_violation(self):
        model = OllamaVideoAnalyzer()
        observation = {"events": [{"code": "other", "certainty": "uncertain", "rule_id": ""}]}
        with patch.object(model, "ready", return_value=True), patch.object(model, "observe", return_value=observation):
            with self.assertRaises(AnalysisError) as error:
                model.analyze(Path("synthetic.mov"), {"duration_sec": 5}, lambda *args: None)
        self.assertEqual(error.exception.code, "uncertain_observations")

    def test_rule_and_category_must_match(self):
        model = OllamaVideoAnalyzer()
        observation = {"events": [{"code": "personal_items", "certainty": "clear", "rule_id": "nb231_31_4", "frame_index": 0}]}
        with patch.object(model, "ready", return_value=True), patch.object(model, "observe", return_value=observation):
            with self.assertRaises(AnalysisError):
                model.analyze(Path("synthetic.mov"), {"duration_sec": 5}, lambda *args: None)


if __name__ == "__main__":
    unittest.main()
