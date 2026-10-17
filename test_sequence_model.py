"""Separate-frame adapter contracts with synthetic pixels and mocked inference.

No dataset video, generated observation, model weight, or GPU is accessed here.
These tests check transport and evidence integrity, not visual accuracy.
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

from PIL import Image, ImageDraw

from analyzers import AnalysisError
from local_model import ROOT
from sequence_model import SequenceVideoAnalyzer
from test_refined_model import SYNTHETIC_FRAME, SyntheticVideo, candidate, checks


class SequenceModelTests(unittest.TestCase):
    def setUp(self):
        self.folder = ROOT / ".qa" / ("sequence-tests-" + uuid.uuid4().hex)
        self.folder.mkdir(parents=True)
        self.model = SequenceVideoAnalyzer(model="synthetic-local-model", endpoint="http://127.0.0.1:11434")
        self.model.scratch = self.folder
        self.model.last_diagnostics = {"calls": 0}

    def tearDown(self):
        self.assertTrue(self.folder.resolve().is_relative_to((ROOT / ".qa").resolve()))
        shutil.rmtree(self.folder)

    @staticmethod
    def response(value):
        return {"done": True, "message": {"content": json.dumps(value)}}

    def deadline(self):
        return time.monotonic() + 30

    def test_numbered_inputs_preserve_order_aspect_ratio_and_color(self):
        specifications = [((1024, 512), (220, 30, 40)), ((512, 1024), (25, 190, 60)), ((768, 768), (30, 70, 210))]
        paths = []
        for index, (size, color) in enumerate(specifications):
            path = self.folder / f"source-{index}.png"
            Image.new("RGB", size, color).save(path)
            paths.append(path)
        text_method = ImageDraw.ImageDraw.text
        with patch("sequence_model.ImageDraw.ImageDraw.text", autospec=True, side_effect=text_method) as draw_text:
            outputs = self.model.visual_input(paths, [0, 2.125, 4.999], self.folder / "sequence.jpg")
        self.assertEqual([path.name for path in outputs], [f"sequence-{index}.jpg" for index in range(3)])
        self.assertEqual([call.args[2] for call in draw_text.call_args_list], ["FRAME 0 | 0.000 s", "FRAME 1 | 2.125 s", "FRAME 2 | 4.999 s"])
        for path, expected_size, (_, color) in zip(outputs, [(768, 416), (384, 800), (768, 800)], specifications):
            with self.subTest(path=path.name), Image.open(path) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, expected_size)
                actual = image.getpixel((image.width // 2, 32 + (image.height - 32) // 2))
                self.assertTrue(all(abs(a - b) <= 3 for a, b in zip(actual, color)))

    def test_empty_or_unpaired_inputs_are_rejected(self):
        for paths, stamps in [([], []), ([SYNTHETIC_FRAME], []), ([SYNTHETIC_FRAME], [0, 1])]:
            with self.subTest(counts=(len(paths), len(stamps))), self.assertRaises(ValueError):
                self.model.visual_input(paths, stamps, self.folder / "invalid.jpg")

    def test_crop_and_context_are_flat_in_correct_transport_order(self):
        context = self.model.visual_input([SYNTHETIC_FRAME] * 3, [0, 1, 2], self.folder / "context.jpg")
        crop = self.folder / "synthetic-crop.jpg"
        Image.new("RGB", (80, 80), (200, 100, 50)).save(crop)
        for board in [context, tuple(context)]:
            with self.subTest(container=type(board).__name__), patch.object(self.model, "request", return_value=self.response(checks(evidence=[2]))) as request:
                decision, _ = self.model.verify(board, "personal_object", 3, self.deadline(), extra_frame=(2, crop, "crop"))
            self.assertEqual(decision, "accepted")
            self.assertEqual(request.call_args.args[0], "/api/chat")
            payload = request.call_args.args[1]
            pictures = payload["messages"][0]["images"]
            self.assertEqual(len(pictures), 4)
            self.assertTrue(all(isinstance(picture, str) for picture in pictures))
            self.assertEqual([base64.b64decode(picture) for picture in pictures], [path.read_bytes() for path in [crop, *context]])
            prompt = payload["messages"][0]["content"]
            self.assertIn("from frame 2", prompt)
            self.assertIn("ownership and location ONLY from the full contextual frames", prompt)
            self.assertNotIn("Each separate image is ONE video frame", prompt)
            self.assertFalse(payload["stream"])
            self.assertEqual(payload["options"]["temperature"], 0)
            self.assertEqual(payload["format"]["required"], ["checks", "evidence"])

    def test_strict_json_and_frame_gate_remain_active(self):
        invalid = [candidate(frame=True), candidate(frame=2), candidate("invented_violation"), {"visibility": "adequate"}]
        for value in invalid:
            with self.subTest(value=value), patch.object(self.model, "request", return_value=self.response(value)):
                with self.assertRaises(AnalysisError) as caught:
                    self.model.screen([SYNTHETIC_FRAME] * 2, 2, self.deadline())
            self.assertEqual(caught.exception.code, "invalid_observation")
        for payload, code in [({"done": True, "message": {"content": "not JSON"}}, "invalid_observation"), ({"done": True, "done_reason": "length", "message": {"content": json.dumps(candidate())}}, "truncated_observation")]:
            with self.subTest(code=code), patch.object(self.model, "request", return_value=payload):
                with self.assertRaises(AnalysisError) as caught:
                    self.model.screen([SYNTHETIC_FRAME], 1, self.deadline())
            self.assertEqual(caught.exception.code, code)

    def test_temporal_confirmation_requires_distinct_original_indexes(self):
        for evidence, expected in [([1, 1], "unresolved"), ([0, 2], "accepted")]:
            with self.subTest(evidence=evidence), patch.object(self.model, "request", return_value=self.response(checks("cashier_departure", evidence=evidence))):
                decision, _ = self.model.verify([SYNTHETIC_FRAME] * 3, "cashier_departure", 3, self.deadline())
            self.assertEqual(decision, expected)

    def test_numbered_derivatives_never_replace_original_jpeg_evidence(self):
        numbered_bytes = []
        visual_input = self.model.visual_input

        def record_inputs(paths, stamps, destination, **kwargs):
            outputs = visual_input(paths, stamps, destination, **kwargs)
            numbered_bytes.extend(path.read_bytes() for path in outputs)
            return outputs

        with patch.object(self.model, "ready", return_value=True), \
             patch("refined_model.LocalVideo", SyntheticVideo), \
             patch.object(self.model, "visual_input", side_effect=record_inputs), \
             patch.object(self.model, "infer", side_effect=deepcopy([candidate(), checks(evidence=[0, 2])])):
            result = self.model.analyze(Path("synthetic-input.mov"), {"duration_sec": 5}, lambda *_: None)
        self.assertEqual(result["verdict"], "violation")
        self.assertEqual(result["analysis"]["strategy"], "separate_temporal_frames")
        self.assertTrue(result["analysis"]["sampled"])
        self.assertTrue(result["analysis"]["coverage_complete"])
        original = SYNTHETIC_FRAME.read_bytes()
        self.assertTrue(numbered_bytes)
        self.assertTrue(all(picture != original for picture in numbered_bytes))
        evidence = result["violations"][0]["evidence_frames"]
        self.assertEqual(len(evidence), 2)
        self.assertEqual([item["timestamp_sec"] for item in evidence], [0, 3])
        self.assertTrue(all(base64.b64decode(item["image_base64"]) == original for item in evidence))

    def test_cloud_models_and_nonloopback_endpoints_remain_rejected(self):
        for endpoint in ["https://example.com", "http://192.168.1.10:11434", "https://127.0.0.1:11434"]:
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                SequenceVideoAnalyzer(endpoint=endpoint)
        with self.assertRaises(ValueError):
            SequenceVideoAnalyzer(model="synthetic-cloud", endpoint="http://127.0.0.1:11434")


if __name__ == "__main__":
    unittest.main()
