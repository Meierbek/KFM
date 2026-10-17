"""Object-assistance contracts using synthetic images and no model inference."""
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import time
import types
import unittest
from unittest.mock import Mock, patch
import uuid

from PIL import Image

from analyzers import AnalysisError
from local_model import ROOT
from object_model import ObjectAssistedAnalyzer
from test_refined_model import SYNTHETIC_FRAME, SyntheticVideo, checks, synthetic_board
from yolo_candidates import LocalYoloCandidates, rank_proposals


class ObjectModelTests(unittest.TestCase):
    def setUp(self):
        self.folder = ROOT / ".qa" / ("object-tests-" + uuid.uuid4().hex)
        self.folder.mkdir(parents=True)
        self.crop_a = self.folder / "synthetic-detail-a.jpg"
        self.crop_b = self.folder / "synthetic-detail-b.jpg"
        with Image.open(SYNTHETIC_FRAME) as image:
            image.crop((0, 0, 80, 80)).save(self.crop_a, "JPEG")
            image.crop((80, 0, 160, 80)).save(self.crop_b, "JPEG")
        self.detector = Mock()
        self.detector.last_diagnostics = {"frames": 3}
        self.detector.propose_video.return_value = []
        with patch("yolo_candidates.LocalYoloCandidates", return_value=self.detector):
            self.model = ObjectAssistedAnalyzer(model="synthetic-local-model")
        self.model.scratch = self.folder
        self.model.last_diagnostics = {"calls": 0}

    def tearDown(self):
        self.assertTrue(self.folder.resolve().is_relative_to((ROOT / ".qa").resolve()))
        shutil.rmtree(self.folder)

    def proposal(self, *, stamp=4.125, crop=None, score=.99):
        return {"timecode": stamp, "crop_path": str(crop or self.crop_a),
                "frame_path": str(SYNTHETIC_FRAME), "score": score, "kind": "phone",
                "bbox_normalized": [0, 0, .25, .25]}

    def analyze_with(self, proposals, verification, duration=12):
        self.detector.propose_video.return_value = deepcopy(proposals)
        ordinary = {"visibility": "adequate", "activities": [{"kind": "ordinary_service", "frame": 0}]}
        with patch.object(self.model, "ready", return_value=True), \
             patch("refined_model.LocalVideo", SyntheticVideo), \
             patch("refined_model.storyboard", synthetic_board), \
             patch.object(self.model, "screen", return_value=ordinary), \
             patch.object(self.model, "infer", return_value=deepcopy(verification)):
            return self.model.analyze(Path("synthetic.mov"), {"duration_sec": duration}, lambda *_: None)

    def test_high_yolo_score_is_only_candidate_and_can_be_rejected(self):
        response = checks()
        response["checks"]["employee_possession"] = "absent"
        result = self.analyze_with([self.proposal(score=.999)], response)
        self.assertEqual(result["verdict"], "no_violation")
        self.assertEqual(result["violations"], [])
        self.assertEqual(result["analysis"]["object_proposals"], 1)
        self.assertEqual(self.model.last_diagnostics["rejected"], 1)

    def test_yolo_candidate_with_unknown_ownership_cannot_become_clean(self):
        response = checks()
        response["checks"]["employee_possession"] = "unknown"
        with self.assertRaises(AnalysisError) as caught:
            self.analyze_with([self.proposal()], response)
        self.assertEqual(caught.exception.code, "uncertain_observations")

    def test_detector_failure_never_turns_into_no_candidates_or_clean(self):
        self.detector.propose_video.side_effect = AnalysisError("analysis_timeout", "Synthetic object timeout")
        with patch.object(self.model, "ready", return_value=True), \
             patch("refined_model.LocalVideo", SyntheticVideo), \
             patch.object(self.model, "screen") as screen:
            with self.assertRaises(AnalysisError) as caught:
                self.model.analyze(Path("synthetic.mov"), {"duration_sec": 12}, lambda *_: None)
        self.assertEqual(caught.exception.code, "analysis_timeout")
        screen.assert_not_called()

    def test_two_distinct_object_crops_are_not_temporally_deduplicated(self):
        self.model.object_proposals = [self.proposal(stamp=4, crop=self.crop_a), self.proposal(stamp=4, crop=self.crop_b, score=.8)]
        candidates = self.model.augment_candidates([])
        self.assertEqual(len(candidates), 2)
        self.assertEqual({candidate["crop_path"] for candidate in candidates}, {str(self.crop_a), str(self.crop_b)})

    def test_generic_overview_event_is_not_replaced_by_unidentified_nearby_object(self):
        generic = {"kind": "personal_object", "stamp": 0.0}
        self.model.object_proposals = [self.proposal(stamp=4)]
        candidates = self.model.augment_candidates([generic])
        self.assertIn(generic, candidates)
        self.assertEqual(len(candidates), 2)

    def test_yolo_nms_keeps_distinct_same_class_boxes_at_same_time(self):
        proposals = [
            {"kind": "phone", "timecode": 4, "score": .9, "bbox_normalized": [0, 0, .2, .2]},
            {"kind": "phone", "timecode": 4, "score": .8, "bbox_normalized": [.7, .7, .9, .9]},
            {"kind": "phone", "timecode": 5, "score": .7, "bbox_normalized": [0, 0, .2, .2]},
        ]
        selected = rank_proposals(proposals, limit=5)
        self.assertEqual(len(selected), 2)
        self.assertEqual({item["score"] for item in selected}, {.9, .8})

    def test_two_images_are_sent_in_detail_then_context_order_without_verdict(self):
        response = {"done": True, "message": {"content": json.dumps(checks(evidence=[2]))}}
        with patch.object(self.model, "request", return_value=response) as request:
            decision, _ = self.model.verify(SYNTHETIC_FRAME, "personal_object", 4, time.monotonic() + 10,
                                             extra_frame=(2, self.crop_a, "crop"))
        self.assertEqual(decision, "accepted")
        body = request.call_args.args[1]
        message = body["messages"][0]
        self.assertEqual([base64.b64decode(item) for item in message["images"]],
                         [self.crop_a.read_bytes(), SYNTHETIC_FRAME.read_bytes()])
        self.assertIn("from frame 2", message["content"])
        self.assertIn("ownership and location ONLY from the full contextual frames", message["content"])
        self.assertNotIn("0.99", message["content"])

    def test_crop_based_acceptance_cannot_publish_only_a_different_timestamp(self):
        with patch.object(self.model, "infer", return_value=checks(evidence=[0])):
            decision, answer = self.model.verify(SYNTHETIC_FRAME, "personal_object", 4, time.monotonic() + 10,
                                                  extra_frame=(2, self.crop_a, "crop"))
        self.assertTrue(decision != "accepted" or 2 in answer["evidence"],
                        "Accepted crop-based evidence must include its original frame, not just another timestamp")

    def test_accepted_crop_is_grounded_at_its_time_and_only_original_jpeg_is_published(self):
        proposal = self.proposal(stamp=4.125)
        boards = {}

        def build_board(paths, stamps, destination, **kwargs):
            boards[str(destination)] = list(stamps)
            return synthetic_board(paths, stamps, destination, **kwargs)

        def verify(board, kind, count, deadline, extra_frame=None):
            hero, crop, crop_kind = extra_frame
            self.assertEqual(crop, str(self.crop_a))
            self.assertEqual(crop_kind, "crop")
            self.assertEqual(boards[str(board)][hero], proposal["timecode"])
            return "accepted", checks(evidence=[hero])

        self.detector.propose_video.return_value = [proposal]
        ordinary = {"visibility": "adequate", "activities": [{"kind": "ordinary_service", "frame": 0}]}
        with patch.object(self.model, "ready", return_value=True), \
             patch("refined_model.LocalVideo", SyntheticVideo), \
             patch("refined_model.storyboard", build_board), \
             patch.object(self.model, "screen", return_value=ordinary), \
             patch.object(self.model, "verify", side_effect=verify):
            result = self.model.analyze(Path("synthetic.mov"), {"duration_sec": 12}, lambda *_: None)
        event = result["violations"][0]
        self.assertEqual(event["timecode_sec"], proposal["timecode"])
        self.assertEqual(event["evidence_frames"][0]["timestamp_sec"], proposal["timecode"])
        evidence = base64.b64decode(event["evidence_frames"][0]["image_base64"])
        self.assertEqual(evidence, SYNTHETIC_FRAME.read_bytes())
        self.assertNotEqual(evidence, self.crop_a.read_bytes())

    def test_object_analyzer_rejects_cloud_endpoint_before_loading_detector(self):
        with patch("yolo_candidates.LocalYoloCandidates") as detector:
            with self.assertRaises(ValueError):
                ObjectAssistedAnalyzer(endpoint="https://cloud.example")
            with self.assertRaises(ValueError):
                ObjectAssistedAnalyzer(model="example-cloud")
        detector.assert_not_called()

    def test_yolo_load_uses_local_weight_cpu_and_disables_cloud_integrations(self):
        detector = LocalYoloCandidates.__new__(LocalYoloCandidates)
        detector.model = None
        detector.weights = self.folder / "synthetic-local.pt"
        detector.threads = 2
        torch = types.ModuleType("torch")
        torch.set_num_threads = Mock()
        ultralytics = types.ModuleType("ultralytics")
        ultralytics.YOLO = Mock(return_value=Mock())
        ultralytics.settings = Mock()
        utilities = types.ModuleType("ultralytics.utils")
        utilities.torch_utils = types.SimpleNamespace(NUM_THREADS=8)
        with patch.dict(sys.modules, {"torch": torch, "ultralytics": ultralytics, "ultralytics.utils": utilities}), \
             patch.dict(os.environ), patch("socket.create_connection", side_effect=AssertionError("Network is forbidden in synthetic test")):
            detector.load()
            self.assertEqual(os.environ["YOLO_OFFLINE"], "true")
            self.assertEqual(os.environ["YOLO_AUTOINSTALL"], "false")
        ultralytics.YOLO.assert_called_once_with(str(detector.weights), task="detect")
        detector.model.to.assert_called_once_with("cpu")
        for disabled in ("sync", "hub", "wandb", "comet", "clearml"):
            self.assertIs(ultralytics.settings.update.call_args.args[0][disabled], False)


if __name__ == "__main__":
    unittest.main()
