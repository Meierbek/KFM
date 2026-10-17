"""8B request adapter tests: synthetic pixels and mocked HTTP, no inference."""
import base64
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import time
import unittest
from unittest.mock import patch
import uuid

from analyzers import AnalysisError
from evaluate_dataset import SOURCE_FILES, make_analyzer
from large_model import LargeVideoAnalyzer
from local_model import ROOT
from refined_model import RefinedVideoAnalyzer
from server import App
from test_refined_model import SYNTHETIC_FRAME, candidate


class LargeModelTests(unittest.TestCase):
    def setUp(self):
        self.model = LargeVideoAnalyzer(endpoint="http://127.0.0.1:11434")
        self.model.last_diagnostics = {"calls": 0}

    def test_default_model_and_version_ignore_previous_model_environment(self):
        with patch.dict(os.environ, {"KFM_OLLAMA_MODEL": "previous-local-model"}):
            model = LargeVideoAnalyzer(endpoint="http://127.0.0.1:11434")
        self.assertEqual(model.model, "qwen3-vl:8b-instruct")
        self.assertEqual(model.version, "large-v1")
        self.assertIn("8B", model.name)

    def test_evaluation_selector_preserves_model_and_tracks_adapter_source(self):
        model = make_analyzer("large", "synthetic-local-large")
        self.assertIsInstance(model, LargeVideoAnalyzer)
        self.assertEqual(model.model, "synthetic-local-large")
        self.assertEqual(model.version, "large-v1")
        self.assertIn("large_model.py", SOURCE_FILES)

    def test_server_selector_records_large_adapter_without_running_inference(self):
        folder = ROOT / ".qa" / ("large-selector-" + uuid.uuid4().hex)
        app = None
        try:
            with patch.dict(os.environ, {"KFM_ANALYZER": "large", "KFM_OLLAMA_MODEL": "synthetic-local-large", "KFM_OLLAMA_URL": "http://127.0.0.1:11434", "KFM_API_KEY": "synthetic-test-key"}):
                app = App(folder, demo_delay=False)
            self.assertIsInstance(app.analyzer, LargeVideoAnalyzer)
            job = app.new_job(filename="synthetic-test.mov")
            self.assertTrue(job["experimental"])
            self.assertEqual(job["analyzer"]["version"], "large-v1")
            self.assertEqual(job["analyzer"]["model"], "synthetic-local-large")
        finally:
            if app is not None:
                app.executor.shutdown(wait=True)
            self.assertTrue(folder.resolve().is_relative_to((ROOT / ".qa").resolve()))
            if folder.exists():
                shutil.rmtree(folder)

    def test_actual_chat_payload_has_bounded_batch_and_inherited_controls(self):
        response = io.BytesIO(json.dumps({"done": True, "message": {"content": json.dumps(candidate())}}).encode())
        with patch.object(self.model.opener, "open", return_value=response) as send:
            result = self.model.screen(SYNTHETIC_FRAME, 3, time.monotonic() + 20)
        self.assertEqual(result, candidate())
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(request.get_method(), "POST")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "qwen3-vl:8b-instruct")
        self.assertEqual(payload["options"]["num_batch"], 128)
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertEqual(payload["options"]["seed"], 17)
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["think"])
        self.assertEqual(payload["format"]["required"], ["visibility", "activities"])
        self.assertEqual(base64.b64decode(payload["messages"][0]["images"][0]), SYNTHETIC_FRAME.read_bytes())
        self.assertGreater(send.call_args.kwargs["timeout"], 0)

    def test_chat_options_are_copied_without_mutating_caller(self):
        body = {"model": self.model.model, "options": {"num_ctx": 4096, "num_batch": 512, "seed": 17}, "messages": [{"role": "user", "content": "Synthetic"}]}
        original = deepcopy(body)
        sentinel = {"synthetic": "response"}
        with patch.object(RefinedVideoAnalyzer, "request", return_value=sentinel) as delegate:
            result = self.model.request("/api/chat", body, timeout=12)
        sent = delegate.call_args.args[1]
        self.assertIs(result, sentinel)
        self.assertEqual(body, original)
        self.assertIsNot(sent, body)
        self.assertIsNot(sent["options"], body["options"])
        self.assertEqual(sent["options"], {"num_ctx": 4096, "num_batch": 128, "seed": 17})
        self.assertEqual(delegate.call_args.kwargs["timeout"], 12)

    def test_ready_and_other_get_requests_remain_unmodified(self):
        self.model.ffmpeg = Path(__file__)  # Only is_file() is called; no decoder runs.
        tags = io.BytesIO(json.dumps({"models": [{"name": self.model.model}]}).encode())
        version = io.BytesIO(b'{"version":"synthetic"}')
        with patch.object(self.model.opener, "open", side_effect=[tags, version]) as send:
            self.assertTrue(self.model.ready())
            self.assertEqual(self.model.request("/api/version"), {"version": "synthetic"})
        self.assertEqual([call.args[0].full_url for call in send.call_args_list], ["http://127.0.0.1:11434/api/tags", "http://127.0.0.1:11434/api/version"])
        for call in send.call_args_list:
            self.assertEqual(call.args[0].get_method(), "GET")
            self.assertIsNone(call.args[0].data)
            self.assertEqual(call.kwargs["timeout"], 10)

    def test_inherited_endpoint_and_cloud_model_guards_remain_active(self):
        for endpoint in ["https://example.com", "http://192.168.1.10:11434", "https://127.0.0.1:11434"]:
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                LargeVideoAnalyzer(endpoint=endpoint)
        for model in ["synthetic-cloud", "synthetic:cloud"]:
            with self.subTest(model=model), self.assertRaises(ValueError):
                LargeVideoAnalyzer(model=model, endpoint="http://127.0.0.1:11434")

    def test_completed_video_releases_model_without_generating_a_prompt(self):
        result = {"verdict": "no_violation", "violations": [], "analysis": {"sampled": True}}
        progress = lambda *_: None
        with patch.object(RefinedVideoAnalyzer, "analyze", return_value=result) as analyze, \
             patch.object(self.model, "request", return_value={"done": True}) as unload:
            actual = self.model.analyze(Path("synthetic.mov"), {"duration_sec": 5}, progress)
        self.assertIs(actual, result)
        analyze.assert_called_once_with(Path("synthetic.mov"), {"duration_sec": 5}, progress)
        unload.assert_called_once_with("/api/generate", {"model": self.model.model, "keep_alive": 0}, timeout=5)

    def test_unload_failure_cannot_replace_successful_video_result(self):
        result = {"verdict": "no_violation", "violations": []}
        for failure in [TimeoutError("Synthetic timeout"), AnalysisError("local_model_unavailable", "Synthetic unavailable")]:
            with self.subTest(error=type(failure).__name__), \
                 patch.object(RefinedVideoAnalyzer, "analyze", return_value=result), \
                 patch.object(self.model, "request", side_effect=failure) as unload:
                self.assertIs(self.model.analyze(Path("synthetic.mov"), {}, lambda *_: None), result)
            unload.assert_called_once_with("/api/generate", {"model": self.model.model, "keep_alive": 0}, timeout=5)

    def test_failed_video_preserves_original_exception_despite_unload_failure(self):
        for unload_failure in [None, TimeoutError("Synthetic timeout"), RuntimeError("Synthetic unload failure")]:
            original = AnalysisError("uncertain_observations", "Synthetic inconclusive video")
            with self.subTest(unload_error=type(unload_failure).__name__), \
                 patch.object(RefinedVideoAnalyzer, "analyze", side_effect=original), \
                 patch.object(self.model, "request", side_effect=unload_failure) as unload:
                with self.assertRaises(AnalysisError) as caught:
                    self.model.analyze(Path("synthetic.mov"), {}, lambda *_: None)
            self.assertIs(caught.exception, original)
            unload.assert_called_once_with("/api/generate", {"model": self.model.model, "keep_alive": 0}, timeout=5)


if __name__ == "__main__":
    unittest.main()
