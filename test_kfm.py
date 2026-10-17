"""Behavioral checks for the upload/API/model boundary. No real user videos."""
import base64
import http.client
import json
from pathlib import Path
import struct
import threading
import time
import unittest
import uuid
from unittest.mock import patch

from analyzers import AnalysisError, LocalHTTPAnalyzer, validate_result
from server import App, ROOT, make_server, read_metadata


def fake_mov(duration=42):
    def atom(kind, data):
        return struct.pack(">I4s", 8 + len(data), kind) + data
    return atom(b"ftyp", b"qt  \x00\x00\x00\x00qt  ") + atom(b"moov", atom(b"mvhd", b"\x00" * 12 + struct.pack(">II", 1000, duration * 1000) + b"\x00" * 80))


class KFMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = ROOT / ".qa" / ("test-" + uuid.uuid4().hex)
        cls.temp.mkdir(parents=True)
        cls.app = App(cls.temp, demo_delay=False)
        cls.server = make_server(port=0, app=cls.app)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.app.executor.shutdown(wait=True)
        import shutil
        assert cls.temp.resolve().is_relative_to((ROOT / ".qa").resolve())
        shutil.rmtree(cls.temp)

    def request(self, method, path, body=None, headers=None, auth=True):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=8)
        headers = dict(headers or {})
        if auth:
            headers["X-API-Key"] = self.app.api_key
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        status, response_headers = response.status, dict(response.getheaders())
        connection.close()
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            result = raw
        return status, result, response_headers

    def wait_job(self, job_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            _, job, _ = self.request("GET", f"/api/v1/jobs/{job_id}")
            if job["status"] != "processing":
                return job
            time.sleep(.02)
        self.fail("Job did not complete")

    def upload(self, name="test.mov", data=None):
        boundary = "kfm-test-boundary"
        body = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\nContent-Type: video/quicktime\r\n\r\n'.encode() + (data if data is not None else fake_mov()) + f"\r\n--{boundary}--\r\n".encode()
        return self.request("POST", "/api/v1/analyze", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})

    def test_api_key_required_even_when_browser_session_exists(self):
        status, _, _ = self.request("GET", "/api/v1/jobs", auth=False, headers={"Cookie": f"kfm_session={self.app.session}"})
        self.assertEqual(status, 401)

    def test_second_server_cannot_share_the_same_port(self):
        with self.assertRaises(OSError):
            make_server(port=self.port, app=self.app)
        status, _, _ = self.request("GET", "/api/v1/jobs", auth=False, headers={"X-API-Key": "wrong"})
        self.assertEqual(status, 401)

    def test_ui_requires_local_session(self):
        self.assertEqual(self.request("GET", "/ui/api/config", auth=False)[0], 401)
        status, config, _ = self.request("GET", "/ui/api/config", auth=False, headers={"Cookie": f"kfm_session={self.app.session}"})
        self.assertEqual(status, 200)
        self.assertFalse(config["model_connected"])
        self.assertNotIn("api_key", config)

    def test_home_and_deep_link_set_session(self):
        status, html, headers = self.request("GET", "/", auth=False)
        self.assertEqual(status, 200)
        self.assertIn("HttpOnly", headers["Set-Cookie"])
        self.assertIn(b"KFM", html)
        self.assertEqual(self.request("GET", "/results/" + "a"*32, auth=False)[0], 200)
        self.assertEqual(self.request("GET", "/static/../server.py", auth=False)[0], 404)
        self.assertEqual(self.request("GET", "/static/%2e%2e/server.py", auth=False)[0], 404)

    def test_demo_positive_and_clean(self):
        for scenario, verdict, count in [("violation", "violation", 2), ("clean", "no_violation", 0)]:
            status, result, _ = self.request("POST", "/api/v1/demo", json.dumps({"scenario": scenario}), {"Content-Type": "application/json"})
            self.assertEqual(status, 202)
            job = self.wait_job(result["job_id"])
            self.assertEqual(job["status"], "done")
            self.assertTrue(job["demo"])
            self.assertEqual(job["verdict"], verdict)
            self.assertEqual(len(job["violations"]), count)
            validate_result(job, 42)

    def test_upload_without_model_never_returns_clean_verdict(self):
        status, result, _ = self.upload()
        self.assertEqual(status, 202)
        job = self.wait_job(result["job_id"])
        self.assertFalse(job["demo"])
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["error"]["code"], "model_not_configured")
        self.assertIsNone(job["verdict"])
        self.assertEqual(job["duration_sec"], 42)
        self.assertTrue((self.app.uploads / f"{job['job_id']}.mov").exists())
        status, chunk, headers = self.request("GET", f"/media/{job['job_id']}", headers={"Range": "bytes=0-15"})
        self.assertEqual(status, 206)
        self.assertEqual(chunk, fake_mov()[:16])
        self.assertEqual(headers["Content-Range"], f"bytes 0-15/{len(fake_mov())}")
        self.assertEqual(self.request("GET", f"/media/{job['job_id']}", headers={"Range": "bytes=99999-"})[0], 416)

    def test_reject_invalid_files_and_limits(self):
        self.assertEqual(self.upload(name="test.exe")[0], 415)
        self.assertEqual(self.upload(data=b"not a video")[0], 415)
        self.assertEqual(self.upload(data=b"")[0], 400)
        with patch.object(self.app, "max_bytes", 10):
            self.assertEqual(self.upload()[0], 413)
        self.assertEqual(self.request("POST", "/api/v1/demo", "{", {"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.request("GET", "/api/v1/jobs/not-a-job")[0], 404)

    def test_model_version_and_sampled_scope_survive_api_storage(self):
        class ScopedAnalyzer:
            name, model, version = "Synthetic", "synthetic-local", "test-v1"
            experimental = connected = True
            max_duration_sec = 300
            supported_rule_ids = ("nb231_12", "nb231_22", "nb231_20")

            def analyze(self, path, metadata, progress):
                return {"verdict": "no_violation", "violations": [], "analysis": {
                    "sampled": True, "coverage_complete": True,
                    "unresolved_candidates": 0, "supported_rule_ids": list(self.supported_rule_ids)}}

        with patch.object(self.app, "analyzer", ScopedAnalyzer()):
            status, result, _ = self.upload()
            self.assertEqual(status, 202)
            job = self.wait_job(result["job_id"])
            self.assertEqual(job["status"], "done")
            self.assertEqual(job["analyzer"]["version"], "test-v1")
            self.assertEqual(job["analyzer"]["model"], "synthetic-local")
            self.assertTrue(job["experimental"])
            self.assertTrue(job["analysis"]["sampled"])
            self.assertEqual(job["analysis"]["supported_rule_ids"], list(ScopedAnalyzer.supported_rule_ids))
            _, config, _ = self.request("GET", "/ui/api/config", auth=False, headers={"Cookie": f"kfm_session={self.app.session}"})
            self.assertEqual(config["max_duration_sec"], 300)
            self.assertEqual(config["supported_rule_ids"], list(ScopedAnalyzer.supported_rule_ids))

    def test_future_local_analyzer_contract(self):
        class StubHandler(__import__("http.server").server.BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.server.received = request
                body = json.dumps({"verdict": "no_violation", "violations": []}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        local = __import__("http.server").server.ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
        thread = threading.Thread(target=local.serve_forever, daemon=True)
        thread.start()
        analyzer = LocalHTTPAnalyzer(f"http://127.0.0.1:{local.server_address[1]}/analyze", "Local stub", [])
        try:
            with patch.object(self.app, "analyzer", analyzer):
                _, result, _ = self.upload(name="later.mp4")
                job = self.wait_job(result["job_id"])
            self.assertEqual(job["status"], "done")
            self.assertEqual(job["verdict"], "no_violation")
            self.assertFalse(job["demo"])
            self.assertTrue(Path(local.received["video_path"]).is_absolute())
        finally:
            local.shutdown()
            local.server_close()

    def test_invalid_model_outputs_are_errors(self):
        invalid = [
            {"verdict": "unclear", "violations": []},
            {"verdict": "violation", "violations": []},
            {"verdict": "no_violation", "violations": [{}]},
            {"verdict": "violation", "violations": [{"type": "event", "description": "test", "timecode_sec": 0, "evidence_frames": []}]},
        ]
        for result in invalid:
            with self.assertRaises(AnalysisError):
                validate_result(result)
        with self.assertRaises(ValueError):
            LocalHTTPAnalyzer("https://remote.example/analyze", "", [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
