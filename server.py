"""KFM local application. Python 3.10+, standard library only."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import socket
import sqlite3
import struct
import threading
import time
import traceback
from urllib.parse import unquote, urlsplit
import uuid
import webbrowser

from analyzers import Analyzer, AnalysisError, LocalHTTPAnalyzer, analyze_demo


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
JOB_ID = re.compile(r"^[0-9a-f]{32}$")


def load_env():
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def read_metadata(path: Path) -> dict:
    """Read MOV/MP4 container timing only; no image decoding or external service."""
    result = {"duration_sec": None, "width": None, "height": None}
    size = path.stat().st_size

    def atoms(handle, start, end, depth=0):
        if depth > 5:
            return
        offset = start
        while offset + 8 <= end:
            handle.seek(offset)
            header = handle.read(8)
            if len(header) < 8:
                break
            length, kind = struct.unpack(">I4s", header)
            header_size = 8
            if length == 1:
                raw = handle.read(8)
                if len(raw) < 8:
                    break
                length = struct.unpack(">Q", raw)[0]
                header_size = 16
            elif length == 0:
                length = end - offset
            if length < header_size or offset + length > end:
                break
            body = offset + header_size
            if kind == b"mvhd":
                handle.seek(body)
                raw = handle.read(min(40, length - header_size))
                if len(raw) >= 20:
                    if raw[0] == 1 and len(raw) >= 32:
                        scale, duration = struct.unpack(">IQ", raw[20:32])
                    else:
                        scale, duration = struct.unpack(">II", raw[12:20])
                    if scale and duration < 2**63:
                        result["duration_sec"] = round(duration / scale, 3)
            elif kind == b"tkhd":
                handle.seek(offset + length - 8)
                raw = handle.read(8)
                if len(raw) == 8:
                    width, height = struct.unpack(">II", raw)
                    if width and height:
                        result.update(width=width >> 16, height=height >> 16)
            if kind in {b"moov", b"trak"}:
                atoms(handle, body, offset + length, depth + 1)
            offset += length

    try:
        with path.open("rb") as handle:
            atoms(handle, 0, size)
    except (OSError, struct.error):
        pass
    return result


class App:
    def __init__(self, data_dir=None, demo_delay=True):
        self.data = Path(data_dir or os.environ.get("KFM_DATA_DIR", ROOT / "data")).resolve()
        self.data.mkdir(parents=True, exist_ok=True)
        self.uploads = self.data / "uploads"
        self.uploads.mkdir(exist_ok=True)
        self.database = self.data / "jobs.sqlite3"
        self.session = secrets.token_urlsafe(32)
        self.max_bytes = int(os.environ.get("KFM_MAX_UPLOAD_MB", "128")) * 1024 * 1024
        self.demo_delay = demo_delay
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kfm-analysis")
        self.lock = threading.Lock()
        key_file = self.data / "api_key.txt"
        override = os.environ.get("KFM_API_KEY")
        if override:
            self.api_key = override
        elif key_file.exists():
            self.api_key = key_file.read_text(encoding="utf-8").strip()
        else:
            self.api_key = secrets.token_urlsafe(32)
            key_file.write_text(self.api_key, encoding="utf-8")
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, created REAL NOT NULL, body TEXT NOT NULL)")
            pending = db.execute("SELECT id, body FROM jobs").fetchall()
            for job_id, body in pending:
                job = json.loads(body)
                if job["status"] == "processing":
                    job.update(status="failed", progress=0, stage="Обработка прервана", error={"code": "interrupted", "message": "Сервер был остановлен во время обработки. Повторите проверку."})
                    db.execute("UPDATE jobs SET body=? WHERE id=?", (json.dumps(job, ensure_ascii=False), job_id))
        rules_path = ROOT / "rules.json"
        rules = json.loads(rules_path.read_text(encoding="utf-8")) if rules_path.exists() else []
        endpoint = os.environ.get("KFM_ANALYZER_URL", "").strip()
        analyzer_kind = os.environ.get("KFM_ANALYZER", "").lower()
        if analyzer_kind == "large":
            from large_model import LargeVideoAnalyzer
            self.analyzer = LargeVideoAnalyzer(model=os.environ.get("KFM_OLLAMA_MODEL"))
        elif analyzer_kind == "sequence":
            from sequence_model import SequenceVideoAnalyzer
            self.analyzer = SequenceVideoAnalyzer(model=os.environ.get("KFM_OLLAMA_MODEL"))
        elif analyzer_kind == "objects":
            from object_model import ObjectAssistedAnalyzer
            self.analyzer = ObjectAssistedAnalyzer()
        elif analyzer_kind == "guided":
            from guided_model import GuidedVideoAnalyzer
            self.analyzer = GuidedVideoAnalyzer()
        elif analyzer_kind == "refined":
            from refined_model import RefinedVideoAnalyzer
            self.analyzer = RefinedVideoAnalyzer()
        elif analyzer_kind == "ollama":
            from local_model import OllamaVideoAnalyzer
            self.analyzer = OllamaVideoAnalyzer()
        else:
            self.analyzer = LocalHTTPAnalyzer(endpoint, os.environ.get("KFM_MODEL_NAME", ""), rules) if endpoint else Analyzer()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=15)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def put(self, job):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO jobs VALUES (?, ?, ?)", (job["job_id"], job["created_at"], json.dumps(job, ensure_ascii=False)))

    def get(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT body FROM jobs WHERE id=?", (job_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def update(self, job_id, **fields):
        with self.lock:
            job = self.get(job_id)
            if job:
                job.update(fields)
                self.put(job)

    def recent(self):
        with self.connect() as db:
            rows = db.execute("SELECT body FROM jobs ORDER BY created DESC LIMIT 8").fetchall()
        return [{key: job.get(key) for key in ("job_id", "filename", "created_at", "status", "demo", "verdict", "duration_sec", "scenario")} for job in (json.loads(row[0]) for row in rows)]

    def new_job(self, *, filename, demo=False, scenario=None, size=0, metadata=None):
        job = {
            "job_id": uuid.uuid4().hex, "status": "processing", "created_at": time.time(),
            "filename": filename, "demo": demo, "scenario": scenario,
            "progress": 5, "stage": "Видео принято" if not demo else "Подготовка демонстрации",
            "size_bytes": size, "duration_sec": (metadata or {}).get("duration_sec"),
            "width": (metadata or {}).get("width"), "height": (metadata or {}).get("height"),
            "verdict": None, "violations": [], "error": None,
            "experimental": bool(getattr(self.analyzer, "experimental", False)) and not demo,
            "analyzer": None if demo else {
                "name": self.analyzer.name,
                "model": getattr(self.analyzer, "model", None),
                "version": getattr(self.analyzer, "version", "legacy"),
            },
        }
        return job

    def submit(self, job, path=None):
        self.put(job)
        self.executor.submit(self.run_job, job["job_id"], path)

    def run_job(self, job_id, path):
        started = time.monotonic()
        job = self.get(job_id)

        def progress(percent, stage):
            self.update(job_id, progress=percent, stage=stage)

        try:
            if job["demo"]:
                if self.demo_delay:
                    result = analyze_demo(job["scenario"], STATIC / "demo", progress)
                else:
                    from unittest.mock import patch
                    with patch("analyzers.time.sleep"):
                        result = analyze_demo(job["scenario"], STATIC / "demo", progress)
            else:
                result = self.analyzer.analyze(path, {key: job[key] for key in ("duration_sec", "width", "height", "size_bytes")}, progress)
            self.update(job_id, status="done", progress=100, stage="Готово", elapsed_sec=round(time.monotonic() - started, 2), **result)
        except AnalysisError as exc:
            self.update(job_id, status="failed", stage="Модель не подключена" if exc.code == "model_not_configured" else "Обработка не завершена", error={"code": exc.code, "message": str(exc)}, elapsed_sec=round(time.monotonic() - started, 2))
        except Exception:
            traceback.print_exc()
            self.update(job_id, status="failed", stage="Ошибка обработки", error={"code": "processing_error", "message": "Не удалось обработать запись. Повторите попытку или проверьте журнал сервера."})


class Handler(BaseHTTPRequestHandler):
    server_version = "KFM/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def app(self):
        return self.server.app

    def log_message(self, fmt, *args):
        # Do not log API keys, uploads or request bodies.
        print(f"[{time.strftime('%H:%M:%S')}] {fmt % args}", flush=True)

    def headers_for(self, code, length, kind):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; media-src 'self' blob:; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'")

    def json(self, code, body):
        raw = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.headers_for(code, len(raw), "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(raw)

    def error_json(self, code, message, name="request_error"):
        self.close_connection = True
        self.json(code, {"error": {"code": name, "message": message}})

    def api_authorized(self, ui=False):
        if ui:
            try:
                cookie = SimpleCookie(self.headers.get("Cookie", ""))
                value = cookie["kfm_session"].value if "kfm_session" in cookie else ""
            except Exception:
                value = ""
            return secrets.compare_digest(value, self.app.session)
        return secrets.compare_digest(self.headers.get("X-API-Key", ""), self.app.api_key)

    def route_api(self, path):
        ui = path.startswith("/ui/api/")
        if not self.api_authorized(ui):
            self.error_json(401, "Необходим действующий ключ X-API-Key." if not ui else "Обновите страницу KFM.", "unauthorized")
            return None
        return path.removeprefix("/ui/api") if ui else path.removeprefix("/api/v1")

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/healthz":
            self.json(200, {"status": "ok", "service": "KFM"})
            return
        if path.startswith(("/api/v1/", "/ui/api/")):
            route = self.route_api(path)
            if route is None:
                return
            if route == "/config":
                ready = self.app.analyzer.ready() if hasattr(self.app.analyzer, "ready") else self.app.analyzer.connected
                self.json(200, {"model_connected": ready, "model_name": self.app.analyzer.name, "experimental": bool(getattr(self.app.analyzer, "experimental", False)), "max_upload_mb": self.app.max_bytes // 1024**2, "max_duration_sec": getattr(self.app.analyzer, "max_duration_sec", None), "supported_rule_ids": list(getattr(self.app.analyzer, "supported_rule_ids", ())), "formats": [".mov", ".mp4"], "local": True})
            elif route == "/jobs":
                self.json(200, {"jobs": self.app.recent()})
            elif route.startswith("/jobs/"):
                job_id = route.removeprefix("/jobs/")
                job = self.app.get(job_id) if JOB_ID.fullmatch(job_id) else None
                if job:
                    self.json(200, job)
                else:
                    self.error_json(404, "Проверка не найдена.", "not_found")
            else:
                self.error_json(404, "Маршрут не найден.", "not_found")
            return
        if path.startswith("/media/"):
            if not (self.api_authorized(True) or self.api_authorized(False)):
                self.error_json(401, "Откройте видео через KFM.", "unauthorized")
                return
            job_id = path.removeprefix("/media/")
            job = self.app.get(job_id) if JOB_ID.fullmatch(job_id) else None
            if not job or job["demo"]:
                self.error_json(404, "Видео не найдено.", "not_found")
                return
            extension = Path(job["filename"]).suffix.lower()
            self.send_file(self.app.uploads / f"{job_id}{extension}", video=True)
            return
        if path == "/" or re.fullmatch(r"/results/[0-9a-f]{32}/?", path):
            self.send_file(STATIC / "index.html", session=True)
            return
        if path.startswith("/static/"):
            candidate = (STATIC / unquote(path.removeprefix("/static/"))).resolve()
            if candidate.is_relative_to(STATIC.resolve()) and candidate.is_file():
                self.send_file(candidate)
                return
        self.error_json(404, "Страница не найдена.", "not_found")

    def do_POST(self):
        path = urlsplit(self.path).path
        if not path.startswith(("/api/v1/", "/ui/api/")):
            self.error_json(404, "Маршрут не найден.", "not_found")
            return
        route = self.route_api(path)
        if route is None:
            return
        if route not in {"/analyze", "/demo", "/retry"}:
            self.error_json(404, "Маршрут не найден.", "not_found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.error_json(400, "Некорректный размер запроса.")
            return
        limit = self.app.max_bytes + 1024 * 1024 if route == "/analyze" else 8192
        if length <= 0:
            self.error_json(400, "Пустой запрос.")
            return
        if length > limit:
            self.error_json(413, f"Максимальный размер видео — {self.app.max_bytes // 1024**2} МБ.", "file_too_large")
            return
        self.connection.settimeout(90)
        raw = self.rfile.read(length)
        if len(raw) != length:
            self.error_json(400, "Загрузка прервалась. Выберите файл повторно.", "upload_incomplete")
            return
        if route == "/analyze":
            self.accept_video(raw)
            return
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError()
        except (ValueError, UnicodeDecodeError):
            self.error_json(400, "Ожидается JSON-объект.")
            return
        if route == "/demo":
            scenario = payload.get("scenario", "violation")
            if scenario not in {"violation", "clean"}:
                self.error_json(400, "Неизвестный учебный сценарий.")
                return
            job = self.app.new_job(filename="Учебный сценарий" if scenario == "violation" else "Сценарий без событий", demo=True, scenario=scenario, metadata={"duration_sec": 42})
            self.app.submit(job)
            self.json(202, {"job_id": job["job_id"], "status": "processing", "demo": True})
        else:
            job_id = payload.get("job_id", "")
            old = self.app.get(job_id) if isinstance(job_id, str) and JOB_ID.fullmatch(job_id) else None
            if not old or old["demo"]:
                self.error_json(404, "Запись не найдена.", "not_found")
                return
            if old["status"] == "processing":
                self.error_json(409, "Запись уже обрабатывается.", "already_processing")
                return
            job = self.app.new_job(filename=old["filename"], size=old["size_bytes"], metadata=old)
            original = self.app.uploads / f"{old['job_id']}{Path(old['filename']).suffix.lower()}"
            destination = self.app.uploads / f"{job['job_id']}{Path(old['filename']).suffix.lower()}"
            try:
                import shutil
                shutil.copyfile(original, destination)
            except OSError:
                self.error_json(404, "Исходное видео отсутствует.", "not_found")
                return
            self.app.submit(job, destination)
            self.json(202, {"job_id": job["job_id"], "status": "processing"})

    def accept_video(self, raw):
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data") or "boundary=" not in content_type:
            self.error_json(400, "Отправьте видео в multipart/form-data, поле file.")
            return
        try:
            message = BytesParser(policy=policy.default).parsebytes(f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii") + raw)
            files = [part for part in message.iter_parts() if part.get_param("name", header="content-disposition") == "file"]
            if len(files) != 1:
                raise ValueError()
            part = files[0]
            filename = str(part.get_filename() or "").replace("\\", "/").split("/")[-1]
            filename = "".join(ch for ch in filename if ch.isprintable())[:220]
            content = part.get_payload(decode=True)
            if not filename or not isinstance(content, bytes) or not content:
                raise ValueError()
        except Exception:
            self.error_json(400, "Не удалось прочитать видео. Используйте поле file.")
            return
        extension = Path(filename).suffix.lower()
        if extension not in {".mov", ".mp4"}:
            self.error_json(415, "Поддерживаются видео MOV и MP4.", "unsupported_format")
            return
        if len(content) > self.app.max_bytes:
            self.error_json(413, "Видео превышает допустимый размер.", "file_too_large")
            return
        if len(content) < 12 or content[4:8] not in {b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip"}:
            self.error_json(415, "Файл не похож на видео MOV или MP4.", "invalid_video")
            return
        job = self.app.new_job(filename=filename, size=len(content))
        path = self.app.uploads / f"{job['job_id']}{extension}"
        path.write_bytes(content)
        job.update(read_metadata(path))
        self.app.submit(job, path)
        self.json(202, {"job_id": job["job_id"], "status": "processing", "demo": False})

    def send_file(self, path: Path, *, session=False, video=False):
        if not path.is_file():
            self.error_json(404, "Файл не найден.", "not_found")
            return
        total = path.stat().st_size
        start, end = 0, total - 1
        partial = False
        header = self.headers.get("Range") if video else None
        if header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
            if not match or not any(match.groups()):
                self.headers_for(416, 0, "video/mp4")
                self.send_header("Content-Range", f"bytes */{total}")
                self.end_headers()
                return
            first, last = match.groups()
            if not first:
                start = max(0, total - int(last))
            else:
                start = int(first)
                end = min(int(last), total - 1) if last else total - 1
            if start > end or start >= total:
                self.headers_for(416, 0, "video/mp4")
                self.send_header("Content-Range", f"bytes */{total}")
                self.end_headers()
                return
            partial = True
        kind = "video/mp4" if video else (mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        if kind in {"text/html", "text/css", "text/javascript", "application/javascript"}:
            kind += "; charset=utf-8"
        self.headers_for(206 if partial else 200, end - start + 1, kind)
        if session:
            self.send_header("Set-Cookie", f"kfm_session={self.app.session}; HttpOnly; SameSite=Strict; Path=/")
        if video:
            self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = handle.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


class KFMHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        # Windows SO_REUSEADDR permits two live servers on the same port.
        # Exclusive binding keeps browser sessions attached to one instance.
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def make_server(host="127.0.0.1", port=8765, app=None):
    server = KFMHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.app = app or App()
    return server


def main():
    load_env()
    host = os.environ.get("KFM_HOST", "127.0.0.1")
    port = int(os.environ.get("KFM_PORT", "8765"))
    try:
        server = make_server(host, port)
    except OSError:
        from urllib.request import ProxyHandler, build_opener
        address = f"http://127.0.0.1:{port}"
        try:
            with build_opener(ProxyHandler({})).open(address + "/healthz", timeout=2) as response:
                existing = json.load(response)
            if existing.get("service") != "KFM":
                raise ValueError("Another application uses the port")
        except Exception:
            raise RuntimeError(f"Port {port} is already in use. Set another KFM_PORT in .env.") from None
        print(f"KFM is already running: {address}", flush=True)
        if os.environ.get("KFM_OPEN_BROWSER") == "1":
            webbrowser.open(address)
        return
    print(f"\n  KFM is ready: http://{host}:{port}\n  Data stays on this computer.\n  API key file: {server.app.data / 'api_key.txt'}\n  Press Ctrl+C to stop.\n", flush=True)
    if os.environ.get("KFM_OPEN_BROWSER") == "1":
        webbrowser.open(f"http://127.0.0.1:{port}")
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.app.executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
