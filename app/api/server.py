"""Loopback-only, read-only AgriMini dashboard (no training or ingestion).

GET /api/status -> {workspace, sources, sources_error, api, session}.
POST /api/answer with {"question": str} -> retrieval.qa.answer_query's object.
Errors -> {error: {code, message}, unknown: true, answer: "UNKNOWN: ...",
           data_used: str, sources: [], model_version: null, confidence: null}.
Questions are limited to 512 characters / 8 KiB JSON and 32 generated tokens.
Counters are process-local, not persisted evaluation metrics. This server is for
trusted local users, not deployment behind a proxy or on a public interface.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
import json
from pathlib import Path
import threading
from typing import Any
from urllib.parse import unquote, urlsplit


MAX_BODY_BYTES = 8192
MAX_QUESTION_CHARS = 512
STATIC_ROOT = Path(__file__).resolve().parent.parent / "dashboard"
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/dashboard.css": ("dashboard.css", "text/css; charset=utf-8"),
    "/dashboard.js": ("dashboard.js", "text/javascript; charset=utf-8"),
}


def _answer(root: Path, question: str) -> dict[str, Any]:
    # Import only when asked: serving status must not initialize a model.
    return import_module("retrieval.qa").answer_query(root, question, max_new_tokens=32)


def _status(root: Path) -> dict[str, Any]:
    return import_module("agri.workspace").status(root)


def _sources(root: Path) -> dict[str, Any]:
    return import_module("agri.source_registry").inspect_sources(root)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def create_server(root: Path, port: int = 8765) -> ThreadingHTTPServer:
    """Create, but do not start, an IPv4 loopback server; port=0 selects a free port."""
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer between 0 and 65535")
    root = Path(root).resolve()
    query_slot = threading.BoundedSemaphore(1)
    counters_lock = threading.Lock()
    counters = {"queries": 0, "answered": 0, "unknown": 0, "errors": 0}

    class Handler(BaseHTTPRequestHandler):
        server_version = "AgriMiniLocal/1"
        sys_version = ""

        def setup(self) -> None:
            self.request.settimeout(5)
            super().setup()

        def log_message(self, format: str, *args: Any) -> None:
            # Questions and local paths are not written into access logs.
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("utf-8")
            self._send(code, body, "application/json; charset=utf-8")

        def _error(self, code: int, kind: str, message: str) -> None:
            self._json(code, {"error": {"code": kind, "message": message},
                              "unknown": True, "answer": "UNKNOWN: " + message,
                              "data_used": "None; no verified evidence returned by the API.",
                              "sources": [], "model_version": None, "confidence": None})

        def send_error(self, code: int, message: str | None = None,
                       explain: str | None = None) -> None:
            self._error(code, "http_error", "HTTP request rejected")

        def _local_request(self, *, post: bool = False) -> bool:
            hosts = self.headers.get_all("Host", [])
            actual_port = self.server.server_address[1]
            allowed = {f"127.0.0.1:{actual_port}", f"localhost:{actual_port}"}
            if actual_port == 80:
                allowed.update({"127.0.0.1", "localhost"})
            if len(hosts) != 1 or hosts[0].lower() not in allowed:
                self._error(403, "host_rejected", "Use this server's loopback address and port")
                return False
            if post:
                origins = self.headers.get_all("Origin", [])
                sites = self.headers.get_all("Sec-Fetch-Site", [])
                if (len(origins) > 1 or (origins and origins[0] != "http://" + hosts[0])
                        or any(site not in ("same-origin", "none") for site in sites)):
                    self._error(403, "origin_rejected", "Cross-origin queries are not allowed")
                    return False
            return True

        def _path(self) -> str:
            parts = urlsplit(self.path)
            if parts.scheme or parts.netloc:
                return ""
            return unquote(parts.path)

        def do_GET(self) -> None:
            if not self._local_request():
                return
            try:
                path = self._path()
            except ValueError:
                self._error(400, "invalid_path", "Malformed request path")
                return
            if path == "/api/status":
                try:
                    workspace = _status(root)
                    sources, sources_error = None, None
                    try:
                        sources = _sources(root)
                    except (ImportError, OSError, ValueError) as exc:
                        sources_error = str(exc)
                    with counters_lock:
                        session = dict(counters)
                    self._json(200, {"workspace": workspace, "sources": sources,
                                     "sources_error": sources_error,
                                     "api": {"status": "GREEN", "local_only": True,
                                             "read_only": True, "max_question_chars": MAX_QUESTION_CHARS,
                                             "max_new_tokens": 32}, "session": session})
                except Exception:
                    self._error(503, "status_unavailable", "Workspace status unavailable; check local data and configuration")
            elif path in ASSETS:
                filename, content_type = ASSETS[path]
                try:
                    body = (STATIC_ROOT / filename).read_bytes()
                except OSError:
                    self._error(503, "asset_unavailable", "Dashboard asset unavailable")
                    return
                self._send(200, body, content_type)
            else:
                self._error(404, "not_found", "No such local resource")

        def do_POST(self) -> None:
            if not self._local_request(post=True):
                return
            try:
                path = self._path()
            except ValueError:
                path = ""
            if path != "/api/answer":
                self._error(404, "not_found", "No such local endpoint")
                return
            if self.headers.get_all("Transfer-Encoding"):
                self._error(400, "invalid_length", "Transfer encoding is not supported")
                return
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
                self._error(411, "invalid_length", "A single valid Content-Length is required")
                return
            if len(lengths[0]) > 8 or int(lengths[0]) > MAX_BODY_BYTES:
                self._error(413, "body_too_large", "JSON body exceeds 8 KiB")
                return
            content_types = self.headers.get_all("Content-Type", [])
            if len(content_types) != 1 or self.headers.get_content_type() != "application/json":
                self._error(415, "invalid_content_type", "Content-Type must be application/json")
                return
            if self.headers.get_content_charset("utf-8").lower() not in ("utf-8", "utf8"):
                self._error(415, "invalid_charset", "JSON must use UTF-8")
                return
            try:
                body = self.rfile.read(int(lengths[0]))
                if len(body) != int(lengths[0]):
                    raise ValueError("Incomplete body")
                payload = json.loads(body.decode("utf-8"), object_pairs_hook=_unique_object,
                                     parse_constant=_invalid_constant)
                if not isinstance(payload, dict) or set(payload) != {"question"}:
                    raise ValueError("Expected question object")
                question = payload["question"]
                if (not isinstance(question, str) or not question.strip()
                        or len(question) > MAX_QUESTION_CHARS):
                    raise ValueError("Invalid question")
                question.encode("utf-8")
                if any(ord(char) < 32 and char not in "\t\n\r" for char in question):
                    raise ValueError("Control character")
            except TimeoutError:
                self._error(408, "request_timeout", "Timed out reading JSON body")
                return
            except (ValueError, UnicodeError, RecursionError):
                self._error(400, "invalid_question", "Supply valid UTF-8 JSON with only a nonempty question of at most 512 characters")
                return
            if not query_slot.acquire(blocking=False):
                self._error(429, "busy", "One local query is already running; retry after it finishes")
                return
            try:
                with counters_lock:
                    counters["queries"] += 1
                result = _answer(root, question.strip())
                if not isinstance(result, dict):
                    raise ValueError("Backend did not return an object")
                encoded = json.dumps(result, allow_nan=False).encode("utf-8")
                with counters_lock:
                    counters["unknown" if result.get("unknown") is not False else "answered"] += 1
                self._send(200, encoded, "application/json; charset=utf-8")
            except (ImportError, FileNotFoundError):
                with counters_lock:
                    counters["errors"] += 1
                self._error(503, "backend_unavailable", "Query backend or required local data/checkpoint is missing")
            except Exception:
                with counters_lock:
                    counters["errors"] += 1
                self._error(503, "answer_unavailable", "Cannot answer from validated local evidence; check data, checkpoint and approvals")
            finally:
                query_slot.release()

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(root: Path, port: int = 8765) -> None:
    """Serve until interrupted, then close the listening socket."""
    server = create_server(root, port)
    try:
        print(f"AgriMini local dashboard: http://127.0.0.1:{server.server_address[1]}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
