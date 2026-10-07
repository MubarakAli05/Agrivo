"""Real loopback HTTP contract checks; no datasets, downloads or model required."""

from contextlib import contextmanager
import http.client
import json
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from app.api.server import MAX_BODY_BYTES, MAX_QUESTION_CHARS, create_server


ROOT = Path(__file__).resolve().parent.parent
MISSING_ROOT = ROOT / "tests" / "dashboard-workspace-does-not-exist"


@contextmanager
def running_server():
    server = create_server(MISSING_ROOT, 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if thread.is_alive():
            raise AssertionError("HTTP test server failed to stop")


def request(server, method="GET", path="/api/status", body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def post(server, payload, headers=None):
    return request(server, "POST", "/api/answer", json.dumps(payload).encode("utf-8"),
                   headers or {"Content-Type": "application/json"})


class DashboardTests(unittest.TestCase):
    def assert_fallback_contract(self, body):
        payload = json.loads(body)
        self.assertIs(payload["unknown"], True)
        self.assertTrue(payload["answer"].startswith("UNKNOWN:"))
        self.assertEqual(payload["sources"], [])
        self.assertIsNone(payload["model_version"])
        self.assertIsNone(payload["confidence"])
        self.assertEqual(payload["data_used"], "None; no verified evidence returned by the API.")
        self.assertTrue(payload["error"]["code"])
        self.assertTrue(payload["error"]["message"])

    def test_loopback_status_missing_workspace_is_honest_and_read_only(self):
        with running_server() as server:
            self.assertEqual(server.server_address[0], "127.0.0.1")
            code, headers, body = request(server)
        payload = json.loads(body)
        self.assertEqual(code, 200)
        self.assertEqual(payload["workspace"]["status"], "RED")
        self.assertTrue(payload["workspace"]["missing_directories"])
        self.assertIsNone(payload["sources"])
        self.assertTrue(payload["sources_error"])
        self.assertTrue(payload["api"]["read_only"])
        self.assertEqual(payload["session"]["queries"], 0)
        self.assertFalse(MISSING_ROOT.exists())
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(int(headers["Content-Length"]), len(body))

    def test_status_contract_calls_workspace_status(self):
        report = {"status": "YELLOW", "training": "Not started", "metrics": None}
        with patch("agri.workspace.status", return_value=report) as status:
            with running_server() as server:
                code, _, body = request(server)
        status.assert_called_once_with(MISSING_ROOT.resolve())
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["workspace"], report)

    def test_security_headers_and_static_assets_are_local(self):
        with running_server() as server:
            for path, content_type in (("/", "text/html"), ("/index.html", "text/html"),
                                       ("/dashboard.css", "text/css"), ("/dashboard.js", "text/javascript")):
                with self.subTest(path=path):
                    code, headers, body = request(server, path=path)
                    self.assertEqual(code, 200)
                    self.assertTrue(headers["Content-Type"].startswith(content_type))
                    self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
                    self.assertEqual(headers["X-Frame-Options"], "DENY")
                    self.assertEqual(headers["Cache-Control"], "no-store")
                    self.assertEqual(headers["Referrer-Policy"], "no-referrer")
                    self.assertIn("connect-src 'self'", headers["Content-Security-Policy"])
                    self.assertNotIn("Access-Control-Allow-Origin", headers)
                    text = body.decode("utf-8")
                    self.assertNotIn("https://", text)
                    self.assertNotIn("http://", text)
                    self.assertNotIn("innerHTML", text)
                    self.assertNotIn("eval(", text)
        self.assertIn('lang="en"', (ROOT / "app" / "dashboard" / "index.html").read_text(encoding="utf-8"))

    def test_path_traversal_never_serves_workspace_files(self):
        paths = ["/../configs/agri-mini.json", "/%2e%2e/configs/agri-mini.json", "/%2e%2e%5cREADME.md",
                 "/%252e%252e/README.md", "/app/api/server.py", "/dashboard.js/../server.py",
                 "/dashboard.js%00", "/data/source_registry.yaml", "http://evil.example/api/status"]
        with running_server() as server:
            for path in paths:
                with self.subTest(path=path):
                    code, _, body = request(server, path=path,
                                            headers={"Host": f"127.0.0.1:{server.server_address[1]}"})
                    self.assertEqual(code, 404)
                    self.assertEqual(json.loads(body)["error"]["code"], "not_found")

    def test_answer_backend_is_lazy_and_token_limit_is_fixed(self):
        answer = {"answer": "<img src=x onerror=alert(1)>", "unknown": False,
                  "evidence": [{"source": "SYNTHETIC", "ph": 6.2}], "confidence": None}
        backend = unittest.mock.Mock()
        backend.answer_query.return_value = answer
        with patch("app.api.server.import_module", return_value=backend) as lazy_import:
            with running_server() as server:
                lazy_import.assert_not_called()
                code, headers, body = post(server, {"question": "  soil pH?  "})
        self.assertEqual(code, 200)
        self.assertIn("application/json", headers["Content-Type"])
        self.assertEqual(json.loads(body), answer)
        lazy_import.assert_called_once_with("retrieval.qa")
        backend.answer_query.assert_called_once_with(MISSING_ROOT.resolve(), "soil pH?", max_new_tokens=32)

    def test_missing_backend_or_checkpoint_returns_unknown(self):
        for error in (ModuleNotFoundError("retrieval.qa"), FileNotFoundError("checkpoint"), ValueError("approval")):
            with self.subTest(error=type(error).__name__):
                with patch("app.api.server._answer", side_effect=error):
                    with running_server() as server:
                        code, _, body = post(server, {"question": "soil?"})
                self.assertEqual(code, 503)
                self.assert_fallback_contract(body)

    def test_status_failure_is_honest(self):
        with patch("app.api.server._status", side_effect=ValueError("invalid config")):
            with running_server() as server:
                code, _, body = request(server)
        self.assertEqual(code, 503)
        self.assertEqual(json.loads(body)["error"]["code"], "status_unavailable")
        self.assert_fallback_contract(body)

    def test_request_rejections_keep_complete_answer_contract(self):
        cases = [
            ("POST", "/api/answer", b"{", {"Content-Type": "application/json"}, 400),
            ("POST", "/api/answer", b"{}", {"Origin": "https://other.example"}, 403),
            ("POST", "/api/absent", b"{}", {}, 404),
            ("POST", "/api/answer", b"", {"Content-Length": "-1"}, 411),
            ("POST", "/api/answer", b"x" * (MAX_BODY_BYTES + 1), {}, 413),
            ("POST", "/api/answer", b"{}", {"Content-Type": "text/plain"}, 415),
            ("OPTIONS", "/api/answer", None, {}, 501),
        ]
        with running_server() as server:
            for method, path, body, headers, expected in cases:
                with self.subTest(status=expected):
                    code, _, response = request(server, method, path, body, headers)
                    self.assertEqual(code, expected)
                    self.assert_fallback_contract(response)

    def test_invalid_json_and_question_are_rejected_before_backend(self):
        bodies = [b"{", b"null", b"[]", b'"question"', b'{"question": NaN}', b'\xff',
                  b'{"question":"one","question":"two"}', b'{"question":"\\ud800"}',
                  b'{"question":"\\u0000"}']
        bodies += [json.dumps(item).encode() for item in ({}, {"question": " "}, {"question": 7},
                   {"question": []}, {"question": "q", "max_new_tokens": 10000},
                   {"question": "x" * (MAX_QUESTION_CHARS + 1)})]
        with patch("app.api.server._answer") as backend:
            with running_server() as server:
                for body in bodies:
                    with self.subTest(body=body[:50]):
                        code, _, response = request(server, "POST", "/api/answer", body,
                                                    {"Content-Type": "application/json"})
                        self.assertEqual(code, 400)
                        self.assertTrue(json.loads(response)["unknown"])
        backend.assert_not_called()

    def test_exact_body_and_question_limits_are_accepted(self):
        with patch("app.api.server._answer", return_value={"answer": "UNKNOWN", "unknown": True}) as backend:
            with running_server() as server:
                body = json.dumps({"question": "x" * MAX_QUESTION_CHARS}).encode("utf-8")
                body += b" " * (MAX_BODY_BYTES - len(body))
                code, _, _ = request(server, "POST", "/api/answer", body,
                                     {"Content-Type": "application/json; charset=utf-8"})
                self.assertEqual(code, 200)
                backend.assert_called_once_with(MISSING_ROOT.resolve(), "x" * MAX_QUESTION_CHARS)

    def test_real_missing_backend_or_data_abstains(self):
        with running_server() as server:
            code, _, body = post(server, {"question": "What is soil pH?"})
        self.assertIn(code, (200, 503))
        self.assertTrue(json.loads(body)["unknown"])
        self.assertFalse(MISSING_ROOT.exists())

    def test_content_type_length_and_body_limits(self):
        with patch("app.api.server._answer") as backend:
            with running_server() as server:
                for content_type in ("text/plain", "application/x-www-form-urlencoded", "application/json; charset=latin1"):
                    code, _, _ = post(server, {"question": "soil?"}, {"Content-Type": content_type})
                    self.assertEqual(code, 415)
                code, _, _ = request(server, "POST", "/api/answer", b"{}")
                self.assertEqual(code, 415)
                code, _, _ = request(server, "POST", "/api/answer", b"x" * (MAX_BODY_BYTES + 1),
                                     {"Content-Type": "application/json"})
                self.assertEqual(code, 413)
                for length, expected in (("-1", 411), ("oops", 411), ("99999999999999", 413)):
                    code, _, _ = request(server, "POST", "/api/answer", b"",
                                         {"Content-Type": "application/json", "Content-Length": length})
                    self.assertEqual(code, expected)
                code, _, _ = request(server, "POST", "/api/answer", b"",
                                     {"Transfer-Encoding": "chunked", "Content-Type": "application/json"})
                self.assertEqual(code, 400)
        backend.assert_not_called()

    def test_origin_and_host_rejections_and_same_origin_success(self):
        with patch("app.api.server._answer", return_value={"answer": "UNKNOWN", "unknown": True}) as backend:
            with running_server() as server:
                for headers in ({"Origin": "https://evil.example"}, {"Origin": "null"},
                                {"Origin": "http://localhost:1"}, {"Sec-Fetch-Site": "cross-site"},
                                {"Sec-Fetch-Site": "same-site"}, {"Host": "evil.example"},
                                {"Host": "127.0.0.1.evil.example"}, {"Host": "127.0.0.1:1"}):
                    code, _, _ = post(server, {"question": "soil?"}, {"Content-Type": "application/json", **headers})
                    self.assertEqual(code, 403)
                backend.assert_not_called()
                code, _, _ = request(server, headers={"Host": "evil.example"})
                self.assertEqual(code, 403)
                origin = f"http://127.0.0.1:{server.server_address[1]}"
                code, _, _ = post(server, {"question": "soil?"}, {"Content-Type": "application/json",
                                                                        "Origin": origin, "Sec-Fetch-Site": "same-origin"})
                self.assertEqual(code, 200)

    def test_mutating_routes_and_preflight_are_unavailable(self):
        with running_server() as server:
            for method, path, expected in (("POST", "/api/train", 404), ("POST", "/api/ingest", 404),
                                           ("PUT", "/api/status", 501), ("OPTIONS", "/api/answer", 501)):
                code, headers, _ = request(server, method, path)
                self.assertEqual(code, expected)
                self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_concurrent_queries_are_bounded_and_counters_are_session_only(self):
        entered, release = threading.Event(), threading.Event()
        results = []

        def answer(root, question):
            entered.set()
            if not release.wait(timeout=5):
                raise AssertionError("Test did not release backend")
            return {"answer": "UNKNOWN", "unknown": True}

        with patch("app.api.server._answer", side_effect=answer):
            with running_server() as server:
                client = threading.Thread(target=lambda: results.append(post(server, {"question": "one"})))
                client.start()
                try:
                    self.assertTrue(entered.wait(timeout=5))
                    code, _, body = post(server, {"question": "two"})
                    self.assertEqual(code, 429)
                    self.assert_fallback_contract(body)
                    code, _, _ = request(server)
                    self.assertEqual(code, 200)
                finally:
                    release.set()
                    client.join(timeout=5)
                self.assertFalse(client.is_alive())
                self.assertEqual(results[0][0], 200)
                _, _, body = request(server)
                self.assertEqual(json.loads(body)["session"], {"queries": 1, "unknown": 1, "answered": 0, "errors": 0})

    def test_startup_does_not_import_torch_or_retrieval(self):
        result = subprocess.run([sys.executable, "-c", "from pathlib import Path; import sys; "
                                 "from app.api.server import create_server; "
                                 "s=create_server(Path('.'),0); s.server_close(); "
                                 "from app.api.server import _status; "
                                 "_status(Path('tests/dashboard-workspace-does-not-exist')); "
                                 "assert 'torch' not in sys.modules; assert 'retrieval.qa' not in sys.modules"],
                                cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_ports(self):
        for port in (-1, 65536, True, "8765"):
            with self.assertRaises(ValueError):
                create_server(ROOT, port)


if __name__ == "__main__":
    unittest.main()
