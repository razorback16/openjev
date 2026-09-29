import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import socket
import subprocess
import pytest

import httpx

from openjev import forjev_service as service


def test_refuses_unrelated_process(tmp_path, monkeypatch):
    pidfile = tmp_path / "process.json"
    pidfile.write_text(json.dumps({"pid": 123, "birth": "456"}))
    monkeypatch.setattr(service, "PIDFILE", pidfile)
    monkeypatch.setattr(service, "birth", lambda pid: "456")
    monkeypatch.setattr(service, "owned_process", lambda pid: False)
    with pytest.raises(RuntimeError, match="refusing"):
        service.stop()


def test_start_probe_stop_preserves_upstream(tmp_path, monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def reply(self, data):
            raw = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def do_GET(self):
            self.reply({"data": [{"id": "qwen"}]} if self.path == "/v1/models" else {"status": "ok"})
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                self.reply({"tokens": [ord(data["prompt"])]})
            else:
                self.reply({"choices": [{"logprobs": {"content": [{"top_logprobs": [
                    {"token": f"token_id:{i}", "logprob": -1.0} for i in data["logprob_token_ids"]]}]}}],
                    "usage": {"prompt_tokens": 10}})
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    for key, value in {"OPENJEV_BACKEND": "forjev", "OPENJEV_UPSTREAM": f"http://127.0.0.1:{server.server_port}",
                       "OPENJEV_UPSTREAM_MODEL": "qwen", "OPENJEV_HOST": "127.0.0.1", "OPENJEV_PORT": str(port),
                       "FORJEV_MAX_CHOICES": "2", "FORJEV_READY_TIMEOUT": "10"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(service, "STATE", tmp_path)
    monkeypatch.setattr(service, "PIDFILE", tmp_path / "process.json")
    # This CI sandbox mounts /proc outside the Python PID namespace. Model
    # process identity with Popen handles; retain real launch, HTTP and signals.
    children = {}
    real_popen = subprocess.Popen
    def spawn(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children[child.pid] = child
        return child
    monkeypatch.setattr(service.subprocess, "Popen", spawn)
    monkeypatch.setattr(service, "birth", lambda pid: str(pid) if pid in children and children[pid].poll() is None else None)
    monkeypatch.setattr(service, "owned_process", lambda pid: pid in children)
    try:
        service.start()
        info = service.current()
        assert info and info["port"] == port
        service.start()
        assert service.current()["pid"] == info["pid"]
        assert httpx.get(f"http://127.0.0.1:{port}/ready", trust_env=False).status_code == 200
        service.stop()
        assert service.current() is None
        assert httpx.get(f"http://127.0.0.1:{server.server_port}/health", trust_env=False).status_code == 200
    finally:
        service.stop()
        server.shutdown()
        server.server_close()
