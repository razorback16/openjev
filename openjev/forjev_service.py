"""Linux controller for the HTTP adapter only. Never manages Docker or Qwen."""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse

import httpx

from .forjev_probe import probe

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / ".forjev-run"
PIDFILE = STATE / "process.json"


def birth(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def owned_process(pid):
    command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    return b"openjev.api:create_app" in command and Path(f"/proc/{pid}/cwd").resolve() == ROOT


def current():
    if not PIDFILE.exists():
        return None
    info = json.loads(PIDFILE.read_text())
    if birth(info["pid"]) != info["birth"]:
        return None
    if not owned_process(info["pid"]):
        raise RuntimeError("Stored PID is not this checkout's ForJev process; refusing to manage it")
    return info


def stop():
    info = current()
    if not info:
        return
    os.kill(info["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 30
    while birth(info["pid"]) == info["birth"]:
        if time.monotonic() > deadline:
            raise RuntimeError("ForJev did not stop within 30s; no forced kill was issued")
        time.sleep(0.2)
    PIDFILE.unlink(missing_ok=True)


def start():
    if current():
        print(json.dumps({"status": "already_running", **current()}))
        return
    host = os.environ.get("OPENJEV_HOST", "0.0.0.0")
    port = int(os.environ.get("OPENJEV_PORT", "8001"))
    upstream = urlparse(os.environ.get("OPENJEV_UPSTREAM", "http://127.0.0.1:8000"))
    if upstream.hostname in {"localhost", "127.0.0.1", "0.0.0.0", "::1"} and port == (upstream.port or 80):
        raise RuntimeError("ForJev must not bind the upstream model's port")
    with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET) as sock:
        sock.bind((host, port))
    # No port changes until the actual candidate-logprob contract is checked.
    result = asyncio.run(probe(int(os.environ.get("FORJEV_MAX_CHOICES", "20"))))
    (STATE / "upstream-probe.json").write_text(json.dumps(result, indent=2))
    log_path = STATE / f"serve-{time.time_ns()}.log"
    with log_path.open("ab") as log:
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "openjev.api:create_app",
                                 "--factory", "--host", host, "--port", str(port)],
                                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                start_new_session=True)
    started = birth(proc.pid)
    if started is None:
        proc.terminate()
        proc.wait(timeout=10)
        raise RuntimeError(f"ForJev exited at launch; read {log_path}")
    info = {"pid": proc.pid, "birth": started, "port": port, "log": str(log_path)}
    PIDFILE.write_text(json.dumps(info))
    address = "127.0.0.1" if host == "0.0.0.0" else ("[::1]" if host == "::" else host)
    base = f"http://{address}:{port}"
    headers = {}
    if os.environ.get("OPENJEV_API_KEY"):
        headers["authorization"] = "Bearer " + os.environ["OPENJEV_API_KEY"]
    if os.environ.get("OPENJEV_ORIGIN_SECRET"):
        headers["x-origin-secret"] = os.environ["OPENJEV_ORIGIN_SECRET"]
    deadline = time.monotonic() + int(os.environ.get("FORJEV_READY_TIMEOUT", "120"))
    try:
        with httpx.Client(base_url=base, headers=headers, timeout=5, trust_env=False) as client:
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"ForJev exited: {proc.returncode}; read {log_path}")
                try:
                    client.get("/ready").raise_for_status()
                    models = client.get("/v1/models")
                    models.raise_for_status()
                    if "forjev-qwen-next" not in {m["name"] for m in models.json()["models"]}:
                        raise RuntimeError("Wrong model registry on ForJev port")
                    break
                except httpx.HTTPError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"ForJev readiness timeout; read {log_path}")
                    time.sleep(1)
            response = client.post("/v1/systemone", json={"model": "forjev-qwen-next",
                "state": "The light is red.", "questions": {"red": {"type": "noul",
                "instructions": "Is the light red?"}}}, timeout=120)
            response.raise_for_status()
            data = response.json()
            if data.get("model") != "forjev-qwen-next" or "red" not in data.get("answers", {}):
                raise RuntimeError("SystemOne smoke returned an unexpected response")
        print(json.dumps({"status": "ready", "url": base, **info}))
    except BaseException:
        stop()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "restart", "stop", "status", "probe"], nargs="?", default="status")
    args = parser.parse_args()
    if os.environ.get("OPENJEV_BACKEND") != "forjev":
        parser.error("OPENJEV_BACKEND must be forjev")
    STATE.mkdir(mode=0o700, exist_ok=True)
    with (STATE / "lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "status":
            print(json.dumps({"status": "running" if current() else "stopped", "process": current()}))
        elif args.action == "probe":
            print(json.dumps(asyncio.run(probe(int(os.environ.get("FORJEV_MAX_CHOICES", "20"))))))
        else:
            if args.action in {"stop", "restart"}:
                # On restart, check Qwen before disrupting an existing adapter.
                if args.action == "restart":
                    asyncio.run(probe(int(os.environ.get("FORJEV_MAX_CHOICES", "20"))))
                stop()
            if args.action != "stop":
                start()


if __name__ == "__main__":
    main()
