"""Download the exact inputs in dataset/sources.lock.json; verify every SHA256."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(item, output):
    relative = Path(item["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe input path")
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and sha256(path) == item["sha256"]:
        return str(relative) + " verified"
    temporary = path.with_name(path.name + ".download")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(item["url"], timeout=120) as src, temporary.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if temporary.stat().st_size != item["bytes"] or sha256(temporary) != item["sha256"]:
                raise ValueError(f"Input checksum mismatch: {relative}")
            temporary.replace(path)
            return str(relative) + " downloaded and verified"
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/openjev-pilot/source")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    lock = json.loads((ROOT / "dataset/sources.lock.json").read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(lambda item: download(item, args.output), lock["files"]):
            print(result, flush=True)


if __name__ == "__main__":
    main()
