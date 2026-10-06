"""Read-only upstream capability check, including real candidate logprobs."""
import argparse
import asyncio
import base64
import json
import mimetypes
from pathlib import Path

from .config import Settings
from .forjev import ForJevEngine


async def probe(count, image=None):
    engine = ForJevEngine(Settings())
    try:
        response = await engine.client.get("/health")
        response.raise_for_status()
        models = await engine.client.get("/v1/models")
        models.raise_for_status()
        if engine.s.upstream_model not in {m["id"] for m in models.json().get("data", [])}:
            raise RuntimeError("OPENJEV_UPSTREAM_MODEL is not in upstream /v1/models")
        if count > engine.max_choices:
            raise ValueError("Set FORJEV_MAX_CHOICES to the requested count first (maximum 255)")
        pairs = await engine._ids(count)
        images = None
        if image:
            path = Path(image)
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            images = [{"type": "image_url", "image_url": {
                "url": f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()}}]
        answers, tokens, _ = await engine.decide({"probe": {
            "type": "choice", "instructions": "Select option_0.",
            "criteria": {f"option_{i}": "target" if i == 0 else "alternative" for i in range(count)}
        }}, "ForJev capability probe", 0, images)
        return {"status": "ok", "model": engine.s.upstream_model,
                "scoring": engine.s.forjev_scoring,
                "candidate_count": count, "labels": dict(pairs), "input_tokens": tokens,
                "answer": answers["probe"], "image_included": bool(image)}
    finally:
        await engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--choices", type=int, default=20, choices=range(2, 256), metavar="2..255")
    parser.add_argument("--image")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(probe(args.choices, args.image)), ensure_ascii=False))


if __name__ == "__main__":
    main()
