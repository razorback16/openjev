"""Replay saved WorkflowEvals questions on serial/concurrent numeric providers.

No retries, serving restart, or cache reset. Reports contain no state/prompts.
Both providers receive the same request, except for require_prefill.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

import httpx

from .config import Settings
from .decision_scores import parse_response
from .forjev import ForJevEngine


def comparisons(records):
    groups = {}
    for row in records:
        if row.get("ok"):
            groups.setdefault(row["question"], []).append(row)
    out = []
    for question, rows in groups.items():
        for i, left in enumerate(rows):
            for right in rows[i + 1:]:
                if left["request_hash"] != right["request_hash"]:
                    raise ValueError("Comparison prompts differ")
                if left["token_ids"] != right["token_ids"]:
                    raise ValueError("Comparison candidates differ")
                a, b = left["probabilities"], right["probabilities"]
                out.append({"question": question, "left": left["phase"],
                            "right": right["phase"],
                            "max_probability_delta": max(abs(x-y) for x, y in zip(a, b)),
                            "winner_changed": a.index(max(a)) != b.index(max(b))})
    return sorted(out, key=lambda r: r["max_probability_delta"], reverse=True)


def saved_call(path, wanted):
    data = json.loads(Path(path).read_text())
    matches = [call for run in data["runs"] for case in run["cases"]
               for call in case["calls"] if set(wanted) <= set(call["questions"])]
    if len(matches) != 1:
        raise ValueError(f"Expected one saved call containing requested questions; found {len(matches)}")
    return matches[0]


async def replay(args):
    wanted = args.questions.split(",")
    call = saved_call(args.results, wanted)
    engine = ForJevEngine(Settings())
    engine.client.timeout = httpx.Timeout(args.timeout, connect=5)
    records = []
    try:
        qs, forced = engine.build_schema({k: call["questions"][k] for k in wanted})
        if forced or {q["key"] for q in qs} != set(wanted):
            raise ValueError("Replay questions must require model scoring")
        state = call["state"]
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        requests = []
        for q in qs:
            pairs = await engine._ids(len(q["choices"]))
            body = {**engine.question_request(state_text, q, None, pairs),
                    "candidate_token_ids": [tid for _, tid in pairs]}
            digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            requests.append((q["key"], body, digest))

        async def score(item, phase, native, slots):
            question, body, digest = item
            row = {"question": question, "phase": phase, "request_hash": digest,
                   "token_ids": body["candidate_token_ids"]}
            async with slots:
                started = time.perf_counter()
                try:
                    # Direct post: intentionally bypass the adapter/SDK retry logic.
                    cache_options = ({"skip_reading_prefix_cache": True}
                                     if getattr(args, "skip_prefix_cache", False) else {})
                    response = await engine.client.post("/v1/decision_scores",
                                                        json={**body, "require_prefill": native, **cache_options})
                    response.raise_for_status()
                    data = response.json()
                    probs, tokens = parse_response(data, row["token_ids"], require_prefill=native)
                    expected = "prefill_logits" if native else "engine_logprobs"
                    if data["execution"] != expected:
                        raise ValueError("Unexpected provider execution")
                    row.update(ok=True, probabilities=probs, scores=data["scores"],
                               input_tokens=tokens, execution=data["execution"],
                               generated_tokens=data["generated_tokens"])
                except Exception as exc:
                    row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                row["seconds"] = time.perf_counter() - started
            return row

        # Reference first: native requests in this run cannot affect its cache.
        providers = {"both": (False, True), "engine": (False,), "native": (True,)}
        phases = [(native, repeat, width) for native in providers[getattr(args, "providers", "both")]
                  for repeat in range(args.repeats) for width in (1, args.parallel)]
        for native, repeat, width in phases:
            phase = f"{'native' if native else 'engine'}-c{width}-r{repeat+1}"
            slots = asyncio.Semaphore(width)
            rows = await asyncio.gather(*(score(item, phase, native, slots) for item in requests))
            records.extend(rows)
            print(json.dumps({"phase": phase, "ok": sum(r["ok"] for r in rows),
                              "total": len(rows), "max_seconds": max(r["seconds"] for r in rows),
                              "errors": [r["error"] for r in rows if not r["ok"]]}), flush=True)
            if any(not r["ok"] for r in rows):
                # Finish the failed phase, then stop instead of accumulating long timeouts.
                return {"skip_reading_prefix_cache": getattr(args, "skip_prefix_cache", False),
                        "records": records, "comparisons": comparisons(records),
                        "stopped_on_error": True}
        return {"skip_reading_prefix_cache": getattr(args, "skip_prefix_cache", False),
                "records": records, "comparisons": comparisons(records), "stopped_on_error": False}
    finally:
        await engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--questions", default="line_0_evidence_1,line_1_unexplained_fee,line_2_scope")
    parser.add_argument("--parallel", type=int, default=2, choices=range(2, 9))
    parser.add_argument("--repeats", type=int, default=2, choices=range(1, 4))
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--providers", choices=("both", "engine", "native"), default="both")
    parser.add_argument("--skip-prefix-cache", action="store_true",
                        help="Recompute prompts without reading existing prefix cache (writes remain enabled)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    report = asyncio.run(replay(args))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(path), "largest_differences": report["comparisons"][:6]}, indent=2))
    raise SystemExit(1 if report["stopped_on_error"] else 0)


if __name__ == "__main__":
    main()
