import argparse
import asyncio
import json

import httpx
import pytest

from openjev.decision_scores import score_response
from openjev.forjev_parity import comparisons, replay


def test_comparisons_reject_different_prompts_and_candidates():
    row = dict(ok=True, question="q", phase="serial", request_hash="same",
               token_ids=[1, 2], probabilities=[.8, .2])
    assert comparisons([row, {**row, "phase": "parallel", "probabilities": [.4, .6]}])[0]["winner_changed"]
    for changed in ({"request_hash": "different"}, {"token_ids": [2, 1]}):
        with pytest.raises(ValueError):
            comparisons([row, {**row, **changed}])


@pytest.mark.parametrize("skip_cache", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_replay_identical_prompts_provider_selection_and_no_retry(tmp_path, monkeypatch, fail, skip_cache):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"runs": [{"cases": [{"calls": [{
        "state": {"invoice": "fixed state"},
        "questions": {k: {"type": "choice", "instructions": "Choose x.",
                           "criteria": {"x": "target", "y": "alternative"}}
                      for k in ("one", "two")}}]}]}]}))
    bodies = []
    def respond(request):
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [ord(body["prompt"])]})
        bodies.append(body)
        if fail:
            return httpx.Response(503, json={"detail": "failure"})
        native = body["require_prefill"]
        return httpx.Response(200, json=score_response(
            body["candidate_token_ids"], [-1, -2], 40,
            score_type="raw_logits" if native else "raw_logprobs",
            execution="prefill_logits" if native else "engine_logprobs"))
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        **{**kwargs, "transport": httpx.MockTransport(respond)}))
    args = argparse.Namespace(results=str(path), questions="one,two", timeout=3,
                              repeats=2, parallel=2, skip_prefix_cache=skip_cache)
    result = asyncio.run(replay(args))
    assert result["stopped_on_error"] == fail
    assert len(bodies) == (2 if fail else 16)
    assert all(b.get("skip_reading_prefix_cache", False) == skip_cache for b in bodies)
    assert all(b["messages"] == bodies[0]["messages"] for b in bodies)
    assert all(b["candidate_token_ids"] == bodies[0]["candidate_token_ids"] for b in bodies)
    if not fail:
        assert {r["execution"] for r in result["records"]} == {"prefill_logits", "engine_logprobs"}
        assert all(r["max_probability_delta"] == 0 for r in result["comparisons"])
