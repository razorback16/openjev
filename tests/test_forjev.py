import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from openjev.api import create_app
from openjev.config import Settings
from openjev.forjev import ForJevEngine, LABELS
from openjev.engine import Upstream
from openjev.decision_scores import score_response


@pytest.fixture
def upstream(monkeypatch):
    calls = []
    ids = {label: i + 10 for i, label in enumerate(LABELS)}
    def respond(request):
        body = json.loads(request.content) if request.content else {}
        calls.append((request, body))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [ids[body["prompt"]]]})
        if body.get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                 stream=httpx.ByteStream(b'data: {"delta":"ok"}\n\ndata: [DONE]\n\n'))
        candidates = body["logprob_token_ids"]
        return httpx.Response(200, json={"choices": [{"logprobs": {"content": [{"top_logprobs": [
            {"token": f"token_id:{i}", "logprob": -0.1 if i == candidates[-1] else -3.0}
            for i in candidates]}]}}], "usage": {"prompt_tokens": 42, "completion_tokens": 1}})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(
        **{**kw, "transport": httpx.MockTransport(respond)}))
    return calls


def settings(**kw):
    return Settings(backend="forjev", upstream="http://qwen", upstream_model="qwen", **kw)


def body():
    return {"model": "forjev-qwen-next", "state": {"health": 4}, "questions": {
        "move": {"type": "choice", "instructions": {"goal": "survive"},
                 "criteria": {"stay": "Stay here", "flee": "Run away"}},
        "risk": {"type": "score", "criteria": ["low", "high"]},
        "urgent": {"type": "noul", "instructions": "Is this urgent?"}}}


@pytest.mark.parametrize("images", [None, [], ["data:image/jpeg;base64,/9j/2Q=="]])
def test_native_api_typed_answers_and_optional_images(upstream, images):
    with TestClient(create_app(settings())) as client:
        response = client.post("/v1/systemone", json={**body(), "images": images})
        assert response.status_code == 200, response.text
        answers = response.json()["answers"]
        assert answers["move"]["choice"] == "flee"
        assert sum(answers["move"]["probabilities"].values()) == pytest.approx(1)
        assert 0 <= answers["risk"]["score"] <= 1
        assert 0 <= answers["urgent"]["noul"] <= 1
        assert response.json()["usage"] == {"input_tokens": 126, "output_tokens": 0}
    chats = [b for r, b in upstream if r.url.path == "/v1/chat/completions"]
    assert len(chats) == 3
    for b in chats:
        assert b["max_tokens"] == 1 and b["logprobs"] is True
        assert b["chat_template_kwargs"] == {"enable_thinking": False}
        assert b["messages"][1]["content"][0]["type"] == ("image_url" if images else "text")


def test_typesafe_sdk_default_model_uses_forjev(upstream):
    from typesafe_sdk import TypeSafeClient

    with TestClient(create_app(settings())) as client:
        sdk = TypeSafeClient(api_key="unused", base_url="http://testserver", http_client=client)
        result = sdk.system_one("A zombie is approaching.", {
            "move": {"type": "choice", "instructions": "Choose a safe action.",
                     "criteria": {"stay": "Remain in place", "flee": "Move away"}},
        })
        assert result.choices["move"].choice == "flee"
        assert result.choices["move"].probabilities["flee"] > 0.9
    assert any(req.url.path == "/v1/chat/completions" for req, _ in upstream)


@pytest.mark.parametrize("field,value", [("think", 1), ("sequential", True), ("steps", 2), ("samples", 2)])
def test_unsupported_options_rejected_before_inference(upstream, field, value):
    with TestClient(create_app(settings())) as client:
        assert client.post("/v1/systemone", json={**body(), field: value}).status_code == 400
    assert not upstream


def test_labels_beyond_single_characters_are_unique(upstream):
    async def run():
        engine = ForJevEngine(settings(forjev_max_choices=255))
        try:
            pairs = await engine._ids(255)
            assert len(set(t for _, t in pairs)) == 255
            assert pairs[62][0] == "AA"
            assert len({label for label, _ in pairs}) == 255
        finally:
            await engine.close()
    asyncio.run(run())


def test_missing_candidates_fail_closed():
    async def run():
        engine = ForJevEngine(settings())
        async def post(path, data):
            if path == "/tokenize":
                return {"tokens": [ord(data["prompt"])]}
            return {"choices": [{"logprobs": {"content": [{"top_logprobs": [
                {"token": "token_id:65", "logprob": -0.1}]}]}}], "usage": {"prompt_tokens": 1}}
        engine._post = post
        try:
            with pytest.raises(Upstream, match="incomplete"):
                await engine.decide(body()["questions"], "state", 0)
        finally:
            await engine.close()
    asyncio.run(run())


@pytest.mark.parametrize("statuses,path,message,expected_calls", [
    ([500, 500, 200], "/v1/chat/completions", "list index out of range", 3),
    ([500, 500, 500], "/v1/chat/completions", "list index out of range", 3),
    ([500, 200], "/tokenize", "list index out of range", 1),
    ([500, 200], "/v1/chat/completions", "other failure", 1),
    ([503, 200], "/v1/chat/completions", "unavailable", 1),
    ([400, 200], "/v1/chat/completions", "invalid request", 1),
])
def test_only_known_chat_logprob_failure_is_retried(monkeypatch, statuses, path, message, expected_calls):
    calls = []
    original = httpx.AsyncClient
    def respond(request):
        calls.append(json.loads(request.content))
        status = statuses[len(calls) - 1]
        if status == 500:
            return httpx.Response(500, json={"error": {"message": message}})
        if status == 400:
            return httpx.Response(400, json={"error": {"message": "invalid request"}})
        return httpx.Response(status, json={"ok": True})
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(
        **{**kw, "transport": httpx.MockTransport(respond)}))

    async def run():
        engine = ForJevEngine(settings())
        try:
            if statuses[-1] == 200 and expected_calls == len(statuses):
                assert await engine._post(path, {"max_tokens": 1}) == {"ok": True}
            elif statuses[0] == 400:
                with pytest.raises(Upstream, match="invalid request"):
                    await engine._post(path, {"max_tokens": 1})
            else:
                with pytest.raises(httpx.HTTPStatusError):
                    await engine._post(path, {"max_tokens": 1})
        finally:
            await engine.close()
    asyncio.run(run())
    assert len(calls) == expected_calls
    assert all(body == calls[0] for body in calls)


def test_chat_proxy_stream_tools_and_separate_credentials(upstream):
    with TestClient(create_app(settings(api_key="frontend", forjev_upstream_api_key="upstream"))) as client:
        assert client.post("/v1/chat/completions", json={}).status_code == 403
        headers = {"Authorization": "Bearer frontend"}
        models = client.get("/v1/models", headers=headers).json()
        assert {m["id"] for m in models["data"]} == {"qwen", "forjev-qwen-next"}
        payload = {"model": "qwen", "stream": True, "temperature": 0.8,
                   "messages": [{"role": "user", "content": "hi"}],
                   "tools": [{"type": "function", "function": {"name": "look"}}]}
        response = client.post("/v1/chat/completions", json=payload, headers=headers)
        assert response.status_code == 200
        assert response.content.endswith(b'data: [DONE]\n\n')
        assert upstream[-1][1] == payload
        assert upstream[-1][0].headers["authorization"] == "Bearer upstream"
        assert client.get("/ready").status_code == 200


def test_choice_cap_and_forced_answer(upstream):
    with TestClient(create_app(settings())) as client:
        data = body()
        data["questions"] = {"one": {"type": "choice", "criteria": {"only": "only"}}}
        assert client.post("/v1/systemone", json=data).json()["answers"]["one"]["choice"] == "only"
        assert not upstream
        data["questions"]["one"]["criteria"] = {str(i): str(i) for i in range(21)}
        assert client.post("/v1/systemone", json=data).status_code == 400
        assert not upstream


@pytest.mark.parametrize("mode,execution", [("engine_scores", "engine_logprobs"),
                                           ("prefill_scores", "prefill_logits")])
def test_numeric_scoring_preserves_typed_answers_and_images(monkeypatch, mode, execution):
    calls = []
    original = httpx.AsyncClient
    def respond(request):
        payload = json.loads(request.content)
        calls.append((request.url.path, payload))
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [ord(payload["prompt"])]})
        assert request.url.path == "/v1/decision_scores"
        ids = payload["candidate_token_ids"]
        assert payload["require_prefill"] == (mode == "prefill_scores")
        assert not any(k in payload for k in ("logprobs", "max_tokens", "top_logprobs"))
        return httpx.Response(200, json=score_response(
            ids[::-1], [-0.1] + [-3.0] * (len(ids) - 1), 42,
            score_type="raw_logprobs", execution=execution))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(
        **{**kw, "transport": httpx.MockTransport(respond)}))
    with TestClient(create_app(settings(forjev_scoring=mode))) as client:
        response = client.post("/v1/systemone", json={**body(),
            "images": ["data:image/jpeg;base64,/9j/2Q=="]})
        assert response.status_code == 200, response.text
        answers = response.json()["answers"]
        assert answers["move"]["choice"] == "flee"
        assert answers["move"]["probabilities"]["flee"] > 0.9
        assert response.json()["usage"]["input_tokens"] == 126
    scores = [b for path, b in calls if path == "/v1/decision_scores"]
    assert len(scores) == 3
    assert all(b["messages"][1]["content"][0]["type"] == "image_url" for b in scores)


def test_scoring_setting_rejects_typos():
    with pytest.raises(ValueError, match="FORJEV_SCORING"):
        settings(forjev_scoring="direct")
