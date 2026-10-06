import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from openjev.decision_scores import score_response
from openjev.vllm_scores import engine_scores, install_routes


def body(**kwargs):
    return {"model": "qwen", "messages": [{"role": "user", "content": "Choose A or B"}],
            "candidate_token_ids": [10, 20], **kwargs}


def make_app(**kwargs):
    app = FastAPI()
    app.state.openai_serving_chat = SimpleNamespace(model_config=SimpleNamespace(
        get_vocab_size=lambda: 100))
    install_routes(app, **kwargs)
    return app


def test_route_uses_existing_renderer_and_unique_engine_request_ids():
    calls = []
    async def render(chat, request):
        calls.append(request.messages)
        return {"prompt_token_ids": [1, 2], "multi_modal_data": {"image": "existing"}}
    async def score(chat, prompt, ids, **kwargs):
        calls.append(kwargs["request_id"])
        assert prompt["multi_modal_data"]["image"] == "existing"
        return score_response(ids, [-2.0, -0.2], 2,
                              score_type="raw_logprobs", execution="engine_logprobs")
    with TestClient(make_app(render=render, score=score)) as client:
        first = client.post("/v1/decision_scores", json=body())
        second = client.post("/v1/decision_scores", json=body())
    assert first.status_code == second.status_code == 200
    assert first.json()["probabilities"][1] > 0.8
    assert calls[1] != calls[3]


@pytest.mark.parametrize("change,status", [
    ({"require_prefill": True}, 501), ({"candidate_token_ids": [10, 10]}, 400),
    ({"candidate_token_ids": [10, 100]}, 400),
    ({"candidate_token_ids": [True, 20]}, 422),
    ({"candidate_token_ids": [10]}, 422), ({"max_tokens": 4}, 422),
])
def test_invalid_or_unavailable_requests_do_not_run_inference(change, status):
    async def render(*args):
        pytest.fail("must reject before rendering")
    with TestClient(make_app(render=render)) as client:
        response = client.post("/v1/decision_scores", json=body(**change))
        assert response.status_code == status, response.text


def test_prefill_provider_is_used_and_must_honor_execution_contract():
    async def render(*args):
        return {"prompt_token_ids": [1, 2]}
    async def provider(chat, prompt, ids, **kwargs):
        return score_response(ids, [2.0, 4.0], 2,
                              score_type="raw_logits", execution="prefill_logits")
    with TestClient(make_app(render=render, prefill_score=provider)) as client:
        response = client.post("/v1/decision_scores", json=body(require_prefill=True))
        assert response.status_code == 200, response.text
        assert response.json()["generated_tokens"] == 0
    async def downgrade(chat, prompt, ids, **kwargs):
        return score_response(ids, [-2.0, -0.2], 2,
                              score_type="raw_logprobs", execution="engine_logprobs")
    with TestClient(make_app(render=render, prefill_score=downgrade)) as client:
        assert client.post("/v1/decision_scores", json=body(require_prefill=True)).status_code == 502


def test_duplicate_route_installation_is_rejected():
    with pytest.raises(RuntimeError, match="already"):
        install_routes(make_app())


@pytest.mark.parametrize("native", [False, True])
def test_cache_bypass_reaches_selected_provider(native):
    async def render(*args):
        return {"prompt_token_ids": [1]}
    async def provider(chat, prompt, ids, **kwargs):
        assert kwargs["skip_reading_prefix_cache"] is True
        return score_response(ids, [-2., -1.], 1,
                              score_type="raw_logits" if native else "raw_logprobs",
                              execution="prefill_logits" if native else "engine_logprobs")
    with TestClient(make_app(render=render, score=provider, prefill_score=provider)) as client:
        response = client.post("/v1/decision_scores", json=body(
            require_prefill=native, skip_reading_prefix_cache=True))
        assert response.status_code == 200, response.text
        assert client.post("/v1/decision_scores", json=body(
            skip_reading_prefix_cache="true")).status_code == 422


def test_bridge_requests_remain_bridge_when_native_provider_is_installed():
    async def render(*args):
        return {"prompt_token_ids": [1]}
    async def bridge(chat, prompt, ids, **kwargs):
        return score_response(ids, [-2.0, -0.2], 1,
                              score_type="raw_logprobs", execution="engine_logprobs")
    async def native(*args, **kwargs):
        pytest.fail("require_prefill=false selects the bridge explicitly")
    with TestClient(make_app(render=render, score=bridge, prefill_score=native)) as client:
        response = client.post("/v1/decision_scores", json=body())
        assert response.status_code == 200
        assert response.json()["execution"] == "engine_logprobs"
        assert response.json()["generated_tokens"] == 1


@pytest.fixture
def sampling_params(monkeypatch):
    module = ModuleType("vllm.sampling_params")
    module.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", module)


@pytest.mark.parametrize("bad", [None, "missing", "misaligned", "nonfinite", "empty_row"])
def test_internal_engine_readout_bypasses_formatter_and_rejects_missing_scores(sampling_params, bad):
    class Engine:
        def generate(self, prompt, params, request_id):
            assert params.detokenize is False
            assert params.seed == 0
            assert params.logprob_token_ids == [10, 20]
            async def results():
                row = {10: SimpleNamespace(logprob=-2.0), 20: SimpleNamespace(logprob=-0.2)}
                if bad == "missing":
                    row.pop(20)
                if bad == "nonfinite":
                    row[20].logprob = float("nan")
                if bad == "empty_row":
                    row = None
                yield SimpleNamespace(prompt_token_ids=[1, 2], outputs=[SimpleNamespace(
                    token_ids=[5, 6] if bad == "misaligned" else [5],
                    logprobs=[row], finish_reason="length")])
            return results()
        async def abort(self, request_id):
            pytest.fail("completed request must not abort")
    chat = SimpleNamespace(engine_client=Engine(), model_config=SimpleNamespace(
        logprobs_mode="raw_logprobs", max_logprobs=20))
    async def connected():
        return False
    async def run():
        return await engine_scores(chat, {"prompt_token_ids": [1, 2]}, [10, 20],
                                   request_id="test", disconnected=connected)
    if bad:
        with pytest.raises(HTTPException) as exc:
            asyncio.run(run())
        assert exc.value.status_code == 502
        if bad == "missing":
            assert exc.value.detail["missing_token_ids"] == [20]
    else:
        response = asyncio.run(run())
        assert response["execution"] == "engine_logprobs"
        assert response["generated_tokens"] == 1
        assert response["probabilities"][1] > 0.8


def test_disconnection_aborts_the_exact_engine_request(sampling_params):
    aborted = []
    closed = []
    class Engine:
        def generate(self, *args):
            async def results():
                try:
                    yield object()
                finally:
                    closed.append(True)
            return results()
        async def abort(self, request_id):
            aborted.append(request_id)
    chat = SimpleNamespace(engine_client=Engine(), model_config=SimpleNamespace(
        logprobs_mode="raw_logprobs", max_logprobs=20))
    async def disconnected():
        return True
    async def run():
        await engine_scores(chat, {}, [10, 20], request_id="only-this-request", disconnected=disconnected)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
    assert aborted == ["only-this-request"]
    assert closed == [True]
