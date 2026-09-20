"""Offline tests: the real tokenizer, a stubbed vLLM read."""
import json as _json
import math

import msgspec
import pytest
from fastapi.testclient import TestClient
from transformers import AutoTokenizer

from openjev.api import create_app
from openjev.config import Settings
from openjev.engine import Engine, confidence, to_answer

TOKENIZER = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
EXAMPLE = {  # Jev's quickstart request, verbatim
    "state": "Hi, I've been trying to connect my Stripe account but keep getting a 403 error.",
    "model": "jev-latest",
    "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle this",
                       "criteria": {"billing": "Payment or subscription issues",
                                    "technical": "Bugs or integration problems",
                                    "sales": "Pricing or account questions"}},
        "frustration": {"type": "score", "instructions": "How frustrated the customer appears",
                        "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]},
        "is_urgent": {"type": "noul", "instructions": "The message conveys urgency or time-sensitivity"},
    },
}


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.fixture
def client(tok, monkeypatch):
    reads = []

    calls = []

    async def fake_read(self, template, slots, sys_text, content, seed, steps=1, prefix=None):
        reads.append(sys_text)
        calls.append({"content": content, "seed": seed, "steps": steps, "prefix": prefix, "template": template})
        # first label 70%, the rest share 30%; odd seeds flip the first two
        # labels of a noul so averaging over samples shows up
        out = []
        for s in slots:
            n = len(s["label_ids"])
            probs = [0.7] + [0.3 / (n - 1)] * (n - 1)
            if n == 2 and seed % 2:
                probs = [0.3, 0.7]
            out.append({"probs": probs, "entropy": 0.05})
        return out, 123

    async def fake_think(self, sys_text, state_text, budget):
        return self.chat_prompt_ids(sys_text, state_text, thinking=True) + self.thought_open + [7, 8, 9] + self.thought_close, 3, 100

    monkeypatch.setattr(Engine, "one_read", fake_read)
    monkeypatch.setattr(Engine, "think", fake_think)
    with TestClient(create_app(Settings(), tokenizer=tok)) as c:
        c.reads = reads
        c.calls = calls
        yield c


def test_labels_are_single_tokens(tok):
    eng = Engine(Settings(), tok)
    assert len(eng.choice_labels) == 128
    assert eng.choice_labels[:3] == ["A", "B", "C"]


def test_many_questions_chunk(tok):
    eng = Engine(Settings(canvas=32), tok)
    schema = eng.build_schema({f"k{i}": {"type": "noul", "instructions": "x"} for i in range(30)})
    groups = eng.groups(schema["questions"], schema["format"])
    assert len(groups) > 1
    for g in groups:
        eng.resolve_template(g, schema["format"])  # every chunk fits and resolves


def test_quickstart_decodes_with_typesafe_sdk(client):
    from typesafe_sdk._core.response_types import _RESPONSE_DECODER

    r = client.post("/v1/systemone", json=EXAMPLE)
    assert r.status_code == 200, r.text
    assert r.headers["x-typesafe-request-id"].startswith("req_")
    body = _RESPONSE_DECODER.decode(r.content)  # the SDK's strict decoder
    assert body.answers["department"].choice == "billing"
    assert set(body.answers["department"].probabilities) == {"billing", "technical", "sales"}
    assert body.answers["frustration"].legend[0] == "Calm, just stating facts"
    assert math.isclose(body.answers["frustration"].score, 0.15 * 1 + 0.15 * 2)
    assert math.isclose(body.answers["is_urgent"].noul, 0.7)
    assert set(r.json()["answers"]["is_urgent"]) == {"type", "noul"}
    assert body.usage.input_tokens == 123 and body.usage.output_tokens == 0
    assert "department" not in client.reads[0]  # question ids stay out of the prompt


def test_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["models"][0]["name"] == "openjev-latest"


def test_validation_shapes(client):
    r = client.post("/v1/systemone", json={"model": "jev-latest", "questions": {"a": {"type": "noul"}}})
    assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", "state"]
    r = client.post("/v1/systemone", json={"state": "x", "model": "jev-latest", "questions": {}})
    assert r.status_code == 422
    r = client.post("/v1/systemone", json={"state": "x", "model": "jev-latest",
                                           "questions": {"a": {"type": "score", "criteria": [f"l{i}" for i in range(11)]}}})
    assert r.status_code == 400 and r.json()["detail"] == "Too many score levels. Must have at most 10 levels."
    r = client.post("/v1/systemone", json={"state": "x", "model": "gpt-4", "questions": {"a": {"type": "noul"}}})
    assert r.status_code == 400 and r.json()["detail"] == {"error_type": "api_usage_error", "message": "Unknown model: gpt-4"}


def test_auth(tok, monkeypatch):
    with TestClient(create_app(Settings(api_key="sk-test"), tokenizer=tok)) as c:
        r = c.get("/v1/models")
        assert r.status_code == 403 and r.json()["detail"]["error_type"] == "authentication_error"
        assert c.get("/v1/models", headers={"authorization": "Bearer nope"}).status_code == 401
        assert c.get("/v1/models", headers={"authorization": "Bearer sk-test"}).status_code == 200
    with TestClient(create_app(Settings(origin_secret="s3"), tokenizer=tok)) as c:
        assert c.get("/v1/models").status_code == 403
        assert c.get("/v1/models", headers={"x-origin-secret": "s3"}).status_code == 200
        assert c.get("/health").status_code == 200


def test_confidence():
    assert confidence([1.0, 0.0, 0.0]) == 1.0
    assert confidence([0.5, 0.5]) == pytest.approx(0.0)
    # matches Jev's documented example (0.84/0.159/0.001 -> ~0.596)
    assert confidence([0.84, 0.159, 0.001]) == pytest.approx(0.596, abs=0.01)


def test_answer_shapes_roundtrip():
    from typesafe_sdk._core.response_types import ScoreAnswer

    q = {"type": "score", "choices": [("0", "a"), ("1", "b")], "legend": ["a", {"k": 1}]}
    a = to_answer(q, [0.25, 0.75])
    decoded = msgspec.json.decode(msgspec.json.encode(a), type=ScoreAnswer)
    assert decoded.score == 0.75 and decoded.legend[1] == {"k": 1}


def test_indexed_format_with_mixed_types(tok):
    """Past ten questions answers are "q1yes q2A q3 4 ..."; every label type must keep one slot."""
    eng = Engine(Settings(), tok)
    qs = {}
    for i in range(12):
        if i % 3 == 0:
            qs[f"n{i}"] = {"type": "noul"}
        elif i % 3 == 1:
            qs[f"s{i}"] = {"type": "score", "criteria": [f"level {k}" for k in range(10)]}
        else:
            qs[f"c{i}"] = {"type": "choice", "criteria": {f"opt{k}": None for k in range(40)}}
    schema = eng.build_schema(qs)
    assert schema["format"] == "indexed"
    for g in eng.groups(schema["questions"], schema["format"]):
        eng.resolve_template(g, schema["format"])


PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="


def test_images_go_ahead_of_the_state(client):
    body = dict(EXAMPLE, images=[f"data:image/png;base64,{PNG}", {"content_type": "image/jpeg", "base64": PNG}])
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    content = client.calls[0]["content"]
    assert [p["type"] for p in content] == ["image_url", "image_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[2]["text"] == EXAMPLE["state"]


def test_image_validation(client):
    for images, needle in [(["https://example.com/a.png"], "data:image"),
                           ([{"content_type": "image/bmp", "base64": PNG}], "not supported"),
                           ([{"content_type": "image/png", "base64": "not base64!"}], "base64"),
                           ([f"data:image/png;base64,{PNG}"] * 9, "at most 8")]:
        r = client.post("/v1/systemone", json=dict(EXAMPLE, images=images))
        assert r.status_code == 400 and needle in r.json()["detail"], (images, r.text)


def test_options_default_to_jevs_behaviour(client):
    plain = client.post("/v1/systemone", json=EXAMPLE).json()
    first = list(client.calls)
    client.calls.clear()
    explicit = client.post("/v1/systemone", json=dict(EXAMPLE, steps=1, think=0, sequential=False)).json()
    assert plain == explicit
    assert [c["steps"] for c in first] == [1] and first[0]["prefix"] is None
    assert [(c["seed"], c["steps"]) for c in client.calls] == [(c["seed"], c["steps"]) for c in first]


def test_steps_and_samples(client):
    r = client.post("/v1/systemone", json=dict(EXAMPLE, steps=4, samples=4))
    assert r.status_code == 200, r.text
    assert len(client.calls) == 4 and all(c["steps"] == 4 for c in client.calls)
    # two of the four seeds flip the noul, so the mean is 0.5; every read is billed
    assert math.isclose(r.json()["answers"]["is_urgent"]["noul"], 0.5)
    assert r.json()["usage"]["input_tokens"] == 4 * 123
    assert client.post("/v1/systemone", json=dict(EXAMPLE, samples=33)).status_code == 422
    assert client.post("/v1/systemone", json=dict(EXAMPLE, steps=9)).status_code == 422


def test_think_continues_after_the_thought(client):
    r = client.post("/v1/systemone", json=dict(EXAMPLE, think=256))
    assert r.status_code == 200, r.text
    call = client.calls[0]
    close = client.app.state.engine.thought_close
    assert call["prefix"][-len(close) - 3:] == [7, 8, 9] + close  # the read continues after the closed thought
    assert call["template"][: len(client.app.state.engine.scaffold)] != client.app.state.engine.scaffold
    assert r.json()["usage"]["output_tokens"] == 3  # the thought's tokens
    # the input is billed for the thought pass and again for the read
    assert r.json()["usage"]["input_tokens"] == 100 + 123


def test_sequential_prefills_earlier_answers(tok, client):
    qs = {f"k{i}": {"type": "noul", "instructions": f"question {i}"} for i in range(24)}
    r = client.post("/v1/systemone", json={"state": "x", "model": "jev-latest", "questions": qs, "sequential": True})
    assert r.status_code == 200, r.text
    assert len(client.calls) > 1
    assert client.calls[0]["prefix"] is None
    later = client.calls[1]["prefix"]
    assert later is not None and len(later) > len(client.calls[0]["template"])


def test_think_and_sequential_need_text(client):
    for opt in ({"think": 64}, {"sequential": True}):
        r = client.post("/v1/systemone", json=dict(EXAMPLE, images=[f"data:image/png;base64,{PNG}"], **opt))
        assert r.status_code == 400 and "text state" in r.json()["detail"]
        assert r.json()["detail"].startswith(next(iter(opt)))  # which option was at fault


def chat_client(tok, handler, **settings):
    import httpx

    app = create_app(Settings(**settings), tokenizer=tok)
    c = TestClient(app)
    c.__enter__()
    app.state.generator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://vllm")
    return c


def test_chat_normalizes_jev_ultrafast_request(tok):
    sent = []

    def handler(request):
        sent.append(_json.loads(request.content))
        return httpx_response({"id": "x", "model": "dgemma", "choices": [{"index": 0, "message": {
            "role": "assistant", "content": 'Sure!\n```json\n{"text": "Zurich"}\n```'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52}})

    c = chat_client(tok, handler)
    body = {"model": "diffusiongemma-26b", "max_tokens": 1024, "response_format": {"type": "json_object"},
            "reasoning": {"enabled": False}, "temperature": 0.2, "seed": 1,
            "messages": [{"role": "system", "content": "Return {\"text\": ...}"}, {"role": "user", "content": "{}"}]}
    r = c.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    up = sent[0]
    assert up["model"] == "dgemma" and "response_format" not in up and "temperature" not in up
    assert "reasoning" not in up and "seed" not in up and up["max_tokens"] == 1024
    assert "JSON object" in up["messages"][0]["content"]
    assert up["chat_template_kwargs"] == {"enable_thinking": False}
    out = r.json()
    assert out["model"] == "diffusiongemma-26b"
    assert _json.loads(out["choices"][0]["message"]["content"]) == {"text": "Zurich"}
    assert c.post("/v1/chat/completions", json=dict(body, model="gpt-4")).status_code == 404


def test_single_option_choice_is_answered_without_a_read(client):
    """Jev sets no minimum on a choice's options; one option has one answer."""
    body = dict(EXAMPLE, questions={"only": {"type": "choice", "instructions": "Which team",
                                             "criteria": {"billing": "anything at all"}}})
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    a = r.json()["answers"]["only"]
    assert a == {"type": "choice", "choice": "billing", "probabilities": {"billing": 1.0}, "confidence": 1.0}
    assert client.reads == []  # the model was never asked
    assert r.json()["usage"]["input_tokens"] == 0


def test_single_option_choice_alongside_read_questions(client):
    qs = dict(EXAMPLE["questions"], only={"type": "choice", "instructions": "Which team", "criteria": {"billing": None}})
    r = client.post("/v1/systemone", json=dict(EXAMPLE, questions=qs))
    assert r.status_code == 200, r.text
    answers = r.json()["answers"]
    assert list(answers) == list(qs)  # answered in the order asked
    assert answers["only"]["choice"] == "billing"
    assert answers["department"]["type"] == "choice" and client.reads  # the rest still went to the model


def test_single_level_score_is_answered_without_a_read(client):
    """Jev answers a one-level score with 0 at probability 1; it used to be a 422 here."""
    r = client.post("/v1/systemone", json=dict(EXAMPLE, questions={"only": {"type": "score", "instructions": "How bad", "criteria": ["fine"]}}))
    assert r.status_code == 200, r.text
    assert r.json()["answers"]["only"] == {"type": "score", "score": 0.0, "legend": {"0": "fine"},
                                           "probabilities": {"0": 1.0}, "confidence": 1.0}
    assert client.reads == []


def test_choice_needs_an_option(client):
    r = client.post("/v1/systemone", json=dict(EXAMPLE, questions={"q": {"type": "choice", "instructions": "x", "criteria": {}}}))
    assert r.status_code == 400 and r.json()["detail"] == "Choice question must have at least one choice: q"


def test_deeply_nested_body_is_rejected_not_crashed(client):
    """A thousand levels of nesting used to blow the stack while the 422 was encoded."""
    deep = {"a": None}
    for _ in range(1000):
        deep = {"a": deep}
    r = client.post("/v1/systemone", json=dict(EXAMPLE, questions={"q": deep}))
    assert r.status_code == 422
    detail = r.json()["detail"][0]
    assert detail["loc"] == ["body", "questions", "q"]
    assert "..." in _json.dumps(detail["input"])  # the offending value is trimmed, not echoed whole


def test_invalid_requests_are_logged_without_their_body(client, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="openjev"):
        client.post("/v1/systemone", json=dict(EXAMPLE, questions={"q": {"type": "nope", "instructions": "x"}}))
        client.post("/v1/systemone", json=dict(EXAMPLE, questions={"q": {"type": "score", "instructions": "x", "criteria": [f"l{i}" for i in range(11)]}}))
    logged = [r.getMessage() for r in caplog.records]
    assert any("questions.q" in m and "tag" in m for m in logged)
    assert any("at most 10 levels" in m for m in logged)
    assert not any("Stripe" in m for m in logged)  # never the state


def test_chat_passes_upstream_errors(tok):
    c = chat_client(tok, lambda request: httpx_response({"error": {"message": "prompt too long"}}, 400))
    r = c.post("/v1/chat/completions", json={"model": "diffusiongemma-26b", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400 and r.json()["error"]["message"] == "prompt too long"


def httpx_response(body, status=200):
    import httpx

    return httpx.Response(status, json=body)
