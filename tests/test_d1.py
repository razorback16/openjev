"""Offline tests for the d1-2B backend: a stub in place of the model, the real API and batching."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from openjev import encoders
from openjev.api import create_app
from openjev.config import Settings
from openjev.encoders import D1Engine
from openjev.engine import SchemaError

NOUL = {"type": "noul", "instructions": "Does the passage answer the question?"}


class FakeD1:
    """system_one_batch in d1's shapes: every noul 0.8, every choice its first option.
    A question whose instructions are "bad" is refused with a ValueError, as d1 refuses one."""

    def __init__(self):
        self.calls = []

    def system_one_batch(self, requests):
        self.calls.append(list(requests))
        out = []
        for state, questions in requests:
            if any(q.get("instructions") == "bad" for q in questions.values()):
                raise ValueError("question 'q': instructions are required")
            answers = {}
            for name, q in reversed(list(questions.items())):  # d1's order need not be the caller's
                if q["type"] == "noul":
                    answers[name] = {"type": "noul", "noul": 0.8}
                else:
                    first = next(iter(q["criteria"]))
                    answers[name] = {"type": "choice", "choice": first, "confidence": 0.9,
                                     "probabilities": {k: (0.9 if k == first else 0.1 / (len(q["criteria"]) - 1))
                                                       for k in q["criteria"]}}
            out.append({"answers": answers, "usage": {"input_tokens": 10 * len(questions), "output_tokens": 0}})
        return out


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    fake = FakeD1()
    monkeypatch.setattr(D1Engine, "load", lambda self: setattr(self, "model", fake))
    return fake


def engine(**kw):
    return D1Engine(Settings(backend="d1", warmup=False, **kw))


def decide_all(eng, requests):
    async def go():
        return await asyncio.gather(*(eng.decide(q, s, 0) for s, q in requests), return_exceptions=True)
    return asyncio.run(go())


@pytest.fixture
def client():
    with TestClient(create_app(Settings(backend="d1", warmup=False))) as c:
        yield c


def test_answers_pass_through_in_the_callers_order(client):
    questions = {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "outage": None}},
                 "urgent": NOUL}
    r = client.post("/v1/systemone", json={"model": "d1-2b", "state": "charged twice", "questions": questions})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "d1-2b"
    assert list(body["answers"]) == ["team", "urgent"]
    assert body["answers"]["urgent"] == {"type": "noul", "noul": 0.8}
    assert body["answers"]["team"]["choice"] == "billing"
    assert body["usage"] == {"input_tokens": 20, "output_tokens": 0}
    assert r.headers["server-timing"].startswith("model;dur=")


def test_jev_model_names_are_accepted_and_d1_is_listed(client):
    r = client.post("/v1/systemone", json={"model": "jev-latest", "state": "x", "questions": {"q": NOUL}})
    assert r.status_code == 200, r.text
    assert [m["name"] for m in client.get("/v1/models").json()["models"]] == ["d1-2b"]


def test_a_refused_question_is_a_400_with_d1s_reason(client):
    r = client.post("/v1/systemone", json={"model": "d1-2b", "state": "x",
                                           "questions": {"q": {"type": "noul", "instructions": "bad"}}})
    assert r.status_code == 400
    assert "instructions are required" in r.json()["detail"]


def test_images_are_refused(client):
    r = client.post("/v1/systemone", json={"model": "d1-2b", "state": "x", "questions": {"q": NOUL},
                                           "images": ["data:image/png;base64,iVBORw0KGgo="]})
    assert r.status_code in (400, 422)


def test_requests_waiting_together_are_read_in_one_batch(fake_model):
    eng = engine(d1_batch=32)
    out = decide_all(eng, [(f"passage {i}", {"q": NOUL}) for i in range(30)])
    assert [len(c) for c in fake_model.calls] == [30]
    assert all(answers["q"]["noul"] == 0.8 for answers, _, _ in out)
    assert [s for s, _ in fake_model.calls[0]] == [f"passage {i}" for i in range(30)]


def test_a_batch_holds_at_most_d1_batch_requests(fake_model):
    eng = engine(d1_batch=4)
    decide_all(eng, [(f"p{i}", {"q": NOUL}) for i in range(10)])
    assert [len(c) for c in fake_model.calls] == [4, 4, 2]


def test_the_default_batch_is_8(fake_model):
    decide_all(engine(), [(f"p{i}", {"q": NOUL}) for i in range(30)])
    assert [len(c) for c in fake_model.calls] == [8, 8, 8, 6]


def test_a_batch_holds_at_most_d1_batch_chars_but_always_one(fake_model):
    eng = engine(d1_batch_chars=25)
    decide_all(eng, [("a" * 10, {"q": NOUL}), ("b" * 10, {"q": NOUL}), ("c" * 100, {"q": NOUL}), ("d", {"q": NOUL})])
    assert [[s[0] for s, _ in c] for c in fake_model.calls] == [["a", "b"], ["c"], ["d"]]


def test_a_refused_request_fails_alone(fake_model):
    eng = engine()
    out = decide_all(eng, [("ok 1", {"q": NOUL}), ("x", {"q": {"type": "noul", "instructions": "bad"}}),
                           ("ok 2", {"q": NOUL})])
    assert isinstance(out[1], SchemaError)
    assert out[0][0]["q"]["noul"] == 0.8 and out[2][0]["q"]["noul"] == 0.8
    # the batch, then each request alone
    assert [len(c) for c in fake_model.calls] == [3, 1, 1, 1]


def test_a_model_failure_reaches_its_request(fake_model, monkeypatch):
    eng = engine()

    def broken(requests):
        raise RuntimeError("MPS backend out of memory")
    monkeypatch.setattr(fake_model, "system_one_batch", broken)
    out = decide_all(eng, [("x", {"q": NOUL})])
    assert isinstance(out[0], RuntimeError)
    # and the engine still serves once the model recovers
    monkeypatch.undo()
    assert decide_all(eng, [("y", {"q": NOUL})])[0][0]["q"]["noul"] == 0.8


def test_the_choice_limit_is_checked_before_any_read(fake_model):
    eng = engine()
    many = {"q": {"type": "choice", "instructions": "x", "criteria": {str(i): None for i in range(256)}}}
    out = decide_all(eng, [("x", many)])
    assert isinstance(out[0], SchemaError)
    assert fake_model.calls == []


def test_warmup_reads_once_before_serving(fake_model):
    D1Engine(Settings(backend="d1", warmup=True))
    assert len(fake_model.calls) == 1
    (state, questions), = fake_model.calls[0]
    assert all(None not in q.values() for q in questions.values())
