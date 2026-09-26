"""The MLX backend's wiring, with a stub in place of the model so it runs on any
machine. tests/test_mlx_model.py runs the same path against real weights."""
import json as _json
import time
import math
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from transformers import AutoTokenizer

from openjev import mlx_backend
from openjev.api import create_app
from openjev.config import Settings
from openjev.engine import PAD, TURN_CLOSE, Engine, SchemaError

from test_api import EXAMPLE, PNG, TOKENIZER

# The stub stands in for MlxRuntime, so it must speak the same protocol: a read
# takes (prompt, canvas, slots, max_tokens) and returns (logprobs, prompt
# tokens). A prompt is either token ids or an ImagePrompt, which only the real
# processor can expand; here it is counted and keyed the way the runtime does.
IMAGE_TOKENS = 256


def stub_images(prompt):
    return list(getattr(prompt, "images", ()) or ())


def stub_key(prompt):
    key = getattr(prompt, "key", None)
    return key if key is not None else tuple(prompt)


def stub_tokens(prompt):
    if isinstance(prompt, (list, tuple)):
        return len(prompt)
    text = prompt.sys_text + prompt.state_text
    return len(text.split()) + IMAGE_TOKENS * len(prompt.images)


class StubRuntime:
    """Every slot reads 0.99 on its first label and splits the rest: sure enough
    that the engine does not re-read."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.reads = []
        self.prefills = set()
        self.passes = 0
        self.generations = []

    def read(self, prompt, canvas, slots, max_tokens=None, steps=1, **kw):
        n = stub_tokens(prompt)
        self.prefills.add(stub_key(prompt))
        self.passes += steps  # the real runtime runs one decoder pass per step
        self.reads.append({"prompt": prompt, "canvas": canvas, "slots": slots, "tokens": n,
                           "key": stub_key(prompt), "images": stub_images(prompt), "steps": steps, **kw})
        if max_tokens is not None and n > max_tokens:
            raise SchemaError(f"the request is {n} tokens; the limit is {max_tokens}")
        out = []
        for s in slots:
            ids = s["label_ids"]
            out.append({i: math.log(0.99 if i == ids[0] else 0.01 / (len(ids) - 1)) for i in ids})
        return out, n

    # Generation stands in for the denoise loop: a fixed reply, one token a word.
    # The real runtime emits a last, token-less segment when it stops, so the stub
    # does too -- TAIL is text nobody is billed for.
    REPLY = ['{"city"', ': "', "Zurich", '"']
    TAIL = "}"

    def generate(self, prompt, max_tokens, stop_ids, emit, skip_special=None):
        self.generations.append({"prompt": list(prompt), "max_tokens": max_tokens,
                                 "stop_ids": list(stop_ids), "skip_special": list(skip_special or ())})
        ids, finish = [], "stop"
        for i, text in enumerate(self.REPLY):
            if len(ids) >= max_tokens:
                finish = "length"
                break
            token = 1000 + i
            if token in stop_ids:
                break
            ids.append(token)
            if not emit(text, token):
                return ids, len(prompt), "cancelled"
        emit(self.TAIL, None)
        return ids, len(prompt), finish

    def close(self):
        self.pool.shutdown()


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.fixture
def client(tok, monkeypatch):
    monkeypatch.setattr(mlx_backend, "MlxRuntime", StubRuntime)
    with TestClient(create_app(Settings(backend="mlx", mlx_model="/models/dg"), tokenizer=tok)) as c:
        yield c


def test_vllm_stays_the_default(tok):
    assert Settings().backend == "vllm"
    with TestClient(create_app(Settings(), tokenizer=tok)) as c:
        assert type(c.app.state.engine) is Engine
        assert "diffusiongemma-26b" in [m["name"] for m in c.get("/v1/models").json()["models"]]


def test_a_read_goes_to_the_runtime(client):
    r = client.post("/v1/systemone", json=EXAMPLE)
    assert r.status_code == 200, r.text
    engine = client.app.state.engine
    assert engine.runtime.model_path == "/models/dg"
    (read,) = engine.runtime.reads  # confident answers: one read, no re-reads
    a = r.json()["answers"]
    assert a["department"]["choice"] == "billing" and math.isclose(a["department"]["probabilities"]["billing"], 0.99)
    assert math.isclose(a["is_urgent"]["noul"], 0.99)
    assert math.isclose(a["frustration"]["score"], 0.005 * 1 + 0.005 * 2)
    # the prompt is the chat prompt, and it is what gets billed
    assert read["prompt"] == engine.chat_prompt_ids(client_sys_text(engine), EXAMPLE["state"])
    assert r.json()["usage"] == {"input_tokens": len(read["prompt"]), "output_tokens": 0}
    # the canvas is the template with noise only in the slots, closed and padded to a step
    template, slots = read_template(engine)
    canvas, positions = read["canvas"], {s["pos"] for s in slots}
    assert len(canvas) % engine.s.canvas_step == 0 and canvas[len(template)] == TURN_CLOSE
    assert all(canvas[i] == t for i, t in enumerate(template) if i not in positions)
    assert set(canvas[len(template) + 1:]) <= {PAD}


def client_sys_text(engine):
    schema = engine.build_schema(EXAMPLE["questions"])
    return engine.system_text(schema["questions"], schema["format"])


def read_template(engine):
    schema = engine.build_schema(EXAMPLE["questions"])
    return engine.resolve_template(schema["questions"], schema["format"])


def test_same_request_same_canvas(client):
    client.post("/v1/systemone", json=EXAMPLE)
    client.post("/v1/systemone", json=EXAMPLE)
    first, second = client.app.state.engine.runtime.reads
    assert first["canvas"] == second["canvas"]


def test_samples_and_sequential_work(client):
    r = client.post("/v1/systemone", json=dict(EXAMPLE, samples=3))
    assert r.status_code == 200 and len(client.app.state.engine.runtime.reads) == 3
    qs = {f"k{i}": {"type": "noul", "instructions": f"question {i}"} for i in range(24)}
    r = client.post("/v1/systemone", json={"state": "x", "model": "jev-latest", "questions": qs, "sequential": True})
    assert r.status_code == 200, r.text
    reads = client.app.state.engine.runtime.reads[3:]
    assert len(reads) > 1 and len(reads[1]["prompt"]) > len(reads[0]["prompt"])  # later chunks carry earlier answers


def test_think_reaches_the_runtime(client):
    """A thought is generated, then read: the read's prompt is the thought
    prefix, and the thought's tokens are billed as output."""
    r = client.post("/v1/systemone", json=dict(EXAMPLE, think=64))
    assert r.status_code == 200, r.text
    rt = client.app.state.engine.runtime
    engine = client.app.state.engine
    (gen,) = rt.generations
    assert gen["max_tokens"] == 64 and gen["stop_ids"] == engine.thought_close
    assert gen["prompt"][-len(engine.thought_open):] == engine.thought_open
    (read,) = rt.reads
    # the read continues the thought: same prefix, then the closed thought
    assert read["prompt"][: len(gen["prompt"])] == gen["prompt"]
    assert read["prompt"][-len(engine.thought_close):] == engine.thought_close
    usage = r.json()["usage"]
    assert usage["output_tokens"] == len(StubRuntime.REPLY)
    # both passes are billed: the thought read the input, the read read it again plus the thought
    assert usage["input_tokens"] == len(gen["prompt"]) + len(read["prompt"])


def test_think_works_with_sequential(client):
    """think + sequential must share one thought: Engine._sequential bills it on
    the first chunk only, and later chunks continue the same prefix."""
    qs = {f"k{i}": {"type": "noul", "instructions": f"question {i}"} for i in range(24)}
    r = client.post("/v1/systemone", json={"state": "x", "model": "jev-latest", "questions": qs,
                                           "sequential": True, "think": 32})
    assert r.status_code == 200, r.text
    rt = client.app.state.engine.runtime
    assert len(rt.generations) == 1, "the thought is written once, not once a chunk"
    assert len(rt.reads) > 1
    assert r.json()["usage"]["output_tokens"] == len(StubRuntime.REPLY)


def test_steps_reach_the_runtime(client):
    """N steps are N decoder passes over ONE prefill: the prompt does not change
    between steps, so more steps must cost GPU time and no extra prompt tokens."""
    rt = client.app.state.engine.runtime
    one = client.post("/v1/systemone", json=dict(EXAMPLE, steps=1))
    four = client.post("/v1/systemone", json=dict(EXAMPLE, steps=4))
    assert one.status_code == 200 and four.status_code == 200, four.text
    assert [r["steps"] for r in rt.reads] == [1, 4]
    assert rt.passes == 5 and len(rt.prefills) == 1
    assert one.json()["usage"]["input_tokens"] == four.json()["usage"]["input_tokens"]


def test_one_step_is_unchanged(client):
    """steps=1 must be the single pass it always was, byte for byte, whether the
    caller asked for it or left it to the default."""
    rt = client.app.state.engine.runtime
    default = client.post("/v1/systemone", json=EXAMPLE)
    explicit = client.post("/v1/systemone", json=dict(EXAMPLE, steps=1))
    assert default.json() == explicit.json()
    first, second = rt.reads
    assert first == second and first["steps"] == 1
    assert rt.passes == 2  # one pass each, no extra work for the default


# a second 1x1 PNG, a different colour, so its bytes differ from PNG's
PNG2 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQ"
        "AAAABJRU5ErkJggg==")


def url(png):
    return f"data:image/png;base64,{png}"


def test_an_image_read_goes_to_the_runtime(client):
    """Images are answered on MLX, not refused, and the runtime is handed them."""
    r = client.post("/v1/systemone", json=dict(EXAMPLE, images=[url(PNG)]))
    assert r.status_code == 200, r.text
    (read,) = client.app.state.engine.runtime.reads
    assert read["images"], "the runtime was given no images"
    assert len(read["images"]) == 1
    assert r.json()["answers"]["department"]["choice"] == "billing"


def test_image_usage_counts_the_expanded_prompt(client):
    """README promises input_tokens covers the image's expanded tokens, so the
    runtime's own count is what gets billed, not the text prompt's length."""
    r = client.post("/v1/systemone", json=dict(EXAMPLE, images=[url(PNG)]))
    assert r.status_code == 200, r.text
    (read,) = client.app.state.engine.runtime.reads
    assert r.json()["usage"]["input_tokens"] == read["tokens"] >= IMAGE_TOKENS


def test_images_do_not_share_a_prefill_entry(client):
    """Two images on one prompt, and the same prompt with no image, must each
    key the prefill cache differently or one read would answer for another."""
    for images in ([url(PNG)], [url(PNG2)], None):
        body = dict(EXAMPLE, images=images) if images else dict(EXAMPLE)
        assert client.post("/v1/systemone", json=body).status_code == 200
    keys = [r["key"] for r in client.app.state.engine.runtime.reads]
    assert len(keys) == 3 and len(set(keys)) == 3, keys


def test_same_image_request_same_canvas(client):
    body = dict(EXAMPLE, images=[url(PNG)])
    client.post("/v1/systemone", json=body)
    client.post("/v1/systemone", json=body)
    first, second = client.app.state.engine.runtime.reads
    assert first["canvas"] == second["canvas"]
    assert first["key"] == second["key"]  # so the second read reuses the prefill


def test_images_with_think_or_sequential_are_still_refused(client):
    for opt in ({"think": 64}, {"sequential": True}):
        r = client.post("/v1/systemone", json=dict(EXAMPLE, images=[url(PNG)], **opt))
        assert r.status_code == 400, r.text
        assert r.json()["detail"].startswith(next(iter(opt)))


def test_long_prompts_are_refused(tok, monkeypatch):
    monkeypatch.setattr(mlx_backend, "MlxRuntime", StubRuntime)
    with TestClient(create_app(Settings(backend="mlx", mlx_max_prompt=200), tokenizer=tok)) as c:
        assert c.post("/v1/systemone", json=EXAMPLE).status_code == 200
        r = c.post("/v1/systemone", json=dict(EXAMPLE, state="word " * 400))
        assert r.status_code == 400 and "limit is 200" in r.json()["detail"]


CHAT = {"model": "diffusiongemma-26b", "messages": [{"role": "user", "content": "Which city?"}]}


def test_generation_model_is_listed(client):
    assert "diffusiongemma-26b" in [m["name"] for m in client.get("/v1/models").json()["models"]]


def test_chat_completion_on_mlx(client):
    r = client.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["model"] == "diffusiongemma-26b" and d["object"] == "chat.completion"
    assert d["id"].startswith("chatcmpl-")
    (choice,) = d["choices"]
    assert choice["message"] == {"role": "assistant", "content": "".join(StubRuntime.REPLY) + StubRuntime.TAIL}
    assert choice["finish_reason"] == "stop" and choice["index"] == 0
    (gen,) = client.app.state.engine.runtime.generations
    assert d["usage"] == {"prompt_tokens": len(gen["prompt"]), "completion_tokens": len(StubRuntime.REPLY),
                          "total_tokens": len(gen["prompt"]) + len(StubRuntime.REPLY)}


def test_chat_json_mode_and_max_tokens(client):
    """normalize() must behave as it does on vLLM: JSON mode adds the
    instruction and extracts the object, and max_tokens is clamped."""
    r = client.post("/v1/chat/completions", json=dict(CHAT, response_format={"type": "json_object"},
                                                      temperature=0.7, seed=3, max_tokens=99999))
    assert r.status_code == 200, r.text
    assert _json.loads(r.json()["choices"][0]["message"]["content"]) == {"city": "Zurich"}
    (gen,) = client.app.state.engine.runtime.generations
    assert gen["max_tokens"] == Settings().gen_max_tokens  # clamped, not the 99999 asked for
    # the JSON instruction went into the prompt the runtime was handed
    prompt = client.app.state.engine.tok.decode(gen["prompt"])
    assert "JSON object" in prompt


def test_chat_stream_on_mlx(client):
    body = dict(CHAT, stream=True, stream_options={"include_usage": True})
    with client.stream("POST", "/v1/chat/completions", json=body) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = [_json.loads(line[6:]) for line in r.iter_lines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in events if e["choices"])
    assert text == "".join(StubRuntime.REPLY) + StubRuntime.TAIL
    assert [e for e in events if e["choices"] and e["choices"][0]["finish_reason"] == "stop"]
    (last,) = [e for e in events if e.get("usage")]
    assert last["usage"]["completion_tokens"] == len(StubRuntime.REPLY)
    assert all(e["model"] == "diffusiongemma-26b" for e in events)


def test_chat_capacity_is_refused(client, monkeypatch):
    monkeypatch.setattr(client.app.state.generator, "running",
                        Settings().gen_max_inflight + Settings().gen_max_queue)
    r = client.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 529 and r.json()["error"]["type"] == "overloaded_error"
    assert client.app.state.engine.runtime.generations == []


def test_chat_rejects_an_unknown_model(client):
    assert client.post("/v1/chat/completions", json=dict(CHAT, model="gpt-4")).status_code == 404


def test_chat_refuses_an_over_long_prompt(tok, monkeypatch):
    monkeypatch.setattr(mlx_backend, "MlxRuntime", StubRuntime)
    with TestClient(create_app(Settings(backend="mlx", mlx_max_prompt=8), tokenizer=tok)) as c:
        r = c.post("/v1/chat/completions", json=dict(CHAT, messages=[{"role": "user", "content": "word " * 200}]))
        assert r.status_code == 400 and "limit is 8" in r.json()["error"]["message"]


def test_a_disconnected_client_stops_generation(tok, monkeypatch):
    """A client that goes away closes the response body. The runtime thread cannot
    be interrupted, so that must reach it as a False from emit, and the slot must
    come back either way."""
    import asyncio

    monkeypatch.setattr(mlx_backend, "MlxRuntime", StubRuntime)
    app = create_app(Settings(backend="mlx", mlx_model="/models/dg"), tokenizer=tok)
    with TestClient(app):  # entered for the lifespan, which builds the generator
        gen = app.state.generator
        emitted = []

        def generate(prompt, max_tokens, stop_ids, emit, skip_special=None):
            # Keep offering tokens until emit says the reply is no longer wanted.
            # The pause matters: the real runtime denoises a block between chunks,
            # and the route only checks for a disconnect while it is idle.
            for i in range(200):
                emitted.append(i)
                if not emit(f"t{i}", 1000 + i):
                    return [1000], len(prompt), "cancelled"
                time.sleep(0.08)
            return list(range(200)), len(prompt), "stop"

        monkeypatch.setattr(app.state.engine.runtime, "generate", generate)
        upstream, _ = gen.normalize(dict(CHAT, stream=True))

        class Gone:
            """A request whose client has already gone away."""

            async def is_disconnected(self):
                return True

        async def drain():
            r = await gen.stream(upstream, Gone())
            it = r.body_iterator
            seen = []
            try:
                while True:
                    seen.append(await it.__anext__())
            except StopAsyncIteration:
                pass
            await it.aclose()
            # the slot comes back from a task of its own, once the runtime stops
            for _ in range(2000):
                if gen.running == 0:
                    break
                await asyncio.sleep(0.005)
            return seen

        seen = asyncio.run(drain())
        assert seen and all(s.startswith("data: ") for s in seen)
        assert not any("[DONE]" in s for s in seen)
        assert len(emitted) < 200, "generation ran to completion after the client left"
        assert gen.running == 0 and gen.slots._value == Settings().gen_max_inflight


def test_a_slow_reader_cancels_rather_than_loses_chunks(tok, monkeypatch):
    """64 chunks of slack, then the buffer is full. Dropping the next chunk
    silently would corrupt a reply a live client is still reading; the reply
    must end instead, and give the slot back."""
    import asyncio

    monkeypatch.setattr(mlx_backend, "MlxRuntime", StubRuntime)
    app = create_app(Settings(backend="mlx", mlx_model="/models/dg"), tokenizer=tok)
    with TestClient(app):  # entered for the lifespan, which builds the generator
        gen = app.state.generator

        def generate(prompt, max_tokens, stop_ids, emit, skip_special=None):
            for i in range(500):  # fast: no block-sized pause, so the buffer fills
                if not emit(f" {i}", 1000 + i):
                    return [1000], len(prompt), "cancelled"
            return [1000], len(prompt), "stop"

        monkeypatch.setattr(app.state.engine.runtime, "generate", generate)
        upstream, _ = gen.normalize(dict(CHAT, stream=True))

        class Here:
            """A request whose client is still connected, just slow."""

            async def is_disconnected(self):
                return False

        async def drain():
            r = await gen.stream(upstream, Here())
            it = r.body_iterator
            seen = []
            try:
                while True:
                    seen.append(await it.__anext__())
                    if len(seen) == 10:
                        await asyncio.sleep(2)  # let the runtime outrun us and fill the buffer
            except StopAsyncIteration:
                pass
            await it.aclose()
            # the slot comes back from a task of its own, once the runtime stops
            for _ in range(2000):
                if gen.running == 0:
                    break
                await asyncio.sleep(0.005)
            return seen

        seen = asyncio.run(drain())
        assert not any("[DONE]" in s for s in seen), "a cancelled reply must not look complete"
        chunks = [_json.loads(s[6:])["choices"][0]["delta"].get("content", "")
                  for s in seen if s.startswith("data: ")]
        got = [int(t) for t in "".join(chunks).split()]
        assert got and len(got) < 500, "the reply ran to completion despite a reader 64 chunks behind"
        # strictly increasing: nothing was silently dropped while the reply ran on
        # (one queued chunk may be displaced by the end marker as it cancels)
        assert got == sorted(set(got))
        assert gen.running == 0 and gen.slots._value == Settings().gen_max_inflight


# The marker tokens for the thought channel the model sometimes opens on its own.
# The detokenizer NEVER hands these out as their own segments: it buffers them and
# flushes them fused into a later token's text, so anything keyed on token ids alone
# cannot see them. This is what the emit protocol really looks like, taken from a
# leaking run against real weights:
#
#   raw ids : [100, 45518, 107, 101, 34699, 6819, ...]   # markers, then "six", " seven"
#   emitted : [(100, ''), (45518, ''), (107, ''), (101, ''), (34699, ''),
#              (6819, '<|channel>thought\n<channel|>six'), (6589, ' seven'), ...]
LEAKY_EMITS = [(100, ""), (45518, ""), (107, ""), (101, ""), (34699, ""),
               (6819, "<|channel>thought\n<channel|>six"), (6589, " seven"),
               (10155, " eight"), (3595, " nine"), (None, " ten")]
CLEAN_EMITS = [(34699, ""), (6819, "six"), (6589, " seven"),
               (10155, " eight"), (3595, " nine"), (None, " ten")]
MARKER_IDS = [100, 45518, 107, 101]


class ReplayRuntime(StubRuntime):
    """Replays a recorded emit stream, so the chat route meets exactly what the
    real detokenizer produces. A generation is clean when the runtime was asked to
    skip the marker ids, and leaks when it was not -- which is what the real
    generator does with skip_special_token_ids."""

    emits = LEAKY_EMITS

    def generate(self, prompt, max_tokens, stop_ids, emit, skip_special=None):
        self.generations.append({"prompt": list(prompt), "max_tokens": max_tokens,
                                 "stop_ids": list(stop_ids), "skip_special": skip_special})
        skipping = set(skip_special or ()) >= set(MARKER_IDS)
        ids = []
        for token, text in (CLEAN_EMITS if skipping else self.emits):
            if token is None:
                emit(text, None)
                break
            ids.append(token)
            if not emit(text, token):
                return ids, len(prompt), "cancelled"
        return ids, len(prompt), "stop"


def replay_client(tok, monkeypatch):
    monkeypatch.setattr(mlx_backend, "MlxRuntime", ReplayRuntime)
    c = TestClient(create_app(Settings(backend="mlx", mlx_model="/models/dg"), tokenizer=tok))
    c.__enter__()
    return c


def test_the_thought_channel_never_reaches_a_chat_client(tok, monkeypatch):
    """The model opens a thought channel of its own accord on some replies. The
    markers reach the detokenizer fused into a later segment, so the reply must be
    kept clean by asking the generator to skip those tokens, not by matching ids."""
    c = replay_client(tok, monkeypatch)
    r = c.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 200, r.text
    text = r.json()["choices"][0]["message"]["content"]
    assert "channel" not in text, text
    assert text == "six seven eight nine ten"


def test_the_thought_channel_never_reaches_a_streaming_client(tok, monkeypatch):
    c = replay_client(tok, monkeypatch)
    with c.stream("POST", "/v1/chat/completions", json=dict(CHAT, stream=True)) as r:
        assert r.status_code == 200
        events = [_json.loads(line[6:]) for line in r.iter_lines()
                  if line.startswith("data: ") and line.strip() != "data: [DONE]"]
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in events if e["choices"])
    assert "channel" not in text, text
    assert text == "six seven eight nine ten"


def test_chat_skips_the_marker_tokens_and_think_does_not(client):
    """Chat wants the thought gone, so the generator is told to drop the markers.
    think WANTS the thought, so it must not be."""
    engine = client.app.state.engine
    markers = set(engine.thought_open) | set(engine.thought_close)
    assert client.post("/v1/chat/completions", json=CHAT).status_code == 200
    (gen,) = client.app.state.engine.runtime.generations
    assert set(gen["skip_special"]) == markers
    client.app.state.engine.runtime.generations.clear()
    assert client.post("/v1/systemone", json=dict(EXAMPLE, think=32)).status_code == 200
    (think,) = client.app.state.engine.runtime.generations
    assert not think["skip_special"]


class OneTokenRuntime(StubRuntime):
    """A reply of a single token, whose text the detokenizer flushes only at the
    end. The shortest replies exposed a race: the generation task finished while
    its last chunk was still in flight to the event loop, and the stream ended
    empty."""

    def generate(self, prompt, max_tokens, stop_ids, emit, skip_special=None):
        self.generations.append({"prompt": list(prompt), "max_tokens": max_tokens,
                                 "stop_ids": list(stop_ids), "skip_special": list(skip_special or ())})
        emit("", 1000)          # the token arrives with no text of its own
        emit("7", None)         # ... which is flushed with the terminal result
        return [1000], len(prompt), "stop"


def test_a_one_token_reply_still_streams(tok, monkeypatch):
    monkeypatch.setattr(mlx_backend, "MlxRuntime", OneTokenRuntime)
    with TestClient(create_app(Settings(backend="mlx", mlx_model="/models/dg"), tokenizer=tok)) as c:
        for _ in range(20):  # it was a race, so once proves nothing
            with c.stream("POST", "/v1/chat/completions", json=dict(CHAT, stream=True)) as r:
                assert r.status_code == 200
                events = [_json.loads(line[6:]) for line in r.iter_lines()
                          if line.startswith("data: ") and line.strip() != "data: [DONE]"]
            text = "".join(e["choices"][0]["delta"].get("content", "") for e in events if e["choices"])
            assert text == "7", (text, events)
