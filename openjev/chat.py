"""POST /v1/chat/completions: ordinary text generation from the same DiffusionGemma.

OpenAI-compatible, so tools that talk to a chat model (jev-ultrafast's text
model, the openai SDKs) can point their base URL here. vLLM refuses a few
things for diffusion models (structured outputs, temperature other than 1,
seeds, logit bias), so requests are normalized first: those fields are
dropped, and JSON mode becomes an instruction plus extraction of the first
JSON object from the reply.

Two generators serve the same routes: Generator proxies to vLLM over HTTP,
MlxGenerator denoises in process on the MLX runtime thread. Normalization,
capacity and the response shape are shared, so an OpenAI client cannot tell
which one answered.
"""
import asyncio
import json
import re
import secrets
import time

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from .config import GEN_MODEL

# everything else -- temperature, seed, min_p, logit_bias, penalties, reasoning
# switches, response_format, n, ... -- is dropped, per the module docstring above.
PASSTHROUGH = {"messages", "max_tokens", "stop", "top_p", "top_k", "stream", "stream_options",
               "tools", "tool_choice", "logprobs", "top_logprobs", "chat_template_kwargs"}
DEFAULT_MAX_TOKENS = 1024
JSON_INSTRUCTION = "Reply with exactly one JSON object and nothing else: no prose, no code fences."
MODEL_NAMES = {GEN_MODEL, "diffusiongemma"}
# end-of-stream marker, distinguishable from any chunk of text including ""
END = object()


def oai_error(status, message, type_="invalid_request_error", code=None, headers=None):
    return JSONResponse({"error": {"message": message, "type": type_, "code": code}}, status_code=status, headers=headers)


class Generator:
    def __init__(self, settings):
        self.s = settings
        self.client = httpx.AsyncClient(base_url=settings.upstream.rstrip("/"),
                                        timeout=httpx.Timeout(300.0, connect=5.0))
        self.slots = asyncio.Semaphore(settings.gen_max_inflight)
        self.running = 0

    async def close(self):
        await self.client.aclose()

    def normalize(self, body):
        """The vLLM request for an OpenAI-style one, and whether the caller
        asked for JSON. Raises ValueError (a 400) for a max_tokens that is not
        an integer: int() used to crash the route on a string, and silently
        turned True into 1 and 3.9 into 3."""
        out = {k: v for k, v in body.items() if k in PASSTHROUGH}
        out["model"] = self.s.upstream_model
        max_tokens = out.get("max_tokens", body.get("max_completion_tokens"))
        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            raise ValueError(f"max_tokens must be an integer, got {max_tokens!r}")
        out["max_tokens"] = max(1, min(max_tokens, self.s.gen_max_tokens))
        out["chat_template_kwargs"] = {"enable_thinking": False, **(out.get("chat_template_kwargs") or {})}
        if out.get("stream"):
            out["stream_options"] = {**(out.get("stream_options") or {}), "include_usage": True}
        fmt = body.get("response_format") or {}
        json_mode = fmt.get("type") in ("json_object", "json_schema")
        if json_mode:
            note = JSON_INSTRUCTION
            schema = (fmt.get("json_schema") or {}).get("schema")
            if schema:
                note += " It must match this JSON schema: " + json.dumps(schema, ensure_ascii=False)
            msgs = [dict(m) for m in out["messages"]]
            if msgs and msgs[0].get("role") == "system" and isinstance(msgs[0].get("content"), str):
                msgs[0]["content"] = msgs[0]["content"].rstrip() + "\n\n" + note
            else:
                msgs.insert(0, {"role": "system", "content": note})
            out["messages"] = msgs
        return out, json_mode

    async def stream(self, upstream, request):
        self.running += 1
        try:
            await self.slots.acquire()
        except BaseException:
            # a client that goes away while waiting for a slot must not keep
            # the capacity it never got; the count only grew, never shrank
            self.running -= 1
            raise

        def release():
            self.slots.release()
            self.running -= 1

        try:
            r = await self.client.send(self.client.build_request("POST", "/v1/chat/completions", json=upstream), stream=True)
        except httpx.HTTPError as e:
            release()
            return oai_error(503, f"inference backend unavailable: {type(e).__name__}", "api_error", headers={"retry-after": "2"})
        except BaseException:
            # cancelled mid-connect: the slot and the count were taken already
            release()
            raise
        if r.status_code >= 400:
            try:
                await r.aread()
            finally:
                await r.aclose()
                release()
            return oai_error(400 if r.status_code < 500 else 503, upstream_message(r), "invalid_request_error" if r.status_code < 500 else "api_error")

        async def lines():
            async for line in r.aiter_lines():
                yield line.replace(f'"model":"{self.s.upstream_model}"', f'"model":"{GEN_MODEL}"') + "\n"

        async def done():
            await r.aclose()
            release()

        return StreamingResponse(lines(), media_type="text/event-stream", background=BackgroundTask(done))

    async def complete(self, upstream, json_mode):
        self.running += 1
        try:
            async with self.slots:
                r = await self.client.post("/v1/chat/completions", json=upstream)
        except httpx.HTTPError as e:
            return oai_error(503, f"inference backend unavailable: {type(e).__name__}", "api_error", headers={"retry-after": "2"})
        finally:
            self.running -= 1
        if r.status_code >= 400:
            return oai_error(400 if r.status_code < 500 else 503, upstream_message(r), "invalid_request_error" if r.status_code < 500 else "api_error")
        d = r.json()
        d["model"] = GEN_MODEL
        if json_mode:
            for c in d.get("choices", []):
                msg = c.get("message") or {}
                if isinstance(msg.get("content"), str):
                    msg["content"] = extract_json(msg["content"])
        return d


class MlxGenerator(Generator):
    """The same routes answered in process. Normalization, capacity and the
    response shape are inherited; only where the tokens come from differs.

    The runtime has one thread, and a reply holds it for its whole length, so
    gen_max_inflight above 1 only queues -- the semaphore still bounds how many
    requests may be waiting on it, and gen_max_queue rejects the rest."""

    def __init__(self, settings, engine):
        super().__init__(settings)
        self.engine = engine

    def prompt_ids(self, upstream):
        """The chat prompt as token ids, ending in the empty thought block.

        Thinking stays off, as it is on vLLM. Seeding the scaffold, the same
        trick a read already uses, has the model start past the thought channel
        on most replies; it still opens one of its own accord now and then,
        which is why the markers are also skipped when generating."""
        tok = self.engine.tok
        thinking = bool((upstream.get("chat_template_kwargs") or {}).get("enable_thinking"))
        out = tok.apply_chat_template(upstream["messages"], tokenize=True,
                                      add_generation_prompt=True, enable_thinking=thinking)
        ids = out["input_ids"] if hasattr(out, "keys") else out
        return [int(t) for t in ids] + self.engine.scaffold

    def stop_ids(self, upstream):
        """Extra stop token ids for the request's `stop` strings. Only stops
        that are a single token can end a denoised block, so multi-token ones
        are dropped rather than half-honoured."""
        stops = upstream.get("stop")
        stops = [stops] if isinstance(stops, str) else list(stops or ())
        out = []
        for s in stops:
            ids = self.engine.enc(s)
            if len(ids) == 1:
                out.append(ids[0])
        return out

    async def generate(self, prompt, upstream, emit):
        """Generate on the runtime thread. ``emit(text, token)`` is called there,
        not on the event loop, and returns False to cancel.

        The thought-channel markers are skipped: the model opens a channel of its
        own accord on some replies, and a chat client asked for the reply, not the
        markers. `think` asks for no skipping, because it wants the thought."""
        engine = self.engine
        return await asyncio.get_running_loop().run_in_executor(
            engine.runtime.pool, engine.runtime.generate,
            prompt, upstream["max_tokens"], self.stop_ids(upstream), emit,
            engine.thought_open + engine.thought_close)

    async def complete(self, upstream, json_mode):
        try:
            prompt = self.prompt_ids(upstream)
            if len(prompt) > self.s.mlx_max_prompt:
                raise ValueError(f"the request is {len(prompt)} tokens; the limit is {self.s.mlx_max_prompt}")
        except ValueError as e:
            return oai_error(400, str(e))
        self.running += 1
        try:
            async with self.slots:
                parts = []
                ids, prompt_tokens, finish = await self.generate(
                    prompt, upstream, lambda text, token: parts.append(text) or True)
        finally:
            self.running -= 1
        text = "".join(parts)
        return {"id": completion_id(), "object": "chat.completion", "created": int(time.time()),
                "model": GEN_MODEL,
                "choices": [{"index": 0, "finish_reason": finish_reason(finish),
                             "message": {"role": "assistant",
                                         "content": extract_json(text) if json_mode else text},
                             "logprobs": None}],
                "usage": usage(prompt_tokens, len(ids))}

    async def stream(self, upstream, request):
        try:
            # rejected here, before the response starts: once the stream is open the
            # status is sent and a 400 can no longer be
            prompt = self.prompt_ids(upstream)
            if len(prompt) > self.s.mlx_max_prompt:
                raise ValueError(f"the request is {len(prompt)} tokens; the limit is {self.s.mlx_max_prompt}")
        except ValueError as e:
            return oai_error(400, str(e))

        loop = asyncio.get_running_loop()
        # a little slack so the runtime thread is not blocked on a slow reader, but
        # also cannot run arbitrarily far ahead of a client that stopped reading
        queue = asyncio.Queue(maxsize=64)
        cancel = False

        def end():
            """Queue the end marker, making room if the buffer is full. A dropped
            marker leaves the drain waiting on a queue nothing else will ever
            fill: the request hung and the slot was never given back."""
            while True:
                try:
                    queue.put_nowait(END)
                    return
                except asyncio.QueueFull:
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass

        def offer(text):
            """Queue a chunk, on the event loop. A full queue means the reader is
            gone or hopelessly behind; cancel the reply rather than wedge the
            runtime, and rather than drop reply text a live client would never
            know it lost."""
            nonlocal cancel
            try:
                queue.put_nowait(text)
            except asyncio.QueueFull:
                cancel = True
                end()

        def emit(text, token):
            # called on the runtime thread: hand the chunk over and report whether
            # the reply is still wanted. The model denoises a whole canvas before
            # emitting any of it, so a cancel lands at the next block, not the next
            # token -- seconds, not the whole reply.
            loop.call_soon_threadsafe(offer, text)
            return not cancel

        self.running += 1
        try:
            await self.slots.acquire()
        except BaseException:
            # as in Generator.stream: a cancelled wait must return its count
            self.running -= 1
            raise
        cid, created = completion_id(), int(time.time())
        usage_wanted = (upstream.get("stream_options") or {}).get("include_usage")

        released = False

        async def reap(task):
            """Wait for the runtime to stop, then give the slot back exactly once."""
            nonlocal released
            if released:
                return
            released = True
            try:
                await task
            except Exception:
                pass
            self.slots.release()
            self.running -= 1

        async def watch_client():
            """Ask on a timer whether the client is still there; on a disconnect,
            tell the runtime to stop and wake the drain with the end marker."""
            nonlocal cancel
            while not cancel:
                await asyncio.sleep(0.1)
                if await request.is_disconnected():
                    cancel = True
                    end()
                    return

        async def produce():
            """Generate, then mark the end of the stream. The sentinel is what the
            drain below stops on: a finished task does NOT mean the queue holds
            everything, because the last chunks may still be in flight on a
            call_soon_threadsafe that has not run yet."""
            try:
                return await self.generate(prompt, upstream, emit)
            finally:
                end()

        async def body():
            nonlocal cancel
            task = asyncio.ensure_future(produce())
            watcher = None
            yield sse(chunk(cid, created, {"role": "assistant", "content": ""}))
            try:
                # Closing this generator is not a reliable cancel on its own: it
                # reaches us only when the server next writes to a dead transport,
                # which may be after the whole reply. So a watcher polls for the
                # disconnect on its own timer rather than inline -- while tokens
                # keep arriving this loop never idles, and a check tied to it would
                # hardly ever run. The watcher ends the loop by queueing END.
                watcher = asyncio.ensure_future(watch_client())
                while True:
                    text = await queue.get()
                    if text is END:
                        break
                    if text:
                        yield sse(chunk(cid, created, {"content": text}))
                if cancel:
                    return  # nobody is listening, so no finish chunk and no [DONE]
                ids, prompt_tokens, finish = await task
                yield sse(chunk(cid, created, {}, finish_reason=finish_reason(finish)))
                if usage_wanted:
                    yield sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                               "model": GEN_MODEL, "choices": [],
                               "usage": usage(prompt_tokens, len(ids))})
                yield "data: [DONE]\n\n"
            finally:
                # Reached normally, and on GeneratorExit when the client goes away.
                # An async generator being closed may not await, so the wait for the
                # runtime thread -- which cannot be interrupted, only asked to stop
                # at its next block -- happens in a task of its own, which is also
                # what releases the slot.
                cancel = True
                if watcher is not None:
                    watcher.cancel()
                asyncio.ensure_future(reap(task))

        return StreamingResponse(body(), media_type="text/event-stream")


def completion_id():
    return "chatcmpl-" + secrets.token_hex(12)


def finish_reason(finish):
    return "stop" if finish == "stop" else "length"


def usage(prompt_tokens, completion_tokens):
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens}


def chunk(cid, created, delta, finish_reason=None):
    return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": GEN_MODEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason, "logprobs": None}]}


def sse(payload):
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def extract_json(text):
    """The first JSON object or array in a reply, as text; the reply unchanged
    when there is none."""
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    dec = json.JSONDecoder()
    for i, ch in enumerate(stripped):
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(stripped[i:])
                return json.dumps(obj, ensure_ascii=False)
            except ValueError:
                continue
    return text


def upstream_message(r):
    try:
        d = r.json()
        msg = (d.get("error") or {}).get("message") or d.get("message") or r.text
    except ValueError:
        msg = r.text
    # as Engine._post does: an upstream error page is not the client's business
    return str(msg)[:500]


def add_chat_routes(app):
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        gen = request.app.state.generator
        try:
            body = await request.json()
        except ValueError:
            return oai_error(400, "The request body is not valid JSON.")
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
            return oai_error(400, "messages must be a non-empty array.")
        model = body.get("model")
        if not isinstance(model, str):
            return oai_error(400, "model is required and must be a string.")
        if model not in MODEL_NAMES:
            return oai_error(404, f"Model {model!r} not found. Available: {GEN_MODEL}.", code="model_not_found")
        if gen.running >= gen.s.gen_max_inflight + gen.s.gen_max_queue:
            return oai_error(529, "Text generation is at capacity. Retry shortly.", "overloaded_error", headers={"retry-after": "2"})
        try:
            upstream, json_mode = gen.normalize(body)
        except ValueError as e:
            return oai_error(400, str(e))
        if upstream.get("stream"):
            return await gen.stream(upstream, request)
        return await gen.complete(upstream, json_mode)
