"""ForJev 0.1.0: typed decisions from an existing Qwen/vLLM HTTP server."""
import asyncio
import json
import math
import string
import time

import httpx

from .encoders import EncoderEngine
from .engine import Overloaded, SchemaError, Upstream, model_ns, to_answer
from .decision_scores import normalize, parse_response

VERSION = "0.1.0"
MODEL_NAME = "forjev-qwen-next"
LABELS = tuple(string.ascii_uppercase + string.ascii_lowercase + string.digits) + tuple(
    a + b for a in string.ascii_uppercase for b in string.ascii_uppercase
)

class ForJevEngine(EncoderEngine):
    model_name = MODEL_NAME
    max_choices = 255

    def __init__(self, settings):
        # EncoderEngine's default constructor is for in-process models and
        # disallows images. This model lives entirely in the existing vLLM.
        self.s = settings
        self.max_choices = settings.forjev_max_choices
        self.waiting = 0
        self.client = httpx.AsyncClient(
            base_url=settings.upstream.rstrip("/"),
            timeout=httpx.Timeout(120.0, connect=5.0),
            trust_env=False,
            headers={"Authorization": f"Bearer {settings.forjev_upstream_api_key}"} if settings.forjev_upstream_api_key else {},
        )
        self.slots = asyncio.Semaphore(settings.max_inflight)
        self.label_ids = []
        self._candidate_index = 0
        self._labels_lock = asyncio.Lock()

    async def close(self):
        await self.client.aclose()

    async def _post(self, path, body):
        # The pinned vLLM build can intermittently return a 500 while building
        # chat logprobs for a one-token response. Retry only that known failure;
        # tokenize, client errors, and other server errors retain their usual
        # behavior. Each retry is a new read-only inference request.
        for attempt in range(3):
            r = await self.client.post(path, json=body)
            if 400 <= r.status_code < 500:
                try:
                    message = r.json().get("error", {}).get("message") or r.text
                except ValueError:
                    message = r.text
                raise Upstream(str(message)[:500])
            if path == "/v1/chat/completions" and r.status_code == 500 and attempt < 2:
                try:
                    message = r.json().get("error", {}).get("message")
                except (ValueError, AttributeError):
                    message = None
                if message == "list index out of range":
                    await asyncio.sleep((0.05, 0.15)[attempt])
                    continue
            r.raise_for_status()
            return r.json()

    async def _ids(self, count):
        # Discover unique, single-token labels in the *served* tokenizer.
        # Never assume Qwen4 will retain the same token IDs as Flash-Next.
        async with self._labels_lock:
            while len(self.label_ids) < count and self._candidate_index < len(LABELS):
                label = LABELS[self._candidate_index]
                tokens = (await self._post("/tokenize", {
                    "model": self.s.upstream_model,
                    "prompt": label,
                    "add_special_tokens": False,
                })).get("tokens", [])
                self._candidate_index += 1
                if len(tokens) == 1 and type(tokens[0]) is int and tokens[0] >= 0 and tokens[0] not in (tid for _, tid in self.label_ids):
                    self.label_ids.append((label, tokens[0]))
            if len(self.label_ids) < count:
                raise SchemaError(
                    f"Qwen tokenizer offers only {len(self.label_ids)} verified single-token labels; "
                    f"this choice needs {count}", ("body", "questions")
                )
            return list(self.label_ids[:count])

    async def _question(self, state_text, q, images):
        pairs = await self._ids(len(q["choices"]))
        ids = [tid for _, tid in pairs]
        request = self.question_request(state_text, q, images, pairs)
        if self.s.forjev_scoring != "chat_logprobs":
            async with self.slots:
                d = await self._post("/v1/decision_scores", {
                    **request, "candidate_token_ids": ids,
                    "require_prefill": self.s.forjev_scoring == "prefill_scores",
                })
            try:
                return parse_response(d, ids, require_prefill=self.s.forjev_scoring == "prefill_scores")
            except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
                raise Upstream(f"Qwen returned invalid decision scores: {exc}") from exc
        async with self.slots:
            d = await self._post("/v1/chat/completions", {
                **request,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": True,
                "top_logprobs": 1,
                "logprob_token_ids": ids,
                "return_tokens_as_token_ids": True,
            })
        try:
            rows = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            got = {int(r["token"].split(":", 1)[1]): float(r["logprob"])
                   for r in rows if str(r.get("token", "")).startswith("token_id:")}
            values = [got[i] for i in ids]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("non-finite logprob")
            return normalize(values), d["usage"]["prompt_tokens"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise Upstream("Qwen returned incomplete candidate logprobs") from exc

    def question_request(self, state_text, q, images, pairs):
        """One prompt builder for inference and controlled scoring diagnostics."""
        options = "\n".join(
            f"{label}. {name}: {description}" if description else f"{label}. {name}"
            for (label, _), (name, description) in zip(pairs, q["choices"])
        )
        prompt = (f"Current state: {state_text}\nQuestion: {q['instructions']}\n"
                  f"Options:\n{options}\nAnswer with one label only:")
        content = list(images or []) + [{"type": "text", "text": prompt}]
        return {
            "model": self.s.upstream_model,
            "messages": [
                {"role": "system", "content": "Select exactly one listed answer from the current state and image, if present."},
                {"role": "user", "content": content},
            ],
            "chat_template_kwargs": {"enable_thinking": False},
        }

    async def decide(self, questions, state, seed, images=None, options=None):
        opts = options or {}
        for field, used in (("steps", (opts.get("steps") or 1) > 1),
                            ("samples", (opts.get("samples") or 1) > 1),
                            ("think", bool(opts.get("think"))),
                            ("sequential", bool(opts.get("sequential")))):
            if used:
                raise SchemaError(f"{self.model_name} does not support {field}", ("body", field))
        if self.waiting >= self.s.max_queue:
            raise Overloaded(f"{self.model_name} is at capacity. Retry shortly.")
        qs, forced = self.build_schema(questions)
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        self.waiting += 1
        try:
            started = time.perf_counter_ns()
            tasks = [asyncio.create_task(self._question(state_text, q, images)) for q in qs]
            try:
                results = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            spent = model_ns.get()
            if spent is not None and qs:
                spent[0] += time.perf_counter_ns() - started
        finally:
            self.waiting -= 1
        answers = dict(forced)
        for q, (probabilities, _) in zip(qs, results):
            answers[q["key"]] = to_answer(q, probabilities)
        return {key: answers[key] for key in questions}, sum(tokens for _, tokens in results), 0
