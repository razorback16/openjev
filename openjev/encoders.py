"""Small System One models: Laya, Verdict, CLM, JevK5 and d1, one per container (or process).

Selected with OPENJEV_BACKEND=laya, verdict, clm, jevk5 or d1. Laya and Verdict read the
questions of a request in batched forward passes of a bidirectional encoder and
a classification head, one sequence per question; CLM scores each option against
the state with contrastive heads; JevK5 reads its answer letters' logits. Either way an
answer is a distribution over the caller's options exactly as with DiffusionGemma. The API layer, the error contract and
the answer shapes are shared with the vLLM backend; only the read differs.

- laya-1.0: Laya by Nandakishor M / Convai Innovations
  (github.com/NandhaKishorM/laya), checkpoint convaiinnovations/laya-typed-decisions
  (ModernBERT-large, 421M, Apache-2.0), through the `laya` package. Its system_one already takes Jev's
  questions; its answers are reshaped here to Jev's exact shapes.
- verdict-1.4: Verdict by Heman10x, checkpoint heman10x/rlcd-modernbert-151m (ModernBERT-base + GLiClass, 151M,
  Apache-2.0). The prompt format and the per-option-count temperatures follow
  core/formatting.py and core/engine_encoder.py of
  github.com/Heman10x-NGU/Verdict-open-jev (v1.4 inference). That package is
  not installed: it ships top-level `openjev` and `core` packages.
- clm-v0.1: CLM by Contrastive-LM (github.com/Contrastive-LM/CLM, Apache-2.0),
  checkpoint Contrastive-LM/CLM-v0.1-8B: a state head and an action head (MLPs
  to 512-d) over the last-token embeddings of a frozen Qwen3-8B. The embeddings
  come from vLLM's pooling runner in the same container; the heads, the prompt
  layout and the scoring are the `contrastive-lm` package's own.
- jevk5-0.2: JevK5 by Alibi Serikbay (github.com/allebee/jevk5, Apache-2.0),
  checkpoint alibiserikbay/JevK5: Qwen3.5-4B with a merged, distilled LoRA. A
  question's options are lettered A-P in a JSON prompt, and the answer is a
  softmax over those letters' next-token logits under one calibration temperature
  (SemIf's readout). vLLM in the same container returns the letters' logprobs;
  the prompt and the reading of more than 16 options are the `jevk5` package's own.
- d1-2b: d1-2B by Liquid AI (huggingface.co/LiquidAI/d1-2B, LFM Open License v1.0),
  through Transformers with the repo's own code at a pinned revision. d1 answers in
  Jev's shapes itself, so its answers are passed through; see D1Engine.

None of these models has images, denoise steps, noise draws or a thought, so those
request options are refused with a 400. torch, gliclass, laya, clm and
jevk5 are imported lazily so the vLLM image never needs them.
"""
import asyncio
import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from .engine import Overloaded, SchemaError, Upstream, model_ns, text_of, to_answer

log = logging.getLogger("openjev")


WARMUP_QUESTIONS = {"c": {"type": "choice", "instructions": "x", "criteria": {"a": None, "b": None}},
                    "s": {"type": "score", "instructions": "x", "criteria": ["low", "high"]},
                    "n": {"type": "noul", "instructions": "x"}}


class EncoderEngine:
    """The Engine contract (decide, close) for an in-process encoder. The model
    lives on one thread, from loading on; a request is one call on it."""

    model_name = ""
    max_choices = 255  # Jev's limit, as for DiffusionGemma
    workers = 1  # threads that read; one for a model that runs in this process

    def __init__(self, settings):
        self.s = settings
        self.waiting = 0
        self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix=f"openjev-{self.model_name}")
        self.pool.submit(self.load).result()
        if self.s.warmup:
            # the first read compiles Triton launchers (about a second); do it before /health is up
            qs, _ = self.build_schema(WARMUP_QUESTIONS)
            self.pool.submit(self.read, "warmup", qs).result()

    async def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def load(self):
        raise NotImplementedError

    def read(self, state, qs):
        """[probabilities per question, in the caller's option order], input tokens.
        Runs on the model's thread. Questions are read in batches of at most
        OPENJEV_ENCODER_BATCH, because every question is a full-length sequence
        and a shared GPU has little room: Laya moves itself to the CPU for good
        after one CUDA out-of-memory error."""
        n = self.s.encoder_batch
        probs, tokens = [], 0
        for i in range(0, len(qs), n):
            p, t = self.read_batch(state, qs[i:i + n])
            probs += p
            tokens += t
        return probs, tokens

    def read_batch(self, state, qs):
        raise NotImplementedError

    def build_schema(self, questions):
        """The questions to read, and the answers that need no read: a choice with
        one option or a score with one level, as in Engine.build_schema."""
        qs, forced = [], {}
        for qid, q in questions.items():
            loc = ("body", "questions", qid, "criteria")
            kind, crit = q["type"], q.get("criteria")
            if kind == "noul":
                crit = crit or {}
                choices = [("yes", text_of(crit.get("true"))), ("no", text_of(crit.get("false")))]
            elif kind == "choice":
                if not crit:
                    raise SchemaError(f"Choice question must have at least one choice: {qid}", loc)
                if len(crit) == 1:
                    only = next(iter(crit))
                    forced[qid] = {"type": "choice", "choice": only, "probabilities": {only: 1.0}, "confidence": 1.0}
                    continue
                if len(crit) > self.max_choices:
                    raise SchemaError(f"Too many choices. Must have at most {self.max_choices} choices.", loc)
                choices = [(name, text_of(desc)) for name, desc in crit.items()]
            elif kind == "score":
                if len(crit) > 10:
                    raise SchemaError("Too many score levels. Must have at most 10 levels.", loc)
                if len(crit) == 1:
                    forced[qid] = {"type": "score", "score": 0.0, "legend": {"0": crit[0]},
                                   "probabilities": {"0": 1.0}, "confidence": 1.0}
                    continue
                choices = [(str(i), text_of(c)) for i, c in enumerate(crit)]
            else:
                raise SchemaError(f"unknown question type {kind!r}", ("body", "questions", qid, "type"))
            # raw_instructions: CLM and JevK5 render an object themselves, not as text_of's JSON
            qs.append({"key": qid, "type": kind, "instructions": text_of(q.get("instructions")),
                       "raw_instructions": q.get("instructions"), "criteria": crit, "choices": choices,
                       "legend": list(crit) if kind == "score" else None})
        return qs, forced

    def refuse_unsupported(self, images, options):
        """A 400 for images and the DiffusionGemma read options, which no encoder has."""
        opts = options or {}
        unsupported = [("images", bool(images)), ("steps", (opts.get("steps") or 1) > 1),
                       ("samples", (opts.get("samples") or 1) > 1), ("think", bool(opts.get("think"))),
                       ("sequential", bool(opts.get("sequential")))]
        for field, used in unsupported:
            if used:
                raise SchemaError(f"{self.model_name} does not support {field}", ("body", field))

    async def decide(self, questions, state, seed, images=None, options=None):
        """Engine.decide's contract. ``seed`` is unused: a read is deterministic."""
        self.refuse_unsupported(images, options)
        if self.waiting >= self.s.max_queue:
            raise Overloaded(f"{self.model_name} is at capacity. Retry shortly.")
        qs, forced = self.build_schema(questions)
        self.waiting += 1
        started = time.perf_counter_ns()
        try:
            probs, tokens = (await asyncio.get_running_loop().run_in_executor(self.pool, self.read, state, qs)
                             if qs else ([], 0))
        finally:
            self.waiting -= 1
            # the model= part of the Server-Timing header, as Engine records it
            spent = model_ns.get()
            if spent is not None and qs:
                spent[0] += time.perf_counter_ns() - started
        answers = dict(forced)
        for q, p in zip(qs, probs):
            answers[q["key"]] = to_answer(q, p)
        return {k: answers[k] for k in questions}, tokens, 0


class LayaEngine(EncoderEngine):
    model_name = "laya-1.0"

    def load(self):
        import laya
        import torch

        # laya sends an empty HF_TOKEN as "Authorization: Bearer ", which httpx refuses;
        # compose files pass HF_TOKEN through even when it is not set
        if not os.environ.get("HF_TOKEN"):
            os.environ.pop("HF_TOKEN", None)
        # Loaded on the CPU and moved by hand: laya would put its fp32 weights on the GPU,
        # and it falls back to the CPU instead of failing when it cannot use the GPU.
        self.agent = laya.load(self.s.laya_model, device="cpu")
        self.device = self.agent.device
        target = encoder_device(self.s)
        if target.type == "cuda":
            dtype = gpu_dtype(torch, target)
            model = self.agent.model.to(dtype)
            model.act_head.float()  # it takes fp32 features; its output is not used
            self.agent.model = model.to(target).eval()
            # system_one runs under autocast in this dtype on a GPU
            self.agent.device = self.device = target
            self.agent.dtype = dtype

    def read_batch(self, state, qs):
        # instructions go as text: laya would send a missing one to the model as "null"
        questions = {str(i): {"type": q["type"], "instructions": q["instructions"], "criteria": q["criteria"]}
                     for i, q in enumerate(qs)}
        try:
            out = self.agent.system_one(state, questions)
        except ValueError:  # the options overflow the head's token budget
            raise SchemaError(f"Too many choices for {self.model_name}: a question's options must fit in "
                              f"{self.agent.cfg.get('head_max_len')} tokens.") from None
        if self.agent.device != self.device:
            # A CUDA out-of-memory error moved the model to the CPU for good. Exit, so the
            # container restarts on the GPU instead of serving slowly with no error.
            log.error("laya moved from %s to %s after a GPU error; exiting", self.device, self.agent.device)
            os._exit(3)
        probs = []
        for i, q in enumerate(qs):
            a = out["answers"][str(i)]
            if q["type"] == "noul":
                p = [a["noul"], 1.0 - a["noul"]]
            else:
                p = list(a["probabilities"].values())
                z = sum(p)  # laya rounds each to 4 places
                p = [v / z for v in p]
            probs.append(p)
        return probs, out["usage"]["input_tokens"]


def encoder_device(settings):
    import torch

    return torch.device(settings.device or ("cuda" if torch.cuda.is_available() else "cpu"))


def gpu_dtype(torch, device):
    """bf16 weights on a GPU: half the memory of fp32, and the answers match fp32's
    to within about 0.02 in probability. fp16 where bf16 is missing (before Ampere)."""
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


# Verdict's prompt contract (core/formatting.py)
LABEL_MARKER, SEP_MARKER = "<<LABEL>>", "<<SEP>>"
ABSTAIN = "insufficient evidence"
VERDICT_MAX_LEN = 512  # the weights were trained on short states; v1.4 cut 1024 to 512


def verdict_prompt(q, context):
    """Verdict's model input for one question: the label prefix, then the text. Every
    question gets an "insufficient evidence" option as its last label."""
    ins = q["instructions"]
    if q["type"] == "noul":
        labels = [f"true: {ins}", f"false: not {ins}"]
        text = f"Context:\n{context}\n\nEvaluate proposition: {ins}"
    else:
        if q["type"] == "choice":
            labels = [f"It is {desc or name}" for name, desc in q["choices"]]
        else:
            labels = [f"{desc} (Value: {float(i)})" for i, (_, desc) in enumerate(q["choices"])]
        text = f"Question: {ins}\n\nContext:\n{context}" if ins else context
    labels.append(ABSTAIN)
    return "".join(LABEL_MARKER + label for label in labels) + SEP_MARKER + text, len(labels)


class VerdictEngine(EncoderEngine):
    model_name = "verdict-1.4"
    max_choices = 24  # the head has 25 logits, the last kept for "insufficient evidence"

    def load(self):
        import torch
        from gliclass import GLiClassModel
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        self.torch = torch
        self.device = encoder_device(self.s)
        model = GLiClassModel.from_pretrained(self.s.verdict_model)
        if self.device.type == "cuda":
            model = model.to(gpu_dtype(torch, self.device))
        self.model = model.to(self.device).eval()
        self.tok = AutoTokenizer.from_pretrained(self.s.verdict_model)
        path = self.s.verdict_model
        cal_file = (os.path.join(path, "calibrator.json") if os.path.isdir(path)
                    else hf_hub_download(path, "calibrator.json"))
        with open(cal_file) as f:
            cal = json.load(f)
        self.temperature = float(cal["temperature"])
        self.per_k = {int(k): float(v) for k, v in cal.get("per_k", {}).items()}

    def read_batch(self, state, qs):
        torch = self.torch
        context = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        prompts, ks = zip(*(verdict_prompt(q, context) for q in qs))
        batch = self.tok(list(prompts), padding=True, truncation=True, max_length=VERDICT_MAX_LEN,
                         return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**batch).logits.float().cpu()
        probs = []
        for row, k in zip(logits, ks):
            z = row[:k] / self.per_k.get(k, self.temperature)
            p = torch.softmax(z, -1)[:-1]
            # Jev's distribution is over the caller's options: the abstention mass is dropped
            p = (p / p.sum()).tolist()
            probs.append(p if math.isfinite(sum(p)) else [1.0 / (k - 1)] * (k - 1))
        return probs, int(batch["attention_mask"].sum())


def _left_truncating_embedder():
    import requests
    from clm.embedder import Embedder

    class LeftTruncating(requests.Session):
        def post(self, url, json=None, **kw):
            return super().post(url, json=dict(json, truncation_side="left"), **kw)

    class LeftTruncatingEmbedder(Embedder):
        """clm's embedder, keeping the end of a long text rather than its start. The state
        head reads context first and the question last, so right truncation (vLLM's
        default) would cut the question off. Contrastive-LM/CLM PR #6 makes the same fix.
        Its connection pool fits every reading thread."""

        def __init__(self, workers, **kw):
            super().__init__(**kw)
            self.session = LeftTruncating()
            adapter = requests.adapters.HTTPAdapter(pool_maxsize=workers)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)

    return LeftTruncatingEmbedder


class ClmEngine(EncoderEngine):
    model_name = "clm-v0.1"

    def __init__(self, settings):
        # The model is the vLLM server; a thread only waits on it, so many read at once
        # and vLLM batches their embeddings together.
        self.workers = settings.clm_workers
        super().__init__(settings)

    def load(self):
        from clm.engine import Engine
        from clm.heads import HF_FILE

        head = self.s.clm_head
        if not os.path.isfile(head):
            from huggingface_hub import hf_hub_download

            head = hf_hub_download(head, HF_FILE)
        device = str(encoder_device(self.s))
        embedder = _left_truncating_embedder()(
            self.workers, url=f"{self.s.upstream}/v1/embeddings", model=self.s.upstream_model,
            max_tokens=self.s.clm_max_tokens, batch=1024, cache_size=self.s.clm_embed_cache)
        self.clm = Engine(embedder=embedder, checkpoint=head, device=device, action_cache=self.s.clm_cache)

    def read(self, state, qs):
        """One call for all of a request's questions, so their texts reach vLLM as one batch."""
        from clm.embedder import EmbedderError

        questions = {str(i): {"type": q["type"], "instructions": q["raw_instructions"], "criteria": q["criteria"]}
                     for i, q in enumerate(qs)}
        try:
            out = self.clm.answer(state, questions)
        except EmbedderError as e:
            # vLLM unreachable or failing: a 503, as for the vLLM backend
            raise httpx.TransportError(str(e)) from e
        probs = []
        for i, q in enumerate(qs):
            a = out["answers"][str(i)]
            # clm keeps the caller's option order, as we do; a noul answer is P(true), our first option
            probs.append([a["noul"], 1.0 - a["noul"]] if q["type"] == "noul" else list(a["probabilities"].values()))
        return probs, out["usage"]["input_tokens"]


class JevK5Engine(EncoderEngine):
    model_name = "jevk5-0.2"

    def __init__(self, settings):
        # As for CLM, the model is the vLLM server. The questions of one request are read at
        # once, on a second pool, so its reading threads never wait on each other.
        self.workers = settings.jevk5_workers
        self.fanout = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="openjev-jevk5-read")
        super().__init__(settings)

    async def close(self):
        await super().close()
        self.fanout.shutdown(wait=True, cancel_futures=True)

    def load(self):
        from jevk5.prompt import LETTERS

        self.http = httpx.Client(base_url=self.s.upstream, timeout=httpx.Timeout(300.0, connect=5.0),
                                 limits=httpx.Limits(max_connections=self.workers))
        self.temperature = jevk5_temperature(self.s.jevk5_model)
        # Each answer letter must be one token, as jevk5's own runtime checks
        ids = [self._post("/tokenize", {"model": self.s.upstream_model, "prompt": letter,
                                        "add_special_tokens": False})["tokens"] for letter in LETTERS]
        if any(len(t) != 1 for t in ids):
            raise ValueError(f"every answer letter must be one token, got {ids}")
        self.letter_ids = [t[0] for t in ids]

    def _post(self, path, body):
        r = self.http.post(path, json=body)
        if 400 <= r.status_code < 500:
            try:
                msg = r.json().get("error", {}).get("message") or r.text
            except ValueError:
                msg = r.text
            raise Upstream(str(msg)[:500])  # a state over the model's 16,384 tokens, say
        r.raise_for_status()
        return r.json()

    def letter_logprobs(self, prompt, count):
        """The first ``count`` letters' logprobs at the answer position, and the prompt's tokens.
        A softmax over them is a softmax over their logits: the full vocabulary's normaliser
        cancels, as jevk5's llama.cpp client notes."""
        ids = self.letter_ids[:count]
        d = self._post("/v1/completions", {
            "model": self.s.upstream_model, "prompt": prompt, "add_special_tokens": False,
            "max_tokens": 1, "temperature": 0, "logprobs": 1, "logprob_token_ids": ids,
            "return_tokens_as_token_ids": True})
        top = {int(k.split(":")[1]): v for k, v in d["choices"][0]["logprobs"]["top_logprobs"][0].items()}
        return [top[i] for i in ids], d["usage"]["prompt_tokens"]

    def read_question(self, state, q):
        from jevk5.prompt import decision_options, prompt_text, spread

        question = {"type": q["type"], "instructions": q["raw_instructions"], "criteria": q["criteria"]}
        options = decision_options(question)
        tokens = 0

        def read(texts):
            nonlocal tokens
            logprobs, n = self.letter_logprobs(prompt_text(state, question["instructions"], texts), len(texts))
            tokens += n
            top = max(logprobs)
            w = [math.exp((v - top) / self.temperature) for v in logprobs]
            return [v / sum(w) for v in w]

        # more than 16 options take several passes, combined as jevk5 combines them
        return spread(read, [text for _, text in options]), tokens

    def read(self, state, qs):
        """[probabilities per question, in the caller's option order], input tokens. jevk5 keeps
        the caller's order: noul is (true, false), P(yes) first as ours; choice and score follow
        the criteria."""
        out = list(self.fanout.map(lambda q: self.read_question(state, q), qs))
        return [p for p, _ in out], sum(t for _, t in out)


def jevk5_temperature(model):
    """The calibration temperature stored with the weights (1.532 for JevK5 v0.2)."""
    path = os.path.join(model, "jevk5_config.json")
    if not os.path.isfile(path):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model, "jevk5_config.json")
    with open(path) as f:
        return float(json.load(f)["temperature"])


class D1Engine(EncoderEngine):
    """Liquid AI's d1-2B through Transformers, on Apple silicon (MPS), CUDA or the CPU.

    d1 answers System One questions itself (`model.system_one`) in Jev's shapes, with its own
    readouts and calibration, so its answers are passed through, not re-read from probabilities.
    Requests that arrive while the model is busy are read together in one
    `system_one_batch` call: a re-ranker sends a whole candidate pool at once, one passage per
    request, and a batch reads it in one pass instead of thirty."""

    model_name = "d1-2b"

    def __init__(self, settings):
        self.pending = []  # (state, questions, future), oldest first
        self.draining = False
        super().__init__(settings)

    def load(self):
        import torch
        from transformers import AutoModel

        device = self.s.device or ("cuda" if torch.cuda.is_available()
                                   else "mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        # trust_remote_code runs the model repo's own Python, so it is pinned to a revision
        self.model = AutoModel.from_pretrained(self.s.d1_model, revision=self.s.d1_revision or None,
                                               trust_remote_code=True, dtype=dtype).to(device).eval()
        log.info("d1: %s@%s on %s", self.s.d1_model, self.s.d1_revision or "main", device)

    def read(self, state, qs):
        """Only the warmup read comes here; requests go through decide's batches."""
        questions = {q["key"]: {k: q[k] for k in ("type", "instructions", "criteria") if q[k] is not None}
                     for q in qs}
        return [], self.read_requests([(state, questions)])[0]["usage"]["input_tokens"]

    def read_requests(self, requests):
        """d1's answers for (state, questions) requests. Runs on the model's thread."""
        return self.model.system_one_batch(requests)

    async def read_each(self, batch):
        """Each request of a batch read alone: an answer, or the exception it raised."""
        loop, out = asyncio.get_running_loop(), []
        for state, questions, _ in batch:
            try:
                out.append((await loop.run_in_executor(self.pool, self.read_requests, [(state, questions)]))[0])
            except Exception as e:
                out.append(e)
        return out

    def next_batch(self):
        """The oldest pending requests that fit OPENJEV_D1_BATCH and OPENJEV_D1_BATCH_CHARS; at least one."""
        batch, chars = [], 0
        for item in self.pending:
            size = len(json.dumps(item[0], ensure_ascii=False)) if not isinstance(item[0], str) else len(item[0])
            if batch and (len(batch) >= self.s.d1_batch or chars + size > self.s.d1_batch_chars):
                break
            batch.append(item)
            chars += size
        del self.pending[:len(batch)]
        return batch

    async def drain(self):
        loop = asyncio.get_running_loop()
        try:
            while self.pending:
                batch = self.next_batch()
                try:
                    results = await loop.run_in_executor(self.pool, self.read_requests,
                                                         [(s, q) for s, q, _ in batch])
                except Exception as e:
                    # one bad request fails the whole call: read each alone, so only it fails
                    results = [e] if len(batch) == 1 else await self.read_each(batch)
                for (_, _, fut), out in zip(batch, results):
                    if fut.done():
                        continue
                    if isinstance(out, ValueError):  # d1 refuses a question it cannot read, saying why
                        fut.set_exception(SchemaError(f"{self.model_name}: {out}"))
                    elif isinstance(out, Exception):
                        fut.set_exception(out)
                    else:
                        fut.set_result(out)
        finally:
            self.draining = False

    async def decide(self, questions, state, seed, images=None, options=None):
        """Engine.decide's contract; the request waits for the next batch."""
        self.refuse_unsupported(images, options)
        if self.waiting >= self.s.max_queue:
            raise Overloaded(f"{self.model_name} is at capacity. Retry shortly.")
        self.build_schema(questions)  # the same limits as the other models, before any read
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.pending.append((state, questions, fut))
        self.waiting += 1
        started = time.perf_counter_ns()
        if not self.draining:
            self.draining = True
            loop.create_task(self.drain())
        try:
            out = await fut
        finally:
            self.waiting -= 1
            spent = model_ns.get()
            if spent is not None:
                spent[0] += time.perf_counter_ns() - started
        answers = out["answers"]
        return {k: answers[k] for k in questions}, out["usage"]["input_tokens"], 0


ENGINES = {"laya": LayaEngine, "verdict": VerdictEngine, "clm": ClmEngine, "jevk5": JevK5Engine, "d1": D1Engine}
