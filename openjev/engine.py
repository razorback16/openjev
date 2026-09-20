"""Structured reads on a DiffusionGemma vLLM server.

Adapted from vLLM's examples/features/diffusion_reads/structured_server.py
(vllm-project/vllm PR #57250, Apache-2.0). A discrete diffusion model denoises
the whole answer canvas per forward pass. Seed the canvas with the answer
template, leave only the label slots as noise, run one read-only denoise step,
and each slot's logprobs are a distribution over that question's labels.

Beyond Jev's contract, a request may opt into the example server's read
options: images ahead of the state, more denoise steps, a fixed number of
noise draws, a thought before the read, and sequential chunks. Left unset,
a read behaves exactly as Jev's contract describes.
"""
import asyncio
import json
import math
import random

import httpx

VOCAB = 262144
TURN_CLOSE = 106
PAD = 0
TOPK = 20
MAX_LABEL_IDS = 128  # vLLM's logprob_token_ids cap per request
SCAFFOLD_TEXT = "<|channel>thought\n<channel|>"  # the empty thought block the chat template leaves to the model

# Answer template shapes: (join between questions, what precedes the label,
# reply instruction). "indexed" costs fewer rows a question; past ten
# questions the saved rows keep a schema in one read.
FORMATS = {
    "lines": ("\n", "{id}: ", 'Reply with one line per question, in this order, formatted as "id: label".'),
    "indexed": (" ", "{id}", "Reply on one line with each question's id immediately followed by its label, separated by single spaces."),
}


class SchemaError(ValueError):
    """A request the model cannot answer as asked; surfaced as a 422."""

    def __init__(self, msg, loc=("body",)):
        super().__init__(msg)
        self.loc = list(loc)


class Overloaded(RuntimeError):
    pass


class Upstream(RuntimeError):
    """vLLM refused a request it was sent (a 4xx); surfaced as a 422."""


DEFAULT_OPTIONS = {"steps": 1, "samples": None, "think": 0, "sequential": False}


def text_of(value):
    """Jev descriptions and instructions may be strings, objects or arrays."""
    if value is None:
        return ""
    return value.strip() if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


class Engine:
    def __init__(self, settings, tokenizer):
        self.s = settings
        self.tok = tokenizer
        self.scaffold = self.enc(SCAFFOLD_TEXT)
        self.thought_open = self.enc("<|channel>thought\n")
        self.thought_close = self.enc("<channel|>")
        self.client = httpx.AsyncClient(base_url=settings.upstream.rstrip("/"),
                                        timeout=httpx.Timeout(120.0, connect=5.0),
                                        limits=httpx.Limits(max_connections=settings.max_inflight * 2))
        self.slots = asyncio.Semaphore(settings.max_inflight)
        self.waiting = 0
        self.choice_labels = self._single_token_labels()
        self._templates = {}

    async def close(self):
        await self.client.aclose()

    def enc(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    # ------------------------------------------------------------------
    # Labels and templates
    # ------------------------------------------------------------------

    def _single_token_labels(self):
        """Choice labels that stay one token after "q1: ", in a stable order."""
        base = self.enc("q1: A")
        cands = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
        cands += [a + b for a in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for b in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        out, seen = [], set()
        for c in cands:
            e = self.enc("q1: " + c)
            if len(e) == len(base) and e[:-1] == base[:-1] and e[-1] not in seen:
                seen.add(e[-1])
                out.append(c)
            if len(out) == MAX_LABEL_IDS:
                break
        return out

    def build_schema(self, questions):
        """Jev questions -> internal question list. Ids are never shown to the
        model; it sees q1, q2, ... and the answers map back by position.
        A choice with one option, or a score with one level, has only one
        possible answer, so it is answered here ("forced") rather than read."""
        qs, forced = [], {}
        for i, (qid, q) in enumerate(questions.items()):
            loc = ("body", "questions", qid, "criteria")
            kind = q["type"]
            if kind == "noul":
                crit = q.get("criteria") or {}
                choices = [("yes", text_of(crit.get("true"))), ("no", text_of(crit.get("false")))]
                labels = ["yes", "no"]
            elif kind == "choice":
                crit = q["criteria"]
                if not crit:
                    raise SchemaError(f"Choice question must have at least one choice: {qid}", loc)
                if len(crit) == 1:
                    only = next(iter(crit))
                    forced[qid] = {"type": "choice", "choice": only, "probabilities": {only: 1.0}, "confidence": 1.0}
                    continue
                if len(crit) > len(self.choice_labels):
                    raise SchemaError(f"Too many choices. Must have at most {len(self.choice_labels)} choices.", loc)
                choices = [(name, text_of(desc)) for name, desc in crit.items()]
                labels = self.choice_labels[: len(choices)]
            elif kind == "score":
                crit = q["criteria"]
                if len(crit) > 10:
                    raise SchemaError("Too many score levels. Must have at most 10 levels.", loc)
                if len(crit) == 1:  # one level, one possible score, as on Jev
                    forced[qid] = {"type": "score", "score": 0.0, "legend": {"0": crit[0]},
                                   "probabilities": {"0": 1.0}, "confidence": 1.0}
                    continue
                choices = [(str(i), text_of(c)) for i, c in enumerate(crit)]
                labels = [str(i) for i in range(len(crit))]
            else:  # the request model rejects this first
                raise SchemaError(f"unknown question type {kind!r}", ("body", "questions", qid, "type"))
            qs.append({"key": qid, "id": f"q{i + 1}", "type": kind, "instructions": text_of(q.get("instructions")),
                       "choices": choices, "labels": labels,
                       # score legends echo the criteria exactly as sent
                       "legend": list(q["criteria"]) if kind == "score" else None})
        for i, q in enumerate(qs):  # the model sees q1, q2, ... with the forced ones left out
            q["id"] = f"q{i + 1}"
        return {"questions": qs, "forced": forced, "format": "lines" if len(qs) <= 10 else "indexed"}

    def system_text(self, qs, fmt, chunked=False):
        s = ("Answer a fixed set of questions about the state the user provides. "
             "Each question lists its allowed answers; reply with exactly one label per question.\n")
        for q in qs:
            s += f"\nQuestion {q['id']}: {q['instructions'] or 'Answer about the state.'}\n"
            for (name, desc), label in zip(q["choices"], q["labels"]):
                if q["type"] == "noul":
                    s += f"  {label}: {desc}\n" if desc else f"  {label}\n"
                elif q["type"] == "score":
                    s += f"  {label}: {desc}\n"
                else:
                    s += f"  {label}: {name} ({desc})\n" if desc else f"  {label}: {name}\n"
        s += "\n" + FORMATS[fmt][2]
        if chunked:
            s += " A reply may cover only some of the questions; answer every line that is present."
        return s

    def answer_text(self, qs, labels, fmt):
        join, lead, _ = FORMATS[fmt]
        return join.join(lead.format(id=q["id"]) + q["labels"][l] for q, l in zip(qs, labels))

    def resolve_template(self, qs, fmt, head=None, lead=""):
        """Tokenize the answer template and find each question's slot. Every
        label must change exactly one token, at the same position for all of a
        question's labels. ``head`` is the token run the canvas starts with:
        the empty thought block for a plain read, nothing when the prompt
        already closes the thought. ``lead`` is the text before the first
        answer when earlier answers are already in the prompt."""
        head = self.scaffold if head is None else head
        key = json.dumps([fmt, head, lead] + [(q["id"], q["labels"]) for q in qs])
        hit = self._templates.get(key)
        if hit:
            return hit
        base_labels = [0] * len(qs)
        base = head + self.enc(lead + self.answer_text(qs, base_labels, fmt))
        if len(base) + 1 > self.s.canvas:
            raise SchemaError(f"answer template is {len(base)} tokens; the canvas holds {self.s.canvas - 1}")
        slots = []
        for qi, q in enumerate(qs):
            pos = None
            ids = [0] * len(q["labels"])
            for li in range(1, len(q["labels"])):
                labels = list(base_labels)
                labels[qi] = li
                e = head + self.enc(lead + self.answer_text(qs, labels, fmt))
                diffs = [i for i in range(min(len(e), len(base))) if e[i] != base[i]]
                if len(e) != len(base) or len(diffs) != 1 or (pos is not None and diffs[0] != pos):
                    raise SchemaError(f"question {q['key']!r}: labels do not share one template slot")
                pos = diffs[0]
                ids[li] = e[pos]
            ids[0] = base[pos]
            slots.append({"pos": pos, "label_ids": ids})
        if len(self._templates) > 4096:
            self._templates.clear()
        self._templates[key] = (base, slots)
        return base, slots

    def groups(self, qs, fmt):
        """Split questions, in order, into the fewest groups whose answer
        templates fit the canvas."""
        out, group = [], []
        for q in qs:
            trial = group + [q]
            rows = len(self.scaffold) + len(self.enc(self.answer_text(trial, [0] * len(trial), fmt))) + 1
            if rows > self.s.canvas and group:
                out.append(group)
                group = [q]
            else:
                group = trial
        out.append(group)
        return out

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def canvas_width(self, template):
        need = len(template) + 1
        step = self.s.canvas_step
        return min(self.s.canvas, -(-need // step) * step)

    def build_canvas(self, template, slots, seed):
        rng = random.Random(seed)
        canvas = list(template) + [TURN_CLOSE]
        canvas += [PAD] * (self.canvas_width(template) - len(canvas))
        for s in slots:
            canvas[s["pos"]] = rng.randrange(VOCAB)
        return canvas

    async def _post(self, path, body):
        async with self.slots:
            r = await self.client.post(path, json=body)
        if 400 <= r.status_code < 500:
            try:
                msg = r.json().get("error", {}).get("message") or r.json().get("message") or r.text
            except ValueError:
                msg = r.text
            raise Upstream(str(msg)[:500])
        r.raise_for_status()
        return r.json()

    def _xargs(self, template, slots, seed, steps):
        return {"diffusion_seed_canvas": self.build_canvas(template, slots, seed),
                "diffusion_canvas_length": self.canvas_width(template),
                "diffusion_max_steps": steps, "diffusion_read_only": True}

    async def one_read(self, template, slots, sys_text, content, seed, steps=1, prefix=None):
        """One read-only denoise over a seeded canvas. ``content`` is the user
        turn: the state text, or image parts followed by it. With ``prefix``
        (prompt token ids that already hold a thought or earlier answers) the
        read continues that prompt through the completions endpoint instead."""
        label_ids = sorted({i for s in slots for i in s["label_ids"]})[:MAX_LABEL_IDS]
        if prefix is not None:
            d = await self._post("/v1/completions", {
                "model": self.s.upstream_model, "prompt": prefix, "max_tokens": len(template) + 1,
                "logprobs": TOPK, "logprob_token_ids": label_ids, "return_tokens_as_token_ids": True,
                "vllm_xargs": self._xargs(template, slots, seed, steps)})
            rows = d["choices"][0]["logprobs"]["top_logprobs"]
            tops = [{int(k.split(":")[1]): v for k, v in rows[s["pos"]].items()} for s in slots]
        else:
            d = await self._post("/v1/chat/completions", {
                "model": self.s.upstream_model,
                "messages": [{"role": "system", "content": sys_text}, {"role": "user", "content": content}],
                "max_tokens": len(template) + 1,
                "logprobs": True,
                "top_logprobs": TOPK,
                # exact logprobs for every label at every position; long option
                # lists rarely rank inside the top-k
                "logprob_token_ids": label_ids,
                "return_tokens_as_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": False},
                "vllm_xargs": self._xargs(template, slots, seed, steps)})
            rows = d["choices"][0]["logprobs"]["content"]
            tops = [{int(t["token"].split(":")[1]): t["logprob"] for t in rows[s["pos"]]["top_logprobs"]} for s in slots]
        out = [slot_distribution(top, s["label_ids"]) for top, s in zip(tops, slots)]
        return out, d.get("usage", {}).get("prompt_tokens", 0)

    def chat_prompt_ids(self, sys_text, state_text, thinking=False):
        """The prompt the chat endpoint would build, as token ids, ending after
        the model turn marker. Text states only."""
        messages = [{"role": "system", "content": sys_text}, {"role": "user", "content": state_text}]
        out = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=thinking)
        ids = out["input_ids"] if hasattr(out, "keys") else out  # newer transformers return a dict
        return [int(t) for t in ids]

    async def think(self, sys_text, state_text, budget):
        """A read prefix ending a thought the model wrote: the chat prompt with
        thinking on, the open tag, up to ``budget`` generated tokens, the close
        tag. Returns (prefix ids, thought token count, prompt tokens processed).
        Billing follows the compute: the thought pass reads the whole input,
        and the read after it reads the input again plus the thought."""
        prompt = self.chat_prompt_ids(sys_text, state_text, thinking=True) + self.thought_open
        d = await self._post("/v1/completions", {
            "model": self.s.upstream_model, "prompt": prompt, "max_tokens": budget, "logprobs": 0,
            "return_tokens_as_token_ids": True, "stop_token_ids": self.thought_close})
        ids = [int(t.split(":")[1]) for t in d["choices"][0]["logprobs"]["tokens"]]
        if self.thought_close[0] in ids:
            ids = ids[: ids.index(self.thought_close[0])]
        return prompt + ids + self.thought_close, len(ids), d.get("usage", {}).get("prompt_tokens", len(prompt))

    async def read_group(self, qs, fmt, sys_text, content, seed, opts, prefix=None, lead=""):
        """The reads for one group of questions, averaged. Returns (mean label
        probabilities per question, billed input tokens, thought tokens)."""
        thought = think_input = 0
        if prefix is None and opts["think"]:
            prefix, thought, think_input = await self.think(sys_text, content, opts["think"])
        template, slots = self.resolve_template(qs, fmt, head=None if prefix is None else [], lead=lead)
        steps = opts["steps"]

        async def read(k):
            return await self.one_read(template, slots, sys_text, content, seed + k * 7919, steps, prefix)

        if opts["samples"]:
            results = await asyncio.gather(*[read(k) for k in range(opts["samples"])])
            reads = [r for r, _ in results]
            billed = sum(t for _, t in results)
        else:
            first, billed = await read(0)
            reads = [first]
            # re-reads are the server's own policy and are not billed
            if self.s.auto_max > 1 and max(r["entropy"] for r in first) > self.s.auto_threshold:
                more = await asyncio.gather(*[read(k) for k in range(1, self.s.auto_max)])
                reads += [m[0] for m in more]
        means = [[sum(r[qi]["probs"][l] for r in reads) / len(reads) for l in range(len(q["labels"]))]
                 for qi, q in enumerate(qs)]
        return means, billed + think_input, thought

    async def decide(self, questions, state, seed, images=None, options=None):
        """Answer a Jev request. ``images`` are OpenAI-style image parts that go
        ahead of the state. Returns (answers keyed by question id, billed input
        tokens, thought tokens)."""
        opts = dict(DEFAULT_OPTIONS, **{k: v for k, v in (options or {}).items() if v is not None})
        if images and (opts["think"] or opts["sequential"]):
            field = "think" if opts["think"] else "sequential"
            raise SchemaError(f"{field} needs a text state; send images without it", ("body", field))
        if self.waiting >= self.s.max_queue:
            raise Overloaded("OpenJev is at capacity. Retry shortly.")
        self.waiting += 1
        try:
            schema = self.build_schema(questions)
            fmt = schema["format"]
            state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
            content = list(images) + [{"type": "text", "text": state_text}] if images else state_text
            groups = self.groups(schema["questions"], fmt) if schema["questions"] else []
            if not groups:  # every question was answered without a read
                results = []
            elif opts["sequential"] and len(groups) > 1:
                results = await self._sequential(groups, fmt, schema["questions"], state_text, seed, opts)
            else:
                chunked = len(groups) > 1
                results = await asyncio.gather(*[
                    self.read_group(g, fmt, self.system_text(g, fmt, chunked), content, seed + 104729 * k, opts)
                    for k, g in enumerate(groups)])
        finally:
            self.waiting -= 1
        answers, billed, thought = dict(schema["forced"]), 0, 0
        for g, (means, group_billed, group_thought) in zip(groups, results):
            billed += group_billed
            thought += group_thought
            for q, mean in zip(g, means):
                answers[q["key"]] = to_answer(q, mean)
        answers = {k: answers[k] for k in questions}  # answered in the order asked
        return answers, billed, thought

    async def _sequential(self, groups, fmt, all_qs, state_text, seed, opts):
        """Chunks continue one answer in order under the full question list.
        Each chunk's chosen labels are written into the prompt before the next
        read, so later answers condition on earlier ones."""
        sys_text = self.system_text(all_qs, fmt)
        thought = think_input = 0
        if opts["think"]:
            base_ids, thought, think_input = await self.think(sys_text, state_text, opts["think"])
        else:
            base_ids = self.chat_prompt_ids(sys_text, state_text) + self.scaffold
        join = FORMATS[fmt][0]
        lines, results = [], []
        for k, group in enumerate(groups):
            if lines:
                prefix, lead = base_ids + self.enc(join.join(lines)), join
            else:
                prefix, lead = (base_ids if opts["think"] else None), ""
            means, billed, _ = await self.read_group(group, fmt, sys_text, state_text, seed + 104729 * k,
                                                     dict(opts, think=0), prefix, lead)
            results.append((means, billed + (think_input if k == 0 else 0), thought if k == 0 else 0))
            chosen = [max(range(len(m)), key=m.__getitem__) for m in means]
            lines.append(self.answer_text(group, chosen, fmt))
        return results


def slot_distribution(top, label_ids):
    """Label probabilities at one slot. Read-only logprobs are at temperature
    1, so the label softmax uses them directly. The entropy is over the
    returned top-k set and drives the re-read policy."""
    floor = min(top.values()) - 5.0
    lp = [top.get(i, floor) for i in label_ids]
    mx = max(lp)
    ex = [math.exp(x - mx) for x in lp]
    z = sum(ex)
    top_p = [math.exp(v) for v in top.values()]
    return {"probs": [e / z for e in ex], "entropy": -sum(p * math.log(p) for p in top_p if p > 0)}


def confidence(p):
    """How peaked a distribution is: 1 - H(p)/ln(K). 1 is certain, 0 uniform."""
    k = len(p)
    h = -sum(x * math.log(x) for x in p if x > 0)
    return max(0.0, min(1.0, 1.0 - h / math.log(k)))


def to_answer(q, p):
    """Jev's answer shapes."""
    if q["type"] == "noul":
        return {"type": "noul", "noul": p[0]}
    if q["type"] == "choice":
        top = max(range(len(p)), key=p.__getitem__)
        return {"type": "choice", "choice": q["choices"][top][0],
                "probabilities": {c[0]: v for c, v in zip(q["choices"], p)}, "confidence": confidence(p)}
    return {"type": "score", "score": sum(i * v for i, v in enumerate(p)),
            "legend": {str(i): q["legend"][i] for i in range(len(p))},
            "probabilities": {str(i): v for i, v in enumerate(p)}, "confidence": confidence(p)}
