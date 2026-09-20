# OpenJev

**Fast, calibrated, typed decisions from an open model.** OpenJev is an open-source
"System One" decision server: send it a state and a set of typed questions (yes/no,
choice, score) and get back probabilities and a confidence for every answer, in tens of
milliseconds. Answers are read straight off the model's probabilities, with no text generated
and nothing parsed, so they cannot be hallucinated off-schema. Questions can be about images too.

It speaks the same wire API as TypeSafe's [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
so their SDKs work against it unchanged. It runs on
[DiffusionGemma 26B-A4B](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4)
(Apache-2.0) through vLLM. The same loaded model also serves ordinary text generation on an
OpenAI-compatible `/v1/chat/completions`, at no extra GPU memory.

> **Hosted for free on [Codiv](https://codiv.ai)**, an inference platform for open System One
> models: sign up and get 100M input tokens, no card required. `https://api.codiv.ai/v1/systemone`

OpenJev is an independent project. It is not affiliated with or endorsed by TypeSafe AI.

## Try it

```bash
pip install typesafe-sdk
export TYPESAFE_BASE_URL=https://api.codiv.ai   # or http://127.0.0.1:8080 for your own server
export TYPESAFE_API_KEY=sk-codiv-...
```

```python
from typesafe_sdk import TypeSafeClient

client = TypeSafeClient()
r = client.system_one(
    "Everything is down and we have a demo with our biggest client at noon.",
    {
        "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
        "team":   {"type": "choice", "instructions": "Which team should handle it?",
                   "criteria": {"outage": "service down", "billing": "charges, refunds", "feature": "requests, how-to"}},
        "tone":   {"type": "score", "instructions": "How upset is the customer?",
                   "criteria": ["calm", "annoyed", "furious"]},
    },
)
r.nouls["urgent"].noul        # 1.00
r.choices["team"].choice      # "outage", confidence 1.00
r.scores["tone"].score        # 2.00 (expected level, 0-indexed)
```

Or with curl:

```bash
curl https://api.codiv.ai/v1/systemone \
  -H "Authorization: Bearer $TYPESAFE_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "openjev-latest", "state": "I was charged twice this month.",
       "questions": {"is_billing": {"type": "noul", "instructions": "Is this a billing issue?"}}}'
```

## API

| | |
|---|---|
| `POST /v1/systemone` | `{state, model, questions}` → `{model, answers, usage}` |
| `POST /v1/chat/completions` | OpenAI-style text generation with model `diffusiongemma-26b` ([below](#text-generation)) |
| `GET /v1/models` | `openjev-0.1` and its alias `openjev-latest`. `jev-latest` and `jev-preview` are also accepted, so TypeSafe SDK defaults work. Also lists `diffusiongemma-26b`. |

Question types:

- **`noul`** (yes/no): takes optional `criteria: {true, false}` and returns `{noul: P(yes)}`.
- **`choice`**: takes `criteria: {name: description}` and returns `{choice, probabilities, confidence}`.
- **`score`**: takes `criteria: [level0, level1, …]` (2–10 levels) and returns `{score: Σ i·pᵢ, legend, probabilities, confidence}`.

`confidence` is `1 − H(p)/ln K`. It is 1 when the model is certain and 0 when the distribution is uniform.
`usage.input_tokens` counts prompt tokens, including image tokens. `usage.output_tokens` is 0
unless `think` is set (below). Errors follow the
same shapes as Jev, checked against the live API: FastAPI `422` validation lists for a field of the
wrong shape, `400` with the reason as plain text for a question that cannot be asked (no options,
too many options or score levels), `400` `api_usage_error` for an unknown model or question type,
`{"detail": {"error_type", "message"}}` for auth errors (`401`/`403`), `429`, and `529` when overloaded.

Known differences from Jev:
- A choice can have at most 128 options (Jev allows 255); the refusal reads the same.
- Model names are OpenJev's own; `jev-latest` and `jev-preview` are aliases, and a pinned Jev version
  such as `jev-1.13.0` answers `400` `Unknown model`.
- A choice with one option, or a score with one level, is answered directly (probability 1) without a
  read, so it bills no tokens. Jev answers the same values and bills for the read.
- A body nested a thousand levels deep answers `422`; Jev answers `500`.
- Many questions are answered in chunks of about 12 per read. They are still answered in parallel.

### Extensions

These optional request fields are OpenJev additions to Jev's contract. Leave them out and a request
behaves exactly like Jev; TypeSafe's SDKs never send them. They come from the example server in
vllm-project/vllm#57250.

| Field | Values | What it does | Cost |
|---|---|---|---|
| `images` | up to 8 | Images the questions are about, placed ahead of the state. Each is a `data:image/...;base64,` URL or `{"content_type", "base64"}`; JPEG, PNG, WebP or GIF, 5 MB each. | about 280 input tokens per image |
| `steps` | 1–8, default 1 | Denoise steps per read. More steps let the answers settle against each other. | same tokens, more GPU time |
| `samples` | 1–32 | Read N times with different noise and average. Replaces the automatic re-reads. | N × input tokens |
| `think` | 0–4096 tokens | Let the model write a thought first, then read the answers after it. The number is a hard cap: a thought that hits it is cut off, so give multi-step problems 512 or more. | input tokens twice, plus the thought as output tokens |
| `sequential` | `true` | For long question lists answered in chunks: read the chunks in order, each seeing the answers already chosen. | one read per chunk, run one after another |

```bash
curl https://api.codiv.ai/v1/systemone \
  -H "Authorization: Bearer $TYPESAFE_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "openjev-latest", "state": "Look at the photo.",
       "images": ["data:image/jpeg;base64,/9j/4AAQ..."],
       "questions": {"hotdog": {"type": "noul", "instructions": "The photo shows a hot dog"}}}'
```

`think` and `sequential` need a text state, so they cannot be combined with `images` (the request
gets a 400).

### Text generation

`POST /v1/chat/completions` generates text with the same DiffusionGemma, OpenAI style, so tools that
talk to a chat model can point their base URL at OpenJev. Use model `diffusiongemma-26b`.

vLLM refuses some fields for diffusion models, so OpenJev adjusts requests instead of failing them:
- `temperature`, `seed`, `min_p`, `logit_bias`, penalties and `reasoning` are ignored.
- `response_format` (`json_object` or `json_schema`) becomes an instruction to reply with JSON only,
  and the first JSON object in the reply is returned as the content.
- `max_tokens` defaults to 1024 and is capped at 8192. Thinking is off unless you set
  `chat_template_kwargs: {"enable_thinking": true}`; the thought then comes back separately in
  `message.reasoning`, never in `content`.
- `stream: true` streams server-sent events and always ends with a usage chunk. Tools
  (`tools`, `tool_choice`) are supported.

Generation denoises the output in 64-token blocks, so it costs far more GPU time than a System One
read. At most 8 generations run at once, so they never crowd out reads.

```python
from openai import OpenAI

client = OpenAI(base_url="https://api.codiv.ai/v1", api_key="sk-codiv-...")
r = client.chat.completions.create(
    model="diffusiongemma-26b",
    messages=[{"role": "user", "content": "Summarize: 3 nonstop flights, cheapest $212 on Delta."}],
)
print(r.choices[0].message.content)
```

## How it works

DiffusionGemma is a discrete diffusion model: it denoises a whole canvas of tokens per
forward pass instead of generating left to right. OpenJev writes the answer template
onto the canvas, for example:

```
q1: ▒
q2: ▒
q3: ▒
```

Only the label slots are left as noise. One read-only denoise step then gives a full
probability distribution over each question's labels, and those distributions are the
answers. If any slot is uncertain (entropy > 0.1), OpenJev re-reads with fresh noise up to
four times and averages the results. Question ids never reach the model.

The vLLM side of this is [vllm-project/vllm#57250](https://github.com/vllm-project/vllm/pull/57250),
which adds seeded canvases, read-only steps and step caps for DiffusionGemma. `openjev/engine.py` is
adapted from that PR's `structured_server.py` example, with async I/O, bounded concurrency and
backpressure added.

## Run your own

You need an NVIDIA GPU with at least 24 GB of memory for the NVFP4 checkpoint (tested on an RTX PRO 6000 Blackwell, sm_120).

A prebuilt image is on Docker Hub, so there is nothing to compile.
[`razorback16/openjev`](https://hub.docker.com/r/razorback16/openjev) runs vLLM with PR #57250
(CUDA 13, [pin below](#caveats)) and the Jev-compatible API server in one container:

```bash
git clone https://github.com/razorback16/openjev && cd openjev
docker compose up -d          # OpenJev on 127.0.0.1:8080 once the model has loaded
curl localhost:8080/v1/models
```

Or without compose:

```bash
docker run -d --gpus all --ipc=host -p 127.0.0.1:8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface razorback16/openjev:0.2.0
```

The model weights (about 18 GB) download on first start into `~/.cache/huggingface`.
Use `docker compose build` to build the image yourself instead. vLLM listens only inside the
container; set `OPENJEV_UPSTREAM` to skip it and use a vLLM server you already run.

Settings are read from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `OPENJEV_UPSTREAM` | unset | external vLLM server URL; when set, the container does not start its own |
| `OPENJEV_MODEL` | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` | weights the built-in vLLM serves |
| `OPENJEV_GPU_UTIL` | `0.9` | vLLM `--gpu-memory-utilization` |
| `OPENJEV_MAX_NUM_SEQS` | `64` | vLLM `--max-num-seqs` |
| `OPENJEV_MAX_MODEL_LEN` | `65536` | vLLM `--max-model-len` |
| `OPENJEV_VLLM_ARGS` | unset | extra `vllm serve` flags |
| `OPENJEV_CANVAS` | `64` | canvas length; also sets the built-in vLLM's `--diffusion-config` |
| `OPENJEV_MAX_INFLIGHT` | `64` | reads in flight to vLLM |
| `OPENJEV_MAX_QUEUE` | `512` | waiting decisions before the server returns 529 |
| `OPENJEV_API_KEY` | unset | require `Authorization: Bearer <key>` |
| `OPENJEV_ORIGIN_SECRET` | unset | require an `X-Origin-Secret` header (for use behind a proxy) |
| `OPENJEV_MAX_IMAGES` | `8` | images per request; also sets the built-in vLLM's `--limit-mm-per-prompt` |
| `OPENJEV_MAX_IMAGE_BYTES` | `5242880` | size limit per image, after base64 decoding |
| `OPENJEV_GEN_MAX_INFLIGHT` | `8` | text generations running at once |
| `OPENJEV_GEN_MAX_QUEUE` | `32` | waiting generations before the server returns 529 |
| `OPENJEV_GEN_MAX_TOKENS` | `8192` | cap on `max_tokens` for text generation |
| `OPENJEV_WARMUP` | `1` | `0` skips the warmup requests sent before the API opens (they save the first users several seconds of compiling) |

Measured on an RTX PRO 6000 using 38% of the GPU, with 3 questions per request and cache-busted states:

| Concurrency | req/s | p50 | p95 |
|---:|---:|---:|---:|
| 1 | 10.7 | 94 ms | 94 ms |
| 16 | 43.3 | 367 ms | 369 ms |
| 32 | 51.7 | 545 ms | 618 ms |
| 64 | 57.4 | 760 ms | 1109 ms |

Without Docker:

```bash
git clone https://github.com/razorback16/vllm -b structured-reads-54309 && cd vllm
VLLM_USE_PRECOMPILED=1 \
  VLLM_PRECOMPILED_WHEEL_COMMIT=2c88fb131c7ae0be01907cd8c276911db5e7aad4 pip install -e .
vllm serve nvidia/diffusiongemma-26B-A4B-it-NVFP4 --served-model-name dgemma \
  --diffusion-config '{"canvas_length": 64}' --max-logprobs 32 --enable-prefix-caching \
  --async-scheduling --attention-backend TRITON_ATTN \
  --limit-mm-per-prompt '{"image": 8, "video": 0}' \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --override-generation-config '{"max_new_tokens": null}'
pip install -e path/to/openjev && python -m openjev
```

## Caveats

- vllm-project/vllm#57250 has not been merged. The request fields it uses (`vllm_xargs`) are
  provisional, so this project pins a commit
  ([`razorback16/vllm` branch `structured-reads-54309`](https://github.com/razorback16/vllm/tree/structured-reads-54309)):
  the PR's head plus fixes that keep vLLM's engine from crashing:
  - vllm-project/vllm#54309, for any request with an image (vllm-project/vllm#56712);
  - the sampler step fell back to eager code with a dtype mismatch once torch.compile hit its
    recompile limit, breaking multi-step reads and text generation;
  - logprobs stashed at different steps could not be joined when reads and generations shared a
    batch.
- Answer quality is the quality of DiffusionGemma 26B-A4B used in this mode. Evaluate it on your
  own tasks before relying on it.

## Development

```bash
pip install -e '.[test]' && pytest
```

## License

Apache-2.0. The DiffusionGemma weights are Apache-2.0 (NVIDIA / Google).
