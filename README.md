# OpenJev

**Fast, calibrated, typed decisions from an open model.** OpenJev is an open-source
"System One" decision server. Send it a state and typed questions (yes/no, choice, score). It
returns a probability and a confidence for each answer in tens of milliseconds. It reads the
answers directly from the model's probabilities and parses no text, so an answer cannot go
off-schema. Questions can also ask about images.

OpenJev uses the same wire API as TypeSafe's
[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), so their SDKs work with it
unchanged. It runs
[DiffusionGemma 26B-A4B](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4)
(Apache-2.0) through vLLM on an NVIDIA GPU, or through MLX on Apple silicon.

> **Hosted for free on [Codiv](https://codiv.ai)**, an inference platform for open System One
> models. Sign up and get 100M input tokens, no card required. `https://api.codiv.ai/v1/systemone`

OpenJev is an independent project. It is not affiliated with or endorsed by TypeSafe AI.

## Models

| Model id | Model | Size | Input | Choices | Runs on |
|---|---|---|---|---|---|
| `openjev-latest` (`openjev-0.1`) | [DiffusionGemma 26B-A4B](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4) (NVIDIA / Google), read as a diffusion canvas | 26B total, 4B active | text and images | up to 255 | vLLM (NVIDIA GPU) or MLX (Apple silicon) |
| `laya-1.0` | [Laya](https://github.com/NandhaKishorM/laya) by Nandakishor M / Convai Innovations | 421M | text, 1,024 tokens | up to 255 | PyTorch, GPU or CPU |
| `verdict-1.4` | [Verdict](https://github.com/Heman10x-NGU/Verdict-open-jev) by Heman10x | 151M | text, 512 tokens | up to 24 | PyTorch, GPU or CPU |
| `clm-v0.1` | [CLM](https://github.com/Contrastive-LM/CLM) by Contrastive-LM: contrastive heads over Qwen3-8B | 8B + 2 × 9.4M | text, 2,048 tokens | up to 255 | vLLM (NVIDIA GPU) |
| `jevk5-0.2` | [JevK5](https://github.com/allebee/jevk5) by Alibi Serikbay: Qwen3.5-4B with a distilled LoRA, read by its answer letters | 4B | text, 16,384 tokens | up to 255 | vLLM (NVIDIA GPU) |

`diffusiongemma-26b` is the same DiffusionGemma for [text generation](#text-generation). All
weights are Apache-2.0. Laya, Verdict, CLM and JevK5 are other people's models: see
[Small encoder models](#small-encoder-models), [CLM](#clm) and [JevK5](#jevk5) for details and credit.

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
| `GET /v1/models` | `openjev-0.1`, its alias `openjev-latest`, `diffusiongemma-26b`, and the [small encoder models](#small-encoder-models) when they run. The server also accepts `jev-latest` and `jev-preview`, so TypeSafe SDK defaults work. |

Question types:

- **`noul`** (yes/no): takes optional `criteria: {true, false}`. Returns `{noul: P(yes)}`.
- **`choice`**: takes `criteria: {name: description}`. Returns `{choice, probabilities, confidence}`.
- **`score`**: takes `criteria: [level0, level1, …]` (2–10 levels). Returns `{score: Σ i·pᵢ, legend, probabilities, confidence}`.

`confidence` is `1 − H(p)/ln K`: 1 when the model is certain, 0 when the distribution is
uniform. `usage.input_tokens` counts prompt tokens, image tokens included.
`usage.output_tokens` is 0 unless you set `think`.

Each response has a `Server-Timing` header:

```
server-timing: model;dur=41.2, server;dur=2.8, total;dur=44.0
```

`model` is the time spent on the model, summed over the request's reads. It can be more than
`total` when reads run in parallel. `server` is the remaining time: schema compile,
tokenization, validation and serialization. Neither includes your network.

Errors use the same shapes as Jev, checked against the live API:

- `422` with a FastAPI validation list for a field of the wrong shape.
- `400` with a plain-text reason for a question the server cannot ask (no options, too many
  options, or too many score levels).
- `400` `api_usage_error` for an unknown model or question type.
- `{"detail": {"error_type", "message"}}` for auth errors (`401`/`403`).
- `429` for rate limits. `529` when the server is overloaded.

Differences from Jev:

- Model names are OpenJev's own. `jev-latest` and `jev-preview` are aliases. A pinned Jev
  version such as `jev-1.13.0` gets `400` `Unknown model`.
- The server reads many questions in chunks of about 12 per read. The chunks run in parallel.

### Extensions

These optional request fields are OpenJev additions. A request without them behaves exactly
like Jev. TypeSafe's SDKs never send them. They come from the example server in
vllm-project/vllm#57250.

| Field | Values | What it does | Cost |
|---|---|---|---|
| `images` | up to 8 | Images for the questions, placed before the state. Each is a `data:image/...;base64,` URL or `{"content_type", "base64"}`. JPEG, PNG, WebP or GIF, 5 MB each. | about 280 input tokens per image |
| `steps` | 1–8, default 1 | Denoise steps per read. More steps let the answers settle against each other. | same tokens, more GPU time |
| `samples` | 1–32 | Read N times with different noise and average. This replaces the automatic re-reads. `samples: 1` gives one read, the fastest answer. | N × input tokens |
| `think` | 0–4096 tokens | The model writes a thought, then reads the answers after it. The number is a hard cap, and a longer thought is cut. Give multi-step problems 512 or more. | input tokens twice, plus the thought as output tokens |
| `sequential` | `true` | For long question lists: read the chunks in order. Each chunk sees the answers before it. | one read per chunk, in series |

```bash
curl https://api.codiv.ai/v1/systemone \
  -H "Authorization: Bearer $TYPESAFE_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "openjev-latest", "state": "Look at the photo.",
       "images": ["data:image/jpeg;base64,/9j/4AAQ..."],
       "questions": {"hotdog": {"type": "noul", "instructions": "The photo shows a hot dog"}}}'
```

`think` and `sequential` need a text state. A request that combines them with `images` gets a 400.

### Text generation

`POST /v1/chat/completions` generates text with the same model, OpenAI style. Use model
`diffusiongemma-26b`. It supports streaming and tools. The server ignores sampling fields
(`temperature`, `seed`, penalties and similar) because vLLM refuses them for diffusion models.
`max_tokens` defaults to 1024, with a cap of 8192. At most 8 generations run at once, so reads
always have room. The MLX backend ignores `tools` and `logprobs` and does not return the thought.

## How it works

DiffusionGemma is a discrete diffusion model. It denoises a full canvas of tokens in each
forward pass, instead of writing left to right. OpenJev uses this to read answers, not write them.

OpenJev builds a canvas in which only the answer slots are masked, one token per question:

```
canvas in                 one read-only pass         answer out
  q1: [?]        ──►      P(yes) 0.001        ──►    noul  0.001
  q2: [?]                 P(A) 0.000                 choice "billing"
                          P(B) 0.999                 confidence 0.997
                          P(C) 0.000
  q3: [?]                 P(0) 0.000                 score 1.00
                          P(1) 0.996
                          P(2) 0.004
```

Each label is one token: `yes`/`no` for a `noul`, `A`/`B`/`C` for a choice, `0`/`1`/`2` for a
score. The model never writes into these slots. One read-only pass gives the probability
distribution for each slot, and that distribution **is** the answer. The numbers above are a
real read of "The invoice looks wrong again. Second time this quarter.": not urgent, billing,
mildly annoyed.

The read scores only the label tokens, so an answer cannot go off-schema. The confidence comes
from the model's own distribution, not from a number that the model reports about itself.

If a slot is uncertain (entropy > 0.1), OpenJev reads three more times with fresh noise and
averages the four results. One uncertain question causes a re-read of all the questions in the
request. These extra reads add no tokens to `usage`. Set `samples: 1` to get one read only.
Question ids never go to the model. It sees `q1`, `q2`, `q3`.

The vLLM part is [vllm-project/vllm#57250](https://github.com/vllm-project/vllm/pull/57250),
merged on 2026-09-22. It adds seeded canvases, read-only steps, step caps and pinned canvas
positions for DiffusionGemma. `openjev/engine.py` adapts that PR's `structured_server.py`
example, with async I/O, bounded concurrency and backpressure.

## Run your own

Two backends serve the same `/v1/systemone`. Select one by hardware:

| | vLLM (default) | MLX |
|---|---|---|
| Hardware | NVIDIA GPU, 24 GB or more | Apple silicon, about 16 GB free |
| Setup | Docker image | `pip install -e '.[mlx]'` |
| Reads | up to 64 in flight | one at a time |
| `images`, `steps` > 1, `think` | yes | yes |
| Text generation | yes | yes, streaming included |

### NVIDIA GPU

You need an NVIDIA GPU with at least 24 GB of memory for the NVFP4 checkpoint. We tested on an
RTX PRO 6000 Blackwell (sm_120).

The prebuilt image [`razorback16/openjev`](https://hub.docker.com/r/razorback16/openjev) runs
vLLM and the API server in one container, on CUDA 13. It pins the upstream vLLM commit below,
with the two changes in [Caveats](#caveats).

```bash
git clone https://github.com/razorback16/openjev && cd openjev
docker compose up -d          # OpenJev on 127.0.0.1:8080 when the model is loaded
curl localhost:8080/v1/models
```

Or without compose:

```bash
docker run -d --gpus all --ipc=host -p 127.0.0.1:8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface razorback16/openjev:0.5.0
```

The first start downloads the weights (about 18 GB) into `~/.cache/huggingface`. To build the
images yourself, build the shared base first, then run `docker compose build`:

```bash
docker build -f docker/Dockerfile.base -t razorback16/openjev-base:cu130-torch2.13 .
docker compose build
```

vLLM listens only inside the container.
Set `OPENJEV_UPSTREAM` to use a vLLM server that you already run.

Measured on an RTX PRO 6000 at 38% of the GPU, with 3 questions per request and cache-busted
states:

| Concurrency | req/s | p50 | p95 |
|---:|---:|---:|---:|
| 1 | 10.7 | 94 ms | 94 ms |
| 16 | 43.3 | 367 ms | 369 ms |
| 32 | 51.7 | 545 ms | 618 ms |
| 64 | 57.4 | 760 ms | 1109 ms |

One request at a time, on the same GPU, with `samples: 1`:

| Request | p50 | p95 |
|---|---:|---:|
| 1 question | 27 ms | 28 ms |
| 3 questions | 31 ms | 32 ms |

Without Docker:

```bash
git clone https://github.com/vllm-project/vllm && cd vllm
VLLM_COMMIT=1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8   # the commit the image pins
git checkout $VLLM_COMMIT
# a choice of more than 128 options needs a larger cap, as in the image
sed -i 's/^MAX_LOGPROB_TOKEN_IDS = 128$/MAX_LOGPROB_TOKEN_IDS = 512/' vllm/sampling_params.py
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=$VLLM_COMMIT pip install -e .
vllm serve nvidia/diffusiongemma-26B-A4B-it-NVFP4 --served-model-name dgemma \
  --diffusion-config '{"canvas_length": 64}' --max-logprobs 32 --enable-prefix-caching \
  --async-scheduling --attention-backend TRITON_ATTN \
  --limit-mm-per-prompt '{"image": 8, "video": 0}' \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --override-generation-config '{"max_new_tokens": null}'
pip install -e path/to/openjev && python -m openjev
```

To match the image's image reads, also apply `docker/patches/vision_prefix_lm.py` to this
checkout. Without it, vLLM prefills images causally.

### Apple silicon

A Mac needs no vLLM and no Docker. `OPENJEV_BACKEND=mlx` runs DiffusionGemma inside the OpenJev
process through [MLX](https://github.com/ml-explore/mlx) and
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm). The 4-bit weights need about 16 GB of memory.

```bash
pip install -e '.[mlx]'
OPENJEV_BACKEND=mlx python -m openjev     # 127.0.0.1:8080
```

This backend uses the same prompts, canvases and seeds as vLLM. It supports `images`,
`samples`, `sequential`, `steps`, `think` and the automatic re-reads, and it bills the same.
Each step after the first reuses one prefill of the prompt. More steps cost GPU time, not
prompt tokens. The re-reads and `samples` of one request share one vision pass.

Reads run one at a time, so this backend is for local use, not for serving. A 3-question request
takes about 0.2–0.4 s on an M3 Ultra and about 0.39 s on an M4 Max, both with the 4-bit weights.
16 concurrent requests finish at about 4 req/s.

### Small encoder models

OpenJev also serves two small System One models that other people built and trained. The
credit is theirs. OpenJev only puts them behind the same API. Each is a bidirectional encoder
with a classification head, not a diffusion model. It reads each question in one forward pass,
so an answer is still a distribution over your options.

| Model id | Model | Size | State limit | Choices |
|---|---|---|---|---|
| `laya-1.0` | **[Laya](https://github.com/NandhaKishorM/laya)** by Nandakishor M / [Convai Innovations](https://huggingface.co/convaiinnovations). The [laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) checkpoint: ModernBERT-large, fine-tuned on the typed-decisions workflows. | 421M | 1,024 tokens, options included | up to 255. The options share 256 tokens, so with many options each option is cut to a few tokens. Use about 20 at most, or split the question. |
| `verdict-1.4` | **[Verdict](https://github.com/Heman10x-NGU/Verdict-open-jev)** by [Heman10x](https://huggingface.co/heman10x). The [rlcd-modernbert-151m](https://huggingface.co/heman10x/rlcd-modernbert-151m) checkpoint with Verdict's v1.4 inference engine: ModernBERT-base with a GLiClass head, calibrated per option count. | 151M | 512 tokens, options included | up to 24 |

Each model runs in its own container (`OPENJEV_BACKEND=laya` or `verdict`).
`docker compose up -d` starts both next to the vLLM container on the same GPU. The `openjev`
container sends their requests to them (`OPENJEV_MODEL_ROUTES`), so `:8080` serves all three
models. `/v1/models` lists a routed model even when its container is stopped. A request for it
then gets a 503. On a GPU both models use bf16 weights and need about 3.7 GB together. Set
`OPENJEV_GPU_UTIL` to keep that memory free.

Measured on an RTX PRO 6000 with 16 questions per request. GPU memory is the `nvidia-smi` value,
CUDA context included. A short state is about 50 tokens. A full state fills the model's limit.

| Model | GPU memory | 16 questions, short state | 16 questions, full state |
|---|---:|---:|---:|
| `laya-1.0` | 2.5 GB | 10 ms | 109 ms (16 × 1,024 tokens) |
| `verdict-1.4` | 1.2 GB | 7 ms | 21 ms (16 × 512 tokens) |

FlashAttention 2 gave no improvement. Memory was the same, speed was within 6%, and short
states were slower.

To run one model alone, on a GPU or on the CPU:

```bash
docker build -f docker/Dockerfile.laya -t openjev-laya .   # after the base, as above
docker run -d --gpus all -p 127.0.0.1:8081:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface openjev-laya
# or without Docker:
pip install -e '.[laya]' && OPENJEV_BACKEND=laya python -m openjev
```

A server that runs one of these models alone also accepts `jev-latest` and `jev-preview` for it.

Each image has its own Dockerfile in `docker/`. All three start from `docker/Dockerfile.base`
and share its CUDA, Python and PyTorch layers (about 8.7 GB on disk). A server that runs all three stores these layers once, about 21 GB in
total.

Differences from the DiffusionGemma model:

- Text only. `images`, `steps` above 1, `samples` above 1, `think` and `sequential` get a 400.
- The server cuts a state that is longer than the limit, with no error. `usage.input_tokens`
  counts what the model read. Each question is a separate sequence, so each question bills the
  state again.
- Verdict adds an "insufficient evidence" option to each question. OpenJev removes it and scales
  the other probabilities to a sum of 1, as Jev's answer shapes require. Verdict also ignores a
  noul's `criteria`.
- Laya rounds each probability to 4 decimal places.
- On a GPU the weights are bf16, not fp32. On 36 test answers per model, no top option changed.
  Probabilities moved by at most 0.021 (Laya) and 0.007 (Verdict).
- The Verdict prompt format and temperatures come from Verdict's inference engine (v1.4). For
  the same input, the probabilities are the same as that engine's output.

For benchmarks, training, fine-tuning and known limits, read the authors' repositories:
[Laya](https://github.com/NandhaKishorM/laya) and
[Verdict](https://github.com/Heman10x-NGU/Verdict-open-jev). The `laya` container runs Laya's
[`laya`](https://pypi.org/project/laya/) package. Verdict uses
[ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) (Answer.AI, LightOn) and
[GLiClass](https://github.com/Knowledgator/GLiClass) (Knowledgator).

### CLM

[CLM](https://github.com/Contrastive-LM/CLM) (Contrastive Language Model) is Contrastive-LM's
model; the credit is theirs. The [CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)
checkpoint is two small heads (a state head and an action head, 9.4M parameters each) on top of a
frozen [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B). Qwen3-8B turns the state (with the
question appended) and each option into its last-token embedding. The heads project them to 512
dimensions, and an answer is the softmax over the scaled cosine of each option with the state.

The `clm` image (`docker/Dockerfile.clm`) runs vLLM's pooling runner for Qwen3-8B and the heads
in one container. vLLM is the same pinned commit as the main image, without its changes. The
prompt layout, the heads and the scoring come from Contrastive-LM's
[`contrastive-lm`](https://pypi.org/project/contrastive-lm/) package (0.1.0).

```bash
docker build -f docker/Dockerfile.clm -t openjev-clm .   # after the base, as above
docker run -d --gpus all --ipc host -p 127.0.0.1:8083:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface openjev-clm
```

Next to DiffusionGemma, run `docker compose --profile clm up -d` and add
`clm-v0.1=http://clm:8080` to `OPENJEV_MODEL_ROUTES`. CLM runs its own vLLM, so lower the
`openjev` service's `OPENJEV_GPU_UTIL` to leave it room; `OPENJEV_CLM_GPU_UTIL` (default 0.12) is its
share of the GPU.

The default weights are [Qwen/Qwen3-8B-FP8](https://huggingface.co/Qwen/Qwen3-8B-FP8). On an
RTX 3090 (Ampere, no FP8 compute) vLLM runs them as weight-only FP8 through Marlin. On 581
four-way SQuAD questions, FP8 and bf16 agreed on 98.5% of top options. The mean embedding cosine
was 0.9992 and accuracy went from 89.7% to 88.8%.
[RedHatAI/Qwen3-8B-FP8-dynamic](https://huggingface.co/RedHatAI/Qwen3-8B-FP8-dynamic) does not
start on Ampere with this vLLM. Set `OPENJEV_MODEL=Qwen/Qwen3-8B` for bf16.

Measured on one RTX 3090 at `OPENJEV_GPU_UTIL=0.85`. Each request has a unique state (a SQuAD
paragraph) and 3 questions, about 550 prompt tokens in all:

| Weights | Weights in GPU memory | KV / prefix cache | 1 request at a time | 64 at a time |
|---|---:|---:|---:|---:|
| Qwen3-8B-FP8 | 7.7 GB | 78k tokens | 99 ms | 18 req/s, 9.6k prompt tokens/s |
| Qwen3-8B (bf16) | 14.1 GB | 33k tokens | 130 ms | 18 req/s, 9.8k prompt tokens/s |

The GPU is the limit: prefill is compute-bound, and weight-only FP8 does not add compute on Ampere.
What FP8 buys there is latency at low load and 2.4 times the prefix cache.
`--max-num-batched-tokens 8192` gave no gain.

Differences from the other models:

- Text only, as for the encoder models. A state longer than 2,048 tokens loses its start, not
  its end, so the question (which comes last) survives. Upstream CLM cuts the end (see
  [CLM PR #6](https://github.com/Contrastive-LM/CLM/pull/6)).
- The server keeps the embeddings and projections of recent texts (`OPENJEV_CLM_EMBED_CACHE`,
  `OPENJEV_CLM_CACHE`). `usage.input_tokens` counts only the texts it had to embed, so a repeated
  request reports 0.
- Score questions can ignore the state. Upstream reports one level winning whatever the state
  says ([CLM issue #3](https://github.com/Contrastive-LM/CLM/issues/3)), and in our checks a
  thankful customer scored "annoyed". Choice and noul questions follow the state. Evaluate score
  questions on your own data before you rely on them.

### JevK5

[JevK5](https://github.com/allebee/jevk5) is Alibi Serikbay's model; the credit is theirs. The
[JevK5](https://huggingface.co/alibiserikbay/JevK5) checkpoint (v0.2) is Qwen3.5-4B with a LoRA
distilled from Qwen3.6-27B, merged. Each question becomes a JSON prompt with its options lettered
A to P, and the answer is a softmax over those letters' next-token logits under one calibration
temperature (1.532, from the checkpoint's `jevk5_config.json`). The readout is
[SemIf's](https://github.com/TheoLeeCJ/SemIf). A question with more than 16 options takes several
passes, combined as JevK5 combines them.

The `jevk5` image (`docker/Dockerfile.jevk5`) runs the checkpoint in bf16 on vLLM, the same pinned
commit as the main image without its changes, and asks it for the letters' logprobs. The prompt
and the combining of passes come from JevK5's own [`jevk5`](https://github.com/allebee/jevk5)
package (0.2.2). JevK5's own server reads one question at a time; here vLLM batches the questions
of all requests together.

```bash
docker build -f docker/Dockerfile.jevk5 -t openjev-jevk5 .   # after the base, as above
docker run -d --gpus all --ipc host -p 127.0.0.1:8084:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface openjev-jevk5
```

Next to DiffusionGemma, run `docker compose --profile jevk5 up -d` and add
`jevk5-0.2=http://jevk5:8080` to `OPENJEV_MODEL_ROUTES`, as for CLM. `OPENJEV_JEVK5_GPU_UTIL`
(default 0.12) is its share of the GPU.

On JevBench's 231 public items, OpenJev and JevK5's own published v0.2 run gave the same top
answer on all 231 and the same input token count on all 231, so the prompts are identical.
Probabilities differed by 0.0012 at the median and 0.055 at most (vLLM's kernels are not
transformers'). Both scored 86.6%.

Measured on one RTX 3090 at `OPENJEV_GPU_UTIL=0.85` (7.9 GB of weights, 284k KV tokens), with the
same requests as for CLM:

| Request | 1 at a time | 32–64 at a time |
|---|---:|---:|
| 3 questions (a 4-way choice, a noul, a 3-level score), about 700 prompt tokens | 116 ms | 8–11 req/s, 8k prompt tokens/s |
| one 10-way choice | 75 ms | 20 req/s |

The GPU is the limit (100% busy at its 350 W cap). `--max-num-batched-tokens 8192` gave no gain.
Qwen3.5's linear-attention layers make vLLM cache prompts in blocks of 528 tokens, so questions
about a state shorter than that do not share its prefill; longer states do.

Differences from the other models:

- Text only, as for the encoder models. A read longer than 16,384 tokens gets a 400, never a cut.
- `usage.input_tokens` counts every pass, as JevK5 does: each question reads the state again.
- The image sets `VLLM_USE_FLASHINFER_SAMPLER=0`. A read takes one greedy token and keeps only
  logprobs, and FlashInfer's sampler would need a CUDA compiler the image does not have.

### Settings

The server reads its settings from the environment.

| Variable | Default | Meaning |
|---|---|---|
| `OPENJEV_BACKEND` | `vllm` | `mlx` to run the model in-process on Apple silicon. `laya` or `verdict` for a [small encoder model](#small-encoder-models), `clm` for [CLM](#clm), `jevk5` for [JevK5](#jevk5) |
| `OPENJEV_MODEL_ROUTES` | unset | `name=url,...`: other OpenJev servers. A request for one of these model names goes to that server unchanged. |
| `OPENJEV_FORWARD_TIMEOUT` | `300` | seconds before a request forwarded to another OpenJev server is a 503 |
| `OPENJEV_LAYA_MODEL` | `convaiinnovations/laya-typed-decisions` | Laya weights: a local directory or a Hugging Face id |
| `OPENJEV_VERDICT_MODEL` | `heman10x/rlcd-modernbert-151m` | Verdict weights: a local directory or a Hugging Face id |
| `OPENJEV_DEVICE` | unset | `laya`/`verdict`/`clm`: `cuda` or `cpu` (for `clm`, the heads). Unset uses CUDA when a GPU is present |
| `OPENJEV_ENCODER_BATCH` | `16` | `laya`/`verdict`: the most questions in one forward pass. A larger request uses more passes. |
| `OPENJEV_CLM_HEAD` | `Contrastive-LM/CLM-v0.1-8B` | `clm`: the heads, a Hugging Face repo holding `CLM_v0.1-8B.pt` or a local `.pt` file |
| `OPENJEV_CLM_MAX_TOKENS` | `2048` | `clm`: longest text sent to Qwen3-8B. A longer one loses its start. |
| `OPENJEV_CLM_WORKERS` | `32` | `clm`: requests read at once, so that vLLM batches them |
| `OPENJEV_CLM_CACHE` | `256MiB` | `clm`: GPU memory for cached projections, a size or a fraction of the GPU. `0` turns it off. |
| `OPENJEV_CLM_EMBED_CACHE` | `20000` | `clm`: embeddings kept in host memory (16 KB each) |
| `OPENJEV_JEVK5_WORKERS` | `32` | `jevk5`: reads in flight to vLLM at once |
| `OPENJEV_UPSTREAM` | unset | external vLLM server URL. When set, the container does not start its own |
| `OPENJEV_MODEL` | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` | weights for the built-in vLLM. `Qwen/Qwen3-8B-FP8` for `clm`, `alibiserikbay/JevK5` for `jevk5` |
| `OPENJEV_MLX_MODEL` | `mlx-community/diffusiongemma-26B-A4B-it-4bit` | MLX weights: a local directory or a Hugging Face id. Also gives the tokenizer. `8bit` and `bf16` builds are also available. |
| `OPENJEV_MLX_MAX_PROMPT` | `32768` | longest request, in tokens, before a 400 |
| `OPENJEV_GPU_UTIL` | `0.9` | vLLM `--gpu-memory-utilization`. `0.85` for `clm` and `jevk5` |
| `OPENJEV_MAX_NUM_SEQS` | `64` | vLLM `--max-num-seqs` |
| `OPENJEV_MAX_MODEL_LEN` | `65536` | vLLM `--max-model-len`. `2048` for `clm`, `16384` for `jevk5` |
| `OPENJEV_VLLM_ARGS` | unset | extra `vllm serve` flags |
| `OPENJEV_CANVAS` | `64` | canvas length. Also sets the built-in vLLM's `--diffusion-config` |
| `OPENJEV_MAX_INFLIGHT` | `64` | reads in flight to vLLM |
| `OPENJEV_MAX_QUEUE` | `512` | waiting decisions before the server returns 529 |
| `OPENJEV_MAX_QUESTIONS` | `256` | questions per request, before a 400 |
| `OPENJEV_MAX_BODY_BYTES` | `67108864` | request body size limit, before a 413 |
| `OPENJEV_API_KEY` | unset | require `Authorization: Bearer <key>` |
| `OPENJEV_ORIGIN_SECRET` | unset | require an `X-Origin-Secret` header (for use behind a proxy) |
| `OPENJEV_MAX_IMAGES` | `8` | images per request. Also sets the built-in vLLM's `--limit-mm-per-prompt` |
| `OPENJEV_MAX_IMAGE_BYTES` | `5242880` | size limit per image, after base64 decoding |
| `OPENJEV_GEN_MAX_INFLIGHT` | `8` | text generations that run at once |
| `OPENJEV_GEN_MAX_QUEUE` | `32` | waiting generations before the server returns 529 |
| `OPENJEV_GEN_MAX_TOKENS` | `8192` | cap on `max_tokens` for text generation |
| `OPENJEV_WARMUP` | `1` | `0` skips the warmup requests before the API opens. Warmup saves the first users several seconds of compilation. |

## Caveats

- The image pins upstream vLLM and makes two changes. A build fails if either change no longer
  applies.
  - It raises the limit of exact label ids per request from 128 to 512, for choices of up to
    255 options.
  - `docker/patches/vision_prefix_lm.py` gives image tokens bidirectional attention, as the
    checkpoint config asks. Upstream vLLM does this for Gemma4 but not yet for DiffusionGemma.
- The `clm` and `jevk5` images pin the same vLLM commit with neither change.
- Answer quality is the quality of DiffusionGemma 26B-A4B in this mode. Evaluate it on your own
  tasks before you rely on it.

## Development

```bash
pip install -e '.[test]' && pytest
OPENJEV_LIVE_URL=http://127.0.0.1:8080 pytest tests/test_live.py   # end to end against a running server
OPENJEV_MLX_TEST_MODEL=path/to/weights pytest tests/test_mlx_model.py   # MLX against the real model
```

Run the live checks after you build an image and before a cutover. They cover each read
option, images, chat, and the encoder models that the server lists.

## License

Apache-2.0. The DiffusionGemma weights are Apache-2.0 (NVIDIA / Google). Laya
(Nandakishor M / Convai Innovations) and Verdict (Heman10x) are Apache-2.0, weights and code.
`openjev/encoders.py` adapts Verdict's prompt format and calibration from its repository.
