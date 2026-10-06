# ForJev 0.1.0 — Qwen backend for OpenJEV

ForJev has a standard HTTP mode and two opt-in numeric scoring modes.
**The native mode requires changes to the vLLM serving process and a restart.**
See the deployment matrix below and
[decision-scores.md](docs/decision-scores.md) for serving changes, installation,
rollback and live acceptance requirements. Native scoring is experimental;
GPU probability parity and shared-engine performance have not been validated.

For the checked B12X V2 build, the preferred next live test is now the
[bridge image with the MTP candidate-logprob correction](docs/decision-scores.md#b12x-bridge-with-candidate-logprobs).
It preserves ordinary scheduling, mixed batches and speculative execution.
The earlier native image isolates scoring steps and showed substantial decode
slowdown during the user's mixed workload; it remains experimental.

ForJev turns an **already running Qwen/vLLM HTTP endpoint** into OpenJEV's
typed decision API. It runs a CPU HTTP adapter and loads no LLM, tokenizer
weights, vision tower or learned classification head. The same Qwen instance
continues serving its existing clients. This branch integrates with OpenJEV's
schema validation, authentication, errors and answer formatting.

| URL | Role |
| --- | --- |
| `http://<spark>:8000` | Existing Qwen on pinned Eugr b12x; harness URL stays unchanged |
| `http://<spark>:8001/v1/systemone` | Typed decisions via ForJev |
| `http://<spark>:8001/v1/chat/completions` | Optional transparent chat/stream/tool proxy to Qwen |
| `http://<spark>:8001/v1/models` | ForJev and upstream chat model discovery |
| `http://<spark>:8001/health` | Adapter liveness |
| `http://<spark>:8001/ready` | Adapter plus upstream health |

The default `chat_logprobs` installation uses the existing serving API and
requires no Qwen restart or second model download when that API is compatible.
`engine_scores` requires registering a route inside the serving API process.
`prefill_scores` additionally requires the checked scheduler/worker integration;
applying it requires restarting vLLM and reloading its resident model. Neither
numeric mode is enabled merely by installing the HTTP adapter. Existing
hidden-state capture and the older gateway on 8088 are independent.

## Scoring modes and serving responsibilities

| `FORJEV_SCORING` | Required serving behavior | Serving changes / restart | Upstream execution |
| --- | --- | --- | --- |
| `chat_logprobs` (default) | `/v1/chat/completions` returns every requested candidate's token ID and logprob | None if the existing API implements the required fields; otherwise serving support is needed | One internal generated token; sampler and configured speculative path remain available |
| `engine_scores` | `/v1/decision_scores` reads raw logprobs from the existing AsyncLLM | Register the numeric route in the API process; restart/redeploy that process | One internal generated token; avoids chat formatting but retains sampler and logprob transport |
| `prefill_scores` (experimental) | `/v1/decision_scores` returns final-prompt raw logits with `generated_tokens: 0` | Checked B12X V2 scheduler, worker, output processor and API integration; restart vLLM | Target prefill and lm_head; no score-request sampling, MTP drafting or speculative verification |

The standard mode retains the observed vLLM chat-logprob 500 failure. The
engine bridge does not repair missing upstream logprobs. Native scoring avoids
those sampling/logprob-formatting paths by design, but has not yet been proven
to resolve the observed failure under live shared-engine load. Defaults remain
standard mode until live acceptance. No mode invents missing probabilities or
silently substitutes an argmax-only result.


## Supported models and deployment profiles

This is the ForJev compatibility list, separate from OpenJEV's other backends.
An upstream name in `/v1/models` selects the running Qwen checkpoint; the
SystemOne API model remains `forjev-qwen-next`.

| Model family | Reference deployment | Inputs | ForJev status |
| --- | --- | --- | --- |
| **Qwen3.8-Flash-Next** | `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`, existing pinned Eugr b12x on DGX Spark | Text; text + image | Initial supported target. Text/image decisions and candidate-logprob API primitives were exercised in the earlier prototype. The integrated OpenJEV backend has offline coverage; each live installation must pass the included probe. |

Reference deployment details:
- Upstream served ID in our setup: `qwen3.8-flash-next-a5b`. This is an
  administrator-defined serving alias, not a tokenizer ID or universal
  provider model name. Use the exact ID in your server's `GET /v1/models`.
- Known serving build from the prototype: vLLM
  `0.1.dev20759+gb40673cd0`, V2 runner, pinned Eugr b12x. Preserve the working
  image and model cache on the existing Spark.
- Adapter defaults: 20 choices and two concurrent decision reads. Larger
  option sets require a successful probe at that size.
- Other quantizations, providers and builds of this family are **candidates
  for compatibility**, not automatically validated by the family name.
- Qwen4 and other Qwen families are not yet listed as supported. Add a profile
  after testing its tokenizer, non-thinking template, candidate logprobs,
  optional vision, and representative decision quality.

### Model portability versus serving portability

The scoring principle applies to autoregressive language models exposing raw
next-token logits: choose single-token labels in the model's own tokenizer and
normalize their scores over the supplied options. No CLM head or model-specific
trained weights are transplanted. The adapter does not hardcode Qwen token IDs
or a vocabulary size; `forjev-qwen-next` remains the existing wire alias.

This does **not** certify every checkpoint or serving stack. Each model needs
enough distinct single-token labels, a compatible chat template that predicts
the answer label without a preceding reasoning segment, and representative
decision-quality tests. The current template request uses
`enable_thinking=false`; another family's template may need a different
implementation. Images additionally require actual vision support. Passing
the small capability probe establishes API functionality, not decision quality
or calibration across tasks.

The native installer is restricted to fingerprinted sources from
`0.1.dev20759+gb40673cd0.d20260913`, V2, TP/PP/DP/CP=1, decoder generation,
and no remote KV transfer. LoRA, streaming input, encoder-decoder and diffusion
models are outside its supported request/configuration scope. Other
autoregressive checkpoints on that runtime are candidates for testing, not
validated profiles. A different vLLM build needs a reviewed provider port and
new source fingerprints; changing only the model name or enabling an environment
variable does not make the patch compatible.

## Suitable providers and endpoint exposure

Compatibility depends on the exposed API and the served model/build, not on
the hosting company's name. Infrastructure below is suitable for deploying a
compatible server; it is not a claim that a provider's ready-made model API
passes the ForJev probe.

| Deployment/provider | Suitability for this backend | Required exposure |
| --- | --- | --- |
| Existing local/VPN Qwen on vLLM | Reference path; uses the current Spark installation | Server root plus `/health`, `/v1/models`, `/tokenize`, `/v1/chat/completions` |
| **RunPod Pods**, with your own compatible vLLM deployment | Suitable hosting option; verify model/build and hardware first | Expose the vLLM HTTP port through the Pod's HTTP proxy or your own network path, including all four routes |
| **RunPod Serverless Load Balancer with a compatible vLLM image** | Conditional candidate; its documented direct-route mode can expose vLLM routes | Use the direct server root; pass the probe. Queue-style `/run` or `/runsync` APIs need a separate adapter and are not accepted by ForJev |
| Your own remote GPU VM/container | Suitable if you control the model server and routing | Same routes and request fields as local vLLM; supply `FORJEV_UPSTREAM_API_KEY` if required |
| Generic managed “OpenAI-compatible” model API | Not certified merely by that label | Must expose tokenizer access and exact candidate logprobs; plain chat, JSON output or top-k-only probabilities are insufficient |

The provider table describes **standard mode**. Numeric modes additionally
require `/v1/decision_scores` and control of the serving API; native mode also
requires the compatible worker integration and a scheduled restart. A managed
API cannot gain native scoring through a ForJev-only configuration change.

The existing Spark image may be hardware/architecture-specific. A provider
must have enough resources for the model and a compatible serving build;
ForJev itself does not make model weights fit or add architecture support.

For remote endpoints, configure the **server root** in `OPENJEV_UPSTREAM`,
without appending `/v1`. A reverse proxy must preserve the required routes,
`logprob_token_ids`, `return_tokens_as_token_ids`, and the response
`top_logprobs` entries. The tokenizer must match the checkpoint actually
served. The launcher expects `/health` as well as `/v1/models`; a proxy
that exposes only chat is not enough for this version.

No global vLLM option is required solely to expose token IDs when the
running build already implements the documented request fields. ForJev sends
them on each request. If the build lacks them or its explicit candidate-ID
limit is too small, serving changes may be needed; the adapter cannot enable
missing server features. Never apply DiffusionGemma-specific patches to Qwen.

Provider/API references, checked 2026-09-29:
- [vLLM deployment on RunPod Pods](https://docs.vllm.ai/en/latest/deployment/frameworks/runpod/)
- [RunPod worker-vllm direct-route deployment](https://github.com/runpod-workers/worker-vllm)
- [vLLM chat protocol: exact candidate IDs and token-ID responses](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/protocol/)

## Install and serve on DGX Spark

Clone this OpenJEV fork in its own directory:

```bash
git clone https://github.com/MhaWay/openjev.git ~/openjev-forjev
```

Python 3.12 on Linux is the tested installer target; `python3-venv` must exist.
Run from the checkout:

```bash
cd ~/openjev-forjev
bash setup-forjev.sh
bash restart-forjev.sh start
```

The installer creates `.venv-forjev`, installs the CPU HTTP requirements,
installs this package with `--no-deps`, and copies `forjev.env.example` to
`.forjev.env` only if absent. It deliberately does not install upstream's
Transformers/model dependencies. Do not use upstream's default Docker Compose
or DiffusionGemma entrypoint for this deployment.

Defaults in `.forjev.env`:

```sh
OPENJEV_BACKEND=forjev
OPENJEV_UPSTREAM=http://127.0.0.1:8000
OPENJEV_UPSTREAM_MODEL=qwen3.8-flash-next-a5b
OPENJEV_HOST=0.0.0.0
OPENJEV_PORT=8001
OPENJEV_MAX_INFLIGHT=2
FORJEV_MAX_CHOICES=20
```

`OPENJEV_UPSTREAM` is the server root, **without `/v1`**. Set the model to an
actual ID returned by upstream `/v1/models`. For an authenticated upstream,
set `FORJEV_UPSTREAM_API_KEY`. Frontend authentication uses the independent
`OPENJEV_API_KEY` and `OPENJEV_ORIGIN_SECRET`; these frontend credentials are
not passed to Qwen. Leave keys empty for the existing VPN/local setup.

```bash
bash restart-forjev.sh status
bash restart-forjev.sh probe
bash restart-forjev.sh restart
bash restart-forjev.sh stop
```

This Linux controller uses a lock, PID plus process creation identity, and
a detached process with logs under `.forjev-run/`. It only signals the
ForJev process it started from this checkout. It never invokes Docker or
changes/stops Qwen. Start verifies the upstream model, token IDs and complete
candidate logprobs before launching, then checks `/ready`, `/v1/models` and
a real SystemOne response. Failed starts stop only the new adapter. Restart
checks the upstream before stopping the old adapter; it does not automatically
roll back source changes. Fix the reported error and run start again.

The detached adapter survives terminal/harness disconnection. It does not
automatically start after a machine reboot. The harness stays on Qwen :8000,
so adapter lifecycle/port changes require no reconnect listener or model
restart. This does not remove the separate vLLM restart required when installing
the native serving patch.

Manual foreground launch, after exporting the settings above:

```bash
.venv-forjev/bin/python -m openjev
```


## Select the listening port

The backend and its listening port are independent settings. OpenJEV already
supports `OPENJEV_BACKEND`, `OPENJEV_HOST` and `OPENJEV_PORT`; no vLLM
restart is needed to change the adapter port.

For the managed script, edit the existing `.forjev.env`, for example:

```sh
OPENJEV_BACKEND=forjev
OPENJEV_HOST=0.0.0.0
OPENJEV_PORT=9001
OPENJEV_UPSTREAM=http://127.0.0.1:8000
OPENJEV_UPSTREAM_MODEL=qwen3.8-flash-next-a5b
```

Then run `bash restart-forjev.sh restart`. Only ForJev restarts and moves to
9001; Qwen and its existing clients stay on 8000. Choose a free port.
The script sources `.forjev.env`, so values in that file override inline
environment assignments. Alternatively, select another config file with
`FORJEV_ENV_FILE=/absolute/path/forjev.env bash restart-forjev.sh start`.
The controller manages one instance per checkout.

For direct foreground startup from the installed checkout, set everything
on one command line (this does not read `.forjev.env`):

```bash
OPENJEV_BACKEND=forjev OPENJEV_HOST=0.0.0.0 OPENJEV_PORT=9001 OPENJEV_UPSTREAM=http://127.0.0.1:8000 OPENJEV_UPSTREAM_MODEL=qwen3.8-flash-next-a5b .venv-forjev/bin/python -m openjev
```

A manually launched instance is managed by that terminal/process manager,
not by `restart-forjev.sh`. Stop an existing instance before reusing its port.
Changing the public adapter port means SystemOne clients must use the new
port; direct Qwen clients remain unchanged.

## How the decision works

For each nontrivial question, ForJev formats state, question and options into
a prompt, assigns each option a distinct single-token label, and optionally
includes the supplied image parts. In standard and engine-bridge modes it
requests one output token with thinking disabled and reads the **next-token
logprobs of every candidate label**. Native mode reads the raw logits at the
final prompt position without generating a token. It normalizes the selected
scores with `exp(score - max) / sum(exp(...))` and delegates
choice/score/noul formatting to OpenJEV.

The target prompt computation still runs, subject to valid prefix-cache reuse.
There is one upstream scoring request per nontrivial question, not a generated
explanation. Standard/bridge requests use `max_tokens=1`; native requests reserve
the same lifecycle limit but finish with no output token. A one-option choice
can be returned without inference. OpenJEV's `output_tokens: 0` denotes the
decision-only wire contract: only verified native execution also means zero
upstream generated tokens. Input usage sums actual upstream prompt tokens.

Probabilities are relative to the provided options, not calibrated real-world
certainty. OpenJEV's entropy-based confidence is preserved. Label order,
prompt wording and quantization can affect results. This backend is not a
trained JevK5 checkpoint or diffusion inference implementation.

## API and token IDs required from Qwen/vLLM

All modes require `/health`, `/v1/models` and the served `/tokenize` API.
**Standard mode** additionally requires the following chat behavior:

1. `GET /health` and `GET /v1/models` for its launcher preflight.
2. `POST /tokenize` accepting `model`, `prompt`, `add_special_tokens:false`,
   returning numeric `tokens`. This resolves the labels in the served tokenizer.
3. `POST /v1/chat/completions` supporting `logprobs:true`,
   `logprob_token_ids:[...]`, and `return_tokens_as_token_ids:true`.
   The answer must include all requested candidates in the first output
   position's `top_logprobs`, with `token:"token_id:<integer>"` and `logprob`.
4. For vision, OpenAI-style `image_url` content and a multimodal model/template.
   The Qwen template must honor `chat_template_kwargs:{"enable_thinking":false}`.

In standard mode these are per-request options. No ForJev worker extension, pooling runner,
dev RPC or hidden-state patch is required. The previously tested pinned b12x
build supports this route for small candidate sets; verify this new integration
live before treating it as production-ready. Do not upgrade a working image
merely to install this adapter.

For numeric modes, replace requirement 3 with the versioned
[`/v1/decision_scores` contract](docs/decision-scores.md#numeric-contract).
The endpoint must use the same served model, tokenizer, chat template and image
preprocessing. Native mode requires `execution: prefill_logits` and
`generated_tokens: 0`; the adapter rejects a generated-token substitute.
The [serving integration guide](docs/decision-scores.md#native-b12x-implementation)
lists the exact five patched vLLM files, restart, dry run, rollback and tests.

In standard mode, the pinned vLLM build has occasionally returned an HTTP 500 while assembling
one-token chat logprobs (`_create_chat_logprobs`: `list index out of range`).
ForJev retries that specific response at most twice, with short delays. Other
errors still propagate. This mitigates an intermittent serving failure; it
does not fix its root cause in vLLM or prove that all future requests succeed.

Inspect the actual IDs and test all configured candidates:

```bash
bash restart-forjev.sh probe
```

The JSON includes `labels`, e.g. `{"A":32,"B":33}` if those are what the live
tokenizer reports, and the probabilities returned by a real request. IDs are
discovered and cached in process; they are never hardcoded or substituted from
another Qwen tokenizer. Restart ForJev after changing the upstream model/tokenizer.

To test an image explicitly, export the environment and run:

```bash
set -a; source .forjev.env; set +a; .venv-forjev/bin/python -m openjev.forjev_probe --choices 20 --image /path/to/image.jpg
```

Default support is 20 choices. You may configure `FORJEV_MAX_CHOICES` up to
255, but only if enough distinct single-token labels exist and vLLM accepts
that many `logprob_token_ids`. Some builds cap this at 128; `--max-logprobs`
and the explicit-token-ID cap are not necessarily the same setting. Inspect
the exact serving build and rerun the probe at the requested count. Missing
candidate logprobs cause an error; ForJev does not silently invent probabilities.

The adapter cannot turn an arbitrary cloud chat endpoint into this exact
readout. A provider needs these capabilities (plus authentication support);
ordinary top-k logprobs do not guarantee that every option is present.

## Requests

Model ID: `forjev-qwen-next`. OpenJEV SDK aliases `jev-latest` and
`jev-preview` also select this backend. The upstream Qwen chat ID is separate.

Text-only state: omit `images` or send `images:null`. The state is still sent
to Qwen; absence of an image does not skip inference.

```bash
curl -fsS http://127.0.0.1:8001/v1/systemone -H 'Content-Type: application/json' -d '{"model":"forjev-qwen-next","state":{"health":4,"threat":"zombie"},"questions":{"action":{"type":"choice","instructions":"Choose the safest action.","criteria":{"retreat":"Move away from the zombie","approach":"Move toward the zombie"}}}}'
```

Images are passed as OpenJEV top-level `images`, for example:

```json
{
  "model": "forjev-qwen-next",
  "state": "Classify this image.",
  "images": ["data:image/jpeg;base64,<actual base64 bytes>"],
  "questions": {
    "letter": {
      "type": "choice",
      "instructions": "Which uppercase letter is visible?",
      "criteria": {"B": "The letter B", "D": "The letter D"}
    }
  }
}
```

`choice`, `score` and `noul`, structured states/instructions and multiple
questions use the existing OpenJEV schema. Images are optional; no frame/session
upload from the older prototype is required. Unsupported `steps>1`,
`samples>1`, `think>0` and `sequential:true` are explicitly rejected.

## Version, compatibility and validation

ForJev backend version: **0.1.0**. The upstream OpenJEV package version is
retained separately. Derived from OpenJEV commit
`a0ddd7d928298eccef2c17153b00b5636b6d996a` and the earlier
`MhaWay/Qwen3.8-Flash-Next-Int4-FAST:feature/forjev` prototype.

Current target: Qwen3.8-Flash-Next served by the existing pinned Eugr b12x
stack. Future Qwen4 compatibility is a goal, not an already tested guarantee;
the HTTP contract, template, labels and quality must be checked with each model.

Offline tests cover the native OpenJEV API, TypeSafe SDK default-model calls,
state/image handling, typed answers, complete candidate probabilities, labels
beyond 52, the narrow HTTP 500 retry, unsupported options, chat streaming/auth,
and adapter start/stop against a fake upstream.
They do not establish accuracy, calibration, latency, 255-choice availability
or compatibility with a real future model.

```bash
.venv-forjev/bin/python -m pip install pytest 'typesafe-sdk>=0.7'
.venv-forjev/bin/python -m pytest -q tests/test_forjev.py tests/test_forjev_service.py
```

### Public JevBench run

These are historical **standard `chat_logprobs`** results. They do not establish
native-scoring probability parity, latency, calibration or immunity to the
observed serving failure; native GPU acceptance is still pending.

On 2026-09-29, one DGX Spark running the pinned Eugr b12x Qwen3.8-Flash-Next
INT4 setup answered all **231 public JevBench tasks** through ForJev's
`/v1/systemone` route: **202/231 correct (87.45%)**, 231/231 valid answers,
0 failed requests, top-label ECE 0.0388 and Brier mean 0.1728. Local
end-to-end latency was p50 0.215 s and p95 1.312 s, with the normal Qwen
harness also active. This run saw no upstream 500s, so the retry was not used.
The dataset hash recorded by JevBench was
`dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`.
The JevBench checkout was `bb05a335bc809e61b20c0f745d25499a82b326fc`.

To reproduce against an already running ForJev server, clone JevBench and run
its `typesafe` adapter with the three public datasets and a fresh results
directory. The endpoint below is loopback on the same host as ForJev:

```bash
git clone https://github.com/fstandhartinger/jevbench.git ~/jevbench
cd ~/jevbench
RUN_DIR="$HOME/forjev-bench-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$RUN_DIR"
python3 -m jevbench.cli run --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl --adapter typesafe --endpoint http://127.0.0.1:8001 --model forjev-qwen-next --key-env '' --cost-basis self_hosted_no_provider_tariff --reserve-usd 0 --results "$RUN_DIR/results.jsonl" --raw-dir "$RUN_DIR/raw" --ledger "$RUN_DIR/ledger.jsonl" --manifest "$RUN_DIR/manifest.json"
python3 -m jevbench.cli summarize --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl --results "$RUN_DIR/results.jsonl" --public-export "$RUN_DIR/summary.json"
```

This is the **public subset**, not a full official JevBench score. No
self-hosted compute cost was measured. Model size, deployment, input ordering
and network conditions differ across published systems; small differences in
correct counts or latency should not be described as proven superiority.
The [JevBench harness](https://github.com/fstandhartinger/jevbench) and an
[independent public-subset comparison](https://github.com/Zefan-Cai/Open-Jev/blob/main/docs/jevbench-public.md)
give the dataset and comparison context. The raw run artifacts are currently
stored on the Spark at `~/forjev-bench-20260929-153053/`.

Credits and licensing: OpenJEV by razorback16 and contributors, Apache-2.0;
ForJev integration uses the same repository license. Existing model licenses
and API provider terms still apply. The other OpenJEV backends are retained.
