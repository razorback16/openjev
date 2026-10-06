# Numeric decision scoring for ForJev

ForJev can request a numeric next-token readout from the resident vLLM model.
Prompt construction, images, tokenizer discovery, labels and typed answers
remain unchanged. No learned head or second model is required.

**Deployment responsibility:** the HTTP adapter alone cannot expose resident
model logits or change inference execution. Standard mode can use an existing
compatible API without changing serving. Both numeric modes require code inside
the serving API process; native mode also requires scheduler/worker changes and
a full vLLM restart/model reload. Installing the package or running the dry run
does not apply those serving changes. Native scoring is experimental until GPU
probability parity, cancellation/cache behavior and concurrent planner traffic
have passed live acceptance. The earlier public benchmark used standard mode.

## Implementation status

| `FORJEV_SCORING` | Upstream path | Status |
| --- | --- | --- |
| `chat_logprobs` | `/v1/chat/completions` | Existing default, backwards compatible |
| `engine_scores` | `/v1/decision_scores` | Numeric bridge implemented; live runtime integration required |
| `prefill_scores` | `/v1/decision_scores` | Native B12X V2 integration implemented; Spark acceptance pending |

The engine bridge reads AsyncLLM results without `_create_chat_logprobs`.
It still schedules one generated token and uses the internal logprob collector,
sampler and scheduler. It cannot repair missing upstream probabilities and does
not promise MTP bypass or latency savings. Missing/misaligned data fails closed.

`prefill_scores` requires genuine prefill-only execution and never downgrades
to generation, argmax or fabricated confidence. Without a native provider the
route returns HTTP 501 before rendering/inference. An environment setting alone
does not implement that provider. The native implementation and checked installer
are described below.

## Model and serving compatibility are separate

The numerical method uses each autoregressive model's own tokenizer and actual
lm_head. For selected logits `z_i`, the conditional action distribution is
`exp(z_i - max(z)) / sum_j exp(z_j - max(z))`; applying that same normalization
to raw vocabulary logprobs yields the same result because their shared
normalizer cancels. Neither requires a new head trained on Qwen or hardcoded
Qwen token IDs. Next-token probabilities still do not measure calibrated action
success.

That mathematical portability is not implementation certification. A new
checkpoint needs label-token discovery, its chat template/answer position,
thinking suppression, optional vision and representative quality checks.
The renderer currently supports only `enable_thinking=false`; templates needing
different controls need a reviewed renderer change. Context limits and
quantization remain the serving model's constraints. The maintained tested
profile is Qwen3.8-Flash-Next on the reference stack; other families and
quantizations are unvalidated candidates.

The native provider calls the loaded model's `compute_logits`, so it uses the
actual resident quantization instead of loading a floating-point copy. A
different quantization still needs quality/probability validation; it is not
assumed equivalent to the original checkpoint. This differs from CLM's learned
projection heads, which require their matching encoder embeddings.

The supplied native integration is source-specific to one B12X build and its
V2 runner. Decoder autoregressive models compatible with that runtime may be
testable through the same hooks, but are not automatically supported by their
architecture name. Diffusion, encoder-decoder, pooling/encoder-only runners,
LoRA, streaming input, parallel/disaggregated execution and arbitrary vLLM
versions are outside this first integration. A generic port must satisfy the
[native provider requirements](#native-provider-requirements), add runtime
compatibility checks and pass live acceptance on its own model/build.

## B12X bridge with candidate logprobs

The preferred next live profile is `deploy/B12X-bridge.Dockerfile` with
`FORJEV_SCORING=engine_scores`. It uses the resident engine's ordinary batching,
sampler/MTP verification and asynchronous output transport. It does not install
native scheduler, output-processor, pooling or prefill-only hooks. Native requests
return 501 in this profile. One internal output token is still generated.

The supplied V2 rejection sampler omitted the ordinary sampler's per-request
`logprob_token_ids` arguments. If another request contributes draft tokens, the
whole batch enters rejection sampling, so a one-token decision can lose requested
candidate columns even though it has no drafts. The checked correction forwards
the candidate state, expanded request-slot mapping and maximum candidate width
through every verification chunk to the existing `compute_topk_scores`. Accepted
sampled tokens, rejection decisions and row offsets retain their existing code.
Custom-only logprob requests also produce readouts when top-k is disabled.

`openjev.b12x_logprobs_patch` verifies original/patched hashes, compiles all changes,
keeps separate backups and defaults to dry run. It changes three files:

| File | Bridge correction |
| --- | --- |
| `v1/worker/gpu/spec_decode/rejection_sampler.py` | Preserve custom candidate columns across expanded request rows and verification chunks |
| `v1/worker/gpu/model_runner.py` | Include custom candidate width when gathering sharded sampler results |
| `entrypoints/launchers/app.py` | Register the numeric bridge without a native provider |

Build from the **original** pinned serving image, never from the experimental
native image. The fingerprints reject the latter; native scheduler hooks must not
remain in a bridge deployment. `FORJEV_B12X_PROFILE=bridge` is stored in the derived
image so the existing launcher's `python -m openjev.b12x_patch` preflight delegates
to the bridge verifier. Use the same `serve.sh` and its existing model/cache/KV
settings. No weight files are copied or downloaded.

```bash
cd ~/openjev-forjev && git fetch origin feature/forjev-direct-scores && FORJEV_BUILD_DIR=$(mktemp -d /tmp/forjev-bridge.XXXXXX) && git archive origin/feature/forjev-direct-scores | tar -x -C "$FORJEV_BUILD_DIR" && docker build --network=none --build-arg B12X_BASE_IMAGE=vllm-node-b12x -f "$FORJEV_BUILD_DIR/deploy/B12X-bridge.Dockerfile" -t vllm-node-b12x-forjev "$FORJEV_BUILD_DIR"
```

Launch the derived image using the existing ForJev-enabled `serve.sh`, then update
the host adapter and select `FORJEV_SCORING=engine_scores`. Both the correction and
route registration need a full serving relaunch/model reload. `docker restart`
alone does not rerun the reference launcher's `docker exec` startup sequence.

The bridge requests `seed=0` and `temperature=0`. This controls token sampling;
it does not guarantee deterministic raw model logits or calibrated probabilities.
Missing candidate IDs now appear explicitly in a 502 response; no truncation,
padding of missing probabilities, retries or argmax substitute repairs them.

CPU tests run the actual patched rejection methods on mixed planner/decision
batches, candidates outside top-k, sparse request slots, verification chunks,
custom-only readouts and device-style adaptive offsets. GPU kernels are stubbed
for these tests. Live candidate completeness and mixed-load decode performance
still require validation. This addresses a confirmed candidate omission in the
source; elimination of the earlier token/logprob-length 500 is not yet proven.

## Native B12X implementation

The supplied runtime is `0.1.dev20759+gb40673cd0.d20260913`. Its files have been
fingerprinted in `openjev/b12x_manifest.json`. `python -m openjev.b12x_patch`
verifies those fingerprints and compiles all modified sources before writing.
It defaults to dry run; `--apply` writes five files and retains original copies;
`--revert` verifies those copies and restores them. A write failure rolls back
the files already written. Stop installation if the source fingerprints differ.

This first native provider requires **V2**, TP/PP/DP/CP=1, an autoregressive decoder generation
runner, and no remote KV transfer. It supports the resident model without LoRA
or streaming input. V1 and parallel configurations return 501 before submission;
do not change runner configuration solely to bypass that check.

### Changes required inside the serving process

The installer changes exactly these files under the installed `vllm` package:

| File | Serving change |
| --- | --- |
| `v1/core/sched/scheduler.py` | Validate score requests; isolate scoring batches; reserve zero draft slots; finish on numeric output; exclude scoring from speculative-depth observations |
| `v1/core/sched/async_scheduler.py` | Fence final prefills with one lifecycle placeholder while scheduling no speculative IDs |
| `v1/worker/gpu/model_runner.py` | Keep candidates by request identity; project completed target prompt rows; bypass sampler and MTP including cached-boundary draft replay; clean up request metadata |
| `v1/engine/output_processor.py` | Convert the transported numeric tensor to an empty-token generation result accepted by AsyncLLM |
| `entrypoints/launchers/app.py` | Register `/v1/decision_scores` with the resident engine's renderer and the native provider |

Hook implementations live in `openjev.b12x_runtime`; `openjev.vllm_scores`
provides HTTP validation and rendering. Existing pooling transport carries the
numeric rows, but the model remains a generation runner: enabling a separate
pooling model or exposing embeddings is not sufficient. Source fingerprints
prevent the installer from applying these changes to another build merely
because filenames happen to match.

The scheduler alternates scoring and generation steps when both classes have
pending work. Scoring batches contain only scoring requests, with zero scheduled
speculative tokens. Async scheduling reserves one lifecycle placeholder to keep
the final prefill from being scheduled again; no output token is created. The
worker projects the actual target lm_head only for completed prompt rows,
gathers requested columns and returns CPU tensors through the existing pooling
transport. Intermediate chunks emit no scores. Full recurrent-cache boundary
hits reuse the saved target hidden state and skip MTP replay. Numeric outputs
finish the request and reach AsyncLLM as zero-token generation results.

No sampler, temperature/penalty processing, MTP proposal or speculative
verification runs on these scoring batches. The planner continues to use its
existing sampler and MTP paths on generation batches. Batch isolation and its
effect on planner throughput still need live measurement. Full vocabulary
logits are projected for completed rows; this is not a selected-row lm_head
implementation. `candidate_mass` is null because no full-vocabulary normalizer
is calculated. Conditional action probabilities remain available.

Applying the patch installs the route on the next normal vLLM startup. The
route retains `/v1` authentication; `FORJEV_NATIVE_SCORES=0` disables route
registration. Installation does not create an engine or load any weights.
Pending scoring is aborted on client disconnection, including during prefill.

### Install in the existing container

The following example is for the reference `qwen38-flash` container and a fork
checkout at `~/openjev-forjev`. Other deployments must replace those names and
retain their original image, entrypoint and configuration. It copies only the
committed source from the scoring branch, not the checkout's credentials or
virtual environment, installs without resolving model/CUDA dependencies, and
runs a read-only source verification. It does not change the host checkout:

```bash
cd ~/openjev-forjev && git fetch origin feature/forjev-direct-scores && git archive --format=tar origin/feature/forjev-direct-scores --output=/tmp/forjev-native.tar && docker cp /tmp/forjev-native.tar qwen38-flash:/tmp/forjev-native.tar && docker exec qwen38-flash sh -c 'mkdir -p /tmp/forjev-native && tar -xf /tmp/forjev-native.tar -C /tmp/forjev-native' && docker exec qwen38-flash python3 -m pip install --no-deps --no-build-isolation /tmp/forjev-native && docker exec qwen38-flash python3 -m openjev.b12x_patch
```

Review and pin the selected commit when building a repeatable deployment.
The last command is dry run. Successful verification prints five files. Ensure
the running configuration uses V2 and the supported parallelism before applying.
Applying alone changes files on disk, not the code already loaded in Python.
Restarting vLLM activates them and reloads the same model weights, interrupting
current inference. Schedule that interruption between agent runs:

```bash
docker exec qwen38-flash python3 -m openjev.b12x_patch --apply
```

Then restart the **vLLM process using its actual process manager**. A Docker
restart is sufficient only when the container entrypoint starts vLLM again.
Eugr's B12X launcher uses `Action: exec`: restarting the container does not
repeat that launch command. The reference Qwen `serve.sh` also removes the
old container before launching a new one, losing any installation made with
`docker exec`. For that launcher, use the persistent image deployment below
instead of installing in the running container and calling `docker restart`.

After vLLM is healthy, install the matching ForJev source revision in the host
adapter environment with `bash setup-forjev.sh`. This is a separate adapter
update and does not manage vLLM. Run the existing read-only capability probe in
native mode, including an image if required:

```bash
cd ~/openjev-forjev && set -a && source .forjev.env && set +a && FORJEV_SCORING=prefill_scores .venv-forjev/bin/python -m openjev.forjev_probe --choices 20
```

The shell override applies only to the probe. After live acceptance, set
`FORJEV_SCORING=prefill_scores` in `.forjev.env` and restart ForJev. Keep the
default legacy mode until then. Native mode fails rather than silently reverting
to generation if its provider is unsupported or malformed.

For an installation retained in the same container, restore the original files:

```bash
docker exec qwen38-flash python3 -m openjev.b12x_patch --revert
```

Restart vLLM through its process manager afterward. For a derived image,
rollback instead means launching the original image as described below.

### Persistent deployment with the Eugr/Qwen launcher

`deploy/B12X.Dockerfile` installs the package and applies the checked patch in
a derivative of the existing B12X image. It preserves the base image's startup
configuration. It copies no weights and resolves no Python dependencies. Build
from a committed source archive, keeping local environment files out of the
Docker context. Replace the base image with the exact pinned image you use:

```bash
cd ~/openjev-forjev && git fetch origin feature/forjev-direct-scores && FORJEV_BUILD_DIR=$(mktemp -d /tmp/forjev-image.XXXXXX) && git archive origin/feature/forjev-direct-scores | tar -x -C "$FORJEV_BUILD_DIR" && docker build --network=none --build-arg B12X_BASE_IMAGE=vllm-node-b12x -f "$FORJEV_BUILD_DIR/deploy/B12X.Dockerfile" -t vllm-node-b12x-forjev "$FORJEV_BUILD_DIR"
```

Build completion verifies the five source fingerprints; it is not GPU runtime
acceptance. A missing build dependency or changed source aborts the build rather
than downloading dependencies or accepting an unverified runtime. The original
image and running server are unaffected by this build.

The reference Qwen launcher accepts `B12X_IMAGE` through `config.env` and
passes it to Eugr's launcher. During a scheduled interruption, launch with:

```bash
cd ~/Qwen3.8-Flash-Next-Int4-FAST && B12X_IMAGE=vllm-node-b12x-forjev ./serve.sh
```

Retain your existing environment overrides and foreground/background choice.
Set `B12X_IMAGE` in your persistent launcher configuration for subsequent runs;
the command above overrides it for that invocation only. Qwen's existing mods
still run before vLLM starts. Confirm `/v1/decision_scores` is registered, then
run the native capability probe and live acceptance checks. This image/launcher
path still needs testing on the actual Spark.

To roll back, launch with `B12X_IMAGE=vllm-node-b12x` (or your original pinned
image), using the same serving configuration. Both launches recreate the
container and reload the resident model; neither downloads another checkpoint.

## Diagnose probability differences under concurrency

`python -m openjev.forjev_parity` replays selected questions from a saved
WorkflowEvals `results.json` using the same prompt builder as ForJev. It sends
the same messages, template options and candidates directly to the numeric
route, comparing serial and concurrent engine-logprob requests, then serial
and concurrent native requests. It bypasses the adapter's inflight limit
and SDK retries; no serving changes or restart are required. Run without other
model traffic. The default is two repeats, concurrency two and a 30-second
per-request timeout. A failed phase stops the diagnostic and saves its errors.

The JSON report includes prompt hashes, candidate IDs, raw scores, probabilities,
execution metadata and timings, but no state, prompt or credentials. It checks
matching prompts/candidates before comparing distributions. This does not reset
the prefix cache, establish a cold-cache baseline or guarantee identical GPU
arithmetic across batch shapes. The engine bridge is a comparison path, not an
independent correctness oracle; failures from its logprob collector are recorded.

Use `--skip-prefix-cache` to set `SamplingParams.skip_reading_prefix_cache=true`
on both providers. This recomputes prompts without reading cached prefixes;
cache writes remain enabled and the existing cache is not cleared. The numeric
HTTP route accepts the same strict boolean `skip_reading_prefix_cache`, default
false. `--providers engine` or `--providers native` selects just one provider.
Updated serving code must be installed and vLLM relaunched before these options
can be used; updating the host-side diagnostic alone is insufficient.

The native recurrent-state postprocess now receives the updated GPU computed
token buffer, matching the ordinary runner's model-state interface. Previously
the missing argument skipped Mamba align postprocessing. This is a lifecycle
correction, not evidence that observed probability differences are resolved.
The uploaded live diagnostic also showed serial-repeat drift in both providers;
prefix-cache bypass and live GPU checks remain required.

Example (replace the result path with your saved run):

```bash
cd ~/openjev-forjev && set -a && source .forjev.env && set +a && .venv-forjev/bin/python -m openjev.forjev_parity --results ~/workflowevals-forjev/runs/invoice_processing/forjev-native-inflight1-20261004-185719/results.json --output .forjev-run/parity.json
```

## Runtime integration

Install this package inside the existing vLLM runtime and call
`openjev.vllm_scores.install_routes(app)` on its existing FastAPI app before
serving. Do not construct another engine or load another model. The exact
insertion point must be checked against the installed runtime sources; this
generic route installer does not deploy or restart a running container. The
explicit B12X source installer above registers the native provider.

For a bridge-only integration, call `install_routes(app)` during the serving
API's app construction, before it starts accepting requests. Restart/redeploy
the API process using that app; do not create an additional engine. Worker
changes are unnecessary for the bridge, but its internal generation/logprob
path remains. This is a serving integration step, not a ForJev-only feature
toggle. For native integration call
`install_routes(app, prefill_score=provider)` only when the checked worker and
scheduler changes are also installed.

The `/v1` route inherits the app's authentication middleware. It validates the
model and uses vLLM's existing multimodal chat renderer. The initial renderer
supports only the resident base model, no LoRA, tools or arbitrary template
overrides, with `enable_thinking=false`. Compatibility code covers legacy
`_preprocess_chat` and newer `render_chat_request`, pending checks on B12X.
The engine bridge requires raw_logprobs mode and sufficient max_logprobs.

Set `FORJEV_SCORING=engine_scores` after installing the route, then run the
existing ForJev capability probe. It reports the chosen scoring mode. Existing
SystemOne clients and TypeSafe answer shapes need no changes.

## Numeric contract

The request contains `model`, `messages`,
`chat_template_kwargs: {"enable_thinking": false}`, `candidate_token_ids`
(unique IDs discovered through the served tokenizer), and `require_prefill`.
`require_prefill=false` selects the engine bridge, even if a native provider is
installed; `true` selects the native provider or returns 501 when it is absent.

| Response field | Meaning |
| --- | --- |
| `schema` | `forjev.decision_scores.v1` |
| `execution` | `engine_logprobs` or `prefill_logits` |
| `score_type` | `raw_logprobs` or `raw_logits`; processed scores are rejected |
| `token_ids`, `scores` | All requested IDs with their raw values |
| `probabilities` | Stable softmax over the candidate set |
| `candidate_mass` | Full vocabulary probability mass, or null without its normalizer |
| `generated_tokens` | 1 for the engine bridge; 0 for genuine prefill execution |
| `usage.prompt_tokens` | Rendered prompt token count |

The gateway reorders scores by ID and computes softmax itself. Missing,
duplicate, nonfinite or incorrectly typed results are errors. These are model
next-token probabilities, not calibrated probabilities of action success.

## Native provider requirements

`install_routes(app, prefill_score=provider)` accepts an async provider with
the same signature as `engine_scores`. It must integrate into the resident
scheduler and return the verified prefill_logits contract. A production
provider must preserve all of the following:

1. Identical prompt/template and multimodal preprocessing.
2. Target-model logits at the position predicting the first answer token,
   before sampler penalties, temperature, truncation and masking.
3. Request identity and position through batch compaction, async execution and
   chunked prefill; no global last-logits slot.
4. No sampling, MTP drafting or speculative verification for scoring requests;
   normal planner generation retains its existing path.
5. Cancellation, cache ownership and resource cleanup. Cache hits may still
   need the final position recomputed to obtain a readout.
6. GPU gathering of candidate columns; optional full-vocabulary logsumexp
   for candidate_mass. Use the actual loaded lm_head and its quantization.

Selected-row lm_head projection is a later optimization requiring equivalence
tests; pooled embeddings and hidden states before final normalization are not
a substitute for its actual input.

## Verification

CPU tests cover logit/logprob normalization equivalence, full vocabulary mass,
reordered/missing candidates, execution metadata, typed answers, image forwarding,
route validation, request identity, cancellation and backwards compatibility.
Native tests use real PyTorch CPU projections for row/column identity, completed
versus intermediate chunks, saved boundaries and nonfinite outputs. With
`B12X_SOURCE_DIR` pointing to the supplied `vllm` source directory, they also
execute the actual patched async-scheduler/output-processor methods and verify
apply, idempotence, fingerprint rejection and rollback. PyTorch is required for
the native numeric tests; without it those cases skip.

Run `python -m pytest tests/test_decision_scores.py tests/test_vllm_scores.py tests/test_forjev.py tests/test_forjev_service.py -q`.
Add `tests/test_b12x_runtime.py` for native tests.

Live acceptance requires the actual B12X build: compare distributions with
successful legacy calls and exercise concurrent planner traffic, batch sizes,
chunked prefill, prefix cache hits/misses and cancellation. Measure latency
on hardware; CPU tests establish no speedup.
