# Generation and thinking retention

The objective is better classification while retaining accuracy, ordinary text
generation, and thinking. One DiffusionGemma model and one shared adapter serve
all paths. There is no separate reasoning architecture or adapter bypass.

## Native prompts and existing inference

The [official template](https://huggingface.co/google/diffusiongemma-26B-A4B-it/blob/f7f5b7f5fa82ffc52addd066915886d497f5517b/chat_template.jinja)
does not inject a fixed natural-language system prompt. It preserves caller
system/developer messages. Thinking is controlled by `enable_thinking=True`,
which inserts `<|think|>` in the system turn. Ordinary conversation uses the
native template with thinking disabled; classification already has OpenJev's
fixed-question system instruction. Do not invent a purported standard prompt.

`Engine.think()` already renders that thinking-enabled classification prompt,
opens `<|channel>thought\n`, generates up to a budget, and closes `<channel|>`.
`read_group()` then puts the decision canvas after this prefix, omitting the
empty-thought scaffold. The new thinking compiler follows exactly that path.

## NVIDIA Nemotron rehearsal

The builder pins NVIDIA Nemotron-SFT-Instruction-Following-Chat-v3 and
Nemotron-SFT-Agentic-v2. Revisions, checksums, and rejections are in report.json;
upstream cards and attribution are preserved in notices/.

The bounded smoke-test build contains 1,425 training and 163 validation records,
covering instruction following and agentic conversations in thinking/nonthinking
views. This is file-prefix sampling, not representative sampling. Sampled chat
rows with withheld content were excluded. Template-prefix mismatches exclude
some tool targets; resolve these before claiming complete tool-call coverage.
Tools are never executed. All turns and variants share a root-prompt split.
Limits are 8,192 prompt and 4,096 response tokens; oversized rows are rejected.
Only final assistant turns are selected for instruction/chat. Agentic records
select the first tool-call and final assistant turns where supported.

Raw and compiled records are excluded from Git. Preserve applicable CC BY,
ODC-By, Apache, and MIT source terms and attribution on redistribution.
See ../RETENTION-REVIEW.md for remaining objective and evaluation checks.

## Base-model thought rehearsal

Run `scripts/collect_thinking_replay.py` against a backend serving the frozen
original DiffusionGemma, **not an adapted checkpoint**. For example:

```bash
python scripts/collect_thinking_replay.py \
  --source data/openjev-pilot/permissive-release \
  --output data/openjev-thinking \
  --upstream http://127.0.0.1:8000 --count 128 --budget 256
```

The teacher sees only the state and question definitions, never gold targets.
The collector calls the existing `Engine.think()` implementation. Ground-truth
decisions are attached locally afterward. Thoughts inherit their parent split
and group; never redistribute variants of a validation state into training.
Cache model identity/revision and serving configuration with a real collection
run. The script does not provision a GPU or claim that collection has occurred.

The resulting record contains the generated thought as prefix and the ordinary
decision slots as targets. It also retains the teacher thought as a generation
rehearsal example. Thoughts are model outputs, not verified explanations; do not
equate a correct decision with a faithful or logically correct thought.

## Shared training objective

`train_slot_pilot.py --generation-data data/openjev-generation/nemotron/release
--thinking-data data/openjev-thinking` enables the initial 60/20/20 example
mixture: direct classification, thought-conditioned classification, ordinary
generation. Without those flags the original pilot remains reproducible.

Classification uses allowed-label slot CE. Generation uses full-vocabulary CE
at randomly corrupted response positions, with no next-token shift. A response
is divided into native 256-token canvases; earlier completed canvases become
encoder context, current/future clean targets never enter that context. Noise
rates are sampled from 0.25, 0.5, 0.75, and 1.0. Padding is not supervised.
Loss is normalized per record, and vocabulary projections are checkpointed in
small chunks to limit memory. Thought-conditioned records add a 0.1-weighted
teacher-thought rehearsal loss using the same shared parameters.

This is a proposed task-adaptation objective, not a reproduction of Google's
original pretraining objective. Decoder-only LoRA is still the small-pilot
parameterization; the architecture and inference paths are unchanged.

## Acceptance tests before scaling

Measure direct and think-before-classify accuracy, NLL, Brier, canvas-seed
stability, and actual reread rate against the original model with the same
thought budget and sampler. Keep a temperature-only control. Accuracy must be
an explicit constraint: predeclare a non-inferiority margin (proposed one
percentage point), use paired confidence intervals, and collect more evaluation
cases if the result is inconclusive. Do not promote a model solely because NLL
improved while accuracy or stability regressed.

The implemented cached-thought evaluation isolates the final decision stage.
It is **not** an end-to-end thinking-retention test. Regenerate thoughts with
the trained model on untouched cases to measure that path. Also compare ordinary
and thinking-enabled free generation, instruction following, and response quality.
The implemented held-out denoising loss is a diagnostic, not perplexity and not
proof of retained generation quality. Short English rehearsal does not establish
retention of multilingual, long-context, tool, or vision capabilities.

No new GPU run, thought collection, or retention-quality claim accompanies this
data/code change. The earlier three runs were classification-only and cannot
answer the retention question. Keep their published results unchanged.
