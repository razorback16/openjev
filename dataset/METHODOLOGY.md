# OpenJev public-data pilot

This is a reproducible, permissively licensed **dataset build**, not a trained
model or an accuracy claim. No Gemini API or other paid generation service is
used. Build outputs live in `data/openjev-pilot/permissive-release/` and the
machine-readable `report.json` is authoritative for counts.

## Reproduce

The compiler supports Python 3.10 or newer. The recorded build used Python
3.12.12; use Python 3.12 and the version pins below for exact reproduction:

```bash
pip install -r dataset/requirements-build.txt
pip install -e '.[data,test]'
python scripts/download_dataset_sources.py
python scripts/build_public_dataset.py
pytest -q tests/test_dataset.py
python scripts/audit_public_dataset.py
python scripts/package_public_dataset.py
```

The downloader needs no Hugging Face credential. It fetches roughly 3.13 GB of
inputs and verifies their byte counts and SHA256 hashes using `sources.lock.json`.
Hugging Face data URLs include commit IDs. Other URLs are content-pinned by SHA256;
an upstream change fails verification instead of silently changing the build.
The final tokenizer audit requires the DiffusionGemma tokenizer to be cached
locally; it does not download or load model weights, or call an inference server.
Its report records tokenizer contents and package versions.

The default target was 60K TaskSource, 20K FLAN, and 5K GoEmotions training
records, plus a separately identified 2K HelpSteer Score supplement. Validation
and calibration targets are each 5% of those counts, from additional held-out
source groups. Sampling never repeats examples to fill quotas. Strict source
matching and permissive-license filtering leave a substantial FLAN shortfall.
The selected FLAN subset is Cosmos QA only, not broad coverage of FLAN's tasks.

## Licensing and provenance

`training_sources.yaml` is an explicit underlying-source allowlist.
`evaluation_sources.yaml` reserves benchmark families and excludes aliases.
Container-level license tags never override source licenses. The accepted
licenses are Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, CC0-1.0, CC-BY-3.0,
and CC-BY-4.0. The actual build uses Apache-2.0, BSD-3-Clause, and CC-BY-4.0.
Noncommercial, research-only, share-alike, missing, and ambiguous terms are not
accepted. License evidence and original dataset cards are retained in `notices/`.

Every record has its source, task, family, declared license, original split,
zero-based original file row, container file row when applicable, and grouping
key. Resolve the file reference through `sources.lock.json` to find the exact
source bytes and URL. Raw source caches contain excluded datasets and are **not**
release artifacts. Release only the compiled files and documented notices.

## Compilation

TaskSource's `inputs/targets/task` representation is parsed only for the exact
supported classification template. The appended answer period is removed only
when it matches a listed label. Ambiguous formats are rejected, not inferred.
Every accepted pair and its label must match an original training example.

BANKING77 states are checked against PolyAI's original training CSV. Its full
77-intent vocabulary replaces TaskSource's reduced option sets. Deterministic
sampling uses 77 options for 50% of these records, 32 for 20%, 8 for 20%, and 4
for 10%. The correct label is always present; distractors are real source labels.
These proportions are sampling probabilities, not exact final quotas.

For other TaskSource examples, source label spaces are preserved. Tasks include
WANLI, bAbI-NLI subtasks, RuleTaker, and formal logical entailment. Existing public
synthetic reasoning data and WANLI's model-assisted examples are identified by
source; "public data" does not mean all text was written by humans.

FLAN is downloaded from `tasksource/flan`, a public mirror. Only single explicit
`OPTIONS` blocks for allowlisted tasks are considered. A candidate must match a
unique original Cosmos QA training context/question, its full answer set, and
its correct answer. Matching ignores case and whitespace; candidate retrieval
also ignores punctuation but final option/label checks do not. Multiple matches,
missing options, generative tasks, and label discrepancies are rejected. All
templates of an original example share one ID; retain one. "None of the above
choices" is rendered as "None of the other options" so shuffling preserves meaning.

Choice criteria are shuffled with a seed derived from the stable record ID.
Targets remain semantic keys rather than positional letters. Independently, 40%
of categorical examples receive one positive and one negative Noul question.
The order of those two questions is shuffled, preventing a fixed-position label
shortcut. They remain in the same record as the Choice question.

GoEmotions retains individual votes. Only comments in the original training
split are eligible; unclear annotations are omitted and at least two usable
raters are required. All emotions with any positive vote and one randomly
selected all-negative emotion become independent Noul questions. One seeded
rater supplies all hard targets for a record, preserving that rater's joint
annotation. `annotation_votes` stores counts for each emitted question. These
are independent Bernoulli judgments, **not** one categorical distribution over
mutually exclusive emotions. The selected-question policy changes prevalence;
this is a disagreement diagnostic, not a representative deployment calibration set.

HelpSteer supplies original integer 0–4 ratings for helpfulness, correctness,
coherence, complexity, and verbosity. The state contains the original prompt
and response as a JSON string; five Score questions share it. Responses were
generated by the source authors' model and annotated by people. Higher
complexity/verbosity means more complex/verbose, not necessarily better.
Rubric wording is abbreviated from the source's annotation guidelines.

## Splits and duplicate controls

Seed: `20260919`. A stable SHA256 of the original context assigns source groups
90%/5%/5% to train/validation/calibration **before** multi-question expansion.
These are sampling probabilities; selection caps are applied afterward.
Premises, BANKING77 queries, Reddit threads, and HelpSteer prompts are grouping
units. Variants or different questions about the same original context cannot
cross splits. Contexts found in original validation/test data are excluded from
all three compiled splits. Official test records are never used to fill quotas.

Per-task deterministic round-robin selection limits domination by the largest
tasks. This does not balance every class, intent, emotion, or Score level.
Duplicate source IDs and normalized states are removed. A 64-permutation MinHash
LSH audit retrieves possible cross-split near duplicates at threshold 0.75 and
checks exact Jaccard similarity of five-token shingles at 0.90. Validation takes
precedence over calibration, which takes precedence over training. Entire source
groups on the lower-priority side are removed. No padding follows removals.
`near_duplicate_audit.json` identifies the matches and removals.

This is approximate text deduplication, not exhaustive semantic matching. External
benchmark contents have not been downloaded and fingerprinted; benchmark defenses
are family exclusions and original-split/context checks. Base-model pretraining
contamination remains unknown. Cosmos QA, WANLI, bAbI, RuleTaker, and other included
sources must not later be described as entirely unseen evaluation tasks.

## Schema and training contract

Each JSONL line contains `state`, `questions`, `targets`, `id`, `split`, `group_id`,
and `provenance`; GoEmotions also contains `annotation_votes`. Send only `state`
and `questions` to inference. Never put targets, provenance, or annotation votes
in the input prompt. Question objects follow OpenJev's existing API.

- Choice target: the correct key in `criteria`.
- Noul target: `"yes"` or `"no"`.
- Score target: integer index into the ordered `criteria` array.

Training must reuse `Engine.build_schema`, `system_text`, `chat_prompt_ids`,
`groups`, and `resolve_template`. Translate semantic targets to the labels and
single-token IDs produced by the tokenizer. Keep scaffold tokens fixed and
corrupt answer slots using the model's supported diffusion training procedure.
Compute cross-entropy on the allowed label IDs at answer slots only; average
slots within each record before averaging records so expanded records do not
automatically receive triple weight. Mask prompt, scaffold, and padding losses.
Do not turn this dataset into ordinary full-response next-token SFT and assume
that it reproduces structured diffusion reads. Training code is not implemented
or exercised by this dataset build.

First experiment: rank-16 LoRA, hard slot cross-entropy, then temperature scaling
on the calibration split. Keep validation separate for model selection and use
untouched external tests for final claims. Fit positive temperatures by question
type and Choice cardinality, with pooled fallback for sparse buckets. Calibrate
the actual deployed single-read/averaged-read policy; changing temperatures can
also change entropy thresholds and reread behavior. Report accuracy, NLL, Brier,
ECE, Score error, seed variance, reread rate, latency, and sample sizes.

## Limits and release scope

The dataset is English-only and NLI-heavy, with one permissively licensed FLAN
task family. It lacks the intended Gemini-generated tool routing, business
workflow, incident triage, and long application-state examples. Task labels can
contain annotation errors; mechanical provenance checks cannot certify truth.
GoEmotions reflects subjective judgments; HelpSteer contains original model
errors by design. Do not mistake those response texts for endorsed answers.

Sampled inspection covered one seeded training record per task (26 task names).
It found two unsupported/underspecified inherited labels, excluded by original
file reference in `quality_exclusions.yaml`, without changing their labels.
This sample is too small and non-independent to estimate the annotation error
rate. Automated schema/source checks and sampled inspection are not a full human
annotation audit. No fine-tuning, calibration fit, model-quality evaluation, or
public upload is performed by these scripts. An eventual model release should
include the base checkpoint/revision/license, this dataset's hashes, training
code/configuration/seeds, adapter weights, calibration parameters, and measured
evaluation results. Do not claim those artifacts exist before training them.
