# OpenJev slot-training feasibility pilot

This experiment tests whether a small decoder adaptation improves one-pass
structured decisions. It does not establish broad model quality or image ability.

## Reproduction

1. Build the public and games datasets using their documented compilers.
2. Run `python scripts/build_procedural_pilot.py` and
   `python scripts/prepare_training_pilot.py` in the repository environment.
3. Copy the resulting train, validation, and calibration JSONL files to a GPU
   directory. Run `python scripts/train_slot_pilot.py --data DIRECTORY` with
   PyTorch 2.9.1 CUDA 12.8, Transformers 5.17.0, and Accelerate 1.15.0.
4. Copy `outputs/` back to `data/openjev-training-pilot/outputs` and run
   `python scripts/analyze_training_pilot.py`.

The pilot reserves 512 training records (384 public, 64 games, 64 original
procedural workflows), 96 validation records, and 48 calibration records. The
128 optimizer steps consume 256 training records once, in frozen shuffled
order, with two records per update. No validation-based early stopping or
checkpoint selection is used. Source groups are disjoint across splits.
The compiler rejects prompts longer than 1,536 tokens rather than truncating
away evidence or options. This introduces a short-context selection bias.

## Training semantics

Base: `google/diffusiongemma-26B-A4B-it`, revision
`f7f5b7f5fa82ffc52addd066915886d497f5517b`, BF16. The production OpenJev compiler
constructs prompts, answer scaffolds, allowed token IDs, and slot positions.
Every answer slot is replaced with a random vocabulary token; the correct target
is never inserted into the input canvas. All slot predictions are made in one
bidirectional decoder pass, with no autoregressive label shift.

The loss is cross entropy over each slot's allowed labels, averaged over the
record's slots. The frozen model's original output projection and logit softcap
are retained. Only decoder attention query/value projections receive rank-16
LoRA updates (alpha 32). The encoder runs frozen, without these adapters. AdamW
uses learning rate 1e-4, weight decay 0.01, gradient clipping 1.0, and seed 818.

This intentionally narrow experiment keeps memory and cost low. Its adapter is
a custom PyTorch state dictionary, not a drop-in PEFT or vLLM adapter. Because
the base ties encoder/decoder weights, merging the decoder delta into both would
change the tested model. Deployment support needs a separate implementation and
parity test. Base weights are not redistributed; this DiffusionGemma checkpoint
is explicitly [Apache-2.0](https://huggingface.co/google/diffusiongemma-26B-A4B-it).
Repository code, the adapter, and original procedural data use Apache-2.0;
public records retain their upstream licenses.

## Evaluation

Three runs use initialization/noise seeds 818, 819, and 820, with the same
hyperparameters and fixed data order. The two replications were added after the
first result to check repeatability; no best-seed selection is used.

The same held-out records and two fixed canvas seeds (19 and 73) are evaluated
before and after training. Report accuracy, NLL, multiclass Brier score,
10-bin ECE, family metrics, and prediction disagreement between seeds.
Bootstrap accuracy deltas by record, keeping each record's seeds and slots
together. Report training-run variation separately from canvas-seed variation.
Main accuracy uses the single reference label; game accuracy also checks all
solver-certified acceptable moves, avoiding penalties for certified alternatives.

Fit one global temperature for the frozen baseline and one per adapter using
only the separate calibration set, then evaluate them on validation. This
controls for the gains achievable by temperature scaling alone.
Do not optimize temperatures or hyperparameters on
validation. The sample is too small to support reliable per-bucket calibration.
No actual production reread rate is measured. No image records are used, and
the locked 10K image evaluation split remains untouched.

## Native generation and release scope

`scripts/generate_native_pilot.py` made 96 samples using exactly
`gemini-3.7-flash`: 3,878 input tokens and 13,041 output/thinking tokens,
approximately $0.0518 at the introductory $0.75/$3.75 per-million rates.
These samples are retained separately for inspection, **excluded from training
and the permissive release**. Google's [Gemini API terms](https://ai.google.dev/gemini-api/terms)
restrict development of competing models. No assumption is made that generated
output ownership removes that usage restriction. The pilot instead uses
original structured workflow states with explicit policies and mechanically
derived labels. The procedural generator does not consume Gemini output.

## Resource controls

One NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB), quoted $2.09/hour.
`scripts/provision_pilot.py` requests provider-side deletion two hours after
creation, with no persistent/network volume. It rejects a quoted rate above
$2.30/hour. The workload also has a 110-minute process timeout. Results must be
copied locally and the pod deleted promptly when finished; the provider deadline
is only a backup. Check the actual final run report for costs and completion.
