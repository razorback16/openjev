# OpenJev pilot results — 2026-09-20

The pipeline works on one RTX PRO 6000, but this pilot supports a probability-quality improvement, not a demonstrated improvement in decision accuracy or game strength.

Three rank-16 decoder-only LoRA runs used the same 256 training records, 128 optimizer updates, and seeds 818/819/820. All were evaluated on the same 96 held-out records (199 slots, each with two canvas seeds). Temperatures were fitted on a separate 48-record calibration set. No checkpoint or seed was selected for best validation performance.

| Model | Accuracy | NLL ↓ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| Frozen base | 64.82% | 1.295 | 0.558 | 0.233 |
| Frozen base + temperature | 64.82% | 0.822 | 0.457 | 0.108 |
| Adapter, mean of 3 runs | 65.66% | 0.833 | 0.437 | 0.117 |
| Adapter + temperature, mean of 3 runs | 65.66% | 0.749 | 0.410 | 0.042 |

Accuracy improvements were only 0.75–1.01 percentage points; every paired record-bootstrap 95% interval included zero. After calibrating both models, the adapter NLL improved by 0.049–0.088. Two of three per-run NLL intervals excluded zero; these are correlated results on one small evaluation set, not three independent evaluation datasets.

Probability calibration is therefore worth testing first: it delivers much of the benefit without training. The adapter adds a smaller, promising improvement. None of this establishes production reliability.

The games slice contains only six chess and six Gomoku records. Solver-acceptable move accuracy was 4.17% for the base and 4.17%, 4.17%, and 0% for the three adapters. Do not claim stronger game play. Simple policy-based workflows were already 100% correct before training, so that subset is too easy to demonstrate a gain.

Canvas-seed prediction disagreement worsened from 4.52% to 9.05–11.06%. The actual serving engine's reread rate was not measured. A next pilot should include a consistency objective or richer noise coverage and a larger untouched game evaluation, alongside a temperature-only control. Do not scale to a large paid run on these results alone.

One 96 GB RTX PRO 6000 completed the experiment. Observed Runpod balance decrease: $0.5834; estimated Gemini generation: $0.0518; combined approximately $0.64. Runpod's itemized billing had not populated at cleanup, so the GPU figure is a balance delta. The pod was deleted and the provider reported no remaining pods and zero hourly spend. No network volume was created.

All three adapters and logits are saved locally. Remote/local adapter SHA-256 hashes match. A fresh-process reload of seed 818 reproduced 28 held-out slot-logit vectors across 12 records exactly; seeds 819/820 passed in-process reload checks. Repository tests: 68 passed.

The adapters use frozen BF16 DiffusionGemma with decoder-only query/value updates. They are custom PyTorch adapters, not directly usable in the current NVFP4/vLLM server. No images were trained on or evaluated in this experiment. Deployment parity and larger held-out evaluation remain necessary.

Gemini 3.7 Flash generated 96 examples for about five cents. Those samples are excluded from training and this release because of the [Gemini Developer API restriction on developing competing models](https://ai.google.dev/gemini-api/terms). Original, mechanically labeled workflow records replaced them. Source records retain Apache-2.0, CC BY 4.0, BSD-3-Clause, or CC0 terms; `sources.jsonl` records provenance. DiffusionGemma itself and these adapter modifications are Apache-2.0.

See [the experiment methodology](../PILOT-TRAINING.md) and `summary.json` for exact metrics and uncertainty. The separate 100K-image pool is complete, audited, and packaged; its locked 10K evaluation split was untouched.
