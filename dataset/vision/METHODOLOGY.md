# OpenJev vision pool v1

100,000 image records with 200,000 finite decision slots. This is image-conditioned
training/evaluation data, not an image-generation dataset or a trained model.
No Gemini or paid generation API is used.

| Family | Images | Training | Calibration | Locked evaluation |
|---|---:|---:|---:|---:|
| Rendered games | 25,000 | 20,900 | 2,049 | 2,051 |
| Charts, tables, invoices, UI | 30,000 | 27,100 | 951 | 1,949 |
| CLEVR scenes | 25,000 | 20,000 | 1,000 | 4,000 |
| Open Images photographs | 20,000 | 17,000 | 1,000 | 2,000 |
| **Total** | **100,000** | **85,000** | **5,000** | **10,000** |

One physical image accompanies each record. Games have one decision, structured
images and photographs have two, and CLEVR images have three. `train.jsonl`,
`calibration.jsonl`, and `evaluation.jsonl` are the canonical record manifests.
Images live under `images/`, named by opaque hashes rather than labels.
WebDataset-compatible tar shards are independently separated by split.

## Games: use existing verified positions

Render 12,000 chess, 7,000 Gomoku, 3,000 Connect Four, and 3,000 Hold'em
fundamental records from the earlier game datasets. Hold'em mixed-strategy
records are excluded here: these pictures teach card reading and fundamental
reasoning, not a new poker strategy distribution. The original game methodology
and limitations still apply: chess engine labels are approximate, Gomoku covers
short tactics, and Connect Four covers solved endgames.

All original parent training positions remain in training. Parent validation
positions become image evaluation positions; calibration remains calibration.
Keep original parent group IDs. Never turn an existing text-training position
into an image-evaluation item. The source snapshots retain parent record IDs,
group IDs, targets, and original split assignments for audit.

The renderers use Pillow, geometric shapes, and bundled DejaVu fonts. They do not
use the playground's CC-BY-SA chess piece artwork. Both chessboard orientations
are included, with explicit coordinate axes. Chess inputs preserve side to move,
castling rights, and en-passant information that cannot be recovered from piece
placement alone. They omit FEN and move history. SAN check/mate suffixes are
removed from displayed candidate moves to avoid giving away mate-in-one labels.

Gomoku uses the same 15x15 freestyle rules and top-left A15 convention as the
playground. Connect Four displays red X and yellow O pieces with numbered
columns. Hold'em displays only the player's hole cards and community board;
text retains necessary probability/range assumptions but removes the pictured
cards. Annotation metadata never enters model inputs.

## Structured images: known data, computed answers

Generate 6,000 images each of bar charts, line plots, inventory tables, invoices,
and application screens. Labels come from the underlying numerical or UI state:

* Charts: largest value and a threshold comparison.
* Tables: largest opening-to-closing increase and a positive-change check.
* Invoices: displayed amount due and a threshold comparison. Arithmetic uses
  integer cents, including tax rounding.
* UI: locate a named action among four marked controls, and identify a job status.

UI button combinations are injectively selected from 20 actions; this is not
thousands of copies of a 24-permutation four-button menu. These are simplified
synthetic documents/interfaces, not representative screenshots of arbitrary
production applications.

Evaluation uses a held-out purple/serif theme. Evaluation bar charts are
horizontal, table columns reverse Opening/Closing order, and UI controls use
a vertical layout rather than a grid. These layouts are absent from training.
Theme choices consume the same random sequence in every split, so rendering a
given underlying example with another theme does not change its labels.
Not every family has an unseen layout: line plots, invoices, and game piece
glyphs retain their basic representation with a held-out visual theme.

## CLEVR

Use the [official CLEVR v1.0 source](https://cs.stanford.edu/people/jcjohns/clevr/),
under CC BY 4.0. Its image scenes, questions, and functional programs are supplied
by the dataset authors. Retain 21,000 official training images, with 1,000
reserved for calibration, and 4,000 official validation images for evaluation.
Do not use the official test set or move official validation data into training.

Selection takes the first requested members in the original ZIP's member order,
which is not the numeric image-name order. It is a deterministic archive-prefix
sample, not a claim of unbiased sampling. The downloader uses verified HTTP
byte ranges and the original ETag to fetch only the required members. The
local `CLEVR_selected.sparse.zip` is deliberately a **partial sparse archive**;
unselected members are not valid and must not be read. Selected members are
verified by ZIP CRC and image SHA-256.

Select three questions per image with a seeded shuffle. Re-execute their
functional programs against the scene graphs and require agreement with the
published answers. Convert yes/no to Noul, and attributes/counts to Choice
with complete answer domains. Retain original questions, programs, scenes,
attribution, and image IDs in provenance. Original image bytes are preserved.

## Real photographs: per-image licensing, verified labels

Use Open Images' official image metadata and human-verified image-level labels.
The annotation license is CC BY 4.0. The upstream image-license column alone
is **not** treated as sufficient permission evidence.

For every selected photograph:

1. Fetch the original Flickr landing page successfully.
2. Find its image-specific JSON-LD `ImageObject` and match its photo ID to the
   requested landing page.
3. Require an allowlisted CC BY or CC0 license. Reject NC, SA, absent or
   unverified licenses, unavailable pages, and public-domain-mark-only entries.
4. Preserve the original metadata, author, source URL, license declaration,
   retrieval timestamp, and landing-page content hash.
5. Download the corresponding image from the official CVDF/S3 mirror. Preserve
   its download hash, apply EXIF orientation, convert to RGB, resize to fit
   1024x1024, and encode at JPEG quality 92. Record these modifications.

Screen decoded photographs for low pixel variation and near duplicates before
final selection. One low-contrast photograph was replaced in this build; no
near-duplicate candidate passed the rejection threshold. Replacement preserves
the requested split quotas.

The selected photographs have 19,896 CC BY 2.0 licenses, 27 CC BY 4.0 licenses,
and 77 CC0 declarations. This is a documented source-license check, not an
independent adjudication of authorship or every possible third-party right.
Retain attribution when redistributing images and annotations.

Select 17,000 official training photos for training. Use 3,000 official
validation photos for evaluation/calibration; none enter training. Candidates
come from bounded metadata prefixes and surviving license checks. This creates
selection bias toward still-accessible, permissively licensed photos.

For each image, select one human-verified positive label and one explicitly
verified negative label, shuffle their slot order, and emit two Noul questions.
An unannotated class is **never** assumed absent. Source annotations can contain
errors; mechanical consistency checks are not a fresh human relabeling of every
photograph. This release does not include bounding-box, counting, or relationship
questions for photographs, because image-level labels alone do not justify them.

## Splits and evaluation discipline

`evaluation.lock.json` fixes the 10K-image evaluation manifest and its image-hash
manifest. Keep those examples, their parent states, and alternative renders out
of training and calibration. The lock provides reproducible split identity;
the open-source evaluation labels are not secret. Preserve upstream splits and
prior game parent splits rather than randomly splitting rendered images.

The audit checks IDs, source group isolation, targets, image hashes, decoded
pixels, image dimensions, and OpenJev's 5 MiB image limit. It rejects exact
decoded-image duplicates across splits. Photographs receive an additional
conservative near-duplicate screen: 64-bit dHash distance <=2 and mean absolute
32x32 RGB difference <3. This does not prove the absence of every crop, burst,
nearby scene, or base-model pretraining overlap.

The tokenizer audit uses the actual cached OpenJev tokenizer and Gemma4 image
processor. It validates all slot templates and image dimension token estimates,
and checks actual image preprocessing on representative/extreme-aspect images.
It does not run the model or reproduce every deployment-specific vLLM setting.

Use `vision_request.py` to construct API payloads. Only `state`, `questions`, and
image bytes are included; targets, FEN, scene graphs, proofs, and source metadata
remain supervision. The exporter supports normal, blank-image, same-family
mismatched-image, and compressed-image arms. Those controls are prepared, not
model-scored in this build. Compare task accuracy, NLL/Brier/ECE, reread rate,
and robustness by family when actually evaluating a checkpoint. For ambiguous
game choices, use the parent's acceptable-target set rather than treating all
alternative moves as errors.

Do not interpret the pool as evidence that OpenJev already sees or plays better.
No model training, inference benchmark, or calibrated quality gain is claimed.

## Build and reproduce

Python 3.12/Linux, with exact package versions in `requirements-build.txt` and
font files/notices under `assets/`. The first acquisition requires internet but
no API key. Existing text game releases are needed for first-time game selection.

```bash
uv pip install --python .venv/bin/python -r dataset/vision/requirements-build.txt
.venv/bin/python scripts/download_vision_sources.py --part metadata
.venv/bin/python scripts/download_vision_sources.py --part clevr
.venv/bin/python scripts/prepare_vision_photos.py
.venv/bin/python scripts/build_vision_pool.py --part games
.venv/bin/python scripts/build_vision_pool.py --part structured
.venv/bin/python scripts/build_vision_pool.py --part clevr
.venv/bin/python scripts/build_vision_pool.py --part photos
.venv/bin/python scripts/snapshot_vision_sources.py
HF_HUB_OFFLINE=1 .venv/bin/python scripts/finalize_vision_pool.py
```

For a historical rebuild, use the frozen release; live Flickr licenses and URLs
can change. `sources/` includes selected game parents, complete selected CLEVR
annotations, and photo attribution/license evidence. Images are already in the
release. Restore offline inputs, then rebuild each family into a new directory:

```bash
python scripts/snapshot_vision_sources.py --restore RELEASE --output restored-source
python scripts/build_vision_pool.py --source restored-source --output rebuilt --part games
python scripts/build_vision_pool.py --source restored-source --output rebuilt --part structured
python scripts/build_vision_pool.py --source restored-source --output rebuilt --part clevr
python scripts/build_vision_pool.py --source restored-source --output rebuilt --part photos
python scripts/package_vision_pool.py --dataset RELEASE --rebuild rebuilt
```

Packaging verifies all four record parts and every regenerated image, then writes
1,000-image tar shards with image/JSON pairs. `shards/index.json` records per-shard
checksums and split counts. The small metadata archive carries code, manifests,
licenses, source annotations, and audit reports; the image shards are separate.
The release remains directly usable through its JSONL manifests and image paths.
