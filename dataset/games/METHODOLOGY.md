# OpenJev games supplement v1

This is a separate, text-only supplement to the public NLP pilot: **45,300
decisions**, produced without Gemini, paid APIs, human-game scraping, or gated
datasets. It contains 40,785 training, 2,246 validation, and 2,269 calibration
records. No model has been trained or evaluated by this data build.

| Game | Total | Label source |
|---|---:|---|
| Chess | 25,000 | 10,000 Lichess puzzles; 15,000 published engine evaluations |
| Gomoku | 15,000 | 5,000 immediate wins, 5,000 necessary blocks, 5,000 proven double threats |
| Connect Four | 5,000 | Exact search of legal endgames, 4–10 empty cells |
| Kuhn poker | 12 | Every distinct player information state; CFR+ average policy |
| Leduc poker | 288 | Every rank-only player information state; CFR+ average policy |

The output is `data/openjev-games/release/{train,validation,calibration}.jsonl`
and equivalent Parquet files. `data/openjev-games/openjev-games-v1.tar.gz` is the
self-contained release archive. It includes the frozen chess source subsets,
compiler, tests, audits, source manifest, and license notices. Large artifacts
are gitignored; the methodology, code, and reports can be committed normally.

## Reproduce

Python 3.12 was used; exact dependency versions are in `requirements-build.txt`.
From the repository root:

```bash
uv pip install --python .venv/bin/python -r dataset/games/requirements-build.txt
# Only needed to obtain a NEW source snapshot; not an exact historical rebuild:
.venv/bin/python scripts/download_game_sources.py
.venv/bin/python scripts/build_games_dataset.py
.venv/bin/python scripts/audit_games_dataset.py
HF_HUB_OFFLINE=1 .venv/bin/python scripts/audit_public_dataset.py --dataset data/openjev-games/release
.venv/bin/python scripts/build_games_dataset.py --output /tmp/openjev-games-rebuild
.venv/bin/python scripts/verify_games_rebuild.py --other /tmp/openjev-games-rebuild
.venv/bin/python scripts/package_games_dataset.py
```

The downloader freezes the first 60,000 puzzle rows and 120,000 evaluation rows
from the [official Lichess exports](https://database.lichess.org/). These are
prefix samples, **not representative random samples** of all Lichess data.
The upstream URLs change over time. For an exact rebuild, extract the release
and use its bundled `source/` with `--source source --output rebuilt`. The builder
verifies every source SHA-256. Pinning just the live download URL is insufficient.
The tokenizer audit additionally requires the cached tokenizer named in its
report; model weights and tokenizer assets are not included in the archive.

## Chess

Standard chess only. OpenSpiel 2.0.2 generates the legal SAN choices. For a
puzzle, apply the first UCI move (the opponent's move), then use the second move
as the answer. Verify the entire published continuation. Require popularity
at least 80, at least 20 plays, and 2–128 legal moves. Include every checkmating
move as acceptable for a mate-in-one puzzle. One puzzle position per original
game is retained. Both colors are equally represented across the full chess set.

For evaluation records, choose the deepest analysis, breaking depth ties by
node count. Require depth >=20 and >=100 thousand nodes. Use the first PV's move;
also retain published moves with the same exact score at that depth as acceptable.
These are bounded-search teacher labels, not proofs of optimality. Unlisted
equally good moves can exist. Centipawn evaluations are **not** confidence labels.
Convert standard-chess castling from the source's UCI_Chess960 king-to-rook
notation before legal-move parsing.

Evaluation FENs lack clocks and history. Inputs explicitly say so. Internal
legal-move generation supplies dummy clocks, which are never presented as actual
history. Do not infer repetition or fifty-move outcomes from these rows.
Puzzles and evaluation positions are deduplicated using the first four FEN
fields. Engine-evaluation rows lack game IDs, so whole-game isolation between
that source and the puzzles cannot be guaranteed. There is no claim of a clean
external chess benchmark or absence of base-model pretraining contamination.

The playground uses FEN and SAN menus, so these records exercise its text path.
This release does not use the playground's CC-BY-SA chess artwork. It does not
train image understanding or change the playground's inference settings.

## Gomoku

Match the playground: 15x15 freestyle, X first, five **or more** in a row wins,
no Renju restrictions. A15 is the top-left cell and O1 the bottom-right.
Generate reachable, nonterminal positions with valid X/O counts, motif stones,
and random surrounding stones. Since the final board has no winning line,
alternating placements of these same stones also cannot have an earlier win.
Balance both sides and the three tactical categories.

Labels are checked across all five-cell windows, including broken lines:

* Immediate win: a move completes five or more.
* Necessary block: the opponent has exactly one immediate winning square and
  the current player has no immediate win. Blocking is necessary to avoid losing
  on the next move; it does **not** guarantee a draw or eventual win.
* Double threat: neither player has an immediate win, and a move creates at least
  two distinct winning squares. The opponent cannot block both in one turn.

Port the playground's radius-two, contiguous-line heuristic to form 16 choices,
generalized to the current side. Measure whether this menu retains a proven
move **before** filtering; 15,000 of 15,001 verified candidates passed in this
build. This is recall on generated tactical cases, not strategic-game recall.
Record all verified moves, and retain all acceptable ones present in the menu.
The data does not teach openings, quiet positional play, or deep Gomoku strategy.

## Connect Four

Generate legal alternating play, avoiding terminal moves until 32–38 discs have
been placed. Keep one position per generated game. Solve every legal root move
to termination using a bitboard negamax solver; discard menus where all moves
have the same game value. Store all action values from the side-to-move
perspective (-1 loss, 0 draw, +1 win), and all best moves as acceptable.

The audit independently reconstructs **every** game in OpenSpiel and uses a
separate complete game-tree minimax implementation to verify every root label.
These are exact endgame labels, not a representative distribution of full games.

## Poker: strategy is not confidence

Use two-player Kuhn and limit Leduc. Leduc removes interchangeable suit identities
(`suit_isomorphism=true`), yielding 288 unique information states. Enumerate the
complete game tree, but emit each observable information state only once.
The input contains the player's own card and public history, never the opponent's
private card or an unrevealed future card. The prose rules and legal actions
match the pinned OpenSpiel configuration.

Run deterministic CFR+ for 10,000 iterations and export its **average** policy.
The build rejects teacher exploitability above 0.01 chips/player. This build's
values are approximately 0.00000963 for Kuhn and 0.00000542 for Leduc. The audit
recomputes exploitability from the actual serialized distributions across all
splits. These measure the **teacher policy**, not OpenJev.

`target_distributions.q1` is a mixed behavioral strategy, with semantic action
keys matching `questions.q1.criteria`. Train with soft cross-entropy, or freshly
sample targets from the stored distribution each epoch. `targets.q1` contains
one reproducible sample for compatibility, **not** the highest-probability action;
repeating that one sample loses mixed-strategy information. No CE trainer has
been added by this dataset change.

Do not fit the ordinary confidence temperature on poker rows, even if a row is
in the `calibration` file. Do not evaluate poker with hard-label accuracy. Use
strategy divergence and full-game exploitability, and sample the emitted policy
when playing. OpenJev currently returns an argmax Choice: a poker integration
must explicitly sample the returned probabilities. Full Texas Hold'em is not
included. The tiny poker set is not padded with duplicate hidden worlds.

## Splits, schema, and training use

Use a seeded SHA-256 90/5/5 group split and deterministic option shuffling.
Group Gomoku by all eight board symmetries plus side to move; group Connect Four
by reflected canonical position; group poker by observable information state.
Chess uses puzzle game IDs or position keys. Reject duplicate IDs and chess
positions. No rotated copies or paraphrases are added.

Kuhn has only 12 rows: this hash split leaves 11 in training and 1 in calibration,
none in validation. Treat it as a tiny protocol curriculum; assess a learned
poker policy through full-game evaluation rather than these tiny split counts.
Leduc's suit aliases have already been removed. These are internal development
splits, not independently sourced benchmarks.

The JSONL schema extends the NLP pilot with `acceptable_targets`,
`target_distributions`, `distribution_semantics`, and `loss_recommendation`.
Parquet keeps nested dictionaries as JSON **strings under the same column names**;
parse those fields after loading. Optional columns are present consistently.
Only `state` and `questions` belong in model inputs. Targets, proofs, teacher
evaluations, and provenance are supervision/metadata and must not enter prompts.
Train/evaluate ordinary decision rows with their hard/acceptable labels, and
poker rows with their strategy distributions.

Keep this supplement separately selectable when mixing with the NLP pilot.
Validation and calibration files must not be concatenated into training. No
mixture weight or improved playing strength is established by creating the data.

## Evidence and limitations

`report.json` records teacher settings, rejection counts, split counts, package
versions, and source/compiler hashes. `game_audit.json` checks labels, menus,
counts, and serialized poker policies. `tokenizer_audit.json` verifies every
record against actual OpenJev templates and tokenizer limits. The release's
`reproducibility.json` compares independent builds. Tests cover puzzle indexing,
castling, Gomoku overlines/symmetries, independent Connect Four solving, and hidden
information isolation. Dataset creation alone is not evidence of model quality.
