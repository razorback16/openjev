# OpenJev Texas Hold'em supplement v1

**15,000 procedural, text-only records**, no Gemini, paid API, hand-history
scraping, or external player data. This adds a separately selectable Hold'em
track alongside the 45,300-record games supplement; it does not replace it.

| Task | Records | Ground truth |
|---|---:|---|
| Best five-card hand category | 4,000 | Exact seven-card evaluation |
| Turn-to-river category improvements | 3,000 | Every one of 46 unseen card identities |
| Showdown equity | 3,000 | Exact opponent-hand and river-card enumeration |
| Restricted river strategy | 5,000 | 625 solved subgames, eight information states each |

Split counts: **13,623 train, 635 validation, 742 calibration**. These are
development splits, not an independent benchmark. The build does not train a
model or establish playing strength.

## Scope of the river solver

This is a deliberately restricted heads-up river game, not unrestricted
no-limit Hold'em:

* The board has five community cards; no cards remain to be dealt.
* Public starting ranges contain four hole-card combinations per player.
* Range entries have equal prior weight. Joint deals are uniform over pairs of
  entries with no shared cards; card blockers are explicitly accounted for.
* A may check, ending the game at showdown, or bet its entire remaining stack.
* B may fold or call the bet. B cannot bet after A checks. There are no raises.
* Both players have the same remaining stack. Pot sizes are 20, 40, 80, 100,
  or 200 chips; the all-in amount is 0.25, 0.5, 1, or 1.5 times that initial pot.
* There is no rake. Ties split the pot. No earlier betting history is assumed:
  the supplied ranges define the game's starting belief distribution.

Every input states these rules, the starting pot, remaining stacks, ranges,
player identity, own cards, and public action history within this subgame.
It never includes an actual opponent hand. Listing possible opponent hands
with prior weights is a range assumption, not disclosure of a dealt card.
B observes A's bet and must account for its strategic information; the solver
does so through joint reach probabilities, rather than treating the prior as
an unchanged posterior.

Solve each game with two zero-sum linear programs. Let `q[i,j]` be the compatible
joint-deal probability, `s[i,j]` be 1/0/-1 for A winning/tying/losing showdown,
`P` the original pot, `B` the bet, `x[i]` A's betting probability, and `y[j]` B's
calling probability. A's centered expected payoff is:

```
c + a @ x + x @ M @ y
c = sum(q * s * P/2)
a[i] = sum_j q[i,j] * (P/2 - s[i,j]*P/2)
M[i,j] = q[i,j] * (s[i,j]*(P/2+B) - P/2)
```

Solve A's maximin and B's minimax separately with SciPy/HiGHS. Require a
best-response gap <=1e-7 chips. Retain games with at least one nontrivial mixed
action and positive betting reach for every B information state. This biases
the curriculum toward mixed decisions; it is not a natural-frequency sample.
The build accepted 625 out of 1,288 candidates.

An independent audit constructs the full **16-by-16 normal-form payoff matrix**
from showdown outcomes and actual chip receipts. It checks all pure contingent
plans against the exported behavioral policies. This certifies equilibrium to
numerical tolerance in the stated restricted game. It is not a certificate
for larger ranges, other bet sizes, other streets, or full Hold'em.

`target_distributions.q1` stores action frequencies, not correctness confidence.
Use soft cross-entropy or freshly sampled action targets each epoch. The hard
`targets.q1` value is one reproducible policy sample for schema compatibility.
Always choosing argmax or repeatedly training the same hard sample loses the
mixed strategy. Do not use these river rows to fit confidence temperatures,
including river rows stored in `calibration.jsonl`.

## Fundamentals

**Hand categories:** construct approximately balanced examples across nine
categories, then randomly assign the seven cards to two hole cards and a
five-card board. Reject constructions whose final best hand has a different
category. Ace-low straights, board-only hands, and kickers are supported.
A royal flush belongs to the straight-flush category. These categories are
deliberately balanced rather than sampled at their natural frequencies.

**Draw counts:** on the turn, enumerate all 46 cards not in the player's hand
or board. Count cards that strictly improve the best-five **category**, excluding
kicker improvements within the same category. This is not "outs to win": some
improvements can still lose to an opponent. The exact question makes this
distinction explicit. The menu contains counts 0 through 46.

**Equity:** 1,500 river states use a uniform opponent distribution over all
990 legal two-card hands. Another 1,500 turn states have an explicitly supplied
uniform range of up to four legal hands, with every legal river enumerated
after removing both players' hole cards. Outcomes include ties. Store integer
win/tie/loss counts, their exact probabilities, and equity `P(win)+P(tie)/2`.
No Monte Carlo is used; there is no sampling error under the stated assumptions.
These assumptions can still differ from a real opponent's range.

The Choice target is one of ten equity intervals: lower bound inclusive,
upper bound exclusive, except the last interval includes 100%. Integer arithmetic
handles boundaries. This is a numerical-reasoning task, **not a bet/fold label**.
Outcome probabilities are metadata with separate semantics from action policies.

## Validation and splits

The compiler uses Treys 0.1.8. The audit implements a separate histogram-based
hand evaluator without Treys tables. It independently checks every fundamental
label, including all equity enumerations, and every river game's normal-form
best responses. Tests cover wheel straights, double trips, kicker order, shared
board ties, card blockers, decile boundaries, suit symmetry, and an analytic
bluffing equilibrium.

Canonicalize each board over all 24 global suit relabelings and all board-card
orders; hash that key into a seeded 90/5/5 split. This intentionally groups
different hole cards, questions, and river subgame information states sharing
an equivalent board. All eight decisions from a river game remain together.
The fixed seed is 20260920. No train/validation/calibration board-group overlap
is permitted. This is internal split isolation, not a claim that generic poker
concepts are absent from the base model's pretraining.

JSONL is canonical. Parquet preserves the same columns, with nested dictionaries
serialized as JSON strings. Only `state` and `questions` belong in model inputs.
Do not feed targets, proof matrices, solver policies, or provenance into prompts.
The package includes all build/audit code, tests, licenses, and reports. It uses
no external dataset inputs and can be rebuilt offline once dependencies are
installed. Dependency packages and model assets are not bundled.

## Reproduce

From the repository root, with Python 3.12:

```bash
uv pip install --python .venv/bin/python -r dataset/holdem/requirements-build.txt
.venv/bin/python scripts/build_holdem_dataset.py
.venv/bin/python scripts/audit_holdem_dataset.py
.venv/bin/python scripts/build_holdem_dataset.py --output /tmp/holdem-rebuild
# Requires the same cached tokenizer and extra environment as the public pilot:
HF_HUB_OFFLINE=1 .venv/bin/python scripts/audit_public_dataset.py --dataset data/openjev-holdem/release
.venv/bin/python scripts/package_holdem_dataset.py --rebuild /tmp/holdem-rebuild
.venv/bin/python -m pytest -q tests/test_holdem_dataset.py
```

For a quick smoke build use `--smoke --output /tmp/holdem-smoke`. The package
command checks two independent builds byte-for-byte, audits' hashes, and Parquet
round trips before creating `data/openjev-holdem/openjev-holdem-v1.tar.gz`.
Reproducibility is established in the pinned environment; different solver
versions can choose different equally valid equilibria.

The release folder contains `report.json`, `audit.json`, `tokenizer_audit.json`,
`reproducibility.json`, and `SHA256SUMS`. No Hugging Face token or Gemini key is
needed. The tokenizer audit uses the locally cached tokenizer identified in its
report; actual model weights are unnecessary.
