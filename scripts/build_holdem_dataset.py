"""Exact Hold'em fundamentals and certified restricted river games; no network."""
from __future__ import annotations
import argparse
import collections
import hashlib
import importlib.metadata
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import linprog
from treys import Card, Evaluator

SEED = 20260920
ROOT = Path(__file__).resolve().parents[1]
RANKS, SUITS = '23456789TJQKA', 'cdhs'
DECK = tuple(r+s for r in RANKS for s in SUITS)
CARDS = {c: Card.new(c) for c in DECK}
EVALUATOR = Evaluator()
CATEGORIES = ['high_card', 'one_pair', 'two_pair', 'three_of_a_kind', 'straight',
              'flush', 'full_house', 'four_of_a_kind', 'straight_flush']


def digest(*args):
    return hashlib.sha256(json.dumps(args, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def rank(cards):
    return EVALUATOR.evaluate([], [CARDS[c] for c in cards])


def category(cards):
    # Treys calls a royal flush a separate class 0; it remains a straight flush here.
    return max(0, min(8, 9-EVALUATOR.get_rank_class(rank(cards))))


def canonical_board(board):
    return min(' '.join(sorted(c[0]+dict(zip(SUITS, perm))[c[1]] for c in board))
               for perm in itertools.permutations(SUITS))


def make_record(task, board, state, instruction, options, target, metadata, policy=None):
    key = digest(task, state, instruction)
    group = digest('holdem-board', canonical_board(board))
    fraction = int(digest(SEED, group)[:12], 16)/16**12
    split = 'train' if fraction < .9 else 'validation' if fraction < .95 else 'calibration'
    rng = random.Random(key)
    items = list(options.items()); rng.shuffle(items)
    if policy is not None:
        target = rng.choices(sorted(policy), [policy[k] for k in sorted(policy)])[0]
    row = {'id': key, 'group_id': group, 'split': split, 'state': state,
           'questions': {'q1': {'type': 'choice', 'instructions': instruction, 'criteria': dict(items)}},
           'targets': {'q1': str(target)},
           'provenance': {'family': 'games', 'task': 'texas_holdem', 'subtask': task,
                          'license': 'Apache-2.0', 'source': 'OpenJev deterministic procedural generator',
                          'seed': SEED, 'board': board, **metadata}}
    if policy is not None:
        row['target_distributions'] = {'q1': {k: float(policy[k]) for k, _ in items}}
        row['distribution_semantics'] = 'behavioral_strategy_not_confidence'
        row['loss_recommendation'] = 'soft_cross_entropy; hard target is one seeded policy sample'
    return row


def category_seed(kind, rng):
    rs = rng.sample(list(RANKS), 5)
    suits = lambda n: rng.sample(list(SUITS), n)
    if kind in (4, 8):
        seq = rng.choice(['A2345'] + [RANKS[i:i+5] for i in range(9)])
        s = rng.choice(SUITS)
        return [r+(s if kind == 8 else rng.choice(SUITS)) for r in seq]
    if kind == 7:
        return [rs[0]+s for s in SUITS] + [rs[1]+rng.choice(SUITS)]
    if kind == 6:
        return [rs[0]+s for s in suits(3)] + [rs[1]+s for s in suits(2)]
    if kind == 5:
        s = rng.choice(SUITS); return [r+s for r in rs]
    if kind == 3:
        return [rs[0]+s for s in suits(3)] + [r+rng.choice(SUITS) for r in rs[1:3]]
    if kind == 2:
        return [r+s for r in rs[:2] for s in suits(2)] + [rs[2]+rng.choice(SUITS)]
    if kind == 1:
        return [rs[0]+s for s in suits(2)] + [r+rng.choice(SUITS) for r in rs[1:4]]
    return [r+rng.choice(SUITS) for r in rs]


def state_cards(hero, board):
    return ('Heads-up Texas Hold\'em; standard 52-card deck, no jokers. Use the best five cards from your hole cards and board. '
            'Ranks 2–9,T,J,Q,K,A; suits c=clubs,d=diamonds,h=hearts,s=spades.\n'
            f'Your hole cards: {" ".join(hero)}. Board: {" ".join(board)}.')


def equity(hero, board, opponent_range=None):
    """Exact uniform conditional range/runout enumeration, not Monte Carlo."""
    remaining = [c for c in DECK if c not in hero+board]
    hands = list(itertools.combinations(remaining, 2)) if opponent_range is None else opponent_range
    counts = collections.Counter(win=0, tie=0, loss=0)
    for opp in hands:
        assert len(set(hero+board+list(opp))) == len(hero)+len(board)+2
        runouts = [()] if len(board) == 5 else [(c,) for c in remaining if c not in opp]
        for runout in runouts:
            community = board+list(runout)
            a, b = rank(hero+community), rank(list(opp)+community)
            counts['win' if a < b else 'tie' if a == b else 'loss'] += 1
    return dict(counts)


def equity_bin(counts):
    # Integer arithmetic protects exact decile boundaries; last bin includes 100%.
    return min(9, (10*(2*counts['win']+counts['tie']))//(2*sum(counts.values())))


def fundamentals(n_category=4000, n_draw=3000, n_equity=3000):
    rng = random.Random(SEED)
    out, seen = [], set()
    while len(out) < n_category:
        kind = len(out) % 9
        five = category_seed(kind, rng)
        cards = five+rng.sample([c for c in DECK if c not in five], 2)
        if category(cards) != kind:
            continue
        rng.shuffle(cards); hero, board = cards[:2], cards[2:]
        key = (tuple(sorted(hero)), tuple(sorted(board)))
        if key in seen:
            continue
        seen.add(key)
        out.append(make_record('hand_category', board, state_cards(hero, board),
            'What is your current best five-card hand category? Treat a royal flush as a straight flush.',
            {x: x.replace('_', ' ') for x in CATEGORIES}, CATEGORIES[kind],
            {'hero': hero, 'teacher': 'exact seven-card evaluator', 'category_index': kind}))
    print(f'Hand categories: {n_category}', flush=True)
    for _ in range(n_draw):
        cards = rng.sample(DECK, 6); hero, board = cards[:2], cards[2:]
        current = category(cards)
        outs = [c for c in DECK if c not in cards and category(cards+[c]) > current]
        out.append(make_record('category_improvement_outs', board, state_cards(hero, board),
            'How many of the 46 unseen cards would strictly improve your hand CATEGORY on the river? '
            'A better kicker within the same category does not count. Count card identities, not ranks. '
            'Opponent cards are unknown; this is not the number of outs to win the pot.',
            {str(i): f'{i} cards' for i in range(47)}, len(outs),
            {'hero': hero, 'teacher': 'exact 46-card enumeration', 'improving_cards': outs,
             'current_category': CATEGORIES[current], 'unknown_cards': 46}))
    print(f'Category-improvement draws: {n_draw}', flush=True)
    for i in range(n_equity):
        cards = rng.sample(DECK, 7 if i % 2 == 0 else 6); hero, board = cards[:2], cards[2:]
        opponent_range = None
        if len(board) == 4:
            remaining = [c for c in DECK if c not in cards]
            opponent_range = sorted({tuple(sorted(rng.sample(remaining, 2))) for _ in range(4)})
        counts = equity(hero, board, opponent_range)
        range_text = ('Opponent is uniformly distributed over every legal two-card hand.' if opponent_range is None else
                      'Opponent range: '+', '.join(' '.join(h) for h in opponent_range)+'. Each listed combination is equally likely; all others have zero weight.')
        state = state_cards(hero, board)+'\n'+range_text+' No further betting. Any remaining community card is uniform over the deck after both hands are removed.'
        total = sum(counts.values())
        out.append(make_record('showdown_equity', board, state,
            'Which interval contains your showdown equity, defined as P(win) + 0.5*P(tie)? This does not ask for a betting action.',
            {str(k): f'{k*10}% inclusive to {(k+1)*10}% '+('inclusive' if k == 9 else 'exclusive') for k in range(10)},
            equity_bin(counts), {'hero': hero, 'opponent_range': opponent_range, 'teacher': 'exact joint hand/runout enumeration',
                'outcome_counts': counts, 'outcome_probabilities': {k: v/total for k, v in counts.items()},
                'equity': (counts['win']+.5*counts['tie'])/total, 'enumerated_outcomes': total,
                'probability_semantics': 'showdown_outcomes_not_action_strategy', 'sampling_error': 0}))
        if (i+1) % 500 == 0:
            print(f'Exact equity: {i+1}', flush=True)
    return out


def solve_river(sign, joint, pot, bet):
    """Zero-sum Bayesian check/bet vs fold/call game, solved by two LPs."""
    n, m = sign.shape
    baseline = float(np.sum(joint*sign*pot/2))
    a = np.sum(joint*(pot/2-sign*pot/2), axis=1)
    b = joint*(sign*(pot/2+bet)-pot/2)
    # max a.x + sum(t), t_j <= 0 and t_j <= sum_i B_ij*x_i.
    first = linprog(-np.r_[a, np.ones(m)], A_ub=np.c_[-b.T, np.eye(m)],
                    b_ub=np.zeros(m), bounds=[(0, 1)]*n+[(None, 0)]*m, method='highs')
    # min sum(u), u_i >= 0 and u_i >= a_i + sum_j B_ij*y_j.
    second = linprog(np.r_[np.zeros(m), np.ones(n)], A_ub=np.c_[b, -np.eye(n)],
                     b_ub=-a, bounds=[(0, 1)]*m+[(0, None)]*n, method='highs')
    if not first.success or not second.success:
        raise ValueError('River LP failed')
    x, y = np.clip(first.x[:n], 0, 1), np.clip(second.x[:m], 0, 1)
    best_a = baseline+float(np.maximum(0, a+b@y).sum())
    best_b = baseline+float(a@x+np.minimum(0, x@b).sum())
    gap = best_a-best_b
    if gap > 1e-7 or gap < -1e-7:
        raise ValueError(f'River best-response gap: {gap}')
    return x, y, {'best_response_gap_chips': max(0., gap), 'game_value_centered_chips': baseline+float(a@x+x@b@y),
                  'primal_dual_gap_chips': abs(float(first.fun+second.fun))}


def river_games(n_spots=625):
    rng = random.Random(SEED+1)
    out, attempts, kept = [], 0, 0
    while kept < n_spots:
        attempts += 1
        if attempts > max(1000, n_spots*100):
            raise ValueError('River sampling exhausted')
        board = rng.sample(DECK, 5); remaining = [c for c in DECK if c not in board]
        ranges = []
        for _ in range(2):
            hands = set()
            while len(hands) < 4:
                hands.add(tuple(sorted(rng.sample(remaining, 2))))
            ranges.append(sorted(hands))
        joint = np.array([[not set(a).intersection(b) for b in ranges[1]] for a in ranges[0]], dtype=float)
        if np.any(joint.sum(axis=0) == 0) or np.any(joint.sum(axis=1) == 0):
            continue
        joint /= joint.sum()
        scores = [[rank(board+list(h)) for h in r] for r in ranges]
        sign = np.sign(np.array(scores[1])[None, :]-np.array(scores[0])[:, None])
        pot = rng.choice([20, 40, 80, 100, 200]); bet = int(pot*rng.choice([.25, .5, 1, 1.5]))
        x, y, certificate = solve_river(sign, joint, pot, bet)
        reach = joint.T@x
        # Keep informative mixed games, and only menus reached with positive probability.
        if np.any(reach < 1e-8) or not np.any((np.r_[x, y] > 1e-6) & (np.r_[x, y] < 1-1e-6)):
            continue
        scenario = digest(board, ranges, pot, bet)
        public = (f'Heads-up Texas Hold\'em RIVER SUBGAME. Board: {" ".join(board)}. '
            f'Pot before this decision: {pot} chips. Both players have exactly {bet} chips remaining. No rake.\n'
            f'Restricted rules: A acts first and may check (immediate showdown) or bet all {bet} chips. '
            'After a bet, B may fold or call; no raises or further actions. A fold awards the existing pot to A; '
            'a call goes to showdown. Tied showdowns split the pot.\n'
            'Public starting ranges, before any action in this subgame:\n'+
            '\n'.join(f'{name}: '+', '.join(' '.join(h) for h in hands) for name, hands in zip('AB', ranges))+
            '\nEach range combination has equal prior weight. Deal the two hands independently, conditioned on no shared cards. '
            'These ranges define this subgame; no earlier betting history is assumed. Opponent private cards remain unknown.')
        for player in (0, 1):
            for index, hand in enumerate(ranges[player]):
                policy = ({'check': 1-x[index], 'bet': x[index]} if player == 0 else {'fold': 1-y[index], 'call': y[index]})
                state = public+f'\nYou are player {"AB"[player]}. Your hole cards: {" ".join(hand)}.\n'
                state += 'A has not acted yet; choose your action.' if player == 0 else f'A bet {bet} chips; choose your response.'
                out.append(make_record('restricted_river_strategy', board, state,
                    'Choose an action using the equilibrium mixed strategy for exactly the stated restricted game.',
                    {k: k.capitalize()+(f' {bet} chips' if k in ('bet', 'call') else '') for k in policy}, None,
                    {'teacher': 'zero-sum Bayesian game linear programs', 'scenario_id': scenario, 'scenario_index': kept,
                     'hero': list(hand), 'player': player, 'hand_index': index, 'ranges': ranges,
                     'pot': pot, 'bet': bet, 'joint_prior': joint.tolist(), 'showdown_sign_matrix': sign.tolist(),
                     'policy_a_bet': x.tolist(), 'policy_b_call': y.tolist(),
                     'certificate': certificate, 'label_scope': 'restricted_single_bet_game_only'}, policy))
        kept += 1
    print(f'River: {kept} certified subgames, {len(out)} information states; {attempts} candidates', flush=True)
    return out, attempts


def write(records, output, attempts):
    assert len({r['id'] for r in records}) == len(records)
    groups = {}
    for r in records:
        assert groups.setdefault(r['group_id'], r['split']) == r['split']
        assert r['targets']['q1'] in r['questions']['q1']['criteria']
    output.mkdir(parents=True, exist_ok=True)
    report = {'version': 'openjev-holdem-v1', 'seed': SEED, 'records': len(records), 'unique_board_groups': len(groups),
              'river_generation_attempts': attempts, 'subtasks': dict(collections.Counter(r['provenance']['subtask'] for r in records)),
              'versions': {p: importlib.metadata.version(p) for p in ('treys', 'numpy', 'scipy', 'pyarrow')},
              'compiler_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'splits': {}}
    optional = ['target_distributions', 'distribution_semantics', 'loss_recommendation']
    for split in ('train', 'validation', 'calibration'):
        rows = sorted([r for r in records if r['split'] == split], key=lambda r: r['id'])
        path = output/(split+'.jsonl')
        path.write_text(''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in rows))
        flat = [{k: json.dumps(v, separators=(',', ':')) if isinstance(v, dict) else v
                 for k, v in {**r, **{k: r.get(k) for k in optional}}.items()} for r in rows]
        pq.write_table(pa.Table.from_pylist(flat), output/(split+'.parquet'), compression='zstd')
        report['splits'][split] = {'records': len(rows), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'subtasks': dict(collections.Counter(r['provenance']['subtask'] for r in rows))}
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'data/openjev-holdem/release')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    data = fundamentals(36, 20, 20) if args.smoke else fundamentals()
    rivers, attempts = river_games(5 if args.smoke else 625)
    write(data+rivers, args.output, attempts)
