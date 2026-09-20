"""Independent hand evaluator and exhaustive pure-strategy best-response audit."""
import argparse
import collections
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

import build_holdem_dataset as build

VALUES = {c: (build.RANKS.index(c[0])+2, c[1]) for c in build.DECK}


def straight(ranks):
    values = set(ranks)
    if 14 in values:
        values.add(1)
    return next((high for high in range(14, 4, -1) if set(range(high-4, high+1)) <= values), 0)


def score(cards):
    """Independent histogram evaluator; greater tuple is stronger. No Treys lookups."""
    counts = collections.Counter(VALUES[c][0] for c in cards)
    suits = collections.defaultdict(list)
    for c in cards:
        value, suit = VALUES[c]; suits[suit].append(value)
    flush = next((sorted(rs, reverse=True) for rs in suits.values() if len(rs) >= 5), None)
    if flush and straight(flush):
        return (8, straight(flush))
    quads = sorted((r for r, n in counts.items() if n == 4), reverse=True)
    if quads:
        return (7, quads[0], max(r for r in counts if r != quads[0]))
    trips = sorted((r for r, n in counts.items() if n >= 3), reverse=True)
    pairs = sorted((r for r, n in counts.items() if n >= 2), reverse=True)
    if trips and any(r != trips[0] for r in pairs):
        return (6, trips[0], max(r for r in pairs if r != trips[0]))
    if flush:
        return (5, *flush[:5])
    run = straight(counts)
    if run:
        return (4, run)
    if trips:
        return (3, trips[0], *sorted((r for r in counts if r != trips[0]), reverse=True)[:2])
    if len(pairs) >= 2:
        return (2, *pairs[:2], max(r for r in counts if r not in pairs[:2]))
    if pairs:
        return (1, pairs[0], *sorted((r for r in counts if r != pairs[0]), reverse=True)[:3])
    return (0, *sorted(counts, reverse=True)[:5])


def independent_equity(hero, board, opponent_range):
    remaining = [c for c in build.DECK if c not in hero+board]
    hands = itertools.combinations(remaining, 2) if opponent_range is None else opponent_range
    counts = collections.Counter(win=0, tie=0, loss=0)
    cached = {}
    for opp in hands:
        runouts = [None] if len(board) == 5 else [c for c in remaining if c not in opp]
        for river in runouts:
            community = board+([river] if river else [])
            if river not in cached:
                cached[river] = score(hero+community)
            a, b = cached[river], score(list(opp)+community)
            counts['win' if a > b else 'tie' if a == b else 'loss'] += 1
    return dict(counts)


def normal_form_gap(board, ranges, pot, bet, x, y):
    """Enumerate all 16 pure contingent plans for each player, independently of LP."""
    plans = np.array(list(itertools.product([0, 1], repeat=4)))
    matrix = np.zeros((16, 16)); deals = 0
    for i, a in enumerate(ranges[0]):
        for j, b in enumerate(ranges[1]):
            if set(a).intersection(b):
                continue
            deals += 1
            sa, sb = score(board+list(a)), score(board+list(b))
            share = 1 if sa > sb else .5 if sa == sb else 0
            # A's share of the original pot plus net subsequent betting receipts.
            check = pot*share
            called = (pot+2*bet)*share-bet
            matrix += np.where(plans[:, i, None] == 0, check,
                               np.where(plans[None, :, j] == 0, pot, called))
    matrix /= deals
    px = np.prod(np.where(plans, np.array(x), 1-np.array(x)), axis=1)
    py = np.prod(np.where(plans, np.array(y), 1-np.array(y)), axis=1)
    return float(np.max(matrix@py)-np.min(px@matrix))


def audit(directory):
    rows = [json.loads(line) for split in ('train', 'validation', 'calibration') for line in (directory/(split+'.jsonl')).open()]
    seen, groups, games = set(), {}, {}
    counts = collections.Counter(); max_gap = 0
    for i, row in enumerate(rows):
        assert row['id'] not in seen; seen.add(row['id'])
        assert groups.setdefault(row['group_id'], row['split']) == row['split']
        m = row['provenance']; board, hero = m['board'], m['hero']; task = m['subtask']
        assert len(hero) == 2 and len(set(hero+board)) == len(hero+board)
        assert all(c in build.DECK for c in hero+board)
        assert row['group_id'] == build.digest('holdem-board', build.canonical_board(board))
        criteria = row['questions']['q1']['criteria']; target = row['targets']['q1']
        assert target in criteria and 2 <= len(criteria) <= 128
        counts[task] += 1
        if task == 'hand_category':
            assert target == build.CATEGORIES[score(hero+board)[0]]
        elif task == 'category_improvement_outs':
            base = score(hero+board)[0]
            cards = [c for c in build.DECK if c not in hero+board and score(hero+board+[c])[0] > base]
            assert cards == m['improving_cards'] and int(target) == len(cards)
        elif task == 'showdown_equity':
            counts_exact = independent_equity(hero, board, m['opponent_range'])
            assert counts_exact == m['outcome_counts']
            assert int(target) == build.equity_bin(counts_exact)
            total = sum(counts_exact.values())
            assert total == m['enumerated_outcomes']
            assert abs(m['equity']-(counts_exact['win']+.5*counts_exact['tie'])/total) < 1e-12
        else:
            policy = row['target_distributions']['q1']
            assert set(policy) == set(criteria)
            assert all(0 <= p <= 1 for p in policy.values()) and abs(sum(policy.values())-1) < 1e-12
            player, h = m['player'], m['hand_index']
            p = m['policy_a_bet'][h] if player == 0 else m['policy_b_call'][h]
            assert policy == ({'check': 1-p, 'bet': p} if player == 0 else {'fold': 1-p, 'call': p})
            assert list(m['ranges'][player][h]) == hero
            scenario = m['scenario_id']
            games.setdefault(scenario, []).append(row)
            if len(games[scenario]) == 1:
                gap = normal_form_gap(board, m['ranges'], m['pot'], m['bet'], m['policy_a_bet'], m['policy_b_call'])
                assert -1e-7 <= gap <= 1e-7
                max_gap = max(max_gap, gap)
        if (i+1) % 3000 == 0:
            print(f'Independently audited {i+1}', flush=True)
    for same_game in games.values():
        assert len(same_game) == 8
        assert len({r['split'] for r in same_game}) == 1
        assert len({(r['provenance']['player'], r['provenance']['hand_index']) for r in same_game}) == 8
    report = {'records': len(rows), 'subtasks': dict(counts), 'certified_river_games': len(games),
              'independent_normal_form_max_best_response_gap_chips': max_gap,
              'independent_evaluator_verified_all_fundamentals': True, 'duplicate_ids': 0,
              'cross_split_board_group_overlap': 0, 'invalid_targets': 0,
              'split_sha256': {s: hashlib.sha256((directory/(s+'.jsonl')).read_bytes()).hexdigest() for s in ('train', 'validation', 'calibration')}}
    (directory/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=Path('data/openjev-holdem/release'))
    audit(p.parse_args().dataset)
