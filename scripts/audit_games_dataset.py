"""Audit all game records, including independent Connect Four game-tree labels."""
import argparse
import collections
import functools
import json
from pathlib import Path

import numpy as np
import pyspiel

import build_games_dataset as g


def audit(directory):
    records = [json.loads(line) for split in ('train', 'validation', 'calibration')
               for line in (directory / (split + '.jsonl')).open()]
    report = g.validate(records)
    counts = collections.Counter()
    labels = collections.Counter()
    positions = collections.Counter()
    sides = collections.Counter()
    depths = collections.Counter()
    pieces = collections.Counter()
    poker = {name: g.poker_states(pyspiel.load_game(name, {'suit_isomorphism': True} if name == 'leduc_poker' else {}))
             for name in ('kuhn_poker', 'leduc_poker')}
    for i, row in enumerate(records):
        meta = row['provenance']; name = meta['task']
        counts[name] += 1; labels[f'{name}/{meta["label_kind"]}'] += 1
        options = row['questions']['q1']['criteria']
        positions[f'{name}/{len(options)}/{list(options).index(row["targets"]["q1"])}'] += 1
        sides[f'{name}/{meta.get("side_to_move", meta.get("player"))}'] += 1
        if name == 'chess':
            fen = meta['fen']
            state = pyspiel.load_game('chess').new_initial_state(fen if len(fen.split()) == 6 else fen + ' 0 1')
            assert set(g.chess_options(state)) == set(options)
            depths[str(meta.get('depth', 'puzzle'))] += 1
            pieces[str(sum(c.isalpha() for c in fen.split()[0]))] += 1
        elif name == 'gomoku':
            board = np.array(meta['board'], dtype=np.int8); turn = meta['side_to_move']
            assert int(np.sum(board == 1)) - int(np.sum(board == 2)) == (turn == 2)
            assert not g.gomoku_terminal(board)
            assert set(options) == {g.coord(c) for c in g.gomoku_candidates(board, turn)}
            mine, theirs = g.gomoku_wins(board, turn), g.gomoku_wins(board, 3-turn)
            for target in row['acceptable_targets']['q1']:
                cell = (15-int(target[1:]))*15 + ord(target[0])-65
                if meta['label_kind'] == 'win_in_one':
                    assert cell in mine
                elif meta['label_kind'] == 'block_only_immediate_win':
                    assert not mine and theirs == {cell}
                else:
                    assert not mine and not theirs
                    board[cell] = turn
                    assert len(g.gomoku_wins(board, turn)) >= 2
                    board[cell] = 0
        elif name == 'connect_four':
            state = pyspiel.load_game('connect_four').new_initial_state()
            for col in meta['history_columns']:
                assert not state.is_terminal()
                state.apply_action(col-1)
            assert not state.is_terminal()
            assert set(options) == {str(a+1) for a in state.legal_actions()}
            player = state.current_player()
            @functools.lru_cache(maxsize=None)
            def solve(history):
                node = state.clone()
                for action in history:
                    node.apply_action(action)
                if node.is_terminal():
                    return int(node.player_return(player))
                scores = [solve(history+(a,)) for a in node.legal_actions()]
                return (max if node.current_player() == player else min)(scores)
            expected = {str(a+1): solve((a,)) for a in state.legal_actions()}
            assert expected == meta['action_values']
            assert set(row['acceptable_targets']['q1']) == {a for a, v in expected.items() if v == max(expected.values())}
        else:
            state = poker[name][meta['information_state']]
            assert row['state'] == g.poker_description(name, state)
            assert set(options) == {str(a) for a in state.legal_actions()}
        if (i+1) % 10000 == 0:
            print(f'Game rules audited: {i+1}', flush=True)
    # Evaluate the actual serialized policies, including float round trips, over the full game.
    report['serialized_policy_exploitability'] = {}
    for name in poker:
        game = pyspiel.load_game(name, {'suit_isomorphism': True} if name == 'leduc_poker' else {})
        table = {r['provenance']['information_state']: [(int(a), p) for a, p in r['target_distributions']['q1'].items()]
                 for r in records if r['provenance']['task'] == name}
        policy = pyspiel.TabularPolicy(table)
        value = pyspiel.exploitability(game, policy)
        assert value < .01
        report['serialized_policy_exploitability'][name] = value
    report.update({'games': dict(counts), 'label_kinds': dict(labels), 'sides': dict(sides),
                   'chess_depths': dict(depths), 'chess_piece_counts': dict(pieces),
                   'target_positions': dict(positions), 'connect_four_independent_solver_verified': counts['connect_four']})
    (directory / 'game_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('target_positions', 'chess_depths', 'chess_piece_counts')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=Path('data/openjev-games/release'))
    audit(p.parse_args().dataset)
