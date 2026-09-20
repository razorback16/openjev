"""Offline, reproducible game decisions. See dataset/games/METHODOLOGY.md."""
from __future__ import annotations

import argparse
import collections
import csv
import functools
import hashlib
import importlib.metadata
import json
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyspiel

SEED = 20260919
ROOT = Path(__file__).resolve().parents[1]


def digest(*parts):
    return hashlib.sha256(json.dumps(parts, separators=(',', ':'), sort_keys=True).encode()).hexdigest()


def split_for(group):
    n = int(digest(SEED, group)[:12], 16) / 16**12
    return 'train' if n < .9 else 'validation' if n < .95 else 'calibration'


def record(game, key, state, options, good, metadata, group=None, distribution=None):
    rng = random.Random(digest(SEED, game, key))
    items = list(options.items())
    rng.shuffle(items)
    if distribution is None:
        target = rng.choice(sorted(good))
    else:
        names = sorted(distribution)
        target = rng.choices(names, [distribution[k] for k in names])[0]
    group = group or digest(game, key)
    out = {'id': digest('games-v1', game, key), 'state': state,
           'questions': {'q1': {'type': 'choice', 'instructions': {
               'chess': 'Choose the strongest legal move for the side to move.',
               'gomoku': 'Choose a move that wins immediately, prevents an immediate loss, or forces a win.',
               'connect_four': 'Choose a legal column giving the best outcome with perfect play.',
               'kuhn_poker': 'Choose an action using the equilibrium mixed strategy.',
               'leduc_poker': 'Choose an action using the equilibrium mixed strategy.',
           }[game], 'criteria': dict(items)}},
           'targets': {'q1': target}, 'split': split_for(group), 'group_id': group,
           'provenance': {'family': 'games', 'task': game,
                          'license': 'CC0-1.0' if game == 'chess' else 'Apache-2.0',
                          'original_split': 'generated' if game != 'chess' else 'public_export',
                          **metadata}}
    if distribution is not None:
        out['target_distributions'] = {'q1': {k: distribution[k] for k, _ in items}}
        out['distribution_semantics'] = 'behavioral_strategy_not_confidence'
        out['loss_recommendation'] = 'soft_cross_entropy; targets is one seeded policy sample'
    else:
        out['acceptable_targets'] = {'q1': sorted(good)}
    return out


def chess_position_key(fen):
    # Ignore clocks: transpositions must not cross splits. Ep is normalized by OpenSpiel.
    return ' '.join(fen.split()[:4])


def parse_uci(state, uci, chess960=False):
    if chess960:
        # Lichess evaluations encode standard-chess castles as king-to-rook UCI.
        castle = {'e1h1': ('K', 'e1g1'), 'e1a1': ('Q', 'e1c1'),
                  'e8h8': ('k', 'e8g8'), 'e8a8': ('q', 'e8c8')}
        if uci in castle and castle[uci][0] in state.board().to_fen().split()[2]:
            uci = castle[uci][1]
    action = state.parse_move_to_action(uci)
    if action not in state.legal_actions():
        raise ValueError(f'Illegal source move {uci}')
    return action


def chess_options(state):
    return {state.action_to_string(a): f'Play {state.action_to_string(a)}.' for a in state.legal_actions()}


def build_chess(source, puzzle_count, evaluation_count, stats):
    game = pyspiel.load_game('chess')
    result, seen, games = [], set(), set()
    side_counts = collections.Counter()
    with (source / 'puzzle.csv').open() as f:
        for i, row in enumerate(csv.DictReader(f)):
            if len(result) >= puzzle_count:
                break
            if int(row['Popularity']) < 80 or int(row['NbPlays']) < 20:
                continue
            game_id = row['GameUrl'].split('/')[3].split('#')[0]
            if game_id in games:
                continue  # One position per source game; never split related puzzle episodes.
            state = game.new_initial_state(row['FEN'])
            moves = row['Moves'].split()
            state.apply_action(parse_uci(state, moves[0]))
            fen = state.board().to_fen()
            key = chess_position_key(fen)
            side = fen.split()[1]
            if key in seen or side_counts[side] >= (puzzle_count + 1) // 2:
                continue
            options = chess_options(state)
            if not 2 <= len(options) <= 128:
                continue
            target = state.action_to_string(parse_uci(state, moves[1]))
            good = [target]
            if 'mateIn1' in row['Themes'].split():
                good = [state.action_to_string(a) for a in state.legal_actions()
                        if state.child(a).is_terminal() and state.child(a).player_return(state.current_player()) == 1]
                if target not in good:
                    raise ValueError('Source mate-in-one did not verify')
            # Validate the complete published continuation, not just the first answer.
            continuation = state.clone()
            for move in moves[1:]:
                continuation.apply_action(parse_uci(continuation, move))
            result.append(record('chess', key,
                f'Chess position (FEN): {fen}\nMove history before this position is unavailable.\n'
                f'{"White" if side == "w" else "Black"} to move.', options, good,
                {'source': 'Lichess puzzles', 'source_ref': f'puzzle.csv:{i}',
                 'upstream': 'https://database.lichess.org/#puzzles', 'game_id': game_id,
                 'puzzle_id': row['PuzzleId'], 'rating': int(row['Rating']), 'themes': row['Themes'].split(),
                 'teacher': 'published Lichess puzzle solution', 'target_uci': moves[1],
                 'fen': fen, 'side_to_move': side, 'label_kind': 'puzzle_solution'}, digest('chess-game', game_id)))
            seen.add(key); games.add(game_id); side_counts[side] += 1
    stats['chess_puzzles'] = len(result)
    evaluation_sides = collections.Counter()
    count = 0
    with (source / 'eval.jsonl').open() as f:
        for i, line in enumerate(f):
            if count >= evaluation_count:
                break
            row = json.loads(line)
            evaluation = max(row['evals'], key=lambda e: (e['depth'], e['knodes']))
            if evaluation['depth'] < 20 or evaluation['knodes'] < 100:
                continue
            # Four-field source FEN has no clocks/history. Reset only for legal-move generation.
            state = game.new_initial_state(row['fen'] + ' 0 1')
            fen = state.board().to_fen()
            key = chess_position_key(fen)
            side = fen.split()[1]
            if key in seen or evaluation_sides[side] >= (evaluation_count + 1) // 2:
                continue
            options = chess_options(state)
            if not 2 <= len(options) <= 128:
                continue
            pv = evaluation['pvs'][0]
            action = parse_uci(state, pv['line'].split()[0], True)
            target = state.action_to_string(action)
            good = [target]
            for other in evaluation['pvs'][1:]:
                if ('cp' in pv and other.get('cp') == pv['cp']) or ('mate' in pv and other.get('mate') == pv['mate']):
                    good.append(state.action_to_string(parse_uci(state, other['line'].split()[0], True)))
            result.append(record('chess', key,
                f'Chess position (FEN, first four fields): {key}\n'
                'Move history and halfmove clock are unavailable; assess the position without repetition or fifty-move claims.\n'
                f'{"White" if side == "w" else "Black"} to move.', options, good,
                {'source': 'Lichess evaluations', 'source_ref': f'eval.jsonl:{i}',
                 'upstream': 'https://database.lichess.org/#evals', 'teacher': 'published Stockfish evaluation',
                 'depth': evaluation['depth'], 'knodes': evaluation['knodes'],
                 'best_pv': pv, 'fen': key, 'side_to_move': side,
                 'label_kind': 'bounded_engine_search_not_proven_optimal',
                 'acceptable_targets_complete': False, 'piece_count': sum(c.isalpha() for c in key.split()[0])}))
            seen.add(key); count += 1; evaluation_sides[side] += 1
    stats['chess_evaluations'] = count
    if len(result) != puzzle_count + evaluation_count:
        raise ValueError(f'Insufficient chess source data: {stats}')
    return result


DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))
WINDOWS = np.array([[15 * (r + dr*k) + c + dc*k for k in range(5)]
    for r in range(15) for c in range(15) for dr, dc in DIRS
    if 0 <= r + 4*dr < 15 and 0 <= c + 4*dc < 15])


def gomoku_wins(board, player):
    cells = board[WINDOWS]
    mask = ((cells == player).sum(axis=1) == 4) & ((cells == 0).sum(axis=1) == 1)
    return set(int(x) for x in WINDOWS[mask][cells[mask] == 0])


def gomoku_terminal(board):
    cells = board[WINDOWS]
    return bool(np.any(np.all(cells == 1, axis=1)) or np.any(np.all(cells == 2, axis=1)))


def gomoku_key(board, turn):
    square = board.reshape(15, 15)
    forms = [np.rot90(square, k) for k in range(4)]
    return f'{turn}:' + min(x.tobytes().hex() for b in forms for x in (b, np.fliplr(b)))


def coord(cell):
    r, c = divmod(int(cell), 15)
    return chr(65+c) + str(15-r)


def gomoku_candidates(board, turn):
    """Port of the playground's radius-two, contiguous-line 16-option heuristic."""
    def line_value(r, c, player, dr, dc):
        run, opened = 1, 0
        for sign in (-1, 1):
            rr, cc = r+dr*sign, c+dc*sign
            while 0 <= rr < 15 and 0 <= cc < 15 and board[rr*15+cc] == player:
                run += 1; rr += dr*sign; cc += dc*sign
            opened += int(0 <= rr < 15 and 0 <= cc < 15 and board[rr*15+cc] == 0)
        if run >= 5:
            return 1000000
        return 0 if not opened else [0, 1, 10, 100, 10000][run] * (10 if opened == 2 else 1)
    near = set()
    for cell in np.flatnonzero(board):
        r, c = divmod(int(cell), 15)
        for rr in range(max(0, r-2), min(15, r+3)):
            for cc in range(max(0, c-2), min(15, c+3)):
                if not board[rr*15+cc]:
                    near.add(rr*15+cc)
    def score(cell):
        r, c = divmod(cell, 15)
        return sum(line_value(r, c, turn, *d)*1.1 + line_value(r, c, 3-turn, *d) for d in DIRS)
    return sorted(near, key=lambda cell: (-score(cell), cell))[:16]


def build_gomoku(count, stats):
    rng = random.Random(SEED + 1)
    result, seen = [], set()
    rejected = collections.Counter()
    recall = collections.Counter()
    kinds = ['win_in_one', 'block_only_immediate_win', 'win_in_two']
    attempts = 0
    while len(result) < count:
        attempts += 1
        if attempts > max(10000, count*100):
            raise ValueError('Gomoku generation exhausted')
        kind = kinds[len(result) % 3]
        turn = 1 + (len(result) // 3) % 2
        board = np.zeros(225, dtype=np.int8)
        window = list(map(int, WINDOWS[rng.randrange(len(WINDOWS))]))
        owner = 3-turn if kind == 'block_only_immediate_win' else turn
        placed = window[1:4]
        if kind != 'win_in_two':
            gap = rng.randrange(5)
            placed = window[:gap] + window[gap+1:]
        board[placed] = owner
        n = rng.randrange(8, 25)
        needed = {1: n, 2: n if turn == 1 else n-1}
        empty = list(map(int, np.flatnonzero(board == 0))); rng.shuffle(empty)
        for player in (1, 2):
            for _ in range(needed[player] - int(np.sum(board == player))):
                board[empty.pop()] = player
        if gomoku_terminal(board):
            rejected['terminal'] += 1; continue
        key = gomoku_key(board, turn)
        if key in seen:
            rejected['duplicate_symmetry'] += 1; continue
        mine, theirs = gomoku_wins(board, turn), gomoku_wins(board, 3-turn)
        if kind == 'win_in_one':
            good = mine
        elif kind == 'block_only_immediate_win':
            good = theirs if not mine and len(theirs) == 1 else set()
        else:
            good = set()
            if not mine and not theirs:
                # A double threat wins next turn against every possible reply.
                for cell in np.flatnonzero(board == 0):
                    board[cell] = turn
                    if len(gomoku_wins(board, turn)) >= 2:
                        good.add(int(cell))
                    board[cell] = 0
        if not good:
            rejected['motif_not_proven'] += 1; continue
        candidates = gomoku_candidates(board, turn)
        recall['positions'] += 1
        recall['positions_with_proven_move_in_top16'] += bool(good.intersection(candidates))
        usable = good.intersection(candidates)
        if not usable:
            rejected['shortlist_omitted_all_verified_moves'] += 1; continue
        rows = ['   ' + ' '.join('ABCDEFGHIJKLMNO')]
        rows += [f'{15-r:2} ' + ' '.join('.XO'[int(x)] for x in board[r*15:(r+1)*15]) for r in range(15)]
        stones = '\n'.join(f'{symbol}: ' + ', '.join(coord(c) for c in np.flatnonzero(board == p)) for p, symbol in [(1, 'X'), (2, 'O')])
        state = 'Freestyle Gomoku, 15x15. Five or more in a row wins. X plays first; no forbidden moves.\n' + '\n'.join(rows)
        state += f'\n{stones}\n{"X" if turn == 1 else "O"} to move.'
        result.append(record('gomoku', key, state, {coord(c): f'Place at {coord(c)}.' for c in candidates},
            [coord(c) for c in usable], {'source': 'OpenJev procedural tactical generator',
             'source_ref': f'seed:{SEED+1}/attempt:{attempts}', 'teacher': 'exhaustive immediate-threat verification',
             'label_kind': kind, 'board': board.tolist(), 'side_to_move': turn,
             'all_verified_moves': sorted(coord(c) for c in good),
             'candidate_policy': 'playground radius2 contiguous heuristic top16, generalized to either side',
             'acceptable_targets_complete': kind != 'win_in_two'}))
        seen.add(key)
        if len(result) % 1000 == 0:
            print(f'Gomoku: {len(result)}', flush=True)
    stats['gomoku'] = {'attempts': attempts, 'rejected': dict(rejected), 'shortlist_recall': dict(recall)}
    return result


def c4_win(bits):
    for shift in (1, 7, 6, 8):
        pairs = bits & (bits >> shift)
        if pairs & (pairs >> (2*shift)):
            return True
    return False


def c4_legal(a, b):
    occupied = a | b
    return [c for c in range(7) if not (occupied >> (c*7+5)) & 1]


def c4_bit(a, b, col):
    return 1 << (col*7 + (((a | b) >> (col*7)) & 63).bit_count())


def c4_values(a, b):
    @functools.lru_cache(maxsize=None)
    def solve(current, other):
        legal = c4_legal(current, other)
        if not legal:
            return 0
        best = -1
        for col in legal:
            played = current | c4_bit(current, other, col)
            val = 1 if c4_win(played) else -solve(other, played)
            best = max(best, val)
            if best == 1:
                break
        return best
    result = {}
    for col in c4_legal(a, b):
        played = a | c4_bit(a, b, col)
        result[col] = 1 if c4_win(played) else -solve(b, played)
    return result, solve.cache_info().currsize


def c4_mirror(bits):
    return sum(((bits >> (7*c)) & 127) << (7*(6-c)) for c in range(7))


def build_connect_four(count, stats):
    rng = random.Random(SEED + 2)
    result, seen = [], set()
    attempts, nodes = 0, 0
    while len(result) < count:
        attempts += 1
        a = b = 0
        moves = []
        target_ply = rng.randrange(32, 39)
        for _ in range(target_ply):
            legal = [c for c in c4_legal(a, b) if not c4_win(a | c4_bit(a, b, c))]
            if not legal:
                break
            col = rng.choice(legal); moves.append(col+1)
            a, b = b, a | c4_bit(a, b, col)
        if len(moves) != target_ply or len(c4_legal(a, b)) < 2:
            continue
        key = min((a, b), (c4_mirror(a), c4_mirror(b)))
        if key in seen:
            continue
        values, visited = c4_values(a, b); nodes += visited
        if len(set(values.values())) < 2:
            continue  # An all-tied move menu teaches no outcome discrimination.
        best = max(values.values()); good = [str(c+1) for c, v in values.items() if v == best]
        current_symbol = 'X' if target_ply % 2 == 0 else 'O'
        other_symbol = 'O' if current_symbol == 'X' else 'X'
        rows = [' '.join(current_symbol if a & (1 << (c*7+r)) else other_symbol if b & (1 << (c*7+r)) else '.' for c in range(7)) for r in reversed(range(6))]
        state = 'Connect Four: 7 columns, 6 rows, gravity; four in a row wins. X plays first.\n'
        state += '\n'.join(rows) + '\n1 2 3 4 5 6 7\n' + current_symbol + ' to move.'
        result.append(record('connect_four', key, state, {str(c+1): f'Drop a disc in column {c+1}.' for c in values}, good,
            {'source': 'OpenJev legal self-play endgames', 'source_ref': f'seed:{SEED+2}/attempt:{attempts}',
             'teacher': 'exact terminal negamax', 'label_kind': 'proven_game_value',
             'history_columns': moves, 'current_bits': a, 'other_bits': b, 'side_to_move': current_symbol,
             'action_values': {str(c+1): v for c, v in values.items()},
             'value_perspective': 'side_to_move; loss=-1 draw=0 win=1',
             'acceptable_targets_complete': True, 'search_states': visited}))
        seen.add(key)
        if len(result) % 1000 == 0:
            print(f'Connect Four: {len(result)}', flush=True)
    stats['connect_four'] = {'attempts': attempts, 'search_states': nodes}
    return result


def poker_states(game):
    """Only retain player information states; never export full hidden world states."""
    states = {}
    def visit(state):
        if state.is_terminal():
            return
        if not state.is_chance_node():
            states.setdefault(state.information_state_string(), state)
        for action in state.legal_actions():
            visit(state.child(action))
    visit(game.new_initial_state())
    return states


def poker_description(name, state):
    info = state.information_state_string()
    if name == 'kuhn_poker':
        private, history = info[0], info[1:]
        return (f'Two-player Kuhn poker. Deck J<Q<K, one of each; each player antes 1. One private card each. '
                'One betting round, bet size 1, no raises. Highest card wins a showdown.\n'
                f'You are player {state.current_player()} (player 0 acts first). Your card: {"JQK"[int(private)]}.\n'
                f'Public action history: {history or "none"}. p=check when no bet, otherwise fold; b=bet when no bet, otherwise call.')
    return ('Two-player limit Leduc poker. Deck J,J,Q,Q,K,K; rank IDs 0=J, 1=Q, 2=K (suit identities removed). '
            'Each player antes 1 and receives one private card. One public card is revealed between two betting rounds. '
            'A pair with the public card beats a high card; equal ranks split the pot. '
            'Bet/raise increments are 2 in round 1 and 4 in round 2; at most two raises per round. '
            'Player 0 starts each round. Money fields are remaining stacks. '
            'Public action IDs: 0=fold, 1=check/call, 2=bet/raise. Only your information state follows:\n' + info)


def build_poker(iterations, stats):
    result = []
    for name in ('kuhn_poker', 'leduc_poker'):
        game = pyspiel.load_game(name, {'suit_isomorphism': True} if name == 'leduc_poker' else {})
        solver = pyspiel.CFRPlusSolver(game)
        for _ in range(iterations):
            solver.evaluate_and_update_policy()
        policy = solver.tabular_average_policy()
        exploitability = pyspiel.exploitability(game, policy)
        if exploitability > .01:
            raise ValueError(f'{name} CFR did not converge sufficiently: {exploitability}')
        states = poker_states(game)
        for info, state in sorted(states.items()):
            actions = state.legal_actions()
            probs = dict(policy.get_state_policy(info))
            options = {str(a): state.action_to_string(a) for a in actions}
            if name == 'kuhn_poker':
                outstanding = info.endswith('b')
                options = {'0': 'Fold' if outstanding else 'Check', '1': 'Call' if outstanding else 'Bet'}
            elif options.get('1') == 'Call':
                options['1'] = 'Check or call the outstanding bet'
            dist = {str(a): probs[a] for a in actions}
            result.append(record(name, info, poker_description(name, state), options, [],
                {'source': 'OpenSpiel CFR+ full game tree', 'source_ref': f'{name}/information_state:{info}',
                 'teacher': 'CFR+ average policy', 'iterations': iterations,
                 'teacher_exploitability': exploitability, 'information_state': info,
                 'player': state.current_player(), 'game_parameters': game.get_parameters(),
                 'label_kind': 'approximate_equilibrium_mixed_strategy'}, distribution=dist))
        stats[name] = {'information_states': len(states), 'iterations': iterations,
                       'exploitability': exploitability, 'units': 'chips per player; OpenSpiel exploitability'}
        print(f'{name}: {len(states)} information states, exploitability {exploitability:.8f}', flush=True)
    return result


def validate(records):
    ids, groups, positions = set(), {}, {}
    for row in records:
        assert row['id'] not in ids
        ids.add(row['id'])
        split, group = row['split'], row['group_id']
        assert groups.setdefault(group, split) == split
        criteria = row['questions']['q1']['criteria']
        assert 2 <= len(criteria) <= 128
        assert row['targets']['q1'] in criteria
        if 'acceptable_targets' in row:
            assert set(row['acceptable_targets']['q1']) <= criteria.keys()
            assert row['targets']['q1'] in row['acceptable_targets']['q1']
        if 'target_distributions' in row:
            p = row['target_distributions']['q1']
            assert p.keys() == criteria.keys()
            assert all(0 <= x <= 1 for x in p.values()) and abs(sum(p.values())-1) < 1e-9
        if row['provenance']['task'] == 'chess':
            key = chess_position_key(row['provenance']['fen'])
            assert key not in positions
            positions[key] = split
    return {'records': len(ids), 'unique_groups': len(groups), 'duplicate_ids': 0,
            'cross_split_group_overlap': 0, 'duplicate_chess_positions': 0,
            'invalid_targets': 0, 'invalid_distributions': 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT / 'data/openjev-games/source')
    p.add_argument('--output', type=Path, default=ROOT / 'data/openjev-games/release')
    p.add_argument('--puzzles', type=int, default=10000)
    p.add_argument('--evaluations', type=int, default=15000)
    p.add_argument('--gomoku', type=int, default=15000)
    p.add_argument('--connect-four', type=int, default=5000)
    p.add_argument('--cfr-iterations', type=int, default=10000)
    args = p.parse_args()
    for source in json.loads((args.source / 'sources.lock.json').read_text()):
        if hashlib.sha256((args.source / source['file']).read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('Frozen source checksum mismatch')
    stats = {}
    records = build_chess(args.source, args.puzzles, args.evaluations, stats)
    print(f'Chess: {len(records)}', flush=True)
    records += build_gomoku(args.gomoku, stats)
    records += build_connect_four(args.connect_four, stats)
    records += build_poker(args.cfr_iterations, stats)
    report = {'version': 'openjev-games-v1', 'seed': SEED, 'validation': validate(records),
              'teachers': stats, 'versions': {k: importlib.metadata.version(k) for k in ('open_spiel', 'numpy', 'pyarrow')},
              'sources_lock_sha256': hashlib.sha256((args.source / 'sources.lock.json').read_bytes()).hexdigest(),
              'build_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'splits': {}}
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'validation', 'calibration'):
        rows = sorted((r for r in records if r['split'] == split), key=lambda r: r['id'])
        path = args.output / (split + '.jsonl')
        path.write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':'))+'\n' for r in rows))
        table = pa.Table.from_pylist([{k: json.dumps(v, separators=(',', ':')) if isinstance(v, (dict, list)) else v
                                     for k, v in {**r, 'acceptable_targets': r.get('acceptable_targets'),
                                                  'target_distributions': r.get('target_distributions'),
                                                  'distribution_semantics': r.get('distribution_semantics'),
                                                  'loss_recommendation': r.get('loss_recommendation')}.items()} for r in rows])
        pq.write_table(table, args.output / (split + '.parquet'), compression='zstd')
        report['splits'][split] = {'records': len(rows), 'games': dict(collections.Counter(r['provenance']['task'] for r in rows)),
                                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
