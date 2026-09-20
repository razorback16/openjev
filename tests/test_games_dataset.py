"""Independent game-rule checks for the generated training labels."""
import importlib.util
from pathlib import Path

import pytest

pyspiel = pytest.importorskip('pyspiel')
np = pytest.importorskip('numpy')
pytest.importorskip('pyarrow')
spec = importlib.util.spec_from_file_location('games', Path(__file__).resolve().parents[1] / 'scripts/build_games_dataset.py')
games = importlib.util.module_from_spec(spec)
spec.loader.exec_module(games)


def test_chess_puzzle_starts_after_opponent_move():
    state = pyspiel.load_game('chess').new_initial_state('q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17')
    state.apply_action(games.parse_uci(state, 'e8d7'))
    assert state.board().to_fen().split()[1] == 'w'
    assert state.action_to_string(games.parse_uci(state, 'a2e6')) == 'Be6+'


def test_chess960_pv_castle_is_converted_for_standard_chess():
    s = pyspiel.load_game('chess').new_initial_state('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1')
    assert s.action_to_string(games.parse_uci(s, 'e1h1', True)) == 'O-O'
    assert s.action_to_string(games.parse_uci(s, 'e1a1', True)) == 'O-O-O'


def test_gomoku_freestyle_overline_and_coordinates():
    b = np.zeros(225, dtype=np.int8)
    b[[0, 1, 2, 4, 5]] = 1
    assert 3 in games.gomoku_wins(b, 1)
    b[3] = 1
    assert games.gomoku_terminal(b)
    assert games.coord(0) == 'A15' and games.coord(224) == 'O1'


def test_gomoku_symmetries_share_group():
    b = np.zeros((15, 15), dtype=np.int8)
    b[1, 4] = 1; b[5, 3] = 2
    for k in range(4):
        assert games.gomoku_key(b.flatten(), 1) == games.gomoku_key(np.fliplr(np.rot90(b, k)).flatten(), 1)


def test_gomoku_tactical_proofs_against_openspiel():
    for row in games.build_gomoku(6, {}):
        meta = row['provenance']
        board = np.array(meta['board'])
        queues = [list(map(int, np.flatnonzero(board == p))) for p in (1, 2)]
        state = pyspiel.load_game('gomoku').new_initial_state()
        for i in range(int(np.count_nonzero(board))):
            assert not state.is_terminal()
            state.apply_action(queues[i % 2].pop())
        assert not state.is_terminal()
        player = state.current_player()
        target = row['targets']['q1']
        action = (15-int(target[1:]))*15 + ord(target[0])-65
        state.apply_action(action)
        if meta['label_kind'] == 'win_in_one':
            assert state.is_terminal() and state.player_return(player) == 1
        elif meta['label_kind'] == 'win_in_two':
            for reply in state.legal_actions():
                child = state.child(reply)
                assert not child.is_terminal()
                assert any(child.child(a).is_terminal() and child.child(a).player_return(player) == 1
                           for a in child.legal_actions())
        else:
            assert all(not state.child(a).is_terminal() for a in state.legal_actions())


def independent_c4_value(state, player):
    if state.is_terminal():
        return int(state.player_return(player))
    values = [independent_c4_value(state.child(a), player) for a in state.legal_actions()]
    return (max if state.current_player() == player else min)(values)


def test_connect_four_solver_matches_independent_openspiel_tree():
    records = games.build_connect_four(30, {})
    for row in records:
        meta = row['provenance']
        state = pyspiel.load_game('connect_four').new_initial_state()
        for col in meta['history_columns']:
            assert not state.is_terminal()
            state.apply_action(col-1)
        assert set(state.legal_actions()) == {int(k)-1 for k in meta['action_values']}
        for action in state.legal_actions():
            assert independent_c4_value(state.child(action), state.current_player()) == meta['action_values'][str(action+1)]


@pytest.mark.parametrize('name,params', [('kuhn_poker', {}), ('leduc_poker', {'suit_isomorphism': True})])
def test_poker_description_cannot_reveal_opponent_card(name, params):
    game = pyspiel.load_game(name, params)
    a, b = game.new_initial_state(), game.new_initial_state()
    a.apply_action(0); b.apply_action(0)
    a.apply_action(1); b.apply_action(2)
    assert str(a) != str(b)
    assert a.information_state_string() == b.information_state_string()
    assert games.poker_description(name, a) == games.poker_description(name, b)


def test_poker_convergence_and_probabilities():
    stats = {}
    records = games.build_poker(1000, stats)
    assert len(records) == 300  # 12 Kuhn + 288 rank-only Leduc, no repeated hidden worlds.
    games.validate(records)
    assert stats['leduc_poker']['exploitability'] < .01
    assert any(.1 < p < .9 for r in records for p in r['target_distributions']['q1'].values())
    assert all(r['distribution_semantics'] == 'behavioral_strategy_not_confidence' for r in records)
