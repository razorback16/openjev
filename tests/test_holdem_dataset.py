"""Hold'em label, blocker, information-boundary, and equilibrium regressions."""
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip('treys')
np = pytest.importorskip('numpy')
pytest.importorskip('scipy')
pytest.importorskip('pyarrow')
SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
import build_holdem_dataset as h
import audit_holdem_dataset as audit


@pytest.mark.parametrize('cards,category', [
    ('As 2s 3s 4s 5s Kd Qh', 8),
    ('As Ah Ad Ac Ks Qs Js', 7),
    ('As Ah Ad Kc Ks Kh 2d', 6),
    ('As Js 8s 5s 2s Kd Qd', 5),
    ('As 2d 3h 4c 5s Kd Qh', 4),
    ('As Ah Ad Kc Qs 9h 2d', 3),
    ('As Ah Kd Kc Qs Qh 2d', 2),
    ('As Ah Kd Qc Js 9h 2d', 1),
    ('As Kd Qh 9c 7s 4h 2d', 0),
])
def test_hand_categories_and_wheel(cards, category):
    assert h.category(cards.split()) == category == audit.score(cards.split())[0]


def test_independent_evaluator_kickers_and_double_trips():
    assert audit.score('As Ah Kd Qc Js 9h 2d'.split()) > audit.score('Ac Ad Kh Qs Td 9c 2h'.split())
    assert audit.score('As Ah Ad Kc Ks Kh 2d'.split()) == (6, 14, 13)
    assert audit.score('2s 3h 4c 5d 6s Ah Kd'.split()) > audit.score('As 2h 3c 4d 5s Qh Kd'.split())


def test_board_royal_flush_ties_every_legal_opponent():
    counts = h.equity(['2c', '3d'], ['As', 'Ks', 'Qs', 'Js', 'Ts'])
    assert counts == {'win': 0, 'tie': 990, 'loss': 0}
    assert h.equity_bin(counts) == 5


def test_turn_equity_removes_opponent_blockers():
    hero, board, ranges = ['As', 'Ah'], ['2c', '3d', '4h', '9s'], [('Ks', 'Kh'), ('Qc', 'Qd')]
    counts = h.equity(hero, board, ranges)
    assert sum(counts.values()) == 88  # 44 possible rivers per specified opponent hand.
    assert counts == audit.independent_equity(hero, board, ranges)
    with pytest.raises(AssertionError):
        h.equity(hero, board, [('As', 'Kd')])


def test_exact_decile_boundaries():
    assert h.equity_bin({'win': 1, 'tie': 0, 'loss': 9}) == 1
    assert h.equity_bin({'win': 0, 'tie': 1, 'loss': 4}) == 1
    assert h.equity_bin({'win': 10, 'tie': 0, 'loss': 0}) == 9


def test_suit_aliases_and_board_order_share_split_group():
    assert h.canonical_board(['As', 'Ks', '2d', '3c', '4h']) == h.canonical_board(['4c', '3h', '2s', 'Kd', 'Ad'])


def test_analytic_bluffing_equilibrium():
    x, y, cert = h.solve_river(np.array([[1, 1], [-1, -1]]), np.full((2, 2), .25), 100, 100)
    assert x == pytest.approx([1, .5])
    assert float(np.mean(y)) == pytest.approx(.5)
    assert cert['best_response_gap_chips'] < 1e-8


def test_river_policy_certified_and_no_actual_opponent_card():
    rows, _ = h.river_games(3)
    assert len(rows) == 24
    for first in rows[::8]:
        m = first['provenance']
        assert audit.normal_form_gap(m['board'], m['ranges'], m['pot'], m['bet'], m['policy_a_bet'], m['policy_b_call']) < 1e-8
    for row in rows:
        assert 'Opponent private cards remain unknown' in row['state']
        assert 'opponent_hand' not in row['provenance']
        assert row['state'].count('Your hole cards:') == 1
        assert row['distribution_semantics'] == 'behavioral_strategy_not_confidence'
    for same_game in (rows[i:i+8] for i in range(0, len(rows), 8)):
        assert len({r['group_id'] for r in same_game}) == 1
        assert len({r['split'] for r in same_game}) == 1
