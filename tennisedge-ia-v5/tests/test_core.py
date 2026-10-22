import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennisedge import markov, rapid  # noqa: E402
from tennisedge.names import NameIndex, key  # noqa: E402
from tennisedge.schema import parse_score  # noqa: E402


def test_symmetry_and_bounds():
    assert abs(markov.match_win(0.62, 0.62) - 0.5) < 1e-9
    p = markov.match_win(np.array([0.65, 0.55]), np.array([0.6, 0.6]))
    assert p[0] > 0.5 > p[1]
    assert markov.match_win(0.65, 0.6, 5) > markov.match_win(0.65, 0.6, 3)


def test_markets_sum_to_one():
    m = markov.markets(0.64, 0.6, 3)
    assert abs(sum(m['set_scores'].values()) - 1) < 1e-9
    assert abs(m['p_a'] - float(markov.match_win(0.64, 0.6))) < 1e-6
    assert abs(sum(m['set1_scores'].values()) - 1) < 1e-9


def test_against_simulation():
    random.seed(3)

    def game(p):
        a = b = 0
        while True:
            if random.random() < p: a += 1
            else: b += 1
            if a >= 4 and a - b >= 2: return 1
            if b >= 4 and b - a >= 2: return 0

    def set_(pa, pb, af):
        a = b = n = 0
        while True:
            srv_a = (n % 2 == 0) == af
            w = game(pa) if srv_a else 1 - game(pb)
            a += w; b += 1 - w; n += 1
            if (a == 6 and b <= 4) or (b == 6 and a <= 4) or (a, b) in ((7, 5), (5, 7)): return a > b
            if a == 6 and b == 6:
                return random.random() < float(markov.tiebreak_win(pa, pb, (n % 2 == 0) == af))
    wins = sum(set_(0.66, 0.6, True) for _ in range(20000)) / 20000
    exact = sum(float(np.ravel(v)[0]) for (i, j), v in markov.set_distribution(np.array([0.66]), np.array([0.6]), True).items() if i > j)
    assert abs(wins - exact) < 0.015


def test_solve_shift_hits_target():
    a, b = markov.solve_shift(np.array([0.62]), np.array([0.6]), np.array([0.7]))
    assert abs(float(markov.match_win(a, b)[0]) - 0.7) < 1e-4


def test_parse_score():
    s = parse_score('7-6(5) 3-6 6-4')
    assert s['w_sets'] == 2 and s['l_sets'] == 1 and s['tiebreaks'] == 1 and s['w_games'] == 16
    assert parse_score('6-4 3-1 RET') is None
    assert parse_score('W/O') is None


def test_names():
    assert key('Carlos Alcaraz') == key('ALCARAZ Carlos')
    idx = NameIndex([key('Carlos Alcaraz'), key('Jannik Sinner')])
    assert idx.resolve('C. Alcaraz') == key('Carlos Alcaraz')
    assert idx.resolve('Alcaraz, Carlos') == key('Carlos Alcaraz')


def test_parse_results_orientation():
    rows = [{'id': 1, 'date': '2026-10-10T12:00:00Z', 'result': '4-6 3-6', 'result_type': 'completed',
             'player1': {'id': 10, 'name': 'Player One'}, 'player2': {'id': 20, 'name': 'Player Two'},
             'match_winner': 20, 'odd1': 1.5, 'odd2': 2.6,
             'tournament': {'name': 'Shanghai', 'courtId': 1, 'rankId': 3}}]
    df = rapid.parse_results(rows, 'atp')
    r = df.iloc[0]
    assert r.winner == 'Player Two' and r.loser == 'Player One'
    assert parse_score(r.score)['w_sets'] == 2
    assert r.odds_w == 2.6 and r.surface == 'Hard'
