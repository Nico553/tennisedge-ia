"""Modelo jerárquico de tenis: punto al saque -> juego -> tiebreak -> set -> partido.

Todas las probabilidades de mercado (sets, juegos, hándicaps, primer set,
tiebreak) salen de solo dos números: la probabilidad de que cada jugador gane
un punto con su propio saque (pa, pb). Así los mercados son coherentes entre sí.

Las funciones de probabilidad de partido aceptan escalares o arrays de numpy
(vectorizadas) para poder evaluar miles de partidos en el entrenamiento.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import numpy as np

P_MIN, P_MAX = 0.20, 0.95


def _arr(x):
    return np.clip(np.asarray(x, dtype=float), P_MIN, P_MAX)


def game_win(p):
    """Probabilidad de que el sacador gane el juego (ventaja clásica)."""
    p = _arr(p)
    q = 1 - p
    deuce = p * p / (1 - 2 * p * q)
    return p**4 * (1 + 4 * q + 10 * q**2) + 20 * p**3 * q**3 * deuce


def tiebreak_win(pa, pb, a_first: bool, target: int = 7):
    """Probabilidad de que A gane un tiebreak a `target` puntos (diferencia de 2)."""
    pa, pb = _arr(pa), _arr(pb)
    first_is_a = a_first

    def a_serves(k: int) -> bool:
        same = ((k + 1) // 2) % 2 == 0
        return first_is_a if same else not first_is_a

    probs = {(0, 0): np.ones_like(pa)}
    win = np.zeros_like(pa)
    t = target - 1
    for n in range(0, 2 * t + 1):
        for i in range(0, min(n, t) + 1):
            j = n - i
            if j > t or (i, j) not in probs:
                continue
            pr = probs.pop((i, j))
            if i == t and j == t:
                # Desde t-t cada par de puntos tiene un saque de cada uno.
                a2 = pa * (1 - pb)
                b2 = (1 - pa) * pb
                win = win + pr * a2 / (a2 + b2)
                continue
            pw = pa if a_serves(n) else 1 - pb
            for (ni, nj, w) in ((i + 1, j, pw), (i, j + 1, 1 - pw)):
                if ni == target and nj <= t - 1:
                    win = win + pr * w
                elif nj == target and ni <= t - 1:
                    pass
                else:
                    probs[(ni, nj)] = probs.get((ni, nj), 0) + pr * w
    return win


def set_distribution(pa, pb, a_first: bool, tb_target: int = 7):
    """Distribución de marcadores finales de un set: {(juegos_A, juegos_B): prob}."""
    pa, pb = _arr(pa), _arr(pb)
    ga, gb = game_win(pa), game_win(pb)
    probs = {(0, 0): np.ones_like(pa)}
    final: dict[tuple[int, int], np.ndarray] = {}
    for n in range(0, 12):
        for i in range(0, n + 1):
            j = n - i
            if (i, j) not in probs:
                continue
            pr = probs.pop((i, j))
            a_serving = (n % 2 == 0) == a_first
            pw = ga if a_serving else 1 - gb
            for (ni, nj, w) in ((i + 1, j, pw), (i, j + 1, 1 - pw)):
                done = (ni == 6 and nj <= 4) or (nj == 6 and ni <= 4) or (ni, nj) in ((7, 5), (5, 7))
                bucket = final if done else probs
                bucket[(ni, nj)] = bucket.get((ni, nj), 0) + pr * w
    tie = probs.pop((6, 6))
    tb = tiebreak_win(pa, pb, a_first, tb_target)
    final[(7, 6)] = tie * tb
    final[(6, 7)] = tie * (1 - tb)
    return final


def match_win(pa, pb, best_of: int = 3, final_tb: int = 7):
    """Probabilidad (vectorizada) de que A gane el partido; promedia quién saca primero."""
    pa, pb = _arr(pa), _arr(pb)
    need = best_of // 2 + 1
    # Para cada primer sacador: P(A gana set y el siguiente set lo abre A/B).
    trans = {}
    for last in (False, True):
        tb = final_tb if last else 7
        for a_first in (True, False):
            dist = set_distribution(pa, pb, a_first, tb)
            out = {('A', True): 0, ('A', False): 0, ('B', True): 0, ('B', False): 0}
            for (i, j), pr in dist.items():
                g = i + j
                nxt_a = a_first if g % 2 == 0 else not a_first
                out[('A' if i > j else 'B', nxt_a)] = out[('A' if i > j else 'B', nxt_a)] + pr
            trans[(last, a_first)] = out
    total = np.zeros_like(pa)
    for start_a in (True, False):
        states = {(0, 0, start_a): np.ones_like(pa)}
        win = np.zeros_like(pa)
        while states:
            new = {}
            for (sa, sb, a_first), pr in states.items():
                last = (sa == need - 1 and sb == need - 1)
                for (who, nxt), w in trans[(last, a_first)].items():
                    na, nb = (sa + 1, sb) if who == 'A' else (sa, sb + 1)
                    if na == need:
                        win = win + pr * w
                    elif nb == need:
                        continue
                    else:
                        key = (na, nb, nxt)
                        new[key] = new.get(key, 0) + pr * w
            states = new
        total = total + 0.5 * win
    return total


def solve_shift(pa, pb, target, best_of: int = 3, iters: int = 40):
    """Busca d tal que match_win(pa+d, pb-d) == target (vectorizado, bisección).

    Mantiene la suma de las dos probabilidades de saque (el "estilo" del partido,
    clave para los totales de juegos) y mueve solo la ventaja relativa.
    """
    pa, pb, target = (np.asarray(x, dtype=float) for x in (pa, pb, target))
    lo = np.full_like(pa, -0.35)
    hi = np.full_like(pa, 0.35)
    for _ in range(iters):
        mid = (lo + hi) / 2
        val = match_win(pa + mid, pb - mid, best_of)
        up = val < target
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    d = (lo + hi) / 2
    return np.clip(pa + d, P_MIN, P_MAX), np.clip(pb - d, P_MIN, P_MAX)


@lru_cache(maxsize=4096)
def _markets_cached(pa: float, pb: float, best_of: int, final_tb: int) -> dict:
    need = best_of // 2 + 1
    set_d = {}
    for last in (False, True):
        for a_first in (True, False):
            dist = set_distribution(np.array([pa]), np.array([pb]), a_first, final_tb if last else 7)
            set_d[(last, a_first)] = {k: float(v[0]) for k, v in dist.items()}

    sets_score = defaultdict(float)
    total_games = defaultdict(float)
    games_a = defaultdict(float)
    games_b = defaultdict(float)
    margin = defaultdict(float)
    tb_any = 0.0
    first_set = defaultdict(float)
    for start_a in (True, False):
        for k, v in set_d[(need == 1, start_a)].items():
            first_set[k] += 0.5 * v
        states = {(0, 0, start_a, 0, 0, False): 0.5}
        while states:
            new = defaultdict(float)
            for (sa, sb, a_first, gA, gB, had_tb), pr in states.items():
                last = sa == need - 1 and sb == need - 1
                for (i, j), w in set_d[(last, a_first)].items():
                    if w <= 0:
                        continue
                    na, nb = (sa + 1, sb) if i > j else (sa, sb + 1)
                    ga2, gb2 = gA + i, gB + j
                    tb2 = had_tb or (i + j == 13)
                    nxt = a_first if (i + j) % 2 == 0 else not a_first
                    p = pr * w
                    if na == need or nb == need:
                        sets_score[(na, nb)] += p
                        total_games[ga2 + gb2] += p
                        games_a[ga2] += p
                        games_b[gb2] += p
                        margin[ga2 - gb2] += p
                        if tb2:
                            tb_any += p
                    else:
                        new[(na, nb, nxt, ga2, gb2, tb2)] += p
            states = new
    return {'sets_score': dict(sets_score), 'total_games': dict(total_games),
            'games_a': dict(games_a), 'games_b': dict(games_b), 'margin': dict(margin),
            'tiebreak': tb_any, 'first_set': dict(first_set)}


def _over(dist: dict, line: float) -> float:
    return float(sum(p for k, p in dist.items() if k > line))


def markets(pa: float, pb: float, best_of: int = 3, final_tb: int = 7) -> dict:
    """Todos los mercados derivados de (pa, pb), listos para mostrar."""
    raw = _markets_cached(round(float(pa), 4), round(float(pb), 4), int(best_of), int(final_tb))
    need = best_of // 2 + 1
    ss = raw['sets_score']
    p_a = sum(p for (a, b), p in ss.items() if a == need)
    tg = raw['total_games']
    exp_games = sum(k * p for k, p in tg.items())
    fs = raw['first_set']
    fs_total = defaultdict(float)
    fs_margin = defaultdict(float)
    for (i, j), p in fs.items():
        fs_total[i + j] += p
        fs_margin[i - j] += p
    if best_of == 3:
        lines = [18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5]
    else:
        base = round(exp_games - 0.5)
        lines = sorted({base - 3 + 0.5 + k for k in range(0, 7)})
    return {
        'p_a': p_a,
        'set_scores': {f'{a}-{b}': p for (a, b), p in sorted(ss.items(), key=lambda kv: -kv[1])},
        'straight_a': ss.get((need, 0), 0.0),
        'straight_b': ss.get((0, need), 0.0),
        'a_wins_set': 1 - ss.get((0, need), 0.0),
        'b_wins_set': 1 - ss.get((need, 0), 0.0),
        'n_sets': {n: sum(p for (a, b), p in ss.items() if a + b == n) for n in range(need, best_of + 1)},
        'expected_games': exp_games,
        'total_over': {ln: _over(tg, ln) for ln in lines},
        'a_handicap': {h: _over(raw['margin'], h) for h in (1.5, 2.5, 3.5, 4.5, 5.5)},
        'b_handicap': {h: float(sum(p for k, p in raw['margin'].items() if -k > h)) for h in (1.5, 2.5, 3.5, 4.5, 5.5)},
        'a_games_over': {ln: _over(raw['games_a'], ln) for ln in (8.5, 9.5, 10.5, 11.5, 12.5)},
        'b_games_over': {ln: _over(raw['games_b'], ln) for ln in (8.5, 9.5, 10.5, 11.5, 12.5)},
        'tiebreak': raw['tiebreak'],
        'set1_a': sum(p for (i, j), p in fs.items() if i > j),
        'set1_scores': {f'{i}-{j}': p for (i, j), p in sorted(fs.items(), key=lambda kv: -kv[1])},
        'set1_total_over': {ln: _over(fs_total, ln) for ln in (8.5, 9.5, 10.5, 12.5)},
        'set1_a_handicap': {h: _over(fs_margin, h) for h in (1.5, 2.5, 3.5)},
        'set1_b_handicap': {h: float(sum(p for k, p in fs_margin.items() if -k > h)) for h in (1.5, 2.5, 3.5)},
    }
