"""Formato común de partido usado por todas las fuentes."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

STAT_COLS = ['svpt', '1stIn', '1stWon', '2ndWon', 'SvGms', 'ace', 'df', 'bpSaved', 'bpFaced']
COLUMNS = (['date', 'tour', 'tourney_id', 'tourney_name', 'level', 'surface', 'round', 'best_of',
            'winner', 'loser', 'winner_rank', 'loser_rank', 'score', 'retired', 'odds_w', 'odds_l', 'source']
           + [f'w_{c}' for c in STAT_COLS] + [f'l_{c}' for c in STAT_COLS])

ROUND_ORDER = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3, 'R128': 4, 'R64': 5, 'R32': 6, 'R16': 7,
               'RR': 7, 'QF': 8, 'SF': 9, 'BR': 9, 'F': 10}
ROUNDS_TO_FINAL = {'R128': 6, 'R64': 5, 'R32': 4, 'R16': 3, 'RR': 3, 'QF': 2, 'SF': 1, 'BR': 1, 'F': 0}

SURFACES = ('Hard', 'Clay', 'Grass')

# Peso del torneo en el Elo (estilo FiveThirtyEight / Tennis Abstract).
LEVEL_K = {'G': 1.10, 'M': 1.05, 'F': 1.05, 'PM': 1.05, 'A': 1.0, 'P': 1.0, 'I': 1.0, 'O': 1.0,
           'D': 0.90, 'W': 0.9, 'C': 0.90, 'ITF': 0.80}
LEVEL_NAME = {'G': 'Grand Slam', 'M': 'Masters 1000', 'PM': 'WTA 1000', 'F': 'Finals', 'A': 'ATP 250/500',
              'P': 'WTA 500', 'I': 'WTA 250', 'D': 'Copa Davis/BJK', 'O': 'Olímpicos', 'W': 'Otros',
              'C': 'Challenger / WTA 125', 'ITF': 'ITF'}

_SET = re.compile(r'^(\d+)-(\d+)(?:\(\d+\))?$')


def norm_surface(value) -> str | None:
    s = str(value or '').strip().lower()
    if 'grass' in s or 'cesped' in s or 'césped' in s:
        return 'Grass'
    if 'clay' in s or 'arcilla' in s or 'tierra' in s:
        return 'Clay'
    if 'hard' in s or 'carpet' in s or 'acrylic' in s or 'dura' in s or 'indoor' in s:
        return 'Hard'
    return None


def norm_level(value) -> str:
    v = str(value or '').strip()
    if v in LEVEL_K:
        return v
    if v.isdigit():  # niveles ITF numéricos (15, 25, 35, 50, 75, 100...)
        return 'ITF'
    return 'A'


def parse_score(score) -> dict | None:
    """Marcador desde la óptica del ganador. None si no es un partido completo."""
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return None
    s = str(score).replace(',', ' ').replace(':', '-').strip()
    if not s or re.search(r'RET|W/?O|DEF|ABD|ABN|UNFINISHED|SUSP|DEFAULT|NA', s, re.I):
        return None
    sets = []
    for tok in s.split():
        m = _SET.match(tok)
        if not m:
            # super tiebreak "[10-8]" o "1-0(10-8)": se cuenta como set 1-0 sin juegos
            if re.match(r'^\[?\d+-\d+\]?$', tok):
                a, b = (int(x) for x in tok.strip('[]').split('-'))
                if max(a, b) >= 10:
                    sets.append((1, 0) if a > b else (0, 1))
                    continue
            return None
        a, b = int(m.group(1)), int(m.group(2))
        if a == b:
            return None
        sets.append((a, b))
    if len(sets) < 2:
        return None
    ws = sum(a > b for a, b in sets)
    ls = sum(b > a for a, b in sets)
    if ws <= ls:
        return None
    return {'sets': sets, 'w_sets': ws, 'l_sets': ls,
            'w_games': sum(a for a, _ in sets), 'l_games': sum(b for _, b in sets),
            'tiebreaks': sum(1 for a, b in sets if (a, b) in ((7, 6), (6, 7)))}


def empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)
