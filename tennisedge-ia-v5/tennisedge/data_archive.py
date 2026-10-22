"""Histórico Jeff Sackmann / Tennis Abstract (CC BY-NC-SA 4.0), copia de Aneeshers.

Uso personal, no comercial, con atribución. Incluye estadísticas de saque por partido.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .schema import COLUMNS, STAT_COLS, ROUNDS_TO_FINAL, norm_level, norm_surface

BASE = 'https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main'
ATTRIBUTION = ('Jeff Sackmann / Tennis Abstract — CC BY-NC-SA 4.0 · copia histórica de '
               'Aneeshers/tennis-sackmann-archive (junio 2026). Uso personal no comercial.')
FILES = {
    'ATP': ['atp/atp_matches_{y}.csv', 'atp/atp_matches_qual_chall_{y}.csv'],
    'WTA': ['wta/wta_matches_{y}.csv', 'wta/wta_matches_qual_itf_{y}.csv'],
}


def _read(path_or_url: str) -> pd.DataFrame:
    if path_or_url.startswith('http'):
        r = requests.get(path_or_url, timeout=60, headers={'User-Agent': 'TennisEdge/5 personal research'})
        if r.status_code == 404:
            return pd.DataFrame()
        r.raise_for_status()
        return pd.read_csv(BytesIO(r.content), low_memory=False)
    p = Path(path_or_url)
    return pd.read_csv(p, low_memory=False) if p.exists() else pd.DataFrame()


def normalize_sackmann(df: pd.DataFrame, tour: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    d = pd.to_datetime(df['tourney_date'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
    level = df['tourney_level'].map(norm_level)
    rnd = df['round'].astype(str)
    slam = level.eq('G')
    final_day = slam.map({True: 13, False: 6})
    to_final = rnd.map(ROUNDS_TO_FINAL)
    qual = rnd.str.match(r'^Q\d')
    offset = final_day - to_final.fillna(0) * slam.map({True: 2, False: 1})
    offset = offset.where(~qual, -4 + rnd.str[1:].map(lambda x: int(x) if str(x).isdigit() else 1))
    out = pd.DataFrame({
        'date': d,
        '_offset': offset.fillna(0),
        'tour': tour,
        'tourney_id': df['tourney_id'].astype(str),
        'tourney_name': df['tourney_name'].astype(str),
        'level': level,
        'surface': df['surface'].map(norm_surface),
        'round': rnd,
        'best_of': pd.to_numeric(df['best_of'], errors='coerce').fillna(3).astype(int),
        'winner': df['winner_name'].astype(str).str.strip(),
        'loser': df['loser_name'].astype(str).str.strip(),
        'winner_rank': pd.to_numeric(df['winner_rank'], errors='coerce'),
        'loser_rank': pd.to_numeric(df['loser_rank'], errors='coerce'),
        'score': df['score'].astype(str),
        'retired': df['score'].astype(str).str.contains(r'RET|W/O|DEF|ABD|Walkover', case=False, regex=True),
        'odds_w': float('nan'), 'odds_l': float('nan'),
        'source': 'sackmann',
    })
    for c in STAT_COLS:
        out[f'w_{c}'] = pd.to_numeric(df.get(f'w_{c}'), errors='coerce')
        out[f'l_{c}'] = pd.to_numeric(df.get(f'l_{c}'), errors='coerce')
    out = out.dropna(subset=['date', 'surface'])
    out = out[(out.winner != '') & (out.loser != '') & (out.winner != out.loser)]
    return out[COLUMNS + ['_offset']]


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Unifica fechas de un mismo torneo entre archivos, estima el día de cada ronda y deduplica."""
    parts = []
    for tour, g in df.groupby('tour'):
        g = g.copy()
        g['date'] = _unify_event_dates(g['tourney_name'].astype(str), g['date'])
        parts.append(g)
    df = pd.concat(parts)
    df = df.drop_duplicates(subset=['tour', 'tourney_name', 'date', 'round', 'winner', 'loser'])
    df['date'] = df['date'] + pd.to_timedelta(df.pop('_offset'), unit='D')
    return df[COLUMNS].reset_index(drop=True)


def _unify_event_dates(names: pd.Series, d: pd.Series) -> pd.Series:
    """Un mismo torneo a veces aparece con fechas distintas en dos archivos (p. ej. WTA 125).
    Agrupa por nombre y fechas a <= 10 días y usa la fecha más tardía del grupo, para que
    una ronda posterior nunca quede antes que una anterior."""
    out = d.copy()
    frame = pd.DataFrame({'n': names.str.lower().str.replace(r'[^a-z0-9]', '', regex=True), 'd': d})
    for _, g in frame.dropna().groupby('n'):
        dates = np.sort(g['d'].unique())
        if len(dates) < 2:
            continue
        clusters, cur = [], [dates[0]]
        for x in dates[1:]:
            if (x - cur[-1]) / np.timedelta64(1, 'D') <= 10:
                cur.append(x)
            else:
                clusters.append(cur); cur = [x]
        clusters.append(cur)
        mapping = {x: c[-1] for c in clusters for x in c}
        out.loc[g.index] = g['d'].map(mapping)
    return out


def load_archive(years, local_dir: str | None = None, tours=('ATP', 'WTA'), log=None) -> pd.DataFrame:
    frames = []
    for tour in tours:
        for y in years:
            for pattern in FILES[tour]:
                rel = pattern.format(y=y)
                src = str(Path(local_dir) / rel) if local_dir else f'{BASE}/{rel}'
                raw = _read(src)
                if log:
                    log(f'{rel}: {len(raw)} filas')
                if not raw.empty:
                    frames.append(normalize_sackmann(raw, tour))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return finalize(pd.concat(frames, ignore_index=True))
