"""Resultados recientes (desde el fin del histórico hasta hoy).

Fuentes, en orden:
  1. data/recent_results.csv  -> en el repositorio; lo actualiza GitHub Actions cada día.
  2. caché local de la instancia -> lo que se descargue desde la app (se pierde al reiniciar Render).
"""
from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from . import rapid
from .schema import COLUMNS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
REPO_FILE = DATA / 'recent_results.csv'
CACHE_DIR = Path(os.environ.get('TENNISEDGE_CACHE', '/tmp/tennisedge'))
CACHE_FILE = CACHE_DIR / 'recent_results.csv'
PROGRESS_FILE = DATA / 'recent_progress.json'


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, low_memory=False)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df[COLUMNS].dropna(subset=['date'])


def merge(*frames: pd.DataFrame) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['retired'] = df['retired'].astype(str).str.lower().isin(['true', '1'])
    df = df.drop_duplicates(subset=['tour', 'date', 'winner', 'loser'], keep='last')
    return df.sort_values(['date', 'tour']).reset_index(drop=True)


def load_recent() -> pd.DataFrame:
    return merge(_read(REPO_FILE), _read(CACHE_FILE))


def covered_until(archive_last: str) -> date:
    """Última fecha completamente descargada (para seguir desde ahí)."""
    best = pd.Timestamp(archive_last).date()
    for p in (PROGRESS_FILE, CACHE_DIR / 'recent_progress.json'):
        try:
            d = pd.Timestamp(json.loads(p.read_text())['covered_until']).date()
            best = max(best, d)
        except Exception:  # noqa: BLE001
            pass
    return best


def update(archive_last: str, budget: int = 40, to_repo: bool = False, log=None) -> tuple[pd.DataFrame, int]:
    """Descarga lo que falta (respetando el presupuesto de llamadas) y guarda."""
    start = covered_until(archive_last) - timedelta(days=2)   # solapamiento: partidos tardíos
    end = date.today() - timedelta(days=1)
    if start > end:
        return load_recent(), 0
    new, used, covered = rapid.fetch_range(start, end, budget, log=log)
    target = REPO_FILE if to_repo else CACHE_FILE
    progress = PROGRESS_FILE if to_repo else CACHE_DIR / 'recent_progress.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    merged = merge(_read(target), new)
    merged.to_csv(target, index=False)
    if covered >= start:
        progress.write_text(json.dumps({'covered_until': covered.isoformat()}))
    return load_recent(), used
