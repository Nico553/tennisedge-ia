"""Descarga resultados recientes desde RapidAPI y los guarda en data/recent_results.csv.

Lo ejecuta GitHub Actions cada día (ver .github/workflows/actualizar-resultados.yml).
Uso local:  RAPIDAPI_TENNIS_KEY=... python scripts/update_results.py --budget 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tennisedge import rapid, store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=int, default=40, help='máximo de llamadas a RapidAPI (plan free: 50/día)')
    args = ap.parse_args()
    if not rapid.available():
        print('Falta RAPIDAPI_TENNIS_KEY (GitHub → Settings → Secrets and variables → Actions).')
        return 1
    archive_last = json.loads((ROOT / 'data' / 'model.json').read_text(encoding='utf-8'))['report']['archive_last_date']
    df, used = store.update(archive_last, budget=args.budget, to_repo=True, log=print)
    print(f'Llamadas usadas: {used}. Resultados recientes guardados: {len(df)}. '
          f'Cubierto hasta: {store.covered_until(archive_last)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
