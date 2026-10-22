"""Entrena TennisEdge desde el histórico y guarda los artefactos en data/.

Uso:
    python scripts/build_model.py                 # descarga el histórico desde GitHub
    python scripts/build_model.py --local DIR     # usa una copia local del archivo
Tarda ~2-4 minutos en un PC normal. En Render NO se ejecuta: la app carga lo ya entrenado.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tennisedge import markov  # noqa: E402
from tennisedge.data_archive import ATTRIBUTION, load_archive  # noqa: E402
from tennisedge.model import (FEATURES, MARKET_FAMILIES, LogitModel, TreeModel,  # noqa: E402
                              calibration_table, feature_frame, fit_platt, market_report,
                              market_samples, metrics)
from tennisedge.ratings import Engine, ordinal  # noqa: E402
from tennisedge.schema import ROUND_ORDER, parse_score  # noqa: E402

DATA = ROOT / 'data'
TRAIN_END = pd.Timestamp('2024-07-01')
VAL_END = pd.Timestamp('2025-07-01')


def group_of(tour: str, level: str) -> str:
    if level == 'ITF':
        return f'{tour} ITF'
    if level == 'C':
        return 'Challenger' if tour == 'ATP' else 'WTA 125'
    return f'{tour} principal'


def run_engine(df: pd.DataFrame):
    engine = Engine()
    rows = []
    stat_cols = ['w_svpt', 'w_1stWon', 'w_2ndWon', 'l_svpt', 'l_1stWon', 'l_2ndWon']
    recs = df.to_dict('records')
    t0 = time.time()
    for i, r in enumerate(recs):
        if i % 50000 == 0:
            print(f'  {i}/{len(recs)}  {time.time()-t0:.0f}s', flush=True)
        if r['retired']:
            continue
        sc = parse_score(r['score'])
        if sc is None:
            continue
        tour, surf, lvl = r['tour'], r['surface'], r['level']
        t = ordinal(r['date'])
        w = engine.player(tour, r['winner'], lvl)
        l = engine.player(tour, r['loser'], lvl)
        a_is_w = w.name.casefold() <= l.name.casefold()
        A, B = (w, l) if a_is_w else (l, w)
        f = engine.features(tour, A, B, surf, t, int(r['best_of']))
        set1 = sc['sets'][0]
        f.update({'date': r['date'], 'tour': tour, 'surface': surf, 'level': lvl, 'best_of': int(r['best_of']),
                  'group': group_of(tour, lvl), 'player_a': A.name, 'player_b': B.name,
                  'a_win': int(a_is_w), 'n_sets': sc['w_sets'] + sc['l_sets'], 'l_sets': sc['l_sets'],
                  'total_games': sc['w_games'] + sc['l_games'],
                  'margin_a': (sc['w_games'] - sc['l_games']) * (1 if a_is_w else -1),
                  'set1_a': int((set1[0] > set1[1]) == a_is_w), 'tb_any': sc['tiebreaks']})
        rows.append(f)
        stats = None
        if all(pd.notna(r[c]) for c in stat_cols):
            stats = {c: float(r[c]) for c in stat_cols}
        p_w = f['p_elo'] if a_is_w else 1 - f['p_elo']
        engine.update(tour, r['winner'], r['loser'], surf, lvl, t, stats, r['winner_rank'], r['loser_rank'], p_w,
                      games=(sc['w_games'], sc['l_games']))
    print(f'  motor listo: {len(rows)} partidos en {time.time()-t0:.0f}s', flush=True)
    return engine, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--local', default=None)
    ap.add_argument('--start', type=int, default=2016)
    ap.add_argument('--end', type=int, default=2026)
    args = ap.parse_args()

    print('Cargando histórico…', flush=True)
    df = load_archive(range(args.start, args.end + 1), args.local, log=lambda m: None)
    df['ro'] = df['round'].map(ROUND_ORDER).fillna(5)
    df = df.sort_values(['date', 'tourney_id', 'ro']).reset_index(drop=True)
    df = df.drop_duplicates(subset=['tour', 'tourney_id', 'winner', 'loser', 'round'])
    print(f'  {len(df)} partidos, {df.date.min().date()} → {df.date.max().date()}', flush=True)

    engine, ex = run_engine(df)

    print('Modelo de saque (Markov) vectorizado…', flush=True)
    p_mk = np.empty(len(ex))
    for bo in (3, 5):
        m = ex.best_of.to_numpy() == bo
        if m.any():
            p_mk[m] = markov.match_win(ex.pa.to_numpy()[m], ex.pb.to_numpy()[m], bo)
    ex['p_markov'] = p_mk
    X = feature_frame(ex)
    y = ex['a_win'].to_numpy()

    # Solo se evalúa/entrena con jugadores que ya tienen algo de historial
    seasoned = (ex.n_a >= 5) & (ex.n_b >= 5)
    start_fit = ex.date >= pd.Timestamp('2018-01-01')     # 2016-17 = calentamiento de ratings
    tr = seasoned & start_fit & (ex.date < TRAIN_END)
    va = seasoned & (ex.date >= TRAIN_END) & (ex.date < VAL_END)
    te = seasoned & (ex.date >= VAL_END)
    print(f'  train {tr.sum()}  val {va.sum()}  test {te.sum()}', flush=True)

    lr = LogitModel.fit(X[tr], y[tr], C=0.3)
    best = None
    for hp in ({'max_iter': 200, 'learning_rate': 0.05}, {'max_iter': 300, 'learning_rate': 0.05},
               {'max_iter': 400, 'learning_rate': 0.03, 'max_leaf_nodes': 20}):
        gb = TreeModel.fit(X[tr], y[tr], **hp)
        ll = metrics(y[va], gb.predict_sym(X[va].to_numpy()))['log_loss']
        print(f'  GB {hp}: log loss val {ll:.4f}', flush=True)
        if best is None or ll < best[0]:
            best = (ll, hp, gb)
    _, HP, mdl = best
    predict = lambda frame: mdl.predict_sym(np.asarray(frame, float))  # noqa: E731

    # Comparativa en test (periodo nunca visto)
    p_te = predict(X[te])
    comp = [{'modelo': 'Elo por superficie (con margen de victoria)', **metrics(y[te], ex.p_elo[te])},
            {'modelo': 'Saque/resto (modelo de puntos)', **metrics(y[te], ex.p_markov[te])},
            {'modelo': 'Regresión logística (referencia)', **metrics(y[te], lr.predict(X[te]))},
            {'modelo': 'TennisEdge final (Gradient Boosting)', **metrics(y[te], p_te)}]
    for c in comp:
        print('  ', c)
    groups = []
    for g, part in ex[te].groupby('group'):
        idx = part.index
        groups.append({'grupo': g, 'elo_accuracy': metrics(y[idx], ex.p_elo[idx])['accuracy'], **metrics(y[idx], predict(X.loc[idx]))})
    surf = []
    for s, part in ex[te].groupby('surface'):
        idx = part.index
        surf.append({'superficie': s, **metrics(y[idx], predict(X.loc[idx]))})

    # Mercados: muestras de validación (para calibrar) y de test (para juzgar), sin mezclar.
    print('Validando mercados…', flush=True)
    t0 = time.time()
    tier = ex.group.isin(['ATP principal', 'WTA principal', 'Challenger', 'WTA 125'])
    vrows = ex[va & tier].tail(4000)
    trows = ex[te & tier].tail(4000)
    s_val = market_samples(vrows, predict(X.loc[vrows.index]))
    s_te = market_samples(trows, predict(X.loc[trows.index]))
    calib = {k: fit_platt(*s_val[k]) for k in MARKET_FAMILIES if len(s_val[k][1])}
    base_rates = {k: float(s_val[k][1].mean()) for k in MARKET_FAMILIES if len(s_val[k][1])}
    for k, ln in s_val['_lines'].items():
        base_rates[k] = {str(x): float(s_val[k][1][ln == x].mean()) for x in np.unique(ln)}
    mv = market_report(s_te, calib, base_rates)
    print(f'  mercados en {time.time()-t0:.0f}s')
    for r in mv:
        print(f"   {r['mercado']}: brier {r['brier']:.4f} (sin calibrar {r['brier_sin_calibrar']:.4f}, "
              f"referencia {r['brier_referencia']:.4f}) acierto {r['acierto']:.3f} vs {r['acierto_referencia']:.3f} "
              f"-> {'AVALADO' if r['avalado'] else 'no avalado'}")

    # Modelo final con TODOS los datos y los mismos hiperparámetros.
    allm = seasoned & start_fit
    final = TreeModel.fit(X[allm], y[allm], **HP)

    DATA.mkdir(exist_ok=True)
    keep_since = ordinal(pd.Timestamp('2023-01-01'))
    state = engine.to_dict(keep_since=keep_since, include_h2h=False)
    with gzip.open(DATA / 'engine_state.json.gz', 'wt', encoding='utf-8') as fh:
        json.dump(state, fh, separators=(',', ':'))
    report = {
        'version': 'tennisedge-v5', 'built_utc': pd.Timestamp.utcnow().isoformat(timespec='seconds'),
        'attribution': ATTRIBUTION, 'archive_last_date': str(df.date.max().date()),
        'n_matches': int(len(ex)), 'split': {'train_end': str(TRAIN_END.date()), 'val_end': str(VAL_END.date()),
                                              'n_train': int(tr.sum()), 'n_val': int(va.sum()), 'n_test': int(te.sum())},
        'hyperparams': HP, 'test_models': comp, 'test_groups': groups, 'test_surfaces': surf,
        'calibration': calibration_table(y[te], p_te), 'markets': mv, 'base_rates': base_rates,
        'market_calibration': calib, 'games_bias': s_val['_games_bias'],
        'lr_coef': dict(zip(FEATURES, lr.coef.tolist())),
    }
    (DATA / 'model.json').write_text(json.dumps({'model': final.to_dict(), 'report': report},
                                                ensure_ascii=False, indent=1), encoding='utf-8')
    # Tabla compacta para historial/H2H en la app (desde 2024, con estadísticas)
    hist = df[df.date >= pd.Timestamp('2024-01-01')].drop(columns=['ro', 'odds_w', 'odds_l', 'source', 'tourney_id'])
    hist.to_csv(DATA / 'history.csv.gz', index=False, compression='gzip')
    print('Artefactos guardados en data/', flush=True)
    for p in sorted(DATA.iterdir()):
        print(f'  {p.name}: {p.stat().st_size/1e6:.1f} MB')


if __name__ == '__main__':
    main()
