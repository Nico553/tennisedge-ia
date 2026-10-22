"""Combinación final (regresión logística) y evaluación temporal.

El modelo final es una regresión logística sobre: Elo (superficie), modelo de saque
(Markov), ranking, forma, cansancio, descanso, experiencia y cara a cara. Se guarda como
coeficientes JSON (sin pickles) para que cargue rápido en Render.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import markov

FEATURES = ['logit_elo', 'logit_markov', 'log_rank', 'rank_missing', 'form_diff', 'load7_diff',
            'rest_diff', 'layoff_diff', 'h2h_diff', 'exp_diff', 'elo_bo5', 'w_serve']

FEATURE_LABEL = {
    'logit_elo': 'Elo en la superficie', 'logit_markov': 'Saque y resto (modelo de puntos)',
    'log_rank': 'Ranking oficial', 'form_diff': 'Forma reciente', 'load7_diff': 'Cansancio (últimos 7 días)',
    'rest_diff': 'Descanso', 'layoff_diff': 'Inactividad prolongada', 'h2h_diff': 'Cara a cara',
    'exp_diff': 'Experiencia', 'elo_bo5': 'Elo × mejor de 5',
}


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def feature_frame(f: pd.DataFrame) -> pd.DataFrame:
    """f: DataFrame con las columnas de Engine.features() + 'p_markov'."""
    x = pd.DataFrame(index=f.index)
    x['logit_elo'] = _logit(f['p_elo'])
    x['logit_markov'] = _logit(f['p_markov'])
    x['log_rank'] = f['log_rank'].astype(float)
    x['rank_missing'] = f['rank_missing'].astype(float)
    x['form_diff'] = f['form_a'] - f['form_b']
    x['load7_diff'] = f['load7_a'] - f['load7_b']
    ra = np.minimum(f['rest_a'].astype(float), 120)
    rb = np.minimum(f['rest_b'].astype(float), 120)
    x['rest_diff'] = np.log1p(ra) - np.log1p(rb)
    x['layoff_diff'] = (ra > 60).astype(float) - (rb > 60).astype(float)
    x['h2h_diff'] = (f['h2h_a'] - f['h2h_b']) / (f['h2h_a'] + f['h2h_b'] + 4)
    x['exp_diff'] = np.log1p(f['n_a']) - np.log1p(f['n_b'])
    x['elo_bo5'] = x['logit_elo'] * f['bo5']
    x['w_serve'] = f['w_serve'].astype(float) * x['logit_markov']
    return x[FEATURES]


class LogitModel:
    def __init__(self, mean, scale, coef, intercept):
        self.mean = np.asarray(mean, float); self.scale = np.asarray(scale, float)
        self.coef = np.asarray(coef, float); self.intercept = float(intercept)

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        z = (np.asarray(x, float) - self.mean) / self.scale
        return 1 / (1 + np.exp(-(z @ self.coef + self.intercept)))

    def contributions(self, x_row: np.ndarray) -> dict:
        z = (np.asarray(x_row, float) - self.mean) / self.scale
        return {name: float(c) for name, c in zip(FEATURES, z * self.coef)}

    def to_dict(self) -> dict:
        return {'features': FEATURES, 'mean': self.mean.tolist(), 'scale': self.scale.tolist(),
                'coef': self.coef.tolist(), 'intercept': self.intercept}

    @classmethod
    def from_dict(cls, d: dict) -> 'LogitModel':
        return cls(d['mean'], d['scale'], d['coef'], d['intercept'])

    @classmethod
    def fit(cls, x: pd.DataFrame, y: np.ndarray, C: float = 1.0) -> 'LogitModel':
        from sklearn.linear_model import LogisticRegression
        xa = np.asarray(x, float)
        # Simetría: duplicamos cada partido desde la óptica del otro jugador.
        sym = -xa.copy()
        sym[:, FEATURES.index('rank_missing')] = xa[:, FEATURES.index('rank_missing')]
        X = np.vstack([xa, sym]); Y = np.concatenate([y, 1 - y])
        mean = np.zeros(X.shape[1]); scale = X.std(axis=0); scale[scale == 0] = 1
        lr = LogisticRegression(C=C, max_iter=2000, fit_intercept=False)
        lr.fit((X - mean) / scale, Y)
        return cls(mean, scale, lr.coef_[0], 0.0)


def metrics(y, p) -> dict:
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    if len(y) == 0:
        return {'n': 0}
    return {'n': int(len(y)), 'accuracy': float(np.mean((p >= .5) == y)),
            'brier': float(np.mean((p - y) ** 2)),
            'log_loss': float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))}


def calibration_table(y, p, bins=(0, .3, .4, .5, .6, .7, .8, .9, 1.0001)) -> list[dict]:
    y = np.asarray(y, float); p = np.asarray(p, float)
    fav = np.maximum(p, 1 - p); yf = np.where(p >= .5, y, 1 - y)
    out = []
    for lo, hi in zip((.5, .6, .7, .8, .9), (.6, .7, .8, .9, 1.0001)):
        m = (fav >= lo) & (fav < hi)
        if m.sum():
            out.append({'tramo': f'{int(lo*100)}–{int(min(hi,1)*100)}%', 'partidos': int(m.sum()),
                        'prob_media': float(fav[m].mean()), 'acierto_real': float(yf[m].mean())})
    return out


# --------------------------------------------------------------------------- mercados
MARKET_FAMILIES = {
    'straight': 'Se decide en sets corridos',
    'set1_fav': 'Favorito gana el 1er set',
    'totals': 'Total de juegos (más de X)',
    'fav_handicap': 'Hándicap de juegos del favorito',
    'dog_set': 'No favorito gana al menos un set',
    'tiebreak': 'Habrá al menos un tiebreak',
}
TOTAL_LINES = (18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5)
HCP_LINES = (1.5, 2.5, 3.5, 4.5)


def market_samples(rows: pd.DataFrame, p_final: np.ndarray) -> dict:
    """Probabilidades crudas del modelo de puntos + resultado real, por familia de mercado."""
    out = {k: ([], []) for k in MARKET_FAMILIES}
    lines = {'totals': [], 'fav_handicap': []}
    games_bias = []
    for i, r in enumerate(rows.itertuples(index=False)):
        bo = int(r.best_of)
        a, b = (float(x[0]) for x in markov.solve_shift(np.array([r.pa]), np.array([r.pb]),
                                                         np.array([p_final[i]]), best_of=bo, iters=25))
        m = markov.markets(a, b, bo)
        raw = markov._markets_cached(round(a, 4), round(b, 4), bo, 7)
        need = bo // 2 + 1
        fav_a = p_final[i] >= .5
        out['straight'][0].append(m['n_sets'][need]); out['straight'][1].append(int(r.n_sets == need))
        out['set1_fav'][0].append(m['set1_a'] if fav_a else 1 - m['set1_a'])
        out['set1_fav'][1].append(int(bool(r.set1_a) == fav_a))
        if bo == 3:
            for ln in TOTAL_LINES:
                out['totals'][0].append(sum(p for k, p in raw['total_games'].items() if k > ln))
                out['totals'][1].append(int(r.total_games > ln)); lines['totals'].append(ln)
            games_bias.append(r.total_games - m['expected_games'])
            fav_margin = r.margin_a if fav_a else -r.margin_a
            for h in HCP_LINES:
                out['fav_handicap'][0].append(m['a_handicap'][h] if fav_a else m['b_handicap'][h])
                out['fav_handicap'][1].append(int(fav_margin > h)); lines['fav_handicap'].append(h)
        fav_won = (r.a_win == 1) == fav_a
        out['dog_set'][0].append(m['b_wins_set'] if fav_a else m['a_wins_set'])
        out['dog_set'][1].append(int((not fav_won) or r.l_sets >= 1))
        out['tiebreak'][0].append(m['tiebreak']); out['tiebreak'][1].append(int(r.tb_any > 0))
    res = {k: (np.asarray(p, float), np.asarray(y, float)) for k, (p, y) in out.items()}
    res['_games_bias'] = float(np.mean(games_bias)) if games_bias else 0.0
    res['_lines'] = {k: np.asarray(v, float) for k, v in lines.items()}
    return res


def fit_platt(p, y) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    z = _logit(p).reshape(-1, 1)
    lr = LogisticRegression(C=100.0, max_iter=1000).fit(z, y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def apply_platt(p, ab) -> np.ndarray:
    a, b = ab
    return 1 / (1 + np.exp(-(a + b * _logit(p))))


def market_report(samples: dict, calib: dict, base_rates: dict) -> list[dict]:
    rows = []
    for k, label in MARKET_FAMILIES.items():
        p_raw, y = samples[k]
        if len(y) == 0:
            continue
        p = apply_platt(p_raw, calib[k]) if k in calib else p_raw
        if k in samples.get('_lines', {}) and isinstance(base_rates.get(k), dict):
            # referencia por línea: la frecuencia histórica de ESA línea (más exigente)
            ln = samples['_lines'][k]
            base = np.array([base_rates[k][str(x)] for x in ln])
        else:
            base = np.full(len(y), base_rates.get(k, float(y.mean())))
        brier = float(np.mean((p - y) ** 2)); brier_raw = float(np.mean((p_raw - y) ** 2))
        brier_base = float(np.mean((base - y) ** 2))
        acc = float(np.mean((p >= .5) == y)); acc_base = float(np.mean((base >= .5) == y))
        ok = brier < brier_base - 1e-4 and acc >= acc_base - 0.002
        rows.append({'clave': k, 'mercado': label, 'muestras': int(len(y)), 'prob_media': float(p.mean()),
                     'frecuencia_real': float(y.mean()), 'brier': brier, 'brier_sin_calibrar': brier_raw,
                     'brier_referencia': brier_base, 'acierto': acc, 'acierto_referencia': acc_base,
                     'avalado': bool(ok)})
    return rows


class TreeModel:
    """Gradient Boosting exportado a JSON y evaluado con numpy (sin depender de scikit-learn)."""

    def __init__(self, baseline: float, trees: list[dict]):
        self.baseline = float(baseline)
        self.trees = [{k: np.asarray(v) for k, v in t.items()} for t in trees]

    def raw(self, x) -> np.ndarray:
        X = np.asarray(x, float)
        if X.ndim == 1:
            X = X[None, :]
        out = np.full(len(X), self.baseline)
        rows = np.arange(len(X))
        for t in self.trees:
            node = np.zeros(len(X), dtype=int)
            for _ in range(64):
                leaf = t['leaf'][node]
                if leaf.all():
                    break
                f = t['feat'][node]
                v = X[rows, f]
                go_left = np.where(np.isnan(v), t['miss_left'][node], v <= t['thr'][node])
                nxt = np.where(go_left, t['left'][node], t['right'][node])
                node = np.where(leaf, node, nxt)
            out += t['value'][node]
        return out

    def predict(self, x) -> np.ndarray:
        return 1 / (1 + np.exp(-self.raw(x)))

    @classmethod
    def from_sklearn(cls, model) -> 'TreeModel':
        trees = []
        for it in model._predictors:
            n = it[0].nodes
            trees.append({'feat': n['feature_idx'].astype(int).tolist(), 'thr': n['num_threshold'].astype(float).tolist(),
                          'miss_left': n['missing_go_to_left'].astype(bool).tolist(),
                          'left': n['left'].astype(int).tolist(), 'right': n['right'].astype(int).tolist(),
                          'leaf': n['is_leaf'].astype(bool).tolist(), 'value': n['value'].astype(float).tolist()})
        return cls(float(np.ravel(model._baseline_prediction)[0]), trees)

    def to_dict(self) -> dict:
        return {'features': FEATURES, 'baseline': self.baseline,
                'trees': [{k: (v.tolist()) for k, v in t.items()} for t in self.trees]}

    @classmethod
    def from_dict(cls, d: dict) -> 'TreeModel':
        return cls(d['baseline'], d['trees'])

    @classmethod
    def fit(cls, x, y, **params) -> 'TreeModel':
        from sklearn.ensemble import HistGradientBoostingClassifier
        xa = np.asarray(x, float)
        sym = -xa.copy()
        sym[:, FEATURES.index('rank_missing')] = xa[:, FEATURES.index('rank_missing')]
        hp = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=300,
                  l2_regularization=1.0, random_state=1, early_stopping=False)
        hp.update(params)
        m = HistGradientBoostingClassifier(**hp)
        m.fit(np.vstack([xa, sym]), np.concatenate([y, 1 - np.asarray(y)]))
        tm = cls.from_sklearn(m)
        tm._sk = m
        return tm

    def predict_sym(self, x) -> np.ndarray:
        """Promedia p(A) y 1-p(B) para que el orden de los jugadores no importe."""
        xa = np.asarray(x, float)
        sym = -xa.copy()
        sym[:, FEATURES.index('rank_missing')] = xa[:, FEATURES.index('rank_missing')]
        return 0.5 * (self.predict(xa) + 1 - self.predict(sym))
