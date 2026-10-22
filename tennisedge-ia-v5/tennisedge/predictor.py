"""Carga el modelo entrenado, lo pone al día con resultados recientes y genera análisis."""
from __future__ import annotations

import gzip
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import markov
from .model import FEATURE_LABEL, TreeModel, apply_platt, feature_frame
from .names import NameIndex, key as name_key
from .ratings import Engine, Player, ordinal
from .schema import LEVEL_NAME, parse_score

DATA = Path(__file__).resolve().parents[1] / 'data'


def value(p: float, odd: float | None) -> dict:
    if not odd or odd <= 1 or p is None:
        return {'cuota': odd, 'cuota_justa': (1 / p) if p else None, 'ev': None, 'ventaja_pp': None}
    return {'cuota': odd, 'cuota_justa': 1 / p, 'implicita': 1 / odd, 'ev': p * odd - 1,
            'ventaja_pp': (p - 1 / odd) * 100}


def fair_probs(o1: float | None, o2: float | None) -> tuple[float | None, float | None]:
    if not o1 or not o2:
        return None, None
    a, b = 1 / o1, 1 / o2
    return a / (a + b), b / (a + b)


class Predictor:
    def __init__(self, data_dir: Path = DATA):
        with gzip.open(data_dir / 'engine_state.json.gz', 'rt', encoding='utf-8') as fh:
            self.engine = Engine.from_dict(json.load(fh))
        bundle = json.loads((data_dir / 'model.json').read_text(encoding='utf-8'))
        self.model = TreeModel.from_dict(bundle['model'])
        self.report = bundle['report']
        self.calib = {k: tuple(v) for k, v in self.report.get('market_calibration', {}).items()}
        self.avalado = {m['clave']: m['avalado'] for m in self.report.get('markets', [])}
        self.archive_last = self.report['archive_last_date']
        hist = pd.read_csv(data_dir / 'history.csv.gz', low_memory=False)
        hist['date'] = pd.to_datetime(hist['date'])
        for c in ('tour', 'tourney_name', 'level', 'surface', 'round', 'winner', 'loser'):
            hist[c] = hist[c].astype('category')
        for c in hist.columns:
            if hist[c].dtype == 'float64':
                hist[c] = hist[c].astype('float32')
        self.history = hist
        for r in hist[['tour', 'winner', 'loser', 'retired']].itertuples(index=False):
            if not r.retired:
                self.engine.add_h2h(r.tour, r.winner, r.loser)
        self.recent = pd.DataFrame()
        self.oos = pd.DataFrame()
        self._reindex()

    # ----------------------------------------------------------- nombres
    def _reindex(self):
        self.index = {t: NameIndex([k for (tt, k) in self.engine.players if tt == t]) for t in ('ATP', 'WTA')}

    def canonical(self, tour: str, name: str) -> str:
        k = self.index[tour].resolve(name)
        if k is not None:
            return self.engine.players[(tour, k)].name
        return str(name).strip()

    def players(self, tour: str, active_days: int = 400) -> list[str]:
        last = self.engine.last_date or 0
        ps = [(p.blend('Hard'), p.name) for (t, _), p in self.engine.players.items()
              if t == tour and p.last and last - p.last <= active_days and p.n >= 5]
        return [n for _, n in sorted(ps, reverse=True)]

    # ------------------------------------------------ resultados recientes
    def apply_results(self, df: pd.DataFrame) -> None:
        """Aplica partidos posteriores al histórico. Antes de actualizar cada uno, guarda la
        predicción del modelo (fuera de muestra) para compararla con las cuotas de la casa."""
        if df is None or df.empty:
            return
        df = df[pd.to_datetime(df['date']) > pd.Timestamp(self.archive_last)].sort_values('date')
        logs, kept = [], []
        for r in df.to_dict('records'):
            if str(r.get('retired')).lower() in ('true', '1') or not r.get('surface'):
                continue
            sc = parse_score(r.get('score'))
            if sc is None:
                continue
            tour = str(r['tour']).upper()
            if tour not in ('ATP', 'WTA'):
                continue
            w = self.canonical(tour, r['winner']); l = self.canonical(tour, r['loser'])
            lvl = r.get('level') or 'A'
            t = ordinal(r['date'])
            pw = self.engine.player(tour, w, lvl); pl = self.engine.player(tour, l, lvl)
            bo = int(r.get('best_of') or 3)
            seasoned = pw.n >= 5 and pl.n >= 5
            f = self.engine.features(tour, pw, pl, r['surface'], t, bo)
            f['p_markov'] = float(markov.match_win(f['pa'], f['pb'], bo))
            p = float(self.model.predict_sym(feature_frame(pd.DataFrame([f])).to_numpy())[0])
            logs.append({'date': r['date'], 'tour': tour, 'level': lvl, 'winner': w, 'loser': l,
                         'p_winner': p, 'p_elo_winner': f['p_elo'], 'odds_w': r.get('odds_w'),
                         'odds_l': r.get('odds_l'), 'seasoned': seasoned})
            self.engine.update(tour, w, l, r['surface'], lvl, t, None, None, None, f['p_elo'],
                               games=(sc['w_games'], sc['l_games']))
            self.engine.add_h2h(tour, w, l)
            kept.append({**r, 'winner': w, 'loser': l})
        self.recent = pd.DataFrame(kept)
        self.oos = pd.DataFrame(logs)
        self._reindex()

    # ------------------------------------------------------------ predicción
    def _get(self, tour: str, name: str, level: str) -> tuple[Player, bool]:
        canon = self.canonical(tour, name)
        p = self.engine.players.get((tour, name_key(canon)))
        if p is None:
            return Player(canon, level), False
        return p, True

    def predict(self, tour: str, a: str, b: str, surface: str, best_of: int = 3, level: str = 'A',
                when: date | None = None, odds_a: float | None = None, odds_b: float | None = None,
                full: bool = True) -> dict:
        tour = tour.upper()
        pa_, known_a = self._get(tour, a, level)
        pb_, known_b = self._get(tour, b, level)
        t = ordinal(pd.Timestamp(when or date.today()))
        f = self.engine.features(tour, pa_, pb_, surface, t, best_of)
        f['p_markov'] = float(markov.match_win(f['pa'], f['pb'], best_of))
        X = feature_frame(pd.DataFrame([f])).to_numpy()
        p = float(self.model.predict_sym(X)[0])
        if full:
            sa, sb = markov.solve_shift(np.array([f['pa']]), np.array([f['pb']]), np.array([p]), best_of, iters=25)
            sa, sb = float(sa[0]), float(sb[0])
            mk = self.calibrated_markets(sa, sb, best_of, p)
        else:
            sa, sb, mk = f['pa'], f['pb'], None
        conf = self.confidence(f, known_a and known_b)
        fair_a, fair_b = fair_probs(odds_a, odds_b)
        return {
            'tour': tour, 'a': pa_.name, 'b': pb_.name, 'surface': surface, 'best_of': best_of, 'level': level,
            'level_name': LEVEL_NAME.get(level, level), 'p_a': p, 'p_b': 1 - p,
            'p_elo': f['p_elo'], 'p_markov': f['p_markov'], 'serve_a': sa, 'serve_b': sb,
            'serve_measured_a': f['pa'], 'serve_measured_b': f['pb'], 'features': f,
            'factors': self.factors(f, pa_.name, pb_.name), 'markets': mk, 'confidence': conf,
            'known': (known_a, known_b),
            'value_a': value(p, odds_a), 'value_b': value(1 - p, odds_b),
            'market_fair': (fair_a, fair_b),
            'data_last': self.data_last_date(),
        }

    def predict_many(self, items: list[dict]) -> list[dict]:
        """Versión vectorizada (solo probabilidad, valor y confianza) para tablas grandes.
        items: dicts con tour, a, b, surface, best_of, level, odds_a, odds_b."""
        if not items:
            return []
        t = ordinal(pd.Timestamp(date.today()))
        feats, meta = [], []
        for it in items:
            tour = it['tour'].upper()
            pa_, ka = self._get(tour, it['a'], it.get('level', 'A'))
            pb_, kb = self._get(tour, it['b'], it.get('level', 'A'))
            f = self.engine.features(tour, pa_, pb_, it['surface'], t, int(it.get('best_of', 3)))
            feats.append(f); meta.append((pa_.name, pb_.name, ka and kb))
        df = pd.DataFrame(feats)
        bo = df['bo5'].map({1: 5, 0: 3}).to_numpy()
        pm = np.empty(len(df))
        for b_ in (3, 5):
            m = bo == b_
            if m.any():
                pm[m] = markov.match_win(df['pa'].to_numpy()[m], df['pb'].to_numpy()[m], b_)
        df['p_markov'] = pm
        p = self.model.predict_sym(feature_frame(df).to_numpy())
        out = []
        for i, it in enumerate(items):
            a, b, known = meta[i]
            f = feats[i]; f['p_markov'] = float(pm[i])
            out.append({'a': a, 'b': b, 'p_a': float(p[i]), 'p_b': float(1 - p[i]),
                        'confidence': self.confidence(f, known),
                        'value_a': value(float(p[i]), it.get('odds_a')),
                        'value_b': value(float(1 - p[i]), it.get('odds_b'))})
        return out

    def data_last_date(self) -> str:
        return str(date.fromordinal(self.engine.last_date)) if self.engine.last_date else self.archive_last

    # ------------------------------------------------------------ mercados
    def calibrated_markets(self, sa: float, sb: float, bo: int, p: float) -> dict:
        m = markov.markets(sa, sb, bo)
        need = bo // 2 + 1
        cal = self.calib
        fav_a = p >= .5
        out = dict(m)
        out['avalado'] = dict(self.avalado)
        out['calibrated'] = bool(cal)
        # Sets corridos: reparte manteniendo P(A gana)
        if 'straight' in cal:
            raw_st = m['n_sets'][need]
            st = float(apply_platt(np.array([raw_st]), cal['straight'])[0])
            k = st / raw_st if raw_st > 0 else 1
            ss = {s: v for s, v in m['set_scores'].items()}
            a_str, b_str = f'{need}-0', f'0-{need}'
            new = {a_str: ss.get(a_str, 0) * k, b_str: ss.get(b_str, 0) * k}
            rest_a = {s: v for s, v in ss.items() if int(s.split('-')[0]) == need and s != a_str}
            rest_b = {s: v for s, v in ss.items() if int(s.split('-')[1]) == need and s != b_str}
            for rest, target in ((rest_a, max(p - new[a_str], 0)), (rest_b, max(1 - p - new[b_str], 0))):
                tot = sum(rest.values()) or 1
                for s, v in rest.items():
                    new[s] = v / tot * target
            norm = sum(new.values())
            out['set_scores'] = {s: v / norm for s, v in sorted(new.items(), key=lambda kv: -kv[1])}
            out['n_sets'] = {n: sum(v for s, v in out['set_scores'].items() if sum(map(int, s.split('-'))) == n)
                             for n in range(need, bo + 1)}
            out['straight_a'] = out['set_scores'].get(a_str, 0); out['straight_b'] = out['set_scores'].get(b_str, 0)
        if 'dog_set' in cal:
            raw = m['b_wins_set'] if fav_a else m['a_wins_set']
            c = float(apply_platt(np.array([raw]), cal['dog_set'])[0])
            if fav_a:
                out['b_wins_set'] = c
            else:
                out['a_wins_set'] = c
        if 'set1_fav' in cal:
            raw = m['set1_a'] if fav_a else 1 - m['set1_a']
            c = float(apply_platt(np.array([raw]), cal['set1_fav'])[0])
            out['set1_a'] = c if fav_a else 1 - c
        if 'tiebreak' in cal:
            out['tiebreak'] = float(apply_platt(np.array([m['tiebreak']]), cal['tiebreak'])[0])
        if bo == 3 and 'totals' in cal:
            out['total_over'] = {ln: float(apply_platt(np.array([v]), cal['totals'])[0])
                                 for ln, v in m['total_over'].items()}
            out['expected_games'] = m['expected_games'] + float(self.report.get('games_bias', 0))
        if bo == 3 and 'fav_handicap' in cal:
            src = m['a_handicap'] if fav_a else m['b_handicap']
            fav = {h: float(apply_platt(np.array([v]), cal['fav_handicap'])[0]) for h, v in src.items()}
            if fav_a:
                out['a_handicap'] = fav
            else:
                out['b_handicap'] = fav
        out['fav_is_a'] = fav_a
        return out

    # --------------------------------------------------------- confianza
    @staticmethod
    def confidence(f: dict, both_known: bool) -> dict:
        n = min(f['n_a'], f['n_b']); sn = min(f['sn_a'], f['sn_b'])
        exp = n / (n + 15)
        serve = min(1.0, f['w_serve'] / 0.8)
        surf = sn / (sn + 6)
        rest = max(f['rest_a'], f['rest_b'])
        recency = 1.0 if rest <= 45 else max(0.0, 1 - (rest - 45) / 120)
        score = 100 * (0.35 * exp + 0.25 * serve + 0.2 * surf + 0.2 * recency) * (1 if both_known else 0.4)
        nivel = 'alta' if score >= 75 else ('media' if score >= 55 else 'baja')
        motivos = []
        if not both_known:
            motivos.append('Algún jugador no aparece en la base (nombre distinto o debutante)')
        if n < 15:
            motivos.append(f'Poca historia: {n} partidos del que menos tiene')
        if sn < 5:
            motivos.append('Pocos partidos en esta superficie')
        if f['w_serve'] < 0.6:
            motivos.append('Poca estadística de saque')
        if rest > 60:
            motivos.append(f'Inactividad: {int(rest)} días sin jugar')
        return {'score': int(round(score)), 'nivel': nivel, 'motivos': motivos}

    # --------------------------------------------------------- explicación
    @staticmethod
    def factors(f: dict, a: str, b: str) -> list[dict]:
        out = []

        def add(label, prob_a, detail):
            if prob_a is None:
                return
            fav = a if prob_a >= .5 else b
            strength = abs(prob_a - .5) * 2
            out.append({'factor': label, 'favorece': fav if strength > 0.02 else 'Neutral',
                        'fuerza': strength, 'prob_a': prob_a, 'detalle': detail})
        add('Elo en la superficie', f['p_elo'], f"{f['elo_a']:.0f} vs {f['elo_b']:.0f}")
        add('Saque y resto', f['p_markov'], f"saque {f['pa']*100:.1f}% vs {f['pb']*100:.1f}% (pts ganados)")
        if not f['rank_missing']:
            pr = 1 / (1 + math.exp(-0.85 * f['log_rank']))
            add('Ranking', pr, f"Nº {int(f['rank_a'])} vs Nº {int(f['rank_b'])}")
        fd = f['form_a'] - f['form_b']
        add('Forma reciente (vs lo esperado)', 1 / (1 + math.exp(-4 * fd)), f"{f['form_a']:+.2f} vs {f['form_b']:+.2f}")
        ld = (f['load7_b'] - f['load7_a'])
        add('Cansancio (últimos 7 días)', 1 / (1 + math.exp(-0.4 * ld)),
            f"{f['load7_a']} vs {f['load7_b']} partidos · descanso {min(f['rest_a'],120)} vs {min(f['rest_b'],120)} días")
        hw, hl = f['h2h_a'], f['h2h_b']
        add('Cara a cara', (hw + 1) / (hw + hl + 2) if hw + hl else 0.5, f"{hw}-{hl}")
        return out

    # ----------------------------------------------------------- perfiles
    def matches_of(self, tour: str, name: str) -> pd.DataFrame:
        h = self.history
        m = h[(h.tour == tour) & ((h.winner == name) | (h.loser == name))]
        if not self.recent.empty:
            r = self.recent
            m = pd.concat([m.astype({c: 'object' for c in m.select_dtypes('category').columns}),
                           r[(r.tour == tour) & ((r.winner == name) | (r.loser == name))]], ignore_index=True)
        return m.sort_values('date', ascending=False)

    def profile(self, tour: str, name: str, surface: str) -> dict:
        m = self.matches_of(tour, name)
        p = self.engine.players.get((tour, name_key(name)))
        res = {'name': name, 'elo': p.elo if p else None, 'elo_surface': p.surf_elo(surface) if p else None,
               'rank': p.rank if p else None, 'matches_total': p.n if p else 0}
        if m.empty:
            return {**res, 'last': pd.DataFrame(), 'stats': {}, 'record_surface': (0, 0)}
        m = m.copy()
        m['ganó'] = m['winner'] == name
        m['rival'] = np.where(m['ganó'], m['loser'], m['winner'])
        last_surf = m[m.surface == surface].head(10)
        year = m[(m.surface == surface) & (m.date >= m.date.max() - pd.Timedelta(days=365))]
        st = {}
        if len(year):
            me = lambda c: np.where(year['ganó'], year[f'w_{c}'], year[f'l_{c}']).astype(float)  # noqa: E731
            op = lambda c: np.where(year['ganó'], year[f'l_{c}'], year[f'w_{c}']).astype(float)  # noqa: E731
            svpt, fin, fwon, swon = me('svpt'), me('1stIn'), me('1stWon'), me('2ndWon')
            ok = ~np.isnan(svpt) & (svpt > 0)
            if ok.sum() >= 2:
                o_svpt, o_fw, o_sw = op('svpt'), op('1stWon'), op('2ndWon')
                sg, bps, bpf = me('SvGms'), me('bpSaved'), me('bpFaced')
                o_bps, o_bpf = op('bpSaved'), op('bpFaced')
                s = lambda x: np.nansum(x[ok])  # noqa: E731
                st = {'partidos_con_dato': int(ok.sum()),
                      '1er saque dentro': s(fin) / s(svpt),
                      'Puntos ganados con 1er saque': s(fwon) / max(s(fin), 1),
                      'Puntos ganados con 2º saque': s(swon) / max(s(svpt) - s(fin), 1),
                      'Puntos ganados al saque': (s(fwon) + s(swon)) / s(svpt),
                      'Puntos ganados al resto': 1 - (s(o_fw) + s(o_sw)) / max(s(o_svpt), 1),
                      'Juegos de saque ganados': 1 - (s(bpf) - s(bps)) / max(s(sg), 1),
                      'Break points salvados': s(bps) / max(s(bpf), 1),
                      'Break points convertidos': (s(o_bpf) - s(o_bps)) / max(s(o_bpf), 1),
                      'Aces por partido': s(me('ace')) / ok.sum(),
                      'Dobles faltas por partido': s(me('df')) / ok.sum()}
        surf_all = m[m.surface == surface]
        return {**res, 'last': last_surf[['date', 'tourney_name', 'round', 'rival', 'ganó', 'score']],
                'stats': st, 'record_surface': (int(surf_all['ganó'].sum()), int((~surf_all['ganó']).sum())),
                'recent_all': m.head(10)[['date', 'tourney_name', 'surface', 'rival', 'ganó', 'score']]}

    def h2h_matches(self, tour: str, a: str, b: str) -> pd.DataFrame:
        m = self.matches_of(tour, a)
        if m.empty:
            return m
        m = m[(m.winner == b) | (m.loser == b)]
        return m[['date', 'tourney_name', 'surface', 'round', 'winner', 'score']]

    # ------------------------------------------------- prueba con cuotas
    def oos_vs_market(self, seasoned_only: bool = True) -> dict | None:
        d = self.oos
        if d is None or d.empty:
            return None
        if seasoned_only:
            d = d[d.seasoned]
        d = d.dropna(subset=['odds_w', 'odds_l'])
        d = d[(d.odds_w > 1) & (d.odds_l > 1)]
        if len(d) < 30:
            return {'n': int(len(d))}
        fair_w = (1 / d.odds_w) / (1 / d.odds_w + 1 / d.odds_l)
        p = d.p_winner.clip(1e-6, 1 - 1e-6)
        res = {'n': int(len(d)), 'desde': str(pd.Timestamp(d.date.min()).date()), 'hasta': str(pd.Timestamp(d.date.max()).date()),
               'modelo_acierto': float((p > .5).mean()), 'casa_acierto': float((fair_w > .5).mean()),
               'modelo_brier': float(((1 - p) ** 2).mean()), 'casa_brier': float(((1 - fair_w) ** 2).mean()),
               'modelo_logloss': float(-np.log(p).mean()), 'casa_logloss': float(-np.log(fair_w.clip(1e-6)).mean())}
        # Simulación de apostar 1 unidad cuando EV > umbral (con la cuota disponible)
        sims = []
        for thr in (0.0, 0.05, 0.10):
            ev_w = p * d.odds_w - 1
            ev_l = (1 - p) * d.odds_l - 1
            stake_w = ev_w > thr
            stake_l = ev_l > thr
            profit = (stake_w * (d.odds_w - 1)).sum() - stake_l.sum()
            bets = int(stake_w.sum() + stake_l.sum())
            sims.append({'umbral_ev': thr, 'apuestas': bets, 'ganancia_unidades': float(profit),
                         'roi': float(profit / bets) if bets else None})
        res['simulacion'] = sims
        return res
