"""Motor cronológico de ratings: Elo (general + superficie), saque/resto ajustados al rival,
forma, cansancio, ranking y cara a cara.

Cada partido se procesa así: 1) se calculan las características con el estado PREVIO,
2) recién después se actualiza el estado con el resultado. Así no hay fuga de información.
El mismo código se usa para entrenar (histórico) y para actualizar en producción.
"""
from __future__ import annotations

import math
from datetime import date as _date

import numpy as np
import pandas as pd

from .names import key as name_key
from .schema import LEVEL_K, SURFACES

HALF_LIFE_DAYS = 365.0          # vida media de las estadísticas de saque/resto
K_SERVE_ALL = 300.0             # puntos "prior" que encogen el rating de saque hacia la media
K_SERVE_SURF = 500.0            # puntos que encogen el rating de superficie hacia el general
MOV_WEIGHT = 1.0                # ajustado con validación 2024-25 (mejora log loss ~0,01)
INIT_ELO = {'ITF': 1300.0, 'C': 1400.0}
DEFAULT_SERVE_AVG = {('ATP', 'Hard'): 0.635, ('ATP', 'Clay'): 0.615, ('ATP', 'Grass'): 0.66,
                     ('WTA', 'Hard'): 0.56, ('WTA', 'Clay'): 0.55, ('WTA', 'Grass'): 0.575}


def _ord(d) -> int:
    if isinstance(d, (int, np.integer)):
        return int(d)
    return pd.Timestamp(d).toordinal()


def elo_k(n: int, level: str) -> float:
    return 250.0 / ((n + 5) ** 0.4) * LEVEL_K.get(level, 1.0)


class Player:
    __slots__ = ('name', 'elo', 'n', 'selo', 'sn', 'last', 'recent', 'srv', 'stamp', 'rank', 'rank_date',
                 'wins', 'first_date')

    def __init__(self, name: str, level: str = 'A'):
        self.name = name
        self.elo = INIT_ELO.get(level, 1500.0)
        self.n = 0
        self.selo: dict[str, float] = {}
        self.sn: dict[str, int] = {}
        self.last: int | None = None
        self.recent: list = []        # [(ordinal, ganó, esperado)] últimos 20
        # srv[surf or 'all'] = [suma_resid_saque, pts_saque, suma_resid_resto, pts_resto]
        self.srv: dict[str, list[float]] = {}
        self.stamp: int | None = None
        self.rank: float | None = None
        self.rank_date: int | None = None
        self.wins = 0
        self.first_date: int | None = None

    # ------------------------------------------------------------------ Elo
    def surf_elo(self, surface: str) -> float:
        return self.selo.get(surface, self.elo)

    def blend(self, surface: str) -> float:
        return 0.5 * self.elo + 0.5 * self.surf_elo(surface)

    # ---------------------------------------------------------- saque/resto
    def decay(self, t: int) -> None:
        if self.stamp is not None and t > self.stamp and self.srv:
            f = 0.5 ** ((t - self.stamp) / HALF_LIFE_DAYS)
            for v in self.srv.values():
                v[0] *= f; v[1] *= f; v[2] *= f; v[3] *= f
        self.stamp = t if self.stamp is None else max(self.stamp, t)

    def serve_rating(self, surface: str) -> tuple[float, float, float, float]:
        """(rating_saque, rating_resto, pts_saque, pts_resto) encogidos hacia 0 (= media)."""
        a = self.srv.get('all', [0.0, 0.0, 0.0, 0.0])
        s = self.srv.get(surface, [0.0, 0.0, 0.0, 0.0])
        rs_all = a[0] / (a[1] + K_SERVE_ALL)
        rr_all = a[2] / (a[3] + K_SERVE_ALL)
        rs = (s[0] + K_SERVE_SURF * rs_all) / (s[1] + K_SERVE_SURF)
        rr = (s[2] + K_SERVE_SURF * rr_all) / (s[3] + K_SERVE_SURF)
        return rs, rr, a[1], a[3]

    def form(self) -> float:
        rec = self.recent[-10:]
        if not rec:
            return 0.0
        return sum(w - e for _, w, e in rec) / (len(rec) + 3)

    def load(self, t: int, days: int) -> int:
        return sum(1 for d, _, _ in self.recent if t - days <= d < t)

    # ------------------------------------------------------------ serializa
    def to_dict(self) -> dict:
        return {'name': self.name, 'elo': round(self.elo, 2), 'n': self.n,
                'selo': {k: round(v, 2) for k, v in self.selo.items()}, 'sn': self.sn,
                'last': self.last, 'recent': [[d, int(w), round(e, 4)] for d, w, e in self.recent],
                'srv': {k: [round(x, 3) for x in v] for k, v in self.srv.items()},
                'stamp': self.stamp, 'rank': self.rank, 'rank_date': self.rank_date,
                'wins': self.wins, 'first_date': self.first_date}

    @classmethod
    def from_dict(cls, d: dict) -> 'Player':
        p = cls(d['name'])
        p.elo = d['elo']; p.n = d['n']; p.selo = d['selo']; p.sn = d['sn']; p.last = d['last']
        p.recent = [tuple(x) for x in d['recent']]; p.srv = d['srv']; p.stamp = d['stamp']
        p.rank = d['rank']; p.rank_date = d['rank_date']; p.wins = d.get('wins', 0)
        p.first_date = d.get('first_date')
        return p


class Engine:
    def __init__(self):
        self.players: dict[tuple[str, str], Player] = {}
        self.h2h: dict[tuple[str, str, str], list[int]] = {}   # (tour, k1, k2) k1<k2 -> [w1, w2]
        # media de puntos ganados al saque por circuito/superficie (decaída)
        self.avg: dict[str, list[float]] = {f'{t}|{s}': [DEFAULT_SERVE_AVG[(t, s)] * 2000, 2000.0]
                                            for (t, s) in DEFAULT_SERVE_AVG}
        self.last_date: int | None = None
        self.n_matches = 0

    # ------------------------------------------------------------- helpers
    def player(self, tour: str, name: str, level: str = 'A', create: bool = True) -> Player | None:
        k = (tour, name_key(name))
        p = self.players.get(k)
        if p is None and create:
            p = self.players[k] = Player(name, level)
        return p

    def serve_avg(self, tour: str, surface: str) -> float:
        a = self.avg.get(f'{tour}|{surface}')
        return a[0] / a[1] if a else 0.6

    def h2h_record(self, tour: str, a: str, b: str) -> tuple[int, int]:
        ka, kb = name_key(a), name_key(b)
        k = (tour, *sorted((ka, kb)))
        rec = self.h2h.get(k, [0, 0])
        return (rec[0], rec[1]) if ka <= kb else (rec[1], rec[0])

    # ------------------------------------------------------------ features
    def features(self, tour: str, a: Player, b: Player, surface: str, t: int, best_of: int) -> dict:
        a.decay(t); b.decay(t)
        avg = self.serve_avg(tour, surface)
        rsa, rra, pts_a, _ = a.serve_rating(surface)
        rsb, rrb, pts_b, _ = b.serve_rating(surface)
        pa = float(np.clip(avg + rsa - rrb, 0.3, 0.9))
        pb = float(np.clip(avg + rsb - rra, 0.3, 0.9))
        ea, eb = a.blend(surface), b.blend(surface)
        p_elo = 1.0 / (1.0 + 10 ** ((eb - ea) / 400.0))
        ra, rb = a.rank, b.rank
        rank_ok = bool(ra and rb and ra > 0 and rb > 0)
        rest_a = (t - a.last) if a.last is not None else 120
        rest_b = (t - b.last) if b.last is not None else 120
        w1, w2 = self.h2h_record(tour, a.name, b.name)
        return {
            'p_elo': p_elo, 'elo_a': ea, 'elo_b': eb, 'elo_all_a': a.elo, 'elo_all_b': b.elo,
            'elo_surf_a': a.surf_elo(surface), 'elo_surf_b': b.surf_elo(surface),
            'pa': pa, 'pb': pb, 'serve_avg': avg,
            'rs_a': rsa, 'rr_a': rra, 'rs_b': rsb, 'rr_b': rrb, 'pts_a': pts_a, 'pts_b': pts_b,
            'w_serve': min(pts_a / (pts_a + K_SERVE_ALL), pts_b / (pts_b + K_SERVE_ALL)),
            'n_a': a.n, 'n_b': b.n, 'sn_a': a.sn.get(surface, 0), 'sn_b': b.sn.get(surface, 0),
            'rank_a': ra, 'rank_b': rb,
            'log_rank': (math.log(rb) - math.log(ra)) if rank_ok else 0.0, 'rank_missing': 0 if rank_ok else 1,
            'form_a': a.form(), 'form_b': b.form(),
            'load7_a': a.load(t, 7), 'load7_b': b.load(t, 7),
            'load14_a': a.load(t, 14), 'load14_b': b.load(t, 14),
            'rest_a': rest_a, 'rest_b': rest_b,
            'h2h_a': w1, 'h2h_b': w2, 'bo5': int(best_of == 5),
        }

    # -------------------------------------------------------------- update
    def update(self, tour: str, winner: str, loser: str, surface: str, level: str, t: int,
               stats: dict | None = None, w_rank=None, l_rank=None, p_elo_pre: float | None = None,
               games: tuple[int, int] | None = None) -> None:
        w = self.player(tour, winner, level)
        l = self.player(tour, loser, level)
        for p in (w, l):
            if p.first_date is None:
                p.first_date = t
            if surface not in p.selo:
                p.selo[surface] = p.elo
        # Margen de victoria: ganar 6-1 6-1 mueve más el Elo que ganar 7-6 7-6.
        mov = 1.0
        if games and sum(games) > 0:
            mov = float(np.clip(1 + MOV_WEIGHT * (games[0] / sum(games) - 0.5) * 4, 0.5, 3.0))
        # Elo general
        exp = 1.0 / (1.0 + 10 ** ((l.elo - w.elo) / 400.0))
        w.elo += elo_k(w.n, level) * mov * (1 - exp)
        l.elo -= elo_k(l.n, level) * mov * (1 - exp)
        # Elo de superficie
        sw, sl = w.selo[surface], l.selo[surface]
        exp_s = 1.0 / (1.0 + 10 ** ((sl - sw) / 400.0))
        w.selo[surface] = sw + elo_k(w.sn.get(surface, 0), level) * mov * (1 - exp_s)
        l.selo[surface] = sl - elo_k(l.sn.get(surface, 0), level) * mov * (1 - exp_s)
        # Saque / resto ajustados al rival
        if stats:
            self._update_serve(tour, w, l, surface, t, stats['w_svpt'], stats['w_1stWon'] + stats['w_2ndWon'])
            self._update_serve(tour, l, w, surface, t, stats['l_svpt'], stats['l_1stWon'] + stats['l_2ndWon'])
        pe = p_elo_pre if p_elo_pre is not None else exp
        for p, won, e in ((w, 1, pe), (l, 0, 1 - pe)):
            p.n += 1
            p.sn[surface] = p.sn.get(surface, 0) + 1
            p.last = t
            p.recent.append((t, won, e))
            if len(p.recent) > 20:
                del p.recent[0]
            p.wins += won
        for p, r in ((w, w_rank), (l, l_rank)):
            if r is not None and not (isinstance(r, float) and math.isnan(r)) and r > 0:
                p.rank = float(r); p.rank_date = t
        k1, k2 = name_key(w.name), name_key(l.name)
        hk = (tour, *sorted((k1, k2)))
        rec = self.h2h.setdefault(hk, [0, 0])
        rec[0 if k1 <= k2 else 1] += 1
        self.last_date = t if self.last_date is None else max(self.last_date, t)
        self.n_matches += 1

    def _update_serve(self, tour, server: Player, receiver: Player, surface, t, svpt, won) -> None:
        if not svpt or svpt < 20 or won is None or won > svpt:
            return
        server.decay(t); receiver.decay(t)
        avg = self.serve_avg(tour, surface)
        rs, _, _, _ = server.serve_rating(surface)
        _, rr, _, _ = receiver.serve_rating(surface)
        actual = won / svpt
        res_srv = actual - (avg - rr)            # habilidad de saque, descontando el resto rival
        res_ret = (1 - actual) - ((1 - avg) - rs)  # habilidad de resto, descontando el saque rival
        for key_s in ('all', surface):
            s = server.srv.setdefault(key_s, [0.0, 0.0, 0.0, 0.0])
            s[0] += res_srv * svpt; s[1] += svpt
            r = receiver.srv.setdefault(key_s, [0.0, 0.0, 0.0, 0.0])
            r[2] += res_ret * svpt; r[3] += svpt
        a = self.avg.setdefault(f'{tour}|{surface}', [0.6 * 2000, 2000.0])
        a[0] = a[0] * 0.9995 + won
        a[1] = a[1] * 0.9995 + svpt

    def add_h2h(self, tour: str, winner: str, loser: str) -> None:
        k1, k2 = name_key(winner), name_key(loser)
        rec = self.h2h.setdefault((tour, *sorted((k1, k2))), [0, 0])
        rec[0 if k1 <= k2 else 1] += 1

    # ------------------------------------------------------------ persist
    def to_dict(self, keep_since: int | None = None, include_h2h: bool = True) -> dict:
        players = {f'{t}|{k}': p.to_dict() for (t, k), p in self.players.items()
                   if keep_since is None or (p.last or 0) >= keep_since}
        kept = set(players)
        h2h = [[t, k1, k2, v[0], v[1]] for (t, k1, k2), v in self.h2h.items()
               if include_h2h and f'{t}|{k1}' in kept and f'{t}|{k2}' in kept]
        return {'players': players, 'h2h': h2h, 'avg': self.avg, 'last_date': self.last_date,
                'n_matches': self.n_matches}

    @classmethod
    def from_dict(cls, d: dict) -> 'Engine':
        e = cls()
        for k, v in d['players'].items():
            t, nk = k.split('|', 1)
            e.players[(t, nk)] = Player.from_dict(v)
        e.h2h = {(t, k1, k2): [a, b] for t, k1, k2, a, b in d.get('h2h', [])}
        e.avg = d['avg']; e.last_date = d['last_date']; e.n_matches = d['n_matches']
        return e


def ordinal(d) -> int:
    return _ord(d)


def from_ordinal(t: int) -> _date:
    return _date.fromordinal(int(t))
