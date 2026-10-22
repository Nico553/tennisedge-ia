"""Tennis API ATP/WTA/ITF (RapidAPI · Matchstat/SportsAPI365), plan FREE: 50 llamadas/día.

Se usa para: (1) partidos próximos con cuotas y superficie, (2) resultados recientes
para mantener los ratings al día desde el fin del histórico (junio 2026) hasta hoy.
La clave vive solo en la variable de entorno RAPIDAPI_TENNIS_KEY (Render / GitHub Secrets).
"""
from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

from .schema import COLUMNS, parse_score

HOST = 'tennis-api-atp-wta-itf.p.rapidapi.com'
BASE = f'https://{HOST}/tennis/v2'
ENV_KEY = 'RAPIDAPI_TENNIS_KEY'
COURT_SURFACE = {1: 'Hard', 2: 'Clay', 3: 'Hard', 4: 'Hard', 5: 'Grass'}
SLAMS = ('australian open', 'roland garros', 'french open', 'wimbledon', 'us open')


class RapidError(RuntimeError):
    pass


class Quota:
    remaining: int | None = None
    calls = 0


def available() -> bool:
    return bool(os.environ.get(ENV_KEY, '').strip())


def request(path: str, params: dict | None = None, timeout: int = 25) -> dict:
    key = os.environ.get(ENV_KEY, '').strip()
    if not key:
        raise RapidError(f'Falta {ENV_KEY}. Configúrala en Render → Environment.')
    if not path.startswith('/') or '..' in path or '://' in path:
        raise ValueError('Ruta no permitida')
    try:
        r = requests.get(BASE + path, params=params or {}, timeout=timeout,
                         headers={'X-RapidAPI-Key': key, 'X-RapidAPI-Host': HOST, 'Accept': 'application/json'})
    except requests.RequestException as exc:
        raise RapidError('RapidAPI no respondió; intenta más tarde.') from exc
    Quota.calls += 1
    rem = r.headers.get('x-ratelimit-requests-remaining')
    if rem is not None and str(rem).isdigit():
        Quota.remaining = int(rem)
    if r.status_code in (401, 403):
        raise RapidError('RapidAPI rechazó la clave o tu plan no incluye este endpoint '
                         '(¿te suscribiste al plan Basic/Free de la API?).')
    if r.status_code == 429:
        raise RapidError('Se agotaron las llamadas gratuitas de hoy en RapidAPI (50/día).')
    if r.status_code != 200:
        raise RapidError(f'RapidAPI respondió HTTP {r.status_code}.')
    try:
        data = r.json()
    except ValueError as exc:
        raise RapidError('RapidAPI devolvió algo que no es JSON.') from exc
    if not isinstance(data, (dict, list)):
        raise RapidError('Respuesta inesperada de RapidAPI.')
    return data if isinstance(data, dict) else {'data': data}


# ----------------------------------------------------------------- utilidades
def _obj(v: Any) -> dict:
    return v if isinstance(v, dict) else {}


def _odd(v) -> float | None:
    try:
        x = float(v)
        return x if 1.0 < x < 1000 else None
    except (TypeError, ValueError):
        return None


def surface_of(tourney: dict, court: Any = None) -> str | None:
    c = court if court is not None else tourney.get('court')
    name = _obj(c).get('name') or (c if isinstance(c, str) else None) or tourney.get('surface')
    if name:
        n = str(name).lower()
        if 'grass' in n:
            return 'Grass'
        if 'clay' in n:
            return 'Clay'
        if 'hard' in n or 'carpet' in n or 'acrylic' in n:
            return 'Hard'
    try:
        return COURT_SURFACE.get(int(tourney.get('courtId') or _obj(c).get('id')))
    except (TypeError, ValueError):
        return None


def level_of(tourney: dict) -> str:
    name = str(tourney.get('name') or '').lower()
    if any(s in name for s in SLAMS):
        return 'G'
    if re.search(r'\b(itf|m15|m25|w15|w35|w50|w75|w100|m15|w25)\b', name):
        return 'ITF'
    if 'challenger' in name or re.search(r'\bch\b|125', name):
        return 'C'
    try:
        rank = int(tourney.get('rankId'))
    except (TypeError, ValueError):
        rank = None
    if rank == 0:
        return 'ITF'
    if rank == 1:
        return 'C'
    return 'A'


def _orient(score: str, p1_won: bool) -> str | None:
    """Devuelve el marcador desde la óptica del ganador."""
    s = str(score or '').replace(',', ' ').strip()
    if parse_score(s) is not None and p1_won:
        return s
    flipped = ' '.join(re.sub(r'^(\d+)-(\d+)', r'\2-\1', t) for t in s.split())
    flipped = re.sub(r'\((\d+)\)', r'(\1)', flipped)
    if parse_score(flipped) is not None:
        return flipped
    if parse_score(s) is not None:
        return s
    return None


# -------------------------------------------------------------- resultados
def get_results(tour: str, start: date, end: date, max_pages: int = 3) -> list[dict]:
    rows: list[dict] = []
    for page in range(1, max_pages + 1):
        payload = request(f'/{tour}/results/{start.isoformat()}/{end.isoformat()}',
                          {'filter': 'PlayerGroup:singles', 'pageSize': 500, 'pageNo': page})
        chunk = payload.get('data', [])
        if not isinstance(chunk, list):
            break
        rows.extend(c for c in chunk if isinstance(c, dict))
        if not payload.get('hasNextPage') or not chunk:
            break
    return rows


def parse_results(rows: list[dict], tour: str) -> pd.DataFrame:
    out = []
    for r in rows:
        p1, p2 = _obj(r.get('player1')), _obj(r.get('player2'))
        n1, n2 = str(p1.get('name') or '').strip(), str(p2.get('name') or '').strip()
        if not n1 or not n2 or '/' in n1 or '/' in n2 or n1 == n2:
            continue
        rtype = str(r.get('result_type') or '').lower()
        score = str(r.get('result') or '')
        retired = bool(re.search(r'ret|w/?o|walkover|def|abd', score + ' ' + rtype, re.I))
        win_id = r.get('match_winner')
        id1 = r.get('player1Id', p1.get('id'))
        id2 = r.get('player2Id', p2.get('id'))
        if win_id is not None and str(win_id) == str(id2):
            p1_won = False
        elif win_id is not None and str(win_id) == str(id1):
            p1_won = True
        else:
            p1_won = True  # convención del proveedor: player1 = ganador
        winner, loser = (n1, n2) if p1_won else (n2, n1)
        o1, o2 = _odd(r.get('odd1')), _odd(r.get('odd2'))
        ow, ol = (o1, o2) if p1_won else (o2, o1)
        sc = None if retired else _orient(score, p1_won)
        t = _obj(r.get('tournament'))
        d = pd.to_datetime(r.get('date'), utc=True, errors='coerce')
        if pd.isna(d):
            continue
        lvl = level_of(t)
        best_of = r.get('best_of')
        try:
            best_of = int(best_of)
        except (TypeError, ValueError):
            best_of = 5 if (lvl == 'G' and tour.upper() == 'ATP') else 3
        row = {c: None for c in COLUMNS}
        row.update({'date': d.tz_convert(None).normalize(), 'tour': tour.upper(),
                    'tourney_id': str(t.get('id') or r.get('tournamentId') or ''),
                    'tourney_name': str(t.get('name') or ''), 'level': lvl,
                    'surface': surface_of(t), 'round': str(r.get('roundId') or ''), 'best_of': best_of,
                    'winner': winner, 'loser': loser, 'score': sc or score, 'retired': retired or sc is None,
                    'odds_w': ow, 'odds_l': ol, 'source': 'rapidapi'})
        out.append(row)
    df = pd.DataFrame(out, columns=COLUMNS)
    return df[df.surface.notna()] if not df.empty else df


def fetch_range(start: date, end: date, budget: int, tours=('atp', 'wta'), log=None) -> tuple[pd.DataFrame, int, date]:
    """Descarga resultados en ventanas de 7 días respetando un presupuesto de llamadas.

    Devuelve (resultados, llamadas_usadas, última_fecha_completa_cubierta).
    """
    frames = []
    used0 = Quota.calls
    cur = start
    covered = start - timedelta(days=1)
    while cur <= end:
        stop = min(cur + timedelta(days=6), end)
        if Quota.calls - used0 + 2 * 3 > budget:
            if log:
                log('Presupuesto de llamadas alcanzado; se continuará en la próxima actualización.')
            break
        ok = True
        for tour in tours:
            try:
                rows = get_results(tour, cur, stop)
                frames.append(parse_results(rows, tour))
                if log:
                    log(f'{tour.upper()} {cur} → {stop}: {len(rows)} partidos')
            except RapidError as exc:
                ok = False
                if log:
                    log(f'{tour.upper()} {cur} → {stop}: {exc}')
                break
        if not ok:
            break
        covered = stop
        cur = stop + timedelta(days=1)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    return df, Quota.calls - used0, covered


# ---------------------------------------------------------------- próximos
def get_upcoming(max_pages: int = 2) -> list[dict]:
    out, seen = [], set()
    for page in range(1, max_pages + 1):
        payload = request('/upcoming/matches', {'group': 'singles', 'includeAll': 'true',
                                                'include': 'preMatchOdds', 'page': page, 'limit': 200})
        rows = payload.get('matches', payload.get('data', []))
        if not isinstance(rows, list):
            break
        for r in rows:
            mid = str(_obj(r).get('matchId') or _obj(r).get('id') or '')
            if mid and mid in seen:
                continue
            seen.add(mid)
            out.append(r)
        if not payload.get('hasNextPage') or not rows:
            break
    return out


def parse_upcoming(rows: list[dict], now: datetime | None = None) -> pd.DataFrame:
    now = now or datetime.now(timezone.utc)
    recs = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        p1, p2 = _obj(r.get('player1')), _obj(r.get('player2'))
        n1, n2 = str(p1.get('name') or '').strip(), str(p2.get('name') or '').strip()
        if not n1 or not n2 or '/' in n1 or '/' in n2 or n1 == n2:
            continue
        t = _obj(r.get('tournament'))
        start = pd.to_datetime(r.get('date') or r.get('startTime'), utc=True, errors='coerce')
        if pd.notna(start) and start.to_pydatetime() <= now:
            continue
        pre, full = _obj(r.get('preMatchOdds')), _obj(r.get('odds'))
        pairs = [(_odd(pre.get('player1')), _odd(pre.get('player2'))), (_odd(full.get('k1')), _odd(full.get('k2'))),
                 (_odd(p1.get('odd')), _odd(p2.get('odd')))]
        o1, o2 = next(((a, b) for a, b in pairs if a and b), (None, None))
        tour_txt = ' '.join(str(r.get(k) or '') for k in ('type', 'tour', 'tourType')).lower() + ' ' + str(t.get('name', '')).lower()
        tour = 'WTA' if 'wta' in tour_txt or 'women' in tour_txt else ('ATP' if ('atp' in tour_txt or 'men' in tour_txt) else None)
        lvl = level_of(t)
        recs.append({'id': str(r.get('matchId') or r.get('id') or ''), 'inicio': start,
                     'torneo': str(t.get('name') or 'Torneo'), 'tour': tour, 'level': lvl,
                     'surface': surface_of(t, r.get('court')), 'jugador_a': n1, 'jugador_b': n2,
                     'cuota_a': o1, 'cuota_b': o2,
                     'best_of': 5 if (lvl == 'G' and tour == 'ATP') else 3})
    df = pd.DataFrame(recs)
    if not df.empty:
        df = df.drop_duplicates(subset=['id', 'jugador_a', 'jugador_b']).sort_values('inicio', na_position='last')
    return df.reset_index(drop=True)


def raw_sample_keys(rows: list[dict]) -> list[str]:
    """Para diagnóstico: lista de campos del primer registro (sin valores)."""
    def walk(o, p=''):
        ks = []
        if isinstance(o, dict):
            for k, v in o.items():
                ks.append(p + k)
                ks += walk(v, p + k + '.')
        elif isinstance(o, list) and o:
            ks += walk(o[0], p + '[].')
        return ks
    return walk(rows[0]) if rows else []
