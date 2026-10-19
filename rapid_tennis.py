"""Conector opcional de Tennis API (RapidAPI): ATP/WTA/Challenger/ITF.

Sin scraping, sin datos inventados y sin pronósticos de terceros.
Solo endpoints documentados; la clave permanece exclusivamente en el servidor.
FREE: 50 llamadas/día según documentación pública; comprobar vigencia.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

import pandas as pd
import requests

RAPID_HOST = 'tennis-api-atp-wta-itf.p.rapidapi.com'
RAPID_BASE = f'https://{RAPID_HOST}/tennis/v2'
MAX_PAGES = 2
PAGE_SIZE = 200

class RapidTennisError(ValueError):
    pass


def available() -> bool:
    return bool(os.environ.get('RAPIDAPI_TENNIS_KEY', '').strip())


def request(path: str, params: dict | None = None, timeout: int = 18) -> dict:
    key = os.environ.get('RAPIDAPI_TENNIS_KEY', '').strip()
    if not key:
        raise RapidTennisError('Configura RAPIDAPI_TENNIS_KEY de forma privada en Render.')
    if (not path.startswith('/') or path.startswith('//') or '://' in path or '?' in path
            or '..' in path or '#' in path or '\n' in path):
        raise ValueError('Ruta no permitida')
    try:
        response = requests.get(
            RAPID_BASE + path,
            params=params or {},
            headers={'X-RapidAPI-Key': key, 'X-RapidAPI-Host': RAPID_HOST,
                     'Accept': 'application/json'},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RapidTennisError('La fuente secundaria no respondió.') from exc
    if response.status_code in (401,403):
        raise RapidTennisError('RapidAPI rechazó la clave o el plan no autoriza este endpoint.')
    if response.status_code == 429:
        raise RapidTennisError('Límite gratuito diario de RapidAPI agotado.')
    if response.status_code != 200:
        raise RapidTennisError(f'Error del proveedor secundario (HTTP {response.status_code}).')
    try:
        payload = response.json()
    except ValueError as exc:
        raise RapidTennisError('El proveedor devolvió datos no válidos.') from exc
    if not isinstance(payload, dict):
        raise RapidTennisError('Respuesta inesperada del proveedor.')
    if payload.get('success') is False:
        raise RapidTennisError('El proveedor informó que la consulta falló.')
    return payload


def get_upcoming(max_pages: int = MAX_PAGES) -> tuple[list[dict], dict]:
    """Una vista ATP+WTA del proveedor, incluye Challenger/ITF si existen.

    Nunca se continúa paginando sin límite; informa explícitamente si es parcial.
    """
    all_matches = []
    last_has_next = False
    seen = set()
    pages = 0
    total = None
    for page in range(1, min(max(1, max_pages), MAX_PAGES) + 1):
        payload = request('/upcoming/matches', {
            'group': 'singles', 'includeAll': 'true',
            'include': 'preMatchOdds', 'page': page, 'limit': PAGE_SIZE,
        })
        rows = payload.get('matches', [])
        if not isinstance(rows, list):
            raise RapidTennisError('La respuesta de partidos no tiene formato de lista.')
        pages = page
        total = payload.get('total', total)
        for row in rows:
            if not isinstance(row, dict):
                continue
            mid = row.get('matchId', row.get('id'))
            ident = str(mid) if mid is not None else ''
            if ident and ident in seen:
                continue
            if ident:
                seen.add(ident)
            all_matches.append(row)
        last_has_next = bool(payload.get('hasNextPage', False))
        if not last_has_next or not rows:
            break
    return all_matches, {'pages': pages, 'total': total,
                         'more_pages': last_has_next, 'source': 'Tennis API / RapidAPI'}


def get_recent_results(tour: str, start: str, end: str, max_pages: int = 1) -> tuple[list[dict],dict]:
    if tour not in ('atp','wta'):
        raise ValueError('Circuito no permitido')
    start_date = pd.to_datetime(start, format='%Y-%m-%d', errors='raise').date()
    end_date = pd.to_datetime(end, format='%Y-%m-%d', errors='raise').date()
    if end_date < start_date or (end_date-start_date).days > 14:
        raise ValueError('Rango de resultados fuera de límites')
    rows = []
    last_has_next = False
    pages = 0
    for page in range(1,min(max(1,max_pages),2)+1):
        payload=request(f'/{tour}/results/{start}/{end}',
                        {'filter':'PlayerGroup:singles','pageSize':500,'pageNo':page})
        chunk=payload.get('data',[])
        if not isinstance(chunk,list):
            raise RapidTennisError('La respuesta de resultados no es una lista.')
        rows.extend(c for c in chunk if isinstance(c,dict))
        pages=page
        last_has_next=bool(payload.get('hasNextPage',False))
        if not last_has_next or not chunk:
            break
    return rows, {'pages':pages,'more_pages':last_has_next}


def _object(value: Any) -> dict:
    return value if isinstance(value,dict) else {}


def _safe_odds(v: Any) -> float | None:
    try:
        n=float(v)
        return n if 1.0 < n < 1000 else None
    except (TypeError,ValueError):
        return None


def _surface(tourney: dict) -> str | None:
    court=tourney.get('court')
    val=_object(court).get('name') or (court if isinstance(court,str) else None) or tourney.get('surface')
    if val:
        n=str(val).lower().strip()
        if 'grass' in n:return 'Grass'
        if 'clay' in n:return 'Clay'
        if 'hard' in n or 'acrylic' in n:return 'Hard'
        if 'carpet' in n:return 'Carpet'
    try:
        cid=int(tourney.get('courtId'))
    except (ValueError,TypeError):
        cid=None
    return {1:'Hard',2:'Clay',3:'Hard',4:'Carpet',5:'Grass'}.get(cid)


def _circuit(tourney: dict, names: tuple[str,str], tour_field: str = '') -> str:
    """El rango 1 mezcla Challenger e ITF; mantener la ambigüedad explícita."""
    txt=(str(tourney.get('name',''))+' '+tour_field).lower()
    if 'itf' in txt or 'm15' in txt or 'w15' in txt or 'm25' in txt or 'w25' in txt:
        return 'ITF'
    if 'challenger' in txt:
        return 'Challenger'
    try:
        rank=int(tourney.get('rankId'))
    except (ValueError,TypeError):
        rank=None
    if rank==0:return 'ITF'
    if rank==1:return 'Challenger/ITF (sin distinguir)'
    if rank in (2,3,4,7):
        if 'wta' in tour_field.lower() or 'wta' in txt:return 'WTA'
        if 'atp' in tour_field.lower() or 'atp' in txt:return 'ATP'
        return 'Circuito superior (sin confirmar)'
    return 'Sin clasificar'


def parse_upcoming(matches: list[dict], *, retrieved_at: datetime | None = None) -> pd.DataFrame:
    retrieved_at = retrieved_at or datetime.now(timezone.utc)
    records=[]
    for row in matches:
        if not isinstance(row,dict):continue
        p1,p2=_object(row.get('player1')),_object(row.get('player2'))
        n1,n2=str(p1.get('name') or '').strip(),str(p2.get('name') or '').strip()
        if not n1 or not n2 or '/' in n1 or '/' in n2 or n1==n2:
            continue
        tourney=_object(row.get('tournament'))
        name=str(tourney.get('name') or '').strip() or 'Torneo sin nombre'
        start=pd.to_datetime(row.get('date') or row.get('startTime'),utc=True,errors='coerce')
        if pd.notna(start) and start.to_pydatetime() <= retrieved_at:
            continue
        line=_object(row.get('preMatchOdds'))
        # Tres campos oficiales documentados. Son precios observados, NO
        # probabilidades independientes ni cuotas con frescura confirmada.
        full=_object(row.get('odds'))
        quoted=[(_safe_odds(line.get('player1')),_safe_odds(line.get('player2')),'preMatchOdds'),
                (_safe_odds(full.get('k1')),_safe_odds(full.get('k2')),'odds.k1/k2'),
                (_safe_odds(p1.get('odd')),_safe_odds(p2.get('odd')),'player.odd')]
        valid_pairs=[(a,b,source) for a,b,source in quoted if a is not None and b is not None]
        if valid_pairs:
            o1,o2,odds_source=valid_pairs[0]
        else:
            o1,o2,odds_source=None,None,'Sin dos cuotas válidas'
        # No retroceder silenciosamente a la probabilidad de un modelo de pago.
        # El feed carece de un sello temporal fiable de la cuota: ninguna pasa
        # un control de frescura para ser considerada una señal confirmada.
        records.append({
            'id':str(row.get('matchId') or row.get('id') or ''),
            'Torneo':name,'Categoría':_circuit(tourney,(n1,n2),str(row.get('type') or row.get('tour') or row.get('tourType') or '')),
            'Superficie':_surface({'court':row.get('court'),'courtId':tourney.get('courtId')}) or _surface(tourney) or 'Sin confirmar',
            'Jugador A':n1,'Jugador B':n2,'ID A':str(p1.get('id') or ''),'ID B':str(p2.get('id') or ''),
            'Inicio UTC':start,
            'Cuota A':o1,'Cuota B':o2,'Origen cuota':odds_source,
            'H2H A':_object(row.get('h2hSummary')).get('player1Wins'),
            'H2H B':_object(row.get('h2hSummary')).get('player2Wins'),
            'Fuente':'Tennis API / RapidAPI',
            'Consulta UTC':pd.Timestamp(retrieved_at),
            'Hora de cotización':'No informada por el proveedor',
            'Validación cuota':'No validada por falta de timestamp',
            'rankId':tourney.get('rankId'),
        })
    cols=['id','Torneo','Categoría','Superficie','Jugador A','Jugador B','ID A','ID B',
          'Inicio UTC','Cuota A','Cuota B','Origen cuota','H2H A','H2H B','Fuente','Consulta UTC',
          'Hora de cotización','Validación cuota','rankId']
    df=pd.DataFrame(records,columns=cols)
    if not df.empty:
        df=df.drop_duplicates(subset=['id','Jugador A','Jugador B'])
        df=df.sort_values(['Inicio UTC','Torneo'],na_position='last').reset_index(drop=True)
    return df


def parse_recent_results(rows: list[dict], tour: str, *, now: datetime | None = None) -> pd.DataFrame:
    """Sólo resultados COMPLETADOS, nunca W/O/RET. No son datos de entrenamiento
    hasta comprobar que el periodo está completo, sin huecos ni duplicados.
    """
    now=pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tzinfo is None:now=now.tz_localize('UTC')
    out=[]
    for row in rows:
        if not isinstance(row,dict) or str(row.get('result_type','')).lower() not in ('completed','complete',''):
            continue
        score=str(row.get('result') or '').strip()
        if not score or any(q in score.lower() for q in ('ret','w/o','walkover','def')):
            continue
        p1,p2=_object(row.get('player1')),_object(row.get('player2'))
        w,l=str(p1.get('name') or '').strip(),str(p2.get('name') or '').strip()
        if not w or not l or w==l or '/' in w or '/' in l:continue
        ts=pd.to_datetime(row.get('date'),utc=True,errors='coerce')
        if pd.isna(ts) or ts>now:continue
        tournament=_object(row.get('tournament'))
        out.append({'match_id':str(row.get('matchId') or row.get('id') or ''),
            'date':ts,'tour':tour.upper(),'winner_name':w,'loser_name':l,
            'surface':_surface(tournament), 'score':score,
            'best_of':row.get('best_of') or 3, 'tournament':tournament.get('name'),
            'category':_circuit(tournament,(w,l),tour)})
    df=pd.DataFrame(out,columns=['match_id','date','tour','winner_name','loser_name','surface',
                                  'score','best_of','tournament','category'])
    if not df.empty:df=df.drop_duplicates(subset=['tour','match_id','winner_name','loser_name'])
    return df
