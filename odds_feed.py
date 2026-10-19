"""Cuotas de tenis desde The Odds API (the-odds-api.com).

API KEY via environment only. Read only, no Betano credentials/scraping.
This is a *bookmaker* data feed, not a validated betting recommendation.
"""
from __future__ import annotations
import os
from datetime import datetime, timezone
from statistics import median
from typing import Any
import requests
import pandas as pd

BASE = 'https://api.the-odds-api.com/v4'
VALID_MARKETS = {'h2h', 'spreads', 'totals'}
MARKET_NAME = {'h2h': 'Ganador del partido', 'spreads': 'Hándicap de juegos', 'totals': 'Total de juegos'}

class OddsFeedError(ValueError):
    pass

def available_key() -> bool:
    return bool(os.getenv('THE_ODDS_API_KEY', '').strip())

def _get(path: str, key: str, params: dict[str, Any] | None = None, timeout: int = 14) -> tuple[Any, dict[str, str]]:
    if not key or not key.strip():
        raise OddsFeedError('Falta THE_ODDS_API_KEY en las variables privadas de Render.')
    p={'apiKey':key, **(params or {})}
    try:
        response=requests.get(f'{BASE}{path}',params=p,timeout=timeout,headers={'User-Agent':'TennisEdgeIA/0.4 ResearchDashboard'})
    except requests.RequestException as exc:
        raise OddsFeedError('No se pudo conectar con la API de cuotas; inténtalo después.') from exc
    quota={k:response.headers.get(k, '?') for k in ('x-requests-remaining','x-requests-used','x-requests-last')}
    if response.status_code == 401:
        raise OddsFeedError('Clave API inválida. Revisa THE_ODDS_API_KEY en Render.')
    if response.status_code == 429:
        raise OddsFeedError('Se agotó el límite de consultas de la API; espera al reinicio de la cuota.')
    if response.status_code != 200:
        raise OddsFeedError(f'Proveedor de cuotas respondió {response.status_code}.')
    try:
        return response.json(),quota
    except (ValueError,TypeError) as exc:
        raise OddsFeedError('El proveedor devolvió una respuesta que no se puede leer.') from exc

def tennis_tournaments(key: str, *, all_sports: bool=False) -> list[dict]:
    sports,_=_get('/sports/',key, {'all':'true'} if all_sports else {})
    if not isinstance(sports,list):
        raise OddsFeedError('No se pudo leer la lista de torneos.')
    return sorted(({'key':str(s.get('key','')),'title':str(s.get('title','')),
                    'active':bool(s.get('active',False))} for s in sports
                   if str(s.get('key','')).startswith('tennis_') and not bool(s.get('has_outrights',False))),
                   key=lambda d:d['title'].lower())

def fetch_tournament(key:str, sport_key:str, *, region:str='eu', markets:tuple[str,...]=('h2h',)) -> tuple[list[dict], dict]:
    if not sport_key.startswith('tennis_') or '/' in sport_key or '?' in sport_key:
        raise OddsFeedError('Torneo no válido.')
    if region not in ('eu','uk','us','au'):
        raise OddsFeedError('Región no válida.')
    if not markets or any(m not in VALID_MARKETS for m in markets):
        raise OddsFeedError('Mercados no válidos.')
    data,quota=_get(f'/sports/{sport_key}/odds/',key,{'regions':region,'markets':','.join(markets),'oddsFormat':'decimal','dateFormat':'iso'})
    if not isinstance(data,list):
        raise OddsFeedError('No se recibió una lista de partidos.')
    return data,quota

def _as_date(value:Any):
    try:
        result=pd.to_datetime(value,utc=True,errors='coerce')
        if pd.isna(result):return pd.NaT
        return result
    except (TypeError,ValueError): return pd.NaT

def _num(v):
    try:
        x=float(v)
        return x if 1.0 < x < 1000 else None
    except (ValueError,TypeError):return None

def flatten_odds(events:list[dict], *, only_prematch:bool=True, now:datetime|None=None) -> pd.DataFrame:
    """One row per bookmaker/market/outcome, no invented prices or market lines.

    The API may omit game handicaps and totals for some bookmakers.
    """
    now=now or datetime.now(timezone.utc)
    stamp=pd.Timestamp(now)
    if stamp.tzinfo is None:stamp=stamp.tz_localize('UTC')
    else:stamp=stamp.tz_convert('UTC')
    records=[]
    for event in events:
        if not isinstance(event,dict):continue
        start=_as_date(event.get('commence_time'))
        if pd.isna(start):continue
        if only_prematch and start<=stamp:continue
        eid=str(event.get('id','')).strip()
        p1=str(event.get('home_team','')).strip();p2=str(event.get('away_team','')).strip()
        if not (eid and p1 and p2):continue
        for book in event.get('bookmakers',[]) or []:
            if not isinstance(book,dict):continue
            bn=str(book.get('title') or book.get('key') or 'sin casa')
            for market in book.get('markets',[]) or []:
                if not isinstance(market,dict):continue
                mk=str(market.get('key',''))
                if mk not in VALID_MARKETS:continue
                updated=_as_date(market.get('last_update') or book.get('last_update'))
                outcomes=market.get('outcomes') or []
                good=[]
                for outcome in outcomes:
                    if not isinstance(outcome,dict): continue
                    price=_num(outcome.get('price'))
                    name=str(outcome.get('name','')).strip()
                    point=outcome.get('point')
                    try:point=float(point) if point is not None else None
                    except (TypeError,ValueError):point=None
                    if price is not None and name:good.append((name,point,price))
                # Fair-price estimates only within one bookmaker's complete, mutually-exclusive two-way market.
                # For tennis all h2h/spreads/totals should have 2 opposing outcomes, not guaranteed.
                invsum=sum(1/p for n,pt,p in good) if len(good)==2 else 0
                for name,point,price in good:
                    delta=(stamp-updated).total_seconds()/60 if not pd.isna(updated) else None
                    records.append({'event_id':eid,'torneo':str(event.get('sport_title','')),
                       'inicio_utc':start,'jugador_a':p1,'jugador_b':p2,'mercado':MARKET_NAME[mk],
                       'market_key':mk,'seleccion':name,'linea':point,'casa':bn,
                       'cuota':price,'ultima_actualizacion':updated,'antiguedad_min':round(delta,1) if delta is not None else None,
                       'prob_impl_pct':round(100/price,2),'prob_sin_margen_pct':round(100*(1/price)/invsum,2) if invsum else None})
    columns=['event_id','torneo','inicio_utc','jugador_a','jugador_b','mercado','market_key','seleccion',
             'linea','casa','cuota','ultima_actualizacion','antiguedad_min','prob_impl_pct','prob_sin_margen_pct']
    return pd.DataFrame(records,columns=columns)

def summarize_prices(df:pd.DataFrame, *, max_age_minutes:int=120, min_books:int=1) -> pd.DataFrame:
    """No-vig consensus is a descriptive market average, not proof of an edge."""
    cols=['torneo','inicio_utc','partido','mercado','seleccion','linea','mejor_cuota','mejor_casa',
          'casas','cuota_mediana','consenso_sin_margen_pct','actualizacion_reciente']
    if df.empty:return pd.DataFrame(columns=cols)
    good=df.loc[(df.antiguedad_min.notna()) &(df.antiguedad_min.between(-5,max_age_minutes))].copy()
    if good.empty:return pd.DataFrame(columns=cols)
    rows=[]
    groupkeys=['event_id','market_key','seleccion','linea']
    # dropna=False is essential to retain h2h outcomes (without point).
    for _,sub in good.groupby(groupkeys,dropna=False):
        if sub.casa.nunique()<min_books:continue
        best=sub.sort_values('cuota',ascending=False).iloc[0]
        estimates=sub.prob_sin_margen_pct.dropna()
        rows.append({'torneo':best.torneo,'inicio_utc':best.inicio_utc,
                     'partido':best.jugador_a+' vs '+best.jugador_b,'mercado':best.mercado,
                     'seleccion':best.seleccion,'linea':best.linea,'mejor_cuota':float(best.cuota),
                     'mejor_casa':best.casa,'casas':int(sub.casa.nunique()),
                     'cuota_mediana':round(float(sub.cuota.median()),2),
                     'consenso_sin_margen_pct':round(float(estimates.median()),2) if len(estimates)>=3 else None,
                     'actualizacion_reciente':str(sub.ultima_actualizacion.max())})
    return pd.DataFrame(rows,columns=cols).sort_values(['inicio_utc','partido','mercado','seleccion']) if rows else pd.DataFrame(columns=cols)
