"""API-Tennis live fixtures and scores (optional; private API_TENNIS_KEY in Render).
No bookmaker scraping; returns factual status, never makes up live scores.
"""
from __future__ import annotations
from datetime import date
from typing import Any
import os
import requests

API_URL = 'https://api.api-tennis.com/tennis/'

class TennisLiveError(ValueError):
    pass


def live_key_available() -> bool:
    return bool(os.getenv('API_TENNIS_KEY','').strip())


def _call(method: str, **extra) -> list[dict]:
    if method not in ('get_livescore', 'get_fixtures'):
        raise TennisLiveError('Operación API no permitida.')
    token = os.getenv('API_TENNIS_KEY','').strip()
    if not token:
        raise TennisLiveError('Falta API_TENNIS_KEY en Render.')
    params={'method':method,'APIkey':token,'timezone':'America/Santiago',**extra}
    try:
        response=requests.get(API_URL,params=params,timeout=18,
          headers={'User-Agent':'TennisEdgeIA/2.2 EducationalResearch'})
    except requests.RequestException as ex:
        raise TennisLiveError('No se pudo contactar el proveedor de marcadores.') from ex
    if response.status_code in (401,403):
        raise TennisLiveError('La clave de API-Tennis no tiene acceso a este recurso.')
    if response.status_code==429:
        raise TennisLiveError('Se alcanzó el límite de peticiones de API-Tennis.')
    if response.status_code!=200:
        raise TennisLiveError(f'API-Tennis respondió HTTP {response.status_code}.')
    try:
        data=response.json()
    except ValueError as ex:
        raise TennisLiveError('API-Tennis no devolvió JSON.') from ex
    if not isinstance(data,dict) or str(data.get('success')) not in ('1','True'):
        raise TennisLiveError('La consulta de tenis no se completó. Revisa créditos y plan de API-Tennis.')
    result=data.get('result',[])
    if isinstance(result,dict):
        result=list(result.values())
    if not isinstance(result,list):
        raise TennisLiveError('Formato de resultados inesperado.')
    return [r for r in result if isinstance(r,dict)]


def _rows(entries:list[dict], *, only_live:bool=False) -> list[dict]:
    rows=[]
    for m in entries:
        category=str(m.get('event_type_type') or '')
        if not ('singles' in category.lower() and ('atp' in category.lower() or 'wta' in category.lower())):
            continue
        a=str(m.get('event_first_player') or '').strip()
        b=str(m.get('event_second_player') or '').strip()
        if not a or not b:
            continue
        is_live=str(m.get('event_live','0'))=='1'
        if only_live and not is_live:
            continue
        games=str(m.get('event_game_result') or '').strip()
        sets=m.get('scores') or []
        score_sets=[]
        for x in sets if isinstance(sets,list) else []:
            if not isinstance(x,dict):continue
            first=str(x.get('score_first','')).strip();second=str(x.get('score_second','')).strip()
            if first and second:score_sets.append(f'{first}-{second}')
        rows.append({
          'event_id':str(m.get('event_key') or ''),
          'torneo':str(m.get('tournament_name') or ''),
          'circuito':'ATP' if 'atp' in category.lower() else 'WTA',
          'jugador_a':a,'jugador_b':b,
          'fecha':str(m.get('event_date') or ''),
          'hora_cl':str(m.get('event_time') or ''),
          'estado':str(m.get('event_status') or ('En juego' if is_live else 'Programado')),
          'en_vivo':is_live,'sets':' · '.join(score_sets) if score_sets else '—',
          'juegos': games if games and games!='-' else '—',
          'sacador': str(m.get('event_serve') or '—'),
        })
    return rows


def fetch_live() -> list[dict]:
    return _rows(_call('get_livescore'),only_live=True)


def fetch_fixtures(day:date) -> list[dict]:
    if not isinstance(day,date):
        raise TennisLiveError('Fecha inválida.')
    value=day.isoformat()
    return _rows(_call('get_fixtures',date_start=value,date_stop=value))
