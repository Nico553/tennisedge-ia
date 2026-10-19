"""Adaptador opcional de calendario provisto por Sportradar bajo licencia Tennis v3.

No hay scraping. Requiere API key y contrato propio, configurados solo en el
servidor como SPORTRADAR_TENNIS_API_KEY. Nunca expone la clave en la UI.
"""
from __future__ import annotations
import os
import requests
import pandas as pd

BASE='https://api.sportradar.com/tennis'


def has_key():
    return bool(os.environ.get('SPORTRADAR_TENNIS_API_KEY','').strip())


def daily_summaries(date_str:str,access:str='trial',limit:int=150):
    if access not in ('trial','production'):
        raise ValueError('Modo de acceso no válido')
    pd.to_datetime(date_str,format='%Y-%m-%d',errors='raise')
    key=os.environ.get('SPORTRADAR_TENNIS_API_KEY','').strip()
    if not key:
        raise RuntimeError('Falta configurar API key Sportradar en Render')
    u=f'{BASE}/{access}/v3/en/schedules/{date_str}/summaries.json'
    r=requests.get(u,headers={'x-api-key':key,'Accept':'application/json'},
                   params={'start':0,'limit':min(200,max(1,int(limit)))},timeout=22)
    if r.status_code in (401,403):
        raise RuntimeError('API Sportradar: credenciales inválidas o acceso sin permiso')
    if r.status_code==429:
        raise RuntimeError('API Sportradar: cuota excedida. Espera antes de volver a consultar.')
    r.raise_for_status()
    return r.json()


def flatten_summaries(payload:dict)->pd.DataFrame:
    rows=[]
    for s in payload.get('summaries',[]):
        e=s.get('sport_event') or {}
        con=e.get('sport_event_context') or {}
        cat=con.get('category') or {}
        comp=con.get('competition') or {}
        competitors=e.get('competitors') or []
        names=[p.get('name','') for p in competitors if isinstance(p,dict)]
        if len(names)!=2:continue
        circuit=str(cat.get('name','')).upper()
        if 'ATP' not in circuit and 'WTA' not in circuit:
            continue
        cname=str(comp.get('name',''))
        if 'double' in cname.lower():continue
        rows.append({'Circuito':circuit,'Torneo':cname,'Jugador A':names[0],
                     'Jugador B':names[1],'Inicio UTC':e.get('start_time',''),
                     'Estado':(s.get('sport_event_status') or {}).get('status',''),
                     'Evento ID':e.get('id',''),'Fuente':'Sportradar licenciada'})
    return pd.DataFrame(rows)
