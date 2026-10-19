"""Escaneo multitorneo, selección prudente y simulaciones de combinadas.

Nunca aprueba apuestas con datos viejos ni afirma ROI sin backtest de cuotas.
"""
from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
from typing import Callable
import math
import pandas as pd

from odds_feed import flatten_odds
from value_engine import infer_tour, resolve_surface, scan_h2h

MIN_ODD = 1.20
MIN_EDGE_STRESS = 0.05  # Heurística de demostración, NO intervalo estadístico.
MIN_P_COMBO = 0.75
MAX_PER_CYCLE = 5
CYCLE_HOURS = 12
RESERVE_CREDITS = 60


def tournament_category(tournament: dict) -> str:
    text = (str(tournament.get('key',''))+' '+str(tournament.get('title',''))).casefold()
    if any(x in text for x in ('doubles','double','mixed')):return 'Excluido (dobles)'
    if 'challenger' in text:return 'Challenger'
    if 'itf' in text:return 'ITF'
    if 'utr' in text:return 'UTR'
    if 'wta' in text:return 'WTA'
    if 'atp' in text:return 'ATP'
    return 'Otro tenis'


def choose_tournaments(catalog: list[dict], slot: int, max_count: int=MAX_PER_CYCLE) -> list[dict]:
    """Lista deterministicamente rotativa y equilibrada entre ATP, WTA y otros.

    La API podría no listar Challenger/ITF; no fabricamos cobertura.
    """
    if max_count <= 0:return []
    active=[]
    for t in catalog:
        k=str(t.get('key',''))
        if t.get('active') and k.startswith('tennis_') and tournament_category(t)!='Excluido (dobles)':
            active.append(t)
    buckets:dict[str,list[dict]]={}
    for item in active:
        buckets.setdefault(tournament_category(item),[]).append(item)
    for key,items in buckets.items():
        items.sort(key=lambda t:t['key'])
        offset=slot % len(items)
        buckets[key]=items[offset:]+items[:offset]
    keys=[k for k in ('ATP','WTA','Challenger','ITF','UTR','Otro tenis') if buckets.get(k)]
    picked=[]
    while len(picked)<max_count and any(buckets.get(k) for k in keys):
        for key in keys:
            if len(picked)>=max_count:break
            if buckets[key]:picked.append(buckets[key].pop(0))
    return picked


def collect_tournament_prices(
    tournaments:list[dict], fetch: Callable[...,tuple[list[dict],dict]], *,
    now:datetime|None=None, region:str='eu', reserve:int=RESERVE_CREDITS,
) -> tuple[pd.DataFrame,list[dict],int|None]:
    """Una sola región y mercado h2h para minimizar créditos; registra cobertura real."""
    now=now or datetime.now(timezone.utc)
    rows=[]; audited=[]; remaining=None
    for t in tournaments:
        title=str(t['title']); key=str(t['key'])
        if remaining is not None and remaining <= reserve:
            audited.append({'Torneo':title,'Circuito':tournament_category(t),'Estado':'Omitido: reserva de créditos','Partidos':0})
            continue
        try:
            events,quota=fetch(key,region=region,markets=('h2h',))
            rem=quota.get('x-requests-remaining','?')
            if str(rem).isdigit():remaining=int(rem)
            raw=flatten_odds(events,only_prematch=True,now=now)
            count=int(raw.event_id.nunique()) if not raw.empty else 0
            audited.append({'Torneo':title,'Circuito':tournament_category(t),
                'Estado':'Cotizaciones consultadas' if count else 'Sin próximos partidos con cuotas',
                'Partidos':count})
            if not raw.empty:
                raw['sport_key']=key
                raw['nombre_torneo']=title
                rows.append(raw)
        except Exception as exc:
            # No mostrar secrets ni URL/params en errores del proveedor.
            name=type(exc).__name__
            audited.append({'Torneo':title,'Circuito':tournament_category(t),
                'Estado':f'Error de proveedor ({name})','Partidos':0})
    if rows:return pd.concat(rows,ignore_index=True),audited,remaining
    cols=list(flatten_odds([],now=now).columns)+['sport_key','nombre_torneo']
    return pd.DataFrame(columns=cols),audited,remaining


def evaluate_all(raw:pd.DataFrame, lab:object, *,now:datetime|None=None,
                 max_age_minutes:int=90)->tuple[pd.DataFrame,list[dict]]:
    """Separa AT P/WTA evaluables; abstiene cuando la identidad o pista es incierta."""
    cols=['Torneo','Circuito','Superficie','Inicio UTC','Partido','Selección','Casa','Cuota',
          'P modelo %','P implícita %','Cuota justa','EV teórico %','Discrepancia modelos pp',
          'Estado','Motivos','Actualizado UTC']
    if raw.empty:return pd.DataFrame(columns=cols),[]
    reports=[];tables=[]
    for key,part in raw.groupby('sport_key',sort=True):
        title=str(part['nombre_torneo'].iloc[0]);tour=infer_tour(key,title)
        surface=resolve_surface(title,key)
        report={'Torneo':title,'Circuito':tournament_category({'key':key,'title':title}),
                'Superficie':surface or 'No confirmada','Partidos':int(part.event_id.nunique())}
        if not tour:
            report['Evaluación']='Precios disponibles; sin modelo validado para este circuito'
        elif not surface:
            report['Evaluación']='Precios disponibles; superficie no verificada'
        else:
            report['Evaluación']='Evaluado con controles de datos y fiabilidad'
            frame=scan_h2h(part,lab,tour,surface,now=now,max_age_minutes=max_age_minutes)
            if not frame.empty:
                frame.insert(0,'Superficie',surface)
                frame.insert(0,'Circuito',tour)
                frame.insert(0,'Torneo',title)
                tables.append(frame)
        reports.append(report)
    if not tables:return pd.DataFrame(columns=cols),reports
    return pd.concat(tables,ignore_index=True)[cols],reports


def enrich_candidates(table:pd.DataFrame,min_odd:float=MIN_ODD,
                      *,min_stress_ev:float=MIN_EDGE_STRESS)->pd.DataFrame:
    """Descuento prudencial heurístico, nunca interpretado como IC formal.

    No admite ROI demostrado: 'Estado' heredado es siempre experimental.
    """
    copy=table.copy()
    copy['P prudente %']=pd.Series(dtype=float)
    copy['EV prudente %']=pd.Series(dtype=float)
    copy['Clasificación']=pd.Series(dtype=str)
    if copy.empty:return copy
    probabilities=pd.to_numeric(copy['P modelo %'],errors='coerce')/100
    odds=pd.to_numeric(copy['Cuota'],errors='coerce')
    divergence=pd.to_numeric(copy['Discrepancia modelos pp'],errors='coerce')/100
    # 3 puntos mínimos o la mitad del desacuerdo entre modelos: regla simple de estrés, no IC.
    haircut=divergence.mul(.5).clip(lower=.03)
    prudent=(probabilities-haircut).clip(lower=0,upper=1)
    copy['P prudente %']=(prudent*100).round(2)
    copy['EV prudente %']=((prudent*odds-1)*100).round(2)
    valid=(copy['Estado'].str.startswith('Candidato experimental') &
           odds.ge(min_odd) & copy['EV prudente %'].ge(min_stress_ev*100) &
           probabilities.gt(0) & divergence.notna())
    copy['Clasificación']='No aprobada'
    copy.loc[valid & probabilities.ge(.70),'Clasificación']='Alta probabilidad · experimental'
    copy.loc[valid & probabilities.lt(.70),'Clasificación']='Subestimado · experimental'
    # Esto NO convierte ninguna fila en recomendación: faltan ROI y revisión de lesiones.
    copy=copy.sort_values(['EV prudente %','P modelo %'],ascending=False,na_position='last')
    return copy.reset_index(drop=True)


def propose_combinations(table:pd.DataFrame,max_legs:int=2,max_shown:int=3)->pd.DataFrame:
    """Escenarios, no apuestas: independencia aproximada, y SIN aprobación real."""
    cols=['Selecciones','Cuota combinada','P conjunta aprox. %','EV aprox. %','Advertencia']
    if table.empty or max_legs<2:return pd.DataFrame(columns=cols)
    good=table.loc[(table['Clasificación']=='Alta probabilidad · experimental') &
          (pd.to_numeric(table['P modelo %'],errors='coerce')>=MIN_P_COMBO*100)].copy()
    if len(good)<2:return pd.DataFrame(columns=cols)
    good=good.sort_values(['EV prudente %','P modelo %'],ascending=False).head(16)
    combos=[]
    for count in range(2,min(max_legs,3)+1):
        for members in combinations(good.to_dict('records'),count):
            games=[(x['Inicio UTC'],x['Partido']) for x in members]
            if len(set(games))!=len(games):continue
            # El mismo tenista no puede ocupar dos selecciones en una combinada.
            players=[{x.strip().casefold() for x in m['Partido'].split(' vs ')} for m in members]
            if any(players[i]&players[j] for i in range(count) for j in range(i+1,count)):continue
            prices=[float(x['Cuota']) for x in members]
            probs=[float(x['P modelo %'])/100 for x in members]
            if any(not math.isfinite(x) or x<=0 for x in prices+probs):continue
            gross=math.prod(prices);p=math.prod(probs)
            # La independencia solo es una aproximación; no garantía de cuota justa.
            combos.append({'Selecciones':' + '.join(f"{x['Torneo']}: {x['Selección']} ({x['Cuota']:.2f})" for x in members),
                'Cuota combinada':round(gross,2),'P conjunta aprox. %':round(p*100,2),
                'EV aprox. %':round((p*gross-1)*100,2),
                'Advertencia':'Simulación sin validar: asume independencia; sin ROI histórico ni control de lesiones'})
    if not combos:return pd.DataFrame(columns=cols)
    return pd.DataFrame(combos,columns=cols).sort_values(['P conjunta aprox. %','EV aprox. %'],ascending=False).head(max_shown).reset_index(drop=True)
