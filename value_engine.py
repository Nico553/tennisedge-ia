"""Análisis reproducible de cuotas prematch. No realiza apuestas ni inventa datos.

La señal conservadora exige historial reciente, superficie verificada, muestra
suficiente y acuerdo de modelos. Incluso así sigue siendo *experimental* hasta
que se valide la rentabilidad fuera de muestra con precios históricos legítimos.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import re
import unicodedata
from typing import Any
import pandas as pd

MODEL_NAMES=('Elo por superficie','Regresión logística','Random Forest','Gradient Boosting')

# Identificación deliberadamente conservadora: se comprueba el nombre del torneo,
# no se infiere una pista por su país o época del año.
_SURFACE_RULES = {
    'Hard': ('australian open','us open','indian wells','miami open','miami masters',
             'cincinnati','canadian open','toronto','montreal','shanghai masters',
             'paris masters','paris indoor','vienna','basel','beijing','tokyo',
             'stockholm','rotterdam','doha','dubai','acapulco','washington',
             'brisbane','auckland','adelaide','seoul','osaka','wuhan'),
    'Clay': ('french open','roland garros','monte carlo','madrid open','mutua madrid',
             'italian open','internazionali bnl','rome masters','rome open',
             'barcelona open','buenos aires','rio open','rio de janeiro',
             'hamburg open','estoril','bastad','umag','geneva','gstaad','bucharest'),
    'Grass': ('wimbledon','queens club','queen s club','cinch championships',
              'halle','s hertogenbosch','rosmalen','eastbourne','mallorca championships',
              'nottingham','bad homburg','berlin grass'),
}


def normalize_name(name: Any) -> str:
    value=unicodedata.normalize('NFKD',str(name).casefold())
    value=''.join(c for c in value if not unicodedata.combining(c))
    value=value.replace('ł','l').replace('ø','o').replace('đ','d').replace('ß','ss')
    value=re.sub(r'[^a-z0-9]+',' ',value)
    return ' '.join(value.split())


def resolve_surface(title:str, sport_key:str='') -> str|None:
    text=normalize_name(title+' '+sport_key)
    possibilities=[surf for surf,phrases in _SURFACE_RULES.items() if any(p in text for p in phrases)]
    return possibilities[0] if len(set(possibilities))==1 else None


def infer_tour(sport_key:str, title:str='') -> str|None:
    k=(sport_key+' '+title).casefold()
    # No incluir torneos de dobles, ITF o Challenger sin datos propios.
    if any(q in k for q in ('doubles','double','challenger','itf','mixed')):return None
    if 'wta' in k:return 'WTA'
    if 'atp' in k:return 'ATP'
    return None


def resolve_player(name:str, known_names:set[str]) -> tuple[str|None,str]:
    """Evita coincidencias ambiguas de apellidos; no usa fuzzy matching incierto."""
    needle=normalize_name(name)
    if not needle:return None,'Nombre vacío'
    exact=[candidate for candidate in known_names if normalize_name(candidate)==needle]
    if len(exact)==1:return exact[0],'Nombre exacto'
    if len(exact)>1:return None,'Nombre ambiguo'
    words=needle.split()
    # Los feeds a veces usan "J. Sinner", comparables sólo si el apellido es único.
    if len(words)>=2 and len(words[0])==1:
        matches=[c for c in known_names if (lambda tokens: len(tokens)>=2 and
                 tokens[0][0]==words[0] and tokens[-1]==words[-1])(normalize_name(c).split())]
        if len(matches)==1:return matches[0],'Inicial + apellido único'
    return None,'Jugador no identificado de forma inequívoca'


def model_data_quality(details:dict, latest_date:pd.Timestamp, *, today:datetime|None=None,
                       max_history_age_days:int=21, min_total:int=25, min_surface:int=8) -> list[str]:
    today=today or datetime.now(timezone.utc)
    ts=pd.Timestamp(latest_date)
    now=pd.Timestamp(today)
    if ts.tzinfo is None:ts=ts.tz_localize('UTC')
    else:ts=ts.tz_convert('UTC')
    if now.tzinfo is None:now=now.tz_localize('UTC')
    else:now=now.tz_convert('UTC')
    problems=[]
    delta=(now-ts).days
    if delta>max_history_age_days: problems.append(f'Historial desactualizado ({delta} días)')
    if delta<0:problems.append('Última fecha histórica futura')
    if min(details.get('matches_a',0),details.get('matches_b',0))<min_total: problems.append('Muestra general insuficiente')
    if min(details.get('surface_a',0),details.get('surface_b',0))<min_surface: problems.append('Pocos partidos en la superficie')
    return problems


def theoretical_value(prob:float, odd:float) -> dict:
    if not (0 < prob < 1 and 1 < odd < 1000):raise ValueError('Probabilidad o cuota fuera de rango')
    return {'p_modelo':round(prob,4), 'cuota':round(odd,3),
            'p_implicita':round(1/odd,4),'cuota_justa':round(1/prob,3),
            'ev_teorico':round(prob*odd-1,4),'ventaja_pp':round((prob-1/odd)*100,2)}


def scan_h2h(raw:pd.DataFrame, lab:Any, tour:str, surface:str|None,
             *, now:datetime|None=None,max_age_minutes:int=90) -> pd.DataFrame:
    """Agrupa precios por evento/selección, estima p SIN mezclar mercados ni líneas.

    Sin superficie/historial reciente/identidad inequívoca se abstiene.
    El estatus nunca afirma una ventaja económicamente demostrada, pues falta
    una evaluación histórica con las cuotas realmente disponibles entonces.
    """
    columns=['Inicio UTC','Partido','Selección','Casa','Cuota','P modelo %',
             'P implícita %','Cuota justa','EV teórico %','Discrepancia modelos pp',
             'Estado','Motivos','Actualizado UTC']
    if raw is None or raw.empty or tour not in ('ATP','WTA'):
        return pd.DataFrame(columns=columns)
    now=pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tzinfo is None:now=now.tz_localize('UTC')
    else:now=now.tz_convert('UTC')
    # Sólo ganador de partido, *prematch*, con precios relativamente recientes.
    frame=raw.loc[(raw.market_key=='h2h') & (raw.inicio_utc>now) &
        (raw.antiguedad_min.notna()) & (raw.antiguedad_min.between(-5,max_age_minutes))].copy()
    if frame.empty:return pd.DataFrame(columns=columns)
    names={name for tour_key,name in lab.states.keys() if tour_key==tour}
    memo={}; rows=[]
    for (event_id, selection),group in frame.groupby(['event_id','seleccion'],dropna=False):
        best=group.sort_values(['cuota','ultima_actualizacion'],ascending=[False,False]).iloc[0]
        label=f'{best.jugador_a} vs {best.jugador_b}'
        cause=[]
        pa,why_a=resolve_player(best.jugador_a,names)
        pb,why_b=resolve_player(best.jugador_b,names)
        ps,why_s=resolve_player(str(selection),names)
        if surface is None:cause.append('Superficie desconocida/no verificada')
        if not pa or not pb or pa==pb or not ps or ps not in (pa,pb):
            cause.append('Identidad de jugadores sin correspondencia inequívoca')
        probability=None; divergence=None
        if not cause:
            # El laboratorio ajusta la calibración usando jugadores en orden
            # alfabético; mantener esa misma orientación evita sesgos al
            # recibir el orden arbitrario que usan las casas de apuestas.
            first,second=sorted((pa,pb),key=str.casefold)
            key=(first,second,surface)
            try:
                if key not in memo:memo[key]=lab.predict(tour,first,second,surface,3)
                result=memo[key]
                probability=float(result['probabilities'][lab.selected])
                all_p=[float(result['probabilities'][m]) for m in MODEL_NAMES]
                if ps==second:
                    probability=1-probability
                    all_p=[1-p for p in all_p]
                divergence=(max(all_p)-min(all_p))*100
                cause.extend(model_data_quality(result['details'],lab.matches.date.max(),today=now.to_pydatetime()))
                if divergence>15:cause.append('Modelos discrepan más de 15 puntos porcentuales')
            except (ValueError,KeyError,AttributeError,TypeError) as exc:
                cause.append('Modelo no pudo evaluar este partido')
        if probability is None:
            values={'p_modelo':None,'p_implicita':round(1/float(best.cuota),4),
                    'cuota_justa':None,'ev_teorico':None}
        else:
            values=theoretical_value(probability,float(best.cuota))
            if values['ev_teorico']<=0:cause.append('Sin EV teórico positivo')
        # Un EV positivo NO implica que una apuesta esté demostrada ni recomendada.
        status=('Candidato experimental (sin ROI validado)' if not cause and probability is not None
                else 'No validado')
        rows.append({'Inicio UTC':str(best.inicio_utc),'Partido':label,'Selección':selection,
             'Casa':best.casa,'Cuota':round(float(best.cuota),3),
             'P modelo %':round(probability*100,2) if probability is not None else None,
             'P implícita %':round(100/float(best.cuota),2),
             'Cuota justa':values['cuota_justa'],
             'EV teórico %':round(values['ev_teorico']*100,2) if values['ev_teorico'] is not None else None,
             'Discrepancia modelos pp':round(divergence,2) if divergence is not None else None,
             'Estado':status,'Motivos':'; '.join(cause) if cause else 'Requiere backtest con cuotas históricas y verificación física',
             'Actualizado UTC':str(best.ultima_actualizacion)})
    df=pd.DataFrame(rows,columns=columns)
    return df.sort_values(['EV teórico %','Inicio UTC'],ascending=[False,True],na_position='last').reset_index(drop=True)
