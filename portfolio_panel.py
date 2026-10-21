"""Inicio automático multitorneo · TennisEdge IA 4.0 (sin alertas).

Presupuesto controlado: 5 competiciones cada 12h (hasta ~300 créditos/30 días
si cada solicitud de la región EU/mercado h2h cuesta 1 crédito). No garantiza
cobertura total de todos los torneos con planes de cuota limitados.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import os
import time
import pandas as pd
import streamlit as st

from odds_feed import tennis_tournaments,fetch_tournament,available_key
from multitour_engine import (
    CYCLE_HOURS, MAX_PER_CYCLE, RESERVE_CREDITS, tournament_category,
    choose_tournaments,collect_tournament_prices,evaluate_all,
    enrich_candidates,propose_combinations,
)
from value_panel import get_calibrated_model
from rapid_tennis import available as rapid_available, RapidTennisError
from rapid_catalog import current_rapid_catalog


def key_hash()->str:
    return sha256(os.getenv('THE_ODDS_API_KEY','').encode()).hexdigest()[:16]


@st.cache_data(ttl=CYCLE_HOURS*3600,show_spinner='Revisando competiciones y cuotas en caché compartida…')
def snapshot_cached(slot:int,token_hash:str):
    """No serializa claves; las lee del entorno de Render solo durante el fetch."""
    key=os.getenv('THE_ODDS_API_KEY','').strip()
    catalog=tennis_tournaments(key)
    chosen=choose_tournaments(catalog,slot,MAX_PER_CYCLE)
    raw,coverage,remaining=collect_tournament_prices(
        chosen,lambda sport_key,**kwargs:fetch_tournament(key,sport_key,**kwargs),region='eu'
    )
    return catalog,chosen,raw,coverage,remaining,datetime.now(timezone.utc)


def render_portfolio_panel():
    st.subheader('🌐 Escáner automático · múltiples torneos')
    st.caption('Cobertura conjunta ATP, WTA y torneos inferiores cuando la fuente los informa. Solo The Odds API ofrece cuotas con hora de actualización para el escáner estadístico actual; RapidAPI amplía el catálogo, pero sus precios no se usan como señales confirmadas.')
    # La consulta de RapidAPI comparte la misma caché de 6h con su página propia.
    # No se envían solicitudes adicionales al pasar entre ambas páginas.
    st.markdown('#### 🌍 Torneos disponibles en la segunda fuente')
    if not rapid_available():
        st.info('RapidAPI aún no está configurada. Abre «Multitorneo: segunda fuente» para ver los pasos; el escáner de cuotas original sigue disponible.')
    else:
        try:
            rapid_frame, rapid_meta, rapid_checked = current_rapid_catalog()
        except RapidTennisError as exc:
            st.warning(f'Segunda fuente: {exc}. El análisis con The Odds API permanece independiente.')
        except Exception:
            st.warning('No fue posible leer el catálogo de RapidAPI. El escáner principal continúa con The Odds API.')
        else:
            if rapid_frame.empty:
                st.info('RapidAPI no publicó partidos individuales futuros en las páginas consultadas. No se simulan torneos.')
            else:
                rt = int(rapid_frame[['Torneo', 'Categoría']].drop_duplicates().shape[0])
                rmatch = len(rapid_frame)
                rquoted = int(rapid_frame[['Cuota A', 'Cuota B']].notna().all(axis=1).sum())
                u,v,x = st.columns(3)
                u.metric('Torneos de RapidAPI', rt)
                v.metric('Partidos programados RapidAPI', rmatch)
                x.metric('Partidos con precios sin validar', rquoted)
                st.caption('Datos de RapidAPI consultados el '+rapid_checked.strftime('%d-%m %H:%M')+' UTC. Estos números no representan señales evaluadas por el modelo.')
                if rapid_meta.get('more_pages'):
                    st.warning('Catálogo incompleto: hay más páginas en el proveedor. Se consultaron solo dos para respetar el plan gratuito.')
                groups=(rapid_frame.groupby('Categoría',dropna=False)
                    .agg(Torneos=('Torneo','nunique'),Partidos=('Torneo','size')).reset_index())
                st.dataframe(groups,hide_index=True,width='stretch')
                with st.expander('📅 Ver partidos y torneos adicionales (sin señales aprobadas)'):
                    show=rapid_frame[['Torneo','Categoría','Superficie','Inicio UTC','Jugador A','Jugador B','Cuota A','Cuota B']].copy()
                    show=show.rename(columns={'Inicio UTC':'Inicio (UTC)', 'Cuota A':'Precio A sin validar', 'Cuota B':'Precio B sin validar'})
                    st.dataframe(show,hide_index=True,width='stretch')
                    st.caption('La API secundaria no garantiza el timestamp de sus precios. La ausencia de hora de cotización impide calcular señales de valor fiables con ellos.')
    st.divider()
    st.markdown('#### 💹 Mercados de cuotas verificables · The Odds API')
    if not available_key():
        st.error('Falta THE_ODDS_API_KEY en Environment de Render.')
        return
    slot=int(time.time()//(CYCLE_HOURS*3600))
    try:
        catalog,selected,raw,coverage,remaining,checked_at=snapshot_cached(slot,key_hash())
    except Exception:
        st.warning('No se pudo obtener el catálogo de cuotas. Revisa tu saldo y conexión con el proveedor.')
        return
    consulted=sum(c['Estado'].startswith(('Cotizaciones consultadas','Sin próximos')) for c in coverage)
    quoted=int(raw.event_id.nunique()) if not raw.empty else 0
    a,b,c=st.columns(3)
    eligible_count=sum(tournament_category(t)!='Excluido (dobles)' and t.get('active') for t in catalog)
    a.metric('Torneos activos The Odds API',eligible_count)
    b.metric('Consultadas en este ciclo',consulted)
    c.metric('Partidos con cuotas',quoted)
    st.caption(f"Última consulta real: {checked_at.strftime('%d-%m %H:%M')} UTC · Región: EU · Créditos restantes reportados: {remaining if remaining is not None else 'no disponible'} · Reserva: {RESERVE_CREDITS} créditos.")
    if eligible_count>len(selected):
        st.info(f'The Odds API ofreció {eligible_count} torneos activos; consultamos hasta {len(selected)} en este ciclo. Los otros torneos de RapidAPI aparecen arriba, pero todavía no generan señales validadas.')
    if coverage:
        with st.expander('📋 Competiciones realmente consultadas y estado'):
            st.dataframe(pd.DataFrame(coverage),hide_index=True,width='stretch')
    if raw.empty:
        st.info('En las competiciones consultadas no encontramos próximos partidos con precios disponibles. No inventamos señales ni precios.')
        return
    st.markdown('**Exploración y observación experimental**')
    st.caption('Ganador del encuentro (prematch). Se necesitarán modelos independientes y datos específicos para sets, juegos y hándicaps.')
    # No usar información cuyo uso no comercial no haya sido expresamente aceptado.
    consent=st.checkbox('Autorizo el uso personal no comercial del archivo Jeff Sackmann/Tennis Abstract (CC BY-NC-SA 4.0) con atribución.',key='global_archive_consent')
    if not consent:
        st.info('Las cuotas y la cobertura ya están consultadas. Marca la casilla de autorización para ejecutar la comparación estadística; no necesitas ajustar modelos ni seleccionar partidos.')
        with st.expander('Ver partidos y cuotas consultadas'):
            prices=raw.loc[raw.market_key.eq('h2h'),['nombre_torneo','inicio_utc','jugador_a','jugador_b','seleccion','cuota','casa']]
            st.dataframe(prices.sort_values('inicio_utc'),hide_index=True,width='stretch')
        return
    try:
        with st.spinner('Preparando el modelo y evaluando partidos; primer cálculo puede tardar…'):
            lab,logs=get_calibrated_model()
            evaluated,analysis=evaluate_all(raw,lab,now=datetime.now(timezone.utc),max_age_minutes=90)
    except Exception:
        st.error('No fue posible completar el cálculo estadístico. Las cuotas siguen disponibles; revisa la conectividad del archivo histórico y la memoria de Render.')
        return
    last=pd.Timestamp(lab.matches.date.max()).date()
    st.caption(f'Historial utilizado: hasta {last}. Modelo temporal ganador: {lab.selected}.')
    if (datetime.now(timezone.utc).date()-last).days>21:
        st.warning('**NINGUNA SEÑAL ESTÁ APROBADA**: el archivo histórico disponible está desactualizado. Los cálculos sirven para investigación; no deben usarse como recomendaciones actuales.')
    with st.expander('Cobertura del modelo por torneo'):
        st.dataframe(pd.DataFrame(analysis),hide_index=True,width='stretch')
    if evaluated.empty:
        st.info('Ningún partido pudo evaluarse: quizá falta la superficie verificada, los nombres no coinciden o el circuito carece de un modelo específico.')
        return
    ranking=enrich_candidates(evaluated)
    demo=ranking[ranking['Clasificación'].ne('No aprobada')]
    approved=0   # Sin verificación física y backtest de odds, cero aprobadas por definición.
    x,y,z=st.columns(3)
    x.metric('Partidos evaluados',ranking['Partido'].nunique())
    y.metric('Hipótesis con filtros superados',len(demo))
    z.metric('Señales validadas',approved)
    st.caption('La penalización prudencial usa el mayor entre 3 puntos porcentuales y media divergencia entre modelos. Es un test de estrés HEURÍSTICO, no un intervalo de confianza estadístico ni garantía de acierto.')
    st.markdown('**Posibles discrepancias para investigar (NO apuestas aprobadas)**')
    display_cols=['Torneo','Circuito','Partido','Selección','Cuota','P modelo %','P prudente %','EV teórico %','EV prudente %','Clasificación','Motivos']
    if demo.empty:
        st.info('No hay señales que superen todos los filtros estadísticos básicos. Esto es normal cuando los históricos están desactualizados o la cobertura es limitada.')
    else:
        st.dataframe(demo[display_cols],hide_index=True,width='stretch')
    with st.expander('📊 Auditoría completa: evaluadas y descartadas'):
        st.dataframe(ranking[display_cols+['Discrepancia modelos pp']],hide_index=True,width='stretch')
        st.download_button('Exportar auditoría CSV',ranking.to_csv(index=False).encode('utf-8-sig'),file_name='tennisedge_auditoria_multitorneo.csv',mime='text/csv')
    st.markdown('**🧩 Combinadas hipotéticas (máximo dos selecciones)**')
    combos=propose_combinations(demo,max_legs=2)
    if combos.empty:
        st.info('No hay dos selecciones distintas con requisitos suficientes para siquiera simular una combinada. No se añaden partidos para elevar la cuota.')
    else:
        st.dataframe(combos,hide_index=True,width='stretch')
        st.warning('Simulaciones NO recomendadas: la probabilidad conjunta presupone independencia y aún no se han verificado lesiones, dependencias ni ROI histórico.')
    st.caption('La cuota mínima de análisis es 1,20. No existe cuota máxima. Combinadas e individuales no son enviadas automáticamente a ninguna casa de apuestas.')
