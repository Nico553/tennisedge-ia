"""Pantalla de valor experimental; usa las mismas cotizaciones ya consultadas.

No se consultan APIs adicionales de cuotas ni se generan apuestas automáticas.
"""
from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
from data_sources import collect_years, ATTRIBUTION
from engine import normalize_matches
from model_lab import ChronologicalLab
from value_engine import scan_h2h, resolve_surface, infer_tour

@st.cache_resource(ttl=24*3600, show_spinner='Preparando modelos y calibración temporal; la primera ejecución puede tardar…')
def get_calibrated_model():
    years=(2023,2024,2025,2026)
    datasets, logs=collect_years(list(years),('ATP','WTA'))
    frames=[normalize_matches(df,tour=tour) for tour,df in datasets.items()]
    if not frames:
        raise ValueError('No fue posible descargar los históricos ATP/WTA autorizados.')
    hist=pd.concat(frames,ignore_index=True).sort_values('date').reset_index(drop=True)
    if hist.empty:raise ValueError('No existen partidos con datos válidos para entrenar.')
    return ChronologicalLab(hist).benchmark(),logs


def render_value_panel(raw:pd.DataFrame, sport_key:str, title:str, max_age:int=90):
    st.subheader('🎯 Escáner experimental de valor · ganador prematch')
    st.caption('Sin límites de cuota. Usa el mejor precio observado por selección, estimaciones calibradas y la misma consulta a The Odds API (no gasta otra petición).')
    st.warning('La fuente histórica disponible llega a mayo de 2026. Por ello el escáner puede mostrar EV teórico, pero **no debe aprobar apuestas actuales** hasta tener resultados recientes y validación económica.')
    tour=infer_tour(sport_key,title)
    surface=resolve_surface(title,sport_key)
    if not tour:
        st.info('No se pudo establecer ATP/WTA individuales de forma inequívoca para este torneo. Escáner desactivado.')
        return
    if not surface:
        st.info('No se conoce con seguridad la superficie de este torneo. Mostramos solo precios; no asignamos probabilidades ni fingimos que la superficie es conocida.')
        return
    st.caption(f'Circuito: {tour} · Superficie verificada por catálogo de torneos: {surface} · Precio en casas externas a Betano.')
    agree=st.checkbox('Utilizar archivo histórico Jeff Sackmann / Tennis Abstract (CC BY-NC-SA 4.0) únicamente para análisis personal no comercial, con atribución.',
        value=False,key='model_archive_consent')
    if not agree:
        st.info('Marca la casilla para preparar los modelos una sola vez. Las consultas posteriores reutilizan el modelo en caché.')
        return
    try:
        lab,log=get_calibrated_model()
        latest=lab.matches.date.max().date()
        st.caption(f'Historial: {len(lab.matches):,} partidos ATP/WTA · Último resultado: {latest} · Modelo seleccionado: {lab.selected} · {ATTRIBUTION}')
        st.caption('Brier en el test temporal separado (menor es mejor); estas métricas NO prueban rentabilidad con cuotas de mercado.')
        b=lab.scoreboard[['Modelo','N','Exactitud','Brier ↓','Log loss ↓']].copy()
        st.dataframe(b,hide_index=True,width='stretch')
        table=scan_h2h(raw,lab,tour,surface,now=datetime.now(timezone.utc),max_age_minutes=max_age)
        if table.empty:
            st.info('Sin precios prematch suficientemente recientes o identificables en el torneo seleccionado.')
            return
        positive=table.loc[table['EV teórico %'].notna() & (table['EV teórico %']>0)]
        st.metric('Discrepancias positivas teóricas (NO recomendaciones)',len(positive))
        st.dataframe(table,hide_index=True,width='stretch')
        st.download_button('Descargar auditoría del escáner (CSV)',table.to_csv(index=False).encode('utf-8-sig'),
            file_name='tennisedge_watchlist_experimental.csv',mime='text/csv')
        st.caption('Los modelos están calibrados con una ventana de validación y evaluados en test posterior, pero se exige además un backtest con cuotas históricas antes de etiquetar apuestas como rentables. No se predicen sets/hándicaps con un modelo de ganador.')
    except Exception:
        st.error('No se pudo preparar el escáner experimental. Comprueba conectividad del archivo y recursos de Render. Los precios siguen funcionando independientemente.')
