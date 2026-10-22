import os

import pandas as pd
import streamlit as st

from tennisedge import odds, rapid, store
from tennisedge.ui import get_predictor, reload_predictor, setup

setup('Datos y APIs', '🗃️')
P = get_predictor()

st.subheader('Estado')
recent = store.load_recent()
c1, c2, c3, c4 = st.columns(4)
c1.metric('Histórico (Sackmann) hasta', P.archive_last)
c2.metric('Resultados recientes', f'{len(recent):,}'.replace(',', '.'),
          f"hasta {pd.Timestamp(recent.date.max()).date()}" if not recent.empty else 'sin descargar', delta_color='off')
c3.metric('Ratings al día hasta', P.data_last_date())
c4.metric('Jugadores en la base', f'{len(P.engine.players):,}'.replace(',', '.'))

st.subheader('Claves de API (variables de entorno en Render)')
keys = pd.DataFrame([
    {'Variable': 'RAPIDAPI_TENNIS_KEY', 'Para qué': 'Partidos próximos + resultados diarios (obligatoria para estar al día)',
     'Plan gratis': '50 llamadas/día', 'Estado': '✅ configurada' if rapid.available() else '❌ falta'},
    {'Variable': 'THE_ODDS_API_KEY', 'Para qué': 'Cuotas de muchas casas (página 💹)',
     'Plan gratis': '500 créditos/mes', 'Estado': '✅ configurada' if odds.available_key() else '❌ falta'},
])
st.dataframe(keys, hide_index=True, width='stretch')
with st.expander('¿Dónde pongo las claves?'):
    st.markdown("""
1. Entra a tu servicio en **dashboard.render.com** → **Environment** → **Add Environment Variable**.
2. Agrega `RAPIDAPI_TENNIS_KEY` (y opcionalmente `THE_ODDS_API_KEY`) con su valor → **Save, rebuild and deploy**.
3. Para la actualización diaria automática, agrega la misma `RAPIDAPI_TENNIS_KEY` en GitHub:
   repositorio → **Settings → Secrets and variables → Actions → New repository secret**.

Nunca pegues las claves en el código ni en el chat.
""")

st.subheader('Actualizar resultados recientes')
st.markdown('Descarga los partidos jugados desde la última actualización (ATP y WTA, incluye Challenger/ITF) para '
            'que los ratings estén al día. Cada semana de datos cuesta ~2–6 llamadas.')
if not rapid.available():
    st.info('Configura RAPIDAPI_TENNIS_KEY para activar esta opción.')
else:
    budget = st.slider('Máximo de llamadas a usar ahora', 6, 45, 20, 1,
                       help='El plan gratis permite 50 al día. La página de partidos usa 2 cada 3 horas.')
    if st.button('⬇️ Actualizar ahora', type='primary'):
        logs = []
        with st.status('Descargando resultados…', expanded=True) as status:
            try:
                df, used = store.update(P.archive_last, budget=budget, to_repo=False, log=lambda m: (logs.append(m), st.write(m)))
                status.update(label=f'Listo: {used} llamadas usadas · {len(df)} resultados recientes en total', state='complete')
            except rapid.RapidError as exc:
                status.update(label=str(exc), state='error')
        reload_predictor()
        st.rerun()
    if rapid.Quota.remaining is not None:
        st.caption(f'Llamadas restantes hoy según RapidAPI: {rapid.Quota.remaining}')
    st.caption('Lo descargado aquí vive mientras Render no reinicie el servicio. Para guardarlo de forma permanente '
               'y que se actualice solo cada día, activa el flujo de GitHub Actions (ver README) o descarga el CSV '
               'y súbelo al repositorio como `data/recent_results.csv`.')
if not recent.empty:
    st.download_button('Descargar recent_results.csv', recent.to_csv(index=False).encode(), 'recent_results.csv', 'text/csv')

with st.expander('🔎 Diagnóstico de RapidAPI (usa 1 llamada)'):
    st.caption('Muestra qué campos trae la respuesta de resultados, para verificar que el lector los interpreta bien.')
    if rapid.available() and st.button('Probar endpoint de resultados'):
        try:
            end = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
            rows = rapid.get_results('atp', (end - pd.Timedelta(days=1)).date(), end.date(), max_pages=1)
            st.write(f'{len(rows)} partidos recibidos')
            st.code('\n'.join(rapid.raw_sample_keys(rows)) or '(vacío)')
            parsed = rapid.parse_results(rows, 'atp')
            st.dataframe(parsed[['date', 'tourney_name', 'level', 'surface', 'winner', 'loser', 'score', 'odds_w', 'odds_l']].head(15),
                         hide_index=True, width='stretch')
        except rapid.RapidError as exc:
            st.error(str(exc))

if not P.oos.empty:
    st.subheader('Últimos partidos aplicados')
    o = P.oos.tail(30).iloc[::-1].copy()
    o['date'] = pd.to_datetime(o['date']).dt.date
    st.dataframe(o[['date', 'tour', 'winner', 'loser', 'p_winner', 'odds_w', 'odds_l']].rename(columns={
        'date': 'Fecha', 'tour': 'Circ.', 'winner': 'Ganador', 'loser': 'Perdedor', 'p_winner': 'Prob. previa del ganador',
        'odds_w': 'Cuota ganador', 'odds_l': 'Cuota perdedor'}), hide_index=True, width='stretch',
        column_config={'Prob. previa del ganador': st.column_config.ProgressColumn(min_value=0, max_value=1, format='percent')})

st.subheader('Fuentes y licencias')
st.markdown(f"""
- **Histórico 2016–jun 2026**: {P.report['attribution']} Incluye estadísticas de saque por partido.
- **Resultados recientes y partidos próximos**: Tennis API ATP/WTA/ITF (RapidAPI · Matchstat/SportsAPI365), plan gratuito.
- **Cuotas de varias casas**: The Odds API, plan gratuito. Ninguna fuente es Betano Chile.
- Proyecto de **uso personal**, sin fines comerciales. No es asesoría de apuestas.
""")
