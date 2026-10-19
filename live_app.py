"""TennisEdge IA 2.2 · Current matches and odds with source-labelled timestamps.

The Odds API: bookmaker prices, upcoming/in-play availability, not confirmed scores.
API-Tennis (optional): live score, serve, games and sets. Never fake missing data.
Quota-minded refresh: auto runs only while page is open, once per 30 minutes.
"""
from __future__ import annotations
from datetime import datetime, date, timezone
from hashlib import sha256
from zoneinfo import ZoneInfo
import os
import pandas as pd
import streamlit as st

from odds_feed import OddsFeedError,tennis_tournaments,fetch_tournament,flatten_odds,summarize_prices,available_key
from live_scores import TennisLiveError,live_key_available,fetch_live,fetch_fixtures
from value_panel import render_value_panel
from portfolio_panel import render_portfolio_panel

CHILE=ZoneInfo('America/Santiago')

def key_fingerprint(name:str)->str:
    return sha256(os.getenv(name,'').encode()).hexdigest()[:12]

@st.cache_data(ttl=7200,show_spinner=False)
def tournaments_cached(token_hash:str):
    return tennis_tournaments(os.getenv('THE_ODDS_API_KEY','').strip())

@st.cache_data(ttl=1800,show_spinner=False)
def odds_cached(sport_key:str,region:str,markets:tuple[str,...],token_hash:str):
    return fetch_tournament(os.getenv('THE_ODDS_API_KEY','').strip(),sport_key,region=region,markets=markets)

@st.cache_data(ttl=1200,show_spinner=False)
def scores_cached(token_hash:str):
    return fetch_live()

@st.cache_data(ttl=7200,show_spinner=False)
def fixtures_cached(day:date,token_hash:str):
    return fetch_fixtures(day)

st.markdown('''<style>
.stApp{background:#0c1323;color:#eef4ff}
[data-testid="stSidebar"]{background-color:#111a30}
[data-testid="stMetric"]{border:1px solid #2a3756;border-radius:13px;padding:14px;background:#151f35}
[data-testid="stMetricValue"]{color:#d6fb68}
.stButton>button[kind="primary"]{background:#d8fb69;color:#132018}
</style>''',unsafe_allow_html=True)

st.caption('TennisEdge IA 4.0 · ESCÁNER GLOBAL DE TENIS · CHILE')
st.title('🎾 TennisEdge · Señales multitorneo')
st.write('Revisa competiciones ATP/WTA y otros torneos disponibles, busca discrepancias entre cuotas y probabilidades históricas y descarta señales poco respaldadas. **Sin notificaciones y sin apuestas automáticas**.')
st.caption('Los precios son de casas externas, **no son cuotas de Betano Chile**. Datos actualizados según la hora que informe cada proveedor.')

if not available_key():
    st.error('Falta THE_ODDS_API_KEY en Render. Abre Environment y guarda la clave de The Odds API como variable privada.')
    st.markdown('[Administrar Render](https://dashboard.render.com/web/srv-dbb3g9mi0phs73bc97tg)')
else:
    render_portfolio_panel()
    st.divider()
    @st.fragment(run_every='30m')
    def current_odds_panel():
        st.subheader('💹 Cuotas automáticas · ATP / WTA')
        st.caption('Actualización automática mientras la página esté abierta: como máximo una consulta nueva cada 30 minutos por torneo, región y mercado. Esta frecuencia conserva los créditos del plan gratuito; no es un flujo segundo a segundo.')
        try:
            tournaments=[x for x in tournaments_cached(key_fingerprint('THE_ODDS_API_KEY')) if x['active']]
            if not tournaments:
                st.info('El proveedor no muestra torneos activos en esta consulta. No significa que no haya tenis hoy. Revisa la sección de marcadores más abajo.')
                return
            choice={f"{x['title']}":x['key'] for x in tournaments}
            col_a,col_b=st.columns([2.2,1])
            label=col_a.selectbox('Torneo con cuotas',list(choice.keys()),key='live_tournament')
            region=col_b.selectbox('Región de casas',['eu','uk','us','au'],key='live_region')
            extended=st.toggle('Consultar también hándicaps y totales (consume más créditos)',value=False,key='live_extended')
            mk=('h2h','spreads','totals') if extended else ('h2h',)
            events,quota=odds_cached(choice[label],region,mk,key_fingerprint('THE_ODDS_API_KEY'))
            raw=flatten_odds(events,only_prematch=False)
            st.caption(f"Créditos restantes: {quota.get('x-requests-remaining','?')} · Coste última consulta: {quota.get('x-requests-last','?')} · La caché puede retrasar cambios hasta 30 min.")
            if raw.empty:
                st.warning('No hay cuotas recientes para este torneo y región. Intenta otra región o torneo. No se muestran precios inventados.')
                return
            x,y,z=st.columns(3)
            x.metric('Partidos encontrados',raw.event_id.nunique())
            y.metric('Casas encontradas',raw.casa.nunique())
            z.metric('Precios disponibles',len(raw))
            maxage=st.select_slider('Antigüedad máxima de cada cuota',options=[15,30,60,90,120,180],value=90,format_func=lambda v:f'{v} min')
            summary=summarize_prices(raw,max_age_minutes=maxage,min_books=1)
            if summary.empty:
                st.info('No hay cuotas suficientemente frescas. Amplía la antigüedad o prueba otra región.')
                return
            chosen=st.segmented_control('Estado del partido',options=['Todos','Por comenzar','Ya comenzó (sin confirmar marcador)'],default='Todos')
            now=pd.Timestamp.now(tz='UTC')
            summary['estado']=summary['inicio_utc'].apply(lambda d:'Por comenzar' if d>now else 'Ya comenzó (sin confirmar marcador)')
            if chosen != 'Todos':summary=summary[summary.estado==chosen]
            if summary.empty:
                st.info('No hay partidos en ese estado según la hora programada del proveedor.')
                return
            summ=summary.copy()
            summ['Hora Chile']=summ.inicio_utc.dt.tz_convert('America/Santiago').dt.strftime('%d-%m %H:%M')
            summ['Línea']=summ.linea.map(lambda n:'' if pd.isna(n) else f'{n:+g}')
            summ['Estado por horario']=summ.estado
            summ=summ.rename(columns={'partido':'Partido','mercado':'Mercado','seleccion':'Selección',
                'mejor_cuota':'Mejor cuota','mejor_casa':'Casa','casas':'N.º casas',
                'consenso_sin_margen_pct':'Consenso sin margen %','actualizacion_reciente':'Hora de cotización UTC'})
            cols=['Hora Chile','Partido','Estado por horario','Mercado','Selección','Línea','Mejor cuota','Casa','N.º casas','Hora de cotización UTC']
            render_value_panel(raw,choice[label],label,max_age=maxage)
            st.markdown('##### Todas las cuotas observadas (sin restricción de cuotas)')
            st.dataframe(summ[cols],hide_index=True,width='stretch')
            st.download_button('Descargar precios reales (CSV)',summ.to_csv(index=False).encode('utf-8-sig'),
                file_name='tennisedge_cuotas_actuales.csv',mime='text/csv')
            st.caption('La columna «Ya comenzó» se infiere del horario, NO confirma que el partido siga en juego. Para eso se requiere un proveedor de marcadores.')
        except OddsFeedError as e:
            st.error(str(e))
        except Exception:
            st.error('No se pudieron cargar las cuotas. Comprueba la clave y la cobertura del torneo en The Odds API.')
    with st.expander('🔎 Explorar un torneo manualmente (opcional; consume créditos adicionales)'):
        if st.toggle('Activar consulta individual',value=False,key='single_tournament_opt_in'):
            current_odds_panel()

st.divider()
st.subheader('🔴 Marcadores y estadísticas en juego')
if live_key_available():
    @st.fragment(run_every='20m')
    def livescore_panel():
        st.caption('Proveedor independiente: API-Tennis. Consulta en caché durante 20 min para respetar los límites del plan gratuito. No es un marcador punto a punto instantáneo.')
        try:
            matches=scores_cached(key_fingerprint('API_TENNIS_KEY'))
            now=datetime.now(CHILE).strftime('%d-%m-%Y %H:%M')
            if matches:
                frame=pd.DataFrame(matches)
                frame['Partido']=frame.jugador_a+' vs '+frame.jugador_b
                frame=frame.rename(columns={'circuito':'Circuito','torneo':'Torneo','estado':'Set actual','sets':'Marcador sets','juegos':'Juegos','sacador':'Saca'})
                st.dataframe(frame[['Circuito','Torneo','Partido','Set actual','Marcador sets','Juegos','Saca']],hide_index=True,width='stretch')
                st.success(f'{len(frame)} partidos en juego identificados por el proveedor · Consultado {now} hora Chile')
            else:
                st.info(f'La API no devolvió partidos ATP/WTA individuales en juego · Consultado {now} hora Chile.')
        except TennisLiveError as e:
            st.error(str(e))
        except Exception:
            st.error('No fue posible consultar el marcador en directo. Revisa la autorización del proveedor.')
    livescore_panel()
    with st.expander('📅 Próximos partidos del calendario de tenis (API-Tennis)'):
        try:
            d=st.date_input('Día en Chile',value=datetime.now(CHILE).date(),key='fixture_day')
            fixtures=fixtures_cached(d,key_fingerprint('API_TENNIS_KEY'))
            if fixtures:
                f=pd.DataFrame(fixtures)
                f['Partido']=f.jugador_a+' vs '+f.jugador_b
                f=f.rename(columns={'circuito':'Circuito','torneo':'Torneo','hora_cl':'Hora Chile','estado':'Estado'})
                st.dataframe(f[['Circuito','Torneo','Partido','Hora Chile','Estado']],hide_index=True,width='stretch')
            else:st.info('No hay partidos en este segmento del calendario.')
        except TennisLiveError as e:st.warning(str(e))
else:
    st.info('Para mostrar sets, puntos, saques y marcadores en juego sin depender del horario, necesitamos una **segunda clave API**, distinta de The Odds API. La app ya tiene preparado el conector a API-Tennis.')
    st.markdown('[Consultar proveedor API-Tennis](https://api-tennis.com/) · [Configurar variable privada en Render](https://dashboard.render.com/web/srv-dbb3g9mi0phs73bc97tg)')
    st.code('API_TENNIS_KEY',language='text')
    st.caption('El plan gratuito de API-Tennis es limitado. Para segundos/puntos en vivo y actualizaciones frecuentes haría falta un plan con más solicitudes o streaming autorizado.')

st.divider()
with st.expander('🧠 Modelos históricos y predicciones experimentales'):
    st.write('El laboratorio permite estudiar Elo, regresión logística, Random Forest y Gradient Boosting con particiones cronológicas. El escáner está arriba y se activa al aceptar el uso no comercial del archivo histórico.')
    st.warning('El conjunto histórico actual termina en mayo de 2026. No debemos convertir sus probabilidades en recomendaciones de partidos de octubre sin incorporar resultados recientes, calibrar y validar de nuevo.')

st.caption('TennisEdge IA es una herramienta de investigación. No ofrece cuotas verificadas de Betano, no realiza apuestas y no garantiza resultados.')
