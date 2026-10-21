"""Exploración multitorneo secundaria: sin pronósticos inventados.

TennisEdge IA v4.1: ATP/WTA/Challenger/ITF cuando el proveedor los entregue.
No presenta cotizaciones sin timestamp como señales validadas.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from zoneinfo import ZoneInfo
import os

import pandas as pd
import streamlit as st
from rapid_tennis import (available,get_recent_results,parse_recent_results,RapidTennisError)
from rapid_catalog import current_rapid_catalog

CHILE=ZoneInfo('America/Santiago')

@st.cache_data(ttl=12*3600,show_spinner='Comprobando resultados recientes ATP y WTA…')
def recent_cached(fingerprint:str,start:str,end:str):
    a,ma=get_recent_results('atp',start,end,max_pages=1)
    b,mb=get_recent_results('wta',start,end,max_pages=1)
    return a,ma,b,mb

st.title('🎾 Cobertura multitorneo · TennisEdge IA 4.1')
st.caption('Segunda fuente de partidos y estadísticas. Sin alertas, sin apuestas automáticas y sin confundir cuotas con pronósticos.')
st.markdown('La sección **Partidos y cuotas actuales** sigue disponible en el menú lateral con tu conexión original a The Odds API.')

if not available():
    st.info('La segunda fuente gratuita todavía no está conectada. TennisEdge IA sigue funcionando con The Odds API; no necesitas cambiar la clave anterior.')
    st.markdown('**Para activar esta cobertura adicional:**')
    st.markdown('1. Abre [Tennis API - ATP WTA ITF en RapidAPI](https://rapidapi.com/jjrm365-kIFr3Nx_odV/api/tennis-api-atp-wta-itf) y selecciona el producto **Tennis API - ATP WTA ITF**. Revisa que el plan elegido sea **FREE** y verifica sus condiciones antes de suscribirte.\n'
                '2. Copia la clave `X-RapidAPI-Key` sin publicarla.\n'
                '3. En [Render → Environment](https://dashboard.render.com/web/srv-dbb3g9mi0phs73bc97tg) agrega `RAPIDAPI_TENNIS_KEY` con el valor de tu clave y selecciona **Save, deploy**.\n'
                '4. Vuelve a abrir esta pantalla. La clave se mantiene privada en Render.')
    st.warning('En la documentación del proveedor figura un plan FREE de 50 solicitudes diarias. La disponibilidad y los límites pueden cambiar; compruébalos en tu cuenta.')
    st.page_link('live_app.py',label='Ir a cuotas y escáner actuales',icon='📡')
    st.stop()

fingerprint=sha256(os.environ['RAPIDAPI_TENNIS_KEY'].strip().encode()).hexdigest()[:16]
try:
    frame,meta,checked=current_rapid_catalog()
except RapidTennisError as exc:
    st.error(str(exc))
    st.page_link('live_app.py',label='Continuar con The Odds API',icon='📡')
    st.stop()
except Exception:
    st.error('No se pudo interpretar la respuesta del proveedor. No se inventan datos alternativos.')
    st.stop()

if frame.empty:
    st.info('La fuente no devolvió encuentros individuales programados en las páginas consultadas.')
    st.stop()

n_tournaments=frame[['Torneo','Categoría']].drop_duplicates().shape[0]
with_odds=frame[['Cuota A','Cuota B']].notna().all(axis=1).sum()
missing_date=int(frame['Inicio UTC'].isna().sum())
a,b,c=st.columns(3)
a.metric('Torneos identificados',n_tournaments)
b.metric('Partidos programados',len(frame))
c.metric('Partidos con dos cuotas',int(with_odds))
st.caption(f'Proveedor: Tennis API / RapidAPI · Consulta del catálogo: {checked.astimezone(CHILE):%d-%m %H:%M} hora Chile · '
           f'Páginas consultadas: {meta["pages"]} · Sin hora de inicio publicada: {missing_date}')
if meta.get('more_pages'):
    st.warning('Cobertura PARCIAL: quedan páginas no consultadas. Se limitó a dos páginas para proteger las 50 peticiones gratuitas por día.')

st.info('**Las cuotas de esta segunda fuente no tienen sello temporal fiable.** Se muestran para explorar mercados, no como señales de valor confirmadas. Las probabilidades de mercado tampoco son probabilidades de nuestros modelos.')

groups=frame.groupby('Categoría',dropna=False).agg(Torneos=('Torneo','nunique'),Partidos=('id','count')).reset_index()
st.subheader('Cobertura real por circuito')
st.dataframe(groups,hide_index=True,width='stretch')

st.subheader('Partidos programados en todas las competiciones')
view=frame.copy()
view['Hora Chile']=view['Inicio UTC'].apply(lambda x:x.tz_convert(CHILE).strftime('%d-%m %H:%M') if pd.notna(x) else 'Por definir')
view=view.rename(columns={'Jugador A':'Jugador A','Jugador B':'Jugador B'})
view['Prob. implícita A %']=(100/view['Cuota A']).round(1)
view['Prob. implícita B %']=(100/view['Cuota B']).round(1)
show=['Torneo','Categoría','Superficie','Hora Chile','Jugador A','Jugador B','Cuota A','Cuota B','Prob. implícita A %','Prob. implícita B %']
st.dataframe(view[show],hide_index=True,width='stretch')
st.caption('Las probabilidades implícitas son 1/cuota, incluyen margen de la casa y NO son una predicción independiente. Falta validar edad del precio.')
with st.expander('Auditoría de cuotas y procedencia'):
    st.dataframe(frame[['Torneo','Jugador A','Jugador B','Origen cuota','Hora de cotización','Validación cuota']],hide_index=True,width='stretch')

with st.expander('🗃️ Resultados recientes · diagnóstico opcional (usa 2 llamadas adicionales)'):
    st.caption('Puedes comprobar resultados terminados de la última semana sin modificar ni contaminar el modelo histórico. El archivo del modelo sigue mostrando su última fecha real.')
    if st.button('Consultar últimos 7 días ATP/WTA',key='rapid_results_fetch'):
        end=datetime.now(timezone.utc).date()-timedelta(days=1)
        start=end-timedelta(days=6)
        try:
            atp,ma,wta,mb=recent_cached(fingerprint,start.isoformat(),end.isoformat())
            aa=parse_recent_results(atp,'atp'); ww=parse_recent_results(wta,'wta')
            results=pd.concat([aa,ww],ignore_index=True)
            l,r=st.columns(2)
            l.metric('Resultados ATP',len(aa))
            r.metric('Resultados WTA',len(ww))
            if ma['more_pages'] or mb['more_pages']:
                st.warning('Resultados parciales: el proveedor tiene más páginas; no utilizar este lote directamente para entrenar.')
            if not results.empty:
                st.dataframe(results[['date','tour','tournament','winner_name','loser_name','surface','score','category']].sort_values('date',ascending=False),hide_index=True,width='stretch')
                st.caption('Datos observados del proveedor secundario. Requieren conciliación histórica completa, validación de licencias y control de duplicados para entrenamiento.')
            else:
                st.info('No hay resultados válidos en el lote consultado.')
        except (RapidTennisError,ValueError) as exc:
            st.warning(str(exc))
        except Exception:
            st.error('No se pudo preparar el informe de resultados recientes.')

st.divider()
st.subheader('¿Cuándo habrá señales de valor de estos torneos?')
st.write('Primero debemos **completar y validar el historial hasta fechas recientes** y confirmar la antigüedad de cada cuota. Para Challenger/ITF también necesitamos entrenamiento y test separados por circuito. Hasta entonces, esta nueva fuente amplía el catálogo informativo, pero no genera señales aprobadas.')
st.page_link('live_app.py',label='Ver escáner y cuotas con actualización horaria verificable',icon='📡')
