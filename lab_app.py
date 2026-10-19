"""TennisEdge IA — V0.3. Investigación cuantitativa de tenis, NO sistema de apuestas."""
from __future__ import annotations
from datetime import date
import os
import base64
from zoneinfo import ZoneInfo
from io import BytesIO
import pandas as pd
import numpy as np
import streamlit as st

from engine import normalize_matches,DataError
from model_lab import ChronologicalLab
from data_sources import collect_years,ATTRIBUTION,ORIGINAL
from official_feed import daily_summaries,flatten_summaries,has_key
from odds_feed import OddsFeedError, available_key, tennis_tournaments, fetch_tournament, flatten_odds, summarize_prices
import streamlit.components.v1 as components

st.markdown('''<style>
.stApp{background-color:#0c1323;color:#eef4ff}
[data-testid="stSidebar"]{background-color:#111a30}
[data-testid="stMetric"]{border:1px solid #2a3756;border-radius:13px;padding:13px;background:#151f35}
[data-testid="stMetricValue"]{color:#d6fb68}
h1,h2{letter-spacing:-.03em}
.stButton>button[kind="primary"]{background:#d8fb69;color:#132018}
</style>''',unsafe_allow_html=True)

@st.cache_data(ttl=12*3600,show_spinner=False)
def get_archive(years:tuple[int,...],tours:tuple[str,...]):
    return collect_years(list(years),tours)

@st.cache_data(show_spinner=False)
def convert_csv(content:bytes,tour:str):
    return normalize_matches(pd.read_csv(BytesIO(content),low_memory=False),tour)

@st.cache_resource(show_spinner='Entrenando y validando modelos con fechas históricas…')
def fit_lab(raw_atp:bytes|None,raw_wta:bytes|None,archive_years:tuple[int,...],
            selected_tours:tuple[str,...],source:str):
    frames=[];logs=[]
    if source=='Archivo histórico (no comercial)':
        dfs,logs=get_archive(archive_years,selected_tours)
        for tour,df in dfs.items():
            frames.append(normalize_matches(df,tour=tour))
    else:
        for tour,data in [('ATP',raw_atp),('WTA',raw_wta)]:
            if data:
                frames.append(convert_csv(data,tour))
    if not frames:
        raise DataError('No se cargaron partidos válidos. Revisa conexión, permisos o archivos.')
    historic=pd.concat(frames,ignore_index=True).sort_values('date').reset_index(drop=True)
    lab=ChronologicalLab(historic).benchmark()
    return lab,logs

st.sidebar.title('🎾 TennisEdge IA')
st.sidebar.caption('V0.3 · Históricos reales + evaluación cronológica')
source=st.sidebar.radio('Origen de datos',['Archivo histórico (no comercial)','CSV propio autorizado'])
atp_bytes=wta_bytes=None;years=tuple();tours=tuple()
if source=='Archivo histórico (no comercial)':
    st.sidebar.info('Datos históricos de Jeff Sackmann / Tennis Abstract. Archivo externo de junio de 2026; **NO es feed ATP/WTA oficial ni en vivo**.')
    start=st.sidebar.selectbox('Primer año histórico',list(range(2022,2014,-1)),index=2)
    end=st.sidebar.selectbox('Último año',[2026,2025,2024],index=0)
    chosen=st.sidebar.multiselect('Circuitos',['ATP','WTA'],default=['ATP','WTA'])
    accept=st.sidebar.checkbox('Entiendo que la fuente es CC BY-NC-SA 4.0: uso personal no comercial con atribución.',value=False)
    years=tuple(range(start,end+1)) if start<=end else tuple()
    tours=tuple(chosen)
    if st.sidebar.button('📥 Cargar y analizar datos reales',type='primary'):
        if not accept or not years or not tours:
            st.sidebar.error('Confirma la licencia, años y circuito antes de cargar.')
        else:
            st.session_state['ready']=True
            st.session_state['config']=(years,tours)
    if st.session_state.get('config')!=(years,tours):
        st.session_state['ready']=False
else:
    upa=st.sidebar.file_uploader('Histórico ATP .csv',type='csv',key='atp')
    upw=st.sidebar.file_uploader('Histórico WTA .csv',type='csv',key='wta')
    atp_bytes=upa.getvalue() if upa else None
    wta_bytes=upw.getvalue() if upw else None
    st.sidebar.caption('Solo importa históricos que tengas autorización para usar en entrenamiento de modelos.')
st.sidebar.markdown('---')
st.sidebar.warning('Sin pronósticos seguros. Sin acceso automatizado a Betano. No efectúa apuestas.')

st.markdown('##### LABORATORIO HISTÓRICO · ATP Y WTA')
st.title('Laboratorio de modelos 🧠')
st.write('**Modelos cuantitativos, precios observados y control estricto del riesgo.**')
st.caption('Las probabilidades se basan en históricos y NO detectan por sí solas lesiones, cambios de pista o noticias.')

# PANEL INDEPENDIENTE: permanece accesible aunque todavía no se haya cargado el histórico.
@st.cache_data(ttl=3600,show_spinner=False)
def cached_tennis_sports(api_key:str):
    return tennis_tournaments(api_key)

@st.cache_data(ttl=1800,show_spinner=False)
def cached_tennis_prices(api_key:str, sport_key:str, region:str, markets:tuple[str,...]):
    return fetch_tournament(api_key,sport_key,region=region,markets=markets)

with st.expander('📡 CUOTAS AUTOMÁTICAS · ATP / WTA (otras casas, no Betano)',expanded=True):
    st.caption('Fuente: The Odds API (the-odds-api.com) · Consulta automática al abrir un torneo · caché compartida de 30 minutos por consulta.')
    st.warning('Cotizaciones de otras casas **NO son cuotas verificadas de Betano Chile**. Pueden variar en importe, líneas, liquidación y disponibilidad. Ningún mercado es seguro.')
    if not available_key():
        st.info('Para habilitar cuotas reales debes crear una clave gratuita en The Odds API y guardarla como **THE_ODDS_API_KEY** en las variables de entorno privadas de Render. No pegues la clave en GitHub.')
        st.markdown('[Obtener clave de The Odds API](https://the-odds-api.com/#get-access) · [Abrir configuración de TennisEdge en Render](https://dashboard.render.com/web/srv-dbb3g9mi0phs73bc97tg)')
        st.caption('El plan gratuito consultado ofrece 500 créditos mensuales; los mercados de hándicaps y totales tienen cobertura limitada en tenis. No se realizan apuestas ni scraping.')
    else:
        api_key=os.getenv('THE_ODDS_API_KEY','').strip()
        try:
            tournaments=[t for t in cached_tennis_sports(api_key) if t['active']]
            if not tournaments:
                st.info('No hay torneos de tenis activos con cuotas en la API en este momento. Esto no significa que no haya partidos de tenis.')
            else:
                choices={f"{t['title']} · {t['key']}":t['key'] for t in tournaments}
                a,b,c=st.columns([2.7,1.1,1.3])
                selected_name=a.selectbox('Torneo con mercado disponible',list(choices.keys()),key='odds_tournament')
                region=b.selectbox('Casas de región',['eu','uk','us','au'],key='odds_region')
                mode=c.selectbox('Tipo de cuota',['Ganador','Ganador + hándicaps + totales'],key='odds_market_type')
                markets=('h2h',) if mode=='Ganador' else ('h2h','spreads','totals')
                st.caption('La búsqueda ampliada usa hasta 3 créditos por torneo y región. Escoge un torneo distinto para consultar sus precios.')
                games,quota=cached_tennis_prices(api_key,choices[selected_name],region,markets)
                raw=flatten_odds(games,only_prematch=True)
                st.caption(f"Créditos restantes según el proveedor: **{quota.get('x-requests-remaining','?')}** · Última petición cobrada: {quota.get('x-requests-last','?')} · Precios en caché hasta 30 min.")
                if raw.empty:
                    st.info('El torneo no tiene cuotas previas al inicio de partidos disponibles en las casas consultadas.')
                else:
                    m1,m2,m3=st.columns(3)
                    m1.metric('Partidos con cuotas',raw.event_id.nunique())
                    m2.metric('Casas encontradas',raw.casa.nunique())
                    m3.metric('Mercados diferentes',raw.market_key.nunique())
                    # Las cuotas antiguas se excluyen de la comparación para no hacer pasar un precio obsoleto por actual.
                    minutes=st.slider('Antigüedad máxima del precio (minutos)',15,180,90,15,key='odds_max_age')
                    minimum=st.select_slider('Mínimo de casas para comparar',options=[1,2,3,4,5],value=1,key='odds_min_books')
                    table=summarize_prices(raw,max_age_minutes=minutes,min_books=minimum)
                    lower,upper=st.slider('Rango de cuota que deseas examinar',min_value=1.05,max_value=4.0,value=(1.30,1.60),step=0.05,key='odds_band')
                    table=table.loc[table.mejor_cuota.between(lower-0.001,upper+0.001)].copy() if not table.empty else table
                    if table.empty:
                        st.warning('No hay precios suficientemente recientes dentro de este rango. Prueba otro torneo, otro mercado o amplía la antigüedad; no se inventan cuotas.')
                    else:
                        view=table.copy()
                        view['inicio_chile']=view['inicio_utc'].dt.tz_convert('America/Santiago').dt.strftime('%d-%m %H:%M')
                        view['linea']=view['linea'].map(lambda x: '' if pd.isna(x) else f'{x:+g}')
                        view=view.rename(columns={'partido':'Partido','mercado':'Mercado','seleccion':'Selección','linea':'Línea','mejor_cuota':'Mejor cuota','mejor_casa':'Casa con mejor cuota','casas':'N.º casas','cuota_mediana':'Mediana de cuotas','consenso_sin_margen_pct':'Consenso sin margen (%)','inicio_chile':'Inicio Chile','actualizacion_reciente':'Actualizado UTC'})
                        st.dataframe(view[['Inicio Chile','Partido','Mercado','Selección','Línea','Mejor cuota','Casa con mejor cuota','N.º casas','Mediana de cuotas','Consenso sin margen (%)','Actualizado UTC']],hide_index=True,use_container_width=True)
                        st.download_button('Descargar cuotas observadas (CSV)',view.to_csv(index=False).encode('utf-8-sig'),file_name='tennisedge_cuotas_mercado.csv',mime='text/csv')
                        st.caption('El consenso sin margen es la mediana de probabilidades normalizadas de **al menos tres casas** con ambas cotizaciones. No es una predicción independiente ni evidencia de valor rentable.')
                    with st.expander('Ver todas las cuotas y procedencia'):
                        full=raw.copy()
                        full['inicio_chile']=full['inicio_utc'].dt.tz_convert('America/Santiago').dt.strftime('%d-%m %H:%M')
                        st.dataframe(full[['inicio_chile','jugador_a','jugador_b','mercado','seleccion','linea','casa','cuota','ultima_actualizacion','antiguedad_min']],hide_index=True,use_container_width=True)
        except OddsFeedError as ex:
            st.error(str(ex))
        except Exception as ex:
            st.error('La consulta de cuotas no pudo completarse. Inténtalo más tarde; si persiste, revisa Render y la vigencia de la clave.')
    st.caption('Esta pantalla solo consulta y muestra precios de un proveedor autorizado, sin iniciar sesión en casas de apuestas ni apostar. La app no recopila datos personales de Betano.')

if source=='Archivo histórico (no comercial)' and not st.session_state.get('ready',False):
    st.info('Para comenzar, confirma la licencia de la fuente en la barra lateral y pulsa **Cargar y analizar datos reales**. La primera descarga puede tardar uno o dos minutos.')
    st.markdown(f'**Procedencia de los históricos:** [{ATTRIBUTION}]({ORIGINAL})')
    st.stop()
if source!='Archivo histórico (no comercial)' and not (atp_bytes or wta_bytes):
    st.info('Carga tus archivos históricos ATP/WTA en el menú lateral. No se muestran jugadores ficticios.')
    st.stop()
try:
    with st.spinner('Preparando datos y comparando Elo, regresión logística, Random Forest y Gradient Boosting…'):
        lab,log=fit_lab(atp_bytes,wta_bytes,years,tours,source)
except (DataError,ValueError,UnicodeError) as e:
    st.error(f'No fue posible procesar el histórico: {e}')
    st.stop()
except Exception as e:
    st.error(f'Error de carga o conexión: {str(e)[:250]}')
    st.stop()

if log:
    errors=[x for x in log if x.get('status')!='ok']
    if errors:
        st.warning(f'No se descargaron {len(errors)} años solicitados. Verifica cobertura en Fuentes. Solo se usan archivos recibidos correctamente.')
max_date=lab.matches.date.max().date()
age=(date.today()-max_date).days
stale=age>60
if stale:
    st.error(f'**Históricos desactualizados para pronósticos actuales:** el último partido registrado es del {max_date}. No usar señales como recomendaciones de apuestas en vivo.')

stats=st.columns(4)
stats[0].metric('Partidos históricos',f'{len(lab.matches):,}'.replace(',','.'))
stats[1].metric('Última fecha',max_date.strftime('%d-%m-%Y'))
stats[2].metric('Test cronológico',f'{lab.n_test:,}'.replace(',','.'))
stats[3].metric('Modelo elegido',lab.selected)

tab1,tab2,tab3,tab4,tab5=st.tabs(['🎯 Partido','📋 Mercados','📈 Modelos','🗃️ Fuentes','📷 OCR captura'])

with tab1:
    st.subheader('Comparación prepartido')
    tours_available=sorted({k[0] for k in lab.states})
    col1,col2,col3=st.columns(3)
    tour=col1.selectbox('Circuito',tours_available)
    surface=col2.selectbox('Superficie',['Hard','Clay','Grass','Carpet'])
    bo=col3.selectbox('Formato',[3,5],format_func=lambda x:f'Mejor de {x} sets')
    players=sorted(name for (t,name) in lab.states if t==tour)
    left,right=st.columns(2)
    a=left.selectbox('Jugador A',players)
    b=right.selectbox('Jugador B',[p for p in players if p!=a])
    physical=st.radio('Control físico confirmado externamente',
                      ['No revisado','Sin incidencias reportadas tras comprobar','Lesiones / fatiga / retiro posible'],
                      horizontal=False)
    pred=lab.predict(tour,a,b,surface,bo)
    probs=pred['probabilities'];detail=pred['details']
    approved=not stale and physical=='Sin incidencias reportadas tras comprobar' and min(detail['matches_a'],detail['matches_b'])>=25 and min(detail['surface_a'],detail['surface_b'])>=8
    if not approved:
        st.warning('PARTIDO SIN APROBACIÓN: faltan datos recientes, verificación física o muestra suficiente por superficie.')
    st.markdown('#### Predicciones experimentales')
    models=pd.DataFrame([{'Modelo':k,'Probabilidad A (%)':round(v*100,2)} for k,v in probs.items() if k in ('Elo por superficie','Regresión logística','Random Forest','Gradient Boosting')])
    st.dataframe(models,use_container_width=True,hide_index=True)
    pwin=probs[lab.selected]
    st.metric(f'{a} gana · {lab.selected}',f'{pwin:.1%}')
    st.caption('Modelo seleccionado por el Brier en validación pasada, no por rendimiento futuro garantizado.')
    st.subheader('Comparar con cuotas introducidas manualmente')
    st.caption('Escribe cuotas que hayas visto en Betano u otro proveedor, identificando fuente y hora. **No se extraen cuotas de Betano automáticamente**.')
    betano_image=st.file_uploader('Captura de la cuota (opcional, solo visualización)',type=['png','jpg','jpeg'],key='image_betano')
    if betano_image is not None:
        st.image(betano_image.getvalue(),caption='Captura subida manualmente. Transcribe abajo las cuotas que aparecen.',width=360)
    st.caption('Las cuotas precargadas a 1,50 son ejemplos ficticios: reemplázalas siempre por las observadas y registra la hora.')
    rows=[]
    terms=[('Ganador A',pwin),('A gana un set',probs['A gana ≥1 set']),('A −2,5 juegos',probs['A −2,5 juegos']),('Más de 22,5 juegos',probs['Más de 22,5 juegos'])]
    columns=st.columns(2)
    for i,(label,prob) in enumerate(terms):
        with columns[i%2]:
            price=st.number_input(f'{label} · cuota decimal',min_value=1.01,max_value=100.,value=1.5,step=.05,key=f'odds_{i}')
        implied=1/price
        edge=prob-implied
        ev=prob*price-1
        rows.append({'Mercado':label,'Cuota':price,'Probabilidad modelo':round(prob,4),
                     'Implicada por cuota':round(implied,4),'EV teórico':round(ev,4),
                     'Estado':'No aprobada' if not approved else 'Experimental / validar'})
    frame=pd.DataFrame(rows)
    st.dataframe(frame,use_container_width=True,hide_index=True)
    st.download_button('Descargar análisis CSV',frame.to_csv(index=False).encode('utf-8-sig'),
                       'tennisedge_mercados.csv','text/csv')
    st.warning('El EV teórico no prueba rentabilidad. Se necesitan cuotas históricas legítimas para calcular ROI y calibración por mercado; un modelo preciso puede perder dinero.')

with tab2:
    st.subheader('Escáner de cuotas observadas · solo lectura')
    st.write('Este módulo funciona como un **robot analítico**: importa mercados de un CSV autorizado, comprueba cobertura y señala cuáles requieren revisión. No accede a Betano ni realiza apuestas.')
    template='tour,player_a,player_b,surface,best_of,market,odds,provider,observed_at,injury_check\nATP,Jugador A,Jugador B,Hard,3,winner,1.55,Betano,2026-10-19 12:00,unknown\n'
    st.download_button('Descargar plantilla CSV',template,'plantilla_cuotas.csv','text/csv')
    upload=st.file_uploader('Subir CSV de mercados observados',type=['csv'],key='odds_csv')
    if upload:
        try:
            table=pd.read_csv(upload)
            expected={'tour','player_a','player_b','surface','market','odds'}
            if not expected.issubset(table.columns):
                st.error('Faltan columnas: '+', '.join(sorted(expected-set(table.columns))))
            else:
                scanned=[]
                mk={'winner':'selected','set':'A gana ≥1 set','minus25':'A −2,5 juegos','over225':'Más de 22,5 juegos'}
                for _,r in table.head(200).iterrows():
                    t=str(r.tour).upper();aa=str(r.player_a);bb=str(r.player_b);ss=str(r.surface).title();key=str(r.market).strip().lower()
                    odds=pd.to_numeric(r.odds,errors='coerce')
                    observation=r.get('observed_at',None)
                    is_fresh=False
                    if pd.notna(observation):
                        when=pd.to_datetime(str(observation),errors='coerce')
                        if pd.notna(when):
                            if when.tzinfo is None:
                                when=when.tz_localize('America/Santiago',ambiguous='NaT',nonexistent='NaT')
                            if pd.notna(when):
                                minutes=(pd.Timestamp.now(tz='America/Santiago')-when.tz_convert('America/Santiago')).total_seconds()/60
                                is_fresh=-5 <= minutes <= 20
                    if key not in mk or pd.isna(odds) or odds<=1:
                        continue
                    info={'Partido':f'{aa} vs {bb}','Mercado':key,'Cuota':float(odds),
                          'Casa':r.get('provider','no especificada')}
                    try:
                        p=lab.predict(t,aa,bb,ss,int(r.get('best_of',3)))
                        pr=p['probabilities'][lab.selected if key=='winner' else mk[key]]
                        d=p['details']
                        ok=str(r.get('injury_check','unknown')).lower()=='verified_ok'
                        risky=stale or not is_fresh or not ok or min(d['matches_a'],d['matches_b'])<25 or min(d['surface_a'],d['surface_b'])<8
                        info.update({'P estimada':round(pr,3),'EV teórico':round(float(pr*odds-1),3),
                                     'Estado':'REVISAR: precio/físico/datos' if risky else 'Experimental / no validado'})
                    except (ValueError,KeyError) as ex:
                        info.update({'Estado':'Datos insuficientes / revisar nombres'})
                    scanned.append(info)
                if scanned:
                    result=pd.DataFrame(scanned).sort_values('EV teórico',ascending=False,na_position='last')
                    st.dataframe(result,use_container_width=True,hide_index=True)
                    st.download_button('Exportar resultados',result.to_csv(index=False).encode('utf-8-sig'),
                                       'escaneo_tennisedge.csv','text/csv')
                else:st.warning('No se identificaron mercados válidos.')
        except Exception as e:st.error(f'CSV inválido: {str(e)[:170]}')
    st.caption('Los precios transcritos no son cuotas verificadas por API. El robot NO recopila sesiones privadas ni evita restricciones, CAPTCHA o geolocalización.')

with tab3:
    st.subheader('Evaluación temporal: no usar resultados futuros para entrenar el pasado')
    st.caption(f"Inicio validación: {lab.split_dates[0]} · Inicio test: {lab.split_dates[1]} · Entrenamiento {lab.n_train:,} · Validación {lab.n_val:,} · Prueba {lab.n_test:,}" )
    st.write('**Modelos de ganador, medidos sobre el mismo período nunca usado para elegir hiperparámetros:**')
    scored=lab.scoreboard.copy()
    for c in ('Exactitud','Brier ↓','Log loss ↓','AUC'):
        scored[c]=scored[c].round(4)
    st.dataframe(scored,use_container_width=True,hide_index=True)
    st.write('**Otros mercados:** comparación de regresión logística retrospectiva. No son probabilidades calibradas para casas de apuestas.')
    extra=lab.market_metrics.copy()
    for c in ('Exactitud','Brier ↓','Log loss ↓','AUC'):extra[c]=extra[c].round(4)
    st.dataframe(extra,use_container_width=True,hide_index=True)
    st.markdown('**Interpretación:** menor Brier y Log loss es mejor. La exactitud sin cuotas y comisiones no equivale a ganar dinero.')
    st.warning('No calculamos ROI, yield ni efectividad del 98 % sin históricos de cuotas permitidos. Los mercados alternativos requieren validación y calibración adicionales.')
    st.download_button('Descargar comparación de modelos',scored.to_csv(index=False).encode('utf-8-sig'),
                       'tennisedge_modelos_test.csv','text/csv')

with tab4:
    st.subheader('Procedencia y límites de los datos')
    st.markdown(f'Histórico: [{ATTRIBUTION}]({ORIGINAL})' if source=='Archivo histórico (no comercial)' else 'Históricos aportados manualmente por el usuario.')
    st.write(f'Última fecha encontrada: **{max_date}**. La fecha del archivo no significa que cubra partidos jugados hoy.')
    if log:
        st.dataframe(pd.DataFrame(log),use_container_width=True,hide_index=True)
    st.write('**Calendario licenciado opcional:** Sportradar Tennis API requiere una licencia autorizada y una clave guardada de forma privada en Render.')
    if has_key():
        st.success('La clave de una API licenciada está configurada en el servidor. No se comparte con los visitantes.')
        selected_day=st.date_input('Consultar calendario ATP/WTA autorizado',value=date.today())
        selected_access=st.selectbox('Nivel API contratado',['trial','production'])
        if st.button('Consultar calendario por API autorizada'):
            try:
                fixtures=flatten_summaries(daily_summaries(str(selected_day),selected_access))
                if fixtures.empty:
                    st.info('No se encontraron partidos ATP/WTA individuales en el segmento de API consultado.')
                else:
                    st.dataframe(fixtures,use_container_width=True,hide_index=True)
            except Exception as exc:
                st.error(f'No se pudo consultar Sportradar: {str(exc)[:160]}')
    else:
        st.info('Conector oficial preparado, inactivo: falta contratar el acceso y definir SPORTRADAR_TENNIS_API_KEY en el entorno privado de Render. Nunca pegues claves API en un repositorio público.')
    st.write('**Cuotas Betano:** no se han obtenido por API ni scraping. Consulta el sitio manualmente, captura la pantalla e introduce la cuota o carga el CSV de mercados observados.')
    st.dataframe(lab.matches[['date','tour','winner','loser','surface','score']].tail(50).iloc[::-1],use_container_width=True,hide_index=True)
    st.warning('Investigación personal no comercial. No hay apuestas seguras, no hay garantía de ganancias y no se realizan operaciones en sitios de apuestas.')


with tab5:
    st.subheader('Robot visual experimental · lectura de captura aportada por ti')
    st.caption('Lee texto y posibles cuotas desde una imagen **que tú seleccionas**. No abre Betano, no consulta cuotas reales y no realiza ninguna apuesta.')
    st.warning('Antes de subir imágenes recorta el saldo, identificadores y cualquier información personal. El OCR es orientativo y puede leer incorrectamente números, nombres y mercados.')
    pic=st.file_uploader('Captura recortada de los mercados (PNG o JPG)',type=['png','jpg','jpeg'],key='ocr_image')
    if pic:
        binary=pic.getvalue()
        if len(binary)>3_000_000:
            st.error('Imagen superior a 3 MB. Recorta/comprime antes de usar el OCR.')
        else:
            st.image(binary,width=380)
            if st.button('Reconocer texto y posibles cuotas (experimental)'):
                mime='image/png' if pic.type=='image/png' else 'image/jpeg'
                src='data:'+mime+';base64,'+base64.b64encode(binary).decode('ascii')
                html=rf"""<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"/>
                <script src="https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js"></script></head>
                <body style="background:#111827;color:#fff;font-family:system-ui;padding:10px">
                <strong>OCR local en tu navegador · No envía apuestas</strong>
                <p id="state">Leyendo captura… Puede tardar unos segundos.</p>
                <div id="odds" style="font-size:20px;color:#d6fb68;margin:10px 0"></div>
                <textarea id="text" rows="12" style="width:97%;background:#1f2937;color:white"></textarea>
                <img id="source" src="{src}" style="display:none"/>
                <script>
                (async()=>{{
                  try {{
                    if(!window.Tesseract) throw new Error('No se pudo cargar el motor OCR');
                    const worker=await Tesseract.createWorker('spa');
                    const result=await worker.recognize(document.getElementById('source'));
                    const raw=result.data.text||'';
                    document.getElementById('text').value=raw;
                    const odds=Array.from(new Set((raw.match(/\b[1-9][.,]\d{{2}}\b/g)||[]))).slice(0,35);
                    document.getElementById('odds').textContent=odds.length ? 'Posibles cuotas: '+odds.join(' · ') : 'No encontré cuotas claras';
                    document.getElementById('state').textContent='Lectura terminada. Compara todo con Betano manualmente antes de registrar una cuota.';
                    await worker.terminate();
                  }} catch(e) {{document.getElementById('state').textContent='OCR no disponible: '+String(e);}}
                }})();
                </script></body></html>"""
                components.html(html,height=400,scrolling=True)
    st.info('Este lector puede equivocarse; el usuario debe verificar la casa, la hora y la cuota. No se utiliza para apostar ni sustituye un feed autorizado.')
