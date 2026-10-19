"""Interfaz TennisEdge IA — ejecutar: streamlit run app.py"""
from __future__ import annotations

from io import BytesIO
import pandas as pd
import numpy as np
import streamlit as st

from engine import TennisModel, DataError, normalize_matches, evaluate_odds

st.set_page_config(page_title='TennisEdge IA · Laboratorio estadístico',
                   page_icon='🎾', layout='wide', initial_sidebar_state='auto')

st.markdown('''
<style>
:root {color-scheme:dark}
.stApp {background:#0c1020; color:#f1f5f9;}
[data-testid="stSidebar"] {background:#121a2d;}
h1,h2,h3 {letter-spacing:-.035em}
h1 {font-weight:800 !important}
[data-testid="stMetric"] {border:1px solid #27314b; border-radius:15px;padding:16px;background:#141e32;}
[data-testid="stMetricValue"] {color:#d4ff60;}
div.stButton button[kind="primary"] {background:#cbfc52;color:#132008;border:none;font-weight:700;}
hr {border-color:#263147;}
div[data-testid="stAlert"] {border-radius:12px;}
.smallcap {font-size:.78rem;color:#9fb0cf;font-weight:700;letter-spacing:.12em;text-transform:uppercase}
</style>
''', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def generate_demo_csv(tour: str) -> bytes:
    """Dataset 100 % sintético, solo para demostrar la app en la nube."""
    rng = np.random.default_rng(426 if tour == 'ATP' else 427)
    count = 30
    names = [f'{tour} · Jugador ficticio {i:02d}' for i in range(1, count + 1)]
    basic = rng.normal(0, .9, count)
    surface_mod = rng.normal(0, .35, (count, 3))
    dates = pd.date_range('2022-01-05', '2026-10-09', freq='D')
    rows=[]
    for k in range(1800):
        ai, bi = rng.choice(count, 2, replace=False)
        s_idx = rng.choice(3, p=[.62, .30, .08])
        surface = ('Hard', 'Clay', 'Grass')[s_idx]
        qa = basic[ai] + surface_mod[ai, s_idx]
        qb = basic[bi] + surface_mod[bi, s_idx]
        win_first = rng.random() < 1/(1 + np.exp(-(qa-qb)*.9))
        w, l = (ai,bi) if win_first else (bi,ai)
        if rng.random() < .70:
            score = f"6-{rng.choice([1,2,3,4],p=[.12,.20,.38,.30])} 6-{rng.choice([2,3,4],p=[.2,.4,.4])}"
        else:
            score = f"6-{rng.choice([3,4])} {rng.choice([3,4])}-6 6-{rng.choice([2,3,4])}"
        rows.append({'tourney_date':dates[int(k/1800*(len(dates)-1))].strftime('%Y%m%d'),
            'winner_name':names[w], 'loser_name':names[l], 'surface':surface,
            'score':score,'best_of':3})
    return pd.DataFrame(rows).to_csv(index=False).encode('utf-8')

@st.cache_data(show_spinner=False)
def read_and_normalize(content:bytes,tour:str):
    df=pd.read_csv(BytesIO(content),low_memory=False)
    return normalize_matches(df,tour=tour)

@st.cache_resource(show_spinner='Construyendo Elo cronológico y estadísticas históricas...')
def build_model(atp_bytes:bytes|None,wta_bytes:bytes|None,mode:str):
    frames=[]
    if atp_bytes:
        frames.append(read_and_normalize(atp_bytes,'ATP'))
    if wta_bytes:
        frames.append(read_and_normalize(wta_bytes,'WTA'))
    if not frames:
        raise DataError('No hay resultados históricos cargados.')
    return TennisModel(pd.concat(frames,ignore_index=True))

st.sidebar.markdown('## 🎾 TENNISEDGE **IA**')
st.sidebar.caption('Análisis cuantitativo · Edición web 0.2')
st.sidebar.markdown('---')
mode=st.sidebar.radio('Origen de los datos', ['Demostración (simulada)','CSV históricos reales'],index=0)
atp=None;wta=None
if mode.startswith('Demostración'):
    atp=generate_demo_csv('ATP')
    wta=generate_demo_csv('WTA')
    st.sidebar.warning('MODO DEMO: resultados inventados. No usar probabilidades ni cuotas para apostar.')
else:
    up_atp=st.sidebar.file_uploader('Historial ATP (CSV)',type='csv',key='atp')
    up_wta=st.sidebar.file_uploader('Historial WTA (CSV)',type='csv',key='wta')
    if up_atp: atp=up_atp.getvalue()
    if up_wta: wta=up_wta.getvalue()
    st.sidebar.caption('Admite columnas Sackmann: winner_name, loser_name, tourney_date, surface, score; o Tennis-Data: Winner, Loser, Date, Surface, W1/L1...')
st.sidebar.markdown('---')
st.sidebar.caption('La herramienta no accede a Betano, no realiza apuestas y no asegura ganancias.')

st.markdown('<div class="smallcap">Laboratorio de modelos deportivos</div>',unsafe_allow_html=True)
st.title('TennisEdge IA  🎾')
st.markdown('**Mercados alternativos · Comparación de cuotas · Control de riesgo**')
st.caption('📱 En iPhone, abre el menú «>» de la esquina superior izquierda para cambiar entre demo y archivos históricos.')
if mode.startswith('Demostración'):
    st.error('**SIMULACIÓN — NO USAR PARA DECISIONES REALES.** Los nombres, resultados, probabilidades y oportunidades mostrados son ficticios.')
else:
    st.info('**Datos aportados por el usuario.** Modelo experimental: las probabilidades no están calibradas frente a Betano ni incluyen información médica automatizada.')

if not atp and not wta:
    st.warning('Carga un CSV de partidos ATP o WTA en el panel izquierdo para activar el modelo. También puedes probar la demostración.')
    st.stop()
try:
    model=build_model(atp,wta,mode)
except (DataError, ValueError, UnicodeDecodeError) as err:
    st.error(f'No se pudo procesar la información: {err}')
    st.stop()

n=len(model.matches); players=len(model.players)
val=model.backtest()
cols=st.columns(4)
cols[0].metric('Partidos procesados',f'{n:,}'.replace(',','.'))
cols[1].metric('Jugadores con historial',f'{players:,}'.replace(',','.'))
cols[2].metric('Precisión retrospectiva',f"{100*val['accuracy']:.1f}%" if val['valid'] else 'Sin validar')
cols[3].metric('Brier retrospectivo',f"{val['brier']:.3f}" if val['valid'] else '—')
st.caption('La precisión es una medición cronológica interna (sin odds). No demuestra rentabilidad ni capacidad de anticipar lesiones o retiros.')

tab1,tab2,tab3,tab4=st.tabs(['🔎 Analizar partido','📋 Buscador de cuotas','📊 Validación','🗃️ Historial y datos'])

with tab1:
    st.subheader('Analizador de partido')
    c1,c2,c3=st.columns([1,1,.8])
    tours=sorted(model.matches.tour.unique().tolist())
    tour=c1.selectbox('Circuito',tours)
    surface=c2.selectbox('Superficie', ['Hard','Clay','Grass','Carpet'])
    best_of=c3.selectbox('Formato',[3,5],format_func=lambda n:f'Al mejor de {n}')
    names=sorted({name for t,name in model.players if t==tour})
    if len(names)<2:
        st.warning('No hay suficientes jugadores para el circuito.')
    else:
        c1,c2=st.columns(2)
        a=c1.selectbox('Jugador A',names,index=0)
        b=c2.selectbox('Jugador B',[x for x in names if x!=a],index=min(1,len(names)-2))
        injured=st.selectbox('Información física (verificada externamente)',
                         ['Sin comprobar / desconocida','Ambos sin lesiones reportadas','Dudas o lesiones relevantes'], index=0)
        snapshot=model.probability(tour,a,b,surface)
        p=snapshot['p_win']
        pcol1,pcol2,pcol3=st.columns(3)
        pcol1.metric(f'Elo {a}',snapshot['elo_a'])
        pcol2.metric(f'Elo {b}',snapshot['elo_b'])
        pcol3.metric('Victoria A · Elo bruto',f'{p*100:.1f} %')
        st.progress(float(p),text=f'{a} {100*p:.1f} % | {b} {100*(1-p):.1f} %')
        low_sample= min(snapshot['matches_a'],snapshot['matches_b'])<15 or min(snapshot['surface_a'],snapshot['surface_b'])<5
        if low_sample:
            st.warning('Historial limitado en general o en esta superficie: estimaciones menos confiables.')
        if injured=='Dudas o lesiones relevantes':
            st.error('FILTRO DE RIESGO: partido con dudas físicas. Evita usarlo como señal de apuesta.')
        if injured=='Sin comprobar / desconocida':
            st.warning('Estado físico sin comprobar. El modelo no detecta lesiones de manera automática.')
        st.markdown('### Comparador de mercados')
        st.caption('Cuotas manuales. Los intervalos son orientativos y NO están certificados por un modelo calibrado.')
        market_keys=[
            ('Victoria A','winner',p,0),
            ('A gana al menos 1 set','set',p,0),
            ('A hándicap −2,5 juegos','handicap',p,-2.5),
            ('B hándicap +2,5 juegos','handicap',1-p,2.5),
            ('Más de 22,5 juegos','over',p,22.5),
            ('Menos de 22,5 juegos','under',p,22.5),
        ]
        with st.form('market_odds'):
            prices={}
            first=st.columns(3)
            for idx,(name,_,__,___) in enumerate(market_keys):
                with first[idx%3]:
                    prices[name]=st.number_input(f'{name}: cuota',min_value=1.01,max_value=100.,value=1.50,step=.05,key=f'price_{idx}')
            submitted=st.form_submit_button('Calcular mercados',type='primary')
        results=[]
        for name,mtype,p_ref,line in market_keys:
            result=model.estimate_market(tour,surface,p_ref,mtype,line,best_of)
            ev=evaluate_odds(result,prices[name])
            results.append({'Mercado':name,'Cuota':prices[name],
                            'Prob. estimada':f"{result['p']:.1%}" if result['p'] is not None else 'Insuficiente',
                            'Intervalo (90 %)':f"{result['lower']:.0%}–{result['upper']:.0%}" if result['p'] is not None else '—',
                            'Prob. implícita':f"{ev['implied']:.1%}",
                            'EV teórico':f"{ev['ev']:+.1%}" if ev['ev'] is not None else '—',
                            'Muestra efectiva':result['n_eff']})
        result_frame=pd.DataFrame(results)
        st.dataframe(result_frame,use_container_width=True,hide_index=True)
        st.caption('EV = p estimada × cuota − 1. Un EV positivo no constituye ventaja demostrada; requiere calibración y contraste contra cuotas de cierre.')
        if best_of==5:
            st.info('Con formato a 5 sets, algunas probabilidades pueden carecer de suficientes partidos análogos.')
        if submitted:
            if low_sample or injured != 'Ambos sin lesiones reportadas' or not val['valid'] or mode.startswith('Demostración'):
                st.warning('No se puede aprobar ninguna selección: quedan controles de calidad, salud, validación o fuente pendiente.')
            else:
                st.info('Cálculo listo para auditoría. El modelo aún no ha demostrado una ventaja rentable sobre el mercado; no se emite una señal de apuesta.')
        st.download_button('Descargar comparación CSV',result_frame.to_csv(index=False).encode('utf-8-sig'),
                           file_name='tennisedge_mercados.csv',mime='text/csv')

with tab2:
    st.subheader('Buscador de mercados · cargar próximos partidos')
    st.write('Carga un CSV de enfrentamientos próximos. El escáner compara una cuota para **ganador A**, **A gana un set** o **hándicap A −2,5**.')
    st.code('tour,player_a,player_b,surface,best_of,odds_a_win,odds_a_set,odds_a_minus25,injury_check\n'
            'ATP,Jugador A,Jugador B,Hard,3,1.60,1.20,1.85,unknown',language='text')
    uploaded=st.file_uploader('Archivo de próximos partidos (CSV)',type='csv',key='upcoming')
    if uploaded is not None:
        try:
            upcoming=pd.read_csv(uploaded)
            required={'tour','player_a','player_b','surface'}
            if not required.issubset(upcoming.columns):
                st.error('Faltan columnas: '+', '.join(sorted(required-upcoming.columns)))
            else:
                data=[]
                for _, row in upcoming.head(150).iterrows():
                    t,pa,pb,s=str(row.tour).upper(),str(row.player_a),str(row.player_b),str(row.surface).title()
                    if t not in tours or (t,pa) not in model.players or (t,pb) not in model.players:
                        data.append({'Partido':f'{pa} — {pb}','Mercado':'—','Estado':'Falta historial','EV estimado':None})
                        continue
                    bo=int(row.get('best_of',3))
                    pp=model.probability(t,pa,pb,s)
                    low=min(pp['matches_a'],pp['matches_b'])<15 or min(pp['surface_a'],pp['surface_b'])<5
                    injury=str(row.get('injury_check','unknown')).strip().lower()
                    risk=low or injury not in ('checked_ok','ok','clear')
                    for market,key,typ,line in [('A ganador','odds_a_win','winner',0),('A gana set','odds_a_set','set',0),('A −2,5 juegos','odds_a_minus25','handicap',-2.5)]:
                        odd=pd.to_numeric(row.get(key),errors='coerce')
                        if pd.isna(odd) or odd<=1:
                            continue
                        est=model.estimate_market(t,s,pp['p_win'],typ,line,bo)
                        metrics=evaluate_odds(est,float(odd))
                        data.append({'Partido':f'{pa} — {pb}','Mercado':market,'Cuota':round(float(odd),2),
                                     'Prob. estimada':round(est['p'],3) if est['p'] is not None else None,
                                     'EV estimado':metrics['ev'],
                                     'Estado':'Revisión obligatoria' if risk else 'Experimental: no validado contra odds',
                                     'Muestra efectiva':est['n_eff']})
                scan=pd.DataFrame(data)
                if not scan.empty:
                    st.dataframe(scan,use_container_width=True,hide_index=True)
                    st.download_button('Descargar resultados del escáner',scan.to_csv(index=False).encode('utf-8-sig'),
                                       'tennisedge_escaneo.csv','text/csv')
                else:
                    st.warning('El archivo no contiene cuotas válidas o partidos reconocidos.')
        except Exception as exc:
            st.error(f'El CSV no se pudo interpretar: {exc}')
    st.info('No se integra a una casa de apuestas ni se rastrean cuotas en vivo. Las lesiones no se infieren del historial.')

with tab3:
    st.subheader('Validación cronológica y auditoría')
    if not val['valid']:
        st.warning(val['reason'])
    else:
        v1,v2,v3,v4=st.columns(4)
        v1.metric('Partidos evaluados',val['n_test'])
        v2.metric('Aciertos ganadores',f"{100*val['accuracy']:.1f}%")
        v3.metric('Brier score',f"{val['brier']:.3f}")
        v4.metric('Log loss',f"{val['logloss']:.3f}")
        st.caption('Se reserva el último 20 % de ejemplos elegibles. Las calificaciones Elo de cada encuentro se calculan con partidos anteriores.')
        test=val['test'].copy()
        test['bucket']=pd.cut(test.p_model,np.linspace(0,1,11),include_lowest=True)
        curve=test.groupby('bucket',observed=True).agg(predicho=('p_model','mean'),real=('winner_flag','mean'),cantidad=('winner_flag','size')).reset_index()
        st.subheader('Curva de calibración · victoria')
        st.line_chart(curve.set_index(curve.bucket.astype(str))[['predicho','real']])
        st.dataframe(curve[['bucket','predicho','real','cantidad']],use_container_width=True,hide_index=True)
        st.warning('No se ha realizado un backtest con cuotas históricas de cierre de Betano. Accuracy NO equivale a rentabilidad. Los intervalos de otros mercados son exploratorios, no calibración validada.')

with tab4:
    st.subheader('Resumen del histórico')
    st.write(f"Se cargaron **{len(model.matches):,} partidos completos** hasta **{model.matches.date.max().strftime('%d-%m-%Y')}**.")
    st.dataframe(model.matches[['date','tour','winner','loser','surface','score']].tail(100).iloc[::-1],use_container_width=True,hide_index=True)
    st.info('Para datos reales: utiliza archivos que tengas permiso para usar y que incluyan partidos previos a la predicción. No mezcles partidos de exhibición, retiros o información posterior al resultado.')
