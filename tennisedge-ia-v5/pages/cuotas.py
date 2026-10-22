import hashlib
import os

import pandas as pd
import streamlit as st

from tennisedge import odds
from tennisedge.surfaces import resolve_level, resolve_surface, resolve_tour
from tennisedge.ui import SURF_EN, SURF_ES, TZ, get_predictor, pct, render_analysis, setup, stale_warning

setup('Cuotas de casas (The Odds API)', '💹')
P = get_predictor()
stale_warning(P)

if not odds.available_key():
    st.info('Configura **THE_ODDS_API_KEY** en Render para ver cuotas de varias casas (plan gratis: 500 créditos/mes). '
            'Ver **🗃️ Datos y APIs**.')
    st.stop()
KEY = os.environ['THE_ODDS_API_KEY'].strip()
FP = hashlib.sha256(KEY.encode()).hexdigest()[:12]


@st.cache_data(ttl=3600, show_spinner=False)
def sports(_fp):
    return odds.tennis_tournaments(KEY)


@st.cache_data(ttl=1800, show_spinner='Consultando cuotas (1 crédito)…')
def prices(_fp, sport_key, region):
    events, quota = odds.fetch_tournament(KEY, sport_key, region=region, markets=('h2h',))
    return odds.flatten_odds(events), quota


try:
    tours = [t for t in sports(FP) if t['active']]
except odds.OddsFeedError as exc:
    st.error(str(exc)); st.stop()
if not tours:
    st.info('No hay torneos de tenis activos en The Odds API ahora mismo.'); st.stop()

c1, c2 = st.columns([3, 1])
title = c1.selectbox('Torneo activo', [t['title'] for t in tours])
region = c2.selectbox('Región de casas', ['eu', 'uk', 'us', 'au'])
sel = next(t for t in tours if t['title'] == title)
tour = resolve_tour(sel['key'], title)
surface = resolve_surface(title, sel['key'])
level = resolve_level(title, sel['key'])
c1, c2 = st.columns(2)
if tour is None:
    tour = c1.radio('No pude deducir el circuito: elige', ['ATP', 'WTA'], horizontal=True)
if surface is None:
    surface = SURF_EN[c2.radio('No pude deducir la superficie: elige', ['Dura', 'Arcilla', 'Césped'], horizontal=True)]
st.caption(f'{tour} · {SURF_ES[surface]} · nivel {level}. Si algo no corresponde, el análisis manual permite corregirlo.')

try:
    raw, quota = prices(FP, sel['key'], region)
except odds.OddsFeedError as exc:
    st.error(str(exc)); st.stop()
st.caption(f"Créditos restantes del mes: {quota.get('x-requests-remaining', '?')}")
h2h = raw[raw.market_key == 'h2h'] if not raw.empty else raw
if h2h.empty:
    st.info('Sin partidos próximos con cuotas en este torneo.'); st.stop()

bo = 5 if (tour == 'ATP' and level == 'G') else 3
items, evs = [], []
for eid, ev in h2h.groupby('event_id'):
    a, b = ev.jugador_a.iloc[0], ev.jugador_b.iloc[0]
    best = ev.sort_values('cuota', ascending=False).groupby('seleccion').head(1).set_index('seleccion')
    oa = float(best.loc[a, 'cuota']) if a in best.index else None
    ob = float(best.loc[b, 'cuota']) if b in best.index else None
    items.append({'tour': tour, 'a': a, 'b': b, 'surface': surface, 'best_of': bo, 'level': level,
                  'odds_a': oa, 'odds_b': ob})
    evs.append((ev, best, a, b, oa, ob))
rows = []
for (ev, best, a, b, oa, ob), r in zip(evs, P.predict_many(items)):
    rows.append({'Inicio (Chile)': ev.inicio_utc.iloc[0].tz_convert(TZ).strftime('%d-%m %H:%M'),
                 'Partido': f'{a} vs {b}', 'P(A)': r['p_a'],
                 'Mejor cuota A': oa, 'Casa A': best.loc[a, 'casa'] if a in best.index else None,
                 'Mejor cuota B': ob, 'Casa B': best.loc[b, 'casa'] if b in best.index else None,
                 'EV A %': r['value_a']['ev'] * 100 if r['value_a'].get('ev') is not None else None,
                 'EV B %': r['value_b']['ev'] * 100 if r['value_b'].get('ev') is not None else None,
                 'Casas': ev.casa.nunique(), 'Confianza': r['confidence']['score'], '_r': (a, b, oa, ob, bo)})
tab = pd.DataFrame(rows).sort_values('Inicio (Chile)')
st.dataframe(tab.drop(columns=['_r']), hide_index=True, width='stretch',
             column_config={'P(A)': st.column_config.ProgressColumn(min_value=0, max_value=1, format='percent'),
                            'EV A %': st.column_config.NumberColumn(format='%+.1f'),
                            'EV B %': st.column_config.NumberColumn(format='%+.1f'),
                            'Confianza': st.column_config.ProgressColumn(min_value=0, max_value=100, format='%d')})
st.caption('Se usa la MEJOR cuota disponible entre casas para cada jugador. Estas casas no son Betano Chile: '
           'compara siempre con tu casa antes de decidir.')
pick = st.selectbox('Análisis completo de:', tab['Partido'].tolist())
a, b, oa, ob, bo = tab[tab.Partido == pick].iloc[0]['_r']
render_analysis(P, P.predict(tour, a, b, surface, bo, level, odds_a=oa, odds_b=ob))
