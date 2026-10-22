import hashlib
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from tennisedge import rapid
from tennisedge.ui import SURF_ES, TZ, get_predictor, pct, render_analysis, setup, stale_warning

setup('Partidos próximos y valor', '📅')
P = get_predictor()
stale_warning(P)


@st.cache_data(ttl=3 * 3600, show_spinner='Consultando partidos próximos (2 llamadas a RapidAPI)…')
def upcoming(fingerprint: str, slot: int) -> tuple[pd.DataFrame, str]:
    rows = rapid.get_upcoming(max_pages=2)
    return rapid.parse_upcoming(rows), datetime.now(timezone.utc).isoformat(timespec='minutes')


if not rapid.available():
    st.info('Para ver los partidos del día automáticamente, configura **RAPIDAPI_TENNIS_KEY** en Render '
            '(ver **🗃️ Datos y APIs**). Mientras tanto puedes analizar cualquier partido en **🎾 Análisis de partido** '
            'o usar **💹 Cuotas de casas** con The Odds API.')
    st.stop()

fp = hashlib.sha256(os.environ[rapid.ENV_KEY].encode()).hexdigest()[:12]
slot = int(datetime.now(timezone.utc).timestamp() // (3 * 3600))
try:
    up, fetched = upcoming(fp, slot)
except rapid.RapidError as exc:
    st.error(str(exc))
    st.stop()

if up.empty:
    st.info('El proveedor no devolvió partidos próximos ahora mismo.')
    st.stop()

ms = [m for m in up.to_dict('records') if m['tour'] in ('ATP', 'WTA') and m['surface']]
preds = P.predict_many([{'tour': m['tour'], 'a': m['jugador_a'], 'b': m['jugador_b'], 'surface': m['surface'],
                         'best_of': m['best_of'], 'level': m['level'], 'odds_a': m['cuota_a'],
                         'odds_b': m['cuota_b']} for m in ms])
LEVEL_NAME = __import__('tennisedge.schema', fromlist=['LEVEL_NAME']).LEVEL_NAME
rows = []
for m, r in zip(ms, preds):
    fav_a = r['p_a'] >= .5
    ev_a, ev_b = r['value_a'].get('ev'), r['value_b'].get('ev')
    best_ev = max([x for x in (ev_a, ev_b) if x is not None], default=None)
    rows.append({
        'Inicio (Chile)': m['inicio'].tz_convert(TZ).strftime('%d-%m %H:%M') if pd.notna(m['inicio']) else '—',
        'Circuito': m['tour'], 'Torneo': m['torneo'], 'Nivel': LEVEL_NAME.get(m['level'], m['level']),
        'Pista': SURF_ES.get(m['surface'], m['surface']),
        'Partido': f"{r['a']} vs {r['b']}", 'Favorito': r['a'] if fav_a else r['b'],
        'Prob. favorito': max(r['p_a'], r['p_b']),
        'Cuota A': m['cuota_a'], 'Cuota B': m['cuota_b'],
        'EV A %': ev_a * 100 if ev_a is not None else None, 'EV B %': ev_b * 100 if ev_b is not None else None,
        'Mejor EV %': best_ev * 100 if best_ev is not None else None,
        'Confianza': r['confidence']['score'],
        '_m': m,
    })
tab = pd.DataFrame(rows)
st.caption(f'{len(tab)} partidos individuales con superficie conocida · consulta {fetched} UTC '
           '(se reutiliza 3 h para cuidar tus 50 llamadas/día).')
if tab.empty:
    st.stop()

c1, c2, c3, c4 = st.columns(4)
tours = c1.multiselect('Circuito', ['ATP', 'WTA'], default=['ATP', 'WTA'])
min_conf = c2.slider('Confianza mínima', 0, 100, 55, 5)
only_odds = c3.toggle('Solo con cuotas', value=False)
min_ev = c4.slider('EV mínimo %', -20, 30, -20, 1)
view = tab[tab.Circuito.isin(tours) & (tab.Confianza >= min_conf)]
if only_odds:
    view = view[view['Cuota A'].notna()]
if min_ev > -20:
    view = view[view['Mejor EV %'].fillna(-99) >= min_ev]
view = view.sort_values(['Mejor EV %', 'Prob. favorito'], ascending=False, na_position='last')
st.dataframe(view.drop(columns=['_m']), hide_index=True, width='stretch', height=460,
             column_config={'Prob. favorito': st.column_config.ProgressColumn(min_value=0.5, max_value=1, format='percent'),
                            'Confianza': st.column_config.ProgressColumn(min_value=0, max_value=100, format='%d'),
                            'EV A %': st.column_config.NumberColumn(format='%+.1f'),
                            'EV B %': st.column_config.NumberColumn(format='%+.1f'),
                            'Mejor EV %': st.column_config.NumberColumn(format='%+.1f')})
st.caption('EV = probabilidad del modelo × cuota − 1. Un EV positivo es una discrepancia con la casa, no una '
           'garantía. Mira **🧠 Rendimiento** para ver si esas discrepancias han rendido en partidos reales.')

st.divider()
choice = st.selectbox('Ver análisis completo de:', view['Partido'].tolist() or tab['Partido'].tolist())
if choice:
    m = tab[tab.Partido == choice].iloc[0]['_m']
    r = P.predict(m['tour'], m['jugador_a'], m['jugador_b'], m['surface'], m['best_of'], m['level'],
                  odds_a=m['cuota_a'], odds_b=m['cuota_b'])
    st.caption(f"{m['torneo']}")
    render_analysis(P, r)
