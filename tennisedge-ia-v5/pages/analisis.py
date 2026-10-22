import streamlit as st

from tennisedge.schema import LEVEL_NAME
from tennisedge.ui import SURF_EN, get_predictor, render_analysis, setup, stale_warning

setup('Análisis de partido', '🎾')
P = get_predictor()
stale_warning(P)

c1, c2, c3 = st.columns([1, 1, 1])
tour = c1.radio('Circuito', ['ATP', 'WTA'], horizontal=True)
surface = SURF_EN[c2.radio('Superficie', ['Dura', 'Arcilla', 'Césped'], horizontal=True)]
levels = {'ATP': ['G', 'M', 'A', 'C', 'ITF'], 'WTA': ['G', 'PM', 'P', 'I', 'C', 'ITF']}[tour]
level = c3.selectbox('Nivel del torneo', levels, index=2 if tour == 'ATP' else 3, format_func=lambda x: LEVEL_NAME.get(x, x))

names = P.players(tour)
c1, c2 = st.columns(2)
a = c1.selectbox('Jugador A', names, index=0, help='Ordenados por Elo. Escribe para buscar.')
b = c2.selectbox('Jugador B', names, index=1)
c1, c2, c3 = st.columns(3)
bo = c1.radio('Formato', [3, 5], horizontal=True, format_func=lambda x: f'Mejor de {x}',
              index=1 if (tour == 'ATP' and level == 'G') else 0)
odds_a = c2.number_input(f'Cuota {a} (opcional)', min_value=0.0, value=0.0, step=0.01, format='%.2f')
odds_b = c3.number_input(f'Cuota {b} (opcional)', min_value=0.0, value=0.0, step=0.01, format='%.2f')

if a == b:
    st.warning('Elige dos jugadores distintos.')
    st.stop()
r = P.predict(tour, a, b, surface, bo, level, odds_a=odds_a or None, odds_b=odds_b or None)
render_analysis(P, r)
