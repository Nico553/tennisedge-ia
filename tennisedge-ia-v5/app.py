"""TennisEdge IA v5 · pronósticos de tenis para análisis personal (sin fines comerciales)."""
import streamlit as st

st.set_page_config(page_title='TennisEdge IA', page_icon='🎾', layout='wide')
nav = st.navigation({
    'Pronósticos': [
        st.Page('pages/partidos.py', title='Partidos y valor', icon='📅', default=True),
        st.Page('pages/analisis.py', title='Análisis de partido', icon='🎾'),
        st.Page('pages/cuotas.py', title='Cuotas de casas', icon='💹'),
    ],
    'Modelo': [
        st.Page('pages/rendimiento.py', title='Rendimiento', icon='🧠'),
        st.Page('pages/datos.py', title='Datos y APIs', icon='🗃️'),
    ],
}, position='sidebar')
st.sidebar.caption('Uso personal · no es asesoría de apuestas. Datos: Jeff Sackmann / Tennis Abstract '
                   '(CC BY-NC-SA 4.0), Tennis API (RapidAPI), The Odds API.')
nav.run()
