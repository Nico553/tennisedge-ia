"""TennisEdge IA · Navigation: real data first; historical model lab second."""
import streamlit as st
st.set_page_config(page_title='TennisEdge IA · Tenis actual',page_icon='🎾',layout='wide')
page = st.navigation([
    st.Page('live_app.py',title='Partidos y cuotas actuales',icon='📡',default=True),
    st.Page('lab_app.py',title='Laboratorio histórico',icon='🧠'),
], position='sidebar')
page.run()
