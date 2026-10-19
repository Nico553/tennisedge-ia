"""TennisEdge IA v4.1 · Multitorneo gratis opcional sin afectar fuentes previas."""
import streamlit as st
st.set_page_config(page_title='TennisEdge IA · Tenis actual',page_icon='🎾',layout='wide')
page=st.navigation([
    st.Page('live_app.py',title='Señales y cuotas actuales',icon='📡',default=True),
    st.Page('rapid_page.py',title='Multitorneo: segunda fuente',icon='🌐'),
    st.Page('lab_app.py',title='Laboratorio histórico',icon='🧠'),
],position='sidebar')
page.run()
