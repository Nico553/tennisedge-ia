import pandas as pd
import streamlit as st

from tennisedge.ui import get_predictor, pct, setup

setup('Rendimiento del modelo', '🧠')
P = get_predictor()
rep = P.report

st.markdown(f"""
**Cómo se evalúa:** el modelo se entrenó con partidos hasta **{rep['split']['train_end']}**, se ajustó con
validación hasta **{rep['split']['val_end']}** y se juzga en un **test que nunca vio**
(desde {rep['split']['val_end']} hasta {rep['archive_last_date']}). Todas las características usan solo
información anterior a cada partido.
""")

st.subheader('Ganador del partido · test')
tm = pd.DataFrame(rep['test_models'])
st.dataframe(tm.rename(columns={'modelo': 'Modelo', 'n': 'Partidos', 'accuracy': 'Acierto', 'brier': 'Brier ↓',
                                'log_loss': 'Log loss ↓'}), hide_index=True, width='stretch',
             column_config={'Acierto': st.column_config.NumberColumn(format='percent')})
st.caption('Brier y log loss miden la calidad de las probabilidades (más bajo = mejor). El acierto solo mira '
           'si el favorito ganó.')

c1, c2 = st.columns(2)
with c1:
    st.markdown('**Por circuito / nivel**')
    g = pd.DataFrame(rep['test_groups'])
    st.dataframe(g.rename(columns={'grupo': 'Grupo', 'n': 'Partidos', 'accuracy': 'Acierto modelo',
                                   'elo_accuracy': 'Acierto Elo', 'brier': 'Brier', 'log_loss': 'Log loss'}),
                 hide_index=True, width='stretch',
                 column_config={'Acierto modelo': st.column_config.NumberColumn(format='percent'),
                                'Acierto Elo': st.column_config.NumberColumn(format='percent')})
with c2:
    st.markdown('**Calibración** (¿cuando dice 70%, gana el 70%?)')
    cal = pd.DataFrame(rep['calibration'])
    st.dataframe(cal.rename(columns={'tramo': 'Prob. del favorito', 'partidos': 'Partidos', 'prob_media': 'Prob. media',
                                     'acierto_real': 'Ganó realmente'}), hide_index=True, width='stretch',
                 column_config={'Prob. media': st.column_config.NumberColumn(format='percent'),
                                'Ganó realmente': st.column_config.NumberColumn(format='percent')})

st.subheader('Mercados · test')
mk = pd.DataFrame(rep['markets'])
mk['Estado'] = mk['avalado'].map({True: '✅ validado', False: '⚠️ no supera la referencia'})
st.dataframe(mk[['mercado', 'muestras', 'brier', 'brier_sin_calibrar', 'brier_referencia', 'acierto',
                 'acierto_referencia', 'Estado']].rename(columns={
    'mercado': 'Mercado', 'muestras': 'Muestras', 'brier': 'Brier', 'brier_sin_calibrar': 'Brier sin calibrar',
    'brier_referencia': 'Brier referencia', 'acierto': 'Acierto', 'acierto_referencia': 'Acierto referencia'}),
    hide_index=True, width='stretch',
    column_config={'Acierto': st.column_config.NumberColumn(format='percent'),
                   'Acierto referencia': st.column_config.NumberColumn(format='percent')})
st.caption('Referencia = decir siempre lo más frecuente (por línea, en totales y hándicaps). Un mercado solo se '
           'marca como validado si mejora el Brier de esa referencia sin perder acierto. La recalibración se '
           'ajustó con validación, nunca con test.')

st.subheader('Prueba real frente a las casas (partidos posteriores al histórico)')
o = P.oos_vs_market()
if not o or o.get('n', 0) < 30:
    st.info('Aún no hay suficientes partidos recientes con cuotas. Cuando se descarguen resultados de RapidAPI '
            '(traen la cuota de cierre de cada partido) aquí verás, sin trampas, si el modelo le gana a la casa.')
else:
    c1, c2, c3 = st.columns(3)
    c1.metric('Partidos con cuota', o['n'], f"{o['desde']} → {o['hasta']}", delta_color='off')
    c2.metric('Acierto modelo', pct(o['modelo_acierto']), f"casa {pct(o['casa_acierto'])}", delta_color='off')
    c3.metric('Brier modelo', f"{o['modelo_brier']:.4f}", f"casa {o['casa_brier']:.4f}", delta_color='off')
    st.markdown('**Simulación: 1 unidad cada vez que el modelo ve EV por encima del umbral**')
    st.dataframe(pd.DataFrame(o['simulacion']).rename(columns={'umbral_ev': 'Umbral EV', 'apuestas': 'Apuestas',
                                                              'ganancia_unidades': 'Ganancia (u)', 'roi': 'ROI'}),
                 hide_index=True, width='stretch',
                 column_config={'Umbral EV': st.column_config.NumberColumn(format='percent'),
                                'ROI': st.column_config.NumberColumn(format='percent')})
    st.caption('Las cuotas del proveedor pueden ser de cierre y de una sola fuente; tómalo como orientación. '
               'Si el ROI es negativo, el modelo NO le gana a la casa en ese periodo.')

with st.expander('Detalles técnicos'):
    st.markdown("""
- **Elo** general + por superficie (mezcla 50/50), K decreciente con la experiencia, peso por nivel del torneo y
  **margen de victoria en juegos** (mejoró el log loss de validación de 0,582 a 0,572).
- **Saque/resto ajustados al rival**: residuo de puntos ganados al saque frente a lo esperado por el resto del
  rival; media móvil con vida media de 1 año y encogimiento hacia la media según los puntos disponibles.
- **Modelo de puntos (Markov)**: con la probabilidad de ganar punto al saque de cada uno se calculan juegos,
  tiebreaks, sets y partido de forma exacta. Para los mercados se ajusta la ventaja para que coincida con la
  probabilidad final del modelo y luego se recalibra cada familia de mercados.
- **Modelo final**: Gradient Boosting (simétrico) sobre Elo, saque, ranking, forma, cansancio, descanso, inactividad,
  experiencia y cara a cara. Se exporta a JSON y se evalúa con numpy.
- Histórico corregido: cuartos de final mal fechados y torneos duplicados entre archivos.
""")
    st.json(rep.get('hyperparams', {}))
    st.caption(rep['attribution'])
