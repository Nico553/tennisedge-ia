"""Componentes de interfaz compartidos (Streamlit)."""
from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from . import store
from .predictor import Predictor

TZ = ZoneInfo('America/Santiago')
SURF_ES = {'Hard': 'Dura', 'Clay': 'Arcilla', 'Grass': 'Césped'}
SURF_EN = {v: k for k, v in SURF_ES.items()}

CSS = """
<style>
.block-container{padding-top:1.6rem;max-width:1200px}
.te-card{border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:14px 18px;margin:6px 0 10px 0;
  background:rgba(128,128,128,.05)}
.te-big{font-size:2.3rem;font-weight:800;line-height:1.1}
.te-sub{opacity:.7;font-size:.85rem}
.te-a{color:#f08c2e}.te-b{color:#3d8bfd}
.te-badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:.75rem;font-weight:600;margin-left:6px}
.te-ok{background:rgba(34,197,94,.18);color:#16a34a}.te-warn{background:rgba(234,179,8,.2);color:#b45309}
.te-bad{background:rgba(239,68,68,.18);color:#dc2626}
.te-bar{height:12px;border-radius:6px;background:linear-gradient(90deg,#f08c2e var(--p),#3d8bfd var(--p));margin:6px 0}
</style>
"""


def setup(title: str, icon: str = '🎾') -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(f'{icon} {title}')


@st.cache_resource(show_spinner='Cargando modelo y poniendo los ratings al día…')
def get_predictor() -> Predictor:
    p = Predictor()
    p.apply_results(store.load_recent())
    return p


def reload_predictor() -> None:
    get_predictor.clear()


def pct(x, d=1) -> str:
    return '—' if x is None or pd.isna(x) else f'{x*100:.{d}f}%'


def badge(ok: bool | None, yes='validado', no='experimental') -> str:
    if ok is None:
        return ''
    return f"<span class='te-badge {'te-ok' if ok else 'te-warn'}'>{'✅ ' + yes if ok else '⚠️ ' + no}</span>"


def conf_badge(conf: dict) -> str:
    cls = {'alta': 'te-ok', 'media': 'te-warn', 'baja': 'te-bad'}[conf['nivel']]
    return f"<span class='te-badge {cls}'>Confianza {conf['nivel']} · {conf['score']}/100</span>"


def stale_warning(P: Predictor) -> None:
    last = pd.Timestamp(P.data_last_date())
    days = (pd.Timestamp.now(tz=TZ).tz_localize(None).normalize() - last).days
    if days > 10:
        st.warning(f'Los resultados llegan hasta **{last.date()}** ({days} días atrás). Las predicciones pierden '
                   'precisión con datos viejos: ve a **🗃️ Datos y APIs** y actualiza, o activa la actualización '
                   'diaria automática (GitHub Actions).', icon='⏳')


def render_analysis(P: Predictor, r: dict) -> None:
    a, b = r['a'], r['b']
    f = r['features']
    m = r['markets']
    av = m['avalado']
    st.markdown(f"""<div class='te-card'>
      <div class='te-sub'>{r['tour']} · {r['level_name']} · {SURF_ES.get(r['surface'], r['surface'])} · mejor de {r['best_of']}
      · datos hasta {r['data_last']} {conf_badge(r['confidence'])}</div>
      <div style='display:flex;justify-content:space-between;align-items:flex-end;margin-top:8px'>
        <div><div class='te-big te-a'>{pct(r['p_a'],0)}</div><b>{a}</b><div class='te-sub'>
          {'Nº ' + str(int(f['rank_a'])) if f['rank_a'] else 'sin ranking'} · Elo {f['elo_a']:.0f}</div></div>
        <div style='text-align:right'><div class='te-big te-b'>{pct(r['p_b'],0)}</div><b>{b}</b><div class='te-sub'>
          {'Nº ' + str(int(f['rank_b'])) if f['rank_b'] else 'sin ranking'} · Elo {f['elo_b']:.0f}</div></div>
      </div>
      <div class='te-bar' style='--p:{r['p_a']*100:.1f}%'></div>
      <div class='te-sub'>Probabilidad de ganar según TennisEdge (modelo final). Elo solo: {pct(r['p_elo'])} ·
      modelo de saque: {pct(r['p_markov'])} para {a}.</div></div>""", unsafe_allow_html=True)
    if r['confidence']['motivos']:
        st.caption('Confianza reducida por: ' + ' · '.join(r['confidence']['motivos']))

    va, vb = r['value_a'], r['value_b']
    if va.get('ev') is not None:
        fa, fb = r['market_fair']
        c1, c2, c3 = st.columns(3)
        c1.metric(f'Cuota {a}', f"{va['cuota']:.2f}", f"EV {va['ev']*100:+.1f}%")
        c2.metric(f'Cuota {b}', f"{vb['cuota']:.2f}", f"EV {vb['ev']*100:+.1f}%")
        c3.metric('Mercado sin margen', f'{pct(fa,0)} / {pct(fb,0)}',
                  f"modelo {(r['p_a']-fa)*100:+.1f} pp", delta_color='off')
        st.caption(f"Cuota justa según el modelo: {a} {va['cuota_justa']:.2f} · {b} {vb['cuota_justa']:.2f}. "
                   'EV positivo no garantiza nada: las casas aciertan más que cualquier modelo público. '
                   'Revisa **🧠 Rendimiento** para ver cómo le ha ido al modelo frente a las cuotas.')

    st.subheader('🧠 Por qué')
    fac = pd.DataFrame(r['factors'])
    st.dataframe(fac[['factor', 'favorece', 'fuerza', 'detalle']], hide_index=True, width='stretch',
                 column_config={'fuerza': st.column_config.ProgressColumn('fuerza', min_value=0, max_value=1, format='%.2f'),
                                'factor': 'Factor', 'favorece': 'Favorece a', 'detalle': 'Dato'})

    st.subheader('📈 Mercados')
    st.caption('Todos salen del mismo modelo punto→juego→set→partido, recalibrado con datos reales. '
               '“Validado” = superó en test a la referencia ingenua (decir siempre lo más frecuente).')
    t1, t2, t3, t4 = st.tabs(['Sets', 'Juegos', 'Hándicap', 'Primer set y otros'])
    with t1:
        st.markdown('**Marcador en sets** ' + badge(av.get('straight')), unsafe_allow_html=True)
        def _lab(s):
            x, y = (int(v) for v in s.split('-'))
            return f'{a} gana {x}-{y}' if x > y else f'{b} gana {y}-{x}'
        st.dataframe(pd.DataFrame([{'Marcador': _lab(s), 'Prob.': v} for s, v in m['set_scores'].items()]),
                     hide_index=True, width='stretch',
                     column_config={'Prob.': st.column_config.ProgressColumn(min_value=0, max_value=1, format='percent')})
        c1, c2 = st.columns(2)
        c1.markdown(f"**{a}** gana al menos un set: **{pct(m['a_wins_set'])}**")
        c2.markdown(f"**{b}** gana al menos un set: **{pct(m['b_wins_set'])}**")
        st.markdown(f"Partido en sets corridos: **{pct(m['n_sets'][r['best_of']//2+1])}** "
                    + badge(av.get('straight')), unsafe_allow_html=True)
        st.caption('“Gana al menos un set” del no favorito ' + ('está validado.' if av.get('dog_set') else 'es experimental.'))
    with t2:
        st.markdown(f"Juegos esperados: **{m['expected_games']:.1f}** " + badge(av.get('totals') if r['best_of'] == 3 else False),
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{'Línea': ln, 'Más de': v, 'Menos de': 1 - v} for ln, v in m['total_over'].items()]),
                     hide_index=True, width='stretch',
                     column_config={'Más de': st.column_config.ProgressColumn(min_value=0, max_value=1, format='percent'),
                                    'Menos de': st.column_config.ProgressColumn(min_value=0, max_value=1, format='percent')})
        st.markdown('**Juegos de cada jugador** ' + badge(False), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{'Línea': ln, f'{a} más de': m['a_games_over'][ln], f'{b} más de': m['b_games_over'][ln]}
                                   for ln in m['a_games_over']]), hide_index=True, width='stretch')
    with t3:
        fav, dog = (a, b) if m['fav_is_a'] else (b, a)
        fh = m['a_handicap'] if m['fav_is_a'] else m['b_handicap']
        st.markdown(f'**Hándicap de juegos · favorito {fav}** ' + badge(av.get('fav_handicap') if r['best_of'] == 3 else False),
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{'Línea': f'−{h}', f'{fav} −{h}': v, f'{dog} +{h}': 1 - v} for h, v in fh.items()]),
                     hide_index=True, width='stretch')
    with t4:
        st.markdown('**Ganador del 1er set** ' + badge(av.get('set1_fav')), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric(a, pct(m['set1_a'])); c2.metric(b, pct(1 - m['set1_a']))
        st.markdown('**Marcador del 1er set** ' + badge(False), unsafe_allow_html=True)
        top = list(m['set1_scores'].items())[:8]
        st.dataframe(pd.DataFrame([{'Marcador (A-B)': s, 'Prob.': v} for s, v in top]), hide_index=True,
                     width='stretch')
        st.markdown(f"Habrá al menos un tiebreak: **{pct(m['tiebreak'])}** " + badge(av.get('tiebreak')),
                    unsafe_allow_html=True)

    st.subheader('📚 Historial')
    cols = st.columns(2)
    for col, name in zip(cols, (a, b)):
        with col:
            pr = P.profile(r['tour'], name, r['surface'])
            w, l = pr['record_surface']
            st.markdown(f"**{name}** · {SURF_ES.get(r['surface'])}: {w}V-{l}D (desde 2024)")
            if pr['stats']:
                stt = pr['stats']
                rows = [{'Estadística (últimos 12 meses)': k, 'Valor': (f'{v:.1f}' if 'partido' in k else pct(v))}
                        for k, v in stt.items() if k != 'partidos_con_dato']
                st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
            if isinstance(pr.get('last'), pd.DataFrame) and not pr['last'].empty:
                last = pr['last'].copy()
                last['date'] = pd.to_datetime(last['date']).dt.date
                last['ganó'] = last['ganó'].map({True: '✅', False: '❌'})
                st.dataframe(last.rename(columns={'date': 'Fecha', 'tourney_name': 'Torneo', 'round': 'Ronda',
                                                  'rival': 'Rival', 'ganó': '', 'score': 'Marcador'}),
                             hide_index=True, width='stretch')
    h = P.h2h_matches(r['tour'], a, b)
    st.markdown(f"**Cara a cara** (base desde 2024 + recientes): {f['h2h_a']}-{f['h2h_b']}")
    if not h.empty:
        h = h.copy(); h['date'] = pd.to_datetime(h['date']).dt.date
        st.dataframe(h.rename(columns={'date': 'Fecha', 'tourney_name': 'Torneo', 'surface': 'Pista', 'round': 'Ronda',
                                       'winner': 'Ganador', 'score': 'Marcador'}), hide_index=True, width='stretch')
