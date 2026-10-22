# 🎾 TennisEdge IA v5

Pronosticador de tenis **para análisis personal** (sin fines comerciales). Calcula la probabilidad de cada
partido ATP/WTA (incluye Challenger, WTA 125 e ITF), todos los mercados de sets y juegos, y los compara con
las cuotas de las casas. Funciona 100% con planes **gratuitos**.

> No es asesoría de apuestas. Ningún modelo público le gana con regularidad a las casas; la app muestra
> honestamente cuándo el modelo acierta y cuándo no.

## Qué hace

| Página | Contenido |
|---|---|
| 📅 **Partidos y valor** | Partidos próximos (RapidAPI) con probabilidad, cuotas, EV y confianza. Filtros y análisis completo. |
| 🎾 **Análisis de partido** | Cualquier enfrentamiento: probabilidad, factores, mercados, estadísticas de saque, últimos partidos, H2H. |
| 💹 **Cuotas de casas** | Mejores cuotas de varias casas (The Odds API) frente al modelo. |
| 🧠 **Rendimiento** | Acierto, Brier y calibración en test, validación de cada mercado y prueba real contra las cuotas. |
| 🗃️ **Datos y APIs** | Estado de los datos, claves, actualización manual y diagnóstico. |

## Cómo funciona el modelo

1. **Datos**: 2016 → junio 2026 de Jeff Sackmann / Tennis Abstract (~393 mil partidos con estadísticas de
   saque), corregidos (fechas de rondas y duplicados). Desde junio 2026 hasta hoy: resultados de RapidAPI.
2. **Elo** general + por superficie, con K decreciente, peso por nivel de torneo y **margen de victoria**.
3. **Saque y resto ajustados al rival** (con encogimiento según muestra y vida media de 1 año).
4. **Modelo de puntos (Markov)**: con la probabilidad de ganar punto al saque se calculan juego → tiebreak →
   set → partido de forma exacta. De ahí salen **todos** los mercados, coherentes entre sí.
5. **Modelo final**: Gradient Boosting sobre Elo, saque, ranking, forma, cansancio, descanso, inactividad,
   experiencia y H2H. Los mercados se recalibran por familia con datos de validación.
6. **Evaluación honesta**: entrenamiento hasta jul-2024, validación hasta jul-2025, test jul-2025 → jun-2026.

Resultado en test (33.837 partidos): **69,3% de acierto**, Brier 0,197 (Elo solo: 68,9% / 0,199).
Seis familias de mercados superan la referencia ingenua (sets corridos, 1er set, totales, hándicap,
"no favorito gana un set", tiebreak).

## Claves de API (gratis)

| Variable | Dónde obtenerla | Plan gratis | Uso |
|---|---|---|---|
| `RAPIDAPI_TENNIS_KEY` | [Tennis API ATP/WTA/ITF](https://rapidapi.com/jjrm365-kIFr3Nx_odV/api/tennis-api-atp-wta-itf) → Pricing → Basic | 50 llamadas/día | Partidos próximos + resultados diarios |
| `THE_ODDS_API_KEY` | [The Odds API](https://the-odds-api.com/#get-access) | 500 créditos/mes | Cuotas de muchas casas |

**Dónde ponerlas:**

1. **Render** → tu servicio → **Environment** → *Add Environment Variable* → agrega las dos → *Save, rebuild and deploy*.
2. **GitHub** (para la actualización diaria automática) → repositorio → **Settings → Secrets and variables →
   Actions → New repository secret** → nombre `RAPIDAPI_TENNIS_KEY`, misma clave.

Nunca las pongas en el código.

## Actualización diaria automática

`.github/workflows/actualizar-resultados.yml` corre cada día (~07:17 Chile): descarga los resultados nuevos
(máx. 40 llamadas), los guarda en `data/recent_results.csv` y hace commit. Render redespliega solo y la app
queda al día. La primera vez tarda 2–3 días en ponerse al día desde junio 2026 (por el límite de 50 llamadas).
Puedes lanzarlo a mano en la pestaña **Actions → Actualizar resultados de tenis → Run workflow**.

## Despliegue en Render

- Build: `pip install -r requirements.txt`
- Start: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`

El modelo ya viene entrenado en `data/` (la app no entrena en Render, solo carga, ~2 s).

## Reentrenar (opcional, en tu PC)

```bash
pip install -r requirements-build.txt
python scripts/build_model.py          # ~10 min; descarga el histórico y regenera data/
python -m pytest -q tests
```

## Estructura

```
app.py                     navegación
pages/                     páginas de la app
tennisedge/
  markov.py                punto → juego → set → partido y todos los mercados
  ratings.py               Elo, saque/resto, forma, cansancio, H2H (cronológico, sin fuga)
  model.py                 características, Gradient Boosting exportado a JSON, calibración de mercados
  predictor.py             carga, actualización con resultados recientes, análisis completo
  data_archive.py          histórico Sackmann (con correcciones)
  rapid.py / odds.py       conectores RapidAPI y The Odds API
  store.py                 resultados recientes (repo + caché)
scripts/build_model.py     entrenamiento y validación
scripts/update_results.py  actualización diaria (GitHub Actions)
data/                      modelo entrenado y datos derivados
```

## Licencias

Datos históricos: Jeff Sackmann / Tennis Abstract, **CC BY-NC-SA 4.0** (copia de
Aneeshers/tennis-sackmann-archive). Los archivos derivados en `data/` se comparten bajo la misma licencia,
con atribución y solo para uso no comercial.
