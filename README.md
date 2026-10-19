# TennisEdge IA v0.3 — históricos y comparación de modelos

Aplicación experimental de investigación deportiva **sin apuestas automáticas**.

## Funciones

- 6 a 10 años de ATP y WTA con atribución y uso permitido no comercial. **La fuente NO es oficial de ATP/WTA** y no es un feed en directo; archivo de junio de 2026.
- Modelos: Elo por superficie, regresión logística, Random Forest, HistGradientBoosting.
- Entrenamiento con fechas previas al partido, mismo día congelado y particiones temporales entrenamiento/validación/test. Selección de modelo con Brier de validación, evaluación con test separado.
- Mercados: ganador, ganar un set, hándicap de juegos -2.5 y total +22.5 mediante modelo separado.
- Escáner de cuotas observadas en CSV; capturas de Betano con OCR **experimental** a petición del usuario (Tesseract.js en el navegador) y verificación humana. No scraping ni apuestas automáticas.
- Filtro de lesiones manual, muestra insuficiente y antigüedad de datos. **No muestra recomendaciones seguras.**

## Despliegue en Render

1. Reemplazar/añadir archivos en la raíz de `Nico553/tennisedge-ia`: `app.py`, `engine.py`, `model_lab.py`, `data_sources.py`, `requirements.txt`, `README.md`.
2. Render desplegará automáticamente cambios en rama `main`.
3. Arranque: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`.
4. URL existente: `https://tennisedge-ia.onrender.com`.

## Licencias y datos

Archivo histórico de Jeff Sackmann / Tennis Abstract, CC BY-NC-SA 4.0. Copia [Aneeshers/tennis-sackmann-archive](https://github.com/Aneeshers/tennis-sackmann-archive), procedente de snapshot de junio 2026. Permitir solo investigación privada/no comercial con atribución. No se redistribuyen datasets históricos en este repositorio: se descargan de la fuente original de archivo solamente después de que el usuario acepte la licencia; la calidad y cobertura depende de los archivos accesibles. Consultar la licencia antes de publicar productos comerciales.

Los feeds oficiales licenciados de ATP/WTA requieren acceso de datos con permisos; [Sportradar Tennis](https://developer.sportradar.com/tennis/reference/overview) permite APIs con autenticación.

El sitio [Tennis-Data](https://www.tennis-data.co.uk/data.php) prohíbe entrenamiento de IA con sus archivos sin permiso; **no se usa**.

No hay integración API de Betano; nunca se automatiza acceso, sesión, scraping protegido, CAPTCHA, colocación o ejecución de apuestas. Cuotas introducidas por el usuario son referencias no verificadas.

## Calidad

Ejecutar `python -m pytest tests -q` con `pytest` instalado. Las pruebas sintéticas sirven para integridad del código, no para demostrar poder predictivo. En producción, la app reporta la fecha máxima de su archivo y métricas reales al cargar históricas.

## Calendario ATP/WTA oficial opcional

El módulo `official_feed.py` está listo para usar Sportradar Tennis v3, pero **requiere una suscripción/acceso legal vigente**. En Render se configura la variable secreta `SPORTRADAR_TENNIS_API_KEY` mediante Dashboard → Environment; nunca se publica ni comparte aquí. La app permite consultar el calendario autorizado (no cuotas de Betano). Más información: [Sportradar Tennis Daily Summaries](https://developer.sportradar.com/tennis/reference/daily-summaries).

## V0.4 — Consultor AUTOMÁTICO de cuotas de tenis (otras casas)

El panel **📡 Cuotas automáticas** está visible desde que cargas la web, incluso si todavía no has entrenado el modelo histórico. Extrae de **The Odds API** (`the-odds-api.com`) precios publicados por las casas disponibles en esa API, con torneo, hora de inicio, nombre de la casa, mercado, línea exacta y última actualización de cada cuota. No recibe información de Betano Chile ni la presenta como tal. Se muestra solo mercado realmente presente; hándicap de juegos y totales pueden faltar.

### Activación paso a paso

1. Registra una clave en [The Odds API](https://the-odds-api.com/#get-access) (dominio con guiones). Su página de precios indica un plan de **500 créditos/mes** gratuito, sujeto a cambios por el proveedor.
2. En [tu servicio Render](https://dashboard.render.com/web/srv-dbb3g9mi0phs73bc97tg), ve a **Environment** > **Add Environment Variable**.
3. **Key**: `THE_ODDS_API_KEY`. **Value**: pega la clave secreta. Pulsa **Save, deploy**. **No la guardes en GitHub ni la envíes por chat**.
4. Abre [TennisEdge IA](https://tennisedge-ia.onrender.com). Selecciona un torneo activo y una región de casas. Los precios se consultan automáticamente; la caché de Streamlit dura 30 minutos por torneo/región/mercados.
5. Los mercados `h2h`, `spreads`, `totals` se consultan **solo cuando el proveedor los ofrece**. Elige primero `Ganador` y región `eu` para gastar menos créditos. Un mercado x una región suele consumir 1 crédito; tres mercados x una región usan 3 créditos. La lista de deportes no consume créditos.

En la pantalla se puede filtrar el intervalo de cuotas (por defecto 1,30–1,60), frescura máxima, número de casas y ver los precios originales. La estimación *consenso sin margen* requiere 3 casas con mercados de dos resultados y representa la **mediana de probabilidades normalizadas del mercado**, no la probabilidad calibrada del modelo ni una apuesta recomendable. El software **no apuesta, no accede a cuentas, no automatiza la web de Betano y no evade restricciones regionales**.

> **Limitaciones:** sin API key no hay consulta en vivo; el módulo de históricos y los modelos requieren su propia carga de datos; las cuotas del proveedor no tienen por qué coincidir con las de Betano. Render Free puede suspender el servicio cuando no se usa, y el tablero actualiza precios al visitarlo (caché), **no realiza vigilancia permanente en segundo plano**.

Documentación: [tenis](https://the-odds-api.com/sports/tennis-odds.html) · [API v4](https://the-odds-api.com/liveapi/guides/v4/) · [términos](https://the-odds-api.com/terms-and-conditions.html).
