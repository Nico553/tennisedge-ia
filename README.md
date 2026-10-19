# TennisEdge IA · Edición web para iPhone

Prototipo experimental de análisis estadístico de tenis en Streamlit.

**NO ES UN SISTEMA DE APUESTAS SEGURAS.** Las predicciones necesitan datos reales autorizados, calibración y verificación histórica. El modo demo es completamente ficticio y no debe utilizarse para apuestas.

## Contenido mínimo del repositorio
- `app.py`: aplicación web. Genera su propio conjunto ficticio para demostrar la interfaz.
- `engine.py`: modelo Elo y comparación de mercados.
- `requirements.txt`: dependencias.

## Publicar gratis (Streamlit Community Cloud)
1. Crea un repositorio nuevo en GitHub llamado `tennisedge-ia-web` (público si es para compartir, sin información privada).
2. Coloca los **tres archivos** de esta carpeta en la **raíz** del repositorio; no subas el ZIP cerrado.
3. Accede a https://share.streamlit.io y entra con tu cuenta GitHub.
4. Selecciona *Create app* > *Yup, I have an app*; elige el repositorio, rama `main`, archivo principal `app.py`.
5. Haz clic en *Deploy* y espera a que termine. Se generará un enlace `https://...streamlit.app`.
6. Abre ese enlace desde Safari del iPhone. Para añadirlo al inicio: Safari > Compartir > Añadir a pantalla de inicio.

Puedes probar el modo de demostración sin subir datos; no representa jugadores ni partidos reales. Para probar datos propios, el menú lateral permite cargar CSV ATP/WTA autorizados. No hay integración automática con Betano ni datos en directo.

Si hay error de despliegue, revisa los registros (*Manage app* > *Logs*). No pongas claves, usuarios ni datos personales en archivos del repositorio.
