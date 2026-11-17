# ARES-X · versión web (Vercel / Netlify)

Esta carpeta es un **sitio estático**: no necesita servidor. El motor de ARES-X (simulación 3D, RNA, lógica difusa,
AG y supervisor) es **el mismo código Python del notebook**, y se ejecuta dentro del navegador con
[Pyodide](https://pyodide.org) (Python compilado a WebAssembly) en un *Web Worker*, para que la animación no se congele.

| Archivo | Qué es |
|---|---|
| `index.html` | Centro de Control. Solo dibuja; cada botón envía un comando al motor Python. |
| `motor_worker.js` | Carga Pyodide, numpy, pandas y scikit-learn; ejecuta `py/ares_motor.py`; responde los comandos. |
| `py/ares_motor.py` | Motor Python generado desde el notebook (sin gráficos matplotlib; la RNA entrena 45 épocas). |
| `vercel.json` | Configuración mínima para Vercel. |

**Primera carga:** alrededor de 1 minuto (descarga Python, entrena la RNA en el navegador). Luego el navegador lo
guarda en caché. Requiere internet, porque Pyodide se descarga desde cdn.jsdelivr.net.

## Publicar en Vercel (sin instalar nada)
1. En https://github.com, cree un repositorio nuevo (por ejemplo `ares-x`, público).
2. En el repositorio: **Add file → Upload files**. Arrastre **el contenido** de esta carpeta (`index.html`,
   `motor_worker.js`, `vercel.json`, `README.md` y la carpeta `py`) → **Commit changes**.
3. En https://vercel.com, inicie sesión con GitHub → **Add New… → Project** → **Import** el repositorio.
4. En *Framework Preset* elija **Other**, sin *Build Command* y con *Root Directory* `./` → **Deploy**.
5. Vercel le entrega una dirección del tipo `https://ares-x-xxxx.vercel.app`. Cada vez que suba cambios a GitHub,
   se vuelve a publicar sola.

Alternativa aún más rápida: https://app.netlify.com/drop → arrastre esta carpeta completa → listo.

## Probar en su computador
Abrir `index.html` con doble clic **no funciona**, porque los navegadores bloquean los workers en `file://`. Use:
```bash
cd ares-x-web
python -m http.server 8000      # y abra http://127.0.0.1:8000
```
