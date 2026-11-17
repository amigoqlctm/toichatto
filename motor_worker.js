// ARES-X · Web Worker que ejecuta el motor Python (simulación 3D + RNA + difuso + AG + supervisor) con Pyodide.
// El panel (index.html) solo dibuja: cada botón envía un comando aquí y Python responde con el nuevo estado.
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
importScripts(PYODIDE + "pyodide.js");

let py = null, apiJson = null;
const progreso = (msg) => postMessage({ tipo: "progreso", msg });

async function arrancar() {
  progreso("Descargando el intérprete de Python (Pyodide)…");
  py = await loadPyodide({ indexURL: PYODIDE });
  py.setStdout({ batched: (msg) => postMessage({ tipo: "log", msg }) });
  py.setStderr({ batched: (msg) => postMessage({ tipo: "log", msg: "[stderr] " + msg }) });
  progreso("Instalando numpy, pandas y scikit-learn…");
  await py.loadPackage(["numpy", "pandas", "scikit-learn"]);
  progreso("Descargando el motor ARES-X (py/ares_motor.py)…");
  const codigo = await (await fetch("py/ares_motor.py", { cache: "no-cache" })).text();
  progreso("Generando datos sintéticos y entrenando la RNA en tu navegador…");
  py.runPython("display = lambda *a, **k: None");
  await py.runPythonAsync(codigo);
  progreso("Creando la misión inicial (mapa de riesgo RNA + primera ruta del AG)…");
  py.runPython("iniciar_web()");
  apiJson = py.globals.get("api_json");
  postMessage({ tipo: "listo" });
}

const listo = arrancar().catch((e) => postMessage({ tipo: "error", msg: String(e && e.message ? e.message : e) }));

onmessage = async (ev) => {
  const { id, cmd, args } = ev.data;
  await listo;
  let data;
  try {
    data = JSON.parse(apiJson(cmd, JSON.stringify(args || {})));
  } catch (e) {
    data = { error: String(e && e.message ? e.message : e) };
  }
  postMessage({ tipo: "resp", id, data });
};
