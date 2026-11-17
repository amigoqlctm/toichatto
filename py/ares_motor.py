"""ARES-X · motor Python para la versión web (se ejecuta en el navegador con Pyodide).
Generado automáticamente desde el notebook ARES_X_v03_CentroControl.ipynb: es el MISMO código
(simulación 3D, RNA, lógica difusa, AG, supervisor). Solo cambia: sin gráficos matplotlib y 45 épocas de RNA."""



# # ARES-X · Rover autónomo de exploración planetaria **en 3D**
# ### Notebook v0.2: Paso 1 (simulador 3D) + Paso 2 (RNA, lógica difusa y AG)
#
# **Mundo 3D:** el terreno es una superficie con relieve (mapa de alturas `z = elevación(fila, col)` en metros)
# con colinas, cráteres y un barranco que es una depresión real. El rover tiene **posición 3D (x, y, z)**,
# **cabeceo** (*pitch*, inclinación hacia adelante) y **alabeo** (*roll*, inclinación lateral). Puede **volcar**
# si supera el límite de inclinación, y gasta más energía al **subir** que al bajar. El LIDAR mide el **perfil
# de alturas** de las próximas 3 celdas, y la visualización principal es una **superficie 3D**.
#
# | Clase | Qué hace |
# |---|---|
# | `PlanetProfile` | Perfil de cada planeta y validación rover/sonda (CA-07) |
# | `World3D` | Terreno 3D con colinas, cráteres, barranco, rocas, refugios, sitios científicos y cajas de instrumentación; capa de nubes en modo sonda |
# | `Rover` + `Physics` | Estado 3D, energía según desnivel y gravedad, vuelco, día/noche, temperatura |
# | `SensorSuite` | Sensores con ruido (LIDAR de perfil, inclinómetro pitch/roll…) y validación |
# | `MissionLogger` | Traza por paso → CSV/JSON |
# | `Simulation` | Ciclo observar → validar → decidir → actuar → registrar |
# | `NeuralPerception` | **RNA**: clasifica la celda de adelante en libre / obstáculo / barranco |
# | `FuzzyController` | **Lógica difusa** Mamdani: velocidad 0–1 y nivel de cautela |
# | `GeneticPlanner` | **AG**: orden de visitas (viajero) + muestras (mochila) + caminos A* (laberinto) |
#
# **Supuestos de simulación:** los valores de daño, energía, ruido, radiación, relieve y velocidades son hipótesis
# pedagógicas ajustables, **no** especificaciones NASA. Los valores planetarios de referencia (temperatura, presión,
# gravedad) son aproximaciones tomadas de NASA Science y del NASA Planetary Fact Sheet.

# Dependencias: todas vienen preinstaladas en Colab gratuito (CPU). Plotly es opcional (vista 3D interactiva).
import json, math, copy, heapq, time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
try:                         # matplotlib solo se usa para los gráficos del notebook (la versión web no lo necesita)
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (activa proyección 3D)
    MOSTRAR_GRAFICOS = True
except ImportError:
    plt = None
    MOSTRAR_GRAFICOS = False

try:
    import plotly.graph_objects as go
    HAY_PLOTLY = True
except ImportError:
    HAY_PLOTLY = False

try:
    display
except NameError:            # fuera de Jupyter/Colab
    display = print

VERSION = "ARES-X v0.2 (3D, H1+H2)"
CELL_M = 5.0            # supuesto: cada celda mide 5 m × 5 m en planta
LIM_VUELCO_DEG = 30.0   # supuesto: inclinación máxima antes de volcar

LIBRE, OBSTACULO, BARRANCO = 0, 1, 2
CLASES = ["libre", "obstaculo", "barranco"]
DIRS = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "O": (0, -1)}
LATERAL = {"N": (0, 1), "S": (0, 1), "E": (1, 0), "O": (1, 0)}   # eje perpendicular al rumbo

# ## 1. `PlanetProfile`

@dataclass
class PlanetProfile:
    nombre: str
    gravedad: float            # m/s² (referencia NASA, aprox.)
    temp_dia: float            # °C exterior típica de día (aprox.)
    temp_noche: float          # °C exterior típica de noche (aprox.)
    presion_bar: float         # bar (aprox.)
    superficie_solida: bool
    factor_solar: float        # supuesto: irradiancia relativa en superficie
    radiacion_base: float      # supuesto: mSv/h
    relieve_m: float = 4.0     # supuesto: amplitud de colinas (m)
    n_crateres: int = 1        # supuesto: cráteres por mapa
    eventos: list = field(default_factory=list)
    nota: str = ""

    def modos_permitidos(self):
        return ["rover", "sonda"] if self.superficie_solida else ["sonda"]

    def validar_modo(self, modo):
        """Devuelve (ok, mensaje). Implementa CA-07."""
        if modo not in ("rover", "sonda"):
            return False, f"Modo '{modo}' desconocido. Use 'rover' o 'sonda'."
        if modo == "rover" and not self.superficie_solida:
            return False, (f"{self.nombre} es un planeta gigante sin superficie sólida accesible: "
                           f"un rover de ruedas no tiene dónde rodar. Use el modo 'sonda' (vehículo atmosférico virtual).")
        if modo == "sonda" and self.superficie_solida:
            return False, (f"En {self.nombre} ARES-X opera como rover de superficie. El modo sonda atmosférica está "
                           f"reservado a los gigantes gaseosos y helados (Júpiter, Saturno, Urano y Neptuno).")
        return True, f"Modo {modo} válido para {self.nombre}."


PLANETAS = {
    "Mercurio": PlanetProfile("Mercurio", 3.7, 430, -180, 0.0, True, 6.7, 0.30, 2.5, 3,
                              ["noche", "fallo_sensor", "baja_conectividad", "meteorito"],
                              "Vacío casi total; contraste térmico extremo; muchos cráteres."),
    "Venus":    PlanetProfile("Venus", 8.87, 464, 464, 92.0, True, 0.05, 0.01, 2.0, 1,
                              ["corrosion", "fallo_sensor", "baja_conectividad"],
                              "≈464 °C, ≈92 bar, nubes de ácido sulfúrico. Rover especial hipotético, misión breve."),
    "Tierra":   PlanetProfile("Tierra", 9.81, 20, 10, 1.0, True, 1.0, 0.0003, 3.0, 0,
                              ["sismo", "noche", "fallo_sensor"], "Banco de calibración."),
    "Marte":    PlanetProfile("Marte", 3.71, -20, -80, 0.006, True, 0.43, 0.03, 3.5, 2,
                              ["tormenta", "sismo", "noche", "meteorito", "fallo_sensor", "baja_conectividad"],
                              "Frío, baja presión, atmósfera de CO₂, polvo."),
    "Júpiter":  PlanetProfile("Júpiter", 24.79, -110, -110, 1.0, False, 0.04, 1.0, 1.0, 0,
                              ["rafaga", "baja_conectividad"], "Gigante gaseoso, sin suelo accesible estable."),
    "Saturno":  PlanetProfile("Saturno", 10.44, -140, -140, 1.0, False, 0.01, 0.1, 1.0, 0,
                              ["rafaga", "baja_conectividad"], "Gigante gaseoso, sin suelo accesible estable."),
    "Urano":    PlanetProfile("Urano", 8.87, -195, -195, 1.0, False, 0.003, 0.05, 1.0, 0,
                              ["rafaga", "baja_conectividad"], "Gigante helado, sin suelo sólido de rodadura."),
    "Neptuno":  PlanetProfile("Neptuno", 11.15, -200, -200, 1.0, False, 0.001, 0.05, 1.0, 0,
                              ["rafaga", "baja_conectividad"], "Gigante helado, sin suelo sólido de rodadura."),
}
for p in ["Marte", "Júpiter"]:
    print(p, "rover →", PLANETAS[p].validar_modo("rover"))

# ## 2. `World3D`: terreno 3D reproducible
#
# - **Colinas:** ruido suavizado con amplitud según el planeta.
# - **Cráteres:** cuencos con borde elevado (más en Mercurio y Marte).
# - **Barranco:** franja serpenteante con una **caída de 3,5 a 10 m** (profundidad variable). Es mortal si el rover entra.
# - **Rocas** (obstáculos): bloques de 0,6 a 2 m de alto, no transitables.
# - **Cajas de instrumentación** (I1, I2) de una misión anterior: la misión pide recuperarlas. Valen 5 puntos, pesan
#   1 kg y traen una batería de respaldo (+5 %). El AG las trata como objetivos más de la mochila.
# - **Modo sonda (gigantes):** el mismo generador crea una **capa de nubes** a 1 bar. Las "rocas" pasan a ser
#   **torres de tormenta** (cumulonimbos) y el "barranco" pasa a ser una **corriente descendente** que hunde a la sonda
#   hacia presiones aplastantes. Las cajas de instrumentación no existen (no hay suelo).
# - **DEM orbital:** una copia **ruidosa** de la altimetría. Es lo que "sabe" el rover antes de explorar,
#   y el AG planifica con este mapa estimado, no con la verdad del terreno.

def _suavizar(a, k):
    p = k // 2
    pad = np.pad(a, p, mode="reflect")
    out = np.zeros_like(a)
    for i in range(k):
        for j in range(k):
            out += pad[i:i + a.shape[0], j:j + a.shape[1]]
    return out / (k * k)


class World3D:
    def __init__(self, planeta: PlanetProfile, tam=20, semilla=42, dificultad="media",
                 n_sitios=4, n_refugios=2, modo="rover", n_instrumentos=None):
        if not isinstance(tam, (int, np.integer)) or not 8 <= tam <= 60:
            raise ValueError("Tamaño de mapa inválido: use un entero entre 8 y 60.")
        if dificultad not in ("facil", "media", "dificil"):
            raise ValueError("Dificultad debe ser 'facil', 'media' o 'dificil'.")
        if not planeta.superficie_solida and modo != "sonda":
            raise ValueError(f"{planeta.nombre} no tiene superficie sólida para generar terreno.")
        self.planeta, self.tam, self.semilla, self.dificultad = planeta, int(tam), int(semilla), dificultad
        self.modo = modo
        self._sal = sum(map(ord, planeta.nombre)) if modo == "sonda" else 0   # cada gigante tiene su propia atmósfera
        self.n_sitios, self.n_refugios = n_sitios, n_refugios
        # Cajas de instrumentación para recuperar (solo rover). En modo sonda no hay suelo donde dejarlas.
        self.n_instrumentos = (2 if modo == "rover" else 0) if n_instrumentos is None else n_instrumentos
        self.zonas_prohibidas = set()
        self._generar()

    # ---------- generación ----------
    def _generar(self):
        for intento in range(50):
            rng = np.random.default_rng(self.semilla + 1000 * intento + self._sal)
            self._generar_una_vez(rng)
            if self._mapa_valido():
                self.dem_orbital = self.elevacion + rng.normal(0, 0.4, self.elevacion.shape)  # altimetría con ruido
                return
        raise RuntimeError("No se pudo generar un mapa válido; cambie semilla o dificultad.")

    def _generar_una_vez(self, rng):
        n, pl = self.tam, self.planeta
        base = _suavizar(rng.normal(0, 1, (n, n)), 5)
        detalle = _suavizar(rng.normal(0, 1, (n, n)), 3)
        elev = base / (np.abs(base).max() + 1e-9) * pl.relieve_m + detalle / (np.abs(detalle).max() + 1e-9) * 0.4
        filas, cols = np.mgrid[0:n, 0:n]
        self.crateres = []
        for _ in range(pl.n_crateres):
            cr, cc = rng.integers(2, n - 2, 2)
            rad, prof = float(rng.uniform(2.5, 4)), float(rng.uniform(1.5, 2.5))
            d = np.hypot(filas - cr, cols - cc)
            elev -= np.where(d < rad, prof * (1 - (d / rad) ** 2), 0)
            elev += np.where((d >= rad) & (d < rad + 1.2), prof * 0.35, 0)
            self.crateres.append({"centro": (int(cr), int(cc)), "radio_celdas": round(rad, 2), "prof_m": round(prof, 2)})
        self.terreno = np.full((n, n), LIBRE, dtype=int)

        prob_obs = {"facil": 0.05, "media": 0.09, "dificil": 0.14}[self.dificultad]
        largo = {"facil": n // 3, "media": n // 2, "dificil": int(n * 0.7)}[self.dificultad]
        vertical = rng.random() < 0.5
        fija = int(rng.integers(n // 3, 2 * n // 3))
        ini = int(rng.integers(0, n - largo))
        for t in range(ini, ini + largo):
            nueva = int(np.clip(fija + rng.integers(-1, 2), 2, n - 3))
            celdas = [(t, fija), (t, nueva)] if vertical else [(fija, t), (nueva, t)]  # sin huecos diagonales
            for (r, c) in celdas:
                if self.terreno[r, c] != BARRANCO:
                    self.terreno[r, c] = BARRANCO
                    elev[r, c] -= float(rng.uniform(3.5, 10.0))   # profundidad variable (supuesto)
            fija = nueva

        mask = (rng.random((n, n)) < prob_obs) & (self.terreno == LIBRE)
        self.terreno[mask] = OBSTACULO
        elev[mask] += rng.uniform(0.6, 2.0, mask.sum())            # rocas de altura variable (supuesto)
        self.elevacion = elev

        libres = list(zip(*np.where(self.terreno == LIBRE)))
        idx = rng.permutation(len(libres))
        n_base = 1 + self.n_refugios + self.n_sitios
        elegidas = [tuple(map(int, libres[i])) for i in idx[: n_base + self.n_instrumentos]]
        self.inicio = elegidas[0]
        self.refugios = elegidas[1:1 + self.n_refugios]
        self.sitios = [{
            "id": f"S{k + 1}", "pos": pos, "valor": int(rng.integers(1, 11)),
            "masa_kg": round(float(rng.uniform(0.5, 4.0)), 2), "recolectado": False,
            "biofirma_simulada": bool(rng.random() < 0.15),   # solo señal anómala; nunca "vida"
            "tipo": "ciencia",
        } for k, pos in enumerate(elegidas[1 + self.n_refugios:n_base])]
        # Instrumentación de una misión anterior: vale 5 puntos de misión, pesa 1 kg y trae una batería de respaldo (+5 %)
        self.sitios += [{"id": f"I{k + 1}", "pos": pos, "valor": 5, "masa_kg": 1.0, "recolectado": False,
                         "biofirma_simulada": False, "tipo": "instrumento"} for k, pos in enumerate(elegidas[n_base:])]

    def _mapa_valido(self):
        alc = self.alcanzables_desde(self.inicio)
        return any(r in alc for r in self.refugios)

    # ---------- consultas ----------
    def dentro(self, pos):
        return 0 <= pos[0] < self.tam and 0 <= pos[1] < self.tam

    def z(self, pos):
        return float(self.elevacion[pos])

    def transitable(self, pos):
        return self.dentro(pos) and self.terreno[pos] == LIBRE and pos not in self.zonas_prohibidas

    def vecinos(self, pos):
        for d, (dr, dc) in DIRS.items():
            q = (pos[0] + dr, pos[1] + dc)
            if self.dentro(q):
                yield d, q

    def alcanzables_desde(self, pos):
        vistos, cola = {pos}, deque([pos])
        while cola:
            p = cola.popleft()
            for _, q in self.vecinos(p):
                if q not in vistos and self.transitable(q) and self.pitch_deg(p, q) < LIM_VUELCO_DEG:
                    vistos.add(q); cola.append(q)
        return vistos

    def pitch_deg(self, a, b):
        """Cabeceo al ir de a → b (ángulo de la pendiente en la dirección de avance)."""
        return float(np.degrees(np.arctan(abs(self.elevacion[b] - self.elevacion[a]) / CELL_M)))

    def roll_deg(self, pos, rumbo):
        """Alabeo en pos: diferencia de altura entre las celdas laterales al rumbo."""
        dr, dc = LATERAL[rumbo]
        a = (pos[0] - dr, pos[1] - dc); b = (pos[0] + dr, pos[1] + dc)
        # Rocas y barrancos son rasgos discretos, no inclinan el chasis: se usa la altura propia
        za = self.elevacion[a] if self.dentro(a) and self.terreno[a] == LIBRE else self.elevacion[pos]
        zb = self.elevacion[b] if self.dentro(b) and self.terreno[b] == LIBRE else self.elevacion[pos]
        return float(np.degrees(np.arctan(abs(zb - za) / (2 * CELL_M))))

    def distancia_peligro(self, pos, rumbo, max_m=20.0, incluir_barrancos=True):
        """Distancia real al primer peligro adelante. Con incluir_barrancos=False solo cuenta lo que
        SOBRESALE del suelo (rocas, escombros de meteorito): es lo que puede ver un sensor de proximidad."""
        dr, dc = DIRS[rumbo]
        for k in range(1, int(max_m / CELL_M) + 1):
            q = (pos[0] + dr * k, pos[1] + dc * k)
            if not self.dentro(q):
                return max_m if not incluir_barrancos else (k - 0.5) * CELL_M
            t = self.terreno[q]
            if t == OBSTACULO or q in self.zonas_prohibidas or (incluir_barrancos and t == BARRANCO):
                return (k - 0.5) * CELL_M
        return max_m

    # ---------- edición manual (RF-02) ----------
    def agregar_barranco(self, pos):
        if not self.dentro(pos):
            return False, "Fuera del mapa."
        if self.terreno[pos] != BARRANCO:
            self.terreno[pos] = BARRANCO
            self.elevacion[pos] -= 10.0
            self.dem_orbital[pos] -= 10.0
        return True, f"Barranco agregado en {pos}."

    def agregar_obstaculo(self, pos):
        if not self.dentro(pos):
            return False, "Fuera del mapa."
        self.terreno[pos] = OBSTACULO
        self.elevacion[pos] += 1.5
        return True, f"Obstáculo agregado en {pos}."

    def resumen(self):
        return {"planeta": self.planeta.nombre, "tam": self.tam, "semilla": self.semilla,
                "dificultad": self.dificultad, "inicio": self.inicio, "refugios": self.refugios,
                "sitios": self.sitios, "crateres": self.crateres,
                "z_min_m": round(float(self.elevacion.min()), 2), "z_max_m": round(float(self.elevacion.max()), 2)}

# ## 3. `Rover` y `Physics` en 3D
#
# Supuestos de simulación:
# - Distancia recorrida 3D: `√(5² + Δz²)` m.
# - Energía por celda: `(0,5·√g_rel·d3D/5 + 0,06·g_rel·max(Δz,0)) · (0,6 + 0,4·velocidad)` % de batería.
#   Subir cuesta más, y en planetas con más gravedad, también.
# - **Vuelco** si el cabeceo o el alabeo superan 30°. Es fatal para la misión.
# - Consumo base: 0,1 %/paso; calefacción si hace menos de −30 °C: +0,3 %/paso. Carga solar solo de día.
# - **La batería nunca baja de 0.** En un refugio se recarga 5 %/paso.

@dataclass
class Rover:
    pos: tuple
    rumbo: str = "E"
    bateria: float = 100.0
    temp_interna: float = 20.0
    salud: float = 100.0
    odometro_m: float = 0.0
    carga_kg: float = 0.0
    capacidad_kg: float = 8.0
    muestras: list = field(default_factory=list)
    estado: str = "CONFIGURACION"
    vivo: bool = True


class Physics:
    TEMP_INT_MIN, TEMP_INT_MAX = -40.0, 85.0

    def __init__(self, planeta: PlanetProfile, pasos_por_sol=40, modo="rover"):
        self.p, self.pasos_por_sol, self.modo = planeta, pasos_por_sol, modo

    def es_de_dia(self, paso, noche_forzada=False):
        return False if noche_forzada else (paso % self.pasos_por_sol) < self.pasos_por_sol / 2

    def temp_exterior(self, paso, noche_forzada=False):
        return self.p.temp_dia if self.es_de_dia(paso, noche_forzada) else self.p.temp_noche

    def costo_movimiento(self, dz, velocidad=1.0):
        if self.modo == "sonda":      # vehículo aerostático: flota, el costo no depende de la gravedad (supuesto)
            return 0.35 * math.hypot(CELL_M, dz) / CELL_M * (0.6 + 0.4 * velocidad)
        g_rel = self.p.gravedad / 9.81
        d3 = math.hypot(CELL_M, dz)
        return (0.5 * math.sqrt(g_rel) * d3 / CELL_M + 0.06 * g_rel * max(dz, 0.0)) * (0.6 + 0.4 * velocidad)

    def recarga_solar(self, de_dia, visibilidad):
        return 0.5 * self.p.factor_solar * visibilidad if de_dia else 0.0

    def actualizar_temperatura(self, rover, t_ext):
        objetivo = 20.0 + 0.15 * (t_ext - 20.0)
        rover.temp_interna += 0.3 * (objetivo - rover.temp_interna)
        return rover.temp_interna

# ## 4. `SensorSuite`: sensores imperfectos en 3D
#
# | Sensor | Variable | Unidad / rango | Uso |
# |---|---|---|---|
# | Proximidad frontal | `proximidad_m` | m, 0–20 | distancia a lo que **sobresale** adelante (rocas); **no ve barrancos**; 6 % ecos falsos y 6 % omisiones |
# | LIDAR de perfil | `lidar_dz1..3` | m, ±20 | desnivel de las 3 celdas siguientes; con polvo pierde retornos y se vuelve muy ruidoso |
# | Inclinómetro | `pitch_deg`, `roll_deg` | °, 0–45 | cabeceo hacia adelante y alabeo lateral |
# | Rugosidad (LIDAR) | `rugosidad` | m, 0–10 | desviación de alturas alrededor de la celda de adelante |
# | Acelerómetro | `vibracion` | 0–1 | sismos y falsas alarmas |
# | Altímetro | `altitud_m` | m | z del rover con ruido |
# | Termómetros | `temp_ext_c`, `temp_int_c` | °C | protección térmica |
# | Dosímetro | `radiacion_msv_h` | mSv/h | resguardo de electrónica |
# | Barómetro | `presion_bar` | bar | compatibilidad planetaria |
# | Partículas | `visibilidad` | 0–1 | polvo; empeora el LIDAR y la cámara |
# | Cámara | `textura` | 0–1 | rasgo sintético del suelo |
#
# `rasgos_percepcion()` es la **única** función que genera las lecturas de percepción. La usan la simulación
# y también el generador de datos de la RNA, así que la red se entrena con exactamente el mismo tipo de dato
# que verá en la misión.

RASGOS_RNA = ["proximidad_m", "lidar_dz1", "lidar_dz2", "lidar_dz3", "pitch_deg",
              "rugosidad", "textura", "vibracion", "visibilidad"]


def rasgos_percepcion(world, pos, rumbo, rng, vis=1.0, sismo=0.0, ruido=1.0):
    dr, dc = DIRS[rumbo]
    z0 = world.elevacion[pos]
    # Proximidad (ultrasonido/radar): solo detecta lo que sobresale; NO ve barrancos. Tiene ecos falsos y omisiones.
    prox = world.distancia_peligro(pos, rumbo, incluir_barrancos=False) + rng.normal(0, 1.5 * ruido)
    u = rng.random()
    if u < 0.06 * ruido:
        prox = rng.uniform(0, 20)            # eco falso
    elif u < 0.12 * ruido:
        prox = 20.0                          # omisión (no detecta)
    rasgos = {"proximidad_m": prox}
    # LIDAR: el polvo degrada fuertemente la medición (ruido ∝ 1/vis²) y puede perder el retorno
    ruido_lidar = 0.5 * ruido / max(vis, 0.2) ** 2
    for k in (1, 2, 3):
        q = (pos[0] + dr * k, pos[1] + dc * k)
        dz = (world.elevacion[q] - z0) if world.dentro(q) else -20.0
        if rng.random() < 0.4 * (1 - vis):
            dz = 0.0                         # sin retorno: el LIDAR reporta "plano"
        rasgos[f"lidar_dz{k}"] = dz + rng.normal(0, ruido_lidar)
    d = (pos[0] + dr, pos[1] + dc)
    if world.dentro(d):
        rasgos["pitch_deg"] = world.pitch_deg(pos, d) + rng.normal(0, 1.5 * ruido)
        r0, r1, c0, c1 = max(d[0] - 1, 0), min(d[0] + 2, world.tam), max(d[1] - 1, 0), min(d[1] + 2, world.tam)
        rasgos["rugosidad"] = float(world.elevacion[r0:r1, c0:c1].std()) + abs(rng.normal(0, 0.3 * ruido))
        textura_real = {LIBRE: 0.35, OBSTACULO: 0.75, BARRANCO: 0.15}[int(world.terreno[d])]
    else:
        rasgos["pitch_deg"], rasgos["rugosidad"], textura_real = 45.0, 0.0, 0.0
    rasgos["textura"] = textura_real + rng.normal(0, 0.25 * ruido / max(vis, 0.2))
    rasgos["vibracion"] = 0.05 + sismo + abs(rng.normal(0, 0.03 * ruido))
    rasgos["visibilidad"] = vis + rng.normal(0, 0.03 * ruido)
    return rasgos


class SensorSuite:
    RANGOS = {
        "proximidad_m": (0.0, 20.0), "lidar_dz1": (-20.0, 20.0), "lidar_dz2": (-20.0, 20.0),
        "lidar_dz3": (-20.0, 20.0), "pitch_deg": (0.0, 45.0), "roll_deg": (0.0, 45.0),
        "rugosidad": (0.0, 10.0), "textura": (0.0, 1.0), "vibracion": (0.0, 1.0), "visibilidad": (0.0, 1.0),
        "altitud_m": (-100.0, 100.0), "temp_ext_c": (-273.0, 600.0), "temp_int_c": (-100.0, 200.0),
        "radiacion_msv_h": (0.0, 100.0), "presion_bar": (0.0, 200.0), "bateria_pct": (0.0, 100.0),
    }
    CRITICOS = ("proximidad_m", "lidar_dz1")

    def __init__(self, semilla=0, ruido=1.0):
        self.rng = np.random.default_rng(semilla)
        self.ruido = ruido
        self.fallos = {}
        self.deriva_odometria = np.array([0.0, 0.0])

    def leer(self, world, rover, physics, paso, ctx, rumbo=None):
        r, s = self.rng, self.ruido
        rumbo = rumbo or rover.rumbo
        lect = rasgos_percepcion(world, rover.pos, rumbo, r, ctx.get("visibilidad", 1.0), ctx.get("sismo", 0.0), s)
        lect.update({
            "roll_deg": world.roll_deg(rover.pos, rumbo) + r.normal(0, 1.0 * s),
            "altitud_m": world.z(rover.pos) + r.normal(0, 0.3 * s),
            "temp_ext_c": max(-273.0, physics.temp_exterior(paso, ctx.get("noche_forzada", False)) + ctx.get("temp_offset", 0.0)) + r.normal(0, 2 * s),
            "temp_int_c": rover.temp_interna + r.normal(0, 0.3 * s),
            "radiacion_msv_h": world.planeta.radiacion_base * (1 + abs(r.normal(0, 0.1 * s))),
            # En modo sonda la presión crece al descender: p = p0·e^(−z/3) (escala de altura simulada)
            "presion_bar": world.planeta.presion_bar * (math.exp(-world.z(rover.pos) / 3.0) if world.modo == "sonda" else 1.0)
                           * (1 + r.normal(0, 0.01 * s)),
            "bateria_pct": rover.bateria,
        })
        self.deriva_odometria += r.normal(0, 0.02 * s, 2)
        lect["pos_estimada"] = (rover.pos[0] + self.deriva_odometria[0], rover.pos[1] + self.deriva_odometria[1])
        for sensor, tipo in self.fallos.items():
            if sensor in lect:
                lect[sensor] = np.nan if tipo == "nan" else self.RANGOS[sensor][1]
        return lect

    def validar(self, lect):
        """Limpia lecturas → (limpias, banderas, critico_fallido). Nunca lanza excepción."""
        limpias, banderas = {}, []
        for k, v in lect.items():
            if k == "pos_estimada" or k not in self.RANGOS:
                limpias[k] = v; continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = float("nan")
            if math.isnan(v) or math.isinf(v):
                banderas.append(f"NAN_{k}"); limpias[k] = np.nan; continue
            lo, hi = self.RANGOS[k]
            if v < lo or v > hi:
                banderas.append(f"FUERA_RANGO_{k}")
            limpias[k] = float(np.clip(v, lo, hi))
        if limpias.get("proximidad_m", 20) < 3 and abs(limpias.get("lidar_dz1", 0)) < 0.3 and limpias.get("visibilidad", 1) > 0.8:
            banderas.append("CONTRADICCION_proximidad_vs_lidar")   # posible eco falso
        critico = any(f"NAN_{c}" in banderas for c in self.CRITICOS)
        return limpias, banderas, critico

# ## 5. `MissionLogger`

class MissionLogger:
    def __init__(self, config):
        self.config, self.filas, self.eventos = config, [], []

    def registrar(self, fila):
        self.filas.append(fila)

    def evento(self, paso, tipo, detalle):
        self.eventos.append({"paso": paso, "tipo": tipo, "detalle": detalle})

    def df(self):
        return pd.DataFrame(self.filas)

    def exportar(self, prefijo="mision"):
        csv_path, json_path = f"{prefijo}_traza.csv", f"{prefijo}_config.json"
        self.df().to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"version": VERSION, "config": self.config, "eventos": self.eventos,
                       "resultado": self.filas[-1] if self.filas else None},
                      f, ensure_ascii=False, indent=2, default=str)
        return csv_path, json_path

# ## 6. `Simulation`: ciclo de misión 3D (todavía con política provisional sin IA)
#
# La política provisional va al sitio más cercano y esquiva solo lo que ve justo adelante: proximidad menor a 3 m,
# caída del LIDAR menor a −3 m o pendiente mayor a 25°. **Es ingenua a propósito**: es la línea base de control.
# En el Paso 3 se reemplaza por RNA + difuso + AG + supervisor.

class Simulation:
    def __init__(self, planeta="Marte", modo="rover", tam=20, semilla=42, dificultad="media",
                 bateria_inicial=100.0, max_pasos=200):
        if planeta not in PLANETAS:
            raise ValueError(f"Planeta desconocido: {planeta}")
        perfil = PLANETAS[planeta]
        ok, msg = perfil.validar_modo(modo)
        if not ok:
            raise ValueError(msg)
        try:
            bateria_inicial = float(bateria_inicial)
        except (TypeError, ValueError):
            raise ValueError("Batería inicial inválida: debe ser un número entre 0 y 100.")
        if math.isnan(bateria_inicial) or not 0 <= bateria_inicial <= 100:
            raise ValueError("Batería inicial inválida: debe estar entre 0 y 100 %.")
        self.config = dict(planeta=planeta, modo=modo, tam=tam, semilla=semilla, dificultad=dificultad,
                           bateria_inicial=bateria_inicial, max_pasos=max_pasos, version=VERSION)
        self.world = World3D(perfil, tam, semilla, dificultad, modo=modo)
        self.physics = Physics(perfil, modo=modo)
        self.sensores = SensorSuite(semilla=semilla + 7)
        self.rover = Rover(pos=self.world.inicio, bateria=bateria_inicial, estado="EXPLORACION")
        self.log = MissionLogger(self.config)
        self.paso, self.max_pasos = 0, max_pasos
        self.ruta_real = [self.rover.pos]
        self.ruta_planificada = []
        self.ctx = {"visibilidad": 1.0, "sismo": 0.0, "noche_forzada": False, "tormenta_pasos": 0}
        self.terminada = False

    # ---------- eventos (RF-09) ----------
    def inyectar(self, tipo, **kw):
        if tipo == "tormenta":
            self.ctx["tormenta_pasos"] = kw.get("duracion", 15)
        elif tipo == "sismo":
            self.ctx["sismo"] = kw.get("intensidad", 0.8)
        elif tipo == "noche":
            self.ctx["noche_forzada"] = kw.get("activa", True)
        elif tipo == "fallo_sensor":
            self.sensores.fallos[kw.get("sensor", "proximidad_m")] = kw.get("modo", "nan")
        elif tipo == "reparar_sensor":
            self.sensores.fallos.pop(kw.get("sensor", "proximidad_m"), None)
        elif tipo == "meteorito":
            centro = kw.get("pos") or self._celda_cercana(3)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (centro[0] + dr, centro[1] + dc)
                    if self.world.dentro(q) and q != self.rover.pos:
                        self.world.zonas_prohibidas.add(q)
        elif tipo == "barranco":
            ok, msg = self.world.agregar_barranco(kw["pos"])
            if not ok:
                self.log.evento(self.paso, "error_evento", msg); return msg
        else:
            msg = f"Evento desconocido: {tipo}"
            self.log.evento(self.paso, "error_evento", msg); return msg
        self.log.evento(self.paso, tipo, kw)
        return f"Evento '{tipo}' inyectado en paso {self.paso}."

    def _celda_cercana(self, radio):
        r, c = self.rover.pos
        rng = np.random.default_rng(self.world.semilla + self.paso)
        while True:
            q = (int(np.clip(r + rng.integers(-radio, radio + 1), 0, self.world.tam - 1)),
                 int(np.clip(c + rng.integers(-radio, radio + 1), 0, self.world.tam - 1)))
            if q != self.rover.pos:
                return q

    # ---------- ciclo ----------
    def observar(self):
        return self.sensores.leer(self.world, self.rover, self.physics, self.paso, self.ctx)

    def objetivo_actual(self):
        rv = self.rover
        pend = [s for s in self.world.sitios if not s["recolectado"]]
        dist = lambda p: abs(p[0] - rv.pos[0]) + abs(p[1] - rv.pos[1])
        if rv.bateria < 25 or not pend:
            return min(self.world.refugios, key=dist)
        return min((s["pos"] for s in pend), key=dist)

    def politica_provisional(self, lect, critico):
        rv = self.rover
        if rv.bateria <= 0:
            return "detener", rv.rumbo, 0.0, "BATERIA_CERO"
        if critico:
            return "detener", rv.rumbo, 0.0, "SENSOR_CRITICO_INVALIDO"
        if self.ctx["sismo"] > 0.3:
            return "esperar", rv.rumbo, 0.0, "SISMO_ESPERAR_Y_REMEDIR"
        obj = self.objetivo_actual()
        if rv.pos == obj and obj in self.world.refugios and rv.bateria < 95:
            return "recargar", rv.rumbo, 0.0, "EN_REFUGIO"
        opciones = sorted(DIRS, key=lambda d: abs(rv.pos[0] + DIRS[d][0] - obj[0]) + abs(rv.pos[1] + DIRS[d][1] - obj[1]))
        vel = 0.5 if self.ctx["tormenta_pasos"] > 0 else 1.0
        for d in opciones:
            q = (rv.pos[0] + DIRS[d][0], rv.pos[1] + DIRS[d][1])
            if not self.world.dentro(q) or q in self.ruta_real[-3:]:
                continue
            l, _, crit = self.sensores.validar(self.sensores.leer(self.world, rv, self.physics, self.paso, self.ctx, rumbo=d))
            if crit or l["proximidad_m"] < 3.0 or l["lidar_dz1"] < -3.0 or l["pitch_deg"] > 25:
                continue
            return "avanzar", d, vel, "AVANZAR_HACIA_OBJETIVO"
        return "detener", rv.rumbo, 0.0, "SIN_MOVIMIENTO_SEGURO_VISIBLE"

    def ejecutar(self, accion, rumbo, velocidad):
        rv, w, ph = self.rover, self.world, self.physics
        consumo, info = 0.1, ""
        if accion == "avanzar" and rv.bateria > 0:
            destino = (rv.pos[0] + DIRS[rumbo][0], rv.pos[1] + DIRS[rumbo][1])
            if not w.dentro(destino):
                info = "BORDE_DEL_MAPA"
            else:
                dz = w.z(destino) - w.z(rv.pos)
                costo = ph.costo_movimiento(dz, velocidad)
                if costo > rv.bateria:
                    info = "BATERIA_INSUFICIENTE_PARA_MOVER"
                else:
                    rv.rumbo = rumbo
                    consumo += costo
                    terreno = int(w.terreno[destino])
                    if terreno == BARRANCO:
                        rv.pos, rv.vivo, rv.salud = destino, False, 0.0
                        info = "CAIDA_EN_BARRANCO"
                    elif terreno == OBSTACULO or destino in w.zonas_prohibidas:
                        rv.salud = max(0.0, rv.salud - 20)
                        info = "COLISION"
                        if rv.salud <= 0:
                            rv.vivo = False
                    else:
                        pitch, roll = w.pitch_deg(rv.pos, destino), w.roll_deg(destino, rumbo)
                        rv.pos = destino
                        rv.odometro_m += math.hypot(CELL_M, dz)
                        if max(pitch, roll) > LIM_VUELCO_DEG:
                            rv.vivo, rv.salud = False, 0.0
                            info = f"VUELCO(pitch={pitch:.0f}°,roll={roll:.0f}°)"
        elif accion == "recargar":
            rv.bateria = min(100.0, rv.bateria + 5.0)

        for s in w.sitios:
            if s["pos"] == rv.pos and not s["recolectado"] and rv.vivo:
                if rv.carga_kg + s["masa_kg"] <= rv.capacidad_kg:
                    s["recolectado"] = True
                    rv.carga_kg += s["masa_kg"]
                    rv.muestras.append(s["id"])
                    info += f" MUESTRA_{s['id']}" + ("_SEÑAL_ANOMALA(no confirma vida)" if s["biofirma_simulada"] else "")
                else:
                    info += f" SIN_CAPACIDAD_PARA_{s['id']}"

        de_dia = ph.es_de_dia(self.paso, self.ctx["noche_forzada"])
        t_ext = ph.temp_exterior(self.paso, self.ctx["noche_forzada"])
        if t_ext < -30:
            consumo += 0.3
        rv.bateria = float(np.clip(rv.bateria - consumo + ph.recarga_solar(de_dia, self.ctx["visibilidad"]), 0.0, 100.0))
        ph.actualizar_temperatura(rv, t_ext)
        if not (ph.TEMP_INT_MIN <= rv.temp_interna <= ph.TEMP_INT_MAX):
            rv.estado = "MODO_SEGURO"; info += " LIMITE_TERMICO"
        return info.strip()

    def actualizar_mundo(self):
        if self.ctx["tormenta_pasos"] > 0:
            self.ctx["tormenta_pasos"] -= 1; self.ctx["visibilidad"] = 0.3
        else:
            self.ctx["visibilidad"] = 1.0
        self.ctx["sismo"] = max(0.0, self.ctx["sismo"] - 0.3)

    def evaluar_fin(self):
        rv = self.rover
        todos = all(s["recolectado"] for s in self.world.sitios)
        if not rv.vivo:
            rv.estado = "FRACASO"
        elif todos and rv.pos in self.world.refugios:
            rv.estado = "EXITO"
        elif rv.bateria <= 0 and rv.pos not in self.world.refugios:
            rv.estado = "FRACASO"
        elif self.paso + 1 >= self.max_pasos:
            rv.estado = "FRACASO"
        self.terminada = rv.estado in ("EXITO", "FRACASO")

    def decidir(self, lect, banderas, critico):
        """Gancho que el Paso 3 reemplaza por RNA + difuso + AG + supervisor."""
        accion, rumbo, vel, motivo = self.politica_provisional(lect, critico)
        return {"accion": accion, "rumbo": rumbo, "velocidad": vel, "motivo": motivo,
                "p_rna": None, "velocidad_difusa": None, "ruta_ag": None}

    def step(self):
        if self.terminada:
            return None
        if self.rover.bateria <= 0 and self.rover.pos not in self.world.refugios:
            # Batería agotada: no se mueve y la misión termina (sección 12)
            self.rover.estado, self.terminada = "FRACASO", True
            fila = {"paso": self.paso, "fila": self.rover.pos[0], "col": self.rover.pos[1],
                    "z_m": round(self.world.z(self.rover.pos), 2), "estado": "FRACASO", "accion": "detener",
                    "velocidad": 0.0, "motivo": "BATERIA_AGOTADA", "resultado": "FIN_MISION",
                    "bateria_pct": 0.0, "salud_pct": self.rover.salud, "muestras": "|".join(self.rover.muestras)}
            self.log.registrar(fila); self.paso += 1
            return fila
        lect, banderas, critico = self.sensores.validar(self.observar())
        dec = self.decidir(lect, banderas, critico)
        if critico:
            self.rover.estado = "MODO_SEGURO"
        elif self.rover.estado in ("MODO_SEGURO", "CONFIGURACION"):
            self.rover.estado = "EXPLORACION"
        info = self.ejecutar(dec["accion"], dec["rumbo"], dec["velocidad"])
        self.actualizar_mundo()
        self.ruta_real.append(self.rover.pos)
        self.evaluar_fin()
        rv = self.rover
        fila = {"paso": self.paso, "fila": rv.pos[0], "col": rv.pos[1], "z_m": round(self.world.z(rv.pos), 2),
                "rumbo": rv.rumbo, "estado": rv.estado, "resultado": info, "bateria_pct": round(rv.bateria, 2),
                "temp_int_c": round(rv.temp_interna, 2), "salud_pct": rv.salud, "muestras": "|".join(rv.muestras),
                "banderas_sensores": "|".join(banderas), **dec,
                **{f"s_{k}": (round(v, 3) if isinstance(v, float) else v) for k, v in lect.items() if k != "pos_estimada"}}
        self.log.registrar(fila)
        self.paso += 1
        return fila

    def correr(self, n=None):
        for _ in range(n or self.max_pasos):
            if self.step() is None:
                break
        return self.log.df()

# ## 7. Visualización 3D (Panel A)
#
# - `dibujar_3d()`: superficie 3D con matplotlib. Siempre funciona en Colab.
# - `dibujar_3d_plotly()`: versión interactiva (rotar y hacer zoom con el mouse) si Plotly está disponible.
# - `dibujar_planta()`: vista desde arriba, útil para leer coordenadas.
#
# Colores: barranco azul oscuro, rocas gris, zonas prohibidas (meteorito) rojo. La ruta ejecutada va en rojo
# continuo y la planificada en cian punteado.

def _colores_terreno(w):
    norm = (w.elevacion - w.elevacion.min()) / (np.ptp(w.elevacion) + 1e-9)
    fc = plt.cm.YlOrBr(0.15 + 0.7 * norm)
    fc[w.terreno == OBSTACULO] = (0.45, 0.45, 0.45, 1)
    fc[w.terreno == BARRANCO] = (0.1, 0.12, 0.35, 1)
    for q in w.zonas_prohibidas:
        fc[q] = (0.85, 0.2, 0.2, 1)
    return fc


def dibujar_3d(w, ruta_real=None, rutas=None, rover_pos=None, ax=None, titulo="", elev=40, azim=-60):
    """rutas: lista de (lista_celdas, color, etiqueta) para rutas planificadas/candidatas."""
    if ax is None:
        fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False   # dibuja rutas y marcadores SOBRE la superficie
    n = w.tam
    X, Y = np.meshgrid(np.arange(n), np.arange(n))
    ax.plot_surface(X, Y, w.elevacion, facecolors=_colores_terreno(w), rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=True, alpha=0.9, zorder=0)
    zc = lambda p: w.z(p) + 0.8
    for r in w.refugios:
        ax.scatter(r[1], r[0], zc(r) + 0.5, marker="s", s=90, c="#2a9d8f", edgecolors="k", depthshade=False, zorder=5)
    for s in w.sitios:
        p = s["pos"]
        ax.scatter(p[1], p[0], zc(p) + 0.5, marker="*", s=200, c="#bbbbbb" if s["recolectado"] else "#f4a261",
                   edgecolors="k", depthshade=False, zorder=5)
        ax.text(p[1], p[0], zc(p) + 2, f"{s['id']}({s['valor']})", fontsize=8, zorder=6)
    for celdas, color, etiqueta in (rutas or []):
        if celdas:
            a = np.array(celdas)
            ax.plot(a[:, 1], a[:, 0], [zc(tuple(c)) for c in a], "--", c=color, lw=2, label=etiqueta, zorder=7)
    if ruta_real:
        a = np.array(ruta_real)
        ax.plot(a[:, 1], a[:, 0], [zc(tuple(c)) for c in a], "-", c="#e63946", lw=3, label="ruta ejecutada", zorder=8)
    if rover_pos is not None:
        ax.scatter(rover_pos[1], rover_pos[0], zc(rover_pos) + 0.8, s=160, c="#e63946", edgecolors="k",
                   depthshade=False, label="rover", zorder=9)
    ax.scatter(w.inicio[1], w.inicio[0], zc(w.inicio), s=60, c="white", edgecolors="k", depthshade=False, label="inicio", zorder=9)
    ax.set_xlabel("columna (×5 m)"); ax.set_ylabel("fila (×5 m)"); ax.set_zlabel("z (m)")
    ax.set_box_aspect((1, 1, 0.4)); ax.view_init(elev=elev, azim=azim)
    ax.set_title(titulo or f"{w.planeta.nombre} · semilla {w.semilla} · terreno 3D")
    ax.legend(loc="upper left", fontsize=8)
    return ax


def dibujar_3d_plotly(w, ruta_real=None, rutas=None, rover_pos=None, titulo=""):
    if not HAY_PLOTLY:
        print("Plotly no está disponible; use dibujar_3d().")
        return None
    color_terreno = w.elevacion.copy()
    zc = lambda p: w.z(p) + 0.8
    fig = go.Figure()
    fig.add_trace(go.Surface(z=w.elevacion, surfacecolor=color_terreno, colorscale="Earth", showscale=False,
                             name="terreno", opacity=0.95))
    for terr, color, nombre in [(BARRANCO, "navy", "barranco"), (OBSTACULO, "gray", "roca")]:
        rr, cc = np.where(w.terreno == terr)
        fig.add_trace(go.Scatter3d(x=cc, y=rr, z=w.elevacion[rr, cc] + 0.3, mode="markers",
                                   marker=dict(size=4, color=color, symbol="square"), name=nombre))
    fig.add_trace(go.Scatter3d(x=[r[1] for r in w.refugios], y=[r[0] for r in w.refugios],
                               z=[zc(r) + 1 for r in w.refugios], mode="markers",
                               marker=dict(size=7, color="#2a9d8f", symbol="square"), name="refugio"))
    fig.add_trace(go.Scatter3d(x=[s["pos"][1] for s in w.sitios], y=[s["pos"][0] for s in w.sitios],
                               z=[zc(s["pos"]) + 1 for s in w.sitios], mode="markers+text",
                               text=[f"{s['id']}({s['valor']})" for s in w.sitios],
                               marker=dict(size=7, color="orange", symbol="diamond"), name="sitio científico"))
    for celdas, color, etiqueta in (rutas or []):
        if celdas:
            fig.add_trace(go.Scatter3d(x=[c[1] for c in celdas], y=[c[0] for c in celdas],
                                       z=[zc(tuple(c)) for c in celdas], mode="lines",
                                       line=dict(color=color, width=5, dash="dash"), name=etiqueta))
    if ruta_real:
        fig.add_trace(go.Scatter3d(x=[c[1] for c in ruta_real], y=[c[0] for c in ruta_real],
                                   z=[zc(tuple(c)) for c in ruta_real], mode="lines",
                                   line=dict(color="red", width=7), name="ruta ejecutada"))
    if rover_pos is not None:
        fig.add_trace(go.Scatter3d(x=[rover_pos[1]], y=[rover_pos[0]], z=[zc(rover_pos) + 1.5], mode="markers",
                                   marker=dict(size=9, color="red"), name="rover"))
    fig.update_layout(title=titulo or f"{w.planeta.nombre} · semilla {w.semilla}", height=600,
                      scene=dict(aspectratio=dict(x=1, y=1, z=0.4), xaxis_title="columna", yaxis_title="fila",
                                 zaxis_title="z (m)"), margin=dict(l=0, r=0, t=40, b=0))
    return fig


def dibujar_planta(w, ruta_real=None, rutas=None, rover_pos=None, ax=None, titulo="", fondo=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(w.elevacion if fondo is None else fondo, cmap="copper" if fondo is None else "Reds")
    rr, cc = np.where(w.terreno == BARRANCO); ax.scatter(cc, rr, marker="s", s=110, c="#1a1f5a", label="barranco")
    rr, cc = np.where(w.terreno == OBSTACULO); ax.scatter(cc, rr, marker="s", s=80, c="#777", label="roca")
    for q in w.zonas_prohibidas:
        ax.add_patch(plt.Rectangle((q[1] - .5, q[0] - .5), 1, 1, color="red", alpha=0.35))
    for r in w.refugios:
        ax.scatter(r[1], r[0], marker="s", s=150, c="#2a9d8f", edgecolors="k", zorder=3)
    for s in w.sitios:
        ax.scatter(s["pos"][1], s["pos"][0], marker="*", s=230, c="#f4a261", edgecolors="k", zorder=3)
        ax.annotate(f"{s['id']}({s['valor']})", (s["pos"][1] + .3, s["pos"][0] - .3), fontsize=8, color="w")
    for celdas, color, etiqueta in (rutas or []):
        if celdas:
            a = np.array(celdas); ax.plot(a[:, 1], a[:, 0], "--", c=color, lw=2, label=etiqueta)
    if ruta_real:
        a = np.array(ruta_real); ax.plot(a[:, 1], a[:, 0], "-", c="#e63946", lw=2.5, label="ruta ejecutada")
    if rover_pos is not None:
        ax.scatter(rover_pos[1], rover_pos[0], s=150, c="#e63946", edgecolors="k", zorder=4)
    ax.scatter(w.inicio[1], w.inicio[0], s=60, c="white", edgecolors="k", zorder=4)
    ax.set_title(titulo or "Vista en planta"); ax.legend(fontsize=7, loc="lower right")
    return ax


def telemetria(df):
    fig, axs = plt.subplots(1, 4, figsize=(17, 3))
    axs[0].plot(df.paso, df.bateria_pct); axs[0].set_title("Batería (%)"); axs[0].set_ylim(0, 100)
    axs[1].plot(df.paso, df.z_m); axs[1].set_title("Altitud z (m)")
    axs[2].plot(df.paso, df.s_pitch_deg, label="pitch adelante"); axs[2].plot(df.paso, df.s_roll_deg, label="roll")
    axs[2].axhline(LIM_VUELCO_DEG, c="r", ls=":", label="límite vuelco"); axs[2].set_title("Inclinación (°)"); axs[2].legend(fontsize=7)
    axs[3].plot(df.paso, df.temp_int_c, label="interna"); axs[3].plot(df.paso, df.s_temp_ext_c, alpha=.6, label="exterior")
    axs[3].set_title("Temperatura (°C)"); axs[3].legend(fontsize=7)
    for a in axs: a.set_xlabel("paso")
    plt.tight_layout()

# ## 8. Demo: misión en Marte 3D (CA-01: al menos 10 pasos con telemetría)

# ## 9. Demo de eventos: tormenta, sismo, fallo de sensor y meteorito

# ## 10. Pruebas de borde de la simulación (sección 12 del enunciado)

# ## 11. Línea base sin IA en 20 semillas (control para comparar después)

# ---
# # PASO 2 (Hito H2): las tres técnicas de IA por separado
#
# ## 12. RNA: percepción (`NeuralPerception`)
#
# **Tarea:** con las lecturas ruidosas de los sensores apuntando hacia adelante, clasificar la celda siguiente como
# **libre / obstáculo / barranco** y entregar la probabilidad de cada clase.
#
# **Datos sintéticos:**
# - Se generan 100 mapas 3D nuevos (semillas 1000–1099), rotando Marte, Mercurio, Tierra y Venus y las 3 dificultades.
# - En cada mapa se toman pares (posición libre, rumbo) cuya celda de adelante es de cada clase, hasta 40 por clase.
# - **Etiqueta (verdad del terreno):** el tipo real de la celda de adelante (`world.terreno`). La red **no** recibe la
#   etiqueta ni ninguna variable idéntica: recibe 9 rasgos **ruidosos** (proximidad, perfil LIDAR, pendiente,
#   rugosidad, textura, vibración, visibilidad). Además, el 40 % de las muestras tiene polvo y el 10 % tiene un sismo
#   que produce falsas alarmas.
#
# **Partición sin fuga:** se divide **por mapa**, no por filas. 60 mapas van a entrenamiento, 20 a validación y 20 a
# prueba. Como dos lecturas del mismo mapa están muy correlacionadas, separarlas por fila inflaría las métricas.
# Además, las semillas de la RNA (1000+) **no se cruzan** con las de las misiones de evaluación (0–99).

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (confusion_matrix, f1_score, recall_score, accuracy_score,
                             classification_report, ConfusionMatrixDisplay)

PLANETAS_ROVER = ["Marte", "Mercurio", "Tierra", "Venus"]
SEMILLAS_RNA = {"train": range(1000, 1060), "valid": range(1060, 1080), "test": range(1080, 1100)}


def generar_dataset(semillas, por_clase=40):
    filas = []
    for sd in semillas:
        planeta, dif = PLANETAS_ROVER[sd % 4], ["facil", "media", "dificil"][sd % 3]
        w = World3D(PLANETAS[planeta], 20, sd, dif)
        rng = np.random.default_rng(50_000 + sd)
        cand = {LIBRE: [], OBSTACULO: [], BARRANCO: []}
        for r in range(w.tam):
            for c in range(w.tam):
                if w.terreno[r, c] == LIBRE:
                    for d, q in w.vecinos((r, c)):
                        cand[int(w.terreno[q])].append(((r, c), d))
        for clase, lst in cand.items():
            for i in rng.choice(len(lst), size=min(por_clase, len(lst)), replace=False):
                pos, d = lst[i]
                vis = 1.0 if rng.random() < 0.6 else float(rng.uniform(0.2, 1.0))
                sismo = float(rng.uniform(0.3, 0.9)) if rng.random() < 0.1 else 0.0
                filas.append({**rasgos_percepcion(w, pos, d, rng, vis, sismo), "clase": clase, "mapa": sd, "planeta": planeta})
    return pd.DataFrame(filas)


t0 = time.time()
datos = {k: generar_dataset(v) for k, v in SEMILLAS_RNA.items()}
print(f"Dataset generado en {time.time() - t0:.1f} s")
for k, d in datos.items():
    print(f"{k:5s}: {len(d):5d} filas · mapas {d.mapa.min()}–{d.mapa.max()} · clases {d.clase.value_counts().sort_index().to_dict()}")
assert not set(datos["train"].mapa) & set(datos["test"].mapa), "Fuga: mapas compartidos entre train y test"
datos["train"].groupby("clase")[RASGOS_RNA].mean().rename(index=dict(enumerate(CLASES))).round(2)

# ### 12.1 Modelo y entrenamiento
#
# MLP **9 → 32 → 16 → 3** con ReLU en las capas ocultas y softmax a la salida, optimizador Adam, en CPU con scikit-learn.
# Se entrena época por época (`partial_fit`) para registrar la pérdida y la exactitud, y se conserva el modelo de la
# época con mejor **macro-F1 en validación** (parada temprana manual). El conjunto de prueba **no se toca** hasta el final.

class NeuralPerception:
    def __init__(self, ocultas=(32, 16), semilla=0, lr=1e-3):
        self.ocultas, self.semilla = ocultas, semilla
        self.scaler = StandardScaler()
        self.mlp = MLPClassifier(hidden_layer_sizes=ocultas, activation="relu", solver="adam",
                                 learning_rate_init=lr, alpha=1e-4, batch_size=64, random_state=semilla)
        self.historial, self.mejor_epoca = [], None
        self.lo = np.array([SensorSuite.RANGOS[k][0] for k in RASGOS_RNA])
        self.hi = np.array([SensorSuite.RANGOS[k][1] for k in RASGOS_RNA])

    def _X(self, df):
        return np.clip(df[RASGOS_RNA].to_numpy(float), self.lo, self.hi)

    def entrenar(self, tr, va, epocas=80):
        tr = tr.sample(frac=1, random_state=self.semilla)
        Xtr, ytr = self.scaler.fit_transform(self._X(tr)), tr.clase.to_numpy()
        Xva, yva = self.scaler.transform(self._X(va)), va.clase.to_numpy()
        mejor, mejor_mlp = -1, None
        for ep in range(1, epocas + 1):
            self.mlp.partial_fit(Xtr, ytr, classes=[0, 1, 2])
            pva = self.mlp.predict(Xva)
            f1 = f1_score(yva, pva, average="macro")
            self.historial.append({"epoca": ep, "loss": self.mlp.loss_, "acc_train": accuracy_score(ytr, self.mlp.predict(Xtr)),
                                   "acc_valid": accuracy_score(yva, pva), "f1_valid": f1})
            if f1 > mejor:
                mejor, mejor_mlp, self.mejor_epoca = f1, copy.deepcopy(self.mlp), ep
        self.mlp = mejor_mlp
        return pd.DataFrame(self.historial)

    def _forward(self, xn):
        """Propagación manual capa por capa: permite mostrar las activaciones internas."""
        a, acts = xn, []
        for i, (W, b) in enumerate(zip(self.mlp.coefs_, self.mlp.intercepts_)):
            z = a @ W + b
            if i < len(self.mlp.coefs_) - 1:
                a = np.maximum(z, 0); acts.append(a)
            else:
                e = np.exp(z - z.max(axis=-1, keepdims=True)); a = e / e.sum(axis=-1, keepdims=True)
        return a, acts

    def predecir_lote(self, df):
        return self.mlp.predict_proba(self.scaler.transform(self._X(df)))

    def predecir(self, lect):
        """lect: dict con los rasgos. Devuelve predicción explicada; nunca lanza excepción por NaN."""
        x = np.array([lect.get(k, np.nan) for k in RASGOS_RNA], dtype=float)
        if np.isnan(x).any():
            faltan = [k for k, v in zip(RASGOS_RNA, x) if np.isnan(v)]
            return {"valido": False, "motivo": "RNA_ENTRADA_INVALIDA", "faltan": faltan,
                    "probs": None, "clase": None, "confianza": 0.0, "p_peligro": None}
        xc = np.clip(x, self.lo, self.hi)
        xn = self.scaler.transform(xc[None, :])
        probs, acts = self._forward(xn)
        probs = probs[0]
        k = int(probs.argmax())
        return {"valido": True, "entrada": dict(zip(RASGOS_RNA, np.round(xc, 3))),
                "normalizada": dict(zip(RASGOS_RNA, np.round(xn[0], 3))),
                "capa1_activas": int((acts[0] > 0).sum()), "capa1": np.round(acts[0][0], 3), "capa2": np.round(acts[1][0], 3),
                "probs": dict(zip(CLASES, np.round(probs, 4))), "clase": CLASES[k], "confianza": float(probs[k]),
                "p_peligro": float(probs[OBSTACULO] + probs[BARRANCO]), "p_barranco": float(probs[BARRANCO])}


rna = NeuralPerception(semilla=0)
t0 = time.time()
EPOCAS_RNA = 45   # la versión web usa menos épocas para cargar más rápido
hist = rna.entrenar(datos["train"], datos["valid"], epocas=EPOCAS_RNA)
T_ENTRENAMIENTO_RNA = time.time() - t0
print(f"Entrenamiento offline: {T_ENTRENAMIENTO_RNA:.1f} s · mejor época (macro-F1 valid): {rna.mejor_epoca}")

if MOSTRAR_GRAFICOS:
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.2))
    axs[0].plot(hist.epoca, hist.loss); axs[0].set_title("Pérdida (entropía cruzada, train)"); axs[0].set_xlabel("época")
    axs[1].plot(hist.epoca, hist.acc_train, label="exactitud train"); axs[1].plot(hist.epoca, hist.acc_valid, label="exactitud valid")
    axs[1].plot(hist.epoca, hist.f1_valid, label="macro-F1 valid"); axs[1].axvline(rna.mejor_epoca, c="k", ls=":", label="modelo elegido")
    axs[1].legend(fontsize=8); axs[1].set_xlabel("época"); axs[1].set_title("Curvas de aprendizaje")
    plt.tight_layout(); plt.show()

# ### 12.2 Evaluación en el conjunto de prueba + baselines
#
# - **Baseline 1 (Dummy):** siempre predice la clase más frecuente.
# - **Baseline 2 (umbral fijo):** si la caída del LIDAR es menor a −3 m, es barranco; si la proximidad es menor a
#   3,5 m, es obstáculo; si no, libre. Es la regla "a mano" que la RNA tiene que superar.
# - **Calibración (ECE):** mide si la "confianza" de la red coincide con su tasa real de acierto. Se usa para decidir
#   cuándo confiar en la red y cuándo pasar a modo conservador.

def regla_umbral(df):
    return np.where(df.lidar_dz1 < -3, BARRANCO, np.where(df.proximidad_m < 3.5, OBSTACULO, LIBRE))


def ece(probs, y, bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    total = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return total


te = datos["test"]; yte = te.clase.to_numpy()
P_test = rna.predecir_lote(te); pred_rna = P_test.argmax(1)
dummy = DummyClassifier(strategy="most_frequent").fit(rna._X(datos["train"]), datos["train"].clase)
comparacion = pd.DataFrame([
    {"modelo": n, "exactitud": accuracy_score(yte, p), "macro_F1": f1_score(yte, p, average="macro"),
     "recall_barranco": recall_score(yte, p, labels=[BARRANCO], average="macro", zero_division=0),
     "recall_obstaculo": recall_score(yte, p, labels=[OBSTACULO], average="macro", zero_division=0)}
    for n, p in [("Dummy (clase mayoritaria)", dummy.predict(rna._X(te))), ("Regla de umbral fijo", regla_umbral(te)),
                 ("RNA MLP 32-16", pred_rna)]]).round(3)
display(comparacion)
METRICAS_RNA = {"macro_F1_test": float(comparacion.iloc[-1].macro_F1),
                "recall_barranco_test": float(comparacion.iloc[-1].recall_barranco),
                "ECE_test": round(float(ece(P_test, yte)), 4), "mejor_epoca": rna.mejor_epoca,
                "t_entrenamiento_s": round(T_ENTRENAMIENTO_RNA, 2)}
print("Métricas RNA (test):", METRICAS_RNA)
print(classification_report(yte, pred_rna, target_names=CLASES, digits=3))

if MOSTRAR_GRAFICOS:
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    ConfusionMatrixDisplay(confusion_matrix(yte, pred_rna), display_labels=CLASES).plot(ax=axs[0], colorbar=False)
    axs[0].set_title("Matriz de confusión (test)")
    conf, acierto = P_test.max(1), (pred_rna == yte)
    bordes = np.linspace(0.3, 1, 8); centros, tasas = [], []
    for lo, hi in zip(bordes[:-1], bordes[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 5:
            centros.append(conf[m].mean()); tasas.append(acierto[m].mean())
    axs[1].plot([0.3, 1], [0.3, 1], "k:", label="calibración perfecta"); axs[1].plot(centros, tasas, "o-", label="RNA")
    axs[1].set_xlabel("confianza predicha"); axs[1].set_ylabel("tasa de acierto real")
    axs[1].set_title(f"Diagrama de confiabilidad · ECE = {METRICAS_RNA['ECE_test']:.3f}"); axs[1].legend()
    plt.tight_layout(); plt.show()

# ### 12.3 Análisis de errores y ejemplo explicado (CA-06)
#
# Los errores más graves son los **barrancos predichos como libres**, porque llevarían a una caída. Por eso, en el
# Paso 3 el supervisor **no confía solo en la RNA**: el LIDAR determinista puede vetar el avance aunque la red diga
# "libre", y si la confianza es baja se pasa a modo conservador.

# ## 13. Lógica difusa: control (`FuzzyController`)
#
# **Entradas** (con 3 conjuntos lingüísticos cada una):
#
# | Variable | Universo | Conjuntos (tipo, parámetros) |
# |---|---|---|
# | distancia al peligro | 0–20 m | cerca trap(0,0,2,6) · media tri(3,8,13) · lejos trap(10,15,20,20) |
# | batería | 0–100 % | crítica trap(0,0,10,20) · baja tri(10,30,50) · alta trap(40,60,100,100) |
# | pendiente | 0–45 ° | plana trap(0,0,5,12) · media tri(8,15,22) · fuerte trap(18,25,45,45) |
# | riesgo (de la RNA) | 0–1 | bajo trap(0,0,.2,.4) · medio tri(.25,.5,.75) · alto trap(.6,.8,1,1) |
#
# **Salida:** velocidad 0–1 con los conjuntos parar, lenta, moderada y rápida.
#
# **Inferencia:** Mamdani. El Y de los antecedentes es el **mínimo**, cada consecuente se **recorta** al nivel de
# activación de su regla, la agregación es el **máximo** y la salida se obtiene por **centroide**.
#
# **Por qué estos parámetros:** "fuerte" empieza en 18° y es total en 25°, antes del límite de vuelco de 30°. "Cerca"
# es total hasta 2 m, que es menos de media celda: no queda distancia de frenado. "Crítica" es total bajo 10 %, que es
# aproximadamente lo que cuesta llegar a un refugio a unas 15 celdas.
#
# **Si no se activa ninguna regla:** la salida de respaldo es **velocidad 0** (detenerse) con el motivo
# `FUZZY_SIN_REGLAS`. Si una entrada es NaN, se reemplaza por su **peor caso** y se registra.

def trapmf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    sube = np.ones_like(x) if b == a else np.clip((x - a) / (b - a), 0, 1)
    baja = np.ones_like(x) if d == c else np.clip((d - x) / (d - c), 0, 1)
    y = np.minimum(sube, baja)
    return np.where((x < a) | (x > d), 0.0, y)

def trimf(x, a, b, c):
    return trapmf(x, a, b, b, c)

def evaluar_mf(tipo, params, x):
    return trimf(x, *params) if tipo == "tri" else trapmf(x, *params)


class FuzzyController:
    ENTRADAS = {
        "distancia": {"universo": (0, 20), "unidad": "m", "peor": 0.0, "conjuntos": {
            "cerca": ("trap", (0, 0, 2, 6)), "media": ("tri", (3, 8, 13)), "lejos": ("trap", (10, 15, 20, 20))}},
        "bateria": {"universo": (0, 100), "unidad": "%", "peor": 0.0, "conjuntos": {
            "critica": ("trap", (0, 0, 10, 20)), "baja": ("tri", (10, 30, 50)), "alta": ("trap", (40, 60, 100, 100))}},
        "pendiente": {"universo": (0, 45), "unidad": "°", "peor": 45.0, "conjuntos": {
            "plana": ("trap", (0, 0, 5, 12)), "media": ("tri", (8, 15, 22)), "fuerte": ("trap", (18, 25, 45, 45))}},
        "riesgo": {"universo": (0, 1), "unidad": "prob.", "peor": 1.0, "conjuntos": {
            "bajo": ("trap", (0, 0, .2, .4)), "medio": ("tri", (.25, .5, .75)), "alto": ("trap", (.6, .8, 1, 1))}},
    }
    SALIDA = {"universo": (0, 1), "conjuntos": {
        "parar": ("trap", (0, 0, .05, .15)), "lenta": ("tri", (.1, .3, .5)),
        "moderada": ("tri", (.4, .6, .8)), "rapida": ("trap", (.7, .85, 1, 1))}}
    REGLAS = [
        ("R1", {"distancia": "cerca"}, "parar", "Peligro a < 5 m: no hay distancia de frenado."),
        ("R2", {"riesgo": "alto"}, "parar", "La RNA cree que adelante hay obstáculo o barranco."),
        ("R3", {"bateria": "critica"}, "lenta", "Con energía crítica se avanza despacio para ahorrar (rumbo a refugio)."),
        ("R4", {"pendiente": "fuerte"}, "lenta", "Inclinación cercana al límite de vuelco."),
        ("R5", {"pendiente": "fuerte", "riesgo": "medio"}, "parar", "Pendiente fuerte + posible borde: combinación peligrosa."),
        ("R6", {"riesgo": "medio"}, "lenta", "Incertidumbre de la RNA: avanzar con cautela y volver a medir."),
        ("R7", {"distancia": "media", "riesgo": "bajo", "pendiente": "plana"}, "moderada", "Terreno despejado a media distancia."),
        ("R8", {"distancia": "media", "pendiente": "media"}, "lenta", "Peligro a media distancia en terreno inclinado."),
        ("R9", {"bateria": "alta", "pendiente": "plana", "distancia": "lejos", "riesgo": "bajo"}, "rapida",
         "Todo favorable: aprovechar para avanzar rápido."),
        ("R10", {"bateria": "alta", "pendiente": "media", "distancia": "lejos", "riesgo": "bajo"}, "moderada",
         "Despejado pero con inclinación: velocidad moderada."),
        ("R11", {"bateria": "baja", "pendiente": "plana", "distancia": "lejos", "riesgo": "bajo"}, "moderada",
         "Despejado con batería baja: no correr para ahorrar energía."),
        ("R12", {"bateria": "baja", "pendiente": "media"}, "lenta", "Subir con batería baja gasta mucho: ir lento."),
    ]

    def __init__(self, resolucion=401):
        self.u = np.linspace(0, 1, resolucion)

    def texto_regla(self, regla):
        rid, ant, cons, _ = regla
        return f"{rid}: SI " + " Y ".join(f"{v} es {c}" for v, c in ant.items()) + f" ENTONCES velocidad es {cons}"

    def tabla_reglas(self):
        return pd.DataFrame([{"id": r[0], "regla": self.texto_regla(r), "justificación": r[3]} for r in self.REGLAS])

    def inferir(self, distancia, bateria, pendiente, riesgo, reglas_desactivadas=()):
        avisos, crisp = [], {}
        for var, val in zip(self.ENTRADAS, (distancia, bateria, pendiente, riesgo)):
            info = self.ENTRADAS[var]
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = float("nan")
            if math.isnan(val):
                val = info["peor"]; avisos.append(f"ENTRADA_NAN_{var}_PEOR_CASO")
            crisp[var] = float(np.clip(val, *info["universo"]))
        mu = {var: {c: float(evaluar_mf(t, p, crisp[var])) for c, (t, p) in info["conjuntos"].items()}
              for var, info in self.ENTRADAS.items()}
        agregada = np.zeros_like(self.u)
        reglas = []
        for regla in self.REGLAS:
            rid, ant, cons, _ = regla
            if rid in reglas_desactivadas:
                continue
            act = min(mu[v][c] for v, c in ant.items())                       # Y = mínimo
            reglas.append({"id": rid, "regla": self.texto_regla(regla), "activacion": round(act, 4), "consecuente": cons})
            if act > 0:
                t, p = self.SALIDA["conjuntos"][cons]
                agregada = np.maximum(agregada, np.minimum(act, evaluar_mf(t, p, self.u)))   # recorte + máximo
        if agregada.sum() == 0:
            vel, fallback = 0.0, True
            avisos.append("FUZZY_SIN_REGLAS")
        else:
            vel, fallback = float((self.u * agregada).sum() / agregada.sum()), False      # centroide
        cautela = "MAXIMA" if vel < 0.2 else "ALTA" if vel < 0.45 else "MEDIA" if vel < 0.7 else "BAJA"
        return {"entradas": crisp, "membresias": mu, "reglas": reglas, "agregada": agregada,
                "velocidad": round(vel, 4), "cautela": cautela, "fallback": fallback, "avisos": avisos,
                "reglas_activas": [r["id"] for r in reglas if r["activacion"] > 0]}

    def graficar(self, res, titulo=""):
        fig, axs = plt.subplots(1, 5, figsize=(20, 3.2))
        for ax, (var, info) in zip(axs, self.ENTRADAS.items()):
            x = np.linspace(*info["universo"], 300)
            for c, (t, p) in info["conjuntos"].items():
                ax.plot(x, evaluar_mf(t, p, x), label=f"{c} = {res['membresias'][var][c]:.2f}")
            ax.axvline(res["entradas"][var], c="k", ls="--")
            ax.set_title(f"{var} = {res['entradas'][var]:.2f} {info['unidad']}"); ax.legend(fontsize=7); ax.set_ylim(0, 1.05)
        ax = axs[-1]
        for c, (t, p) in self.SALIDA["conjuntos"].items():
            ax.plot(self.u, evaluar_mf(t, p, self.u), ":", lw=1, label=c)
        ax.fill_between(self.u, res["agregada"], alpha=0.5, color="tab:orange", label="agregada (max)")
        ax.axvline(res["velocidad"], c="r", lw=2, label=f"centroide = {res['velocidad']:.3f}")
        ax.set_title(f"Salida velocidad · cautela {res['cautela']}"); ax.legend(fontsize=7); ax.set_ylim(0, 1.05)
        fig.suptitle(titulo or f"Reglas activas: {res['reglas_activas']}", y=1.03)
        plt.tight_layout(); plt.show()

    def tabla_activacion(self, res):
        return pd.DataFrame(res["reglas"]).query("activacion > 0").sort_values("activacion", ascending=False)


fuzzy = FuzzyController()
display(fuzzy.tabla_reglas())

# ### 13.1 Diagnóstico difuso (CA-05): la distancia baja de 10 m a 1 m
# Se ve cómo cambian las membresías, las reglas que se activan y la velocidad.

# ### 13.2 Superficie de control 3D: continuidad del controlador
# Muestra la velocidad en función de la distancia y del riesgo (con batería y pendiente fijas). Es una superficie
# suave, sin saltos: esa es la ventaja de lo difuso frente a umbrales rígidos.

# ## 14. Algoritmo genético: planificación (`GeneticPlanner`)
#
# **Se combinan las tres familias de subproblemas de los ejercicios anteriores:**
# 1. **Vendedor viajero:** en qué **orden** visitar los sitios científicos. Es una permutación.
# 2. **Mochila binaria:** **qué** muestras recolectar sin superar 8 kg. Es un vector de bits.
# 3. **Laberinto:** cada tramo entre dos puntos se resuelve con **A\*** sobre la cuadrícula 3D. El costo depende del
#    desnivel y del **riesgo estimado por la RNA**, y no se permiten rampas de 30° o más.
#
# **Cromosoma:** `(permutación, bits)`. Por ejemplo, `([2,0,3,1], [1,0,1,1])` significa visitar S3 → S4 → S2 (S1 no se
# incluye) y terminar en el refugio más conveniente.
#
# **Fitness** (todas las magnitudes normalizadas a aproximadamente 0–1; los pesos son configurables):
#
# `fitness = w1·ciencia − w2·distancia − w3·energía − w4·riesgo − w5·tiempo − P(inviabilidad)`
#
# - ciencia = valor recolectado / valor total disponible
# - distancia = metros 3D / (4 · tamaño · 5 m)
# - energía = % de batería estimado / batería disponible
# - riesgo = 0,5 · riesgo medio + 0,5 · riesgo máximo a lo largo de la ruta
# - tiempo = (celdas + 3 pasos por muestra) / (6 · tamaño)
#
# **Restricciones duras** (penalización P grande, y la ruta queda marcada como **no factible**):
# - masa de las muestras mayor que la capacidad,
# - energía mayor que la batería menos una reserva del 5 %, porque **el rover debe poder llegar al refugio**,
# - no existe camino.
#
# Celdas confirmadas como precipicio y zonas de meteorito están **prohibidas** dentro de A\*.
#
# **El AG planifica con el mapa estimado** (DEM orbital con ruido + riesgo predicho por la RNA), nunca con la verdad
# del terreno.
#
# **Operadores:** selección por torneo (k = 3), cruce OX en la permutación y uniforme en los bits (p = 0,85),
# mutación por intercambio en la permutación y volteo de bits (p = 0,3) y elitismo de 2. Se guarda la población
# completa de cada generación para poder inspeccionarla.

def mapa_riesgo_rna(world, red, semilla=0, vis=0.6, ruido=1.3):
    """Mapa de riesgo estimado: la RNA evalúa cada celda con observaciones degradadas tomadas desde 2 vecinos
    (simula el reconocimiento previo de baja calidad). Devuelve P(obstáculo) + P(barranco) por celda."""
    rng = np.random.default_rng(semilla)
    filas, idx = [], []
    for r in range(world.tam):
        for c in range(world.tam):
            dirs = [d for d, (dr, dc) in DIRS.items() if world.dentro((r - dr, c - dc))]
            for d in rng.choice(dirs, size=min(2, len(dirs)), replace=False):
                dr, dc = DIRS[d]
                filas.append(rasgos_percepcion(world, (r - dr, c - dc), d, rng, vis, 0.0, ruido)); idx.append((r, c))
    P = red.predecir_lote(pd.DataFrame(filas))
    riesgo = np.zeros((world.tam, world.tam)); cuenta = np.zeros_like(riesgo)
    for (r, c), p in zip(idx, P):
        riesgo[r, c] += p[OBSTACULO] + p[BARRANCO]; cuenta[r, c] += 1
    return riesgo / np.maximum(cuenta, 1)


def a_estrella_mapa(world, dem, riesgo, bloqueadas, a, b, w_riesgo=15.0):
    """A* en la cuadrícula 3D: costo = distancia 3D (en celdas) + w·riesgo estimado; prohíbe celdas bloqueadas
    y rampas ≥ 30° según el DEM. Devuelve la lista de celdas o None si no hay camino (laberinto)."""
    h = lambda p: abs(p[0] - b[0]) + abs(p[1] - b[1])
    abiertos, g, padre = [(h(a), 0.0, a)], {a: 0.0}, {a: None}
    while abiertos:
        _, gc, p = heapq.heappop(abiertos)
        if p == b:
            camino = []
            while p is not None:
                camino.append(p); p = padre[p]
            return camino[::-1]
        if gc > g.get(p, np.inf):
            continue
        for _, q in world.vecinos(p):
            if q in bloqueadas:
                continue
            dz = dem[q] - dem[p]
            if np.degrees(np.arctan(abs(dz) / CELL_M)) >= LIM_VUELCO_DEG:
                continue
            ng = gc + math.hypot(CELL_M, dz) / CELL_M + w_riesgo * riesgo[q]
            if ng < g.get(q, np.inf):
                g[q], padre[q] = ng, p
                heapq.heappush(abiertos, (ng + h(q), ng, q))
    return None


class GeneticPlanner:
    PESOS_DEF = {"ciencia": 1.0, "distancia": 0.2, "energia": 0.3, "riesgo": 1.0, "tiempo": 0.2}

    def __init__(self, world, inicio, riesgo_est, bateria, physics, capacidad_kg=8.0, sitios=None, bloqueadas=(),
                 pesos=None, poblacion=40, generaciones=40, p_cruce=0.85, p_mut=0.3, elite=2, torneo=3,
                 semilla=0, paciencia=None, reserva_pct=5.0, w_riesgo_astar=15.0, velocidad_media=0.5, consumo_paso=0.1):
        if poblacion < 2 or generaciones < 1:
            raise ValueError("Población ≥ 2 y generaciones ≥ 1.")
        self.w, self.inicio, self.riesgo, self.dem = world, tuple(inicio), riesgo_est, world.dem_orbital
        self.bateria, self.ph, self.cap = float(bateria), physics, capacidad_kg
        self.sitios = [s for s in (sitios if sitios is not None else world.sitios) if not s["recolectado"]]
        self.bloqueadas = set(bloqueadas) | set(world.zonas_prohibidas)
        self.pesos = {**self.PESOS_DEF, **(pesos or {})}
        self.N, self.G, self.pc, self.pm, self.elite, self.k = poblacion, generaciones, p_cruce, p_mut, elite, torneo
        self.rng = np.random.default_rng(semilla)
        self.semilla, self.paciencia, self.reserva, self.w_ra = semilla, paciencia, reserva_pct, w_riesgo_astar
        self.vm, self.consumo_paso = velocidad_media, consumo_paso   # supuestos para estimar energía por celda
        self.n = len(self.sitios)
        self.valor_total = sum(s["valor"] for s in self.sitios)
        self._cache = {}
        self.historial, self.ops_log = [], []
        self.parada_temprana = None

    # ---------- laberinto: A* sobre el mapa estimado ----------
    def a_estrella(self, a, b):
        key = (a, b)
        if key not in self._cache:
            self._cache[key] = a_estrella_mapa(self.w, self.dem, self.riesgo, self.bloqueadas, a, b, self.w_ra)
        return self._cache[key]

    # ---------- genotipo ----------
    def aleatorio(self):
        return (list(self.rng.permutation(self.n)), [int(b) for b in self.rng.integers(0, 2, self.n)])

    def reparar(self, crom):
        """Corrige genes inválidos (índices fuera de rango o repetidos) sin detener el notebook."""
        perm, bits = crom
        vistos, nueva = set(), []
        for g in perm:
            if isinstance(g, (int, np.integer)) and 0 <= g < self.n and g not in vistos:
                nueva.append(int(g)); vistos.add(int(g))
        nueva += [i for i in range(self.n) if i not in vistos]
        bits = [1 if b else 0 for b in list(bits)[:self.n]] + [0] * max(0, self.n - len(bits))
        return (nueva, bits)

    def decodificar(self, crom):
        perm, bits = crom
        return [i for i in perm if bits[i]]

    # ---------- fitness desglosado ----------
    def evaluar(self, crom):
        orden = self.decodificar(crom)
        ruta, viol, prev = [self.inicio], [], self.inicio
        for i in orden:
            seg = self.a_estrella(prev, self.sitios[i]["pos"])
            if seg is None:
                viol.append(f"SIN_CAMINO_a_{self.sitios[i]['id']}"); break
            ruta += seg[1:]; prev = self.sitios[i]["pos"]
        segs = [(self.a_estrella(prev, r), r) for r in self.w.refugios]
        segs = [(s_, r) for s_, r in segs if s_ is not None]
        refugio = None
        if not segs:
            viol.append("SIN_CAMINO_A_REFUGIO")
        else:
            seg, refugio = min(segs, key=lambda t: len(t[0]))
            ruta += seg[1:]
        dist_m = energia = 0.0
        for p, q in zip(ruta[:-1], ruta[1:]):
            dz = self.dem[q] - self.dem[p]
            dist_m += math.hypot(CELL_M, dz)
            energia += self.ph.costo_movimiento(dz, self.vm) + self.consumo_paso / self.vm   # movimiento + consumo base
        riesgos = np.array([self.riesgo[c] for c in ruta[1:]]) if len(ruta) > 1 else np.array([0.0])
        masa = sum(self.sitios[i]["masa_kg"] for i in orden)
        T = self.w.tam
        comp = {"ciencia": (sum(self.sitios[i]["valor"] for i in orden) / self.valor_total) if self.valor_total else 0.0,
                "distancia": dist_m / (4 * T * CELL_M), "energia": energia / max(self.bateria, 1e-6),
                "riesgo": 0.5 * riesgos.mean() + 0.5 * riesgos.max(), "tiempo": (len(ruta) - 1 + 3 * len(orden)) / (6 * T)}
        P = 0.0
        if masa > self.cap:
            P += 1 + (masa - self.cap) / self.cap; viol.append(f"MASA({masa:.1f}>{self.cap}kg)")
        if energia > self.bateria - self.reserva:
            P += 2 + (energia - (self.bateria - self.reserva)) / 100; viol.append(f"ENERGIA({energia:.1f}%>{self.bateria - self.reserva:.1f}%)")
        P += 5 * sum(v.startswith("SIN_CAMINO") for v in viol)
        fit = (self.pesos["ciencia"] * comp["ciencia"] - self.pesos["distancia"] * comp["distancia"]
               - self.pesos["energia"] * comp["energia"] - self.pesos["riesgo"] * comp["riesgo"]
               - self.pesos["tiempo"] * comp["tiempo"] - P)
        return {"fitness": fit, **{k: round(v, 4) for k, v in comp.items()}, "penalizacion": round(P, 4),
                "factible": not viol, "violaciones": viol, "masa_kg": round(masa, 2), "energia_pct": round(energia, 2),
                "visitas": [self.sitios[i]["id"] for i in orden], "refugio": refugio, "ruta": ruta}

    # ---------- operadores ----------
    def seleccion_torneo(self, pob, fits):
        idx = self.rng.choice(len(pob), size=min(self.k, len(pob)), replace=False)
        return pob[max(idx, key=lambda i: fits[i])]

    def cruce(self, p1, p2):
        n = self.n
        if n < 2:
            return (list(p1[0]), list(p1[1]))
        a, b = sorted(self.rng.choice(n, 2, replace=False))
        hijo = [None] * n; hijo[a:b + 1] = p1[0][a:b + 1]
        resto = [g for g in p2[0] if g not in hijo]
        for i in range(n):
            if hijo[i] is None:
                hijo[i] = resto.pop(0)
        mask = self.rng.random(n) < 0.5
        bits = [p1[1][i] if mask[i] else p2[1][i] for i in range(n)]
        return (hijo, bits)

    def mutar(self, crom):
        perm, bits = list(crom[0]), list(crom[1])
        if self.n >= 2:
            i, j = self.rng.choice(self.n, 2, replace=False); perm[i], perm[j] = perm[j], perm[i]
        for i in range(self.n):
            if self.rng.random() < 1.0 / max(self.n, 1):
                bits[i] = 1 - bits[i]
        return (perm, bits)

    # ---------- bucle evolutivo ----------
    def evolucionar(self):
        pob = [self.aleatorio() for _ in range(self.N)]
        sin_mejora, mejor_hist = 0, -np.inf
        for gen in range(self.G):
            evals = [self.evaluar(c) for c in pob]
            fits = [e["fitness"] for e in evals]
            orden = np.argsort(fits)[::-1]
            self.historial.append({"gen": gen, "poblacion": [(pob[i], evals[i]) for i in orden],
                                   "mejor": fits[orden[0]], "media": float(np.mean(fits)), "peor": fits[orden[-1]]})
            if fits[orden[0]] > mejor_hist + 1e-9:
                mejor_hist, sin_mejora = fits[orden[0]], 0
            else:
                sin_mejora += 1
            if self.paciencia and sin_mejora >= self.paciencia:
                self.parada_temprana = gen; break
            if gen == self.G - 1:
                break
            nueva = [pob[i] for i in orden[:self.elite]]                  # elitismo
            ops_gen = []
            while len(nueva) < self.N:
                p1, p2 = self.seleccion_torneo(pob, fits), self.seleccion_torneo(pob, fits)
                hijo = self.cruce(p1, p2) if self.rng.random() < self.pc else (list(p1[0]), list(p1[1]))
                if len(ops_gen) < 3 and hijo != p1:
                    ops_gen.append({"gen": gen, "op": "cruce OX + uniforme", "padre1": p1, "padre2": p2, "hijo": hijo})
                if self.rng.random() < self.pm:
                    antes = hijo; hijo = self.mutar(hijo)
                    if len(ops_gen) < 6:
                        ops_gen.append({"gen": gen, "op": "mutación swap + bit-flip", "antes": antes, "despues": hijo})
                nueva.append(self.reparar(hijo))
            self.ops_log += ops_gen
            pob = nueva
        return self.resultado()

    def resultado(self, k=3):
        """Mejor solución + al menos k candidatos distintos (por visitas decodificadas)."""
        vistos, cands = set(), []
        for h in reversed(self.historial):
            for crom, ev in h["poblacion"]:
                clave = tuple(ev["visitas"])
                if clave not in vistos:
                    vistos.add(clave); cands.append((crom, ev))
            if len(cands) >= 12:
                break
        cands.sort(key=lambda t: t[1]["fitness"], reverse=True)
        mejor_factible = next((c for c in cands if c[1]["factible"]), None)
        return {"mejor": mejor_factible or cands[0], "candidatos": cands[:k], "hay_factible": mejor_factible is not None}

    # ---------- inspección ----------
    def tabla_generacion(self, g, top=6):
        h = self.historial[min(g, len(self.historial) - 1)]
        return pd.DataFrame([{"rank": i + 1, "permutación": c[0], "bits": c[1], "visitas": "→".join(e["visitas"]) or "(solo retorno)",
                              **{k: e[k] for k in ("fitness", "ciencia", "distancia", "energia", "riesgo", "tiempo", "penalizacion",
                                                   "factible")}} for i, (c, e) in enumerate(h["poblacion"][:top])]).round(4)

    def operaciones_generacion(self, g):
        return [o for o in self.ops_log if o["gen"] == g]

    def graficar_fitness(self, marcar=None):
        h = pd.DataFrame([{k: x[k] for k in ("gen", "mejor", "media", "peor")} for x in self.historial])
        plt.figure(figsize=(8, 3.2))
        plt.plot(h.gen, h.mejor, label="mejor"); plt.plot(h.gen, h.media, label="media")
        plt.plot(h.gen, h.peor.clip(lower=h.media.min() - 2), alpha=.4, label="peor (recortado)")
        if marcar is not None:
            plt.axvline(marcar, c="k", ls=":", label=f"gen {marcar}")
        plt.xlabel("generación"); plt.ylabel("fitness"); plt.legend(); plt.title("Evolución del fitness"); plt.show()

# ### 14.1 Ejecución del AG con el mapa de riesgo que estima la RNA (conexión RNA → AG)

# ### 14.2 Tres rutas candidatas con desglose del fitness (RF-07)

# ### 14.3 Diagnóstico genético (CA-04): generación 5
# Se ven los cromosomas, su fitness desglosado y operaciones reales de cruce y mutación de esa generación.

# ## 15. Primera prueba causal de la conexión RNA → difuso y RNA → AG
#
# La rúbrica pide **demostrar** que cambiar la salida de una técnica cambia la decisión. Hay dos pruebas:
# 1. **RNA → AG:** se reemplaza el mapa de riesgo de la RNA por uno uniforme (ablación "sin RNA"). Se cuenta cuántas
#    celdas **realmente peligrosas** atraviesa la mejor ruta en cada caso.
# 2. **RNA → difuso:** misma situación, pero el riesgo de la RNA sube de 0,1 a 0,9 y se observa cómo cae la velocidad.
#
# La integración completa con el supervisor viene en el Paso 3.

# ## 16. Pruebas de borde de las técnicas de IA

# ## 17. Exportación

# ---
# # PASO 3 (Hito H3): integración híbrida
#
# ## 18. Ciclo híbrido, supervisor de seguridad y máquina de estados
#
# Cada paso de simulación ejecuta exactamente el pseudociclo del enunciado:
#
# ```
# observar() → validar_sensores() → predecir_riesgo_rna() → inferir_control_difuso()
# → replanificar_ag_si_corresponde() → aplicar_restricciones() → ejecutar_accion() → actualizar_mundo() → registrar_traza()
# ```
#
# | Etapa | Qué hace en ARES-X |
# |---|---|
# | Percepción 360° | El rover mira las 4 celdas vecinas. La **RNA** clasifica cada una y actualiza su **mapa de creencias** de riesgo. |
# | Sensor de borde | Si el LIDAR mide una caída mayor a 3 m con buena visibilidad, el borde queda **confirmado** como precipicio (determinista). |
# | AG | Se recalcula **solo** al iniciar, al llegar a un objetivo, si la ruta queda bloqueada, si hay un evento grave o si la batería obliga a volver. |
# | Difuso | Distancia al peligro, batería, pendiente y riesgo de la RNA → **velocidad** 0–1 y nivel de cautela. |
# | Supervisor | Revisa las restricciones duras **en orden fijo** y puede vetar. Cada decisión lleva un **código de motivo**. |
# | Física | La velocidad difusa determina cuánto avanza el rover por paso (avance fraccional) y cuánta energía gasta. |
#
# **Prioridades del supervisor** (deterministas; gana la primera que se cumpla):
# 1. Batería agotada → fin. 2. Daño total → fin. 3. Límite térmico → modo seguro (aislar).
# 4. LIDAR caído → modo seguro (no avanzar a ciegas). 5. Sismo en curso → esperar y volver a medir.
# 6. Precipicio confirmado / zona prohibida / borde LIDAR / pendiente o alabeo > 25° → **veto** + replanificar.
# 7. RNA: peligro > 0,6 → veto; confianza < 0,6 → modo conservador (v ≤ 0,3); RNA inválida → modo degradado (v ≤ 0,25).
# 8. Viento fuerte → v ≤ 0,35, y espera si la deriva lo podría empujar a un peligro.
# 9. Batería insuficiente para volver con margen → modo **REFUGIO**.
# 10. El difuso ordena parar → esperar; tras 3 esperas seguidas → replanificar.
#
# **Políticas para experimentos y ablaciones:**
#
# | Política | RNA | Difuso | AG | Creencias en línea | Supervisor |
# |---|---|---|---|---|---|
# | `hibrido` | ✔ | ✔ | ✔ | ✔ | ✔ |
# | `sin_rna` (regla de umbral en su lugar) | ✘ | ✔ | ✔ | ✔ | ✔ |
# | `sin_difuso` (velocidad fija 0,7) | ✔ | ✘ | ✔ | ✔ | ✔ |
# | `sin_ag` (vecino más cercano + A\*) | ✔ | ✔ | ✘ | ✔ | ✔ |
# | `ag_puro` (riesgos estimados fijos) | solo mapa inicial | ✘ | ✔ | ✘ | ✘ |
# | `control` (ruta más corta, ignora el riesgo) | ✘ | ✘ | ✘ | ✘ | ✘ |

POLITICAS = {
    "hibrido":    dict(rna=True,  difuso=True,  ag=True,  online=True,  supervisor=True,  riesgo_cero=False),
    "sin_rna":    dict(rna=False, difuso=True,  ag=True,  online=True,  supervisor=True,  riesgo_cero=False),
    "sin_difuso": dict(rna=True,  difuso=False, ag=True,  online=True,  supervisor=True,  riesgo_cero=False),
    "sin_ag":     dict(rna=True,  difuso=True,  ag=False, online=True,  supervisor=True,  riesgo_cero=False),
    "ag_puro":    dict(rna=True,  difuso=False, ag=True,  online=False, supervisor=False, riesgo_cero=False),
    "control":    dict(rna=False, difuso=False, ag=False, online=False, supervisor=False, riesgo_cero=True),
}

PLANETAS_SUP = ["Mercurio", "Venus", "Tierra", "Marte"]
GIGANTES = ["Júpiter", "Saturno", "Urano", "Neptuno"]
TODOS = set(PLANETAS_SUP) | set(GIGANTES)
EVENTOS = {
    # tipo: (planetas donde aplica, ¿grave? (fuerza replanificación), descripción)
    "tormenta":           ({"Marte", "Tierra", "Venus"} | set(GIGANTES), False, "Tormenta de polvo: baja visibilidad, menos energía solar y LIDAR muy ruidoso."),
    "viento":             ({"Marte", "Tierra", "Venus"} | set(GIGANTES), False, "Vientos fuertes: pueden desviar al rover y aumentan el consumo."),
    "tornado":            ({"Marte", "Tierra"} | set(GIGANTES), True, "Tornado / remolino de polvo (dust devil en Marte) / vórtice atmosférico en los gigantes: zona móvil muy peligrosa."),
    "meteorito":          (TODOS, True, "Impacto de meteorito: cráter nuevo, eyecta y zona caliente temporal."),
    "derrumbe":           (set(PLANETAS_SUP), True, "Derrumbe de ladera: el relieve cambia y aparecen rocas caídas."),
    "sismo":              (set(PLANETAS_SUP), True, "Sismo (marsquake en Marte; hipotético en otros): vibración y relieve alterado."),
    "noche":              (TODOS, False, "Fuerza la noche: sin carga solar y más frío."),
    "dia":                (TODOS, False, "Vuelve el ciclo normal día/noche."),
    "frio_extremo":       (TODOS, False, "Ola de frío extremo: la electrónica se enfría hacia su límite."),
    "calor_extremo":      (TODOS, False, "Calor extremo: riesgo de sobrecalentamiento."),
    "lluvia_acida":       ({"Venus"}, False, "Ambiente corrosivo hipotético: desgaste del chasis."),
    "fallo_sensor":       (TODOS, False, "Falla de un sensor (lecturas NaN)."),
    "reparar_sensor":     (TODOS, False, "Restablece todos los sensores."),
    "baja_conectividad":  (TODOS, False, "Enlace degradado: los comandos llegan con 4 pasos de retraso; el rover sigue autónomo."),
    "biofirma":           (TODOS, True, "Señal biológica anómala SIMULADA en un sitio: sube su prioridad. Nunca confirma vida."),
    "objeto_desconocido": (TODOS, False, "Objeto desconocido móvil (evento narrativo FICTICIO): obstáculo dinámico."),
    "barranco":           (TODOS, False, "Agrega un barranco (en modo sonda: una corriente descendente) en la celda elegida; el vehículo debe descubrirlo con sus sensores."),
    "obstaculo":          (TODOS, False, "Agrega una roca (en modo sonda: una celda de tormenta) en la celda elegida."),
}
RAZON_NO_APLICA = {
    ("tormenta", "Mercurio"): "Mercurio casi no tiene atmósfera (exosfera): no hay tormentas de polvo.",
    ("viento", "Mercurio"): "Mercurio casi no tiene atmósfera: no hay viento.",
    ("tornado", "Mercurio"): "Mercurio casi no tiene atmósfera: no pueden formarse tornados.",
    ("tornado", "Venus"): "En la superficie de Venus los vientos son muy lentos (≈1 m/s): no se forman tornados.",
}
for pl in ["Mercurio", "Tierra", "Marte"] + GIGANTES:
    RAZON_NO_APLICA[("lluvia_acida", pl)] = "El ambiente ácido corrosivo es un escenario hipotético exclusivo de Venus."
for pl in GIGANTES:
    RAZON_NO_APLICA[("derrumbe", pl)] = f"{pl} no tiene superficie sólida: no hay laderas que puedan derrumbarse."
    RAZON_NO_APLICA[("sismo", pl)] = f"{pl} no tiene corteza sólida accesible: la sonda flota en la atmósfera, no hay sismos que medir."


def evento_aplica(tipo, planeta):
    if tipo not in EVENTOS:
        return False, f"Evento desconocido: {tipo}"
    if planeta in EVENTOS[tipo][0]:
        return True, ""
    return False, RAZON_NO_APLICA.get((tipo, planeta), f"El evento '{tipo}' no aplica en {planeta}.")


class ReglaUmbral:
    """Sustituto de la RNA para la ablación 'sin_rna': regla fija sobre los mismos rasgos."""
    def predecir_lote(self, df):
        cls = regla_umbral(df)
        P = np.full((len(df), 3), 0.05); P[np.arange(len(df)), cls] = 0.9
        return P

    def predecir(self, lect):
        x = [lect.get(k, np.nan) for k in RASGOS_RNA]
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in x):
            return {"valido": False, "motivo": "REGLA_ENTRADA_INVALIDA", "probs": None, "clase": None,
                    "confianza": 0.0, "p_peligro": None}
        cls = BARRANCO if lect["lidar_dz1"] < -3 else OBSTACULO if lect["proximidad_m"] < 3.5 else LIBRE
        p = np.full(3, 0.05); p[cls] = 0.9
        return {"valido": True, "probs": dict(zip(CLASES, p.round(3))), "clase": CLASES[cls], "confianza": 0.9,
                "p_peligro": float(p[1] + p[2]), "p_barranco": float(p[2])}


def _dir_entre(a, b):
    d = (b[0] - a[0], b[1] - a[1])
    for k, v in DIRS.items():
        if v == d:
            return k
    return None

class HybridSimulation(Simulation):
    def __init__(self, rna, fuzzy, politica="hibrido", pesos=None, ga_pob=40, ga_gen=40, max_pasos=400, **kw):
        if politica not in POLITICAS:
            raise ValueError(f"Política desconocida: {politica}")
        super().__init__(max_pasos=max_pasos, **kw)
        self.config.update(politica=politica, ga_pob=ga_pob, ga_gen=ga_gen)
        self.rna_model, self.regla, self.fuzzy = rna, ReglaUmbral(), fuzzy
        self.politica, self.mod = politica, dict(POLITICAS[politica])
        self.pesos = {**GeneticPlanner.PESOS_DEF, **(pesos or {})}
        self.ga_pob, self.ga_gen = ga_pob, ga_gen
        self.rng_ev = np.random.default_rng(self.world.semilla + 333)
        w = self.world
        if self.mod["riesgo_cero"]:
            self.belief = np.zeros((w.tam, w.tam))
        else:
            self.belief = mapa_riesgo_rna(w, rna if self.mod["rna"] else self.regla, semilla=w.semilla)
        self.confirmados = set()
        self.plan, self.plan_visitas, self.plan_info = [self.rover.pos], [], {}
        self.necesita_replan, self.motivo_replan = True, "INICIO_MISION"
        self.progreso, self.esperas, self.n_replan = 0.0, 0, 0
        self.modo_refugio, self.inalcanzable = False, False
        self.ga_obj, self.ga_resumen = None, None
        self.ultimo = {}
        self.efectos, self.mensajes, self.transiciones = [], [], []
        self.version_mundo = 0
        self.zonas_dinamicas = {}           # celda → pasos restantes (polvo caliente de meteorito, derrumbe)
        self.cola_comandos = []
        self.violaciones, self.vetos_total = 0, 0
        self.t_decision = []
        self.sonda = self.world.modo == "sonda"
        # Vientos zonales permanentes en los gigantes (Neptuno tiene los más rápidos del sistema solar, ~2 000 km/h)
        self.viento_base = {"Júpiter": 0.25, "Saturno": 0.3, "Urano": 0.2, "Neptuno": 0.45}.get(self.world.planeta.nombre, 0.0) if self.sonda else 0.0
        self.ctx.update({"viento": self.viento_base, "viento_dir": "E", "viento_pasos": 0, "tornado": None, "temp_offset": 0.0,
                         "aislamiento": 0.15, "termico_pasos": 0, "acido_pasos": 0, "com_pasos": 0, "alien": None,
                         "tipo_termico": None})
        self._msg(f"Misión iniciada en {w.planeta.nombre} · política {politica} · semilla {w.semilla}")

    # ======================= utilidades =======================
    def _msg(self, texto, nivel="info"):
        self.mensajes.append({"paso": self.paso, "texto": texto, "nivel": nivel})

    def _set_estado(self, nuevo):
        if nuevo != self.rover.estado:
            self.transiciones.append({"paso": self.paso, "de": self.rover.estado, "a": nuevo})
            self.rover.estado = nuevo

    def _eventos_activos(self):
        c, act = self.ctx, []
        if c["tormenta_pasos"] > 0: act.append(f"tormenta ({c['tormenta_pasos']})")
        if c["viento_pasos"] > 0: act.append(f"viento {c['viento']:.1f} hacia {c['viento_dir']} ({c['viento_pasos']})")
        if c["tornado"]: act.append(f"tornado ({c['tornado']['pasos']})")
        if c["sismo"] > 0.05: act.append("sismo")
        if c["noche_forzada"]: act.append("noche forzada")
        if c["termico_pasos"] > 0: act.append(f"{c['tipo_termico']} ({c['termico_pasos']})")
        if c["acido_pasos"] > 0: act.append(f"lluvia ácida ({c['acido_pasos']})")
        if c["com_pasos"] > 0: act.append(f"baja conectividad ({c['com_pasos']})")
        if c["alien"]: act.append("objeto desconocido")
        for s_, t in self.sensores.fallos.items(): act.append(f"fallo {s_}")
        return act

    def _celda_cerca(self, dmin, dmax, condicion=None):
        r0, c0 = self.rover.pos
        cand = [(r, c) for r in range(self.world.tam) for c in range(self.world.tam)
                if dmin <= max(abs(r - r0), abs(c - c0)) <= dmax and (condicion is None or condicion((r, c)))]
        if not cand:
            return None
        return cand[int(self.rng_ev.integers(len(cand)))]

    def _t_ext(self):
        return max(-273.0, self.physics.temp_exterior(self.paso, self.ctx["noche_forzada"]) + self.ctx["temp_offset"])

    def _recalcular_zonas(self):
        antes = set(self.world.zonas_prohibidas)
        zonas = set(self.zonas_dinamicas)
        t = self.ctx["tornado"]
        if t:
            for r in range(self.world.tam):
                for c in range(self.world.tam):
                    if math.hypot(r - t["pos"][0], c - t["pos"][1]) <= t["radio"]:
                        zonas.add((r, c))
        a = self.ctx["alien"]
        if a:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (a["pos"][0] + dr, a["pos"][1] + dc)
                    if self.world.dentro(q):
                        zonas.add(q)
        zonas.discard(self.rover.pos) if not t else None
        self.world.zonas_prohibidas = zonas
        restante = set(self.plan[self._idx_plan():]) if self.plan else set()
        if (zonas - antes) & restante:
            self._pedir_replan("ZONA_PELIGROSA_EN_RUTA")

    def _pedir_replan(self, motivo):
        if not self.necesita_replan:
            self.necesita_replan, self.motivo_replan = True, motivo

    def _idx_plan(self):
        """Posición del rover dentro del plan, avanzando siempre hacia adelante (una ruta puede repetir celdas)."""
        i = getattr(self, "plan_i", 0)
        for j in range(i, len(self.plan)):
            if self.plan[j] == self.rover.pos:
                self.plan_i = j
                return j
        return -1

    # ======================= eventos (RF-09) =======================
    def inyectar(self, tipo, **kw):
        ok, razon = evento_aplica(tipo, self.world.planeta.nombre)
        if not ok:
            self._msg(f"⛔ {razon}", "error"); self.log.evento(self.paso, "rechazado", {"tipo": tipo, "razon": razon})
            return razon
        if self.ctx["com_pasos"] > 0 and tipo not in ("baja_conectividad",):
            self.cola_comandos.append({"paso_ejecucion": self.paso + 4, "tipo": tipo, "kw": kw})
            m = f"📡 Comando '{tipo}' en cola por baja conectividad: se ejecutará en el paso {self.paso + 4}."
            self._msg(m, "aviso"); return m
        return self._aplicar_evento(tipo, **kw)

    def _aplicar_evento(self, tipo, **kw):
        w, rv, c = self.world, self.rover, self.ctx
        pos = tuple(kw["pos"]) if kw.get("pos") is not None else None
        m = ""
        if tipo == "tormenta":
            c["tormenta_pasos"] = int(kw.get("duracion", 25))
            m = ("⛈️ Tormenta atmosférica (nubes de amoníaco): visibilidad 30 %, radar muy ruidoso." if self.sonda
                 else "🌫️ Tormenta de polvo: visibilidad 30 %, menos carga solar.")
        elif tipo == "viento":
            c["viento"] = float(kw.get("intensidad", 0.8)); c["viento_pasos"] = int(kw.get("duracion", 30))
            c["viento_dir"] = kw.get("dir") or str(self.rng_ev.choice(list(DIRS)))
            m = f"💨 Vientos fuertes (intensidad {c['viento']:.1f}) soplando hacia {c['viento_dir']}."
        elif tipo == "tornado":
            ini = pos or self._celda_cerca(5, 8) or (0, 0)
            ang = math.atan2(rv.pos[0] - ini[0], rv.pos[1] - ini[1]) + self.rng_ev.normal(0, 0.4)
            c["tornado"] = {"pos": [float(ini[0]), float(ini[1])], "vel": [0.35 * math.sin(ang), 0.35 * math.cos(ang)],
                            "radio": 1.6, "pasos": int(kw.get("duracion", 35))}
            m = (f"🌀 Vórtice atmosférico formado en {ini}, desplazándose hacia la sonda." if self.sonda else
                 f"🌪️ Tornado formado en {ini}, desplazándose hacia la zona del rover.")
        elif tipo == "meteorito":
            pos = pos or self._celda_cerca(2, 4) or rv.pos
            m = self._impacto_meteorito_sonda(pos) if self.sonda else self._impacto_meteorito(pos)
        elif tipo == "derrumbe":
            m = self._derrumbe(pos)
        elif tipo == "sismo":
            c["sismo"] = float(kw.get("intensidad", 0.9))
            w.elevacion += self.rng_ev.normal(0, 0.12, w.elevacion.shape)
            self.efectos.append({"tipo": "sismo", "paso": self.paso})
            m = "🌐 Sismo: vibración fuerte; el rover debe detenerse y volver a medir."
            if self.rng_ev.random() < 0.4:
                m += " " + self._derrumbe(None, causa="sismo")
            self.version_mundo += 1
        elif tipo == "noche":
            c["noche_forzada"] = True; m = "🌙 Noche forzada: sin carga solar."
        elif tipo == "dia":
            c["noche_forzada"] = False; m = "☀️ Ciclo día/noche normal."
        elif tipo in ("frio_extremo", "calor_extremo"):
            frio = tipo == "frio_extremo"
            c["temp_offset"] = -70.0 if frio else 80.0
            c["aislamiento"], c["termico_pasos"], c["tipo_termico"] = 0.45, int(kw.get("duracion", 30)), tipo
            m = "❄️ Frío extremo: la electrónica se enfría." if frio else "🔥 Calor extremo: riesgo de sobrecalentamiento."
        elif tipo == "lluvia_acida":
            c["acido_pasos"] = int(kw.get("duracion", 25)); m = "🧪 Ambiente ácido (hipotético): desgaste progresivo del chasis."
        elif tipo == "fallo_sensor":
            sensor = kw.get("sensor", "lidar_dz1")
            self.sensores.fallos[sensor] = kw.get("modo", "nan"); m = f"⚠️ Falla del sensor {sensor} (lecturas NaN)."
        elif tipo == "reparar_sensor":
            self.sensores.fallos.clear(); m = "🔧 Sensores restablecidos."
        elif tipo == "baja_conectividad":
            c["com_pasos"] = int(kw.get("duracion", 20)); m = "📡 Enlace degradado: comandos con 4 pasos de retraso; el rover opera autónomo."
        elif tipo == "biofirma":
            pend = [s for s in w.sitios if not s["recolectado"] and s.get("tipo") == "ciencia"]
            if not pend:
                return "No quedan sitios por analizar."
            s = pend[int(self.rng_ev.integers(len(pend)))]
            s["biofirma_simulada"], s["valor"] = True, s["valor"] + 6
            m = f"🧬 Señal anómala SIMULADA en {s['id']}: su prioridad científica sube a {s['valor']}. (No confirma vida.)"
        elif tipo == "objeto_desconocido":
            ini = pos or self._celda_cerca(3, 5, w.transitable) or (0, 0)
            c["alien"] = {"pos": list(ini), "pasos": int(kw.get("duracion", 40))}
            m = f"👽 Objeto desconocido en {ini} (evento narrativo ficticio): se trata como obstáculo móvil."
        elif tipo in ("barranco", "obstaculo"):
            pos = pos or ((rv.pos[0] + DIRS[rv.rumbo][0] * 2, rv.pos[1] + DIRS[rv.rumbo][1] * 2))
            if pos == rv.pos or not w.dentro(pos):
                return "Celda inválida (fuera del mapa o bajo el rover)."
            (w.agregar_barranco if tipo == "barranco" else w.agregar_obstaculo)(pos)
            w.dem_orbital[pos] = w.dem_orbital[pos]   # el DEM orbital NO se entera: el rover debe percibirlo
            self.version_mundo += 1
            if self.sonda:
                m = f"{'🌀 Corriente descendente' if tipo == 'barranco' else '⛈️ Celda de tormenta'} en {pos}. La sonda no lo sabe: debe detectarlo."
            else:
                m = f"{'🕳️ Barranco' if tipo == 'barranco' else '🪨 Roca'} agregado en {pos}. El rover no lo sabe: debe detectarlo."
        if EVENTOS[tipo][1]:
            self._pedir_replan(f"EVENTO_GRAVE_{tipo.upper()}")
        self.log.evento(self.paso, tipo, {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()})
        self._msg(m, "evento")
        self._recalcular_zonas()
        return m

    def _impacto_meteorito(self, pos):
        w, rv = self.world, self.rover
        self.efectos.append({"tipo": "meteorito", "pos": list(pos), "paso": self.paso})
        n = w.tam
        for r in range(max(0, pos[0] - 2), min(n, pos[0] + 3)):
            for c_ in range(max(0, pos[1] - 2), min(n, pos[1] + 3)):
                d = math.hypot(r - pos[0], c_ - pos[1])
                if d < 1.7:
                    w.elevacion[r, c_] -= 3.0 * (1 - (d / 1.7) ** 2)
                elif d < 2.3 and w.terreno[r, c_] == LIBRE and self.rng_ev.random() < 0.35 and (r, c_) != rv.pos:
                    w.terreno[r, c_] = OBSTACULO; w.elevacion[r, c_] += 0.8
                if d < 1.5 and (r, c_) != rv.pos:
                    self.zonas_dinamicas[(r, c_)] = 10
        if w.terreno[pos] != BARRANCO:
            w.terreno[pos] = BARRANCO; w.elevacion[pos] -= 2.0
        self.ctx["sismo"] = max(self.ctx["sismo"], 0.5)
        self.version_mundo += 1
        d_rover = math.hypot(rv.pos[0] - pos[0], rv.pos[1] - pos[1])
        if d_rover < 0.5:
            rv.vivo, rv.salud = False, 0.0
            return f"☄️ ¡Impacto directo de meteorito sobre el rover en {pos}!"
        if d_rover < 1.7:
            rv.salud = max(0.0, rv.salud - 40); self.violaciones += 1
            return f"☄️ Meteorito en {pos}, muy cerca del rover: daño por la onda de choque (salud {rv.salud:.0f} %)."
        return f"☄️ Meteorito impactó en {pos}: nuevo cráter, eyecta y zona caliente por 10 pasos."

    def _impacto_meteorito_sonda(self, pos):
        """En un gigante no hay suelo: el bólido explota en la atmósfera y deja una zona de onda de choque y plasma."""
        self.efectos.append({"tipo": "meteorito", "pos": list(pos), "paso": self.paso})
        for r in range(self.world.tam):
            for c_ in range(self.world.tam):
                if math.hypot(r - pos[0], c_ - pos[1]) < 2.2 and (r, c_) != self.rover.pos:
                    self.zonas_dinamicas[(r, c_)] = 12
        self.ctx["sismo"] = max(self.ctx["sismo"], 0.5)      # la onda de choque sacude la sonda
        d = math.hypot(self.rover.pos[0] - pos[0], self.rover.pos[1] - pos[1])
        if d < 2.2:
            self.rover.salud = max(0.0, self.rover.salud - 35); self.violaciones += 1
            return f"☄️ Bólido explotó en {pos}, junto a la sonda: daño por la onda de choque (salud {self.rover.salud:.0f} %)."
        return f"☄️ Bólido explotó en la atmósfera en {pos} (como Shoemaker-Levy 9 en Júpiter): zona de plasma por 12 pasos."

    def _derrumbe(self, pos, causa="ladera inestable"):
        w, rv = self.world, self.rover
        if pos is None:
            def pend_local(q):
                return max((abs(w.elevacion[q] - w.elevacion[v]) for _, v in w.vecinos(q)), default=0)
            cand = [(r, c_) for r in range(w.tam) for c_ in range(w.tam)
                    if 2 <= max(abs(r - rv.pos[0]), abs(c_ - rv.pos[1])) <= 5 and w.terreno[r, c_] == LIBRE]
            if not cand:
                return "No hay laderas cercanas para un derrumbe."
            cand.sort(key=pend_local, reverse=True)
            pos = cand[int(self.rng_ev.integers(min(4, len(cand))))]
        # dirección cuesta abajo
        vec = [(w.elevacion[pos] - w.elevacion[q], d) for d, q in w.vecinos(pos)]
        _, d_baja = max(vec)
        celdas = []
        for k in range(0, 4):
            base = (pos[0] + DIRS[d_baja][0] * k, pos[1] + DIRS[d_baja][1] * k)
            for lat in (-1, 0, 1):
                q = (base[0] + LATERAL[d_baja][0] * lat, base[1] + LATERAL[d_baja][1] * lat)
                if w.dentro(q) and w.terreno[q] != BARRANCO:
                    celdas.append((q, k))
        for q, k in celdas:
            if k == 0:
                w.elevacion[q] -= 1.5
            else:
                w.elevacion[q] += 1.0
                if q != rv.pos and w.terreno[q] == LIBRE and self.rng_ev.random() < 0.45:
                    w.terreno[q] = OBSTACULO; w.elevacion[q] += 0.6
            self.zonas_dinamicas[q] = max(self.zonas_dinamicas.get(q, 0), 5) if q != rv.pos else 0
        self.zonas_dinamicas = {q: t for q, t in self.zonas_dinamicas.items() if t > 0}
        self.efectos.append({"tipo": "derrumbe", "pos": list(pos), "celdas": [list(q) for q, _ in celdas], "paso": self.paso})
        self.version_mundo += 1
        self._pedir_replan("EVENTO_GRAVE_DERRUMBE")
        txt = f"⛰️ Derrumbe en {pos} por {causa}: rocas caídas hacia {d_baja}."
        if rv.pos in [q for q, _ in celdas]:
            rv.salud = max(0.0, rv.salud - 30); self.violaciones += 1
            txt += f" El rover fue alcanzado (salud {rv.salud:.0f} %)."
        return txt

    def _procesar_cola(self):
        listos = [cmd for cmd in self.cola_comandos if cmd["paso_ejecucion"] <= self.paso]
        self.cola_comandos = [cmd for cmd in self.cola_comandos if cmd["paso_ejecucion"] > self.paso]
        for cmd in listos:
            self._msg(f"📡 Comando retrasado '{cmd['tipo']}' recibido y ejecutado.", "aviso")
            self._aplicar_evento(cmd["tipo"], **cmd["kw"])

    def _mover_eventos(self):
        c, w = self.ctx, self.world
        t = c["tornado"]
        if t:
            t["vel"][0] += self.rng_ev.normal(0, 0.08); t["vel"][1] += self.rng_ev.normal(0, 0.08)
            for i in (0, 1):
                t["pos"][i] += t["vel"][i]
                if not 0 <= t["pos"][i] <= w.tam - 1:
                    t["vel"][i] *= -1; t["pos"][i] = float(np.clip(t["pos"][i], 0, w.tam - 1))
            t["pasos"] -= 1
            if t["pasos"] <= 0:
                c["tornado"] = None; self._msg("🌪️ El tornado se disipó.", "info")
        a = c["alien"]
        if a:
            opciones = [q for _, q in w.vecinos(tuple(a["pos"])) if w.transitable(q) and q != self.rover.pos]
            if opciones and self.rng_ev.random() < 0.6:
                a["pos"] = list(opciones[int(self.rng_ev.integers(len(opciones)))])
            a["pasos"] -= 1
            if a["pasos"] <= 0:
                c["alien"] = None; self._msg("👽 El objeto desconocido se alejó (registro ficticio cerrado).", "info")
        self._recalcular_zonas()

    # ======================= percepción (RNA) =======================
    def _percibir(self):
        rv, w = self.rover, self.world
        self.obs = {}
        modelo = self.rna_model if self.mod["rna"] else self.regla
        for d, q in w.vecinos(rv.pos):
            lect, band, crit = self.sensores.validar(self.sensores.leer(w, rv, self.physics, self.paso, self.ctx, rumbo=d))
            out = modelo.predecir(lect) if not self.mod["riesgo_cero"] else {"valido": False, "p_peligro": None}
            if self.mod["online"]:
                if out["valido"]:
                    self.belief[q] = 0.3 * self.belief[q] + 0.7 * out["p_peligro"]
                else:
                    self.belief[q] = max(self.belief[q], 0.5)
            lidar_ok = not math.isnan(lect.get("lidar_dz1", np.nan))
            if self.mod["supervisor"] and lidar_ok and lect["lidar_dz1"] < -3.0 and lect["visibilidad"] >= 0.6:
                if q not in self.confirmados:
                    self.confirmados.add(q); self.belief[q] = 1.0
                    if q in self.plan[max(self._idx_plan(), 0):]:
                        self._pedir_replan("BORDE_CONFIRMADO_EN_RUTA")
            self.obs[d] = (lect, band, crit, out)
        sig = self._siguiente()
        if sig and self.mod["online"] and self.belief[sig] > 0.6:
            self._pedir_replan("RIESGO_ALTO_EN_RUTA")

    # ======================= planificación (AG) =======================
    def _replanificar(self):
        rv, w = self.rover, self.world
        t0 = time.perf_counter()
        sitios = [] if self.modo_refugio else [s for s in w.sitios if not s["recolectado"]]
        bloq = set(self.confirmados) | set(w.zonas_prohibidas)
        bloq.discard(rv.pos)
        ga = GeneticPlanner(w, rv.pos, self.belief, rv.bateria, self.physics, capacidad_kg=rv.capacidad_kg - rv.carga_kg,
                            sitios=sitios, bloqueadas=bloq, pesos=self.pesos, poblacion=self.ga_pob,
                            generaciones=self.ga_gen if self.mod["ag"] else 1, semilla=w.semilla * 100 + self.n_replan,
                            consumo_paso=self._consumo_base_medio())
        if self.mod["ag"]:
            res = ga.evolucionar()
            crom, ev = res["mejor"]
            self.ga_obj = ga
        else:
            # Vecino más cercano + mochila codiciosa; se descartan sitios lejanos hasta que alcance la energía
            orden, p = [], rv.pos
            restantes = list(range(ga.n))
            while restantes:
                i = min(restantes, key=lambda j: abs(ga.sitios[j]["pos"][0] - p[0]) + abs(ga.sitios[j]["pos"][1] - p[1]))
                orden.append(i); restantes.remove(i); p = ga.sitios[i]["pos"]
            bits = [0] * ga.n
            masa = 0
            for i in orden:
                if masa + ga.sitios[i]["masa_kg"] <= ga.cap:
                    bits[i], masa = 1, masa + ga.sitios[i]["masa_kg"]
            crom = (orden, bits); ev = ga.evaluar(crom)
            while not ev["factible"] and any(bits):
                ultimo = [i for i in orden if bits[i]][-1]; bits[ultimo] = 0
                crom = (orden, bits); ev = ga.evaluar(crom)
            res = {"candidatos": [(crom, ev)], "hay_factible": ev["factible"]}
            self.ga_obj = None
        if not ev["factible"]:
            ev_ret = ga.evaluar((list(range(ga.n)), [0] * ga.n))
            solo_energia = ev_ret["refugio"] is not None and all(v.startswith("ENERGIA") for v in ev_ret["violaciones"])
            if ev_ret["factible"]:
                ev = ev_ret; self._msg("AG: ninguna ruta con muestras es segura → solo retorno al refugio.", "aviso")
                self.inalcanzable = False
            elif solo_energia and ev_ret["energia_pct"] <= rv.bateria:
                ev = ev_ret; self.inalcanzable = False; self.modo_refugio = True
                self._msg(f"🔋 Retorno de EMERGENCIA usando la reserva: se necesitan ≈{ev_ret['energia_pct']:.1f} % y hay {rv.bateria:.1f} %. "
                          "Se abandona la ciencia.", "aviso")
            else:
                ya_avisado = self.inalcanzable
                self.inalcanzable = True
                if ya_avisado:
                    pass
                elif ev_ret["refugio"] is None:
                    self._msg("❌ No existe camino seguro a ningún refugio con la información actual. Se espera y se vuelve a medir.", "error")
                else:
                    self._msg(f"❌ Refugio inalcanzable con la batería actual ({rv.bateria:.1f} %; se necesitan ≈{ev_ret['energia_pct'] + 5:.1f} %). "
                              "No se promete un rescate imposible: se espera recarga solar.", "error")
                ev = ev_ret
        else:
            self.inalcanzable = False
        self.plan = ev["ruta"] if ev["ruta"] else [rv.pos]
        self.plan_i = 0
        self.plan_visitas = ev["visitas"]
        self.n_replan += 1
        self.necesita_replan = False
        dt = time.perf_counter() - t0
        self.plan_info = {"n": self.n_replan, "motivo": self.motivo_replan, "visitas": ev["visitas"], "refugio": ev["refugio"],
                          "fitness": round(ev["fitness"], 4), "factible": ev["factible"], "energia_pct": ev["energia_pct"],
                          "t_s": round(dt, 3), "paso": self.paso}
        cands = [{"visitas": e["visitas"], "refugio": e["refugio"], "ruta": [list(x) for x in e["ruta"]],
                  **{k: e[k] for k in ("fitness", "ciencia", "distancia", "energia", "riesgo", "tiempo", "penalizacion", "factible")},
                  "violaciones": e["violaciones"]} for _, e in res["candidatos"][:3]]
        hist = [{"gen": h["gen"], "mejor": round(h["mejor"], 4), "media": round(h["media"], 4)} for h in ga.historial] if self.mod["ag"] else []
        self.ga_resumen = {"n": self.n_replan, "motivo": self.motivo_replan, "historial": hist, "candidatos": cands,
                           "poblacion": ga.N, "generaciones": len(ga.historial), "elegida": self.plan_info}
        self._msg(f"🧬 Replanificación #{self.n_replan} ({self.motivo_replan}): visitas {ev['visitas'] or '—'} → refugio {ev['refugio']} "
                  f"· fitness {ev['fitness']:.3f} · {dt * 1000:.0f} ms", "plan")
        self._set_estado("REPLANIFICACION")

    def _consumo_base_medio(self):
        """Consumo base por paso promediado en un sol (incluye calefacción si la noche o el día bajan de −30 °C)."""
        if self.sonda:
            return 0.1          # la sonda usa un generador de radioisótopos (RTG): su calor residual la calefacciona
        frio = sum(t < -30 for t in (self.physics.p.temp_dia + self.ctx["temp_offset"], self.physics.p.temp_noche + self.ctx["temp_offset"])) / 2
        return 0.1 + 0.3 * frio

    def _siguiente(self):
        i = self._idx_plan()
        if i < 0:
            return None
        return self.plan[i + 1] if i + 1 < len(self.plan) else None

    def _energia_para_volver(self):
        rv, w = self.rover, self.world
        mejor = None
        for r in w.refugios:
            camino = a_estrella_mapa(w, w.dem_orbital, self.belief, set(self.confirmados) | w.zonas_prohibidas - {rv.pos}, rv.pos, r, 15.0)
            if camino:
                e = sum(self.physics.costo_movimiento(w.dem_orbital[q] - w.dem_orbital[p], 0.5) * 0.5 + 0.2
                        for p, q in zip(camino[:-1], camino[1:])) * 2   # a v≈0,5: 2 pasos por celda
                mejor = e if mejor is None else min(mejor, e)
        return mejor

    # ======================= decisión: difuso + supervisor =======================
    def _decidir(self):
        rv, w, c = self.rover, self.world, self.ctx
        sig = self._siguiente()
        vetos, codigo = [], None
        rna_out, fz, vel = None, None, 0.0
        d = _dir_entre(rv.pos, sig) if sig else None
        lect = self.obs[d][0] if d else self.obs[rv.rumbo][0] if rv.rumbo in self.obs else next(iter(self.obs.values()))[0]
        if d:
            lect, band, crit, rna_out = self.obs[d]
            p_pel = rna_out.get("p_peligro") if rna_out else None
            if self.mod["difuso"]:
                dist = lect.get("proximidad_m", np.nan)
                for k in range(1, 5):
                    q = (rv.pos[0] + DIRS[d][0] * k, rv.pos[1] + DIRS[d][1] * k)
                    if not w.dentro(q) or q in self.confirmados or self.belief[q] > 0.5 or q in w.zonas_prohibidas:
                        db = (k - 0.5) * CELL_M
                        dist = db if math.isnan(dist) else min(dist, db); break
                if math.isnan(dist):
                    dist = 20.0                     # sin proximidad: se confía en el mapa de creencias (modo degradado)
                if p_pel is None:
                    p_pel = float(self.belief[sig])  # RNA sin entrada válida: se usa la memoria de riesgo
                fz = self.fuzzy.inferir(dist, rv.bateria, lect.get("pitch_deg", np.nan), p_pel)
                vel = fz["velocidad"]
            else:
                vel = 1.0 if self.politica == "control" else 0.7
        accion = "avanzar" if sig else "detener"
        sup = self.mod["supervisor"]
        lidar_nan = math.isnan(lect.get("lidar_dz1", np.nan))
        temp = rv.temp_interna

        def veto(cod):
            vetos.append(cod)

        # ---- prioridades fijas ----
        if not (self.physics.TEMP_INT_MIN <= temp <= self.physics.TEMP_INT_MAX):
            codigo, accion = "LIMITE_TERMICO_AISLAR_SUBSISTEMA", "detener"
        elif lidar_nan:
            codigo, accion = "SENSOR_CRITICO_LIDAR_MODO_SEGURO", "detener"
        elif c["sismo"] > 0.35:
            codigo, accion = "SISMO_ESPERAR_Y_REMEDIR", "esperar"
        elif sig is None:
            if rv.pos in w.refugios:
                codigo, accion = "EN_REFUGIO", "recargar"
            elif self.inalcanzable:
                codigo, accion = "REFUGIO_INALCANZABLE_ESPERA", "esperar"
            else:
                codigo, accion = "SIN_PLAN_REPLANIFICAR", "esperar"; self._pedir_replan("FIN_DE_PLAN_SIN_REFUGIO")
        elif self.inalcanzable and rv.pos not in w.refugios:
            codigo, accion = "REFUGIO_INALCANZABLE_HIBERNAR_Y_RECARGAR", "esperar"
            if self.paso % 5 == 0:
                self._pedir_replan("REEVALUAR_ENERGIA")
        elif sup:
            if sig in self.confirmados:
                veto("VETO_PRECIPICIO_CONFIRMADO")
            if sig in w.zonas_prohibidas:
                veto("VETO_ZONA_PROHIBIDA")
            if lect["lidar_dz1"] < -3.0:
                veto("VETO_BORDE_LIDAR"); self.belief[sig] = max(self.belief[sig], 0.9)
            if lect["pitch_deg"] > 25:
                veto("VETO_PENDIENTE"); self.belief[sig] = max(self.belief[sig], 0.8)
            if w.roll_deg(sig, d) > 25 and w.terreno[sig] == LIBRE:
                veto("VETO_ALABEO")
            if rna_out and rna_out.get("valido") and rna_out["p_peligro"] > 0.6:
                veto("VETO_RNA_PELIGRO")
            if vetos:
                codigo, accion = vetos[0], "detener"
                self.vetos_total += 1
                self._pedir_replan(vetos[0])
            else:
                codigo = "AVANZAR_SEGUN_PLAN"
                if rna_out is not None and not rna_out.get("valido", True):
                    vel, codigo = min(vel, 0.25), "RNA_INVALIDA_MODO_DEGRADADO"
                elif rna_out and rna_out.get("valido") and rna_out["confianza"] < 0.6:
                    vel, codigo = min(vel, 0.3), "RNA_BAJA_CONFIANZA_MODO_CONSERVADOR"
                if c["viento"] > 0.5:
                    vel = min(vel, 0.35); codigo = "VIENTO_LIMITA_VELOCIDAD"
                    deriva = (sig[0] + DIRS[c["viento_dir"]][0], sig[1] + DIRS[c["viento_dir"]][1])
                    if (not w.dentro(deriva) or deriva in self.confirmados or self.belief[deriva] > 0.5) and _dir_entre(rv.pos, sig) != c["viento_dir"]:
                        codigo, accion = "VIENTO_RIESGO_DERIVA_ESPERAR", "esperar"
                t = c["tornado"]
                if t and math.hypot(rv.pos[0] - t["pos"][0], rv.pos[1] - t["pos"][1]) < t["radio"] + 3:
                    codigo = "TORNADO_CERCANO_ALEJARSE"
                if rv.salud < 30 and not self.modo_refugio:
                    self.modo_refugio = True; self._pedir_replan("DANO_CRITICO_VOLVER")
                    self._msg("🛠️ Daño crítico: se abandona la ciencia y se vuelve al refugio.", "aviso")
                if not self.modo_refugio and self.paso % 3 == 0:
                    e = self._energia_para_volver()
                    if e is not None and rv.bateria < e * 1.3 + 4:
                        self.modo_refugio = True; self._pedir_replan("BATERIA_BAJA_IR_A_REFUGIO")
                        self._msg(f"🔋 Batería {rv.bateria:.1f} % ≈ lo necesario para volver ({e:.1f} %): modo REFUGIO.", "aviso")
                if accion == "avanzar" and vel < 0.1:
                    codigo, accion = "FUZZY_ORDENA_DETENER", "esperar"
        else:
            codigo = "AVANZAR_SIN_SUPERVISOR"
        if accion != "avanzar":
            self.esperas += 1
            if self.esperas >= 3 and sig is not None and codigo in ("FUZZY_ORDENA_DETENER", "VIENTO_RIESGO_DERIVA_ESPERAR"):
                self.belief[sig] = max(self.belief[sig], 0.7); self._pedir_replan("BLOQUEO_PERSISTENTE"); self.esperas = 0
        else:
            self.esperas = 0

        # ---- explicación legible ----
        partes = []
        if rna_out and rna_out.get("valido"):
            partes.append(f"RNA: celda {sig} = {rna_out['clase']} ({rna_out['confianza']:.0%})")
        elif rna_out is not None and d:
            partes.append("RNA: entrada inválida")
        if fz:
            top = sorted([r for r in fz["reglas"] if r["activacion"] > 0], key=lambda r: -r["activacion"])[:2]
            partes.append(f"Difuso: v={fz['velocidad']:.2f} ({fz['cautela']}) por " + ", ".join(f"{r['id']}({r['activacion']:.2f})" for r in top))
        partes.append(f"AG: plan #{self.n_replan} {self.plan_visitas or 'retorno'}")
        partes.append(f"Supervisor: {codigo}" + (f" (vetos: {', '.join(vetos)})" if vetos else ""))
        return {"accion": accion, "rumbo": d or rv.rumbo, "velocidad": round(vel if accion == "avanzar" else 0.0, 3),
                "motivo": codigo, "vetos": vetos, "siguiente": sig, "rna": rna_out, "fuzzy": fz, "lect": lect,
                "explicacion": " · ".join(partes)}

    # ======================= física =======================
    def _mover_a(self, destino, rumbo, forzado=False):
        rv, w = self.rover, self.world
        if not w.dentro(destino):
            return "BORDE_DEL_MAPA"
        dz = w.z(destino) - w.z(rv.pos)
        rv.rumbo = rumbo
        t = int(w.terreno[destino])
        if t == BARRANCO:
            rv.pos, rv.vivo, rv.salud = destino, False, 0.0; self.violaciones += 1
            return "ARRASTRADA_POR_CORRIENTE_DESCENDENTE" if self.sonda else "CAIDA_EN_BARRANCO"
        if t == OBSTACULO:
            rv.salud = max(0.0, rv.salud - 20); self.violaciones += 1
            if rv.salud <= 0: rv.vivo = False
            self.confirmados.add(destino); self.belief[destino] = 1.0; self._pedir_replan("COLISION")
            return "TURBULENCIA_CELDA_DE_TORMENTA" if self.sonda else "COLISION"
        pitch, roll = w.pitch_deg(rv.pos, destino), w.roll_deg(destino, rumbo)
        rv.pos = destino
        rv.odometro_m += math.hypot(CELL_M, dz)
        info = ""
        if destino in w.zonas_prohibidas:
            rv.salud = max(0.0, rv.salud - 15); self.violaciones += 1; info = "INGRESO_ZONA_PELIGROSA"
        if max(pitch, roll) > LIM_VUELCO_DEG and not self.sonda:
            rv.vivo, rv.salud = False, 0.0; self.violaciones += 1
            return f"VUELCO(pitch={pitch:.0f}°,roll={roll:.0f}°)"
        return info

    def _ejecutar(self, dec):
        rv, w, ph, c = self.rover, self.world, self.physics, self.ctx
        consumo, info = 0.1, []
        if dec["accion"] == "avanzar" and dec["siguiente"] is not None and rv.bateria > 0:
            sig, v = dec["siguiente"], dec["velocidad"]
            if dec["rumbo"] != rv.rumbo:
                self.progreso = 0.0
            rv.rumbo = dec["rumbo"]
            dz = w.z(sig) - w.z(rv.pos)
            consumo += ph.costo_movimiento(dz, v) * v + 0.3 * c["viento"] * v
            self.progreso += v
            if self.progreso >= 1.0:
                self.progreso = 0.0
                r = self._mover_a(sig, dec["rumbo"])
                if r: info.append(r)
                if rv.vivo and c["viento"] > 0 and self.rng_ev.random() < 0.35 * c["viento"] * (0.4 + v):
                    q = (rv.pos[0] + DIRS[c["viento_dir"]][0], rv.pos[1] + DIRS[c["viento_dir"]][1])
                    r2 = self._mover_a(q, rv.rumbo, forzado=True)
                    info.append("DERIVA_POR_VIENTO" + (f"→{r2}" if r2 else ""))
                    self._pedir_replan("DESVIO_POR_VIENTO")
        else:
            self.progreso = 0.0
            if dec["accion"] == "recargar":
                rv.bateria = min(100.0, rv.bateria + 5.0)
        # tornado: daño y arrastre si el rover queda dentro
        t = c["tornado"]
        if t and rv.vivo and math.hypot(rv.pos[0] - t["pos"][0], rv.pos[1] - t["pos"][1]) <= t["radio"]:
            rv.salud = max(0.0, rv.salud - 12); self.violaciones += 1
            d = str(self.rng_ev.choice(list(DIRS)))
            r = self._mover_a((rv.pos[0] + DIRS[d][0], rv.pos[1] + DIRS[d][1]), rv.rumbo, forzado=True)
            info.append("ARRASTRE_TORNADO" + (f"→{r}" if r else "")); self._pedir_replan("ARRASTRE_TORNADO")
        if c["acido_pasos"] > 0:
            rv.salud = max(0.0, rv.salud - 1.5)
        if rv.salud <= 0:
            rv.vivo = False
        # muestras
        for s in w.sitios:
            if s["pos"] == rv.pos and not s["recolectado"] and rv.vivo and s["id"] in self.plan_visitas:
                if rv.carga_kg + s["masa_kg"] <= rv.capacidad_kg:
                    s["recolectado"] = True; rv.carga_kg += s["masa_kg"]; rv.muestras.append(s["id"])
                    if s.get("tipo") == "instrumento":
                        rv.bateria = min(100.0, rv.bateria + 5.0)
                        info.append(f"INSTRUMENTO_{s['id']}_RECUPERADO")
                        self._msg(f"📦 Instrumentación {s['id']} recuperada (1 kg). Trae una batería de respaldo: +5 %.", "exito")
                    else:
                        info.append(f"MUESTRA_{s['id']}" + ("_SEÑAL_ANOMALA_SIMULADA" if s["biofirma_simulada"] else ""))
                        nombre = "Muestra atmosférica" if self.sonda else "Muestra"
                        self._msg(f"🧪 {nombre} {s['id']} recolectada (valor {s['valor']}, {s['masa_kg']} kg)"
                                  + (" · señal anómala simulada: requiere análisis, NO confirma vida." if s["biofirma_simulada"] else ""), "exito")
                    self._pedir_replan("OBJETIVO_ALCANZADO")
        # energía y temperatura
        de_dia = ph.es_de_dia(self.paso, c["noche_forzada"])
        t_ext = self._t_ext()
        hiberna = dec["accion"] != "avanzar" and (self.inalcanzable or (dec["motivo"] or "").startswith(("LIMITE_TERMICO", "SENSOR_CRITICO")))
        if hiberna:
            consumo = 0.03                      # hibernación: solo electrónica mínima
        if t_ext < -30 and not self.sonda:
            consumo += 0.1 if hiberna else 0.3
        # Sonda: lejos del Sol los paneles casi no rinden; un RTG (supuesto) entrega +0,08 %/paso constante
        recarga = 0.08 if self.sonda else ph.recarga_solar(de_dia, c["visibilidad"])
        rv.bateria = float(np.clip(rv.bateria - consumo + recarga, 0.0, 100.0))
        objetivo = 20.0 + c["aislamiento"] * (t_ext - 20.0)
        rv.temp_interna += 0.3 * (objetivo - rv.temp_interna)
        return " ".join(info)

    def actualizar_mundo(self):
        c = self.ctx
        for k in ("tormenta_pasos", "viento_pasos", "termico_pasos", "acido_pasos", "com_pasos"):
            if c[k] > 0:
                c[k] -= 1
                if c[k] == 0:
                    if k == "viento_pasos": c["viento"] = self.viento_base
                    if k == "termico_pasos": c["temp_offset"], c["aislamiento"], c["tipo_termico"] = 0.0, 0.15, None
                    self._msg(f"Fin del evento: {k.replace('_pasos', '')}.", "info")
        vis = 1.0
        if c["tormenta_pasos"] > 0: vis = 0.3
        if c["viento"] > 0 and self.world.planeta.nombre in ("Marte", "Tierra"): vis = min(vis, 1 - 0.3 * c["viento"])
        if c["acido_pasos"] > 0: vis = min(vis, 0.6)
        t = c["tornado"]
        if t and math.hypot(self.rover.pos[0] - t["pos"][0], self.rover.pos[1] - t["pos"][1]) < t["radio"] + 2:
            vis = min(vis, 0.4)
        c["visibilidad"] = vis
        c["sismo"] = max(0.0, c["sismo"] - 0.25)
        self.zonas_dinamicas = {q: n - 1 for q, n in self.zonas_dinamicas.items() if n > 1}

    def _evaluar_fin(self, info):
        rv, w = self.rover, self.world
        if not rv.vivo:
            self._set_estado("FRACASO"); self._msg(f"💥 Misión fallida: {info or 'daño total'}.", "error")
        elif (rv.pos in w.refugios and self._siguiente() is None and not self.necesita_replan and self.paso > 0
              and all(x["recolectado"] for x in w.sitios if x["id"] in self.plan_visitas)):
            self._set_estado("EXITO")
            self._msg(f"🏁 Misión completada: {len(rv.muestras)} muestras, batería {rv.bateria:.1f} %.", "exito")
        elif self.paso + 1 >= self.max_pasos:
            self._set_estado("FRACASO"); self._msg("⌛ Tiempo de misión agotado.", "error")
        self.terminada = rv.estado in ("EXITO", "FRACASO")

    def _estado_operativo(self, dec):
        cod = dec["motivo"] or ""
        if cod.startswith(("LIMITE_TERMICO", "SENSOR_CRITICO")) or cod.startswith("REFUGIO_INALCANZABLE"):
            return "MODO_SEGURO"
        if self.modo_refugio:
            return "REFUGIO"
        if self.rover.estado == "REPLANIFICACION" and self.plan_info.get("paso") == self.paso:
            return "REPLANIFICACION"
        r = dec.get("rna") or {}
        if dec["vetos"] or cod.startswith(("SISMO", "VIENTO", "TORNADO", "RNA_", "FUZZY")) or (r.get("valido") and r["p_peligro"] > 0.4):
            return "ALERTA"
        return "EXPLORACION"

    # ======================= un paso completo =======================
    def step(self):
        if self.terminada:
            return None
        t0 = time.perf_counter()
        rv = self.rover
        self._procesar_cola()
        self._mover_eventos()
        if rv.bateria <= 0 and rv.pos not in self.world.refugios:
            self._set_estado("FRACASO"); self.terminada = True
            self._msg("🔋 Batería agotada fuera del refugio: el rover no puede moverse.", "error")
            fila = {"paso": self.paso, "fila": rv.pos[0], "col": rv.pos[1], "estado": "FRACASO", "accion": "detener",
                    "velocidad": 0.0, "motivo": "BATERIA_AGOTADA", "resultado": "FIN_MISION", "bateria_pct": 0.0}
            self.log.registrar(fila); self.paso += 1
            return fila
        if rv.estado == "REPLANIFICACION":
            rv.estado = "EXPLORACION"
        self._percibir()
        if self.necesita_replan:
            self._replanificar()
        dec = self._decidir()
        if self.necesita_replan and dec["vetos"]:
            self._replanificar()
        info = self._ejecutar(dec)
        self.actualizar_mundo()
        self.ruta_real.append(rv.pos)
        if rv.vivo:
            self._set_estado(self._estado_operativo(dec))
        self._evaluar_fin(info)
        dt = time.perf_counter() - t0
        self.t_decision.append(dt)
        self.ultimo = dec
        r, fz, lect = dec["rna"] or {}, dec["fuzzy"] or {}, dec["lect"]
        fila = {"paso": self.paso, "fila": rv.pos[0], "col": rv.pos[1], "z_m": round(self.world.z(rv.pos), 2), "rumbo": rv.rumbo,
                "estado": rv.estado, "accion": dec["accion"], "velocidad": dec["velocidad"], "motivo": dec["motivo"],
                "vetos": "|".join(dec["vetos"]), "resultado": info, "explicacion": dec["explicacion"],
                "bateria_pct": round(rv.bateria, 2), "temp_int_c": round(rv.temp_interna, 2), "temp_ext_c": round(self._t_ext(), 1),
                "salud_pct": round(rv.salud, 1), "muestras": "|".join(rv.muestras),
                "rna_clase": r.get("clase"), "rna_confianza": round(r.get("confianza") or 0, 3),
                "p_libre": (r.get("probs") or {}).get("libre"), "p_obstaculo": (r.get("probs") or {}).get("obstaculo"),
                "p_barranco": (r.get("probs") or {}).get("barranco"),
                "fz_distancia": (fz.get("entradas") or {}).get("distancia"), "fz_velocidad": fz.get("velocidad"),
                "fz_reglas": "|".join(fz.get("reglas_activas", [])), "fz_cautela": fz.get("cautela"),
                "plan_n": self.n_replan, "plan_visitas": "→".join(self.plan_visitas), "ga_fitness": self.plan_info.get("fitness"),
                "eventos": "|".join(self._eventos_activos()), "visibilidad": round(self.ctx["visibilidad"], 2),
                "s_proximidad_m": lect.get("proximidad_m"), "s_lidar_dz1": lect.get("lidar_dz1"), "s_pitch_deg": lect.get("pitch_deg"),
                "t_decision_ms": round(dt * 1000, 2)}
        self.log.registrar(fila)
        self.paso += 1
        return fila

    # ======================= métricas =======================
    def metricas(self):
        d = self.log.df()
        ciencia_total = sum(s["valor"] for s in self.world.sitios)
        ciencia = sum(s["valor"] for s in self.world.sitios if s["recolectado"])
        return {"politica": self.politica, "semilla": self.world.semilla, "exito": self.rover.estado == "EXITO",
                "cayo_barranco": bool(d.resultado.astype(str).str.contains("CAIDA_EN_BARRANCO").any()),
                "vuelco": bool(d.resultado.astype(str).str.contains("VUELCO").any()),
                "colisiones": int(d.resultado.astype(str).str.contains("COLISION").sum()),
                "violaciones_seguridad": self.violaciones, "vetos": self.vetos_total, "replanificaciones": self.n_replan,
                "ciencia_pct": 100 * ciencia / ciencia_total if ciencia_total else 0.0, "muestras": len(self.rover.muestras),
                "instrumentos": sum(1 for s in self.world.sitios if s.get("tipo") == "instrumento" and s["recolectado"]),
                "bateria_final": self.rover.bateria, "distancia_m": self.rover.odometro_m, "pasos": len(d),
                "t_decision_ms": 1000 * float(np.mean(self.t_decision)) if self.t_decision else 0.0}

# Escenario B de los experimentos: eventos fijos en pasos fijos (condiciones idénticas para todas las políticas)
EVENTOS_B = {15: ("tormenta", {"duracion": 20}), 30: ("meteorito", {}), 45: ("viento", {"intensidad": 0.8, "duracion": 25})}


def correr_mision_metricas(politica, semilla, escenario="A", planeta="Marte", bateria=80, ga_pob=30, ga_gen=30, max_pasos=400):
    """Corre una misión completa sin interfaz y devuelve sus métricas (la usan los experimentos y el panel 'Comparar')."""
    t0 = time.perf_counter()
    s = HybridSimulation(rna, fuzzy, politica=politica, planeta=planeta, semilla=semilla, bateria_inicial=bateria,
                         ga_pob=ga_pob, ga_gen=ga_gen, max_pasos=max_pasos)
    while not s.terminada:
        if escenario == "B" and s.paso in EVENTOS_B:
            tipo, kw = EVENTOS_B[s.paso]; s.inyectar(tipo, **kw)
        if s.step() is None:
            break
    return {**s.metricas(), "escenario": escenario, "t_mision_s": round(time.perf_counter() - t0, 2)}

# ### 18.1 Demo del ciclo híbrido con eventos (CA-01 y CA-02)
#
# Primero se colocan un barranco y una roca **justo en la ruta** del rover. Como el mapa orbital no los conoce, el rover
# debe descubrirlos con la RNA y el LIDAR, frenar y replanificar. Después se inyectan una tormenta y un meteorito.

# ### 18.2 Pruebas causales controladas: cada técnica cambia la conducta
#
# Misma semilla y misma situación; solo se altera **una salida** a la vez.

# ### 18.3 Energía crítica (CA-03): batería inicial 12 % y 3 %

# ### 18.4 Modo sonda atmosférica (CA-07 completo) y recuperación de instrumentación
#
# En Júpiter, Saturno, Urano y Neptuno no hay suelo: el rover de ruedas se **bloquea** con una explicación científica y
# se despliega una **sonda aerostática** que flota a ~1 bar. Usa **las mismas tres técnicas**:
#
# | Rover (superficie) | Sonda (atmósfera) |
# |---|---|
# | barranco (caída mortal) | corriente descendente (hunde la sonda hacia presiones aplastantes) |
# | roca (colisión) | torre de tormenta / cumulonimbo (turbulencia, daño) |
# | refugio | punto de relevo de comunicaciones |
# | energía solar + calefacción | generador de radioisótopos (RTG): +0,08 %/paso y calor residual |
# | viento solo como evento | vientos zonales permanentes (Neptuno: los más rápidos, intensidad 0,45) |
# | derrumbe y sismo | **no aplican** (sin corteza sólida): se rechazan con explicación |
#
# Todos los valores son supuestos de simulación.

# ### 18.5 Pruebas de borde de la integración

# ---
# # PASO 4 (Hito H4): Centro de Control en HTML, en tiempo real
#
# El motor (simulación 3D + RNA + difuso + AG + supervisor) **corre en Python**. Un servidor HTTP mínimo, hecho solo
# con la biblioteca estándar, entrega el panel HTML y responde a los botones mediante `POST /api`. Cada clic del usuario
# llega al motor Python, que avanza la simulación y devuelve el estado. El panel solo **dibuja**: no decide nada.
#
# - **En Colab:** ejecute la celda de la sección 20 y haga clic en el enlace "Abrir Centro de Control". Se abre una
#   pestaña nueva a pantalla completa, ideal para presentar.
# - **En su computador:** `python ares_x.py` y abra http://localhost:8765.
#
# **API** (`POST /api` con `{"cmd": ..., "args": {...}}`): `estado`, `reiniciar`, `paso`, `evento`, `modulos`, `pesos`,
# `ga_generacion`, `lab_difuso`, `lab_rna`, `exportar`.

import threading


def _limpiar_json(o):
    """Convierte tipos numpy y NaN/inf a tipos JSON válidos (NaN → null)."""
    if isinstance(o, dict):
        return {str(k): _limpiar_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_limpiar_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return _limpiar_json(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


class CentroControl:
    def __init__(self, rna, fuzzy, ga_pob=40, ga_gen=40):
        self.rna, self.fuzzy = rna, fuzzy
        self.ga_pob, self.ga_gen = ga_pob, ga_gen
        self.lock = threading.Lock()
        self.config = dict(planeta="Marte", modo="rover", semilla=42, bateria_inicial=80, dificultad="media",
                           politica="hibrido", pesos=None)
        self.sim = None
        self.reiniciar({})

    # ---------- metadatos para construir la interfaz ----------
    def meta(self):
        pl = self.config["planeta"]
        return {"version": VERSION, "planetas": {k: {"superficie": v.superficie_solida, "gravedad": v.gravedad, "nota": v.nota,
                                                      "modos": v.modos_permitidos()} for k, v in PLANETAS.items()},
                "eventos": {t: {"aplica": evento_aplica(t, pl)[0], "razon": evento_aplica(t, pl)[1], "descripcion": d[2], "grave": d[1]}
                            for t, d in EVENTOS.items()},
                "politicas": list(POLITICAS), "config": self.config,
                "fuzzy": {"entradas": {v: {"universo": i["universo"], "unidad": i["unidad"], "conjuntos": i["conjuntos"]}
                                       for v, i in self.fuzzy.ENTRADAS.items()},
                          "salida": self.fuzzy.SALIDA, "reglas": [self.fuzzy.texto_regla(r) for r in self.fuzzy.REGLAS]},
                "metricas_rna": METRICAS_RNA, "pesos": GeneticPlanner.PESOS_DEF, "rna_info": self._rna_info(),
                "planetas_info": {k: {"gravedad": v.gravedad, "temp_dia": v.temp_dia, "temp_noche": v.temp_noche,
                                      "presion_bar": v.presion_bar, "superficie": v.superficie_solida, "nota": v.nota}
                                  for k, v in PLANETAS.items()}}

    def _rna_info(self):
        """Datos del entrenamiento para el Laboratorio RNA (calculados una sola vez, en el entrenamiento offline)."""
        if getattr(self, "_rna_cache", None) is None:
            try:
                self._rna_cache = {"arquitectura": [len(RASGOS_RNA), *self.rna.ocultas, 3], "rasgos": RASGOS_RNA,
                                   "rangos": {k: SensorSuite.RANGOS[k] for k in RASGOS_RNA}, "clases": CLASES,
                                   "matriz_confusion": confusion_matrix(yte, pred_rna).tolist(),
                                   "historial": hist[["epoca", "loss", "acc_train", "acc_valid", "f1_valid"]].round(4).to_dict("list"),
                                   "comparacion": comparacion.to_dict("records"), "mejor_epoca": self.rna.mejor_epoca,
                                   "n": {k: len(v) for k, v in datos.items()}}
            except NameError:
                self._rna_cache = {}
        return self._rna_cache

    def reiniciar(self, cfg):
        cfg = {**self.config, **{k: v for k, v in cfg.items() if v is not None}}
        pl, modo = cfg["planeta"], cfg["modo"]
        if pl not in PLANETAS:
            return {"ok": False, "error": f"Planeta desconocido: {pl}"}
        ok, msg = PLANETAS[pl].validar_modo(modo)
        if not ok:
            return {"ok": False, "error": msg, "sugerencia": "sonda"}
        try:
            semilla = int(cfg["semilla"]); bat = float(cfg["bateria_inicial"])
            sim = HybridSimulation(self.rna, self.fuzzy, politica=cfg["politica"], planeta=pl, modo=modo, semilla=semilla,
                                   bateria_inicial=bat, dificultad=cfg["dificultad"], pesos=cfg.get("pesos"),
                                   ga_pob=self.ga_pob, ga_gen=self.ga_gen)
        except (ValueError, NotImplementedError, TypeError) as e:
            return {"ok": False, "error": str(e)}
        self.sim, self.config = sim, cfg
        sim.config["pesos"] = cfg.get("pesos")
        sim._percibir(); sim._replanificar()        # plan inicial visible antes de empezar
        sim._set_estado("EXPLORACION")
        self._msg_i, self._ef_i, self._v_mundo, self._v_ga = 0, 0, -1, -1
        return {"ok": True}

    # ---------- estado para dibujar ----------
    def estado(self, completo=False):
        s, w, rv = self.sim, self.sim.world, self.sim.rover
        dec = s.ultimo or {}
        fz = dec.get("fuzzy")
        sig = s._siguiente()
        d = _dir_entre(rv.pos, sig) if sig else None
        out = {
            "paso": s.paso, "estado": rv.estado, "terminada": s.terminada, "planeta": w.planeta.nombre, "modo": w.modo,
            "politica": s.politica, "modulos": s.mod, "pesos": s.pesos, "semilla": w.semilla,
            "dia": s.physics.es_de_dia(s.paso, s.ctx["noche_forzada"]), "t_ext": round(s._t_ext(), 1),
            "rover": {"pos": list(rv.pos), "z": round(w.z(rv.pos), 2), "rumbo": rv.rumbo, "bateria": round(rv.bateria, 2),
                      "temp_int": round(rv.temp_interna, 1), "salud": round(rv.salud, 1), "carga": round(rv.carga_kg, 2),
                      "capacidad": rv.capacidad_kg, "muestras": rv.muestras, "progreso": round(s.progreso, 3),
                      "dir_mov": d, "vivo": rv.vivo, "odometro": round(rv.odometro_m, 1),
                      "pitch": round(w.pitch_deg(rv.pos, sig), 1) if sig else 0.0, "roll": round(w.roll_deg(rv.pos, rv.rumbo), 1)},
            "ctx": {"visibilidad": s.ctx["visibilidad"], "viento": s.ctx["viento"], "viento_dir": s.ctx["viento_dir"],
                    "tornado": s.ctx["tornado"], "alien": s.ctx["alien"], "sismo": s.ctx["sismo"],
                    "noche_forzada": s.ctx["noche_forzada"], "com_pasos": s.ctx["com_pasos"], "tormenta": s.ctx["tormenta_pasos"],
                    "termico": s.ctx["tipo_termico"], "acido": s.ctx["acido_pasos"], "fallos": list(s.sensores.fallos)},
            "eventos_activos": s._eventos_activos(), "cola": len(s.cola_comandos),
            "belief": np.round(s.belief, 2), "confirmados": [list(q) for q in s.confirmados],
            "zonas": [list(q) for q in w.zonas_prohibidas], "plan": [list(q) for q in s.plan],
            "plan_info": s.plan_info, "ruta_real": [list(q) for q in s.ruta_real[-500:]],
            "sitios": [{"id": x["id"], "pos": list(x["pos"]), "valor": x["valor"], "masa": x["masa_kg"], "ok": x["recolectado"],
                        "tipo": x.get("tipo", "ciencia"),
                        "bio": x["biofirma_simulada"], "en_plan": x["id"] in s.plan_visitas} for x in w.sitios],
            "decision": {"accion": dec.get("accion"), "velocidad": dec.get("velocidad"), "motivo": dec.get("motivo"),
                         "vetos": dec.get("vetos", []), "explicacion": dec.get("explicacion", "Esperando el primer paso…"),
                         "siguiente": list(dec["siguiente"]) if dec.get("siguiente") else None},
            "rna": None, "fuzzy": None,
            "sensores": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in (dec.get("lect") or {}).items() if k != "pos_estimada"},
            "metricas": {"violaciones": s.violaciones, "vetos": s.vetos_total, "replan": s.n_replan,
                         "t_decision_ms": round(1000 * s.t_decision[-1], 1) if s.t_decision else None},
            "transiciones": s.transiciones[-8:],
        }
        r = dec.get("rna")
        if r and r.get("valido"):
            out["rna"] = {"probs": r["probs"], "clase": r["clase"], "confianza": r["confianza"], "p_peligro": r["p_peligro"],
                          "normalizada": r.get("normalizada"), "celda": out["decision"]["siguiente"]}
        elif r:
            out["rna"] = {"invalida": True, "faltan": r.get("faltan")}
        if fz:
            out["fuzzy"] = {"entradas": fz["entradas"], "membresias": fz["membresias"], "velocidad": fz["velocidad"],
                            "cautela": fz["cautela"], "reglas": fz["reglas"], "avisos": fz["avisos"],
                            "agregada": [round(float(v), 3) for v in fz["agregada"][::4]]}
        if completo or s.version_mundo != self._v_mundo:
            out["mundo"] = {"tam": w.tam, "elevacion": np.round(w.elevacion, 2), "terreno": w.terreno,
                            "refugios": [list(x) for x in w.refugios], "inicio": list(w.inicio), "version": s.version_mundo}
            self._v_mundo = s.version_mundo
        if s.ga_resumen and (completo or s.n_replan != self._v_ga):
            out["ga"] = s.ga_resumen; self._v_ga = s.n_replan
        out["mensajes"] = s.mensajes[0 if completo else self._msg_i:]
        out["efectos"] = s.efectos[self._ef_i:]
        self._msg_i, self._ef_i = len(s.mensajes), len(s.efectos)
        if completo:
            out["meta"] = self.meta()
        return out

    # ---------- comandos ----------
    def ejecutar(self, cmd, args):
        with self.lock:
            s = self.sim
            if cmd == "estado":
                return self.estado(completo=True)
            if cmd == "reiniciar":
                r = self.reiniciar(args or {})
                if not r["ok"]:
                    return {"error": r["error"], "sugerencia": r.get("sugerencia")}
                return self.estado(completo=True)
            if cmd == "paso":
                for _ in range(max(1, min(int(args.get("n", 1)), 10))):
                    if s.step() is None:
                        break
                return self.estado()
            if cmd == "evento":
                a = dict(args); tipo = a.pop("tipo")
                if a.get("pos") is not None:
                    a["pos"] = tuple(int(x) for x in a["pos"])
                msg = s.inyectar(tipo, **a)
                out = self.estado(); out["respuesta"] = msg
                return out
            if cmd == "modulos":
                for k in ("rna", "difuso", "ag", "supervisor"):
                    if k in args:
                        s.mod[k] = bool(args[k])
                s.mod["online"] = s.mod["rna"] or s.mod["supervisor"]
                s._pedir_replan("CAMBIO_DE_MODULOS")
                s._msg(f"⚙️ Módulos: RNA {'ON' if s.mod['rna'] else 'OFF'} · Difuso {'ON' if s.mod['difuso'] else 'OFF'} · "
                       f"AG {'ON' if s.mod['ag'] else 'OFF'} · Supervisor {'ON' if s.mod['supervisor'] else 'OFF'}", "aviso")
                return self.estado()
            if cmd == "pesos":
                s.pesos.update({k: float(v) for k, v in args.items() if k in GeneticPlanner.PESOS_DEF})
                self.config["pesos"] = dict(s.pesos)
                s._pedir_replan("CAMBIO_DE_PRIORIDADES"); s._msg(f"🎯 Nuevos pesos del fitness: {s.pesos}", "aviso")
                return self.estado()
            if cmd == "ga_generacion":
                if s.ga_obj is None:
                    return {"error": "El AG está desactivado o aún no se ejecuta."}
                g = int(args.get("g", 0))
                tabla = s.ga_obj.tabla_generacion(g, top=6)
                ops = s.ga_obj.operaciones_generacion(min(g, len(s.ga_obj.historial) - 1))[:4]
                return {"g": g, "max": len(s.ga_obj.historial) - 1, "tabla": tabla.to_dict(orient="records"),
                        "ops": [{k: (str(v) if k not in ("gen", "op") else v) for k, v in o.items()} for o in ops]}
            if cmd == "lab_difuso":
                r = self.fuzzy.inferir(args.get("distancia"), args.get("bateria"), args.get("pendiente"), args.get("riesgo"))
                return {**{k: r[k] for k in ("entradas", "membresias", "velocidad", "cautela", "reglas", "avisos")},
                        "agregada": [round(float(v), 3) for v in r["agregada"][::4]]}
            if cmd == "lab_rna":
                return self.rna.predecir(args)
            if cmd == "rna_muestra":
                # Una fila real del conjunto de PRUEBA (mapas que la red nunca vio) para comparar predicción vs. verdad
                fila = datos["test"].sample(1, random_state=int(args.get("i", 0))).iloc[0]
                pred = self.rna.predecir(fila[RASGOS_RNA].to_dict())
                return {"rasgos": {k: float(fila[k]) for k in RASGOS_RNA}, "verdad": CLASES[int(fila.clase)],
                        "mapa": int(fila.mapa), "planeta": fila.planeta, "prediccion": pred}
            if cmd == "replanificar":
                s._pedir_replan("SOLICITUD_DEL_OPERADOR"); s._replanificar()
                return self.estado()
            if cmd == "comparar_uno":
                if s.world.modo == "sonda":
                    return {"error": "La comparación de políticas está disponible en modo rover."}
                return correr_mision_metricas(args.get("politica", "hibrido"), int(args.get("semilla", 0)),
                                              args.get("escenario", "A"), planeta=s.world.planeta.nombre,
                                              bateria=float(args.get("bateria", 80)), ga_pob=30, ga_gen=30,
                                              max_pasos=int(args.get("max_pasos", 300)))
            if cmd == "exportar":
                d = s.log.df()
                return {"csv": d.to_csv(index=False),
                        "json": json.dumps(_limpiar_json({"version": VERSION, "config": s.config, "eventos": s.log.eventos,
                                                          "mensajes": s.mensajes, "metricas": s.metricas(), "metricas_rna": METRICAS_RNA}),
                                           ensure_ascii=False, indent=2)}
            return {"error": f"Comando desconocido: {cmd}"}


# ---------------- puente con el panel web ----------------
CENTRO = None

def iniciar_web():
    global CENTRO
    CENTRO = CentroControl(rna, fuzzy, ga_pob=30, ga_gen=30)
    print("Centro de Control listo.")

def api_json(cmd, args_json):
    try:
        res = CENTRO.ejecutar(cmd, json.loads(args_json or "{}"))
    except Exception as e:
        import traceback
        res = {"error": f"{type(e).__name__}: {e}", "traza": traceback.format_exc()[-800:]}
    return json.dumps(_limpiar_json(res), ensure_ascii=False)
