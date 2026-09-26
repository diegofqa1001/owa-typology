#!/usr/bin/env python3
"""
cadena_cap3.py — Reconstrucción versionada de la cadena del Capítulo 3 de la tesis
(taxonomía difusa-OWA de ocho perfiles, OE2):

    vector lingüístico (Tabla 3.3) -> defuzzificación COG (Tabla 3.2)
    -> orientación direccional (D5 y D12 invertidas) -> regla de orden
    -> rango k -> centroide de diseño c_k (v1) -> exponente RIM exacto
    -> contraste con los valores congelados v1 (Tabla 3.4)
    -> anclas canónicas por octiles (2k-1)/16 (§3.6, Tabla 3.6)

Salida: results/cadena_cap3.json (y resumen por consola).

Qué documenta:
  1. Valores COG en forma cerrada de las cinco etiquetas de la Tabla 3.2.
  2. Dos reglas de orden sobre el mismo vector direccional:
       (a) media uniforme de las siete coordenadas (puntaje actitudinal de
           owa_typology.attitude_score, línea base del artículo);
       (b) regla de la tesis: orden lexicográfico por el núcleo de actitud
           ante el riesgo (media de D1 y de D5 invertida) y, a igualdad,
           por la media de las cinco dimensiones moduladoras
           (D4, D10, D8, D7 y D12 invertida).
     Se reporta además el umbral de peso del núcleo omega* a partir del cual
     cualquier ponderación convexa núcleo/moduladoras reproduce el orden (b).
  3. Exponentes RIM exactos (Brent, xtol 1e-12) para los centroides de diseño,
     contraste con los alfa congelados de A-Fuzzy-OWA-Taxonomy (Tabla 3.4) y
     verificación de la tolerancia |orness - c_k| < 0,01.
  4. Búsqueda (negativa) de una configuración de optimizador acotado que
     reproduzca los alfa congelados.
  5. Anclas canónicas por octiles y sus exponentes RIM para n = 7.
  6. Ejemplo numérico de §3.4 con los pesos congelados (brecha 42,4 pp).

Uso:  OMP_NUM_THREADS=1 python tesis_cap3/cadena_cap3.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
from scipy.optimize import brentq, minimize_scalar

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
from owa_typology import LABELS, rim_weights, orness  # noqa: E402

# ---------------------------------------------------------------- Tabla 3.3
# Orden canónico de las coordenadas: (d1, d5, d4, d10, d8, d7, d12)
DIM_TESIS = ["D1", "D5", "D4", "D10", "D8", "D7", "D12"]
INVERTIDAS = (1, 6)            # D5 aversión a la pérdida, D12 influencia social
NUCLEO = (0, 1)                # D1 tolerancia al riesgo, D5 aversión a la pérdida
MODULADORAS = (2, 3, 4, 5, 6)  # D4, D10, D8, D7, D12

PERFILES = {  # P#: (nombre, vector lingüístico, centroide de diseño v1)
    "P1": ("Guardián",   ["VL", "VH", "VL", "VL", "VL", "VL", "VH"], 0.15),
    "P2": ("Centinela",  ["L",  "H",  "M",  "L",  "M",  "L",  "H"],  0.25),
    "P3": ("Pragmático", ["M",  "M",  "M",  "M",  "M",  "M",  "M"],  0.50),
    "P6": ("Analista",   ["M",  "M",  "VH", "H",  "H",  "H",  "VL"], 0.60),
    "P4": ("Estratega",  ["H",  "M",  "H",  "M",  "H",  "H",  "L"],  0.65),
    "P5": ("Aventurero", ["H",  "L",  "H",  "M",  "H",  "M",  "H"],  0.70),
    "P7": ("Innovador",  ["VH", "L",  "VH", "H",  "VH", "H",  "L"],  0.75),
    "P8": ("Visionario", ["VH", "VL", "VH", "VH", "VH", "VH", "VL"], 0.90),
}
ORDEN_TESIS = ["P1", "P2", "P3", "P6", "P4", "P5", "P7", "P8"]

# Valores congelados v1 (A-Fuzzy-OWA-Taxonomy-of-Investor-Risk-Profiles,
# data/owa/owa_profiles.json), citados en la Tabla 3.4.
ALFA_CONGELADO = {"P1": 4.0, "P2": 2.4843, "P3": 0.9881, "P6": 0.6949,
                  "P4": 0.5798, "P5": 0.4771, "P7": 0.3889, "P8": 0.1765}
ORNESS_CONGELADO = {"P1": 0.158, "P2": 0.257, "P3": 0.503, "P6": 0.600,
                    "P4": 0.647, "P5": 0.693, "P7": 0.738, "P8": 0.865}
EPS = 0.01
N = 7
S_EJEMPLO = np.array([0.8, 0.6, 0.7, 0.5, 0.4, 0.3, 0.9])


def cog_trapecio(a: float, b: float, c: float, d: float) -> float:
    """Centro de gravedad en forma cerrada de un trapecio (a, b, c, d)."""
    return ((d * d + c * c + c * d) - (a * a + b * b + a * b)) / (3.0 * (d + c - a - b))


def alfa_exacto(objetivo: float, n: int = N) -> float:
    if abs(objetivo - 0.5) < 1e-15:
        return 1.0
    return float(brentq(lambda a: orness(rim_weights(a, n)) - objetivo, 1e-9, 500.0, xtol=1e-12))


def main() -> int:
    out: dict = {}
    cog = {k: cog_trapecio(*v) for k, v in LABELS.items()}
    out["cog_etiquetas"] = {k: round(v, 6) for k, v in cog.items()}

    # ---------------------------------------------- vector direccional
    direc = {}
    for p, (_, vec, _) in PERFILES.items():
        s = np.array([cog[l] for l in vec])
        t = s.copy()
        for i in INVERTIDAS:
            t[i] = 1.0 - s[i]
        direc[p] = t
    tabla = {}
    for p, (nombre, vec, c) in PERFILES.items():
        t = direc[p]
        tabla[p] = {"nombre": nombre, "vector": vec, "centroide_diseno_v1": c,
                    "direccional": [round(x, 4) for x in t],
                    "media_uniforme_7": round(float(t.mean()), 4),
                    "nucleo_D1_D5inv": round(float(t[list(NUCLEO)].mean()), 4),
                    "moduladoras_5": round(float(t[list(MODULADORAS)].mean()), 4)}

    orden_uniforme = sorted(PERFILES, key=lambda p: direc[p].mean())
    orden_lex = sorted(PERFILES, key=lambda p: (round(direc[p][list(NUCLEO)].mean(), 12),
                                                round(direc[p][list(MODULADORAS)].mean(), 12)))
    orden_centroide = sorted(PERFILES, key=lambda p: PERFILES[p][2])

    # umbral omega*: score = w*nucleo + (1-w)*moduladoras reproduce ORDEN_TESIS
    def orden_w(w):
        sc = {p: w * direc[p][list(NUCLEO)].mean() + (1 - w) * direc[p][list(MODULADORAS)].mean()
              for p in PERFILES}
        return sorted(PERFILES, key=sc.get)
    malla = np.linspace(0, 1, 100001)
    validos = [w for w in malla if orden_w(w) == ORDEN_TESIS]
    # umbral analítico de los pares críticos (P6<P4 y P4<P5)
    def umbral(p, q):
        dn = direc[q][list(NUCLEO)].mean() - direc[p][list(NUCLEO)].mean()
        dm = direc[p][list(MODULADORAS)].mean() - direc[q][list(MODULADORAS)].mean()
        return dm / (dn + dm) if dm > 0 else 0.0
    out["reglas_de_orden"] = {
        "orden_tesis_tabla_3_3": ORDEN_TESIS,
        "orden_por_centroide_diseno": orden_centroide,
        "a_media_uniforme_7_dimensiones": orden_uniforme,
        "a_coincide_con_tesis": orden_uniforme == ORDEN_TESIS,
        "b_lexicografica_nucleo_luego_moduladoras": orden_lex,
        "b_coincide_con_tesis": orden_lex == ORDEN_TESIS,
        "peso_nucleo_uniforme": round(2 / 7, 4),
        "umbral_P6_P4": round(umbral("P6", "P4"), 4),
        "umbral_P4_P5": round(umbral("P4", "P5"), 4),
        "intervalo_peso_nucleo_que_reproduce_tesis": [round(min(validos), 4), round(max(validos), 4)]
        if validos else None,
        "nota": "Con peso del núcleo w en (omega*, 1) la ponderación convexa reproduce el orden de la "
                "tesis; en w = 1 Pragmático y Analista empatan (0,5) y el desempate lexicográfico "
                "por moduladoras reproduce el mismo orden.",
    }
    out["perfiles"] = tabla

    # ---------------------------------------------- calibración RIM
    cal = {}
    for p in ORDEN_TESIS:
        c = PERFILES[p][2]
        a_ex = alfa_exacto(c)
        a_cg = ALFA_CONGELADO[p]
        o_cg = orness(rim_weights(a_cg))
        cal[p] = {"nombre": PERFILES[p][0], "centroide": c,
                  "alfa_exacto": round(a_ex, 4),
                  "orness_con_alfa_exacto": round(orness(rim_weights(a_ex)), 10),
                  "alfa_congelado": a_cg, "orness_congelado": round(o_cg, 4),
                  "desviacion_orness": round(abs(o_cg - c), 4),
                  "cumple_eps_0_01": bool(abs(o_cg - c) < EPS),
                  "orness_uniforme_owa_typology": tabla[p]["media_uniforme_7"],
                  "alfa_uniforme_owa_typology": round(alfa_exacto(tabla[p]["media_uniforme_7"]), 4)}
    out["calibracion_rim_n7"] = cal
    out["perfiles_fuera_de_tolerancia"] = [p for p in cal if not cal[p]["cumple_eps_0_01"]]

    # ------------------------------ búsqueda de un optimizador que reproduzca alfa congelado
    warnings.filterwarnings("ignore")
    mejor = None
    for lo in (0.01, 0.05, 0.1, 0.2):
        for hi in (4.0, 5.0, 10.0, 20.0, 50.0):
            for tol in (1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1):
                for obj in ("cuadratico", "absoluto"):
                    r = {}
                    for p in ORDEN_TESIS:
                        c = PERFILES[p][2]
                        f = ((lambda a, c=c: (orness(rim_weights(a)) - c) ** 2) if obj == "cuadratico"
                             else (lambda a, c=c: abs(orness(rim_weights(a)) - c)))
                        r[p] = minimize_scalar(f, bounds=(lo, hi), method="bounded",
                                               options={"xatol": tol}).x
                    err = max(abs(r[p] - ALFA_CONGELADO[p]) for p in ORDEN_TESIS)
                    if mejor is None or err < mejor[0]:
                        mejor = (err, lo, hi, tol, obj)
    out["busqueda_optimizador_congelado"] = {
        "configuraciones_probadas": 4 * 5 * 6 * 2,
        "menor_error_max_alfa": round(float(mejor[0]), 4),
        "config": {"cota_inf": mejor[1], "cota_sup": mejor[2], "xatol": mejor[3], "objetivo": mejor[4]},
        "reproduce": bool(mejor[0] < 5e-4)}

    # ---------------------------------------------- octiles canónicos
    oct_ = {}
    for k, p in enumerate(ORDEN_TESIS, start=1):
        a = (2 * k - 1) / 16
        oct_[p] = {"nombre": PERFILES[p][0], "rango_k": k, "ancla": a,
                   "alfa_rim_n7": round(alfa_exacto(a), 4)}
    out["octiles_canonicos"] = oct_
    out["octiles_si_orden_uniforme"] = {p: {"rango_k": k, "ancla": (2 * k - 1) / 16}
                                        for k, p in enumerate(orden_uniforme, start=1)}

    # ---------------------------------------------- ejemplo numérico §3.4 (pesos congelados)
    b = np.sort(S_EJEMPLO)[::-1]
    F = {p: float(rim_weights(ALFA_CONGELADO[p]) @ b) for p in ORDEN_TESIS}
    out["ejemplo_3_4"] = {"S": S_EJEMPLO.tolist(), "F": {p: round(v, 4) for p, v in F.items()},
                          "brecha": round(max(F.values()) - min(F.values()), 4),
                          "monotono": all(F[ORDEN_TESIS[i]] < F[ORDEN_TESIS[i + 1]] for i in range(7))}

    os.makedirs(os.path.join(RAIZ, "results"), exist_ok=True)
    ruta = os.path.join(RAIZ, "results", "cadena_cap3.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------- resumen
    print("COG:", out["cog_etiquetas"])
    print(f"{'P':3s} {'perfil':11s} {'unif7':>7s} {'núcleo':>7s} {'modul':>7s} {'c_k':>5s}")
    for p in ORDEN_TESIS:
        t = tabla[p]
        print(f"{p:3s} {t['nombre']:11s} {t['media_uniforme_7']:7.4f} {t['nucleo_D1_D5inv']:7.4f} "
              f"{t['moduladoras_5']:7.4f} {t['centroide_diseno_v1']:5.2f}")
    print("orden uniforme  :", orden_uniforme)
    print("orden lexicogr. :", orden_lex, "== tesis:", orden_lex == ORDEN_TESIS)
    print("intervalo peso núcleo que reproduce la tesis:",
          out["reglas_de_orden"]["intervalo_peso_nucleo_que_reproduce_tesis"])
    for p, v in cal.items():
        print(f"{p} {v['nombre']:11s} c={v['centroide']:.2f} alfa_ex={v['alfa_exacto']:.4f} "
              f"alfa_cong={v['alfa_congelado']:.4f} orness_cong={v['orness_congelado']:.4f} "
              f"|dev|={v['desviacion_orness']:.4f} ok={v['cumple_eps_0_01']}")
    print("búsqueda optimizador:", out["busqueda_optimizador_congelado"])
    print("octiles:", {p: (v['ancla'], v['alfa_rim_n7']) for p, v in oct_.items()})
    print("ejemplo:", out["ejemplo_3_4"])
    print("->", ruta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
