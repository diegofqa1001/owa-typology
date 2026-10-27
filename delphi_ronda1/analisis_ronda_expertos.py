"""Análisis de la ronda de validación por juicio experto (Objetivo Específico 2).

Lee la matriz codificada de calificaciones (expertos E01..E14 x 65 ítems, escala
Likert 1-5) y produce únicamente resultados agregados por ítem y por bloque.
Conforme al consentimiento informado aprobado por el comité de ética (Acta
NP 07-2026), la matriz codificada se custodia en acceso restringido y no forma
parte de este repositorio; aquí se publican el guion y sus salidas agregadas.

Uso:
    python analisis_ronda_expertos.py matriz_codificada.csv [carpeta_salida]

Salidas:
    items_agregado.csv     frecuencias 1-5, mediana, IQR, media, % >= 4, CVR,
                           rwg y decisión por ítem
    concordancia.csv       W de Kendall (con y sin corrección por empates),
                           chi-cuadrado, p asintótico y p por permutación,
                           por sección y subpanel
    resumen.json           cifras de síntesis

Reglas de decisión (fijadas de antemano en el instrumento, Anexo A.5.2):
    mediana >= 4 y CVR >= 0,78  -> Validado
    mediana >= 4 y CVR <  0,78  -> Segunda ronda con retroalimentación dirigida
    mediana <  4                -> Revisión mayor
Como lectura estadística complementaria se informa si el CVR supera el valor
crítico de Wilson, Pan y Schumsky (2012) para el N observado.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, rankdata

UMBRAL_CVR = 0.78
CVR_CRIT_WILSON = {5: .736, 6: .672, 7: .622, 8: .582, 9: .548, 10: .520, 11: .496,
                   12: .475, 13: .456, 14: .440, 15: .425, 16: .411, 17: .399,
                   18: .388, 19: .377, 20: .368}
N_PERM = 5000
SEMILLA = 2026


def kendall_w(M, empates=True):
    """W de Kendall para una matriz expertos x ítems (corrección de empates de Schmidt, 1997)."""
    M = np.asarray(M, float)
    m, n = M.shape
    R = np.zeros(n)
    T = 0.0
    for fila in M:
        R += rankdata(fila)
        _, c = np.unique(fila, return_counts=True)
        T += float(np.sum(c ** 3 - c))
    S = float(np.sum((R - R.mean()) ** 2))
    den = m * m * (n ** 3 - n) - (m * T if empates else 0.0)
    return 12 * S / den if den > 0 else float("nan")


def p_permutacion(M, w_obs, rng, n_perm=N_PERM):
    """p por permutación de los ítems dentro de cada experto (conserva sus empates)."""
    M = np.asarray(M, float)
    aciertos = sum(kendall_w(np.array([rng.permutation(f) for f in M])) >= w_obs - 1e-12
                   for _ in range(n_perm))
    return (aciertos + 1) / (n_perm + 1)


def main():
    entrada = Path(sys.argv[1])
    salida = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("resultados")
    salida.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(entrada)
    items = [c for c in df.columns if c[:2] in ("A-", "B-", "C-")]
    X = df[items].astype(float)
    N = len(df)
    crit = CVR_CRIT_WILSON[N]

    filas = []
    for c in items:
        x = X[c].dropna()
        n = len(x)
        ne = int((x >= 4).sum())
        cvr = (ne - n / 2) / (n / 2)
        md = float(x.median())
        iqr = float(x.quantile(.75) - x.quantile(.25))
        if md >= 4 and cvr >= UMBRAL_CVR:
            dec = "Validado"
        elif md >= 4:
            dec = "Segunda ronda con retroalimentación dirigida"
        else:
            dec = "Revisión mayor"
        filas.append({"item": c, "seccion": c[0], "objeto": c.split("-")[1],
                      "criterio": c.split("-")[2] if c[0] != "C" else "SIS",
                      "N": n, **{f"f{k}": int((x == k).sum()) for k in range(1, 6)},
                      "mediana": md, "IQR": iqr, "media": round(float(x.mean()), 3),
                      "pct_4o5": round(100 * ne / n, 1), "CVR": round(cvr, 3),
                      "rwg": round(1 - float(x.var(ddof=1)) / 2, 3),
                      "supera_CVR_critico_Wilson": cvr >= crit, "decision": dec})
    T = pd.DataFrame(filas)
    T.to_csv(salida / "items_agregado.csv", index=False)

    rng = np.random.default_rng(SEMILLA)
    conc = []
    grupos = [("Panel completo", df.index)]
    for g in ("A", "B"):
        grupos.append((f"Subpanel {g}", df.index[df["subpanel"] == g]))
    for nombre, idx in grupos:
        for sec in ("A", "B", "C", "Global"):
            cols = [c for c in items if sec == "Global" or c.startswith(sec)]
            M = X.loc[idx, cols].dropna().values
            m, n = M.shape
            w = kendall_w(M)
            x2 = m * (n - 1) * w
            conc.append({"grupo": nombre, "seccion": sec, "n_items": n, "m_expertos": m,
                         "W_corregido": round(w, 3), "W_sin_correccion": round(kendall_w(M, False), 3),
                         "chi2": round(x2, 2), "gl": n - 1, "p_asintotico": round(float(chi2.sf(x2, n - 1)), 4),
                         "p_permutacion": round(p_permutacion(M, w, rng), 4)})
    C = pd.DataFrame(conc)
    C.to_csv(salida / "concordancia.csv", index=False)

    v = X.values.ravel()
    v = v[~np.isnan(v)]
    resumen = {
        "N_expertos": N, "subpaneles": df["subpanel"].value_counts().to_dict(),
        "n_items": len(items), "n_calificaciones": int(v.size),
        "distribucion": {str(k): int((v == k).sum()) for k in range(1, 6)},
        "pct_calificaciones_4o5": round(100 * float((v >= 4).mean()), 1),
        "validados": int((T.decision == "Validado").sum()),
        "segunda_ronda": T.loc[T.decision.str.startswith("Segunda"), "item"].tolist(),
        "revision_mayor": int((T.decision == "Revisión mayor").sum()),
        "CVR_critico_Wilson": crit,
        "rwg_medio": round(float(T.rwg.mean()), 3),
        "W_global": C.query("grupo == 'Panel completo' and seccion == 'Global'").iloc[0].to_dict(),
    }
    (salida / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    print(C.to_string(index=False))


if __name__ == "__main__":
    main()
