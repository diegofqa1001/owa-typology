# Ronda de validación por juicio experto (Objetivo Específico 2)

Esta carpeta contiene el guion de análisis y los resultados agregados de la ronda
Delphi con la que un panel de expertos validó el contenido de la taxonomía
difusa-OWA de ocho perfiles conductuales de riesgo (Capítulo 3, sección 3.5.3 y
Anexo A.6 de la tesis).

## Ronda

- Instrumento: 65 ítems Likert de cinco puntos (`../instrument/instrumento_validacion_v3_1.md`).
- Aplicación: formulario en línea, del 19 al 26 de octubre de 2026, con el aval del
  comité de ética de la sede Manizales de la Universidad Nacional de Colombia
  (Acta NP 07-2026) y consentimiento informado conforme a la Ley 1581 de 2012.
- Panel analizado: 14 expertos (6 en el subpanel de contenido y 8 en el de método).
  Se excluyeron dos respuestas cuyos autores declararon vínculo con el investigador.
- Regla de decisión fijada de antemano: mediana ≥ 4 y CVR ≥ 0,78 → validado;
  mediana ≥ 4 y CVR < 0,78 → segunda ronda dirigida; mediana < 4 → revisión mayor.

## Resultados principales

| Indicador | Valor |
|---|---|
| Ítems validados | 57 de 65 (87,7 %) |
| Ítems remitidos a segunda ronda | 8 (A-D10-IND, A-D7-OPE, A-D12-IND, A-D12-OPE, B-P3-DIF, B-P7-DIF, B-P8-DIF, C-C5) |
| Ítems en revisión mayor | 0 |
| Calificaciones 4 o 5 | 94,6 % de 910 |
| W de Kendall corregida por empates (65 ítems) | 0,066 (p por permutación = 0,67) |

## Archivos

| Archivo | Contenido |
|---|---|
| `analisis_ronda_expertos.py` | Guion de análisis: CVR, mediana, IQR, decisión por ítem y W de Kendall con prueba de permutación |
| `resultados/items_agregado.csv` | Frecuencias de calificación 1–5, mediana, IQR, media, % 4–5, CVR y decisión por ítem |
| `resultados/concordancia.csv` | W de Kendall (con y sin corrección por empates), χ², p asintótico y p por permutación por sección y subpanel |
| `resultados/resumen.json` | Cifras de síntesis |

## Reproducción

```bash
python analisis_ronda_expertos.py matriz_codificada.csv resultados
```

El guion recibe la matriz codificada de calificaciones (expertos E01–E14 × 65 ítems).
Conforme al consentimiento aprobado por el comité de ética, esa matriz se custodia
en acceso restringido y los resultados se publican de forma agregada.
