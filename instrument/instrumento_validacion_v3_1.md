# Instrumento de validación de contenido — versión 3.1 (Delphi confirmatorio)

Instrumento de validación de la taxonomía difusa-OWA de perfiles de riesgo del
inversionista (Objetivo Específico 2, Capítulo 3 y Anexo A de la tesis doctoral
*Motor de recomendación adaptativo para la toma de decisiones financieras bajo
incertidumbre*, Universidad Nacional de Colombia, Sede Manizales). Este archivo
es la versión principal del instrumento y coincide con el Anexo A de la
monografía; la numeración de dimensiones y perfiles es la de las Tablas 3.1 y 3.3.

**Estado.** El instrumento se sometió a una pre-validación computacional con un
panel sintético de doce agentes (simulación de control del protocolo; ver
`A-Fuzzy-OWA-Taxonomy-of-Investor-Risk-Profiles/code/panel_sintetico.py`). La
ronda con expertos humanos está pendiente del aval del comité de ética
institucional; sus resultados no forman parte de la monografía.

## 1. Alcance del juicio experto

El juicio recae sobre la claridad, la relevancia y la diferenciación de las
definiciones, los ítems y las descripciones de perfil (sección 3.6). Los
valores de las anclas de orness no se someten a votación: los fija la
derivación por octiles. Cada perfil se presenta con su ancla canónica (octil)
y, entre paréntesis, con el orness de la parametrización histórica v1 con la que
se redactaron las descripciones arquetípicas (Tabla 3.6).

## 2. Escala de respuesta

Likert de cinco puntos: 1 = totalmente en desacuerdo; 2 = en desacuerdo;
3 = neutral; 4 = de acuerdo; 5 = totalmente de acuerdo. Cada sección admite
comentarios abiertos.

## 3. Estructura (65 ítems)

| Sección | Objeto evaluado | Estructura | Ítems |
|---|---|---|---|
| A — Dimensiones | Las 7 dimensiones conductuales | 7 dimensiones × 4 criterios | 28 |
| B — Perfiles | Los 8 perfiles canónicos | 8 perfiles × 4 criterios | 32 |
| C — Sistema global | Coherencia y aporte del sistema completo | 5 ítems globales | 5 |
| | | Total | 65 |

### Sección A — Dimensiones (Tabla 3.1)

| ID | Dimensión | Dominio | Definición operativa |
|---|---|---|---|
| D1 | Tolerancia al riesgo | Cognitivo | Grado de aceptación de pérdidas financieras potenciales en busca de mayores retornos esperados |
| D10 | Tolerancia a la ambigüedad | Cognitivo | Disposición a actuar en situaciones donde las probabilidades son desconocidas o incognoscibles |
| D5 | Aversión a la pérdida (inversa) | Emocional-afectivo | Ponderación asimétrica de pérdidas frente a ganancias equivalentes (teoría prospectiva) |
| D7 | Regulación emocional | Emocional-afectivo | Capacidad de modular las respuestas emocionales durante la volatilidad del mercado |
| D4 | Autoeficacia financiera | Contextual-situacional | Confianza específica de dominio en la propia capacidad de gestionar eficazmente las decisiones financieras |
| D8 | Horizonte de inversión | Contextual-situacional | Marco temporal para evaluar los resultados de la inversión (de menos de 1 a más de 10 años) |
| D12 | Influencia social percibida (inversa) | Sociocultural-identitario | Susceptibilidad a la manada, las redes sociales y los efectos de pares |

Criterios por dimensión (cuatro ítems por dimensión):

- A(i) Claridad: la definición está formulada de manera clara y sin ambigüedad.
- A(ii) Relevancia: la dimensión es relevante para el perfilamiento de riesgo.
- A(iii) Diferenciación (independencia conceptual): la dimensión es distinguible de las demás dimensiones.
- A(iv) Operacionalizabilidad: la dimensión puede medirse con indicadores observables.

Ejemplos de ítems (Anexo A.4):

- A-D1 (Tolerancia al riesgo · claridad). «La definición de tolerancia al riesgo como grado de aceptación de pérdidas potenciales a cambio de mayores retornos esperados está formulada de manera clara y sin ambigüedad.»
- A-D5 (Aversión a la pérdida · diferenciación). «La aversión a la pérdida (ponderación asimétrica de pérdidas frente a ganancias equivalentes) constituye una dimensión empíricamente distinguible de la tolerancia al riesgo.»
- A-D4 (Autoeficacia financiera · operacionalizabilidad). «La autoeficacia financiera puede ser medida de forma fiable mediante indicadores observables de confianza específica en la gestión de decisiones financieras.»
- A-D10 (Tolerancia a la ambigüedad · diferenciación). «La tolerancia a la ambigüedad (situaciones con probabilidades desconocidas) es conceptualmente separable de la tolerancia al riesgo (probabilidades conocidas).»
- A-D8 (Horizonte de inversión · relevancia). «El horizonte de inversión es una dimensión relevante para diferenciar el comportamiento de riesgo del inversionista.»
- A-D7 (Regulación emocional · claridad). «La definición de regulación emocional como capacidad de modular respuestas emocionales durante la volatilidad del mercado está formulada con claridad.»
- A-D12 (Influencia social percibida · relevancia). «La influencia social percibida (susceptibilidad a efectos de manada y de pares), codificada de forma inversa, aporta poder discriminante a la caracterización del perfil.»

### Sección B — Perfiles (Tabla 3.3)

Vector difuso en el orden (d1, d5, d4, d10, d8, d7, d12); ancla canónica por
octiles y, entre paréntesis, orness v1.

| # | Perfil | Vector difuso | Orness: octil (v1) | Caracterización arquetípica |
|---|---|---|---|---|
| P1 | Guardián | (VL, VH, VL, VL, VL, VL, VH) | 0,0625 (0,158) | Extremadamente averso al riesgo; muy alta aversión a la pérdida; baja autoeficacia; horizonte corto; alta influencia social |
| P2 | Centinela | (L, H, M, L, M, L, H) | 0,1875 (0,257) | Conservador, con autoeficacia y horizonte moderados |
| P3 | Pragmático | (M, M, M, M, M, M, M) | 0,3125 (0,503) | Todas las dimensiones moderadas; perfil de referencia |
| P6 | Analista | (M, M, VH, H, H, H, VL) | 0,4375 (0,600) | Riesgo moderado; autoeficacia muy alta; influencia social muy baja |
| P4 | Estratega | (H, M, H, M, H, H, L) | 0,5625 (0,647) | Alta tolerancia al riesgo, autoeficacia y regulación emocional |
| P5 | Aventurero | (H, L, H, M, H, M, H) | 0,6875 (0,693) | Alta tolerancia al riesgo; baja aversión a la pérdida; alta influencia social |
| P7 | Innovador | (VH, L, VH, H, VH, H, L) | 0,8125 (0,738) | Tolerancia al riesgo muy alta; horizonte largo |
| P8 | Visionario | (VH, VL, VH, VH, VH, VH, VL) | 0,9375 (0,865) | Buscador de riesgo extremo; autoeficacia y tolerancia a la ambigüedad muy altas |

Criterios por perfil (cuatro ítems por perfil):

- B(i) Coherencia interna del vector lingüístico difuso.
- B(ii) Diferenciación respecto de los perfiles vecinos.
- B(iii) Representatividad de un arquetipo real de inversionista.
- B(iv) Utilidad para la recomendación adaptativa.

Ejemplos de ítems (Anexo A.4; el orness citado es el v1 de la descripción arquetípica y en la aplicación se acompaña del ancla canónica):

- B-P1 (Guardián · representatividad). «El perfil Guardián —fuertemente averso al riesgo, con alta aversión a la pérdida y horizonte corto (orness = 0,158)— representa un arquetipo reconocible de inversionista conservador extremo.»
- B-P2 (Centinela · coherencia interna). «El vector lingüístico difuso del Centinela (conservador con autoeficacia y horizonte moderados, orness = 0,257) es internamente coherente.»
- B-P3 (Pragmático · diferenciación). «El perfil Pragmático (todas las dimensiones moderadas, orness = 0,503) se distingue con claridad de los perfiles adyacentes Centinela y Analista.»
- B-P6 (Analista · utilidad). «El perfil Analista (autoeficacia muy alta e influencia social muy baja, orness = 0,600) es útil para diferenciar la recomendación de portafolio.»
- B-P4 (Estratega · coherencia interna). «El vector del Estratega (alta tolerancia al riesgo, autoeficacia y regulación emocional, orness = 0,647) es internamente coherente.»
- B-P5 (Aventurero · diferenciación). «El perfil Aventurero (alta tolerancia al riesgo con alta influencia social, orness = 0,693) se distingue suficientemente del Estratega y del Innovador.»
- B-P7 (Innovador · representatividad). «El perfil Innovador (tolerancia al riesgo muy alta y horizonte largo, orness = 0,738) representa un arquetipo real de inversionista.»
- B-P8 (Visionario · utilidad). «El perfil Visionario (buscador de riesgo extremo, orness = 0,865) aporta valor diferencial a la recomendación frente a la categoría agresiva tradicional.»

### Sección C — Sistema global (5 ítems)

- C1. «El conjunto de siete dimensiones cubre adecuadamente las facetas conductuales relevantes del riesgo del inversionista.»
- C2. «Los ocho perfiles, en su conjunto, abarcan el espectro conductual sin solapes injustificados ni vacíos relevantes.»
- C3. «La taxonomía de ocho perfiles es superior a la tríada clásica (conservador / moderado / agresivo) para fines de recomendación personalizada.»
- C4. «La formalización mediante lógica difusa (vectores lingüísticos trapezoidales) es adecuada para representar la heterogeneidad del inversionista.»
- C5. «La calibración del operador OWA a través del cuantificador RIM (anclaje perfil → orness) es coherente y está justificada.»

## 4. Criterios de análisis y reglas de decisión

Razón de validez de contenido (Lawshe, 1975): CVR = (n_e − N/2)/(N/2), con N
el número de expertos y n_e el número que califica el ítem con 4 o 5. Se adopta
el umbral de 0,78 (Zamanzadeh et al., 2015), más estricto que el valor crítico
de Lawshe para N = 12 (0,56).

| Condición | Decisión |
|---|---|
| Mediana ≥ 4 y CVR ≥ 0,78 | Validado (con ajustes menores) |
| Mediana ≥ 4 y CVR < 0,78 | Segunda ronda con retroalimentación dirigida |
| Mediana < 4 | Revisión mayor del ítem |

Concordancia: W de Kendall corregida por empates (Schmidt, 1997) y razón
W_obs/W_máx (Meijering et al., 2013) como indicador complementario; los
criterios primarios de aceptación son la mediana y el CVR.

Ítems de atención prioritaria según la pre-validación con panel sintético
(consenso en menos de la mitad de las réplicas): diferenciación de D10 y de
D12, relevancia de D12, diferenciación del Pragmático y del Aventurero y
utilidad del Aventurero.

## 5. Composición del panel

Entre diez y doce expertos en dos subpaneles: Subpanel A — Contenido (n ≈ 6):
finanzas conductuales, psicología financiera y perfilamiento del inversionista.
Subpanel B — Método (n ≈ 6): operadores OWA, lógica difusa y sistemas de
recomendación.

## 6. Ética y protección de datos

La aplicación requiere el aval del comité de ética institucional y el
consentimiento informado de cada experto conforme a la Ley 1581 de 2012 de
protección de datos personales (Colombia). El consentimiento informa el
propósito, la participación voluntaria, la confidencialidad de las respuestas
individuales y el reporte exclusivamente agregado. La lista nominal de expertos
y el registro de consentimientos se custodian conforme al protocolo aprobado por
el comité y no se publican en este repositorio.
