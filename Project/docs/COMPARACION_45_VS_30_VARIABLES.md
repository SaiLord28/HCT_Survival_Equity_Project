# Comparacion 60 vs 45 vs 30 y hallazgos de feature engineering

Fecha de ejecucion: 2026-04-21

## Resumen ejecutivo

- Se ejecutaron pruebas comparativas del pipeline con `n_features` en 60, 45 y 30.
- La reduccion 60 -> 45 mantiene rendimiento muy similar.
- La reduccion 45 -> 30 muestra una caida pequena en AUC y en metrica de equidad, pero aun pasa fairness.
- Hallazgo critico: el escenario de 30 no selecciona 30 reales en la implementacion actual.

## Metodologia

- Dataset: `Project/ai_service/data/raw/train.csv`.
- Pipeline evaluado: `Project/ai_service/src/pipeline.py` (entrenamiento completo M1-M7).
- Configuracion: `model_type='gbm'` y `n_features` en 60, 45, 30.
- Metricas analizadas: CV AUC, CV Accuracy, AUC de entrenamiento, Stratified C-index, C-index disparity, fairness_passed y estabilidad.

## Resultados de comparacion

| n_features solicitado | n_features seleccionado | CV AUC | CV Accuracy | Train AUC | Stratified C-index | C-index disparity | Fairness | Estabilidad |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 60 | 60 | 0.74196 | 0.67965 | 0.81015 | 0.81004 | 0.01598 | True | High |
| 45 | 45 | 0.74148 | 0.67931 | 0.80905 | 0.80880 | 0.01598 | True | Medium |
| 30 | 44 | 0.74004 | 0.67931 | 0.80896 | 0.80867 | 0.01965 | True | Medium |

## Interpretacion

1. 60 vs 45
- Degradacion minima en AUC y accuracy.
- Equidad practicamente igual en disparidad.

2. 45 vs 30
- Caida moderada de CV AUC.
- Ligero aumento en disparidad (peor que 45), aunque sigue en estado `fairness_passed=True`.

3. Efecto practico
- La estrategia actual justifica el recorte a 45 con bajo costo.
- Para 30, antes de concluir, hay que corregir la logica del selector porque hoy no aplica un tope real de 30.

## Hallazgo clave: el modo 30 no esta seleccionando 30

En `Project/ai_service/src/m3_features.py`, la funcion `select_features` inicia con un bloque de variables forzadas:

- Comorbilidades forzadas
- `comorbidity_score`
- Variables clinicas adicionales
- Variables clinicas prioritarias

Con el dataset actual, ese bloque forzado suma 43 variables. Por eso, al solicitar 30, el selector termina con 44 (el piso forzado ya supera el objetivo).

Implicacion:
- El parametro `n_features` no funciona como limite estricto cuando `forced_features > n_features`.

## Feature engineering ya implementado

En `Project/ai_service/src/m1_preprocessing.py` se crean estas features:

1. `age_donor_diff`
- Diferencia entre edad del paciente y del donante.

2. `high_risk_comorbidity`
- Indicador de alta comorbilidad severa.

3. `hla_match_quality`
- Puntaje compuesto de matching HLA.

Adicionalmente, se aplica:
- Imputacion equity-aware por `race_group`.
- Escalado de variables numericas.
- Codificacion de categoricas con manejo de categorias no vistas.

## Hallazgo tecnico en feature engineering

`high_risk_comorbidity` esta quedando constante (sin varianza) en estas pruebas.

Posible causa en la implementacion actual:
- La logica revisa condiciones numericas en columnas que originalmente pueden venir como categoricas tipo `Y/N`.
- Resultado: la bandera no se activa como se espera y pierde poder predictivo.

## Pruebas automatizadas del proyecto

Se intento ejecutar `Project/ai_service/test_pipeline.py` y fallo por:

- `IndentationError: unexpected indent` en la linea 1.

Implicacion:
- El script de pruebas integradas necesita correccion para ser util como verificacion automatizada continua.

## Recomendaciones priorizadas

1. Corregir `high_risk_comorbidity`
- Detectar explicitamente valores afirmativos (`Y`, `Yes`, `1`, `true`) antes o despues del encoding de forma consistente.

2. Hacer `n_features` un tope real
- Separar `must_have` (minimo clinico obligatorio) de `should_have` (ranking).
- Si se pide 30, seleccionar exactamente 30 o reportar claramente que no es posible por restricciones clinicas.

3. Agregar indicadores de missingness
- Especialmente para variables con alta ausencia (por ejemplo `tce_match`, `mrd_hct`, `tce_imm_match`, `cyto_score_detail`).

4. Crear interacciones clinicas con respaldo de dominio
- Ejemplos: `age_at_hct x comorbidity_score`, `dri_score x hla_match_quality`, `karnofsky_score x conditioning_intensity`.

## Decisiones sugeridas

1. Mantener 45 como baseline operativo inmediato.
2. Corregir selector y feature engineering.
3. Repetir benchmark con 30 real y comparar de nuevo contra 45 y 60.
4. Adoptar 30 solo si la perdida de rendimiento y equidad se mantiene dentro del umbral clinico acordado.
