# Analisis Lasso para HCT Survival - Variables Mas Importantes

Fecha de ejecucion: 2026-04-21
Script: `analysis/lasso_analysis.py`
Resultados: `analysis/lasso_results.json`

---

## 1. Resumen del Dataset

Los archivos en `Project/ai_service/data/raw/` contienen:

| Archivo | Descripcion | Tamano |
|---|---|---|
| train.csv | Dataset de entrenamiento | 28,800 registros x 60 columnas |
| test.csv | Dataset de prueba (sin target) | 3 registros x 58 columnas |
| data_dictionary.csv | Diccionario de variables | 59 variables documentadas |
| sample_submission.csv | Formato de envio | Ejemplo de formato |

### Variables Target

- **efs**: Event-Free Survival - binaria (Event / Censoring)
- **efs_time**: Tiempo a evento en meses - numerica continua (media: ~29.1, mediana: ~17.8)

### Distribucion de eventos

- Event: ~53.2% de los registros
- Censoring: ~46.8% de los registros

### Tipos de variables (57 predictoras)

- Numericas: 20 variables (mayoria son variables HLA)
- Categoricas: 37 variables (comorbilidades, datos clinicos, demograficos)

---

## 2. Metodologia

Se ejecuto un **LassoCV** (Lasso con validacion cruzada de 5 folds) sobre `efs_time` como target, excluyendo `ID` y `efs` para evitar fuga de informacion.

- Rango de alphas: 100 valores en escala logaritmica [10^-4, 10^2]
- Preprocesamiento: imputacion (mediana para numericas, moda para categoricas), escalado StandardScaler, OneHotEncoder para categoricas
- Split: 80% train / 20% validacion, random_state=42

### Configuracion del modelo

| Parametro | Valor |
|---|---|
| Alpha optimo | 0.01322 |
| Features totales (post one-hot) | 183 |
| Features seleccionadas (coef != 0) | 122 (66.7%) |
| Features eliminadas (coef = 0) | 61 (33.3%) |

### Metricas de desempeno

| Metrica | Train | Validacion |
|---|---:|---:|
| RMSE | 22.647 | 22.830 |
| MAE | 17.768 | 17.879 |
| R2 | 0.1632 | 0.1637 |

**Nota sobre el R2 bajo (~0.16)**: Es esperado para datos de supervivencia con censura. Los datos censurados (~47%) distorsionan la regresion lineal. Sin embargo, el Lasso sigue siendo valido para identificar que variables son mas importantes, que es el objetivo principal de este analisis.

---

## 3. Variables Mas Importantes

### ALTA importancia (|coef| > 0.5) - 30 variables originales

| Rank | Variable | Tipo | Max Coef | Descripcion |
|---:|---|---|---:|---|
| 1 | ethnicity | Categorica | 7.538 | Etnicidad del paciente |
| 2 | prior_tumor | Categorica | 6.974 | Tumor solido previo |
| 3 | year_hct | Numerica | 6.147 | Ano del trasplante |
| 4 | prim_disease_hct | Categorica | 4.664 | Enfermedad primaria (AML, ALL, MDS, etc.) |
| 5 | cardiac | Categorica | 4.562 | Condicion cardiaca |
| 6 | diabetes | Categorica | 4.366 | Diabetes |
| 7 | conditioning_intensity | Categorica | 3.717 | Intensidad del acondicionamiento |
| 8 | dri_score | Categorica | 3.695 | Indice de riesgo de enfermedad |
| 9 | cmv_status | Categorica | 3.420 | Estatus CMV donante/receptor |
| 10 | gvhd_proph | Categorica | 3.054 | Profilaxis GVHD |
| 11 | sex_match | Categorica | 2.981 | Concordancia de sexo donante/receptor |
| 12 | cyto_score | Categorica | 2.486 | Score citogenetico |
| 13 | in_vivo_tcd | Categorica | 2.459 | Deplecion T-cell in vivo |
| 14 | tbi_status | Categorica | 2.221 | Irradiacion corporal total |
| 15 | pulm_severe | Categorica | 2.114 | Pulmonar severa |
| 16 | vent_hist | Categorica | 2.045 | Historia de ventilacion mecanica |
| 17 | tce_imm_match | Categorica | 1.975 | Match de epitope T-cell |
| 18 | race_group | Categorica | 1.620 | Grupo racial |
| 19 | prod_type | Categorica | 1.542 | Tipo de producto |
| 20 | hepatic_severe | Categorica | 1.499 | Hepatica severa |
| 21 | hla_match_b_low | Numerica | 1.489 | HLA-B baja resolucion |
| 22 | rheum_issue | Categorica | 1.453 | Problema reumatologico |
| 23 | hla_high_res_8 | Numerica | 1.431 | Match HLA 8 loci alta resolucion |
| 24 | hla_low_res_8 | Numerica | 1.377 | Match HLA 8 loci baja resolucion |
| 25 | comorbidity_score | Numerica | 1.331 | Score de comorbilidad Sorror |
| 26 | donor_related | Categorica | 1.296 | Tipo de donante (relacionado/no) |
| 27 | psych_disturb | Categorica | 1.127 | Perturbacion psiquiatrica |
| 28 | hla_nmdp_6 | Numerica | 1.092 | Match HLA NMDP 6 loci |
| 29 | obesity | Categorica | 1.067 | Obesidad |
| 30 | peptic_ulcer | Categorica | 1.055 | Ulcera peptica |

### MEDIA importancia (0.1 < |coef| <= 0.5) - 8 variables

| Rank | Variable | Tipo | Max Coef |
|---:|---|---|---:|
| 31 | tce_match | Categorica | 0.907 |
| 32 | hla_high_res_10 | Numerica | 0.886 |
| 33 | rituximab | Categorica | 0.880 |
| 34 | karnofsky_score | Numerica | 0.725 |
| 35 | mrd_hct | Categorica | 0.700 |
| 36 | hla_match_c_high | Numerica | 0.635 |
| 37 | tce_div_match | Categorica | 0.612 |
| 38 | hla_match_drb1_high | Numerica | 0.604 |

**Nota**: `karnofsky_score` aparece en posicion 34, lo cual podria parecer inesperado clinicamente. Esto se debe a que el Lasso penaliza variables correlacionadas: `comorbidity_score` (rank 25) captura parte de la misma informacion.

### BAJA importancia (|coef| > 0 pero <= 0.1) - 2 variables

| Variable | Coef |
|---|---:|
| hla_match_dqb1_high | 0.089 |
| hla_match_dqb1_low | 0.009 |

### ELIMINADAS por Lasso (|coef| = 0) - 5 variables originales

| Variable | Tipo | Razon probable |
|---|---|---|
| hla_high_res_6 | Numerica | Redundante con hla_high_res_8 y hla_high_res_10 (r > 0.97) |
| donor_age | Numerica | Capturada por age_at_hct y donor_related |
| hla_match_a_high | Numerica | Redundante con variables HLA agregadas |
| hla_low_res_10 | Numerica | Redundante con hla_low_res_8 (r = 0.96) |
| renal_issue | Categorica | Capturada por comorbidity_score |

---

## 4. Analisis de Multicolinealidad

Las variables HLA presentan altisima multicolinealidad: 83 pares con |r| > 0.7.

### Top 10 pares mas correlacionados

| Variable 1 | Variable 2 | r |
|---|---|---:|
| hla_low_res_6 | hla_low_res_8 | 0.975 |
| hla_high_res_8 | hla_high_res_6 | 0.970 |
| hla_low_res_8 | hla_low_res_10 | 0.960 |
| hla_high_res_8 | hla_high_res_10 | 0.951 |
| hla_low_res_6 | hla_low_res_10 | 0.935 |
| hla_high_res_6 | hla_high_res_10 | 0.923 |
| hla_low_res_6 | hla_match_b_low | 0.884 |
| hla_low_res_6 | hla_match_drb1_low | 0.881 |
| hla_match_b_low | hla_low_res_8 | 0.876 |
| hla_high_res_6 | hla_match_b_high | 0.869 |

**Recomendacion**: Crear un score compuesto `hla_match_quality` que combine las variables HLA, reduciendo ~20 variables a 1-3.

---

## 5. Recomendaciones para Feature Engineering

### Fase 1: Correcciones inmediatas

1. Corregir `high_risk_comorbidity` en `m1_preprocessing.py` (actualmente constante, sin varianza).

### Fase 2: Features nuevas sugeridas

1. `age_x_comorbidity` = age_at_hct x comorbidity_score
2. `dri_x_conditioning` = dri_score_encoded x conditioning_encoded
3. `karnofsky_x_comorbidity` = karnofsky_score x comorbidity_score
4. Consolidar HLA en `hla_composite` (media de hla_high_res_8, hla_high_res_10, hla_nmdp_6)
5. Indicadores de missingness para variables con alta ausencia: tce_match, mrd_hct, tce_imm_match, cyto_score_detail
6. `n_comorbidities` = conteo de comorbilidades activas

### Fase 3: Variables a eliminar o consolidar

- Eliminar: hla_high_res_6, hla_low_res_10, hla_match_a_high (redundantes)
- Evaluar: donor_age (puede aportar via age_donor_diff)
- Evaluar: renal_issue (puede estar capturada por comorbidity_score)

---

## 6. Conclusion

Las 57 variables predictoras se agrupan en 7 dominios clinicos por importancia:

1. **Factores demograficos/temporales**: ethnicity, year_hct, race_group, age_at_hct
2. **Comorbilidades**: prior_tumor, cardiac, diabetes, pulm_severe, hepatic_severe, rheum_issue, obesity, peptic_ulcer, psych_disturb, vent_hist
3. **Caracteristicas de la enfermedad**: prim_disease_hct, dri_score, cyto_score, mrd_hct
4. **Tratamiento/trasplante**: conditioning_intensity, gvhd_proph, tbi_status, in_vivo_tcd, rituximab, prod_type, melphalan_dose
5. **Compatibilidad donante**: cmv_status, sex_match, donor_related, tce_imm_match, tce_match, tce_div_match
6. **HLA matching**: hla_high_res_8, hla_low_res_8, hla_nmdp_6, hla_match_b_low, hla_match_c_high, hla_match_drb1_high
7. **Estado funcional**: comorbidity_score, karnofsky_score

52 de 57 variables son relevantes (coeficiente no nulo). Solo 5 son eliminadas por redundancia.
