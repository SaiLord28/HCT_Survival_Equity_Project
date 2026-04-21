# Comparacion: Modelo Optimizado vs Modelo de 60 Variables

Fecha de ejecucion: 2026-04-21
Proyecto: HCT Survival Equity
Modelo usado en ambas comparaciones: GBM

## 1) Objetivo

Realizar una comparacion directa entre:
- Baseline original con grupo de 60 variables (dataset original).
- Nuevo dataset optimizado (adjunto `train_optimizado.csv`) evaluado con el mismo pipeline.

## 2) Preparacion del dataset optimizado

Archivo de entrada:
- `/home/leo/Descargas/train_optimizado.csv`

Archivo usado por pipeline:
- `Project/ai_service/data/raw/train_optimizado_pipeline.csv`

Ajustes realizados para que sea compatible y valido:
1. Se creo columna `efs` (0/1) a partir de `efs_numeric`.
2. Se elimino `efs_numeric` de features para evitar fuga de target (data leakage).
3. Se mantuvo `efs_time` como columna original (el pipeline ya la excluye de features automaticamente).

## 3) Variables usadas en el modelo optimizado

Cantidad final seleccionada: 20

Lista de variables seleccionadas por M3 en optimizado:
1. comorbidity_score
2. hla_match_quality
3. year_hct
4. cond_RIC
5. in_vivo_tcd_Yes
6. race_group
7. dri_Intermediate
8. gvhd_FK_MMF
9. Total_HLA_High_Match
10. KPS_Under_80
11. sex_FM
12. disease_IEA
13. cmv_plus_minus
14. Comorb_x_OrganFail
15. disease_IIS
16. tbi_TBI_Cy
17. hla_match_b_low
18. Organ_Failure_Index
19. sex_MF
20. dri_Pediatric

Nota: aunque se solicitan 45 features en configuracion, el dataset optimizado solo ofrece 20 features modelables tras preprocesamiento; por eso `all_variables_model` y `feature_engineered_model` quedan en 20.

## 4) Comparacion numerica

### 4.1 Baseline original (grupo de 60 variables)

| Configuracion | N features | CV Accuracy | CV AUC |
|---|---:|---:|---:|
| Todas las variables (original) | 60 | 0.6798958333 | 0.7417819508 |
| Feature engineering (original) | 45 | 0.6793055556 | 0.7414772292 |

Delta (45 - 60) en original:
- CV AUC: -0.0003047216
- CV Accuracy: -0.0005902778
- `performance_maintained = true`

### 4.2 Dataset optimizado

| Configuracion | N features | CV Accuracy | CV AUC |
|---|---:|---:|---:|
| Todas las variables (opt) | 20 | 0.6522569444 | 0.7032091233 |
| Feature engineering (opt) | 20 | 0.6529166667 | 0.7033182111 |

Delta (FE - all) en optimizado:
- CV AUC: +0.0001090878
- CV Accuracy: +0.0006597222
- `performance_maintained = true`

### 4.3 Comparacion directa Optimizado vs 60 (baseline)

Tomando como referencia el baseline de 60 variables:
- CV AUC optimizado (20): 0.7033182111 vs 0.7417819508 (60)
  - Diferencia absoluta: -0.0384637397
- CV Accuracy optimizado (20): 0.6529166667 vs 0.6798958333 (60)
  - Diferencia absoluta: -0.0269791666

## 5) Interpretacion

1. El optimizado funciona correctamente con su set reducido y muestra una mejora marginal interna (FE vs all dentro del propio optimizado).
2. Frente al baseline de 60 variables, el optimizado pierde capacidad predictiva (AUC y Accuracy).
3. Conclusión operativa: hoy, el modelo de 60 variables sigue siendo superior en performance; el optimizado reduce dimensionalidad y complejidad, pero con costo de precisión.

## 6) Archivos clave involucrados

- `Project/ai_service/src/m1_preprocessing.py`
- `Project/ai_service/src/m3_features.py`
- `Project/ai_service/src/pipeline.py`
- `Project/ai_service/api.py`
- `Project/ai_service/data/raw/train_optimizado_pipeline.csv`
- `Project/docs/FEATURE_ENGINEERING_VS_60_VARIABLES.md`
