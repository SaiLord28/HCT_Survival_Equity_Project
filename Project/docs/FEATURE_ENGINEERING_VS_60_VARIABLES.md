# Comparacion de Feature Engineering vs Modelo con 60 Variables

Fecha de ejecucion: 2026-04-21
Modelo: GBM
Dataset: /app/data/raw/raw/train.csv

## 1) Variables usadas por Feature Engineering (M3)

Cantidad seleccionada: 45

Lista completa seleccionada:
1. cardiac
2. arrhythmia
3. diabetes
4. hepatic_mild
5. hepatic_severe
6. obesity
7. peptic_ulcer
8. prior_tumor
9. psych_disturb
10. pulm_moderate
11. pulm_severe
12. renal_issue
13. rheum_issue
14. vent_hist
15. comorbidity_score
16. cmv_status
17. tbi_status
18. in_vivo_tcd
19. gvhd_proph
20. prod_type
21. rituximab
22. cyto_score_detail
23. hla_low_res_6
24. hla_low_res_8
25. hla_nmdp_6
26. tce_match
27. tce_imm_match
28. age_donor_diff
29. age_at_hct
30. dri_score
31. conditioning_intensity
32. karnofsky_score
33. hla_high_res_8
34. hla_high_res_10
35. hla_match_quality
36. graft_type
37. donor_related
38. donor_age
39. sex_match
40. prim_disease_hct
41. cyto_score
42. mrd_hct
43. year_hct
44. race_group
45. tce_div_match

### Subgrupos relevantes dentro de las 45
- Comorbilidades incluidas: 14 forzadas + comorbidity_score
- Variables clinicas prioritarias incluidas: 16
- Variable sensible incluida en esta corrida: race_group

## 2) Archivos donde se define y usa esta logica

- Project/ai_service/src/m3_features.py
  - Define la logica de seleccion de variables (clinicas, comorbilidades forzadas, combinacion estadistica + ML, y filtro por disparidad de disponibilidad).
- Project/ai_service/src/m1_preprocessing.py
  - Define variables de entrada preprocesadas y features creadas (por ejemplo age_donor_diff, hla_match_quality).
- Project/ai_service/src/pipeline.py
  - Ejecuta la seleccion M3 dentro del flujo de entrenamiento y expone la comparacion entre modelo con todas las variables y modelo con feature engineering.
- Project/ai_service/api.py
  - Expone el endpoint /train con compare_with_all_features para ejecutar la comparacion en una sola corrida.
- Project/ai_service/data/raw/data_dictionary.csv
  - Diccionario de variables base del dataset.

## 3) Comparacion real contra el grupo de 60 variables

### Resultados obtenidos

| Configuracion | N features | CV Accuracy | CV AUC |
|---|---:|---:|---:|
| Todas las variables preprocesadas | 60 | 0.6798958333 | 0.7417819508 |
| Feature engineering (M3) | 45 | 0.6793055556 | 0.7414772292 |

Deltas (45 - 60):
- Delta CV AUC: -0.0003047216
- Delta CV Accuracy: -0.0005902778
- performance_maintained: true (tolerancia 0.01)

### Cuanto mejora realmente frente al grupo de 60

No hay mejora en metrica predictiva central; hay una reduccion minima:
- CV AUC cae ~0.041% relativo
- CV Accuracy cae ~0.087% relativo

Sin embargo, si hay mejora de eficiencia y control clinico:
- Reduccion de dimensionalidad: 60 -> 45 (25% menos variables)
- Se mantiene el rendimiento practicamente igual dentro de tolerancia
- Se garantiza inclusion de comorbilidades y variables clinicas criticas

## 4) Conclusion ejecutiva

Feature Engineering (M3) no incrementa el AUC/Accuracy respecto al conjunto de 60 variables, pero conserva performance con perdida marginal y ofrece un modelo mas compacto y controlado clinicamente. Para este proyecto, el beneficio principal es robustez clinica + simplificacion del modelo, no una ganancia numerica de AUC.
