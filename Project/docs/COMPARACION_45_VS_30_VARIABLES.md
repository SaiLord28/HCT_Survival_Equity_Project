# Comparación y plan de ajuste: modelo de 45 a 30 variables

## ¿Es viable pasar de 45 a 30?

Sí, es viable si se mantiene la lógica clínica actual:

1. Conservar variables clínicamente críticas forzadas (comorbilidades + variables clínicas prioritarias).
2. Completar hasta 30 con ranking combinado estadístico + ML.
3. Validar que la reducción no degrade de forma relevante AUC y métricas de equidad.

## Qué cambiar

- Unificar el valor por defecto de selección de variables en **30** dentro del flujo de entrenamiento.
- Mantener la estrategia de selección existente (no cambia el método, solo el tamaño objetivo).
- Ajustar el script de reentrenamiento para entrenar con 30 variables por defecto.

## Qué unificar

Para evitar inconsistencias, el valor por defecto quedó alineado en:

- `ai_service/api.py` (`TrainRequest.n_features`)
- `ai_service/src/pipeline.py` (`HCTPipeline.train`)
- `ai_service/src/m3_features.py` (`FeatureSelector.select_features`)
- `scripts/retrain_model.py` (payload de reentrenamiento)

## Cómo realizarlo (flujo recomendado)

1. Reentrenar con `n_features=30`.
2. Revisar salida de validación cruzada (AUC media y desviación).
3. Verificar métricas de equidad (`fairness_passed`, disparidad entre grupos).
4. Comparar contra el modelo de 45:
   - Si caída de desempeño/equidad es baja, adoptar 30.
   - Si hay deterioro importante, subir a un punto intermedio (ej. 35).
