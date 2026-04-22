import json

with open("Project/submission/HCT_Survival_Model.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        text = "".join(cell["source"])
        
        # Simple replacements for headers
        text = text.replace("# HCT Survival Equity - Modelo de Predicción Completo", "# HCT Survival Equity - Complete Prediction Model")
        text = text.replace("**Competencia**", "**Competition**")
        text = text.replace("**Objetivo**", "**Objective**")
        text = text.replace("Predecir la probabilidad de supervivencia libre de eventos", "Predict the probability of event-free survival")
        text = text.replace("optimizando el", "optimizing the")
        text = text.replace("por grupo racial.", "by race group.")
        text = text.replace("Carga y exploración de datos", "Data loading and exploration")
        text = text.replace("Preprocesamiento", "Preprocessing")
        text = text.replace("Entrenamiento con Cross-Validation", "Training with Cross-Validation")
        text = text.replace("Evaluación con métrica de competencia", "Evaluation with competition metric")
        text = text.replace("Generación de submission", "Submission generation")
        text = text.replace("Instalación de dependencias", "Dependencies Installation")
        text = text.replace("Imports y Configuración", "Imports and Setup")
        text = text.replace("Carga de Datos", "Data Loading")
        text = text.replace("Métrica de Competencia", "Competition Metric")
        text = text.replace("Basado en el análisis Lasso y las soluciones ganadoras:", "Based on the Lasso analysis and winning solutions:")
        text = text.replace("Convertir numéricas a categóricas (técnica del 1er lugar)", "Convert numerical to categorical (1st place technique)")
        text = text.replace("Crear interacciones clínicas", "Create clinical interactions")
        text = text.replace("Score compuesto HLA", "Composite HLA score")
        text = text.replace("Indicadores de missingness", "Missingness indicators")
        text = text.replace("Entrenamiento: GBM Clasificador", "Training: GBM Classifier")
        text = text.replace("Estrategia inspirada en la solución del 1er lugar:", "Strategy inspired by the 1st place solution:")
        text = text.replace("Entrenar un **clasificador** que prediga", "Train a **classifier** to predict")
        text = text.replace("Entrenar un **regresor** que prediga", "Train a **regressor** to predict")
        text = text.replace("Combinar ambas predicciones para el score final", "Combine both predictions for the final score")
        text = text.replace("Entrenamiento: GBM Regresor", "Training: GBM Regressor")
        text = text.replace("Combinar Predicciones y Evaluar", "Combine Predictions and Evaluate")
        text = text.replace("Estrategia del 1er lugar: combinar clasificador y regresor.", "1st place strategy: combine classifier and regressor.")
        text = text.replace("El **clasificador** da P(Event) → invertir: `1 - P(Event)` = score de supervivencia", "The **classifier** gives P(Event) → invert: `1 - P(Event)` = survival score")
        text = text.replace("El **regresor** da tiempo predicho → directamente proporcional a supervivencia", "The **regressor** gives predicted time → directly proportional to survival")
        text = text.replace("El score final es un blend de ambos", "The final score is a blend of both")
        text = text.replace("Análisis de Equidad por Grupo Racial", "Equity Analysis by Race Group")
        text = text.replace("Generar Submission", "Generate Submission")
        text = text.replace("Resumen Final", "Final Summary")
        
        cell["source"] = [text]

with open("Project/submission/HCT_Survival_Model.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

