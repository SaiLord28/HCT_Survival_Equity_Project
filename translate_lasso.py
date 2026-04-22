import re

with open("Project/docs/analysis/lasso_analysis.py", "r") as f:
    text = f.read()

replacements = {
    "Análisis de Importancia de Variables con Lasso": "Variable Importance Analysis with Lasso",
    "Este script realiza": "This script performs",
    "Carga de datos": "Data loading",
    "Preprocesamiento": "Preprocessing",
    "Entrenamiento": "Training",
    "Evaluación": "Evaluation",
    "Análisis de multicolinealidad": "Multicollinearity analysis",
    "Guardado": "Saving",
    "INICIANDO ANÁLISIS LASSO": "STARTING LASSO ANALYSIS",
    "Leyendo datos": "Reading data",
    "Preparando preprocesamiento": "Preparing preprocessing",
    "Numericas": "Numerical",
    "Categoricas": "Categorical",
    "Ajustando el modelo LassoCV": "Fitting LassoCV model",
    "Entrenando modelo final con todo el train": "Training final model with all train",
    "Evaluando modelo": "Evaluating model",
    "RMSE Validación": "Validation RMSE",
    "MAE Validación": "Validation MAE",
    "R2 Validación": "Validation R2",
    "procesos terminados": "processes finished",
    "Extrayendo coeficientes": "Extracting coefficients",
    "Analizando multicolinealidad en vars originales": "Analyzing multicollinearity in original vars",
    "Pares altamente correlacionados": "Highly correlated pairs",
    "Guardando resultados": "Saving results",
    "Resultados guardados en": "Results saved in",
    "Variables eliminadas por Lasso": "Variables eliminated by Lasso",
    "Top 10 variables más importantes": "Top 10 most important variables",
    "Completado exitosamente": "Completed successfully"
}

for k, v in replacements.items():
    text = text.replace(k, v)

with open("Project/docs/analysis/lasso_analysis.py", "w") as f:
    f.write(text)

