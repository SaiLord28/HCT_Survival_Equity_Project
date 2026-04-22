#!/usr/bin/env python3
"""
Generador de Dataset Optimizado para HCT Survival
===================================================
Basado en las recomendaciones de optimización:
  1. ELIMINAR variables redundantes y con fuga
  2. MANTENER las top predictors
  3. CREAR nuevas features que REEMPLAZAN las originales

Entrada:  data/raw/train.csv, data/raw/test.csv
Salida:   data/optimized/train_optimized.csv, data/optimized/test_optimized.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──
BASE_DIR = Path(__file__).parent
DATA_RAW = BASE_DIR / 'data' / 'raw'
DATA_OUT = BASE_DIR / 'data' / 'optimized'
DATA_OUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("GENERACIÓN DE DATASET OPTIMIZADO")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
# 1. CARGAR DATOS
# ══════════════════════════════════════════════════════════════════════
train = pd.read_csv(DATA_RAW / 'train.csv')
test = pd.read_csv(DATA_RAW / 'test.csv')

print(f"\n📥 Datos cargados:")
print(f"   Train: {train.shape[0]} filas × {train.shape[1]} columnas")
print(f"   Test:  {test.shape[0]} filas × {test.shape[1]} columnas")

# Guardar targets y metadata antes de transformar
train_id = train['ID'].copy()
test_id = test['ID'].copy()
train_efs = train['efs'].copy()
train_efs_time = train['efs_time'].copy()

# Contar features originales (sin ID, efs, efs_time)
original_feature_count = len(train.columns) - 3  # 57


# ══════════════════════════════════════════════════════════════════════
# 2. DEFINIR QUÉ SE ELIMINA, QUÉ SE CREA, QUÉ SE REEMPLAZA
# ══════════════════════════════════════════════════════════════════════

# --- PASO 1: Eliminación directa (redundancia / fuga) ---
DROP_DIRECT = [
    'ID', 'efs', 'efs_time',          # Fuga de datos
    'hla_high_res_6',                   # r=0.97 con hla_high_res_8
    'hla_low_res_10',                   # r=0.96 con hla_low_res_8
    'hla_low_res_6',                    # r=0.97 con hla_low_res_8
    'hla_match_a_high',                 # redundante con HLA agregadas
    'renal_issue',                      # coef Lasso = 0, capturada por comorbidity_score
    'donor_age',                        # capturada por age_at_hct + donor_related
]

# --- PASO 2: Variables HLA individuales → reemplazadas por hla_composite ---
# hla_composite = media(hla_high_res_8, hla_high_res_10, hla_nmdp_6)
# Una vez creada, las individuales de locus sobran
HLA_REPLACED_BY_COMPOSITE = [
    'hla_high_res_8',       # entra en el cálculo del composite
    'hla_high_res_10',      # entra en el cálculo del composite
    'hla_nmdp_6',           # entra en el cálculo del composite
    'hla_match_b_low',      # individual → capturada por composite
    'hla_match_b_high',     # individual → capturada por composite
    'hla_match_c_low',      # individual → capturada por composite
    'hla_match_c_high',     # individual → capturada por composite
    'hla_match_drb1_low',   # individual → capturada por composite
    'hla_match_drb1_high',  # individual → capturada por composite
    'hla_match_dqb1_low',   # individual → capturada por composite
    'hla_match_dqb1_high',  # individual → capturada por composite
    'hla_match_a_low',      # individual → capturada por composite
    'hla_low_res_8',        # agregada baja res → capturada por composite
]

# --- PASO 3: Categóricas reemplazadas por sus encodings ordinales ---
# dri_score → dri_ordinal, conditioning_intensity → conditioning_ordinal
# Las categóricas originales ya no son necesarias
REPLACED_BY_ORDINAL = [
    'dri_score',               # → dri_ordinal
    'conditioning_intensity',  # → conditioning_ordinal
]

# --- PASO 4: Comorbilidades individuales → reemplazadas por n_comorbidities + risk ---
# Se mantiene comorbidity_score (Sorror) como numérica, pero las 13
# banderas individuales se consolidan en n_comorbidities + n_comorbidities_risk
COMORBIDITIES_INDIVIDUAL = [
    'cardiac', 'arrhythmia', 'diabetes', 'hepatic_mild', 'hepatic_severe',
    'obesity', 'peptic_ulcer', 'prior_tumor', 'psych_disturb',
    'pulm_moderate', 'pulm_severe', 'rheum_issue', 'vent_hist',
]

# --- PASO 5: graft_type y prod_type son redundantes entre sí ---
# prod_type es básicamente lo mismo que graft_type (PB/BM)
DROP_DUPLICATE_COLS = [
    'prod_type',  # duplicado de graft_type
]

# Total a eliminar
ALL_TO_DROP = list(set(
    DROP_DIRECT + HLA_REPLACED_BY_COMPOSITE + REPLACED_BY_ORDINAL +
    COMORBIDITIES_INDIVIDUAL + DROP_DUPLICATE_COLS
))


# ══════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING (crear ANTES de eliminar, porque usamos las originales)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PASO 1: CREAR NUEVAS FEATURES")
print("=" * 70)

def crear_features(df, df_original):
    """
    Crea features con respaldo clínico. Usa las columnas originales
    para calcular, luego se eliminan las originales.
    """
    df = df.copy()

    # ── A) hla_composite: Carga Genética Total ──
    hla_for_composite = ['hla_high_res_8', 'hla_high_res_10', 'hla_nmdp_6']
    hla_for_composite = [c for c in hla_for_composite if c in df.columns]
    if hla_for_composite:
        df['hla_composite'] = df[hla_for_composite].mean(axis=1)
        print("   ✓ hla_composite = media(hla_high_res_8, hla_high_res_10, hla_nmdp_6)")
        print(f"     → REEMPLAZA {len(HLA_REPLACED_BY_COMPOSITE)} variables HLA individuales")

    # ── B) age_x_comorbidity: El Choque Físico ──
    if 'age_at_hct' in df.columns and 'comorbidity_score' in df.columns:
        df['age_x_comorbidity'] = (
            df['age_at_hct'].fillna(df['age_at_hct'].median()) *
            df['comorbidity_score'].fillna(0)
        )
        print("   ✓ age_x_comorbidity = age_at_hct × comorbidity_score")

    # ── C) dri_x_conditioning: El Balance del Tratamiento ──
    dri_map = {
        'Low': 0,
        'Intermediate': 1,
        'Intermediate - TED AML case <missing cytogenetics': 1,
        'High': 2,
        'High - TED AML case <missing cytogenetics': 2,
        'Very high': 3,
        'N/A - non-malignant indication': 0,
        'N/A - pediatric': 0,
        'N/A - disease not classifiable': 1,
        'TBD cytogenetics': 1,
        'Missing disease status': 1,
    }
    cond_map = {
        'NMA': 0,
        'RIC': 1,
        'MAC': 2,
        'TBD': 1,
        'No drugs reported': 0,
        'N/A, F(pre-TED) not submitted': 1,
    }
    if 'dri_score' in df.columns and 'conditioning_intensity' in df.columns:
        df['dri_ordinal'] = df['dri_score'].map(dri_map).fillna(1)
        df['conditioning_ordinal'] = df['conditioning_intensity'].map(cond_map).fillna(1)
        df['dri_x_conditioning'] = df['dri_ordinal'] * df['conditioning_ordinal']
        print("   ✓ dri_ordinal, conditioning_ordinal, dri_x_conditioning")
        print(f"     → REEMPLAZAN dri_score y conditioning_intensity categóricas")

    # ── D) missing_count: Vulnerabilidad/Acceso ──
    missing_mask = df_original.drop(columns=['ID', 'efs', 'efs_time'], errors='ignore').isna()
    not_done_mask = df_original.drop(columns=['ID', 'efs', 'efs_time'], errors='ignore').apply(
        lambda col: col.astype(str).str.strip().str.lower().isin(['not done'])
    )
    df['missing_count'] = (missing_mask | not_done_mask).sum(axis=1).values
    print("   ✓ missing_count = conteo de NaN + 'Not done' por paciente")

    # ── E) n_comorbidities + n_comorbidities_risk ──
    def is_yes(series):
        return series.astype(str).str.strip().str.lower().isin(['yes', 'y', '1', 'true']).astype(int)

    comorbidity_cols = [c for c in COMORBIDITIES_INDIVIDUAL if c in df.columns]
    if comorbidity_cols:
        df['n_comorbidities'] = sum(is_yes(df[c]) for c in comorbidity_cols)
        print(f"   ✓ n_comorbidities = conteo de {len(comorbidity_cols)} comorbilidades")
        print(f"     → REEMPLAZA las {len(comorbidity_cols)} banderas individuales")

    if all(c in df.columns for c in ['obesity', 'diabetes', 'psych_disturb']):
        metabolic = (is_yes(df['obesity']) | is_yes(df['diabetes']))
        mental = is_yes(df['psych_disturb'])
        df['n_comorbidities_risk'] = (metabolic & mental).astype(int)
        print("   ✓ n_comorbidities_risk = (obesidad|diabetes) & psych_disturb")

    # ── F) age_donor_diff ──
    if 'donor_age' in df_original.columns and 'age_at_hct' in df.columns:
        df['age_donor_diff'] = df['age_at_hct'] - df_original['donor_age'].values[:len(df)]
        print("   ✓ age_donor_diff = age_at_hct - donor_age")

    # ── G) karnofsky_x_comorbidity ──
    if 'karnofsky_score' in df.columns and 'comorbidity_score' in df.columns:
        df['karnofsky_x_comorbidity'] = (
            df['karnofsky_score'].fillna(90) *
            df['comorbidity_score'].fillna(0)
        )
        print("   ✓ karnofsky_x_comorbidity = karnofsky × comorbidity_score")

    return df


print("\n   --- Train ---")
train_fe = crear_features(train, train)
print("\n   --- Test ---")
test_fe = crear_features(test, test)


# ══════════════════════════════════════════════════════════════════════
# 4. ELIMINAR LAS ORIGINALES QUE FUERON REEMPLAZADAS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PASO 2: ELIMINAR VARIABLES REEMPLAZADAS")
print("=" * 70)

train_dropped = [c for c in ALL_TO_DROP if c in train_fe.columns]
test_dropped = [c for c in ALL_TO_DROP if c in test_fe.columns]

print(f"\n   Eliminando de Train: {len(train_dropped)} columnas")
print(f"   Eliminando de Test:  {len(test_dropped)} columnas")

# Categorizar las eliminaciones
print("\n   Detalle de eliminaciones:")
for c in sorted(train_dropped):
    if c in DROP_DIRECT:
        reason = "fuga/redundancia directa"
    elif c in HLA_REPLACED_BY_COMPOSITE:
        reason = "→ reemplazada por hla_composite"
    elif c in REPLACED_BY_ORDINAL:
        reason = "→ reemplazada por encoding ordinal"
    elif c in COMORBIDITIES_INDIVIDUAL:
        reason = "→ reemplazada por n_comorbidities"
    elif c in DROP_DUPLICATE_COLS:
        reason = "duplicado"
    else:
        reason = "otro"
    print(f"     ✗ {c:<30s} ({reason})")

train_clean = train_fe.drop(columns=[c for c in ALL_TO_DROP if c in train_fe.columns])
test_clean = test_fe.drop(columns=[c for c in ALL_TO_DROP if c in test_fe.columns])


# ══════════════════════════════════════════════════════════════════════
# 5. RE-AGREGAR TARGETS AL TRAIN
# ══════════════════════════════════════════════════════════════════════
train_final = train_clean.copy()
train_final.insert(0, 'ID', train_id.values)
train_final['efs'] = train_efs.values
train_final['efs_time'] = train_efs_time.values

test_final = test_clean.copy()
test_final.insert(0, 'ID', test_id.values)


# ══════════════════════════════════════════════════════════════════════
# 6. RESUMEN Y VERIFICACIÓN
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RESUMEN FINAL")
print("=" * 70)

final_feature_cols = set(train_final.columns) - {'ID', 'efs', 'efs_time'}
original_features = set(train.columns) - {'ID', 'efs', 'efs_time'}
dropped_features = original_features - final_feature_cols
created_features = final_feature_cols - original_features
kept_features = original_features & final_feature_cols

print(f"\n   📊 Variables originales (features): {len(original_features)}")
print(f"   ✗ Eliminadas:                       {len(dropped_features)}")
print(f"   ✓ Mantenidas sin cambio:             {len(kept_features)}")
print(f"   ✚ Creadas (reemplazo):               {len(created_features)}")
print(f"   ─────────────────────────────────────")
print(f"   📦 Variables finales (features):     {len(final_feature_cols)}")
print(f"   📉 Reducción neta:                   {len(original_features)} → {len(final_feature_cols)} ({len(original_features) - len(final_feature_cols)} menos)")

print(f"\n   Variables MANTENIDAS ({len(kept_features)}):")
for c in sorted(kept_features):
    print(f"     • {c}")

print(f"\n   Variables CREADAS ({len(created_features)}):")
for c in sorted(created_features):
    print(f"     + {c}")

print(f"\n   Train final: {train_final.shape[0]} filas × {train_final.shape[1]} columnas")
print(f"   Test final:  {test_final.shape[0]} filas × {test_final.shape[1]} columnas")

# Verificación: las nuevas features no deben ser constantes
print("\n" + "=" * 70)
print("VERIFICACIÓN DE NUEVAS FEATURES")
print("=" * 70)
for feat in sorted(created_features):
    col = train_final[feat]
    n_unique = col.nunique()
    n_null = col.isna().sum()
    if col.dtype in ['float64', 'int64', 'float32', 'int32']:
        print(f"   {feat:<30s} unique={n_unique:>5d}  nulls={n_null:>5d}  "
              f"min={col.min():>8.2f}  median={col.median():>8.2f}  max={col.max():>8.2f}")
    else:
        print(f"   {feat:<30s} unique={n_unique:>5d}  nulls={n_null:>5d}  type={col.dtype}")

    # Alerta si es constante
    if n_unique <= 1:
        print(f"     ⚠️ ALERTA: {feat} es CONSTANTE — revisar lógica")


# ══════════════════════════════════════════════════════════════════════
# 7. GUARDAR
# ══════════════════════════════════════════════════════════════════════
train_path = DATA_OUT / 'train_optimized.csv'
test_path = DATA_OUT / 'test_optimized.csv'

train_final.to_csv(train_path, index=False)
test_final.to_csv(test_path, index=False)

print(f"\n💾 Archivos guardados:")
print(f"   {train_path}")
print(f"   {test_path}")
print(f"\n✅ Dataset optimizado: {len(original_features)} → {len(final_feature_cols)} features")
