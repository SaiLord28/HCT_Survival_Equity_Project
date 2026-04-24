#!/usr/bin/env python3
"""
Optimized Dataset Generator for HCT Survival
===================================================
Based on optimization recommendations:
  1. REMOVE redundant and leaked variables
  2. KEEP top predictors
  3. CREATE new features that REPLACE original ones

Input:  data/raw/train.csv, data/raw/test.csv
Output:   data/optimized/train_optimized.csv, data/optimized/test_optimized.csv
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
print("OPTIMIZED DATASET GENERATION")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════
train = pd.read_csv(DATA_RAW / 'train.csv')
test = pd.read_csv(DATA_RAW / 'test.csv')

print(f"\n📥 Data loaded:")
print(f"   Train: {train.shape[0]} rows × {train.shape[1]} columns")
print(f"   Test:  {test.shape[0]} rows × {test.shape[1]} columns")

# Save targets and metadata before transforming
train_id = train['ID'].copy()
test_id = test['ID'].copy()
train_efs = train['efs'].copy()
train_efs_time = train['efs_time'].copy()

# Count original features (sin ID, efs, efs_time)
original_feature_count = len(train.columns) - 3  # 57


# ══════════════════════════════════════════════════════════════════════
# 2. DEFINE WHAT IS REMOVED, CREATED, REPLACED
# ══════════════════════════════════════════════════════════════════════

# --- STEP 1: Direct removal (redundancy / leakage) ---
DROP_DIRECT = [
    'ID', 'efs', 'efs_time',          # Data leakage
    'hla_high_res_6',                   # r=0.97 con hla_high_res_8
    'hla_low_res_10',                   # r=0.96 con hla_low_res_8
    'hla_low_res_6',                    # r=0.97 con hla_low_res_8
    'hla_match_a_high',                 # redundant with aggregated HLAs
    'renal_issue',                      # coef Lasso = 0, captured by comorbidity_score
    'donor_age',                        # captured by age_at_hct + donor_related
]

# --- STEP 2: Individual HLA variables -> replaced by hla_composite ---
# hla_composite = mean(hla_high_res_8, hla_high_res_10, hla_nmdp_6)
# Once created, locus individual ones are redundant
HLA_REPLACED_BY_COMPOSITE = [
    'hla_high_res_8',       # enters composite calculation
    'hla_high_res_10',      # enters composite calculation
    'hla_nmdp_6',           # enters composite calculation
    'hla_match_b_low',      # individual -> captured by composite
    'hla_match_b_high',     # individual -> captured by composite
    'hla_match_c_low',      # individual -> captured by composite
    'hla_match_c_high',     # individual -> captured by composite
    'hla_match_drb1_low',   # individual -> captured by composite
    'hla_match_drb1_high',  # individual -> captured by composite
    'hla_match_dqb1_low',   # individual -> captured by composite
    'hla_match_dqb1_high',  # individual -> captured by composite
    'hla_match_a_low',      # individual -> captured by composite
    'hla_low_res_8',        # aggregated low res -> captured by composite
]

# --- PASO 3: Categoricals replaced by their ordinal encodings ---
# dri_score → dri_ordinal, conditioning_intensity → conditioning_ordinal
# Original categoricals are no longer needed
REPLACED_BY_ORDINAL = [
    'dri_score',               # → dri_ordinal
    'conditioning_intensity',  # → conditioning_ordinal
]

# --- PASO 4: Individual comorbidities -> replaced by n_comorbidities + risk ---
# Keep comorbidity_score (Sorror) as numerical, but the 13
# individual flags consolidate into n_comorbidities + n_comorbidities_risk
COMORBIDITIES_INDIVIDUAL = [
    'cardiac', 'arrhythmia', 'diabetes', 'hepatic_mild', 'hepatic_severe',
    'obesity', 'peptic_ulcer', 'prior_tumor', 'psych_disturb',
    'pulm_moderate', 'pulm_severe', 'rheum_issue', 'vent_hist',
]

# --- PASO 5: graft_type and prod_type are mutually redundant ---
# prod_type es básicamente lo mismo que graft_type (PB/BM)
DROP_DUPLICATE_COLS = [
    'prod_type',  # duplicate of graft_type
]

# Total to remove
ALL_TO_DROP = list(set(
    DROP_DIRECT + HLA_REPLACED_BY_COMPOSITE + REPLACED_BY_ORDINAL +
    COMORBIDITIES_INDIVIDUAL + DROP_DUPLICATE_COLS
))


# ══════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING (create BEFORE deleting, since we use the originals)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1: CREATE NEW FEATURES")
print("=" * 70)

def crear_features(df, df_original):
    """
    Crea features con respaldo clínico. Usa las columns originales
    to calculate, then the originals are removed.
    """
    df = df.copy()

    # ── A) hla_composite: Total Genetic Load ──
    hla_for_composite = ['hla_high_res_8', 'hla_high_res_10', 'hla_nmdp_6']
    hla_for_composite = [c for c in hla_for_composite if c in df.columns]
    if hla_for_composite:
        df['hla_composite'] = df[hla_for_composite].mean(axis=1)
        print("   ✓ hla_composite = mean(hla_high_res_8, hla_high_res_10, hla_nmdp_6)")
        print(f"     → REPLACES {len(HLA_REPLACED_BY_COMPOSITE)} individual HLA variables")

    # ── B) age_x_comorbidity: Physical Shock ──
    if 'age_at_hct' in df.columns and 'comorbidity_score' in df.columns:
        df['age_x_comorbidity'] = (
            df['age_at_hct'].fillna(df['age_at_hct'].meann()) *
            df['comorbidity_score'].fillna(0)
        )
        print("   ✓ age_x_comorbidity = age_at_hct × comorbidity_score")

    # ── C) dri_x_conditioning: Treatment Balance ──
    dri_map = {
        'Low': 0,
        'Intermeante': 1,
        'Intermeante - TED AML case <missing cytogenetics': 1,
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
        print(f"     → REPLACESN dri_score y conditioning_intensity categoricals")

    # ── D) missing_count: Vulnerability/Access ──
    missing_mask = df_original.drop(columns=['ID', 'efs', 'efs_time'], errors='ignore').isna()
    not_done_mask = df_original.drop(columns=['ID', 'efs', 'efs_time'], errors='ignore').apply(
        lambda col: col.astype(str).str.strip().str.lower().isin(['not done'])
    )
    df['missing_count'] = (missing_mask | not_done_mask).sum(axis=1).values
    print("   ✓ missing_count = NaN + 'Not done' count per patient")

    # ── E) n_comorbidities + n_comorbidities_risk ──
    def is_yes(series):
        return series.astype(str).str.strip().str.lower().isin(['yes', 'y', '1', 'true']).astype(int)

    comorbidity_cols = [c for c in COMORBIDITIES_INDIVIDUAL if c in df.columns]
    if comorbidity_cols:
        df['n_comorbidities'] = sum(is_yes(df[c]) for c in comorbidity_cols)
        print(f"   ✓ n_comorbidities = count of {len(comorbidity_cols)} comorbidities")
        print(f"     → REPLACES las {len(comorbidity_cols)} individual flags")

    if all(c in df.columns for c in ['obesity', 'diabetes', 'psych_disturb']):
        metabolic = (is_yes(df['obesity']) | is_yes(df['diabetes']))
        mental = is_yes(df['psych_disturb'])
        df['n_comorbidities_risk'] = (metabolic & mental).astype(int)
        print("   ✓ n_comorbidities_risk = (obesity|diabetes) & psych_disturb")

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
# 4. ELIMINAR LAS ORIGINALES QUE FUERON REPLACESDAS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PASO 2: ELIMINAR VARIABLES REPLACESDAS")
print("=" * 70)

train_dropped = [c for c in ALL_TO_DROP if c in train_fe.columns]
test_dropped = [c for c in ALL_TO_DROP if c in test_fe.columns]

print(f"\n   Removing from Train: {len(train_dropped)} columns")
print(f"   Removing from Test:  {len(test_dropped)} columns")

# Categorizar las eliminaciones
print("\n   Removal details:")
for c in sorted(train_dropped):
    if c in DROP_DIRECT:
        reason = "leakage/direct redundancy"
    elif c in HLA_REPLACED_BY_COMPOSITE:
        reason = "→ reemplazada por hla_composite"
    elif c in REPLACED_BY_ORDINAL:
        reason = "→ replaced by ordinal encoding"
    elif c in COMORBIDITIES_INDIVIDUAL:
        reason = "→ replaced by n_comorbidities"
    elif c in DROP_DUPLICATE_COLS:
        reason = "duplicate"
    else:
        reason = "other"
    print(f"     ✗ {c:<30s} ({reason})")

train_clean = train_fe.drop(columns=[c for c in ALL_TO_DROP if c in train_fe.columns])
test_clean = test_fe.drop(columns=[c for c in ALL_TO_DROP if c in test_fe.columns])


# ══════════════════════════════════════════════════════════════════════
# 5. RE-ADD TARGETS TO TRAIN
# ══════════════════════════════════════════════════════════════════════
train_final = train_clean.copy()
train_final.insert(0, 'ID', train_id.values)
train_final['efs'] = train_efs.values
train_final['efs_time'] = train_efs_time.values

test_final = test_clean.copy()
test_final.insert(0, 'ID', test_id.values)


# ══════════════════════════════════════════════════════════════════════
# 6. SUMMARY AND VERIFICATION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

final_feature_cols = set(train_final.columns) - {'ID', 'efs', 'efs_time'}
original_features = set(train.columns) - {'ID', 'efs', 'efs_time'}
dropped_features = original_features - final_feature_cols
created_features = final_feature_cols - original_features
kept_features = original_features & final_feature_cols

print(f"\n   📊 Original variables (features): {len(original_features)}")
print(f"   ✗ Removed:                       {len(dropped_features)}")
print(f"   ✓ Maintained without changes:             {len(kept_features)}")
print(f"   ✚ Created (replacement):               {len(created_features)}")
print(f"   ─────────────────────────────────────")
print(f"   📦 Final variables (features):     {len(final_feature_cols)}")
print(f"   📉 Net reduction:                   {len(original_features)} → {len(final_feature_cols)} ({len(original_features) - len(final_feature_cols)} less)")

print(f"\n   Variables KEPT ({len(kept_features)}):")
for c in sorted(kept_features):
    print(f"     • {c}")

print(f"\n   Variables CREATED ({len(created_features)}):")
for c in sorted(created_features):
    print(f"     + {c}")

print(f"\n   Final Train: {train_final.shape[0]} rows × {train_final.shape[1]} columns")
print(f"   Final Test:  {test_final.shape[0]} rows × {test_final.shape[1]} columns")

# Verification: new features should not be constant
print("\n" + "=" * 70)
print("NEW FEATURES VERIFICATION")
print("=" * 70)
for feat in sorted(created_features):
    col = train_final[feat]
    n_unique = col.nunique()
    n_null = col.isna().sum()
    if col.dtype in ['float64', 'int64', 'float32', 'int32']:
        print(f"   {feat:<30s} unique={n_unique:>5d}  nulls={n_null:>5d}  "
              f"min={col.min():>8.2f}  meann={col.meann():>8.2f}  max={col.max():>8.2f}")
    else:
        print(f"   {feat:<30s} unique={n_unique:>5d}  nulls={n_null:>5d}  type={col.dtype}")

    # Alerta si es constante
    if n_unique <= 1:
        print(f"     ⚠️ ALERT: {feat} is CONSTANT — check logic")


# ══════════════════════════════════════════════════════════════════════
# 7. SAVE
# ══════════════════════════════════════════════════════════════════════
train_path = DATA_OUT / 'train_optimized.csv'
test_path = DATA_OUT / 'test_optimized.csv'

train_final.to_csv(train_path, index=False)
test_final.to_csv(test_path, index=False)

print(f"\n💾 Files saved:")
print(f"   {train_path}")
print(f"   {test_path}")
print(f"\n✅ Optimized dataset: {len(original_features)} → {len(final_feature_cols)} features")
