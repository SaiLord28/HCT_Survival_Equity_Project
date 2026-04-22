#!/usr/bin/env python3
"""
Comprehensive Lasso Regression Analysis for HCT Survival Prediction
===================================================================
This script performs a proper Lasso (L1) regularized regression analysis
on the HCT survival dataset to identify the most important variables
for feature engineering.

Key aspects:
- Proper handling of censored data (efs/efs_time)
- Robust preprocessing (imputation, encoding, scaling)
- LassoCV with cross-validation to find optimal alpha
- Coefficient analysis to rank variable importance
- Correlation matrix for multicollinearity detection
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, Lasso
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ============================
# 1. Load and inspect data
# ============================
data_dir = Path(__file__).parent.parent / 'ai_service' / 'data' / 'raw'
train_path = data_dir / 'train.csv'
dict_path = data_dir / 'data_dictionary.csv'

if not train_path.exists():
    raise FileNotFoundError(f'train.csv not found at: {train_path}')

print("=" * 80)
print("HCT SURVIVAL - LASSO REGRESSION ANALYSIS")
print("=" * 80)

train = pd.read_csv(train_path)
data_dict = pd.read_csv(dict_path)

print(f"\nDataset shape: {train.shape}")
print(f"  - Rows: {train.shape[0]}")
print(f"  - Columns: {train.shape[1]}")

# Target variable
TARGET = 'efs_time'
EVENT_COL = 'efs'

print(f"\nTarget variable: {TARGET}")
print(f"  - Mean: {train[TARGET].mean():.3f}")
print(f"  - Median: {train[TARGET].median():.3f}")
print(f"  - Std: {train[TARGET].std():.3f}")
print(f"  - Min: {train[TARGET].min():.3f}")
print(f"  - Max: {train[TARGET].max():.3f}")

# EFS distribution
efs_encoded = train[EVENT_COL].map({'Event': 1, 'Censoring': 0}).fillna(0)
print(f"\nEvent distribution (efs):")
print(f"  - Event: {(efs_encoded == 1).sum()} ({(efs_encoded == 1).mean()*100:.1f}%)")
print(f"  - Censoring: {(efs_encoded == 0).sum()} ({(efs_encoded == 0).mean()*100:.1f}%)")

# ============================
# 2. Data quality assessment
# ============================
print("\n" + "=" * 80)
print("DATA QUALITY ASSESSMENT")
print("=" * 80)

# Missing values analysis
missing_pct = (train.isnull().sum() / len(train) * 100).sort_values(ascending=False)
print("\nVariables with missing values (top 20):")
for col, pct in missing_pct.head(20).items():
    if pct > 0:
        print(f"  {col:40s}: {pct:6.2f}%")

# Variable types
categorical_cols_raw = train.select_dtypes(include=['object']).columns.tolist()
numerical_cols_raw = train.select_dtypes(include=['number']).columns.tolist()
print(f"\nVariable types:")
print(f"  - Categorical: {len(categorical_cols_raw)}")
print(f"  - Numerical: {len(numerical_cols_raw)}")

# ============================
# 3. Prepare features
# ============================
print("\n" + "=" * 80)
print("FEATURE PREPARATION")
print("=" * 80)

# Drop ID, target, and event indicator
cols_to_drop = ['ID', TARGET, EVENT_COL]
X = train.drop(columns=[c for c in cols_to_drop if c in train.columns])
y = train[TARGET]

# Identify column types
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = [c for c in X.columns if c not in categorical_cols]

print(f"\nFeatures used: {len(X.columns)}")
print(f"  - Numerical features: {len(numerical_cols)}")
print(f"    {numerical_cols}")
print(f"  - Categorical features: {len(categorical_cols)}")
print(f"    {categorical_cols}")

# ============================
# 4. Build preprocessing pipeline
# ============================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols)
    ],
    remainder='drop'
)

# ============================
# 5. Train/validation split
# ============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain/Val split:")
print(f"  - Train: {X_train.shape[0]} samples")
print(f"  - Validation: {X_val.shape[0]} samples")

# ============================
# 6. Fit preprocessing and transform
# ============================
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

# Get feature names after preprocessing
feature_names = []
# Numerical feature names
feature_names.extend(numerical_cols)
# Categorical feature names from OneHotEncoder
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
feature_names.extend(cat_feature_names)

print(f"\nTotal features after preprocessing: {len(feature_names)}")
print(f"  - From numerical: {len(numerical_cols)}")
print(f"  - From categorical (one-hot): {len(cat_feature_names)}")

# ============================
# 7. LassoCV - Find optimal alpha
# ============================
print("\n" + "=" * 80)
print("LASSO REGRESSION WITH CROSS-VALIDATION")
print("=" * 80)

# Use LassoCV to automatically find the best alpha
alphas = np.logspace(-4, 2, 100)  # Range of alphas to test

lasso_cv = LassoCV(
    alphas=alphas,
    cv=5,
    random_state=42,
    max_iter=10000,
    n_jobs=-1
)

lasso_cv.fit(X_train_processed, y_train)

print(f"\nOptimal alpha: {lasso_cv.alpha_:.6f}")
print(f"Number of alphas tested: {len(alphas)}")

# ============================
# 8. Evaluate the model
# ============================
y_pred_train = lasso_cv.predict(X_train_processed)
y_pred_val = lasso_cv.predict(X_val_processed)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
train_mae = mean_absolute_error(y_train, y_pred_train)
val_mae = mean_absolute_error(y_val, y_pred_val)
train_r2 = r2_score(y_train, y_pred_train)
val_r2 = r2_score(y_val, y_pred_val)

print(f"\nModel Performance:")
print(f"  {'Metric':<20s} {'Train':>12s} {'Validation':>12s}")
print(f"  {'-'*44}")
print(f"  {'RMSE':<20s} {train_rmse:>12.4f} {val_rmse:>12.4f}")
print(f"  {'MAE':<20s} {train_mae:>12.4f} {val_mae:>12.4f}")
print(f"  {'R²':<20s} {train_r2:>12.4f} {val_r2:>12.4f}")

# ============================
# 9. Coefficient analysis
# ============================
print("\n" + "=" * 80)
print("LASSO COEFFICIENT ANALYSIS")
print("=" * 80)

coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': lasso_cv.coef_,
    'abs_coefficient': np.abs(lasso_cv.coef_)
}).sort_values('abs_coefficient', ascending=False)

# Non-zero coefficients (selected by Lasso)
nonzero_mask = coef_df['abs_coefficient'] > 0
n_selected = nonzero_mask.sum()
n_eliminated = len(coef_df) - n_selected

print(f"\nVariable Selection by Lasso:")
print(f"  - Total features: {len(feature_names)}")
print(f"  - Selected (non-zero coef): {n_selected}")
print(f"  - Eliminated (zero coef): {n_eliminated}")
print(f"  - Selection rate: {n_selected/len(feature_names)*100:.1f}%")

# Top 30 most important features
print(f"\n{'='*80}")
print("TOP 30 MOST IMPORTANT FEATURES (by absolute Lasso coefficient)")
print(f"{'='*80}")
print(f"\n{'Rank':>4s}  {'Feature':<50s} {'Coefficient':>12s} {'|Coeff|':>12s} {'Direction':>10s}")
print(f"{'-'*90}")

top_features = coef_df[nonzero_mask].head(30)
for i, (_, row) in enumerate(top_features.iterrows(), 1):
    direction = "↑ risk" if row['coefficient'] > 0 else "↓ risk"
    print(f"{i:>4d}  {row['feature']:<50s} {row['coefficient']:>12.6f} {row['abs_coefficient']:>12.6f} {direction:>10s}")

# ============================
# 10. Map back to original variables
# ============================
print(f"\n{'='*80}")
print("ORIGINAL VARIABLE IMPORTANCE (aggregated from one-hot encoded features)")
print(f"{'='*80}")

# For one-hot encoded features, aggregate back to original variable
original_var_importance = {}
for _, row in coef_df.iterrows():
    feature = row['feature']
    abs_coef = row['abs_coefficient']
    
    # Check if this is a one-hot encoded feature (contains underscore from encoding)
    matched = False
    for cat_col in categorical_cols:
        if feature.startswith(cat_col + '_'):
            if cat_col not in original_var_importance:
                original_var_importance[cat_col] = {
                    'max_abs_coef': 0, 
                    'sum_abs_coef': 0, 
                    'n_categories': 0,
                    'n_nonzero': 0,
                    'type': 'Categorical'
                }
            original_var_importance[cat_col]['max_abs_coef'] = max(
                original_var_importance[cat_col]['max_abs_coef'], abs_coef
            )
            original_var_importance[cat_col]['sum_abs_coef'] += abs_coef
            original_var_importance[cat_col]['n_categories'] += 1
            if abs_coef > 0:
                original_var_importance[cat_col]['n_nonzero'] += 1
            matched = True
            break
    
    if not matched:
        # This is a numerical feature
        original_var_importance[feature] = {
            'max_abs_coef': abs_coef,
            'sum_abs_coef': abs_coef,
            'n_categories': 1,
            'n_nonzero': 1 if abs_coef > 0 else 0,
            'type': 'Numerical'
        }

# Sort by max absolute coefficient
orig_importance_df = pd.DataFrame.from_dict(original_var_importance, orient='index')
orig_importance_df.index.name = 'variable'
orig_importance_df = orig_importance_df.sort_values('max_abs_coef', ascending=False)

print(f"\n{'Rank':>4s}  {'Variable':<35s} {'Type':<12s} {'Max|Coef|':>10s} {'Sum|Coef|':>10s} {'#Cat':>5s} {'#NonZero':>8s}")
print(f"{'-'*88}")

for i, (var, row) in enumerate(orig_importance_df.iterrows(), 1):
    print(f"{i:>4d}  {var:<35s} {row['type']:<12s} {row['max_abs_coef']:>10.6f} {row['sum_abs_coef']:>10.6f} {int(row['n_categories']):>5d} {int(row['n_nonzero']):>8d}")
    if i >= 40:
        break

# ============================
# 11. Final Summary for Feature Engineering
# ============================
print(f"\n{'='*80}")
print("SUMMARY: KEY VARIABLES FOR FEATURE ENGINEERING")
print(f"{'='*80}")

# Define thresholds
high_importance = orig_importance_df[orig_importance_df['max_abs_coef'] > 0.5]
medium_importance = orig_importance_df[
    (orig_importance_df['max_abs_coef'] > 0.1) & 
    (orig_importance_df['max_abs_coef'] <= 0.5)
]
low_importance = orig_importance_df[
    (orig_importance_df['max_abs_coef'] > 0) & 
    (orig_importance_df['max_abs_coef'] <= 0.1)
]
zero_importance = orig_importance_df[orig_importance_df['max_abs_coef'] == 0]

print(f"\n🔴 HIGH IMPORTANCE (|coef| > 0.5): {len(high_importance)} variables")
for var, row in high_importance.iterrows():
    print(f"   - {var} ({row['type']}, max |coef| = {row['max_abs_coef']:.4f})")

print(f"\n🟡 MEDIUM IMPORTANCE (0.1 < |coef| ≤ 0.5): {len(medium_importance)} variables")
for var, row in medium_importance.iterrows():
    print(f"   - {var} ({row['type']}, max |coef| = {row['max_abs_coef']:.4f})")

print(f"\n🟢 LOW IMPORTANCE (0 < |coef| ≤ 0.1): {len(low_importance)} variables")
for var, row in low_importance.iterrows():
    print(f"   - {var} ({row['type']}, max |coef| = {row['max_abs_coef']:.4f})")

print(f"\n⚪ ELIMINATED by Lasso (|coef| = 0): {len(zero_importance)} variables")
for var, row in zero_importance.iterrows():
    print(f"   - {var} ({row['type']})")

# ============================
# 12. Correlation analysis (numerical only)
# ============================
print(f"\n{'='*80}")
print("HIGHLY CORRELATED PAIRS (|r| > 0.7, numerical features only)")
print(f"{'='*80}")

# Compute correlation on processed numerical features
num_data = pd.DataFrame(
    preprocessor.named_transformers_['num'].transform(X_train[numerical_cols]),
    columns=numerical_cols
)
corr_matrix = num_data.corr()

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(numerical_cols)):
    for j in range(i+1, len(numerical_cols)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.7:
            high_corr_pairs.append((numerical_cols[i], numerical_cols[j], r))

high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

print(f"\n{'Variable 1':<30s} {'Variable 2':<30s} {'Correlation':>12s}")
print(f"{'-'*74}")
for v1, v2, r in high_corr_pairs:
    print(f"{v1:<30s} {v2:<30s} {r:>12.4f}")

if not high_corr_pairs:
    print("  No highly correlated pairs found.")

# ============================
# 13. Save results
# ============================
results = {
    'optimal_alpha': float(lasso_cv.alpha_),
    'train_rmse': float(train_rmse),
    'val_rmse': float(val_rmse),
    'train_r2': float(train_r2),
    'val_r2': float(val_r2),
    'n_features_total': len(feature_names),
    'n_features_selected': int(n_selected),
    'n_features_eliminated': int(n_eliminated),
    'top_features': [
        {
            'variable': var, 
            'type': row['type'],
            'max_abs_coef': float(row['max_abs_coef']),
            'sum_abs_coef': float(row['sum_abs_coef'])
        }
        for var, row in orig_importance_df.head(30).iterrows()
    ],
    'eliminated_features': [
        var for var, row in zero_importance.iterrows()
    ],
    'high_corr_pairs': [
        {'var1': v1, 'var2': v2, 'correlation': float(r)}
        for v1, v2, r in high_corr_pairs
    ]
}

results_path = Path(__file__).parent / 'lasso_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n\nResults saved to: {results_path}")
print("\nDone!")
