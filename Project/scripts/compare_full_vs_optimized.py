#!/usr/bin/env python3
"""
Compare two HCT survival models:
1) Full model: uses all available predictors (with one-hot encoding).
2) Optimized model: uses the curated 18-feature "elite command" set.

This script trains both models with the same split and prints a side-by-side
comparison including stratified C-index by race group.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42


@dataclass
class ModelSummary:
    model_name: str
    n_features: int
    cv_auc_mean: float
    cv_auc_std: float
    test_auc: float
    test_accuracy: float
    test_f1: float
    stratified_c_index: float
    c_index_disparity: float
    fairness_passed: bool
    group_c_indices: Dict[str, float]


def _to_binary_yes(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("yes").astype(int)


def _safe_numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def build_optimized_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 18 optimized predictors exactly as defined in the request."""
    out = pd.DataFrame(index=df.index)

    # 1-3: direct numeric survivors
    out["year_hct"] = _safe_numeric(df, "year_hct")
    out["comorbidity_score"] = _safe_numeric(df, "comorbidity_score")
    out["hla_match_b_low"] = _safe_numeric(df, "hla_match_b_low")

    # 4: KPS_Under_80
    out["KPS_Under_80"] = (_safe_numeric(df, "karnofsky_score") < 80).astype(int)

    # 5: Organ_Failure_Index = cardiac + pulm_severe + hepatic_severe + renal_issue
    organ_cols = ["cardiac", "pulm_severe", "hepatic_severe", "renal_issue"]
    organ_bin = []
    for col in organ_cols:
        if col in df.columns:
            organ_bin.append(_to_binary_yes(df[col]))
        else:
            organ_bin.append(pd.Series(0, index=df.index, dtype="int64"))
    out["Organ_Failure_Index"] = sum(organ_bin)

    # 6: Total_HLA_High_Match
    hla_high_cols = [
        "hla_match_a_high",
        "hla_match_b_high",
        "hla_match_c_high",
        "hla_match_drb1_high",
        "hla_match_dqb1_high",
    ]
    out["Total_HLA_High_Match"] = sum(_safe_numeric(df, c) for c in hla_high_cols)

    # 7: Comorb_x_OrganFail
    out["Comorb_x_OrganFail"] = out["comorbidity_score"] * out["Organ_Failure_Index"]

    # 8-18: critical binaries extracted from categories
    out["cond_RIC"] = df["conditioning_intensity"].astype(str).eq("RIC").astype(int)
    out["sex_FM"] = df["sex_match"].astype(str).eq("F-M").astype(int)
    out["sex_MF"] = df["sex_match"].astype(str).eq("M-F").astype(int)
    out["dri_Intermediate"] = df["dri_score"].astype(str).eq("Intermediate").astype(int)
    out["dri_Pediatric"] = df["dri_score"].astype(str).eq("N/A - pediatric").astype(int)
    out["disease_IIS"] = df["prim_disease_hct"].astype(str).eq("IIS").astype(int)
    out["disease_IEA"] = df["prim_disease_hct"].astype(str).eq("IEA").astype(int)
    out["cmv_plus_minus"] = df["cmv_status"].astype(str).eq("+/-").astype(int)
    out["gvhd_FK_MMF"] = df["gvhd_proph"].astype(str).eq("FK+ MMF +- others").astype(int)
    out["in_vivo_tcd_Yes"] = df["in_vivo_tcd"].astype(str).eq("Yes").astype(int)
    out["tbi_TBI_Cy"] = df["tbi_status"].astype(str).eq("TBI + Cy +- Other").astype(int)

    return out


def stratified_c_index(y_true: np.ndarray, y_proba: np.ndarray, groups: np.ndarray) -> Dict[str, object]:
    """Compute group-wise AUC proxy and weighted stratified C-index."""
    group_c_indices: Dict[str, float] = {}
    group_sizes: Dict[str, int] = {}

    unique_groups = pd.Series(groups).astype(str).fillna("Unknown").unique()
    total_samples = len(y_true)

    weighted_sum = 0.0
    for group in unique_groups:
        mask = pd.Series(groups).astype(str).fillna("Unknown").eq(group).values
        group_sizes[group] = int(mask.sum())
        if mask.sum() < 20:
            continue

        try:
            c_idx = float(roc_auc_score(y_true[mask], y_proba[mask]))
            group_c_indices[group] = c_idx
            weighted_sum += c_idx * (mask.sum() / total_samples)
        except ValueError:
            continue

    disparity = float(max(group_c_indices.values()) - min(group_c_indices.values())) if group_c_indices else 0.0

    return {
        "stratified_c_index": float(weighted_sum),
        "group_c_indices": group_c_indices,
        "c_index_disparity": disparity,
        "fairness_passed": disparity < 0.10,
    }


def build_full_model_pipeline(df: pd.DataFrame) -> Tuple[Pipeline, List[str]]:
    """Build preprocessing + regularized logistic pipeline for all predictors."""
    feature_cols = [c for c in df.columns if c not in {"ID", "efs", "efs_time"}]
    X = df[feature_cols]

    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    model = SGDClassifier(
        random_state=RANDOM_STATE,
        loss="log_loss",
        penalty="l2",
        max_iter=1000,
        tol=1e-3,
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)], memory=None)

    return pipeline, feature_cols


def build_optimized_model_pipeline() -> Pipeline:
    """Build simple median-impute + regularized logistic pipeline."""
    preprocessor = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    model = SGDClassifier(
        random_state=RANDOM_STATE,
        loss="log_loss",
        penalty="l2",
        max_iter=1000,
        tol=1e-3,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)], memory=None)


def evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    groups_test: np.ndarray,
    cv_folds: int,
) -> ModelSummary:
    if cv_folds and cv_folds > 1:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        cv_auc = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=1)
        cv_auc_mean = float(np.mean(cv_auc))
        cv_auc_std = float(np.std(cv_auc))
    else:
        cv_auc_mean = float("nan")
        cv_auc_std = float("nan")

    pipeline.fit(X_train, y_train)

    # Count effective feature dimensionality after preprocessing (e.g., one-hot expansion).
    try:
        n_effective_features = int(pipeline.named_steps["preprocessor"].transform(X_train).shape[1])
    except Exception:
        n_effective_features = int(X.shape[1])

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    fairness = stratified_c_index(y_test.values, y_proba, groups_test)

    return ModelSummary(
        model_name=model_name,
        n_features=n_effective_features,
        cv_auc_mean=cv_auc_mean,
        cv_auc_std=cv_auc_std,
        test_auc=float(roc_auc_score(y_test, y_proba)),
        test_accuracy=float(accuracy_score(y_test, y_pred)),
        test_f1=float(f1_score(y_test, y_pred, zero_division=0)),
        stratified_c_index=float(fairness["stratified_c_index"]),
        c_index_disparity=float(fairness["c_index_disparity"]),
        fairness_passed=bool(fairness["fairness_passed"]),
        group_c_indices=fairness["group_c_indices"],
    )


def run_comparison(
    data_path: Path,
    output_json: Path | None = None,
    cv_folds: int = 3,
) -> Dict[str, object]:
    df = pd.read_csv(data_path, encoding="utf-8", encoding_errors="replace")

    if "efs" not in df.columns:
        raise ValueError("Dataset must include target column 'efs'.")
    if "race_group" not in df.columns:
        raise ValueError("Dataset must include 'race_group' for stratified C-index.")

    y = pd.to_numeric(df["efs"], errors="coerce").fillna(0).astype(int)

    # Full dataset model
    full_pipeline, full_feature_cols = build_full_model_pipeline(df)
    x_full = df[full_feature_cols]

    # Optimized 18-feature model
    x_opt = build_optimized_features(df)
    opt_pipeline = build_optimized_model_pipeline()

    # Shared split for fair comparison
    split_idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        split_idx,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    y_train = y.iloc[idx_train]
    y_test = y.iloc[idx_test]
    groups_test = df.iloc[idx_test]["race_group"].astype(str).fillna("Unknown").values

    full_summary = evaluate_model(
        model_name="full_all_features",
        pipeline=full_pipeline,
        X=x_full,
        X_train=x_full.iloc[idx_train],
        X_test=x_full.iloc[idx_test],
        y_train=y_train,
        y_test=y_test,
        groups_test=groups_test,
        cv_folds=cv_folds,
    )

    opt_summary = evaluate_model(
        model_name="optimized_18_features",
        pipeline=opt_pipeline,
        X=x_opt,
        X_train=x_opt.iloc[idx_train],
        X_test=x_opt.iloc[idx_test],
        y_train=y_train,
        y_test=y_test,
        groups_test=groups_test,
        cv_folds=cv_folds,
    )

    comparison = {
        "data_path": str(data_path),
        "n_rows": int(len(df)),
        "target_event_rate": float(y.mean()),
        "optimized_feature_names": x_opt.columns.tolist(),
        "models": [asdict(full_summary), asdict(opt_summary)],
        "delta_optimized_minus_full": {
            "cv_auc_mean": opt_summary.cv_auc_mean - full_summary.cv_auc_mean,
            "test_auc": opt_summary.test_auc - full_summary.test_auc,
            "test_accuracy": opt_summary.test_accuracy - full_summary.test_accuracy,
            "test_f1": opt_summary.test_f1 - full_summary.test_f1,
            "stratified_c_index": opt_summary.stratified_c_index - full_summary.stratified_c_index,
            "c_index_disparity": opt_summary.c_index_disparity - full_summary.c_index_disparity,
        },
    }

    print("\n" + "=" * 78)
    print("HCT MODEL COMPARISON: FULL vs OPTIMIZED-18")
    print("=" * 78)
    print(f"Rows: {comparison['n_rows']}")
    print(f"Event rate (efs=1): {comparison['target_event_rate']:.4f}")
    print(f"Estimator: SGDClassifier(log_loss, L2), CV folds: {cv_folds}")

    for model in comparison["models"]:
        print("\n" + "-" * 78)
        print(f"Model: {model['model_name']}")
        print(f"Features: {model['n_features']}")
        print(f"CV AUC: {model['cv_auc_mean']:.4f} +/- {model['cv_auc_std']:.4f}")
        print(f"Test AUC: {model['test_auc']:.4f}")
        print(f"Test Accuracy: {model['test_accuracy']:.4f}")
        print(f"Test F1: {model['test_f1']:.4f}")
        print(f"Stratified C-index: {model['stratified_c_index']:.4f}")
        print(f"C-index disparity: {model['c_index_disparity']:.4f}")
        print(f"Fairness passed (<0.10 disparity): {model['fairness_passed']}")

    delta = comparison["delta_optimized_minus_full"]
    print("\n" + "-" * 78)
    print("Delta (optimized - full):")
    print(f"CV AUC mean: {delta['cv_auc_mean']:+.4f}")
    print(f"Test AUC: {delta['test_auc']:+.4f}")
    print(f"Test Accuracy: {delta['test_accuracy']:+.4f}")
    print(f"Test F1: {delta['test_f1']:+.4f}")
    print(f"Stratified C-index: {delta['stratified_c_index']:+.4f}")
    print(f"C-index disparity: {delta['c_index_disparity']:+.4f}")

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(f"\nSaved report to: {output_json}")

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full vs optimized HCT survival models")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("../ai_service/data/raw/train.csv"),
        help="Path to train.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("comparison_results/full_vs_optimized_report.json"),
        help="Where to save comparison JSON report",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of CV folds for both models (default: 3)",
    )
    args = parser.parse_args()

    run_comparison(data_path=args.data_path, output_json=args.output_json, cv_folds=args.cv_folds)


if __name__ == "__main__":
    main()
