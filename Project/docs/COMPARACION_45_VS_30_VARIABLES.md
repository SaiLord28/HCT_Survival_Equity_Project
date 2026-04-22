# Comparison 60 vs 45 vs 30 and Feature Engineering Findings

Execution Date: 2026-04-21

## Executive Summary

- Comparative tests of the pipeline were run with `n_features` at 60, 45, and 30.
- The 60 -> 45 reduction maintains very similar performance.
- The 45 -> 30 reduction shows a small drop in AUC and the equity metric, but it still passes fairness.
- Critical finding: the 30 scenario does not select 30 actual features in the current implementation.

## Methodology

- Dataset: `Project/ai_service/data/raw/train.csv`.
- Evaluated pipeline: `Project/ai_service/src/pipeline.py` (full training M1-M7).
- Configuration: `model_type='gbm'` and `n_features` at 60, 45, 30.
- Metrics analyzed: CV AUC, CV Accuracy, Training AUC, Stratified C-index, C-index disparity, fairness_passed, and stability.

## Comparison Results

| requested n_features | selected n_features | CV AUC | CV Accuracy | Train AUC | Stratified C-index | C-index disparity | Fairness | Stability |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 60 | 60 | 0.74196 | 0.67965 | 0.81015 | 0.81004 | 0.01598 | True | High |
| 45 | 45 | 0.74148 | 0.67931 | 0.80905 | 0.80880 | 0.01598 | True | Medium |
| 30 | 44 | 0.74004 | 0.67931 | 0.80896 | 0.80867 | 0.01965 | True | Medium |

## Interpretation

1. 60 vs 45
- Minimal degradation in AUC and accuracy.
- Equity is practically the same in terms of disparity.

2. 45 vs 30
- Moderate drop in CV AUC.
- Slight increase in disparity (worse than 45), although it remains in `fairness_passed=True` state.

3. Practical effect
- The current strategy justifies the cut to 45 with low cost.
- For 30, before concluding, the selector logic must be corrected because today it does not apply a real cap of 30.

## Key finding: 30 mode is not selecting 30

In `Project/ai_service/src/m3_features.py`, the `select_features` function starts with a forced variables block:

- Forced comorbidities
- `comorbidity_score`
- Additional clinical variables
- Priority clinical variables

With the current dataset, that forced block sums up to 43 variables. Therefore, when requesting 30, the selector ends up with 44 (the forced floor already exceeds the target).

Implication:
- The `n_features` parameter does not work as a strict limit when `forced_features > n_features`.

## Feature engineering already implemented

In `Project/ai_service/src/m1_preprocessing.py` the following features are created:

1. `age_donor_diff`
- Difference between the patient's age and the donor's age.

2. `high_risk_comorbidity`
- Indicator of severe high comorbidity.

3. `hla_match_quality`
- Composite HLA matching score.

Additionally, the following are applied:
- Equity-aware imputation by `race_group`.
- Scaling of numerical variables.
- Categorical encoding with handling of unseen categories.

## Technical finding in feature engineering

`high_risk_comorbidity` is remaining constant (without variance) in these tests.

Possible cause in the current implementation:
- The logic checks for numerical conditions in columns that originally may come as categorical such as `Y/N`.
- Result: the flag does not trigger as expected and loses predictive power.

## Project automated tests

An attempt was made to run `Project/ai_service/test_pipeline.py` and it failed because of:

- `IndentationError: unexpected indent` on line 1.

Implication:
- The integrated test script needs correction to be useful as continuous automated verification.

## Prioritized Recommendations

1. Fix `high_risk_comorbidity`
- Explicitly detect affirmative values (`Y`, `Yes`, `1`, `true`) before or after encoding consistently.

2. Make `n_features` a real cap
- Separate `must_have` (mandatory clinical minimum) from `should_have` (ranking).
- If 30 is requested, select exactly 30 or clearly report that it is not possible due to clinical constraints.

3. Add missingness indicators
- Especially for variables with high missing rates (e.g. `tce_match`, `mrd_hct`, `tce_imm_match`, `cyto_score_detail`).

4. Create clinical interactions backed by domain knowledge
- Examples: `age_at_hct x comorbidity_score`, `dri_score x hla_match_quality`, `karnofsky_score x conditioning_intensity`.

## Suggested Decisions

1. Keep 45 as the immediate operational baseline.
2. Fix selector and feature engineering.
3. Repeat benchmark with an actual 30 and compare again against 45 and 60.
4. Adopt 30 only if the loss of performance and equity remains within the agreed clinical threshold.
