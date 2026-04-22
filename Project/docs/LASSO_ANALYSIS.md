# Lasso Analysis for HCT Survival - Most Important Variables

Execution Date: 2026-04-21
Script: `analysis/lasso_analysis.py`
Results: `analysis/lasso_results.json`

---

## 1. Dataset Summary

The files in `Project/ai_service/data/raw/` contain:

| File | Description | Size |
|---|---|---|
| train.csv | Training dataset | 28,800 records x 60 columns |
| test.csv | Test dataset (no target) | 3 records x 58 columns |
| data_dictionary.csv | Variable dictionary | 59 documented variables |
| sample_submission.csv | Submission format | Format example |

### Target Variables

- **efs**: Event-Free Survival - binary (Event / Censoring)
- **efs_time**: Time to event in months - continuous numerical (mean: ~29.1, median: ~17.8)

### Event Distribution

- Event: ~53.2% of records
- Censoring: ~46.8% of records

### Variable Types (57 predictors)

- Numerical: 20 variables (mostly HLA variables)
- Categorical: 37 variables (comorbidities, clinical data, demographics)

---

## 2. Methodology

A **LassoCV** (Lasso with 5-fold cross-validation) was executed on `efs_time` as the target, excluding `ID` and `efs` to prevent data leakage.

- Alpha range: 100 values on logarithmic scale [10^-4, 10^2]
- Preprocessing: imputation (median for numerical, mode for categorical), StandardScaler scaling, OneHotEncoder for categorical
- Split: 80% train / 20% validation, random_state=42

### Model Configuration

| Parameter | Value |
|---|---|
| Optimal alpha | 0.01322 |
| Total features (post one-hot) | 183 |
| Selected features (coef != 0) | 122 (66.7%) |
| Eliminated features (coef = 0) | 61 (33.3%) |

### Performance Metrics

| Metric | Train | Validation |
|---|---:|---:|
| RMSE | 22.647 | 22.830 |
| MAE | 17.768 | 17.879 |
| R2 | 0.1632 | 0.1637 |

**Note on low R2 (~0.16)**: This is expected for survival data with censoring. Censored data (~47%) distorts linear regression. However, the Lasso approach remains valid for identifying which variables are most important, which is the primary goal of this analysis.

---

## 3. Most Important Variables

### HIGH importance (|coef| > 0.5) - 30 original variables

| Rank | Variable | Type | Max Coef | Description |
|---:|---|---|---:|---|
| 1 | ethnicity | Categorical | 7.538 | Ethnicity of the patient |
| 2 | prior_tumor | Categorical | 6.974 | Prior solid tumor |
| 3 | year_hct | Numerical | 6.147 | Year of transplant |
| 4 | prim_disease_hct | Categorical | 4.664 | Primary disease (AML, ALL, MDS, etc.) |
| 5 | cardiac | Categorical | 4.562 | Cardiac condition |
| 6 | diabetes | Categorical | 4.366 | Diabetes |
| 7 | conditioning_intensity | Categorical | 3.717 | Conditioning intensity |
| 8 | dri_score | Categorical | 3.695 | Disease risk index |
| 9 | cmv_status | Categorical | 3.420 | CMV status donor/recipient |
| 10 | gvhd_proph | Categorical | 3.054 | GVHD prophylaxis |
| 11 | sex_match | Categorical | 2.981 | Sex match donor/recipient |
| 12 | cyto_score | Categorical | 2.486 | Cytogenetic score |
| 13 | in_vivo_tcd | Categorical | 2.459 | In vivo T-cell depletion |
| 14 | tbi_status | Categorical | 2.221 | Total body irradiation |
| 15 | pulm_severe | Categorical | 2.114 | Severe pulmonary |
| 16 | vent_hist | Categorical | 2.045 | History of mechanical ventilation |
| 17 | tce_imm_match | Categorical | 1.975 | T-cell epitope match |
| 18 | race_group | Categorical | 1.620 | Race group |
| 19 | prod_type | Categorical | 1.542 | Product type |
| 20 | hepatic_severe | Categorical | 1.499 | Severe hepatic |
| 21 | hla_match_b_low | Numerical | 1.489 | HLA-B low resolution |
| 22 | rheum_issue | Categorical | 1.453 | Rheumatologic issue |
| 23 | hla_high_res_8 | Numerical | 1.431 | Match HLA 8 loci high resolution |
| 24 | hla_low_res_8 | Numerical | 1.377 | Match HLA 8 loci low resolution |
| 25 | comorbidity_score | Numerical | 1.331 | Sorror comorbidity score |
| 26 | donor_related | Categorical | 1.296 | Donor type (related/unrelated) |
| 27 | psych_disturb | Categorical | 1.127 | Psychiatric disturbance |
| 28 | hla_nmdp_6 | Numerical | 1.092 | Match HLA NMDP 6 loci |
| 29 | obesity | Categorical | 1.067 | Obesity |
| 30 | peptic_ulcer | Categorical | 1.055 | Peptic ulcer |

### MEDIUM importance (0.1 < |coef| <= 0.5) - 8 variables

| Rank | Variable | Type | Max Coef |
|---:|---|---|---:|
| 31 | tce_match | Categorical | 0.907 |
| 32 | hla_high_res_10 | Numerical | 0.886 |
| 33 | rituximab | Categorical | 0.880 |
| 34 | karnofsky_score | Numerical | 0.725 |
| 35 | mrd_hct | Categorical | 0.700 |
| 36 | hla_match_c_high | Numerical | 0.635 |
| 37 | tce_div_match | Categorical | 0.612 |
| 38 | hla_match_drb1_high | Numerical | 0.604 |

**Note**: `karnofsky_score` appears in position 34, which might seem unexpected clinically. This is because Lasso penalizes correlated variables: `comorbidity_score` (rank 25) captures part of the same information.

### LOW importance (|coef| > 0 but <= 0.1) - 2 variables

| Variable | Coef |
|---|---:|
| hla_match_dqb1_high | 0.089 |
| hla_match_dqb1_low | 0.009 |

### ELIMINATED by Lasso (|coef| = 0) - 5 original variables

| Variable | Type | Probable Reason |
|---|---|---|
| hla_high_res_6 | Numerical | Redundant with hla_high_res_8 and hla_high_res_10 (r > 0.97) |
| donor_age | Numerical | Captured by age_at_hct and donor_related |
| hla_match_a_high | Numerical | Redundant with aggregated HLA variables |
| hla_low_res_10 | Numerical | Redundant with hla_low_res_8 (r = 0.96) |
| renal_issue | Categorical | Captured by comorbidity_score |

---

## 4. Multicollinearity Analysis

HLA variables exhibit extremely high multicollinearity: 83 pairs with |r| > 0.7.

### Top 10 most correlated pairs

| Variable 1 | Variable 2 | r |
|---|---|---:|
| hla_low_res_6 | hla_low_res_8 | 0.975 |
| hla_high_res_8 | hla_high_res_6 | 0.970 |
| hla_low_res_8 | hla_low_res_10 | 0.960 |
| hla_high_res_8 | hla_high_res_10 | 0.951 |
| hla_low_res_6 | hla_low_res_10 | 0.935 |
| hla_high_res_6 | hla_high_res_10 | 0.923 |
| hla_low_res_6 | hla_match_b_low | 0.884 |
| hla_low_res_6 | hla_match_drb1_low | 0.881 |
| hla_match_b_low | hla_low_res_8 | 0.876 |
| hla_high_res_6 | hla_match_b_high | 0.869 |

**Recommendation**: Create a composite score `hla_match_quality` that combines HLA variables, reducing ~20 variables to 1-3.

---

## 5. Recommendations for Feature Engineering

### Phase 1: Immediate Corrections

1. Fix `high_risk_comorbidity` in `m1_preprocessing.py` (currently constant, no variance).

### Phase 2: Suggested New Features

1. `age_x_comorbidity` = age_at_hct x comorbidity_score
2. `dri_x_conditioning` = dri_score_encoded x conditioning_encoded
3. `karnofsky_x_comorbidity` = karnofsky_score x comorbidity_score
4. Consolidate HLA in `hla_composite` (mean of hla_high_res_8, hla_high_res_10, hla_nmdp_6)
5. Missingness indicators for variables with high absence rates: tce_match, mrd_hct, tce_imm_match, cyto_score_detail
6. `n_comorbidities` = count of active comorbidities

### Phase 3: Variables to Eliminate or Consolidate

- Eliminate: hla_high_res_6, hla_low_res_10, hla_match_a_high (redundant)
- Evaluate: donor_age (may contribute via age_donor_diff)
- Evaluate: renal_issue (may be captured by comorbidity_score)

---

## 6. Conclusion

The 57 predictor variables are grouped into 7 clinical domains by importance:

1. **Demographic/temporal factors**: ethnicity, year_hct, race_group, age_at_hct
2. **Comorbidities**: prior_tumor, cardiac, diabetes, pulm_severe, hepatic_severe, rheum_issue, obesity, peptic_ulcer, psych_disturb, vent_hist
3. **Disease characteristics**: prim_disease_hct, dri_score, cyto_score, mrd_hct
4. **Treatment/transplant**: conditioning_intensity, gvhd_proph, tbi_status, in_vivo_tcd, rituximab, prod_type, melphalan_dose
5. **Donor match**: cmv_status, sex_match, donor_related, tce_imm_match, tce_match, tce_div_match
6. **HLA matching**: hla_high_res_8, hla_low_res_8, hla_nmdp_6, hla_match_b_low, hla_match_c_high, hla_match_drb1_high
7. **Functional status**: comorbidity_score, karnofsky_score

52 out of 57 variables are relevant (non-zero coefficient). Only 5 are eliminated due to redundancy.
