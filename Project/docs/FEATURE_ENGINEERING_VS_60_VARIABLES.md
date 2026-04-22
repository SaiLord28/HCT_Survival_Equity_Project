# Comparison of Feature Engineering vs 60-Variable Model

Execution Date: 2026-04-21
Model: GBM
Dataset: /app/data/raw/raw/train.csv

## 1) Variables used by Feature Engineering (M3)

Selected count: 45

Full selected list:
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

### Relevant subgroups within the 45
- Included comorbidities: 14 forced + comorbidity_score
- Priority clinical variables included: 16
- Sensitive variable included in this run: race_group

## 2) Files where this logic is defined and used

- Project/ai_service/src/m3_features.py
  - Defines the variable selection logic (clinical, forced comorbidities, statistical + ML combination, and availability disparity filter).
- Project/ai_service/src/m1_preprocessing.py
  - Defines preprocessed input variables and created features (e.g., age_donor_diff, hla_match_quality).
- Project/ai_service/src/pipeline.py
  - Executes M3 selection within the training flow and exposes the comparison between the all-variable model and the feature engineering model.
- Project/ai_service/api.py
  - Exposes the `/train` endpoint with `compare_with_all_features` to execute the comparison in a single run.
- Project/ai_service/data/raw/data_dictionary.csv
  - Base variable dictionary of the dataset.

## 3) Actual comparison against the 60-variable group

### Results obtained

| Configuration | N features | CV Accuracy | CV AUC |
|---|---:|---:|---:|
| All preprocessed variables | 60 | 0.6798958333 | 0.7417819508 |
| Feature engineering (M3) | 45 | 0.6793055556 | 0.7414772292 |

Deltas (45 - 60):
- Delta CV AUC: -0.0003047216
- Delta CV Accuracy: -0.0005902778
- performance_maintained: true (0.01 tolerance)

### How much it really improves versus the 60 group

There is no improvement in the central predictive metric; there is a minimal reduction:
- CV AUC drops ~0.041% relative
- CV Accuracy drops ~0.087% relative

However, there is an improvement in efficiency and clinical control:
- Dimensionality reduction: 60 -> 45 (25% fewer variables)
- Performance is maintained practically the same within tolerance
- Inclusion of critical clinical variables and comorbidities is guaranteed

## 4) Executive conclusion

Feature Engineering (M3) does not increase AUC/Accuracy compared to the 60-variable set, but it retains performance with marginal loss and offers a more compact and clinically controlled model. For this project, the main benefit is clinical robustness + model simplification, not a numerical AUC gain.
