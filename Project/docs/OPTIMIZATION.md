1. VARIABLES TO ELIMINATE (DROP)
These variables must be removed from your training dataset to avoid data leakage, severe multicollinearity (repeated variables that confuse the algorithm), or because the information is already captured in another column.

Data Leakage:

ID, efs, efs_time: Must be strictly removed from the input variables (X). Using the outcome to predict the outcome ruins the model.

Severe Genetic Redundancy (Correlation > 0.90):

hla_high_res_6, hla_low_res_10, hla_low_res_6, hla_match_a_high: The Lasso analysis showed that these variables are almost identical to other higher-resolution ones. Medically, compatibility at 8 alleles (hla_high_res_8) is the gold standard, so these low-resolution columns only generate statistical noise and should be deleted.

Redundant Sub-comorbidities:

renal_issue: The Lasso model gave it a coefficient of 0. This is because renal issues are already evaluated and summed within the general variable comorbidity_score (HCT-CI).

Absorbed variables:

donor_age: Lasso eliminated it because its biological impact is already being captured by the patient's age (age_at_hct) and the type of donor (donor_related).

2. VARIABLES TO KEEP (KEEP - Top Predictors)
These variables obtained the highest importance in your Lasso model and have undisputed biological and demographic weight. Do not alter them, but ensure they are coded correctly (e.g., using ordinal numbers for severity levels).

Demographic and Systemic: ethnicity (the number 1 variable, strongly linked to social determinants of health and access to optimal donors), year_hct, and race_group.

Cancer Characteristics: prior_tumor (history of solid tumors), prim_disease_hct (type of leukemia/disease), dri_score (Disease Risk Index), and cyto_score.

Patient Physical Status: cardiac, diabetes, pulm_severe, vent_hist, and comorbidity_score.

Treatment and Compatibility: conditioning_intensity (chemotherapy intensity), cmv_status, gvhd_proph (prophylaxis), sex_match, and in_vivo_tcd.

3. VARIABLES TO COMBINE (FEATURE ENGINEERING)
This is where you will gain precision. You must create these new columns by mixing existing information, since algorithms work better when given pre-calculated "clinical context".

A) Total Genetic Load (hla_composite):

How to create it: Calculate the mean (or sum of mismatches) of the variables hla_high_res_8, hla_high_res_10, and hla_nmdp_6.

Medical reason: The human immune system and T-cell receptors (TCR) do not evaluate gene by gene; they react to the "total load" of foreign proteins or differences in peptide binding, which triggers rejection. Summarizing HLA into a single quality score reduces 20 noisy variables to a single powerful metric.

B) Physical Shock (age_x_comorbidity):

How to create it: Multiply age_at_hct * comorbidity_score.

Medical reason: Studies show that a comorbidity in a 20-year-old patient is tolerated well, but when the patient's age exceeds 40 or 50, the same comorbidity exponentially spikes non-relapse mortality (NRM).

C) Treatment Balance (dri_x_conditioning):

How to create it: Multiply disease risk (dri_score coded from 0 to 3) by treatment intensity (conditioning_intensity).

Medical reason: There is a proven clinical interaction between these two metrics. A highly aggressive cancer (High DRI) requires lethal conditioning (MAC). The model needs this interaction to understand if the toxicity risk is worth taking to prevent relapse.

D) Vulnerability/Access Indicator (missing_count):

How to create it: Sum the amount of null or "Not done" values (empty cells) in each patient's row.

Medical reason: The data creators confirmed that the missing values pattern reflects real-life situations. A patient missing prior exams usually is a strong indicator of poor healthcare access or hospital urgency, which impacts survival.

E) Metabolic-Mental Risk (n_comorbidities_risk):

How to create it: Create a binary flag (1 or 0) if the patient has high-risk combinations like obesity/diabetes plus psychiatric issues.

Medical reason: Diabetes and obesity cause a chronic inflammatory state that modulates the immune response and worsens Graft-versus-Host Disease (GVHD). If psychiatric disorders are added to this, the risk of non-compliance with the complex recovery treatment skyrockets.

Next step: Once you apply this cleaning filter and create the new columns, your dataset will be ready to be processed by a decision tree model (Gradient Boosting) evaluated with the stratified concordance index.
