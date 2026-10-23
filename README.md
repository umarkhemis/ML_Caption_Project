# Predicting Telecom Customer Churn

BCS 3101 Capstone - 
Ahmed Umar Khemis - Reg. No. 2024/A/KCS/3970/F
Kimuli Edrine - 
Benard
Clever
John Rockey
Naome


Binary classification: will a telecom customer cancel their subscription?
Dataset: Telco Customer Churn (IBM sample data), 7,043 customers, 21 columns.

This project looks at a telecom company's customer data and tries to answer a simple but
important question: which customers are about to cancel their subscription (churn), before
they actually do it?

We use a real-style dataset of 7,043 customers, each with details like their contract type,
monthly charges, internet service, and how long they've been a customer. Using this data, we
built a machine learning pipeline that cleans the data safely, tries four different types of
models, tunes the two best ones, and picks a final model based on fair, repeated testing rather than a single lucky guess.

The final model can look at a customer it has never seen before and estimate how likely they
are to cancel. We also checked where the model gets things wrong, and whether it's equally
good at spotting churn across different types of customers (for example, by contract length),
rather than just trusting one overall accuracy number.

The end goal: a working, saved model that a telecom company could use to flag at-risk
customers early enough to try to keep them.

## Layout

    data/raw/       original csv, never edited
    src/setup.py     loading, type fixes, stratified split leakage-safe preprocessing (ColumnTransformer)
    notebooks/      numbered notebooks, to be run in order
    models/         saved best pipeline (joblib)
    

## Run it

    pip install -r requirements.txt
    cd notebooks
    jupyter notebook

Notebooks assume they are opened from inside `notebooks/`.

## Updated workflow

1. `01_data_pipeline_baseline.ipynb`: reusable data loading, summary statistics, duplicate and missing-value checks, target imbalance, training-set EDA, preprocessing explanation, and a majority-class baseline.
2. `02_model_comparison.ipynb`: four model families evaluated using the same stratified cross-validation folds; PR-AUC (average precision) is reported alongside F1, ROC-AUC and accuracy. Class weighting is used where supported.
3. `03_tuning_and_final_model.ipynb`: tunes Logistic Regression and Random Forest using PR-AUC, selects a threshold from out-of-fold training probabilities using F2, and evaluates on the held-out test set.
4. `04_error_and_subgroup_analysis.ipynb`: evaluates errors and subgroup recall at the selected style of threshold and adds permutation importance and confident-error review.
5. `05_save_and_reload_pipeline.ipynb`: saves and reloads the full pipeline together with its threshold, input schema, selected parameters, selection metric, random seed and scikit-learn version.

Run notebooks from inside the `notebooks/` directory, in order. Model scores may change after these methodological improvements; use the newly executed notebook outputs rather than carrying forward old score claims. The assumed business costs of missed churners and retention offers are not provided, so F2 is used as a transparent provisional threshold-selection objective, not as a proven financial optimum.


## Important before submission

Re-run notebooks 01–05 from top to bottom after installing `requirements.txt`. This refreshes outputs and regenerates `models/final_pipeline.joblib` with the new threshold and metadata. Do not present the old saved model or old score figures as the results of the updated workflow until notebook 05 has been rerun.
