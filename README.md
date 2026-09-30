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

## Status

- [x] 01 data loading, pipeline, baseline
- [ ] 02 four models with cross-validation
- [ ] 03 tuning and comparison
- [ ] 04 error and subgroup analysis
- [ ] 05 save and reload best pipeline
