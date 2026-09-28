# Predicting Telecom Customer Churn

BCS 3101 Capstone - Ahmed Umar Khemis - Reg. No. 2024/A/KCS/3970/F

Binary classification: will a telecom customer cancel their subscription?
Dataset: Telco Customer Churn (IBM sample data), 7,043 customers, 21 columns.

## Layout

    data/raw/       original csv, never edited
    src/data.py     loading, type fixes, stratified split
    src/pipeline.py leakage-safe preprocessing (ColumnTransformer)
    notebooks/      numbered notebooks, run in order
    models/         saved best pipeline (joblib)
    reports/        report drafts and final PDF
    figures/        charts used in the report

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
