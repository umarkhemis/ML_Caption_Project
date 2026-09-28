"""Preprocessing that is fitted on training data only.

Everything that learns something (median, mean/std, category list) lives in
the ColumnTransformer, so cross-validation refits it inside every fold and
the test set never influences it.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
ALREADY_BINARY = ["SeniorCitizen"]  # 0/1 already, pass through untouched
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def make_preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(
        handle_unknown="ignore",  # a category seen only in test becomes all zeros
        drop="if_binary",         # Yes/No columns become one 0/1 column, not two
        sparse_output=False,
    )
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
        ("bin", "passthrough", ALREADY_BINARY),
    ])


def make_model(estimator):
    """Attach an estimator to a fresh preprocessor."""
    return Pipeline([("prep", make_preprocessor()), ("model", estimator)])
