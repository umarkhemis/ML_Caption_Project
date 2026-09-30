

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_validate

DATA_PATH = "../data/raw/telco.csv"

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORY_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

SCORING = ["accuracy", "f1", "roc_auc"]


def load_and_clean(path=DATA_PATH):
    """Read the csv and fix the two problems notebook 01 found:
    TotalCharges stored as text, and the target stored as Yes/No instead of 1/0."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop(columns=["customerID"])
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def make_preprocessing():
    """Build the exact same leakage-safe pipeline from notebook 01:
    fill missing numeric values and scale them, one-hot encode the categories."""
    numeric_steps = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    category_steps = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([
        ("numbers", numeric_steps, NUMERIC_COLUMNS),
        ("categories", category_steps, CATEGORY_COLUMNS),
    ])


def prepare_everything(path=DATA_PATH, seed=42):
    
    df = load_and_clean(path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    preprocessing = make_preprocessing()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return X_train, X_test, y_train, y_test, preprocessing, cv


def evaluate(model, name, X_train, y_train, preprocessing, cv):
    """Run 5-fold cross-validation on a model and return its average
    accuracy / F1 / ROC-AUC (plus the spread across folds) as one row."""
    full_model = Pipeline([("preprocessing", preprocessing), ("model", model)])
    results = cross_validate(full_model, X_train, y_train, cv=cv, scoring=SCORING)
    row = {"model": name}
    for score_name in SCORING:
        scores = results["test_" + score_name]
        row[score_name + " (mean)"] = round(scores.mean(), 3)
        row[score_name + " (spread)"] = round(scores.std(), 3)
    return row
