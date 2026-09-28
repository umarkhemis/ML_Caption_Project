"""Loading, typing fixes and the train/test split for the Telco churn data."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
           "master/data/Telco-Customer-Churn.csv")
RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco.csv"
TARGET = "Churn"
SEED = 42


def load_raw(path=RAW_PATH):
    """Read the raw csv. If it is missing, fetch it once from the IBM repo."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(RAW_URL).to_csv(path, index=False)
    return pd.read_csv(path)


def tidy(df):
    """Type fixes only. Nothing in here learns from the data (no means, no
    modes), so it is safe to run before the split. Filling the blank
    TotalCharges values is left to the imputer inside the pipeline."""
    df = df.copy()
    # blanks come through as " ", which turns the whole column into text
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop(columns=["customerID"])  # an identifier, not a feature
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def split(df, test_size=0.2, seed=SEED):
    """Stratified split so both halves keep the ~26.5% churn rate."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
