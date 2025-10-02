from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd

FEATURES = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
    "BMI","DiabetesPedigreeFunction","Age"
]

TARGET = "Outcome"

@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series

def load_dataset(path: str) -> Dataset:
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    return Dataset(X, y)

def standardize(X: pd.DataFrame) -> pd.DataFrame:
    # Simple z-score standardization for continuous features
    z = X.copy()
    for col in X.columns:
        mean, std = X[col].mean(), X[col].std()
        if std == 0 or np.isnan(std):
            z[col] = 0.0
        else:
            z[col] = (X[col] - mean) / std
    return z

def metrics_to_json(metrics: Dict[str, Any], path: str):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
