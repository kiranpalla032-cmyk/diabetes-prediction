import argparse
import os
import joblib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from utils import load_dataset, standardize, FEATURES

def evaluate(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = None
    return metrics

def plot_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, str(z), ha='center', va='center')
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main(args):
    data_path = args.data
    out_dir = "models"
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset(data_path)
    X = ds.X.copy()
    y = ds.y.copy()

    # Keep a standardized copy for linear model
    X_std = standardize(X)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    Xs_tr, Xs_te, _, _ = train_test_split(X_std, y, test_size=0.25, random_state=42, stratify=y)

    # Models
    logreg = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)

    logreg.fit(Xs_tr, y_tr)
    rf.fit(X_tr, y_tr)

    # Evaluate
    pred_lr = logreg.predict(Xs_te)
    proba_lr = logreg.predict_proba(Xs_te)[:,1]
    m_lr = evaluate(y_te, pred_lr, proba_lr)

    pred_rf = rf.predict(X_te)
    proba_rf = rf.predict_proba(X_te)[:,1]
    m_rf = evaluate(y_te, pred_rf, proba_rf)

    # Choose best by f1
    best_model, best_name, best_metrics = (logreg, "logistic_regression", m_lr) if m_lr["f1"] >= m_rf["f1"] else (rf, "random_forest", m_rf)

    # Save artifacts
    joblib.dump(best_model, f"{out_dir}/best_model.joblib")
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump({"best_model": best_name, "metrics": best_metrics}, f, indent=2)

    # Save confusion matrices
    plot_confusion(y_te, pred_lr, f"{out_dir}/confusion_logreg.png")
    plot_confusion(y_te, pred_rf, f"{out_dir}/confusion_rf.png")

    print("Training complete.")
    print("Best model:", best_name)
    print("Metrics:", best_metrics)
    print("Saved to:", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/diabetes_sample.csv", help="Path to CSV data file")
    args = parser.parse_args()
    main(args)
