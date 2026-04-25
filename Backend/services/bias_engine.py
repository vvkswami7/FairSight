import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Optional


def detect_sensitive_columns(df: pd.DataFrame) -> list:
    sensitive_keywords = ['gender','sex','race','ethnicity','age','religion','nationality','disability','marital','color','origin']
    return [col for col in df.columns if any(kw in col.lower() for kw in sensitive_keywords)]


def compute_demographic_parity(df: pd.DataFrame, sensitive_col: str, label_col: str) -> dict:
    groups = df[sensitive_col].unique()
    group_rates = {}
    for g in groups:
        subset = df[df[sensitive_col] == g]
        if len(subset) > 0:
            group_rates[str(g)] = round(float(subset[label_col].mean()), 4)

    if not group_rates:
        return {"group_rates": {}, "disparity": 0, "max_group": "", "min_group": "", "is_biased": False}

    max_rate = max(group_rates.values())
    min_rate = min(group_rates.values())
    disparity = round(max_rate - min_rate, 4)
    return {
        "group_rates": group_rates,
        "disparity": disparity,
        "max_group": max(group_rates, key=group_rates.get),
        "min_group": min(group_rates, key=group_rates.get),
        "is_biased": disparity > 0.1
    }


def compute_equalized_odds(df: pd.DataFrame, sensitive_col: str, label_col: str, pred_col: str) -> dict:
    groups = df[sensitive_col].unique()
    tpr_per_group = {}
    fpr_per_group = {}
    for g in groups:
        subset = df[df[sensitive_col] == g]
        positives = subset[subset[label_col] == 1]
        negatives = subset[subset[label_col] == 0]
        tpr_per_group[str(g)] = round(float(positives[pred_col].mean()), 4) if len(positives) > 0 else 0.0
        fpr_per_group[str(g)] = round(float(negatives[pred_col].mean()), 4) if len(negatives) > 0 else 0.0

    tpr_vals = list(tpr_per_group.values()) or [0]
    fpr_vals = list(fpr_per_group.values()) or [0]
    return {
        "tpr_per_group": tpr_per_group,
        "fpr_per_group": fpr_per_group,
        "tpr_disparity": round(max(tpr_vals) - min(tpr_vals), 4),
        "fpr_disparity": round(max(fpr_vals) - min(fpr_vals), 4),
    }


def compute_fairness_score(demographic_parity: dict, equalized_odds: dict) -> int:
    dp_score = max(0, 1 - demographic_parity["disparity"] * 5)
    tpr_score = max(0, 1 - equalized_odds["tpr_disparity"] * 5)
    fpr_score = max(0, 1 - equalized_odds["fpr_disparity"] * 5)
    raw = (dp_score * 0.4 + tpr_score * 0.3 + fpr_score * 0.3) * 100
    return int(round(min(100, max(0, raw))))


def encode_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Encode all non-numeric columns. Returns encoded df and label map."""
    df = df.copy()
    le_map = {}
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_map[col] = list(le.classes_)
    return df, le_map


def train_and_analyze(df: pd.DataFrame, label_col: str, sensitive_cols: list) -> dict:
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")

    # Keep original values for fairness metrics BEFORE encoding
    original_df = df.copy()

    # Encode everything for model training
    encoded_df, le_map = encode_df(df)

    feature_cols = [c for c in encoded_df.columns if c != label_col]
    X = encoded_df[feature_cols].astype(float)
    y = encoded_df[label_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred), 4)

    # For fairness metrics, use ORIGINAL (string) values so group names are readable
    orig_test = original_df.iloc[X_test.index].copy()
    orig_test["__label__"] = y_test.values
    orig_test["__pred__"] = y_pred

    results = {
        "model_accuracy": accuracy,
        "total_samples": len(df),
        "test_samples": len(X_test),
        "label_col": label_col,
        "sensitive_cols": sensitive_cols,
        "bias_analysis": {}
    }

    for col in sensitive_cols:
        if col not in orig_test.columns:
            continue

        # Bin numeric columns (like age) into readable ranges
        working = orig_test.copy()
        if pd.api.types.is_numeric_dtype(working[col]):
            bins = pd.cut(working[col], bins=3, precision=0)
            working[col] = bins.astype(str)

        dp = compute_demographic_parity(working, col, "__label__")
        eo = compute_equalized_odds(working, col, "__label__", "__pred__")
        score = compute_fairness_score(dp, eo)

        results["bias_analysis"][col] = {
            "fairness_score": score,
            "demographic_parity": dp,
            "equalized_odds": eo,
            "severity": "High" if score < 50 else "Medium" if score < 75 else "Low"
        }

    return results


def get_column_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for col in df.columns:
        if df[col].dtype == object or df[col].nunique() <= 20:
            stats[col] = {"type": "categorical", "unique_values": int(df[col].nunique()),
                          "top_values": {str(k): int(v) for k, v in df[col].value_counts().head(5).items()}}
        else:
            stats[col] = {"type": "numeric", "mean": round(float(df[col].mean()), 2),
                          "std": round(float(df[col].std()), 2),
                          "min": round(float(df[col].min()), 2),
                          "max": round(float(df[col].max()), 2)}
    return stats