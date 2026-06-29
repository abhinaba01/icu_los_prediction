"""Metric computations for classification and regression stages."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict[str, float]:
    """Compute Stage 1 classification metrics.

    Args:
        y_true: True LOS class labels.
        y_pred: Predicted LOS class labels.
        y_proba: Optional class probability matrix.

    Returns:
        Dictionary of accuracy, F1, AUC, and kappa metrics.
    """
    per_class = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro-F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Short-F1": per_class[0],
        "Medium-F1": per_class[1],
        "Long-F1": per_class[2],
        "Kappa": cohen_kappa_score(y_true, y_pred),
    }
    metrics["AUC"] = _safe_auc(y_true, y_proba)
    return metrics


def regression_metrics(y_true_days: np.ndarray, y_pred_days: np.ndarray) -> dict[str, float]:
    """Compute Stage 2 regression metrics in raw LOS days.

    Args:
        y_true_days: True LOS in days.
        y_pred_days: Predicted LOS in days.

    Returns:
        Dictionary of regression metrics.
    """
    y_true = np.asarray(y_true_days, dtype=float)
    y_pred = np.asarray(y_pred_days, dtype=float)
    mask = y_true > 0.5
    if mask.any():
        mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100.0
    else:
        mape = np.nan
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mape,
        "MedianAE": median_absolute_error(y_true, y_pred),
    }


def build_cohort_table(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Build a cohort characteristics table with simple summaries.

    Args:
        feature_df: Feature DataFrame containing `los_class`.

    Returns:
        Cohort summary DataFrame.
    """
    rows = []
    groups = {"Full cohort": feature_df}
    labels = {0: "Short-LOS", 1: "Medium-LOS", 2: "Long-LOS"}
    for label, class_id in labels.items():
        groups[label] = feature_df[feature_df["los_class"].eq(class_id)]
    for column in ["age", "los_days", "gender_male"]:
        if column not in feature_df.columns:
            continue
        row = {"variable": column}
        for group_name, group in groups.items():
            series = pd.to_numeric(group[column], errors="coerce")
            if column == "gender_male":
                value = f"{int(series.sum())} ({series.mean() * 100:.1f}%)"
            else:
                value = f"{series.mean():.2f} +/- {series.std():.2f}"
            row[group_name] = value
        row["p_value"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_auc(y_true: np.ndarray, y_proba: np.ndarray | None) -> float:
    """Compute macro one-vs-rest ROC-AUC when probabilities are available.

    Args:
        y_true: True labels.
        y_proba: Probability matrix.

    Returns:
        Macro ROC-AUC or NaN when unavailable.
    """
    if y_proba is None:
        return np.nan
    try:
        return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        return np.nan
