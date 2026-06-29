"""Spearman correlation visualization replicating Hempel Figure 4."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DISPLAY_NAME_MAP = {
    "heart_rate_mean": "Heart Rate (mean, 24h)",
    "spo2_mean": "SpO2 (mean, 24h)",
    "resp_rate_mean": "Respiratory Rate (mean, 24h)",
    "temperature_mean": "Temperature (mean, 24h)",
    "gcs_eye_mean": "GCS Eye Opening (mean, 24h)",
    "gcs_verbal_mean": "GCS Verbal (mean, 24h)",
    "gcs_motor_mean": "GCS Motor (mean, 24h)",
    "age": "Age",
    "gender_male": "Gender (Male=1)",
    "anion_gap_mean": "Anion Gap (mean, 24h)",
    "bicarbonate_mean": "Bicarbonate (mean, 24h)",
    "chloride_mean": "Chloride (mean, 24h)",
    "creatinine_mean": "Creatinine (mean, 24h)",
    "glucose_mean": "Glucose (mean, 24h)",
    "sodium_mean": "Sodium (mean, 24h)",
    "magnesium_mean": "Magnesium (mean, 24h)",
    "potassium_mean": "Potassium (mean, 24h)",
    "phosphate_mean": "Phosphate (mean, 24h)",
    "bun_mean": "BUN (mean, 24h)",
    "hematocrit_mean": "Hematocrit (mean, 24h)",
    "hemoglobin_mean": "Hemoglobin (mean, 24h)",
    "mch_mean": "MCH (mean, 24h)",
    "mchc_mean": "MCHC (mean, 24h)",
    "mcv_mean": "MCV (mean, 24h)",
    "rdw_mean": "RDW (mean, 24h)",
    "rbc_mean": "RBC (mean, 24h)",
    "wbc_mean": "WBC (mean, 24h)",
    "platelets_mean": "Platelets (mean, 24h)",
    "concurrent_patients": "ICU Bed Occupancy (concurrent patients)",
    "occupancy_rate": "ICU Occupancy Rate",
    "occupancy_percentile": "ICU Occupancy Percentile",
}

EXCLUDE_COLUMNS = {"stay_id", "subject_id", "hadm_id", "los_days", "los_log", "los_class", "los_hours"}


def plot_correlation_with_los(
    feature_df: pd.DataFrame,
    los_col: str = "los_days",
    output_path: str | Path = "results/hempel/figures/fig4_correlation.png",
) -> pd.DataFrame:
    """Plot Spearman correlation of all numeric features with LOS.

    Args:
        feature_df: Feature matrix with outcome column.
        los_col: LOS outcome column.
        output_path: Figure output path.

    Returns:
        Correlation table.

    Raises:
        KeyError: If the LOS column is missing.
    """
    if los_col not in feature_df.columns:
        raise KeyError(f"{los_col} is missing")
    rows = []
    y = pd.to_numeric(feature_df[los_col], errors="coerce")
    for column in feature_df.columns:
        if column in EXCLUDE_COLUMNS or column == los_col:
            continue
        x = pd.to_numeric(feature_df[column], errors="coerce")
        valid = x.notna() & y.notna()
        if valid.sum() < 3 or x[valid].nunique() < 2:
            continue
        correlation, p_value = spearmanr(x[valid], y[valid])
        rows.append({"feature": column, "display_name": display_name(column), "spearman_r": correlation, "p_value": p_value})
    table = pd.DataFrame(rows).sort_values("spearman_r", key=lambda s: s.abs(), ascending=False)
    _plot_correlation_table(table, output_path)
    table_path = Path(output_path).parents[1] / "tables" / "table_correlations.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    return table


def display_name(feature: str) -> str:
    """Map a raw feature name to a readable display name.

    Args:
        feature: Raw feature name.

    Returns:
        Display-friendly feature name.
    """
    return DISPLAY_NAME_MAP.get(feature, feature.replace("_", " ").title())


def _plot_correlation_table(table: pd.DataFrame, output_path: str | Path) -> None:
    """Render and save the correlation bar chart.

    Args:
        table: Correlation table.
        output_path: Figure output path.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if table.empty:
        return
    plot_df = table.sort_values("spearman_r")
    colors = np.where(plot_df["spearman_r"] >= 0, "#d95f02", "#1f77b4")
    labels = [
        f"{name}{'***' if p < 0.001 else '*' if p < 0.05 else ''}"
        for name, p in zip(plot_df["display_name"], plot_df["p_value"])
    ]
    height = max(8, len(plot_df) * 0.28)
    fig, ax = plt.subplots(figsize=(8, height))
    ax.barh(labels, plot_df["spearman_r"], color=colors)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman r")
    ax.set_title("Spearman Correlation of Features with ICU Length of Stay")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
