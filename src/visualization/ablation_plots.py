"""Visualization helpers for occupancy ablation analyses."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OCCUPANCY_COLUMNS = {"concurrent_patients", "occupancy_rate", "occupancy_percentile"}


def plot_ablation_summary(
    delta_df: pd.DataFrame,
    output_path: str | Path = "results/ablation/figures/ablation_summary.png",
) -> None:
    """Create the 2x2 ablation summary plot.

    Args:
        delta_df: Delta metrics table.
        output_path: Figure output path.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    panels = [
        ("stage2", "R2", "Delta R2"),
        ("stage2", "MAE", "Delta MAE"),
        ("stage1", "Macro-F1", "Delta Macro-F1"),
        ("stage1", "AUC", "Delta AUC"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (stage, metric, title) in zip(axes.ravel(), panels):
        panel = delta_df[(delta_df["stage"].eq(stage)) & (delta_df["metric"].eq(metric))].copy()
        panel = panel.sort_values("delta")
        colors = [_delta_color(metric, value) for value in panel["delta"]]
        labels = [
            f"{model}{'*' if significant else ''}"
            for model, significant in zip(panel["model"], panel.get("significant", False))
        ]
        ax.barh(labels, panel["delta"], color=colors)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
    fig.suptitle("Impact of ICU Bed Occupancy Feature on Model Performance")
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_occupancy_feature_rank(
    hempel_importance_df: pd.DataFrame,
    extended_importance_df: pd.DataFrame,
    output_path: str | Path = "results/ablation/figures/occupancy_rank.png",
) -> None:
    """Plot feature ranks with occupancy features highlighted.

    Args:
        hempel_importance_df: Baseline importance table.
        extended_importance_df: Extended importance table.
        output_path: Figure output path.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    _plot_rank_panel(axes[0], hempel_importance_df, "Hempel Feature Importance")
    _plot_rank_panel(axes[1], extended_importance_df, "Extended Feature Importance")
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _delta_color(metric: str, delta: float) -> str:
    """Return green for improvement and red for degradation.

    Args:
        metric: Metric name.
        delta: Extended minus baseline metric value.

    Returns:
        Hex color string.
    """
    error_metric = metric in {"MAE", "RMSE", "MAPE", "MedianAE"}
    improved = delta < 0 if error_metric else delta > 0
    return "#59a14f" if improved else "#e15759"


def _plot_rank_panel(ax, importance_df: pd.DataFrame, title: str) -> None:
    """Plot a feature-importance ranking panel.

    Args:
        ax: Matplotlib axes.
        importance_df: Importance table with `feature` and importance column.
        title: Panel title.
    """
    if importance_df.empty:
        ax.axis("off")
        return
    value_col = "importance" if "importance" in importance_df.columns else "mean_abs_shap"
    plot_df = importance_df.sort_values(value_col, ascending=False).head(30).iloc[::-1]
    colors = ["#b22222" if feature in OCCUPANCY_COLUMNS else "#4c78a8" for feature in plot_df["feature"]]
    ax.barh(plot_df["feature"], plot_df[value_col], color=colors)
    ax.set_title(title)
    ax.set_xlabel(value_col)
