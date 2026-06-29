"""Model comparison plots for Stage 1 and Stage 2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def plot_stage1_comparison(metrics_df: pd.DataFrame, output_dir: str | Path = "results/hempel/figures/") -> None:
    """Generate comparison plots for Stage 1 classifiers.

    Args:
        metrics_df: Stage 1 metrics table.
        output_dir: Figure output directory.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics = ["Accuracy", "Macro-F1", "AUC"]
    metrics_df.set_index("model")[metrics].plot(kind="bar", ax=axes[0], color=["#4c78a8", "#59a14f", "#f28e2b"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Stage 1 Overall Metrics")
    axes[0].set_ylabel("Score")
    class_metrics = ["Short-F1", "Medium-F1", "Long-F1"]
    metrics_df.set_index("model")[class_metrics].plot(kind="bar", ax=axes[1], color=["#8cd17d", "#b6992d", "#e15759"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Stage 1 Per-Class F1")
    axes[1].set_ylabel("F1")
    fig.tight_layout()
    fig.savefig(directory / "stage1_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_stage2_comparison(
    metrics_df: pd.DataFrame,
    output_dir: str | Path = "results/hempel/figures/",
    predictions: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Generate comparison plots for Stage 2 regressors.

    Args:
        metrics_df: Stage 2 metrics table.
        output_dir: Figure output directory.
        predictions: Optional mapping of model names to prediction tables.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics_df.set_index("model")[["MAE", "RMSE", "R2"]].plot(kind="bar", ax=axes[0], color=["#4c78a8", "#f28e2b", "#59a14f"])
    axes[0].set_title("Stage 2 Regression Metrics")
    axes[0].set_ylabel("Metric value")
    if predictions:
        best_model = metrics_df.sort_values("MAE").iloc[0]["model"]
        pred_df = predictions.get(best_model)
        if pred_df is not None:
            axes[1].scatter(pred_df["y_true_days"], pred_df["y_pred_days"], s=14, alpha=0.45)
            max_value = np.nanmax([pred_df["y_true_days"].max(), pred_df["y_pred_days"].max()])
            axes[1].plot([0, max_value], [0, max_value], color="black", linestyle="--")
            axes[1].set_xlabel("Actual LOS (days)")
            axes[1].set_ylabel("Predicted LOS (days)")
            axes[1].set_title(f"Predicted vs Actual ({best_model})")
    else:
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(directory / "stage2_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
