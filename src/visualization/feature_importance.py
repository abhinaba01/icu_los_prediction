"""XGBoost feature importance visualization replicating Hempel Figure 5."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.visualization.correlation_matrix import display_name

LOGGER = logging.getLogger(__name__)


def plot_xgboost_feature_importance(
    xgb_model: object,
    feature_names: list[str],
    top_n: int = 20,
    output_path: str | Path = "results/hempel/figures/fig5_importance.png",
    X_test: pd.DataFrame | None = None,
    shap_output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Plot XGBoost feature importance and optional SHAP summary.

    Args:
        xgb_model: Fitted XGBoost-like model exposing `feature_importances_`.
        feature_names: Names of processed features.
        top_n: Number of top features to plot.
        output_path: Feature-importance figure path.
        X_test: Optional test matrix for SHAP summary.
        shap_output_path: Optional SHAP output path.

    Returns:
        Feature importance DataFrame.

    Raises:
        AttributeError: If the model lacks `feature_importances_`.
    """
    if not hasattr(xgb_model, "feature_importances_"):
        raise AttributeError("xgb_model must expose feature_importances_")
    importance_df = pd.DataFrame({"feature": feature_names, "importance": xgb_model.feature_importances_})
    importance_df["display_name"] = importance_df["feature"].map(display_name)
    importance_df = importance_df.sort_values("importance", ascending=False)
    _plot_importance(importance_df.head(top_n), output_path)
    table_path = Path(output_path).parents[1] / "tables" / "xgboost_feature_importance.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(table_path, index=False)
    if X_test is not None:
        _plot_shap_summary(xgb_model, X_test, shap_output_path or Path(output_path).with_name("shap_summary.png"))
    return importance_df


def _plot_importance(importance_df: pd.DataFrame, output_path: str | Path) -> None:
    """Render and save a horizontal feature-importance chart.

    Args:
        importance_df: Top feature importance rows.
        output_path: Figure output path.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_df = importance_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(5, len(plot_df) * 0.35)))
    ax.barh(plot_df["display_name"], plot_df["importance"], color="#2ca25f")
    ax.set_xlabel("Gain-based importance")
    ax.set_title("XGBoost Feature Importance (Gain) - Stage 2 Regression")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_shap_summary(model: object, X_test: pd.DataFrame, output_path: str | Path) -> None:
    """Render and save a SHAP summary plot when SHAP is installed.

    Args:
        model: Fitted tree model.
        X_test: Test feature matrix.
        output_path: Figure output path.
    """
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        LOGGER.warning("shap is not installed; skipping SHAP summary plot")
        return
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=list(X_test.columns), show=False)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
