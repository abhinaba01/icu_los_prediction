"""Master script for the ICU occupancy ablation study."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_config
from src.evaluation.ablation import (
    ablation_by_los_class,
    compute_delta_table,
    permutation_importance_occupancy,
    plot_learning_curves_ablation,
    shap_interaction_occupancy,
    stratified_analysis_by_occupancy,
    test_significance,
)
from src.preprocessing.pipeline import PreprocessingPipeline
from src.visualization.ablation_plots import plot_ablation_summary, plot_occupancy_feature_rank


def configure_logging(config: dict) -> None:
    """Configure file and console logging.

    Args:
        config: Parsed project configuration.
    """
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, config["project"].get("log_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_dir / "ablation_run.log"), logging.StreamHandler()],
    )


def choose_common_model(hempel_stage2: pd.DataFrame, extended_stage2: pd.DataFrame) -> str:
    """Choose the best common model by baseline MAE.

    Args:
        hempel_stage2: Baseline Stage 2 metrics.
        extended_stage2: Extended Stage 2 metrics.

    Returns:
        Selected model name.
    """
    common = set(hempel_stage2["model"]).intersection(extended_stage2["model"])
    ranked = hempel_stage2[hempel_stage2["model"].isin(common)].sort_values("MAE")
    return str(ranked.iloc[0]["model"])


def load_prediction_pair(model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load baseline and extended Stage 2 prediction tables.

    Args:
        model_name: Model name.

    Returns:
        Tuple of Hempel and extended prediction DataFrames.
    """
    hempel = pd.read_csv(PROJECT_ROOT / "results" / "hempel" / "tables" / "predictions" / f"{model_name}_stage2_predictions.csv")
    extended = pd.read_csv(PROJECT_ROOT / "results" / "extended" / "tables" / "predictions" / f"{model_name}_stage2_predictions.csv")
    return hempel, extended


def main() -> None:
    """Run the full occupancy ablation study."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    configure_logging(config)
    logger = logging.getLogger(__name__)
    hempel_stage1 = pd.read_csv(PROJECT_ROOT / "results" / "hempel" / "tables" / "stage1_metrics.csv")
    hempel_stage2 = pd.read_csv(PROJECT_ROOT / "results" / "hempel" / "tables" / "stage2_metrics.csv")
    extended_stage1 = pd.read_csv(PROJECT_ROOT / "results" / "extended" / "tables" / "stage1_metrics.csv")
    extended_stage2 = pd.read_csv(PROJECT_ROOT / "results" / "extended" / "tables" / "stage2_metrics.csv")
    delta_df = compute_delta_table(pd.concat([hempel_stage1, hempel_stage2]), pd.concat([extended_stage1, extended_stage2]), PROJECT_ROOT / "results" / "ablation" / "tables")
    plot_ablation_summary(delta_df, PROJECT_ROOT / "results" / "ablation" / "figures" / "ablation_summary.png")

    model_name = choose_common_model(hempel_stage2, extended_stage2)
    hempel_pred, extended_pred = load_prediction_pair(model_name)
    significance = test_significance(
        hempel_pred["y_pred_days"].to_numpy(),
        extended_pred["y_pred_days"].to_numpy(),
        hempel_pred["y_true_days"].to_numpy(),
        PROJECT_ROOT / "results" / "ablation" / "tables",
        model_name=model_name,
        n_models=len(set(hempel_stage2["model"]).intersection(extended_stage2["model"])),
    )

    _run_model_dependent_ablation(config, model_name)
    conclusion = "did" if significance["conclusion"] == "extended_improved" else "did not"
    logger.info("The ICU bed occupancy feature %s improve Stage 2 performance at corrected p < 0.05.", conclusion)


def _run_model_dependent_ablation(config: dict, model_name: str) -> None:
    """Run ablation analyses requiring fitted models and feature matrices.

    Args:
        config: Parsed project configuration.
        model_name: Selected Stage 2 model name.
    """
    logger = logging.getLogger(__name__)
    try:
        hempel_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "hempel_features.parquet")
        extended_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "extended_features.parquet")
        split_loader = PreprocessingPipeline(config)
        splits = split_loader.load_split_indices(PROJECT_ROOT / "results" / "split_indices.npz")
        hempel_pre = joblib.load(PROJECT_ROOT / "results" / "hempel" / "models" / "preprocessor.pkl")
        extended_pre = joblib.load(PROJECT_ROOT / "results" / "extended" / "models" / "preprocessor.pkl")
        X_h, y_class = hempel_pre.get_X_y_classification(hempel_df)
        X_e, _ = extended_pre.get_X_y_classification(extended_df)
        _, y_reg = hempel_pre.get_X_y_regression(hempel_df)
        X_h_test = hempel_pre.transform(X_h.iloc[splits["test_idx"]])
        X_e_test = extended_pre.transform(X_e.iloc[splits["test_idx"]])
        y_class_test = y_class.iloc[splits["test_idx"]].to_numpy()
        y_reg_test = y_reg.iloc[splits["test_idx"]].to_numpy()
        hempel_stage2 = joblib.load(PROJECT_ROOT / "results" / "hempel" / "models" / f"{model_name}_stage2.pkl")
        extended_stage2 = joblib.load(PROJECT_ROOT / "results" / "extended" / "models" / f"{model_name}_stage2.pkl")
        stratified_analysis_by_occupancy(extended_stage2, X_e_test, y_reg_test, hempel_stage2, X_h_test, PROJECT_ROOT / "results" / "ablation")
        permutation_importance_occupancy(extended_stage2, X_e_test, y_reg_test, output_path=PROJECT_ROOT / "results" / "ablation" / "figures" / "permutation_importance.png")
        plot_learning_curves_ablation(hempel_stage2, extended_stage2, X_h_test, X_e_test, y_reg_test, PROJECT_ROOT / "results" / "ablation" / "figures" / "learning_curves.png")
        if model_name == "xgboost":
            shap_interaction_occupancy(extended_stage2, X_e_test, output_dir=PROJECT_ROOT / "results" / "ablation" / "figures" / "shap_interactions")
        _run_stage1_class_ablation(y_class_test, X_h_test, X_e_test)
        _plot_rank_if_available()
    except FileNotFoundError as exc:
        logger.warning("Skipping model-dependent ablation because a prerequisite file is missing: %s", exc)


def _run_stage1_class_ablation(y_class_test: np.ndarray, X_h_test: pd.DataFrame, X_e_test: pd.DataFrame) -> None:
    """Run class-specific Stage 1 ablation when matching models exist.

    Args:
        y_class_test: Test labels.
        X_h_test: Baseline test features.
        X_e_test: Extended test features.
    """
    for candidate in ["xgboost", "random_forest", "logistic_regression"]:
        h_path = PROJECT_ROOT / "results" / "hempel" / "models" / f"{candidate}_stage1.pkl"
        e_path = PROJECT_ROOT / "results" / "extended" / "models" / f"{candidate}_stage1.pkl"
        if h_path.exists() and e_path.exists():
            ablation_by_los_class(joblib.load(h_path), joblib.load(e_path), X_h_test, X_e_test, y_class_test, PROJECT_ROOT / "results" / "ablation" / "tables")
            return


def _plot_rank_if_available() -> None:
    """Plot occupancy feature rank when importance tables exist."""
    h_path = PROJECT_ROOT / "results" / "hempel" / "tables" / "xgboost_feature_importance.csv"
    e_path = PROJECT_ROOT / "results" / "extended" / "tables" / "xgboost_feature_importance.csv"
    if h_path.exists() and e_path.exists():
        plot_occupancy_feature_rank(pd.read_csv(h_path), pd.read_csv(e_path), PROJECT_ROOT / "results" / "ablation" / "figures" / "occupancy_rank.png")


if __name__ == "__main__":
    main()
