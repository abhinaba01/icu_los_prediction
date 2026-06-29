"""Master script for the SOFA-feature ablation study.

Usage:
    python scripts/run_ablation.py

This script assumes `run_hempel.py` and the SOFA-based `run_extended.py`
have already completed. It compares the Hempel baseline against the extended
feature set and writes SOFA-specific outputs under `results/ablation_sofa/`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_config
from src.evaluation.ablation import (
    ablation_by_los_class,
    compute_delta_table,
    plot_learning_curves_ablation,
    shap_interaction_occupancy,
    test_significance,
)
from src.preprocessing.pipeline import PreprocessingPipeline

ABLATION_ROOT = PROJECT_ROOT / "results" / "ablation_sofa"
SOFA_FEATURES = [
    "sofa_total",
    "sofa_resp",
    "sofa_cardio",
    "sofa_hepatic",
    "sofa_coag",
    "sofa_renal",
    "sofa_neuro",
    "urine_24h",
    "bilirubin_max",
]


def configure_logging(config: dict[str, Any]) -> None:
    """Configure file and console logging.

    Args:
        config: Parsed project configuration.
    """
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, config["project"].get("log_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_dir / "sofa_ablation_run.log"), logging.StreamHandler()],
    )


def choose_common_model(hempel_stage2: pd.DataFrame, extended_stage2: pd.DataFrame) -> str:
    """Choose the best common Stage 2 model by baseline MAE.

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
    """Run the complete SOFA-feature ablation workflow."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    configure_logging(config)
    logger = logging.getLogger(__name__)
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    (ABLATION_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    (ABLATION_ROOT / "figures").mkdir(parents=True, exist_ok=True)

    hempel_stage1 = pd.read_csv(PROJECT_ROOT / "results" / "hempel" / "tables" / "stage1_metrics.csv")
    hempel_stage2 = pd.read_csv(PROJECT_ROOT / "results" / "hempel" / "tables" / "stage2_metrics.csv")
    extended_stage1 = pd.read_csv(PROJECT_ROOT / "results" / "extended" / "tables" / "stage1_metrics.csv")
    extended_stage2 = pd.read_csv(PROJECT_ROOT / "results" / "extended" / "tables" / "stage2_metrics.csv")

    delta_df = compute_delta_table(
        pd.concat([hempel_stage1, hempel_stage2], ignore_index=True),
        pd.concat([extended_stage1, extended_stage2], ignore_index=True),
        ABLATION_ROOT / "tables",
    )
    plot_sofa_ablation_summary(delta_df, ABLATION_ROOT / "figures" / "sofa_ablation_summary.png")

    model_name = choose_common_model(hempel_stage2, extended_stage2)
    hempel_pred, extended_pred = load_prediction_pair(model_name)
    significance = test_significance(
        hempel_pred["y_pred_days"].to_numpy(),
        extended_pred["y_pred_days"].to_numpy(),
        hempel_pred["y_true_days"].to_numpy(),
        ABLATION_ROOT / "tables",
        model_name=model_name,
        n_models=len(set(hempel_stage2["model"]).intersection(extended_stage2["model"])),
    )

    _run_model_dependent_ablation(config, model_name)
    conclusion = "did" if significance["conclusion"] == "extended_improved" else "did not"
    logger.info("The SOFA feature set %s improve Stage 2 performance at corrected p < 0.05.", conclusion)


def _run_model_dependent_ablation(config: dict[str, Any], model_name: str) -> None:
    """Run SOFA analyses requiring fitted models and feature matrices.

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

        sofa_features = _detect_sofa_features(X_e_test)
        logger.info("Detected SOFA features for ablation: %s", sofa_features)

        hempel_stage2 = joblib.load(PROJECT_ROOT / "results" / "hempel" / "models" / f"{model_name}_stage2.pkl")
        extended_stage2 = joblib.load(PROJECT_ROOT / "results" / "extended" / "models" / f"{model_name}_stage2.pkl")
        stratified_analysis_by_sofa(
            extended_stage2,
            X_e_test,
            y_reg_test,
            hempel_stage2,
            X_h_test,
            sofa_col="sofa_total",
            output_root=ABLATION_ROOT,
        )
        permutation_importance_sofa(
            extended_stage2,
            X_e_test,
            y_reg_test,
            sofa_features,
            output_root=ABLATION_ROOT,
        )
        plot_learning_curves_ablation(
            hempel_stage2,
            extended_stage2,
            X_h_test,
            X_e_test,
            y_reg_test,
            ABLATION_ROOT / "figures" / "sofa_learning_curves.png",
        )
        if model_name == "xgboost" and "sofa_total" in X_e_test.columns:
            shap_sample = X_e_test.sample(n=min(2000, len(X_e_test)), random_state=42)
            shap_interaction_occupancy(
                extended_stage2,
                shap_sample,
                occupancy_col="sofa_total",
                output_dir=ABLATION_ROOT / "figures" / "shap_interactions",
            )
        _run_stage1_class_ablation(y_class_test, X_h_test, X_e_test)
    except FileNotFoundError as exc:
        logger.warning("Skipping model-dependent SOFA ablation because a prerequisite file is missing: %s", exc)


def _detect_sofa_features(X_extended: pd.DataFrame) -> list[str]:
    """Find SOFA feature columns present in the processed extended matrix.

    Args:
        X_extended: Processed extended feature matrix.

    Returns:
        SOFA feature columns found in the matrix.
    """
    return [feature for feature in SOFA_FEATURES if feature in X_extended.columns]


def permutation_importance_sofa(
    model: object,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    sofa_features: list[str],
    output_root: str | Path,
    n_repeats: int = 30,
) -> pd.DataFrame:
    """Compute permutation importance and highlight SOFA features.

    Args:
        model: Fitted extended model.
        X_test: Extended test feature matrix.
        y_test: Test target in log-LOS scale.
        sofa_features: SOFA columns to isolate.
        output_root: SOFA ablation output root.
        n_repeats: Number of permutation repeats.

    Returns:
        Full permutation-importance table.
    """
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "is_sofa_feature": [feature in sofa_features for feature in X_test.columns],
        }
    ).sort_values("importance_mean", ascending=False)
    root = Path(output_root)
    _save_table(importance_df, root / "tables", "sofa_permutation_importance")
    _plot_permutation_importance(importance_df, root / "figures" / "sofa_permutation_importance.png")
    return importance_df


def stratified_analysis_by_sofa(
    extended_model: object,
    X_test_extended: pd.DataFrame,
    y_test_log: np.ndarray,
    hempel_model: object,
    X_test_hempel: pd.DataFrame,
    sofa_col: str,
    output_root: str | Path,
) -> pd.DataFrame:
    """Compare baseline and extended MAE across SOFA severity strata.

    Args:
        extended_model: Fitted extended Stage 2 model.
        X_test_extended: Extended test features.
        y_test_log: Test target in log-LOS scale.
        hempel_model: Fitted Hempel Stage 2 model.
        X_test_hempel: Hempel test features.
        sofa_col: SOFA feature used for stratification.
        output_root: SOFA ablation output root.

    Returns:
        Stratified MAE table.

    Raises:
        KeyError: If `sofa_col` is missing.
    """
    if sofa_col not in X_test_extended.columns:
        raise KeyError(f"{sofa_col} is not present in X_test_extended")
    y_true_days = np.expm1(np.asarray(y_test_log, dtype=float))
    hempel_pred = np.clip(np.expm1(hempel_model.predict(X_test_hempel)), 0.0, None)
    extended_pred = np.clip(np.expm1(extended_model.predict(X_test_extended)), 0.0, None)
    q33, q66 = X_test_extended[sofa_col].quantile([0.33, 0.66])
    strata = [
        ("Low SOFA", X_test_extended[sofa_col] < q33),
        ("Medium SOFA", X_test_extended[sofa_col].between(q33, q66, inclusive="both")),
        ("High SOFA", X_test_extended[sofa_col] > q66),
    ]
    rows = []
    for label, mask in strata:
        if not mask.any():
            continue
        hempel_mae = mean_absolute_error(y_true_days[mask], hempel_pred[mask])
        extended_mae = mean_absolute_error(y_true_days[mask], extended_pred[mask])
        rows.append(
            {
                "sofa_stratum": label,
                "hempel_mae": hempel_mae,
                "extended_mae": extended_mae,
                "delta_mae": extended_mae - hempel_mae,
                "n": int(mask.sum()),
            }
        )
    table = pd.DataFrame(rows)
    root = Path(output_root)
    _save_table(table, root / "tables", "table8_sofa_stratified")
    _plot_stratified_mae(table, root / "figures" / "sofa_stratified_mae.png")
    return table


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
            ablation_by_los_class(
                joblib.load(h_path),
                joblib.load(e_path),
                X_h_test,
                X_e_test,
                y_class_test,
                ABLATION_ROOT / "tables",
            )
            return


def plot_sofa_ablation_summary(delta_df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot the master SOFA ablation summary.

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
    fig.suptitle("Impact of SOFA Severity Features on Model Performance")
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


def _save_table(df: pd.DataFrame, output_dir: str | Path, stem: str) -> None:
    """Save a table as CSV and LaTeX.

    Args:
        df: Table to save.
        output_dir: Output directory.
        stem: Filename stem.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    df.to_csv(directory / f"{stem}.csv", index=False)
    df.to_latex(directory / f"{stem}.tex", index=False)


def _plot_permutation_importance(importance_df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot top permutation importances with SOFA features highlighted.

    Args:
        importance_df: Permutation-importance table.
        output_path: Figure output path.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    top = importance_df.head(25).iloc[::-1]
    colors = ["#b22222" if is_sofa else "#4c78a8" for is_sofa in top["is_sofa_feature"]]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color=colors)
    ax.set_xlabel("Permutation importance (decrease in neg MAE)")
    ax.set_title("Permutation Importance with SOFA Features Highlighted")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_stratified_mae(table: pd.DataFrame, output_path: str | Path) -> None:
    """Plot grouped MAE bars by SOFA stratum.

    Args:
        table: Stratified MAE table.
        output_path: Figure output path.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    x = np.arange(len(table))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, table["hempel_mae"], width=width, label="Hempel", color="#4c78a8")
    ax.bar(x + width / 2, table["extended_mae"], width=width, label="Extended + SOFA", color="#59a14f")
    ax.set_xticks(x)
    ax.set_xticklabels(table["sofa_stratum"])
    ax.set_ylabel("MAE (days)")
    ax.set_title("MAE by SOFA Severity Stratum")
    ax.legend()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
