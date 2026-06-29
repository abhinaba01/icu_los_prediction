"""Master script for the extended ICU LOS model with occupancy features."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import MIMICLoader, load_config
from src.features.build_features import build_base_cohort_from_csv, build_extended_features, build_hempel_features
from src.models.stage1_classification import get_stage1_models, get_stage1_param_grids
from src.models.stage2_regression import get_stage2_models, get_stage2_param_grids
from src.models.trainer import ModelTrainer
from src.preprocessing.pipeline import PreprocessingPipeline
from src.visualization.correlation_matrix import plot_correlation_with_los
from src.visualization.feature_importance import plot_xgboost_feature_importance
from src.visualization.model_comparison import plot_stage1_comparison, plot_stage2_comparison


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
        handlers=[logging.FileHandler(log_dir / "extended_run.log"), logging.StreamHandler()],
    )


def load_or_build_hempel(loader: MIMICLoader, config: dict) -> pd.DataFrame:
    """Load existing Hempel features or build them.

    Args:
        loader: Configured MIMIC loader.
        config: Parsed project configuration.

    Returns:
        Hempel feature DataFrame.
    """
    path = PROJECT_ROOT / "data" / "processed" / "hempel_features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return build_hempel_features(loader, config)


def load_cohort(loader: MIMICLoader, config: dict) -> pd.DataFrame:
    """Load or build the base cohort for occupancy features.

    Args:
        loader: Configured MIMIC loader.
        config: Parsed project configuration.

    Returns:
        Base cohort DataFrame.
    """
    if loader.source == "postgresql":
        return loader.run_sql_extract("cohort")
    return build_base_cohort_from_csv(loader, config)


def save_table(df: pd.DataFrame, csv_path: Path, tex_path: Path) -> None:
    """Save a table as CSV and LaTeX.

    Args:
        df: Table to save.
        csv_path: CSV path.
        tex_path: LaTeX path.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_latex(tex_path, index=False)


def main() -> None:
    """Run the complete extended-model workflow."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    configure_logging(config)
    logger = logging.getLogger(__name__)
    random_state = config["project"]["random_state"]
    loader = MIMICLoader(config, PROJECT_ROOT)
    hempel_df = load_or_build_hempel(loader, config)
    cohort_df = load_cohort(loader, config)
    extended_df = build_extended_features(hempel_df, cohort_df, PROJECT_ROOT / "data" / "processed" / "extended_features.parquet")

    results_dir = PROJECT_ROOT / "results" / "extended"
    tables_dir = results_dir / "tables"
    split_path = PROJECT_ROOT / "results" / "split_indices.npz"
    preprocessor_for_split = PreprocessingPipeline(config)
    if split_path.exists():
        splits = preprocessor_for_split.load_split_indices(split_path)
    else:
        splits = preprocessor_for_split.split_indices(hempel_df, split_path)

    preprocessor = PreprocessingPipeline(config)
    X, y_class = preprocessor.get_X_y_classification(extended_df)
    _, y_reg = preprocessor.get_X_y_regression(extended_df)
    X_train = preprocessor.fit_transform(X.iloc[splits["train_idx"]])
    X_val = preprocessor.transform(X.iloc[splits["val_idx"]])
    X_test = preprocessor.transform(X.iloc[splits["test_idx"]])
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_class_train_full = pd.concat([y_class.iloc[splits["train_idx"]], y_class.iloc[splits["val_idx"]]], axis=0)
    y_reg_train_full = pd.concat([y_reg.iloc[splits["train_idx"]], y_reg.iloc[splits["val_idx"]]], axis=0)
    y_class_test = y_class.iloc[splits["test_idx"]].reset_index(drop=True)
    y_reg_test = y_reg.iloc[splits["test_idx"]].reset_index(drop=True)
    joblib.dump(preprocessor, results_dir / "models" / "preprocessor.pkl")

    trainer = ModelTrainer(
        results_dir,
        random_state=random_state,
        svm_max_train_samples=config["models"]["stage1"].get("svm", {}).get("max_train_samples", 5000),
        mlp_max_train_samples=config["models"]["stage1"].get("mlp", {}).get("max_train_samples", 10000),
    )
    stage1_models = trainer.train_stage1(
        X_train_full,
        y_class_train_full.reset_index(drop=True),
        get_stage1_models(random_state),
        get_stage1_param_grids(config),
        config["preprocessing"]["cv_folds"],
    )
    stage1_metrics = trainer.evaluate_stage1(stage1_models, X_test, y_class_test)
    save_table(stage1_metrics, tables_dir / "table4_stage1_extended.csv", tables_dir / "table4_stage1_extended.tex")
    stage2_models = trainer.train_stage2(
        X_train_full,
        y_reg_train_full.reset_index(drop=True),
        get_stage2_models(random_state),
        get_stage2_param_grids(config),
        config["preprocessing"]["cv_folds"],
    )
    stage2_metrics = trainer.evaluate_stage2(stage2_models, X_test, y_reg_test)
    save_table(stage2_metrics, tables_dir / "table5_stage2_extended.csv", tables_dir / "table5_stage2_extended.tex")

    plot_correlation_with_los(extended_df, output_path=results_dir / "figures" / "fig4_correlation.png")
    if "xgboost" in stage2_models:
        plot_xgboost_feature_importance(
            stage2_models["xgboost"],
            preprocessor.get_feature_names_out(),
            output_path=results_dir / "figures" / "fig5_importance.png",
            X_test=X_test,
            shap_output_path=results_dir / "figures" / "shap_summary.png",
        )
    plot_stage1_comparison(stage1_metrics, results_dir / "figures")
    plot_stage2_comparison(stage2_metrics, results_dir / "figures", trainer.stage2_predictions_)
    logger.info("Extended run complete. Best Stage 2 MAE: %.3f", stage2_metrics["MAE"].min())


if __name__ == "__main__":
    main()
