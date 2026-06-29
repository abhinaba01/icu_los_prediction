"""Validation checks for cohorts, features, and preprocessing outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def validate_feature_extraction(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Run feature extraction sanity checks and return missing-rate table.

    Args:
        feature_df: Feature matrix with outcomes.

    Returns:
        DataFrame containing missing rates per column.
    """
    n_rows = len(feature_df)
    if 40000 <= n_rows <= 70000:
        LOGGER.info("Cohort size is in full MIMIC-IV expected range: %s", n_rows)
    elif 2000 <= n_rows <= 5000:
        LOGGER.info("Cohort size resembles a demo/sample dataset: %s", n_rows)
    else:
        LOGGER.warning("Cohort size outside expected ranges: %s", n_rows)

    if "los_days" in feature_df:
        los = feature_df["los_days"].dropna()
        LOGGER.info("LOS median %.2f days; mean %.2f days", los.median(), los.mean())
    if "los_class" in feature_df:
        LOGGER.info("LOS class balance:\n%s", feature_df["los_class"].value_counts(normalize=True))

    missing = feature_df.isna().mean().sort_values(ascending=False).rename("missing_rate").reset_index()
    missing = missing.rename(columns={"index": "feature"})
    high_missing = missing[missing["missing_rate"] > 0.50]
    if not high_missing.empty:
        LOGGER.warning("Features over 50%% missing:\n%s", high_missing)
    return missing


def validate_preprocessing(X_train: pd.DataFrame, X_test: pd.DataFrame, test_size: float) -> None:
    """Validate post-imputation matrices and split proportions.

    Args:
        X_train: Processed training feature matrix.
        X_test: Processed test feature matrix.
        test_size: Expected test proportion.

    Raises:
        ValueError: If NaNs remain or the split proportion is far from config.
    """
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("NaN values remain after preprocessing")
    observed = len(X_test) / (len(X_train) + len(X_test))
    if not np.isclose(observed, test_size, atol=0.05):
        raise ValueError(f"Unexpected test split proportion: {observed:.3f}")
    LOGGER.info("Preprocessing validation passed; observed test size %.3f", observed)


def save_table(df: pd.DataFrame, csv_path: str | Path, tex_path: str | Path | None = None) -> None:
    """Save a table as CSV and optionally LaTeX.

    Args:
        df: Table to save.
        csv_path: CSV output path.
        tex_path: Optional LaTeX output path.
    """
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_file, index=False)
    if tex_path is not None:
        tex_file = Path(tex_path)
        tex_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_latex(tex_file, index=False)
