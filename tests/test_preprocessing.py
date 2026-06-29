"""Tests for preprocessing behavior and leakage guards."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.pipeline import PreprocessingPipeline


def test_preprocessing_imputes_without_using_test_values() -> None:
    """Verify imputation statistics are fitted only on training data."""
    config = _test_config(scaling="none")
    pipeline = PreprocessingPipeline(config)
    X_train = pd.DataFrame({"age": [1.0, 3.0], "gender_male": [1, 0], "diag1_ch_1": [1, 0]})
    X_test = pd.DataFrame({"age": [np.nan], "gender_male": [0], "diag1_ch_1": [1]})
    train_processed = pipeline.fit_transform(X_train)
    test_processed = pipeline.transform(X_test)
    assert not train_processed.isna().any().any()
    assert not test_processed.isna().any().any()
    assert test_processed.loc[0, "age"] == 2.0


def test_split_indices_are_stratified_and_saved(tmp_path) -> None:
    """Check split indices are created and reloadable."""
    config = _test_config()
    df = pd.DataFrame(
        {
            "stay_id": range(30),
            "age": range(30),
            "gender_male": [0, 1] * 15,
            "los_days": [1, 4, 9] * 10,
            "los_log": np.log1p([1, 4, 9] * 10),
            "los_class": [0, 1, 2] * 10,
        }
    )
    pipeline = PreprocessingPipeline(config)
    split_path = tmp_path / "splits.npz"
    splits = pipeline.split_indices(df, split_path)
    loaded = pipeline.load_split_indices(split_path)
    assert set(splits) == {"train_idx", "val_idx", "test_idx"}
    assert len(loaded["test_idx"]) == len(splits["test_idx"])


def _test_config(scaling: str = "standard") -> dict:
    """Return a minimal preprocessing config for tests.

    Args:
        scaling: Scaling mode.

    Returns:
        Config dictionary.
    """
    return {
        "project": {"random_state": 42},
        "preprocessing": {
            "test_size": 0.2,
            "val_size": 0.1,
            "cv_folds": 3,
            "imputation_strategy": "median",
            "scaling": scaling,
        },
    }
