"""Sklearn-compatible preprocessing for ICU LOS feature matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

NON_FEATURE_COLUMNS = {
    "stay_id",
    "subject_id",
    "hadm_id",
    "los_days",
    "los_hours",
    "los_log",
    "los_class",
    "intime",
    "outtime",
    "first_careunit",
    "last_careunit",
    "gender",
    "admission_type",
    "admission_location",
}

CONTINUOUS_FEATURES = {
    "age",
    "heart_rate_mean",
    "spo2_mean",
    "resp_rate_mean",
    "temperature_mean",
    "gcs_eye_mean",
    "gcs_verbal_mean",
    "gcs_motor_mean",
    "anion_gap_mean",
    "bicarbonate_mean",
    "chloride_mean",
    "creatinine_mean",
    "glucose_mean",
    "sodium_mean",
    "magnesium_mean",
    "potassium_mean",
    "phosphate_mean",
    "bun_mean",
    "hematocrit_mean",
    "hemoglobin_mean",
    "mch_mean",
    "mchc_mean",
    "mcv_mean",
    "rdw_mean",
    "rbc_mean",
    "wbc_mean",
    "platelets_mean",
    "concurrent_patients",
    "occupancy_rate",
    "occupancy_percentile",
    "sofa_total", 
    "sofa_resp", 
    "sofa_cardio",
    "sofa_hepatic", 
    "sofa_coag", 
    "sofa_renal",
    "sofa_neuro", 
    "urine_24h", 
    "bilirubin_max"
}


class PreprocessingPipeline:
    """Sklearn-compatible preprocessing pipeline for LOS features."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the preprocessing pipeline.

        Args:
            config: Parsed project configuration.
        """
        self.config = config
        self.random_state = config["project"].get("random_state", 42)
        self.preprocessing_config = config["preprocessing"]
        self.continuous_features_: list[str] = []
        self.categorical_features_: list[str] = []
        self.feature_names_in_: list[str] = []
        self.feature_names_out_: list[str] = []
        self.transformer_: ColumnTransformer | None = None

    def get_X_y_classification(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Return feature matrix and LOS class labels.

        Args:
            df: Feature matrix with `los_class`.

        Returns:
            Tuple of X DataFrame and y Series.

        Raises:
            KeyError: If `los_class` is missing.
        """
        if "los_class" not in df.columns:
            raise KeyError("los_class is required for classification")
        return self._feature_matrix(df), df["los_class"].astype(int)

    def get_X_y_regression(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Return feature matrix and log-transformed LOS target.

        Args:
            df: Feature matrix with `los_log`.

        Returns:
            Tuple of X DataFrame and y Series.

        Raises:
            KeyError: If `los_log` is missing.
        """
        if "los_log" not in df.columns:
            raise KeyError("los_log is required for regression")
        return self._feature_matrix(df), df["los_log"].astype(float)

    def split_indices(self, df: pd.DataFrame, output_path: str | Path | None = None) -> dict[str, np.ndarray]:
        """Create stratified train, validation, and test row indices.

        Args:
            df: Feature DataFrame containing `los_class`.
            output_path: Optional `.npz` path for saving split indices.

        Returns:
            Dictionary with `train_idx`, `val_idx`, and `test_idx`.
        """
        indices = np.arange(len(df))
        y = df["los_class"].astype(int)
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=self.preprocessing_config["test_size"],
            random_state=self.random_state,
            stratify=y,
        )
        val_fraction = self.preprocessing_config["val_size"]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_fraction,
            random_state=self.random_state,
            stratify=y.iloc[train_val_idx],
        )
        splits = {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **splits)
        return splits

    def load_split_indices(self, split_path: str | Path) -> dict[str, np.ndarray]:
        """Load train, validation, and test row indices.

        Args:
            split_path: Path to a `.npz` split file.

        Returns:
            Dictionary with split arrays.

        Raises:
            FileNotFoundError: If the split file does not exist.
        """
        path = Path(split_path)
        if not path.exists():
            raise FileNotFoundError(f"Split indices not found: {path}")
        loaded = np.load(path)
        return {name: loaded[name] for name in ["train_idx", "val_idx", "test_idx"]}

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessing on training data and transform it.

        Args:
            X_train: Raw training feature matrix.

        Returns:
            Processed training DataFrame.
        """
        self.feature_names_in_ = list(X_train.columns)
        self.continuous_features_, self.categorical_features_ = self._split_columns(X_train)
        self.transformer_ = self._build_transformer()
        processed = self.transformer_.fit_transform(self._coerce_features(X_train))
        self.feature_names_out_ = self.continuous_features_ + self.categorical_features_
        return pd.DataFrame(processed, columns=self.feature_names_out_, index=X_train.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using training-fitted imputers and scalers.

        Args:
            X: Raw feature matrix.

        Returns:
            Processed feature DataFrame.

        Raises:
            RuntimeError: If `fit_transform` has not been called.
        """
        if self.transformer_ is None:
            raise RuntimeError("PreprocessingPipeline must be fitted before transform")
        aligned = X.copy()
        for column in self.feature_names_in_:
            if column not in aligned.columns:
                aligned[column] = np.nan
        aligned = aligned[self.feature_names_in_]
        processed = self.transformer_.transform(self._coerce_features(aligned))
        return pd.DataFrame(processed, columns=self.feature_names_out_, index=X.index)

    def fit_transform_splits(
        self,
        df: pd.DataFrame,
        split_indices: dict[str, np.ndarray],
        target: str = "classification",
    ) -> dict[str, Any]:
        """Fit preprocessing on the training split and transform all splits.

        Args:
            df: Full feature DataFrame.
            split_indices: Split index dictionary.
            target: Either `classification` or `regression`.

        Returns:
            Dictionary containing processed split matrices and targets.

        Raises:
            ValueError: If target is unsupported.
        """
        if target == "classification":
            X, y = self.get_X_y_classification(df)
        elif target == "regression":
            X, y = self.get_X_y_regression(df)
        else:
            raise ValueError("target must be 'classification' or 'regression'")

        train_idx = split_indices["train_idx"]
        val_idx = split_indices["val_idx"]
        test_idx = split_indices["test_idx"]
        X_train = self.fit_transform(X.iloc[train_idx])
        return {
            "X_train": X_train,
            "X_val": self.transform(X.iloc[val_idx]),
            "X_test": self.transform(X.iloc[test_idx]),
            "y_train": y.iloc[train_idx].reset_index(drop=True),
            "y_val": y.iloc[val_idx].reset_index(drop=True),
            "y_test": y.iloc[test_idx].reset_index(drop=True),
        }

    def get_feature_names_out(self) -> list[str]:
        """Return feature names after preprocessing.

        Returns:
            List of output feature names.
        """
        return list(self.feature_names_out_)

    def _feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract model feature columns from a full feature frame.

        Args:
            df: Full feature DataFrame.

        Returns:
            Raw model feature matrix.
        """
        feature_columns = [column for column in df.columns if column not in NON_FEATURE_COLUMNS]
        return df[feature_columns].copy()

    def _split_columns(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Split feature columns into continuous and categorical groups.

        Args:
            X: Raw feature matrix.

        Returns:
            Tuple of continuous and categorical feature-name lists.
        """
        continuous = [column for column in X.columns if column in CONTINUOUS_FEATURES]
        categorical = [column for column in X.columns if column not in continuous]
        return continuous, categorical

    def _build_transformer(self) -> ColumnTransformer:
        """Build the sklearn column transformer.

        Returns:
            Configured ColumnTransformer.
        """
        scaling = self.preprocessing_config.get("scaling", "standard")
        if scaling == "standard":
            scaler = StandardScaler()
        elif scaling == "minmax":
            scaler = MinMaxScaler()
        elif scaling == "none":
            scaler = "passthrough"
        else:
            raise ValueError(f"Unsupported scaling option: {scaling}")

        continuous_steps = [("imputer", SimpleImputer(strategy=self.preprocessing_config["imputation_strategy"], keep_empty_features=True))]
        if scaler != "passthrough":
            continuous_steps.append(("scaler", scaler))
        continuous_pipe = Pipeline(continuous_steps)
        categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True))])
        return ColumnTransformer(
            transformers=[
                ("continuous", continuous_pipe, self.continuous_features_),
                ("categorical", categorical_pipe, self.categorical_features_),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def _coerce_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Coerce feature matrix columns to numeric values.

        Args:
            X: Raw or aligned feature matrix.

        Returns:
            Numeric feature matrix.
        """
        output = X.copy()
        for column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
        return output
