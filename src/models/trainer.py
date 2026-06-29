"""Training, evaluation, and persistence for Stage 1 and Stage 2 models."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_validate
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

from src.evaluation.metrics import classification_metrics, regression_metrics

LOGGER = logging.getLogger(__name__)


class ModelTrainer:
    """Trains, evaluates, and saves models for Stage 1 and Stage 2."""

    def __init__(
        self,
        output_dir: str | Path,
        random_state: int = 42,
        svm_max_train_samples: int | None = 5000,
        mlp_max_train_samples: int | None = 10000,
    ):
        """Initialize a trainer.

        Args:
            output_dir: Experiment output directory such as `results/hempel`.
            random_state: Random seed for CV splitters.
            svm_max_train_samples: Optional stratified training cap for RBF
                SVM. Full-cohort RBF SVM grid search is impractical at
                MIMIC-IV scale.
            mlp_max_train_samples: Optional training cap for MLP models.
        """
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.svm_max_train_samples = svm_max_train_samples
        self.mlp_max_train_samples = mlp_max_train_samples
        self.models_dir = self.output_dir / "models"
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir = self.output_dir / "figures"
        self.predictions_dir = self.tables_dir / "predictions"
        for directory in [self.models_dir, self.tables_dir, self.figures_dir, self.predictions_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        self.stage1_predictions_: dict[str, pd.DataFrame] = {}
        self.stage2_predictions_: dict[str, pd.DataFrame] = {}

    def train_stage1(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        models_dict: dict[str, object],
        param_grids: dict[str, dict[str, list[Any]]] | None = None,
        cv_folds: int = 5,
    ) -> dict[str, object]:
        """Train all Stage 1 classifiers with stratified CV grid search.

        Args:
            X_train: Processed training features.
            y_train: Training class labels.
            models_dict: Mapping of model names to estimators.
            param_grids: Optional parameter grids.
            cv_folds: Requested number of CV folds.

        Returns:
            Mapping of model names to fitted estimators.
        """
        fitted = {}
        for name, model in tqdm(models_dict.items(), desc="Stage 1 models"):
            X_model, y_model = self._maybe_subsample_classifier(name, X_train, y_train)
            X_model, y_model = self._maybe_smote(name, X_model, y_model)
            estimator = self._fit_grid_or_direct(
                name=name,
                model=model,
                X=X_model,
                y=y_model,
                param_grid=(param_grids or {}).get(name, {}),
                cv=self._classification_cv(y_model, cv_folds),
                scoring="f1_macro",
                fit_params=self._stage1_fit_params(name, y_model),
            )
            fitted[name] = estimator
            self._save_model(estimator, f"{name}_stage1.pkl")
            LOGGER.info("Completed Stage 1 model %s", name)
        return fitted

    def evaluate_stage1(self, fitted_models: dict[str, object], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Evaluate Stage 1 classifiers and save metrics and predictions.

        Args:
            fitted_models: Mapping of model names to fitted estimators.
            X_test: Processed test features.
            y_test: Test class labels.

        Returns:
            DataFrame of Stage 1 metrics.
        """
        rows = []
        for name, model in fitted_models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            metrics = classification_metrics(y_test.to_numpy(), y_pred, y_proba)
            row = {"model": name, "stage": "stage1", **metrics}
            rows.append(row)
            self._save_confusion_matrix(name, y_test.to_numpy(), y_pred)
            predictions = pd.DataFrame({"y_true": y_test.to_numpy(), "y_pred": y_pred})
            self.stage1_predictions_[name] = predictions
            predictions.to_csv(self.predictions_dir / f"{name}_stage1_predictions.csv", index=False)
        metrics_df = pd.DataFrame(rows)
        metrics_df.to_csv(self.tables_dir / "stage1_metrics.csv", index=False)
        return metrics_df

    def train_stage2(
        self,
        X_train: pd.DataFrame,
        y_train_log: pd.Series,
        models_dict: dict[str, object],
        param_grids: dict[str, dict[str, list[Any]]] | None = None,
        cv_folds: int = 5,
    ) -> dict[str, object]:
        """Train all Stage 2 regressors with K-fold CV grid search.

        Args:
            X_train: Processed training features.
            y_train_log: Training target `log(los_days + 1)`.
            models_dict: Mapping of model names to regressors.
            param_grids: Optional parameter grids.
            cv_folds: Requested number of CV folds.

        Returns:
            Mapping of model names to fitted estimators.
        """
        fitted = {}
        for name, model in tqdm(models_dict.items(), desc="Stage 2 models"):
            X_model, y_model = self._maybe_subsample_regressor(name, X_train, y_train_log)
            estimator = self._fit_grid_or_direct(
                name=name,
                model=model,
                X=X_model,
                y=y_model,
                param_grid=(param_grids or {}).get(name, {}),
                cv=self._regression_cv(len(y_model), cv_folds),
                scoring="neg_mean_absolute_error",
                fit_params={},
            )
            fitted[name] = estimator
            self._save_model(estimator, f"{name}_stage2.pkl")
            LOGGER.info("Completed Stage 2 model %s", name)
        return fitted

    def evaluate_stage2(self, fitted_models: dict[str, object], X_test: pd.DataFrame, y_test_log: pd.Series) -> pd.DataFrame:
        """Evaluate Stage 2 regressors in raw LOS days.

        Args:
            fitted_models: Mapping of model names to fitted estimators.
            X_test: Processed test features.
            y_test_log: Test target `log(los_days + 1)`.

        Returns:
            DataFrame of Stage 2 metrics.
        """
        rows = []
        y_true_days = np.expm1(y_test_log.to_numpy())
        for name, model in fitted_models.items():
            pred_log = model.predict(X_test)
            pred_days = np.clip(np.expm1(pred_log), 0.0, None)
            metrics = regression_metrics(y_true_days, pred_days)
            row = {"model": name, "stage": "stage2", **metrics}
            rows.append(row)
            predictions = pd.DataFrame({"y_true_days": y_true_days, "y_pred_days": pred_days, "y_true_log": y_test_log.to_numpy(), "y_pred_log": pred_log})
            self.stage2_predictions_[name] = predictions
            predictions.to_csv(self.predictions_dir / f"{name}_stage2_predictions.csv", index=False)
        metrics_df = pd.DataFrame(rows)
        metrics_df.to_csv(self.tables_dir / "stage2_metrics.csv", index=False)
        return metrics_df

    def cross_validate_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models_dict: dict[str, object],
        stage: str = "classification",
        cv_folds: int = 5,
    ) -> pd.DataFrame:
        """Run 5-fold CV on the full dataset for reporting.

        Args:
            X: Processed feature matrix.
            y: Target vector.
            models_dict: Mapping of model names to estimators.
            stage: Either `classification` or `regression`.
            cv_folds: Requested number of folds.

        Returns:
            DataFrame of mean and standard deviation CV scores.
        """
        rows = []
        for name, model in models_dict.items():
            if stage == "classification":
                cv = self._classification_cv(y, cv_folds)
                scoring = ["accuracy", "f1_macro"]
            else:
                cv = self._regression_cv(len(y), cv_folds)
                scoring = ["neg_mean_absolute_error", "r2"]
            if cv is None:
                continue
            result = cross_validate(clone(model), X, y, cv=cv, scoring=scoring, n_jobs=-1)
            row = {"model": name, "stage": stage}
            for key, values in result.items():
                if key.startswith("test_"):
                    row[f"{key}_mean"] = float(np.mean(values))
                    row[f"{key}_std"] = float(np.std(values))
            rows.append(row)
        return pd.DataFrame(rows)

    def _fit_grid_or_direct(
        self,
        name: str,
        model: object,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: dict[str, list[Any]],
        cv: StratifiedKFold | KFold | None,
        scoring: str,
        fit_params: dict[str, Any],
    ) -> object:
        """Fit a model with GridSearchCV when possible, otherwise directly.

        Args:
            name: Model name.
            model: Unfitted estimator.
            X: Training features.
            y: Training target.
            param_grid: Parameter grid.
            cv: CV splitter or None.
            scoring: Scoring metric for grid search.
            fit_params: Optional fit parameters.

        Returns:
            Fitted estimator.
        """
        estimator = clone(model)
        if cv is None or not param_grid:
            LOGGER.info("Fitting %s directly", name)
            return estimator.fit(X, y, **fit_params)
        LOGGER.info("Running grid search for %s", name)
        search = GridSearchCV(estimator, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=-1, error_score=np.nan)
        search.fit(X, y, **fit_params)
        LOGGER.info("Best parameters for %s: %s", name, search.best_params_)
        return search.best_estimator_

    def _classification_cv(self, y: pd.Series, cv_folds: int) -> StratifiedKFold | None:
        """Build a valid stratified CV splitter for classification.

        Args:
            y: Class labels.
            cv_folds: Requested folds.

        Returns:
            StratifiedKFold or None when data are too small.
        """
        counts = pd.Series(y).value_counts()
        if counts.empty or counts.min() < 2:
            return None
        n_splits = min(cv_folds, int(counts.min()))
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

    def _regression_cv(self, n_samples: int, cv_folds: int) -> KFold | None:
        """Build a valid K-fold splitter for regression.

        Args:
            n_samples: Number of training samples.
            cv_folds: Requested folds.

        Returns:
            KFold or None when data are too small.
        """
        if n_samples < 2:
            return None
        n_splits = min(cv_folds, n_samples)
        return KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

    def _stage1_fit_params(self, name: str, y: pd.Series) -> dict[str, Any]:
        """Return optional fit parameters for Stage 1 estimators.

        Args:
            name: Model name.
            y: Training labels.

        Returns:
            Fit-parameter dictionary.
        """
        if name == "xgboost":
            return {"sample_weight": compute_sample_weight(class_weight="balanced", y=y)}
        return {}

    def _maybe_smote(self, name: str, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE to MLP training data when feasible.

        Args:
            name: Model name.
            X: Training features.
            y: Training labels.

        Returns:
            Possibly resampled feature matrix and labels.
        """
        if name != "mlp" or pd.Series(y).value_counts().min() < 2:
            return X, y
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            LOGGER.warning("imbalanced-learn is not installed; MLP uses original class balance")
            return X, y
        smote = SMOTE(random_state=self.random_state, k_neighbors=1)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)

    def _maybe_subsample_classifier(self, name: str, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Subsample expensive classification models for tractable grid search.

        Args:
            name: Model name.
            X: Training feature matrix.
            y: Training labels.

        Returns:
            Original or stratified-subsampled feature matrix and labels.
        """
        max_samples = self._max_train_samples_for_model(name)
        if max_samples is None or len(X) <= max_samples:
            return X, y
        from sklearn.model_selection import train_test_split

        _, X_sub, _, y_sub = train_test_split(
            X,
            y,
            test_size=max_samples,
            random_state=self.random_state,
            stratify=y,
        )
        LOGGER.info("Subsampled %s training data from %s to %s rows", name, len(X), len(X_sub))
        return X_sub.reset_index(drop=True), y_sub.reset_index(drop=True)

    def _maybe_subsample_regressor(self, name: str, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Subsample expensive regression models for tractable grid search.

        Args:
            name: Model name.
            X: Training feature matrix.
            y: Training target.

        Returns:
            Original or subsampled feature matrix and target.
        """
        max_samples = self._max_train_samples_for_model(name)
        if name != "mlp" or max_samples is None or len(X) <= max_samples:
            return X, y
        rng = np.random.RandomState(self.random_state)
        positions = rng.choice(len(X), size=max_samples, replace=False)
        LOGGER.info("Subsampled %s training data from %s to %s rows", name, len(X), len(positions))
        return X.iloc[positions].reset_index(drop=True), y.iloc[positions].reset_index(drop=True)

    def _max_train_samples_for_model(self, name: str) -> int | None:
        """Return the configured training cap for an expensive model.

        Args:
            name: Model name.

        Returns:
            Maximum training samples, or None when uncapped.
        """
        if name == "svm":
            return self.svm_max_train_samples
        if name == "mlp":
            return self.mlp_max_train_samples
        return None

    def _save_model(self, model: object, filename: str) -> None:
        """Persist a fitted model with joblib.

        Args:
            model: Fitted estimator.
            filename: Output filename under the models directory.
        """
        joblib.dump(model, self.models_dir / filename)

    def _save_confusion_matrix(self, name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Save a confusion-matrix figure for a classifier.

        Args:
            name: Model name.
            y_true: True labels.
            y_pred: Predicted labels.
        """
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        labels = [0, 1, 2]
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        display = ConfusionMatrixDisplay(matrix, display_labels=["Short", "Medium", "Long"])
        fig, ax = plt.subplots(figsize=(6, 5))
        display.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name} Confusion Matrix")
        out_dir = self.figures_dir / "confusion_matrices"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{name}_confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def paired_error_ttest(y_true: np.ndarray, baseline_pred: np.ndarray, candidate_pred: np.ndarray) -> tuple[float, float]:
    """Run a paired t-test on absolute prediction errors.

    Args:
        y_true: Ground-truth values.
        baseline_pred: Baseline predictions.
        candidate_pred: Candidate predictions.

    Returns:
        Tuple of t-statistic and p-value.
    """
    baseline_errors = np.abs(np.asarray(y_true) - np.asarray(baseline_pred))
    candidate_errors = np.abs(np.asarray(y_true) - np.asarray(candidate_pred))
    result = stats.ttest_rel(baseline_errors, candidate_errors)
    return float(result.statistic), float(result.pvalue)
