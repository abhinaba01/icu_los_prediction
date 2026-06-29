"""Smoke tests for model training and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.models.trainer import ModelTrainer


def test_model_trainer_smoke_classification_and_regression(tmp_path) -> None:
    """Fit and evaluate tiny classification and regression models."""
    X = pd.DataFrame({"age": [20, 30, 40, 50, 60, 70, 80, 90, 45], "lab": [1, 2, 1, 3, 2, 5, 4, 6, 3]})
    y_class = pd.Series([0, 0, 1, 1, 2, 2, 0, 1, 2])
    y_reg = pd.Series(np.log1p([1.0, 1.5, 3.0, 4.0, 8.0, 9.0, 2.0, 5.0, 7.0]))
    trainer = ModelTrainer(tmp_path)
    classifiers = {"logistic_regression": LogisticRegression(max_iter=200)}
    fitted_classifiers = trainer.train_stage1(X, y_class, classifiers, param_grids={"logistic_regression": {}}, cv_folds=2)
    class_metrics = trainer.evaluate_stage1(fitted_classifiers, X, y_class)
    regressors = {"linear_regression": LinearRegression()}
    fitted_regressors = trainer.train_stage2(X, y_reg, regressors, param_grids={"linear_regression": {}}, cv_folds=2)
    reg_metrics = trainer.evaluate_stage2(fitted_regressors, X, y_reg)
    assert class_metrics.loc[0, "model"] == "logistic_regression"
    assert reg_metrics.loc[0, "model"] == "linear_regression"
