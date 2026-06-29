"""Stage 1 LOS classification model definitions."""

from __future__ import annotations

import logging
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

LOGGER = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - depends on optional runtime package
    XGBClassifier = None


def get_stage1_models(random_state: int = 42) -> dict[str, object]:
    """Create Stage 1 classifier instances.

    Args:
        random_state: Random seed used by stochastic estimators.

    Returns:
        Dictionary of model names to sklearn-compatible estimators.
    """
    models: dict[str, object] = {
        "logistic_regression": LogisticRegression(
            C=1.0,
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "svm": SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=random_state,
            class_weight="balanced",
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=random_state,
        ),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        LOGGER.warning("xgboost is not installed; Stage 1 XGBoost model is unavailable")
    return models


def get_stage1_param_grids(config: dict[str, Any]) -> dict[str, dict[str, list[Any]]]:
    """Build Stage 1 GridSearchCV parameter grids from config.

    Args:
        config: Parsed project configuration.

    Returns:
        Mapping of model name to parameter grid.
    """
    cfg = config["models"]["stage1"]
    grids: dict[str, dict[str, list[Any]]] = {
        "logistic_regression": {
            "C": cfg["logistic_regression"]["C"],
        },
        "random_forest": {
            "n_estimators": cfg["random_forest"]["n_estimators"],
            "max_depth": cfg["random_forest"]["max_depth"],
            "min_samples_leaf": cfg["random_forest"]["min_samples_leaf"],
        },
        "svm": {
            "C": cfg["svm"]["C"],
            "kernel": cfg["svm"]["kernel"],
            "gamma": cfg["svm"]["gamma"],
        },
        "mlp": {
            "hidden_layer_sizes": [tuple(value) for value in cfg["mlp"]["hidden_layer_sizes"]],
            "activation": cfg["mlp"]["activation"],
        },
    }
    if "xgboost" in cfg:
        grids["xgboost"] = {
            "n_estimators": cfg["xgboost"]["n_estimators"],
            "max_depth": cfg["xgboost"]["max_depth"],
            "learning_rate": cfg["xgboost"]["learning_rate"],
            "subsample": cfg["xgboost"]["subsample"],
        }
    return grids


STAGE1_MODELS = get_stage1_models()
