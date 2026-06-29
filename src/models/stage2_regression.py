"""Stage 2 LOS regression model definitions."""

from __future__ import annotations

import logging
from typing import Any

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor

LOGGER = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - depends on optional runtime package
    XGBRegressor = None


def get_stage2_models(random_state: int = 42) -> dict[str, object]:
    """Create Stage 2 regression model instances.

    Args:
        random_state: Random seed used by stochastic estimators.

    Returns:
        Dictionary of model names to sklearn-compatible regressors.
    """
    models: dict[str, object] = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "huber": HuberRegressor(epsilon=1.35, alpha=0.0001),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=random_state,
        ),
        "mlp": MLPRegressor(
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
    if XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        LOGGER.warning("xgboost is not installed; Stage 2 XGBoost model is unavailable")
    return models


def get_stage2_param_grids(config: dict[str, Any]) -> dict[str, dict[str, list[Any]]]:
    """Build Stage 2 GridSearchCV parameter grids from config.

    Args:
        config: Parsed project configuration.

    Returns:
        Mapping of model name to parameter grid.
    """
    cfg = config["models"]["stage2"]
    grids: dict[str, dict[str, list[Any]]] = {
        "linear_regression": {},
        "ridge": {"alpha": cfg["ridge"]["alpha"]},
        "huber": {
            "epsilon": cfg.get("huber", {}).get("epsilon", [1.35]),
            "alpha": cfg.get("huber", {}).get("alpha", [0.0001]),
        },
        "random_forest": {
            "n_estimators": cfg["random_forest"]["n_estimators"],
            "max_depth": cfg["random_forest"]["max_depth"],
            "min_samples_leaf": cfg["random_forest"]["min_samples_leaf"],
        },
        "gradient_boosting": {
            "n_estimators": cfg["gradient_boosting"]["n_estimators"],
            "max_depth": cfg["gradient_boosting"]["max_depth"],
            "learning_rate": cfg["gradient_boosting"]["learning_rate"],
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


STAGE2_MODELS = get_stage2_models()
