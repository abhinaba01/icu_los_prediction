"""Ablation analyses for ICU occupancy features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import KFold, learning_curve

from src.evaluation.significance import diebold_mariano_test

LOGGER = logging.getLogger(__name__)

OCCUPANCY_COLUMNS = ["concurrent_patients", "occupancy_rate", "occupancy_percentile"]


def compute_delta_table(
    hempel_metrics: pd.DataFrame,
    extended_metrics: pd.DataFrame,
    output_dir: str | Path = "results/ablation/tables",
) -> pd.DataFrame:
    """Compute metric deltas between Hempel and extended models.

    Args:
        hempel_metrics: Baseline metrics DataFrame.
        extended_metrics: Extended metrics DataFrame.
        output_dir: Directory where CSV and LaTeX tables are saved.

    Returns:
        Delta metrics DataFrame.
    """
    rows: list[dict[str, Any]] = []
    common_keys = ["model", "stage"]
    merged = hempel_metrics.merge(extended_metrics, on=common_keys, suffixes=("_hempel", "_extended"))
    metric_names = [
        column.replace("_hempel", "")
        for column in merged.columns
        if column.endswith("_hempel") and pd.api.types.is_numeric_dtype(merged[column])
    ]
    for _, row in merged.iterrows():
        for metric in metric_names:
            baseline = row[f"{metric}_hempel"]
            candidate = row[f"{metric}_extended"]
            if pd.isna(baseline) or pd.isna(candidate):
                continue
            delta = candidate - baseline
            delta_pct = delta / abs(baseline) * 100.0 if baseline != 0 else np.nan
            rows.append(
                {
                    "model": row["model"],
                    "stage": row["stage"],
                    "metric": metric,
                    "hempel_value": baseline,
                    "extended_value": candidate,
                    "delta": delta,
                    "delta_pct": delta_pct,
                    "significant": False,
                }
            )
    delta_df = pd.DataFrame(rows)
    _save_table(delta_df, output_dir, "delta_metrics", "table6_delta_metrics")
    return delta_df


def test_significance(
    hempel_preds: np.ndarray,
    extended_preds: np.ndarray,
    y_true: np.ndarray,
    output_dir: str | Path = "results/ablation/tables",
    model_name: str = "model",
    n_models: int = 1,
) -> dict[str, Any]:
    """Test whether baseline and extended regression errors differ.

    Args:
        hempel_preds: Baseline predictions in LOS days.
        extended_preds: Extended predictions in LOS days.
        y_true: Ground truth LOS in days.
        output_dir: Directory where significance table is saved.
        model_name: Model label for saved output.
        n_models: Number of model comparisons for Bonferroni correction.

    Returns:
        Dictionary with test statistics, p-values, and conclusion.
    """
    y = np.asarray(y_true, dtype=float)
    hempel_errors = np.abs(y - np.asarray(hempel_preds, dtype=float))
    extended_errors = np.abs(y - np.asarray(extended_preds, dtype=float))
    t_stat, t_p = stats.ttest_rel(hempel_errors, extended_errors, nan_policy="omit")
    try:
        w_stat, w_p = stats.wilcoxon(hempel_errors, extended_errors)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    dm = diebold_mariano_test(hempel_errors, extended_errors)
    corrected_alpha = 0.05 / max(int(n_models), 1)
    conclusion = (
        "extended_improved"
        if np.nanmean(extended_errors) < np.nanmean(hempel_errors) and np.nan_to_num(w_p, nan=1.0) < corrected_alpha
        else "not_significant"
    )
    result = {
        "model": model_name,
        "paired_t_stat": float(t_stat),
        "paired_t_p": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
        **dm,
        "corrected_alpha": corrected_alpha,
        "conclusion": conclusion,
    }
    table = pd.DataFrame([result])
    _save_table(table, output_dir, "significance_tests", "table7_significance")
    return result


def ablation_by_los_class(
    hempel_model: object,
    extended_model: object,
    X_test_hempel: pd.DataFrame,
    X_test_extended: pd.DataFrame,
    y_test: np.ndarray,
    output_dir: str | Path = "results/ablation/tables",
) -> pd.DataFrame:
    """Compare Stage 1 F1 deltas by LOS class.

    Args:
        hempel_model: Fitted baseline classifier.
        extended_model: Fitted extended classifier.
        X_test_hempel: Baseline test features.
        X_test_extended: Extended test features.
        y_test: True class labels.
        output_dir: Directory where output table is saved.

    Returns:
        Class-specific ablation DataFrame.
    """
    y = np.asarray(y_test)
    pred_h = hempel_model.predict(X_test_hempel)
    pred_e = extended_model.predict(X_test_extended)
    labels = {0: "Short", 1: "Medium", 2: "Long"}
    rows = []
    for class_id, label in labels.items():
        mask = y == class_id
        if not mask.any():
            continue
        f1_h = f1_score(y[mask], pred_h[mask], labels=[class_id], average="macro", zero_division=0)
        f1_e = f1_score(y[mask], pred_e[mask], labels=[class_id], average="macro", zero_division=0)
        p_value = _paired_correctness_sign_test(y[mask], pred_h[mask], pred_e[mask])
        rows.append({"class": label, "hempel_f1": f1_h, "extended_f1": f1_e, "delta_f1": f1_e - f1_h, "p_value": p_value})
    output = pd.DataFrame(rows)
    _save_table(output, output_dir, "class_specific_ablation", "class_specific_ablation")
    return output


def permutation_importance_occupancy(
    model: object,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_repeats: int = 30,
    output_path: str | Path = "results/ablation/figures/permutation_importance.png",
) -> dict[str, Any]:
    """Estimate permutation importance for occupancy columns.

    Args:
        model: Fitted estimator.
        X_test: Test feature matrix.
        y_test: Test target vector in the estimator's target scale.
        n_repeats: Number of permutation repeats.
        output_path: Figure output path.

    Returns:
        Dictionary containing full and occupancy-only importance tables.
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
        }
    ).sort_values("importance_mean", ascending=False)
    occupancy_df = importance_df[importance_df["feature"].isin(OCCUPANCY_COLUMNS)].copy()
    _plot_permutation_importance(importance_df, output_path)
    return {"all": importance_df, "occupancy": occupancy_df}


def plot_learning_curves_ablation(
    hempel_model: object,
    extended_model: object,
    X_hempel: pd.DataFrame,
    X_extended: pd.DataFrame,
    y: np.ndarray,
    output_path: str | Path = "results/ablation/figures/learning_curves.png",
) -> None:
    """Plot learning curves for baseline and extended Stage 2 models.

    Args:
        hempel_model: Baseline regressor.
        extended_model: Extended regressor.
        X_hempel: Baseline feature matrix.
        X_extended: Extended feature matrix.
        y: Regression target in model scale.
        output_path: Figure output path.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes = np.linspace(0.1, 1.0, 10)
    h_sizes, h_train, h_val = learning_curve(hempel_model, X_hempel, y, cv=cv, train_sizes=train_sizes, scoring="neg_mean_absolute_error")
    _, e_train, e_val = learning_curve(extended_model, X_extended, y, cv=cv, train_sizes=train_sizes, scoring="neg_mean_absolute_error")
    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_curve(ax, h_sizes, -h_val, "Hempel", "-")
    _plot_curve(ax, h_sizes, -e_val, "Extended", "--")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Validation MAE (log LOS)")
    ax.set_title("Learning Curves: Hempel vs Extended")
    ax.legend()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def shap_interaction_occupancy(
    xgb_extended: object,
    X_test: pd.DataFrame,
    occupancy_col: str = "concurrent_patients",
    output_dir: str | Path = "results/ablation/figures/shap_interactions",
) -> pd.DataFrame:
    """Analyze SHAP interaction values involving an occupancy column.

    Args:
        xgb_extended: Fitted XGBoost model.
        X_test: Extended test feature matrix.
        occupancy_col: Occupancy feature to analyze.
        output_dir: Directory for interaction scatter plots.

    Returns:
        DataFrame of top interactions by mean absolute SHAP interaction.
    """
    if occupancy_col not in X_test.columns:
        raise KeyError(f"{occupancy_col} is not present in X_test")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        LOGGER.warning("shap is not installed; skipping SHAP interaction analysis")
        return pd.DataFrame()

    explainer = shap.TreeExplainer(xgb_extended)
    interactions = explainer.shap_interaction_values(X_test)
    occ_idx = list(X_test.columns).index(occupancy_col)
    occ_interactions = np.asarray(interactions)[:, occ_idx, :]
    scores = np.abs(occ_interactions).mean(axis=0)
    rows = pd.DataFrame({"feature": X_test.columns, "mean_abs_interaction": scores})
    rows = rows[rows["feature"].ne(occupancy_col)].sort_values("mean_abs_interaction", ascending=False).head(5)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for feature in rows["feature"]:
        feat_idx = list(X_test.columns).index(feature)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X_test[feature], occ_interactions[:, feat_idx], s=12, alpha=0.5)
        ax.set_xlabel(feature)
        ax.set_ylabel(f"SHAP interaction with {occupancy_col}")
        ax.set_title(f"{occupancy_col} x {feature}")
        fig.savefig(out / f"{occupancy_col}_x_{feature}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    return rows


def stratified_analysis_by_occupancy(
    extended_model: object,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    hempel_model: object,
    X_test_hempel: pd.DataFrame,
    output_dir: str | Path = "results/ablation",
) -> pd.DataFrame:
    """Compare MAE by low, medium, and high occupancy strata.

    Args:
        extended_model: Fitted extended Stage 2 model.
        X_test: Extended test features.
        y_test: True LOS target in log scale.
        hempel_model: Fitted baseline Stage 2 model.
        X_test_hempel: Baseline test features.
        output_dir: Ablation output root.

    Returns:
        Stratified performance DataFrame.
    """
    if "concurrent_patients" not in X_test.columns:
        raise KeyError("X_test must contain concurrent_patients")
    y_true_days = np.expm1(np.asarray(y_test, dtype=float))
    h_pred = np.clip(np.expm1(hempel_model.predict(X_test_hempel)), 0.0, None)
    e_pred = np.clip(np.expm1(extended_model.predict(X_test)), 0.0, None)
    q33, q66 = X_test["concurrent_patients"].quantile([0.33, 0.66])
    strata = [
        ("Low", X_test["concurrent_patients"] < q33),
        ("Medium", X_test["concurrent_patients"].between(q33, q66, inclusive="both")),
        ("High", X_test["concurrent_patients"] > q66),
    ]
    rows = []
    for label, mask in strata:
        if not mask.any():
            continue
        h_mae = mean_absolute_error(y_true_days[mask], h_pred[mask])
        e_mae = mean_absolute_error(y_true_days[mask], e_pred[mask])
        rows.append({"occupancy_stratum": label, "hempel_mae": h_mae, "extended_mae": e_mae, "delta_mae": e_mae - h_mae, "n": int(mask.sum())})
    table = pd.DataFrame(rows)
    root = Path(output_dir)
    _save_table(table, root / "tables", "stratified_analysis", "table8_stratified")
    _plot_stratified_mae(table, root / "figures" / "stratified_occupancy_mae.png")
    return table


def _save_table(df: pd.DataFrame, output_dir: str | Path, base_name: str, table_name: str) -> None:
    """Save a table with both descriptive and numbered names.

    Args:
        df: Table to save.
        output_dir: Output directory.
        base_name: Descriptive filename stem.
        table_name: Numbered filename stem.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for stem in {base_name, table_name}:
        df.to_csv(directory / f"{stem}.csv", index=False)
        df.to_latex(directory / f"{stem}.tex", index=False)


def _paired_correctness_sign_test(y_true: np.ndarray, pred_h: np.ndarray, pred_e: np.ndarray) -> float:
    """Run a paired sign test on correctness indicators.

    Args:
        y_true: True labels.
        pred_h: Baseline predictions.
        pred_e: Extended predictions.

    Returns:
        Two-sided binomial-test p-value.
    """
    h_correct = pred_h == y_true
    e_correct = pred_e == y_true
    wins = int((e_correct & ~h_correct).sum())
    losses = int((h_correct & ~e_correct).sum())
    total = wins + losses
    if total == 0:
        return np.nan
    return float(stats.binomtest(wins, total, p=0.5).pvalue)


def _plot_permutation_importance(importance_df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot top permutation importances with occupancy highlighted.

    Args:
        importance_df: Permutation importance table.
        output_path: Figure output path.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    top = importance_df.head(25).iloc[::-1]
    colors = ["#b22222" if feature in OCCUPANCY_COLUMNS else "#4c78a8" for feature in top["feature"]]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color=colors)
    ax.set_xlabel("Permutation importance (decrease in neg MAE)")
    ax.set_title("Permutation Importance with Occupancy Highlighted")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_curve(ax: Any, sizes: np.ndarray, scores: np.ndarray, label: str, linestyle: str) -> None:
    """Plot a learning curve mean and standard deviation band.

    Args:
        ax: Matplotlib axes.
        sizes: Training sizes.
        scores: Score matrix where lower is better.
        label: Curve label.
        linestyle: Matplotlib linestyle.
    """
    mean = scores.mean(axis=1)
    std = scores.std(axis=1)
    ax.plot(sizes, mean, linestyle=linestyle, label=label)
    ax.fill_between(sizes, mean - std, mean + std, alpha=0.15)


def _plot_stratified_mae(table: pd.DataFrame, output_path: str | Path) -> None:
    """Plot grouped MAE bars by occupancy stratum.

    Args:
        table: Stratified MAE table.
        output_path: Figure output path.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(table))
    width = 0.35
    ax.bar(x - width / 2, table["hempel_mae"], width=width, label="Hempel", color="#4c78a8")
    ax.bar(x + width / 2, table["extended_mae"], width=width, label="Extended", color="#59a14f")
    ax.set_xticks(x)
    ax.set_xticklabels(table["occupancy_stratum"])
    ax.set_ylabel("MAE (days)")
    ax.set_title("MAE by Occupancy Stratum")
    ax.legend()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
