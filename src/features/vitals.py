"""Vital-sign feature construction from first-24-hour ICU observations."""

from __future__ import annotations

import pandas as pd

VITAL_ITEMIDS = {
    220045: "heart_rate",
    220277: "spo2",
    220210: "resp_rate",
    223761: "temp_f",
    223762: "temp_c",
    220739: "gcs_eye",
    223900: "gcs_verbal",
    223901: "gcs_motor",
}

VITAL_RANGES = {
    "heart_rate": (20, 300),
    "spo2": (70, 100),
    "resp_rate": (4, 60),
    "temp_f": (85, 108),
    "temp_c": (28, 42),
    "gcs_eye": (1, 4),
    "gcs_verbal": (1, 5),
    "gcs_motor": (1, 6),
}

VITAL_OUTPUT_COLUMNS = [
    "heart_rate_mean",
    "spo2_mean",
    "resp_rate_mean",
    "temperature_mean",
    "gcs_eye_mean",
    "gcs_verbal_mean",
    "gcs_motor_mean",
]


def build_vitals(
    cohort_df: pd.DataFrame,
    chartevents_df: pd.DataFrame | None = None,
    vitals_precomputed_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build mean vital-sign features from the first 24 hours of ICU stay.

    Args:
        cohort_df: Cohort DataFrame with `stay_id` and `intime`.
        chartevents_df: Raw MIMIC `chartevents` table.
        vitals_precomputed_df: Optional SQL-derived vital-sign table.

    Returns:
        DataFrame indexed by `stay_id` with seven vital mean columns.

    Raises:
        ValueError: If neither raw nor precomputed vital data is provided.
        KeyError: If required raw columns are missing.
    """
    if vitals_precomputed_df is not None:
        return _format_precomputed(vitals_precomputed_df)
    if chartevents_df is None:
        raise ValueError("Either chartevents_df or vitals_precomputed_df must be provided")

    required = {"stay_id", "intime"}
    missing = required.difference(cohort_df.columns)
    if missing:
        raise KeyError(f"Missing cohort columns for vitals: {sorted(missing)}")
    raw_required = {"stay_id", "itemid", "charttime", "valuenum"}
    raw_missing = raw_required.difference(chartevents_df.columns)
    if raw_missing:
        raise KeyError(f"Missing chartevents columns: {sorted(raw_missing)}")

    cohort = cohort_df[["stay_id", "intime"]].copy()
    cohort["intime"] = pd.to_datetime(cohort["intime"], errors="coerce")
    events = chartevents_df[["stay_id", "itemid", "charttime", "valuenum"]].copy()
    events["itemid"] = pd.to_numeric(events["itemid"], errors="coerce").astype("Int64")
    events = events[events["itemid"].isin(VITAL_ITEMIDS)]
    events["feature_name"] = events["itemid"].astype(int).map(VITAL_ITEMIDS)
    events["valuenum"] = pd.to_numeric(events["valuenum"], errors="coerce")
    events["charttime"] = pd.to_datetime(events["charttime"], errors="coerce")
    events = events.merge(cohort, on="stay_id", how="inner")
    in_window = events["charttime"].between(events["intime"], events["intime"] + pd.Timedelta(hours=24), inclusive="both")
    events = events[in_window & events["valuenum"].notna()].copy()
    events = _filter_ranges(events)
    events.loc[events["feature_name"].eq("temp_f"), "valuenum"] = (
        events.loc[events["feature_name"].eq("temp_f"), "valuenum"] - 32.0
    ) * 5.0 / 9.0
    events["feature_name"] = events["feature_name"].replace({"temp_f": "temperature", "temp_c": "temperature"})
    means = events.groupby(["stay_id", "feature_name"])["valuenum"].mean().unstack()
    means = means.rename(columns={feature: f"{feature}_mean" for feature in means.columns})
    return _ensure_columns(means, VITAL_OUTPUT_COLUMNS)


def _filter_ranges(events: pd.DataFrame) -> pd.DataFrame:
    """Apply physiologic plausibility ranges to vital events.

    Args:
        events: Vital-event DataFrame with `feature_name` and `valuenum`.

    Returns:
        Filtered vital-event DataFrame.
    """
    mask = pd.Series(False, index=events.index)
    for feature, (lower, upper) in VITAL_RANGES.items():
        feature_mask = events["feature_name"].eq(feature) & events["valuenum"].between(lower, upper)
        mask = mask | feature_mask
    return events[mask].copy()


def _format_precomputed(df: pd.DataFrame) -> pd.DataFrame:
    """Format a precomputed vital table.

    Args:
        df: Precomputed vital-sign DataFrame.

    Returns:
        DataFrame indexed by `stay_id` with expected columns.
    """
    if "stay_id" not in df.columns:
        raise KeyError("Precomputed vitals must contain stay_id")
    formatted = df.copy().set_index("stay_id")
    return _ensure_columns(formatted, VITAL_OUTPUT_COLUMNS)


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Ensure a DataFrame has a fixed set of output columns.

    Args:
        df: Input DataFrame.
        columns: Required output columns.

    Returns:
        DataFrame containing exactly the requested columns.
    """
    output = df.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = pd.NA
    return output[columns]
