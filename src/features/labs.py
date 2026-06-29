"""Laboratory feature construction from first-24-hour observations."""

from __future__ import annotations

import pandas as pd

LAB_ITEMIDS = {
    50868: "anion_gap",
    50882: "bicarbonate",
    50902: "chloride",
    50912: "creatinine",
    50931: "glucose",
    50983: "sodium",
    50960: "magnesium",
    50971: "potassium",
    50970: "phosphate",
    51006: "bun",
    51221: "hematocrit",
    51222: "hemoglobin",
    51248: "mch",
    51249: "mchc",
    51250: "mcv",
    51277: "rdw",
    51279: "rbc",
    51301: "wbc",
    51265: "platelets",
}

LAB_RANGES = {
    "anion_gap": (1, 40),
    "bicarbonate": (5, 50),
    "chloride": (70, 140),
    "creatinine": (0.1, 20),
    "glucose": (33, 1000),
    "sodium": (100, 180),
    "magnesium": (0.5, 5),
    "potassium": (1.5, 10),
    "phosphate": (0.5, 10),
    "bun": (1, 150),
    "hematocrit": (5, 65),
    "hemoglobin": (2, 22),
    "mch": (15, 45),
    "mchc": (20, 45),
    "mcv": (50, 130),
    "rdw": (9, 35),
    "rbc": (1, 10),
    "wbc": (0.1, 80),
    "platelets": (5, 1500),
}

LAB_OUTPUT_COLUMNS = [f"{name}_mean" for name in LAB_ITEMIDS.values()]


def build_labs(
    cohort_df: pd.DataFrame,
    labevents_df: pd.DataFrame | None = None,
    labs_precomputed_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build lab-value means from the first 24 hours of ICU stay.

    Args:
        cohort_df: Cohort DataFrame with `stay_id`, `hadm_id`, and `intime`.
        labevents_df: Raw MIMIC `labevents` table.
        labs_precomputed_df: Optional SQL-derived lab table.

    Returns:
        DataFrame indexed by `stay_id` with 19 lab mean columns.

    Raises:
        ValueError: If neither raw nor precomputed lab data is provided.
        KeyError: If required columns are missing.
    """
    if labs_precomputed_df is not None:
        return _format_precomputed(cohort_df, labs_precomputed_df)
    if labevents_df is None:
        raise ValueError("Either labevents_df or labs_precomputed_df must be provided")

    required = {"stay_id", "hadm_id", "intime"}
    missing = required.difference(cohort_df.columns)
    if missing:
        raise KeyError(f"Missing cohort columns for labs: {sorted(missing)}")
    raw_required = {"hadm_id", "itemid", "charttime", "valuenum"}
    raw_missing = raw_required.difference(labevents_df.columns)
    if raw_missing:
        raise KeyError(f"Missing labevents columns: {sorted(raw_missing)}")

    cohort = cohort_df[["stay_id", "hadm_id", "intime"]].copy()
    cohort["intime"] = pd.to_datetime(cohort["intime"], errors="coerce")
    events = labevents_df[["hadm_id", "itemid", "charttime", "valuenum"]].copy()
    events["itemid"] = pd.to_numeric(events["itemid"], errors="coerce").astype("Int64")
    events = events[events["itemid"].isin(LAB_ITEMIDS)]
    events["feature_name"] = events["itemid"].astype(int).map(LAB_ITEMIDS)
    events["valuenum"] = pd.to_numeric(events["valuenum"], errors="coerce")
    events["charttime"] = pd.to_datetime(events["charttime"], errors="coerce")
    events = events.merge(cohort, on="hadm_id", how="inner")
    in_window = events["charttime"].between(events["intime"], events["intime"] + pd.Timedelta(hours=24), inclusive="both")
    events = events[in_window & events["valuenum"].notna()].copy()
    events = _filter_ranges(events)
    means = events.groupby(["stay_id", "feature_name"])["valuenum"].mean().unstack()
    means = means.rename(columns={feature: f"{feature}_mean" for feature in means.columns})
    return _ensure_columns(means, LAB_OUTPUT_COLUMNS)


def _filter_ranges(events: pd.DataFrame) -> pd.DataFrame:
    """Apply physiologic plausibility ranges to lab events.

    Args:
        events: Lab-event DataFrame with `feature_name` and `valuenum`.

    Returns:
        Filtered lab-event DataFrame.
    """
    mask = pd.Series(False, index=events.index)
    for feature, (lower, upper) in LAB_RANGES.items():
        feature_mask = events["feature_name"].eq(feature) & events["valuenum"].between(lower, upper)
        mask = mask | feature_mask
    return events[mask].copy()


def _format_precomputed(cohort_df: pd.DataFrame, labs_df: pd.DataFrame) -> pd.DataFrame:
    """Format a precomputed lab table indexed by `stay_id`.

    Args:
        cohort_df: Cohort table used to map `hadm_id` to `stay_id`.
        labs_df: Precomputed lab table from SQL.

    Returns:
        DataFrame indexed by `stay_id` with expected lab columns.

    Raises:
        KeyError: If required identifier columns are missing.
    """
    formatted = labs_df.copy()
    if "stay_id" not in formatted.columns:
        if "hadm_id" not in formatted.columns:
            raise KeyError("Precomputed labs must contain stay_id or hadm_id")
        formatted = cohort_df[["stay_id", "hadm_id"]].merge(formatted, on="hadm_id", how="left")
    formatted = formatted.set_index("stay_id")
    return _ensure_columns(formatted, LAB_OUTPUT_COLUMNS)


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
