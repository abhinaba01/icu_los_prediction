"""Feature-matrix assembly for Hempel replication and occupancy extension."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.loader import MIMICLoader
from src.data.validator import validate_feature_extraction
from src.features.administrative import build_administrative
from src.features.demographics import build_demographics
from src.features.labs import build_labs
from src.features.labs import LAB_ITEMIDS, LAB_OUTPUT_COLUMNS, LAB_RANGES
from src.features.occupancy import build_occupancy
from src.features.vitals import build_vitals
from src.features.vitals import VITAL_ITEMIDS, VITAL_OUTPUT_COLUMNS, VITAL_RANGES

LOGGER = logging.getLogger(__name__)


def build_hempel_features(loader: MIMICLoader, config: dict[str, Any]) -> pd.DataFrame:
    """Orchestrate extraction and merging of all Hempel et al. features.

    Args:
        loader: Configured MIMIC loader.
        config: Parsed project configuration.

    Returns:
        DataFrame containing Hempel feature columns plus outcomes.

    Raises:
        FileNotFoundError: If required CSV source files are missing.
    """
    if loader.source == "postgresql":
        cohort_df = loader.run_sql_extract("cohort")
        admissions_df = loader.load_table("admissions")
        diag_df = loader.run_sql_extract("diagnoses")
        if _table_source(loader, "chartevents") == "csv":
            vitals_df = build_vitals_from_csv_chunks(loader, cohort_df)
        else:
            vitals_df = build_vitals(cohort_df, vitals_precomputed_df=loader.run_sql_extract("vitals"))
        if _table_source(loader, "labevents") == "csv":
            labs_df = build_labs_from_csv_chunks(loader, cohort_df)
        else:
            labs_df = build_labs(cohort_df, labs_precomputed_df=loader.run_sql_extract("labs"))
    else:
        cohort_df = build_base_cohort_from_csv(loader, config)
        admissions_df = loader.load_table("admissions")
        diag_df = loader.load_table("diagnoses_icd")
        vitals_df = build_vitals(cohort_df, chartevents_df=loader.load_table("chartevents"))
        labs_df = build_labs(cohort_df, labevents_df=loader.load_table("labevents"))

    cohort_with_adm = _merge_admissions(cohort_df, admissions_df)
    base = _base_feature_frame(cohort_with_adm)
    feature_groups = [
        build_demographics(cohort_with_adm),
        build_administrative(cohort_with_adm, diag_df, admissions_df=admissions_df),
        vitals_df,
        labs_df,
    ]
    merged = base.join(feature_groups, how="left")
    merged = _add_outcomes(merged.reset_index(), config)
    _save_processed(merged, loader.project_root / "data" / "processed" / "hempel_features.parquet")
    missing = validate_feature_extraction(merged)
    _save_processed(missing, loader.project_root / "results" / "hempel" / "tables" / "feature_missingness.parquet")
    return merged


def build_vitals_from_csv_chunks(loader: MIMICLoader, cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Build vital-sign features by streaming the full chartevents CSV.

    Args:
        loader: Configured loader with CSV path information.
        cohort_df: Base cohort DataFrame.

    Returns:
        DataFrame indexed by `stay_id` with vital mean columns.
    """
    cohort = cohort_df[["stay_id", "intime"]].copy()
    cohort["stay_id"] = pd.to_numeric(cohort["stay_id"], errors="coerce").astype("Int64")
    cohort["intime"] = pd.to_datetime(cohort["intime"], errors="coerce")
    cohort = cohort.dropna(subset=["stay_id", "intime"])
    stay_ids = set(cohort["stay_id"].astype(int))
    aggregates = []
    for chunk in loader.iter_csv_table("chartevents", usecols=["stay_id", "itemid", "charttime", "valuenum"]):
        chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce")
        chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
        chunk = chunk[chunk["stay_id"].isin(stay_ids) & chunk["itemid"].isin(VITAL_ITEMIDS)]
        if chunk.empty:
            continue
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk.merge(cohort, on="stay_id", how="inner")
        in_window = chunk["charttime"].between(chunk["intime"], chunk["intime"] + pd.Timedelta(hours=24), inclusive="both")
        chunk = chunk[in_window & chunk["valuenum"].notna()].copy()
        if chunk.empty:
            continue
        chunk["feature_name"] = chunk["itemid"].map(VITAL_ITEMIDS)
        chunk = _filter_feature_ranges(chunk, VITAL_RANGES)
        if chunk.empty:
            continue
        fahrenheit = chunk["feature_name"].eq("temp_f")
        chunk.loc[fahrenheit, "valuenum"] = (chunk.loc[fahrenheit, "valuenum"] - 32.0) * 5.0 / 9.0
        chunk["feature_name"] = chunk["feature_name"].replace({"temp_f": "temperature", "temp_c": "temperature"})
        aggregates.append(chunk.groupby(["stay_id", "feature_name"])["valuenum"].agg(["sum", "count"]).reset_index())
    return _finalize_chunked_means(aggregates, VITAL_OUTPUT_COLUMNS)


def build_labs_from_csv_chunks(loader: MIMICLoader, cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Build lab features by streaming the full labevents CSV.

    Args:
        loader: Configured loader with CSV path information.
        cohort_df: Base cohort DataFrame.

    Returns:
        DataFrame indexed by `stay_id` with lab mean columns.
    """
    cohort = cohort_df[["stay_id", "hadm_id", "intime"]].copy()
    cohort["stay_id"] = pd.to_numeric(cohort["stay_id"], errors="coerce").astype("Int64")
    cohort["hadm_id"] = pd.to_numeric(cohort["hadm_id"], errors="coerce").astype("Int64")
    cohort["intime"] = pd.to_datetime(cohort["intime"], errors="coerce")
    cohort = cohort.dropna(subset=["stay_id", "hadm_id", "intime"])
    hadm_ids = set(cohort["hadm_id"].astype(int))
    aggregates = []
    for chunk in loader.iter_csv_table("labevents", usecols=["hadm_id", "itemid", "charttime", "valuenum"]):
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce")
        chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
        chunk = chunk[chunk["hadm_id"].isin(hadm_ids) & chunk["itemid"].isin(LAB_ITEMIDS)]
        if chunk.empty:
            continue
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk.merge(cohort, on="hadm_id", how="inner")
        in_window = chunk["charttime"].between(chunk["intime"], chunk["intime"] + pd.Timedelta(hours=24), inclusive="both")
        chunk = chunk[in_window & chunk["valuenum"].notna()].copy()
        if chunk.empty:
            continue
        chunk["feature_name"] = chunk["itemid"].map(LAB_ITEMIDS)
        chunk = _filter_feature_ranges(chunk, LAB_RANGES)
        if chunk.empty:
            continue
        aggregates.append(chunk.groupby(["stay_id", "feature_name"])["valuenum"].agg(["sum", "count"]).reset_index())
    return _finalize_chunked_means(aggregates, LAB_OUTPUT_COLUMNS)


# def build_extended_features(hempel_df: pd.DataFrame, cohort_df: pd.DataFrame, output_path: str | Path | None = None) -> pd.DataFrame:
    """Add ICU bed occupancy features to the Hempel feature set.

    Args:
        hempel_df: Hempel feature matrix containing `stay_id`.
        cohort_df: Cohort DataFrame with admission and discharge timestamps.
        output_path: Optional output parquet path.

    Returns:
        Extended feature matrix.

    Raises:
        KeyError: If `stay_id` is missing.
    """
    if "stay_id" not in hempel_df.columns:
        raise KeyError("hempel_df must contain stay_id")
    occupancy = build_occupancy(cohort_df)
    extended = hempel_df.merge(occupancy.reset_index(), on="stay_id", how="left")
    output = Path(output_path) if output_path is not None else Path("data/processed/extended_features.parquet")
    _save_processed(extended, output)
    return extended

# AFTER:
def build_extended_features(hempel_df, cohort_df, loader):
    sofa = build_sofa(cohort_df, loader)
    extended = hempel_df.join(sofa, on='stay_id', how='left')
    extended.to_parquet('data/processed/extended_features.parquet')
    return extended


def _table_source(loader: MIMICLoader, table_name: str) -> str:
    """Return the configured source for a table.

    Args:
        loader: Configured MIMIC loader.
        table_name: MIMIC table name.

    Returns:
        Source name, either `csv` or `postgresql`.
    """
    return loader.config["data"].get("table_sources", {}).get(table_name, loader.source).lower()


def _filter_feature_ranges(events: pd.DataFrame, ranges: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Apply feature-specific plausible value ranges.

    Args:
        events: Event DataFrame with `feature_name` and `valuenum`.
        ranges: Mapping from feature name to lower and upper bounds.

    Returns:
        Filtered event DataFrame.
    """
    mask = pd.Series(False, index=events.index)
    for feature, (lower, upper) in ranges.items():
        mask = mask | (events["feature_name"].eq(feature) & events["valuenum"].between(lower, upper))
    return events[mask].copy()


def _finalize_chunked_means(aggregates: list[pd.DataFrame], output_columns: list[str]) -> pd.DataFrame:
    """Convert chunk-level sums and counts into stay-level means.

    Args:
        aggregates: Chunk-level aggregate DataFrames with `sum` and `count`.
        output_columns: Required output column names with `_mean` suffixes.

    Returns:
        DataFrame indexed by `stay_id` with requested mean columns.
    """
    if not aggregates:
        return pd.DataFrame(columns=output_columns).rename_axis("stay_id")
    combined = pd.concat(aggregates, ignore_index=True)
    totals = combined.groupby(["stay_id", "feature_name"])[["sum", "count"]].sum()
    totals["mean"] = totals["sum"] / totals["count"]
    means = totals["mean"].unstack()
    means = means.rename(columns={column: f"{column}_mean" for column in means.columns})
    for column in output_columns:
        if column not in means.columns:
            means[column] = pd.NA
    return means[output_columns]


def build_base_cohort_from_csv(loader: MIMICLoader, config: dict[str, Any]) -> pd.DataFrame:
    """Build the base adult first-stay ICU cohort from CSV tables.

    Args:
        loader: Configured CSV loader.
        config: Parsed project configuration.

    Returns:
        Base cohort DataFrame.
    """
    icustays = loader.load_table("icustays")
    patients = loader.load_table("patients")
    cohort = icustays.copy()
    cohort["intime"] = pd.to_datetime(cohort["intime"], errors="coerce")
    cohort["outtime"] = pd.to_datetime(cohort["outtime"], errors="coerce")
    cohort["los_hours"] = (cohort["outtime"] - cohort["intime"]).dt.total_seconds() / 3600.0
    if "los" in cohort.columns:
        cohort["los_days"] = pd.to_numeric(cohort["los"], errors="coerce")
    else:
        cohort["los_days"] = cohort["los_hours"] / 24.0

    patient_cols = ["subject_id", "gender"]
    age_col = "anchor_age" if "anchor_age" in patients.columns else "age"
    patient_cols.append(age_col)
    cohort = cohort.merge(patients[patient_cols], on="subject_id", how="inner")
    cohort = cohort.rename(columns={age_col: "age"})
    cohort = cohort.sort_values(["subject_id", "intime"])
    cohort["stay_rank"] = cohort.groupby("subject_id").cumcount() + 1
    cohort_cfg = config["cohort"]
    mask = (
        cohort["los_hours"].ge(cohort_cfg["min_los_hours"])
        & cohort["los_days"].le(cohort_cfg["max_los_days"])
        & pd.to_numeric(cohort["age"], errors="coerce").ge(cohort_cfg["min_age"])
    )
    if cohort_cfg.get("first_icu_stay_only", True):
        mask = mask & cohort["stay_rank"].eq(1)
    columns = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "first_careunit",
        "last_careunit",
        "intime",
        "outtime",
        "los_hours",
        "los_days",
        "gender",
        "age",
    ]
    return cohort.loc[mask, columns].copy()


def _merge_admissions(cohort_df: pd.DataFrame, admissions_df: pd.DataFrame) -> pd.DataFrame:
    """Merge admission metadata onto the cohort.

    Args:
        cohort_df: Base cohort.
        admissions_df: Admissions table.

    Returns:
        Cohort with admission metadata.
    """
    columns = ["hadm_id", "admission_type", "admission_location"]
    available = [column for column in columns if column in admissions_df.columns]
    if len(available) <= 1:
        output = cohort_df.copy()
        output["admission_type"] = "Unknown"
        output["admission_location"] = "Unknown"
        return output
    return cohort_df.merge(admissions_df[available].drop_duplicates("hadm_id"), on="hadm_id", how="left")


def _base_feature_frame(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Create the identifier and raw outcome base frame.

    Args:
        cohort_df: Cohort DataFrame.

    Returns:
        DataFrame indexed by `stay_id`.
    """
    columns = ["stay_id", "subject_id", "hadm_id", "los_days", "los_hours", "first_careunit", "intime", "outtime"]
    available = [column for column in columns if column in cohort_df.columns]
    return cohort_df[available].drop_duplicates("stay_id").set_index("stay_id")


def _add_outcomes(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add log LOS and LOS class labels.

    Args:
        df: Feature DataFrame containing `los_days`.
        config: Parsed project configuration.

    Returns:
        DataFrame with `los_log` and `los_class`.
    """
    output = df.copy()
    output["los_days"] = pd.to_numeric(output["los_days"], errors="coerce")
    output["los_log"] = np.log1p(output["los_days"])
    short_threshold = config["los_bins"]["short_threshold_days"]
    long_threshold = config["los_bins"]["long_threshold_days"]
    output["los_class"] = np.select(
        [output["los_days"].le(short_threshold), output["los_days"].gt(long_threshold)],
        [0, 2],
        default=1,
    ).astype(int)
    return output


def _save_processed(df: pd.DataFrame, path: str | Path) -> None:
    """Save a processed DataFrame as parquet.

    Args:
        df: DataFrame to save.
        path: Output parquet path.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
