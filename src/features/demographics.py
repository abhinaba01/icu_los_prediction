"""Demographic feature construction."""

from __future__ import annotations

import pandas as pd


def build_demographics(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Extract and encode demographic features.

    Args:
        cohort_df: Cohort DataFrame containing `stay_id`, `age`, and `gender`.

    Returns:
        DataFrame indexed by `stay_id` with `age` and `gender_male`.

    Raises:
        KeyError: If required columns are missing.
    """
    required = {"stay_id", "age", "gender"}
    missing = required.difference(cohort_df.columns)
    if missing:
        raise KeyError(f"Missing demographic columns: {sorted(missing)}")

    output = cohort_df[["stay_id", "age", "gender"]].copy()
    output["age"] = pd.to_numeric(output["age"], errors="coerce")
    output["gender_male"] = output["gender"].astype(str).str.upper().eq("M").astype(int)
    return output.set_index("stay_id")[["age", "gender_male"]]
