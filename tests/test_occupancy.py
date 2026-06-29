"""Tests for ICU occupancy feature construction."""

from __future__ import annotations

import pandas as pd

from src.features.occupancy import build_occupancy


def test_build_occupancy_counts_concurrent_patients() -> None:
    """Verify concurrent counts include simultaneous admits and exclude ended stays."""
    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "first_careunit": ["MICU", "MICU", "MICU", "MICU", "SICU"],
            "intime": pd.to_datetime(
                ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 05:00", "2020-01-01 05:00", "2020-01-01 05:00"]
            ),
            "outtime": pd.to_datetime(
                ["2020-01-01 10:00", "2020-01-01 05:00", "2020-01-01 08:00", "2020-01-01 06:00", "2020-01-01 07:00"]
            ),
        }
    )
    output = build_occupancy(cohort)
    assert output.loc[1, "concurrent_patients"] == 0
    assert output.loc[2, "concurrent_patients"] == 1
    assert output.loc[3, "concurrent_patients"] == 2
    assert output.loc[4, "concurrent_patients"] == 2
    assert output.loc[5, "concurrent_patients"] == 0
    assert output["occupancy_percentile"].between(0, 1).all()
