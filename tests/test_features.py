"""Tests for feature engineering modules."""

from __future__ import annotations

import pandas as pd

from src.features.administrative import build_administrative, map_icd10_chapter
from src.features.demographics import build_demographics
from src.features.labs import build_labs
from src.features.vitals import build_vitals


def test_demographics_and_administrative_features() -> None:
    """Check demographic encoding and administrative one-hot construction."""
    cohort = pd.DataFrame(
        {
            "stay_id": [10, 20],
            "hadm_id": [100, 200],
            "age": [65, 44],
            "gender": ["M", "F"],
            "first_careunit": ["Medical Intensive Care Unit (MICU)", "Surgical Intensive Care Unit (SICU)"],
        }
    )
    admissions = pd.DataFrame(
        {
            "hadm_id": [100, 200],
            "admission_type": ["EW EMER.", "ELECTIVE"],
            "admission_location": ["EMERGENCY ROOM", "PHYSICIAN REFERRAL"],
        }
    )
    diagnoses = pd.DataFrame(
        {
            "hadm_id": [100, 100, 200, 200],
            "icd_code": ["A419", "I219", "J189", "N179"],
            "icd_version": [10, 10, 10, 10],
            "seq_num": [1, 2, 1, 2],
        }
    )
    demo = build_demographics(cohort)
    admin = build_administrative(cohort, diagnoses, admissions)
    assert demo.loc[10, "gender_male"] == 1
    assert demo.loc[20, "gender_male"] == 0
    assert "diag1_ch_1" in admin.columns
    assert "diag2_ch_9" in admin.columns
    assert map_icd10_chapter("I219") == 9


def test_vitals_and_labs_first_24h_means() -> None:
    """Check raw first-24h vital and lab aggregation."""
    cohort = pd.DataFrame(
        {
            "stay_id": [1],
            "hadm_id": [100],
            "intime": pd.to_datetime(["2020-01-01 00:00"]),
        }
    )
    chartevents = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1],
            "itemid": [220045, 220045, 223761, 220045],
            "charttime": pd.to_datetime(["2020-01-01 01:00", "2020-01-01 02:00", "2020-01-01 03:00", "2020-01-02 02:00"]),
            "valuenum": [80, 100, 98.6, 120],
        }
    )
    labevents = pd.DataFrame(
        {
            "hadm_id": [100, 100, 100],
            "itemid": [50912, 50912, 50931],
            "charttime": pd.to_datetime(["2020-01-01 01:00", "2020-01-01 03:00", "2020-01-02 02:00"]),
            "valuenum": [1.0, 2.0, 100.0],
        }
    )
    vitals = build_vitals(cohort, chartevents_df=chartevents)
    labs = build_labs(cohort, labevents_df=labevents)
    assert vitals.loc[1, "heart_rate_mean"] == 90
    assert round(vitals.loc[1, "temperature_mean"], 1) == 37.0
    assert labs.loc[1, "creatinine_mean"] == 1.5
    assert pd.isna(labs.loc[1, "glucose_mean"])
