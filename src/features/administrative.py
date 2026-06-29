"""Administrative and diagnosis feature construction."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def map_icd10_chapter(icd_code: str | float | None) -> int:
    """Map an ICD-10 code to one of 22 chapter numbers.

    Args:
        icd_code: ICD-10 diagnosis code.

    Returns:
        Integer chapter code from 1 to 22, or 0 when unknown.
    """
    if pd.isna(icd_code):
        return 0
    code = str(icd_code).strip().upper()
    if not code:
        return 0
    if re.match(r"^[AB]", code):
        return 1
    if re.match(r"^C|^D[0-4]", code):
        return 2
    if re.match(r"^D[5-8]", code):
        return 3
    if code.startswith("E"):
        return 4
    if code.startswith("F"):
        return 5
    if code.startswith("G"):
        return 6
    if re.match(r"^H[0-5]", code):
        return 7
    if re.match(r"^H[6-9]", code):
        return 8
    if code.startswith("I"):
        return 9
    if code.startswith("J"):
        return 10
    if code.startswith("K"):
        return 11
    if code.startswith("L"):
        return 12
    if code.startswith("M"):
        return 13
    if code.startswith("N"):
        return 14
    if code.startswith("O"):
        return 15
    if code.startswith("P"):
        return 16
    if code.startswith("Q"):
        return 17
    if code.startswith("R"):
        return 18
    if re.match(r"^[ST]", code):
        return 19
    if re.match(r"^[V-WXY]", code):
        return 20
    if code.startswith("Z"):
        return 21
    if code.startswith("U"):
        return 22
    return 0


def build_diagnosis_chapters(diag_df: pd.DataFrame) -> pd.DataFrame:
    """Build top-two ICD-10 chapter columns by admission.

    Args:
        diag_df: Diagnosis table, either raw `diagnoses_icd` or precomputed
            chapter columns.

    Returns:
        DataFrame with `hadm_id`, `diag1_chapter`, and `diag2_chapter`.

    Raises:
        KeyError: If the diagnosis table lacks required raw or precomputed
            columns.
    """
    precomputed = {"hadm_id", "diag1_chapter", "diag2_chapter"}
    if precomputed.issubset(diag_df.columns):
        return diag_df[list(precomputed)].copy()

    required = {"hadm_id", "icd_code", "icd_version", "seq_num"}
    missing = required.difference(diag_df.columns)
    if missing:
        raise KeyError(f"Missing diagnosis columns: {sorted(missing)}")

    work = diag_df.copy()
    work = work[pd.to_numeric(work["icd_version"], errors="coerce").eq(10)]
    work["seq_num"] = pd.to_numeric(work["seq_num"], errors="coerce")
    work = work.sort_values(["hadm_id", "seq_num"])
    work["diag_rank"] = work.groupby("hadm_id").cumcount() + 1
    work = work[work["diag_rank"].le(2)].copy()
    work["icd10_chapter"] = work["icd_code"].map(map_icd10_chapter)
    pivot = work.pivot_table(
        index="hadm_id",
        columns="diag_rank",
        values="icd10_chapter",
        aggfunc="first",
    ).rename(columns={1: "diag1_chapter", 2: "diag2_chapter"})
    for column in ["diag1_chapter", "diag2_chapter"]:
        if column not in pivot:
            pivot[column] = 0
    return pivot.reset_index()[["hadm_id", "diag1_chapter", "diag2_chapter"]]


def build_administrative(
    cohort_df: pd.DataFrame,
    diag_df: pd.DataFrame,
    admissions_df: pd.DataFrame | None = None,
    top_n_units: int = 8,
) -> pd.DataFrame:
    """Build administrative and ICD chapter one-hot features.

    Args:
        cohort_df: Cohort DataFrame containing `stay_id`, `hadm_id`, and
            `first_careunit`. It may also contain admission columns.
        diag_df: Raw or precomputed diagnosis DataFrame.
        admissions_df: Optional admissions DataFrame containing
            `admission_type` and `admission_location`.
        top_n_units: Number of most common care units to keep before grouping
            remaining units into `Other`.

    Returns:
        DataFrame indexed by `stay_id` with one-hot administrative features.

    Raises:
        KeyError: If required cohort columns are missing.
    """
    required = {"stay_id", "hadm_id", "first_careunit"}
    missing = required.difference(cohort_df.columns)
    if missing:
        raise KeyError(f"Missing administrative cohort columns: {sorted(missing)}")

    work = cohort_df[["stay_id", "hadm_id", "first_careunit"]].copy()
    if admissions_df is not None:
        adm_cols = ["hadm_id", "admission_type", "admission_location"]
        available = [column for column in adm_cols if column in admissions_df.columns]
        work = work.merge(admissions_df[available].drop_duplicates("hadm_id"), on="hadm_id", how="left")
    else:
        for column in ["admission_type", "admission_location"]:
            if column in cohort_df.columns:
                work[column] = cohort_df[column]
            else:
                work[column] = "Unknown"

    chapters = build_diagnosis_chapters(diag_df)
    work = work.merge(chapters, on="hadm_id", how="left")
    work["diag1_chapter"] = work["diag1_chapter"].fillna(0).astype(int)
    work["diag2_chapter"] = work["diag2_chapter"].fillna(0).astype(int)
    work["first_careunit"] = _consolidate_units(work["first_careunit"], top_n_units)
    work["admission_type"] = work["admission_type"].map(_map_admission_type).fillna("Other")
    work["admission_location"] = work["admission_location"].map(_map_admission_location).fillna("Other")

    encoded_parts = [
        pd.get_dummies(work["first_careunit"], prefix="first_careunit", drop_first=True, dtype=int),
        pd.get_dummies(work["admission_type"], prefix="admission_type", drop_first=True, dtype=int),
        pd.get_dummies(work["admission_location"], prefix="admission_location", drop_first=True, dtype=int),
        _chapter_dummies(work["diag1_chapter"], "diag1_ch"),
        _chapter_dummies(work["diag2_chapter"], "diag2_ch"),
    ]
    encoded = pd.concat(encoded_parts, axis=1)
    encoded.insert(0, "stay_id", work["stay_id"].to_numpy())
    return encoded.set_index("stay_id")


def _consolidate_units(series: pd.Series, top_n: int) -> pd.Series:
    """Map care-unit names to compact labels and group rare values.

    Args:
        series: Care-unit names.
        top_n: Number of most common units to keep.

    Returns:
        Consolidated care-unit series.
    """
    mapped = series.astype(str).str.strip().replace(
        {
            "Coronary Care Unit (CCU)": "CCU",
            "Medical Intensive Care Unit (MICU)": "MICU",
            "Medical/Surgical Intensive Care Unit (MICU/SICU)": "MICU_SICU",
            "Surgical Intensive Care Unit (SICU)": "SICU",
            "Trauma SICU (TSICU)": "TSICU",
            "Neuro Surgical Intensive Care Unit (Neuro SICU)": "Neuro_SICU",
            "Cardiac Vascular ICU": "CVICU",
            "Neuro Intermediate": "Neuro_Intermediate",
            "Neuro Stepdown": "Neuro_Stepdown",
            "MICU/SICU": "MICU_SICU",
        }
    )
    counts = mapped.value_counts(dropna=False)
    keep = set(counts.head(top_n).index)
    return mapped.where(mapped.isin(keep), "Other")


def _map_admission_type(value: object) -> str:
    """Map raw admission type to a compact category.

    Args:
        value: Raw admission type.

    Returns:
        Compact admission type category.
    """
    text = str(value).upper()
    if "ELECTIVE" in text or "SURGICAL SAME DAY" in text:
        return "Elective"
    if "URGENT" in text:
        return "Urgent"
    if "EMER" in text or "EW " in text:
        return "Emergency"
    return "Other"


def _map_admission_location(value: object) -> str:
    """Map raw admission location to a compact category.

    Args:
        value: Raw admission location.

    Returns:
        Compact admission location category.
    """
    text = str(value).upper()
    if "EMERGENCY" in text:
        return "Emergency Room"
    if "SNF" in text or "SKILLED" in text:
        return "Transfer from SNF"
    if "TRANSFER" in text or "HOSPITAL" in text:
        return "Transfer from Hospital"
    if "CLINIC" in text:
        return "Clinic Referral"
    if "PHYSICIAN" in text:
        return "Physician Referral"
    if "WALK" in text or "SELF" in text:
        return "Walk-in/Self Referral"
    return "Other"


def _chapter_dummies(series: pd.Series, prefix: str) -> pd.DataFrame:
    """One-hot encode ICD chapter numbers 1 through 22, dropping chapter 0.

    Args:
        series: Chapter numbers.
        prefix: Column prefix for dummy variables.

    Returns:
        Dummy-coded DataFrame.
    """
    values = pd.Categorical(series.fillna(0).astype(int), categories=np.arange(23))
    return pd.get_dummies(values, prefix=prefix, dtype=int).drop(columns=[f"{prefix}_0"], errors="ignore")
