"""SOFA severity-score feature construction from the first 24h of ICU stay."""

from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)

# SOFA component scores: each should vary across a real MIMIC-IV cohort.
SOFA_COMPONENT_COLUMNS = [
    "sofa_total",
    "sofa_resp",
    "sofa_cardio",
    "sofa_hepatic",
    "sofa_coag",
    "sofa_renal",
    "sofa_neuro",
]
# Raw supporting signals: should not be entirely missing for a real cohort.
SOFA_RAW_COLUMNS = ["urine_24h", "bilirubin_max"]


def build_sofa(
    cohort_df: pd.DataFrame,
    loader,
    sofa_precomputed_df: pd.DataFrame = None,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Build SOFA score features from first 24h of ICU stay.

    If sofa_precomputed_df provided (from SQL query), use directly.
    Otherwise runs the SQL via loader.

    Output columns:
        sofa_total        (int, 0–24)  ← primary feature
        sofa_resp         (int, 0–4)
        sofa_cardio       (int, 0–4)   ← new information vs Hempel
        sofa_hepatic      (int, 0–4)   ← new information vs Hempel
        sofa_coag         (int, 0–4)
        sofa_renal        (int, 0–4)
        sofa_neuro        (int, 0–4)
        urine_24h         (float, mL)  ← bonus: raw renal signal
        bilirubin_max     (float)      ← bonus: raw hepatic signal

    All are continuous features — handled by the existing StandardScaler
    in preprocessing/pipeline.py with no code changes required.

    Missing value handling:
        SOFA components default to 0 when data is absent (standard
        clinical convention: absence of vasopressors = score 0).
        urine_24h and bilirubin_max: impute with median in pipeline.

    Args:
        cohort_df: Cohort DataFrame (kept for API symmetry with other
            feature builders; the SQL derives its own cohort from
            ``base_cohort``).
        loader: Configured MIMIC loader used to run the SOFA SQL extraction.
        sofa_precomputed_df: Optional precomputed SOFA table.
        strict: When True (default), raise if any SOFA component is constant
            or any raw signal is entirely missing — a fail-loud guard against
            silently training on degenerate severity features (e.g. when
            ``inputevents``/``outputevents`` are absent from the database).

    Returns:
        DataFrame indexed by ``stay_id`` with the SOFA feature columns.

    Raises:
        ValueError: If ``strict`` is True and one or more SOFA features carry
            no usable signal.
    """
    if sofa_precomputed_df is not None:
        df = sofa_precomputed_df.copy()
    else:
        df = loader.run_sql_extract('sofa')

    sofa = df.set_index('stay_id')
    _check_sofa_quality(sofa, strict=strict)
    return sofa


def _check_sofa_quality(sofa: pd.DataFrame, strict: bool) -> None:
    """Flag SOFA components that carry no usable signal.

    A component score is "dead" if it is missing entirely or has a single
    distinct value across the cohort; a raw signal is "dead" if it is
    entirely null. These are exactly the symptoms of an upstream data gap
    (missing source tables, wrong item ids, or a broken join), so we surface
    them loudly instead of letting the model train on constant features.

    Args:
        sofa: SOFA feature matrix indexed by ``stay_id``.
        strict: Raise on dead features when True; otherwise only warn.

    Raises:
        ValueError: If ``strict`` is True and any dead feature is found.
    """
    dead: list[str] = []
    for column in SOFA_COMPONENT_COLUMNS:
        if column not in sofa.columns:
            dead.append(f"{column} (missing from SQL output)")
            continue
        values = pd.to_numeric(sofa[column], errors="coerce")
        if values.notna().sum() == 0:
            dead.append(f"{column} (all null)")
        elif values.nunique(dropna=True) <= 1:
            only = values.dropna().iloc[0]
            dead.append(f"{column} (constant = {only})")
    for column in SOFA_RAW_COLUMNS:
        if column not in sofa.columns:
            dead.append(f"{column} (missing from SQL output)")
        elif pd.to_numeric(sofa[column], errors="coerce").notna().sum() == 0:
            dead.append(f"{column} (all null)")

    if not dead:
        LOGGER.info("SOFA quality check passed; all %d features carry signal.", len(sofa.columns))
        return

    detail = "; ".join(dead)
    message = (
        f"SOFA features carry no usable signal: {detail}. "
        "This usually means a source table is missing or empty in the database "
        "(cardiovascular needs mimiciv_icu.inputevents; renal urine needs "
        "mimiciv_icu.outputevents), or an item id / time window matched no rows. "
        "Load the required tables (or pass strict=False to bypass this guard)."
    )
    if strict:
        raise ValueError(message)
    LOGGER.warning(message)
