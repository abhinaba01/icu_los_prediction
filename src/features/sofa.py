def build_sofa(cohort_df: pd.DataFrame,
               loader,
               sofa_precomputed_df: pd.DataFrame = None) -> pd.DataFrame:
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
    """
    if sofa_precomputed_df is not None:
        df = sofa_precomputed_df.copy()
    else:
        df = loader.run_sql('sql/06_extract_sofa.sql')

    return df.set_index('stay_id')