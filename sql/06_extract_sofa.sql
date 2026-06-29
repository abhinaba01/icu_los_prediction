-- SOFA Score: Day 1, computed from first 24h of ICU stay
-- Adapted from MIT-LCP mimic-code/mimic-iv/concepts/score/sofa.sql
-- Components: Respiratory, Cardiovascular, Hepatic, Coagulation, Renal, Neurological

WITH cohort AS (
    SELECT stay_id, hadm_id, subject_id, intime, outtime
    FROM base_cohort  -- your existing cohort CTE
),

-- ----------------------------------------------------------------
-- 1. RESPIRATORY: PaO2/FiO2 ratio (use SpO2/FiO2 if PaO2 absent)
-- ----------------------------------------------------------------
bg AS (
    SELECT c.stay_id,
           MIN(CASE WHEN ce.itemid = 50821 THEN ce.valuenum END) AS pao2,
           MAX(CASE WHEN ce.itemid = 223835 THEN ce.valuenum END) AS fio2_chart,
           MAX(CASE WHEN ce.itemid = 220277 THEN ce.valuenum END) AS spo2
    FROM cohort c
    JOIN mimiciv_icu.chartevents ce ON ce.stay_id = c.stay_id
    WHERE ce.itemid IN (50821, 223835, 220277)
      AND ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
      AND ce.valuenum IS NOT NULL
    GROUP BY c.stay_id
),
resp_sofa AS (
    SELECT stay_id,
           COALESCE(pao2, spo2) AS o2_value,
           fio2_chart,
           CASE
               WHEN fio2_chart IS NULL OR fio2_chart = 0 THEN 0
               WHEN (COALESCE(pao2, spo2) / (fio2_chart / 100.0)) < 100 THEN 4
               WHEN (COALESCE(pao2, spo2) / (fio2_chart / 100.0)) < 200 THEN 3
               WHEN (COALESCE(pao2, spo2) / (fio2_chart / 100.0)) < 300 THEN 2
               WHEN (COALESCE(pao2, spo2) / (fio2_chart / 100.0)) < 400 THEN 1
               ELSE 0
           END AS sofa_resp
    FROM bg
),

-- ----------------------------------------------------------------
-- 2. CARDIOVASCULAR: vasopressor use and MAP
-- ----------------------------------------------------------------
vaso AS (
    SELECT c.stay_id,
           -- Norepinephrine
           MAX(CASE WHEN ie.itemid IN (221906, 30047, 30120)
               THEN ie.rate END) AS rate_norepi,
           -- Epinephrine
           MAX(CASE WHEN ie.itemid IN (221289, 30044)
               THEN ie.rate END) AS rate_epi,
           -- Dopamine
           MAX(CASE WHEN ie.itemid IN (221662, 30043)
               THEN ie.rate END) AS rate_dopa,
           -- Dobutamine
           MAX(CASE WHEN ie.itemid IN (221653, 30042)
               THEN ie.rate END) AS rate_dobu,
           -- Vasopressin (binary)
           MAX(CASE WHEN ie.itemid = 222315 THEN 1 ELSE 0 END) AS on_vaso
    FROM cohort c
    JOIN mimiciv_icu.inputevents ie ON ie.stay_id = c.stay_id
    WHERE ie.itemid IN (221906, 30047, 30120, 221289, 30044,
                        221662, 30043, 221653, 30042, 222315)
      AND ie.starttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
    GROUP BY c.stay_id
),
map_vals AS (
    SELECT c.stay_id,
           AVG(ce.valuenum) AS map_mean
    FROM cohort c
    JOIN mimiciv_icu.chartevents ce ON ce.stay_id = c.stay_id
    WHERE ce.itemid IN (220052, 220181, 225312)  -- invasive + non-invasive MAP
      AND ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
      AND ce.valuenum BETWEEN 0 AND 300
    GROUP BY c.stay_id
),
cardio_sofa AS (
    SELECT c.stay_id,
           CASE
               WHEN v.rate_dopa > 15
                 OR v.rate_epi  > 0.1
                 OR v.rate_norepi > 0.1  THEN 4
               WHEN v.rate_dopa > 5
                 OR v.rate_epi  <= 0.1
                 OR v.rate_norepi <= 0.1 THEN 3
               WHEN v.rate_dopa <= 5
                 OR v.on_vaso = 1        THEN 2
               WHEN m.map_mean < 70      THEN 1
               ELSE 0
           END AS sofa_cardio
    FROM cohort c
    LEFT JOIN vaso v ON c.stay_id = v.stay_id
    LEFT JOIN map_vals m ON c.stay_id = m.stay_id
),

-- ----------------------------------------------------------------
-- 3. HEPATIC: Bilirubin (itemid 50885)
-- ----------------------------------------------------------------
hepatic_sofa AS (
    SELECT c.stay_id,
           MAX(le.valuenum) AS bilirubin_max,
           CASE
               WHEN MAX(le.valuenum) >= 12.0 THEN 4
               WHEN MAX(le.valuenum) >= 6.0  THEN 3
               WHEN MAX(le.valuenum) >= 2.0  THEN 2
               WHEN MAX(le.valuenum) >= 1.2  THEN 1
               ELSE 0
           END AS sofa_hepatic
    FROM cohort c
    LEFT JOIN mimiciv_hosp.labevents le
           ON le.hadm_id = c.hadm_id
          AND le.itemid = 50885
          AND le.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
          AND le.valuenum IS NOT NULL
    GROUP BY c.stay_id
),

-- ----------------------------------------------------------------
-- 4. COAGULATION: Platelets (itemid 51265, already in your labs)
-- ----------------------------------------------------------------
coag_sofa AS (
    SELECT c.stay_id,
           MIN(le.valuenum) AS platelets_min,
           CASE
               WHEN MIN(le.valuenum) < 20  THEN 4
               WHEN MIN(le.valuenum) < 50  THEN 3
               WHEN MIN(le.valuenum) < 100 THEN 2
               WHEN MIN(le.valuenum) < 150 THEN 1
               ELSE 0
           END AS sofa_coag
    FROM cohort c
    LEFT JOIN mimiciv_hosp.labevents le
           ON le.hadm_id = c.hadm_id
          AND le.itemid = 51265
          AND le.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
          AND le.valuenum IS NOT NULL
    GROUP BY c.stay_id
),

-- ----------------------------------------------------------------
-- 5. RENAL: Creatinine (itemid 50912) + Urine Output
-- ----------------------------------------------------------------
renal_sofa AS (
    SELECT c.stay_id,
           MAX(cr.valuenum) AS creatinine_max,
           SUM(oe.value)    AS urine_24h,
           CASE
               WHEN MAX(cr.valuenum) >= 5.0
                 OR SUM(oe.value) < 200 THEN 4
               WHEN MAX(cr.valuenum) >= 3.5
                 OR SUM(oe.value) < 500 THEN 3
               WHEN MAX(cr.valuenum) >= 2.0 THEN 2
               WHEN MAX(cr.valuenum) >= 1.2 THEN 1
               ELSE 0
           END AS sofa_renal
    FROM cohort c
    LEFT JOIN mimiciv_hosp.labevents cr
           ON cr.hadm_id = c.hadm_id
          AND cr.itemid = 50912
          AND cr.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
          AND cr.valuenum BETWEEN 0.1 AND 20
    LEFT JOIN mimiciv_icu.outputevents oe
           ON oe.stay_id = c.stay_id
          AND oe.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
          AND oe.itemid IN (226559, 226560, 226561, 226584,
                            226563, 226564, 226565, 226567,
                            226557, 226558, 227488, 227489)
    GROUP BY c.stay_id
),

-- ----------------------------------------------------------------
-- 6. NEUROLOGICAL: GCS (already in your vitals, re-score here)
-- ----------------------------------------------------------------
neuro_sofa AS (
    SELECT c.stay_id,
           MIN(eye.valuenum + verb.valuenum + mot.valuenum) AS gcs_min,
           CASE
               WHEN MIN(eye.valuenum + verb.valuenum + mot.valuenum) < 6  THEN 4
               WHEN MIN(eye.valuenum + verb.valuenum + mot.valuenum) < 10 THEN 3
               WHEN MIN(eye.valuenum + verb.valuenum + mot.valuenum) < 13 THEN 2
               WHEN MIN(eye.valuenum + verb.valuenum + mot.valuenum) < 15 THEN 1
               ELSE 0
           END AS sofa_neuro
    FROM cohort c
    JOIN mimiciv_icu.chartevents eye
      ON eye.stay_id = c.stay_id
     AND eye.itemid = 220739
     AND eye.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
    JOIN mimiciv_icu.chartevents verb
      ON verb.stay_id = c.stay_id
     AND verb.itemid = 223900
     AND verb.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
     AND verb.charttime BETWEEN eye.charttime - INTERVAL '30 minutes'
                             AND eye.charttime + INTERVAL '30 minutes'
    JOIN mimiciv_icu.chartevents mot
      ON mot.stay_id = c.stay_id
     AND mot.itemid = 223901
     AND mot.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
     AND mot.charttime BETWEEN eye.charttime - INTERVAL '30 minutes'
                             AND eye.charttime + INTERVAL '30 minutes'
    GROUP BY c.stay_id
)

-- ----------------------------------------------------------------
-- FINAL ASSEMBLY
-- ----------------------------------------------------------------
SELECT
    c.stay_id,
    COALESCE(rs.sofa_resp,    0) AS sofa_resp,
    COALESCE(cs.sofa_cardio,  0) AS sofa_cardio,
    COALESCE(hs.sofa_hepatic, 0) AS sofa_hepatic,
    COALESCE(gs.sofa_coag,    0) AS sofa_coag,
    COALESCE(re.sofa_renal,   0) AS sofa_renal,
    COALESCE(ns.sofa_neuro,   0) AS sofa_neuro,
    -- Total score (0–24)
    COALESCE(rs.sofa_resp,    0)
    + COALESCE(cs.sofa_cardio,  0)
    + COALESCE(hs.sofa_hepatic, 0)
    + COALESCE(gs.sofa_coag,    0)
    + COALESCE(re.sofa_renal,   0)
    + COALESCE(ns.sofa_neuro,   0) AS sofa_total,
    -- Raw supporting values for interpretability
    re.urine_24h,
    hs.bilirubin_max
FROM cohort c
LEFT JOIN resp_sofa    rs ON c.stay_id = rs.stay_id
LEFT JOIN cardio_sofa  cs ON c.stay_id = cs.stay_id
LEFT JOIN hepatic_sofa hs ON c.stay_id = hs.stay_id
LEFT JOIN coag_sofa    gs ON c.stay_id = gs.stay_id
LEFT JOIN renal_sofa   re ON c.stay_id = re.stay_id
LEFT JOIN neuro_sofa   ns ON c.stay_id = ns.stay_id;