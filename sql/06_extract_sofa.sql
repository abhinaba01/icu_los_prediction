-- SOFA Score: Day 1, computed from first 24h of ICU stay
-- Adapted from MIT-LCP mimic-code/mimic-iv/concepts/score/sofa.sql
-- Components: Respiratory, Cardiovascular, Hepatic, Coagulation, Renal, Neurological

WITH cohort AS (
    SELECT stay_id, hadm_id, subject_id, intime, outtime
    FROM base_cohort  -- your existing cohort CTE
),

-- ----------------------------------------------------------------
-- 1. RESPIRATORY: PaO2/FiO2 ratio (estimate from SpO2/FiO2 if PaO2 absent)
--
-- Bug fixes vs. original:
--   * PaO2 (itemid 50821) is a LABEVENTS measurement, not chartevents.
--     Reading it from chartevents always returned NULL. Now sourced from
--     labevents (joined on hadm_id), aggregated separately to avoid any
--     cartesian product with the chartevents-derived SpO2/FiO2.
--   * SpO2 was previously scored with PaO2/FiO2 thresholds (invalid) and
--     used MAX(spo2) (best, not worst). Now we take the worst (MIN) SpO2
--     and convert SpO2/FiO2 -> estimated PaO2/FiO2 via the Rice et al.
--     (2007) relationship  S/F = 64 + 0.84 * (P/F)  =>  P/F = (S/F - 64)/0.84,
--     valid for SpO2 <= 97 percent, then apply the standard P/F thresholds once.
-- ----------------------------------------------------------------
pao2_24h AS (
    -- Worst (lowest) PaO2 in the first 24h, from blood-gas labs.
    SELECT c.stay_id,
           MIN(le.valuenum) AS pao2
    FROM cohort c
    JOIN mimiciv_hosp.labevents le
      ON le.hadm_id = c.hadm_id
     AND le.itemid = 50821
     AND le.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
     AND le.valuenum IS NOT NULL
    GROUP BY c.stay_id
),
spo2_fio2_24h AS (
    -- Worst (lowest) SpO2 and highest FiO2 requirement in the first 24h.
    SELECT c.stay_id,
           MIN(CASE WHEN ce.itemid = 220277 AND ce.valuenum BETWEEN 50 AND 100
                    THEN ce.valuenum END) AS spo2,
           MAX(CASE WHEN ce.itemid = 223835 THEN ce.valuenum END) AS fio2_chart
    FROM cohort c
    JOIN mimiciv_icu.chartevents ce ON ce.stay_id = c.stay_id
    WHERE ce.itemid IN (220277, 223835)
      AND ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
      AND ce.valuenum IS NOT NULL
    GROUP BY c.stay_id
),
resp_ratio AS (
    -- Normalise FiO2 to a fraction (chartevents records it as a percent,
    -- occasionally already as a fraction), then derive a single P/F ratio:
    -- prefer measured PaO2/FiO2; otherwise estimate from SpO2/FiO2.
    SELECT stay_id,
           CASE
               WHEN fio2_frac IS NULL OR fio2_frac = 0 THEN NULL
               WHEN pao2 IS NOT NULL THEN pao2 / fio2_frac
               WHEN spo2 IS NOT NULL AND spo2 <= 97
                   THEN ((spo2 / fio2_frac) - 64.0) / 0.84
               ELSE NULL
           END AS pf_ratio
    FROM (
        SELECT c.stay_id,
               CASE
                   WHEN sf.fio2_chart IS NULL THEN NULL
                   WHEN sf.fio2_chart > 1.0 THEN sf.fio2_chart / 100.0
                   ELSE sf.fio2_chart
               END AS fio2_frac,
               p.pao2,
               sf.spo2
        FROM cohort c
        LEFT JOIN pao2_24h      p  ON p.stay_id  = c.stay_id
        LEFT JOIN spo2_fio2_24h sf ON sf.stay_id = c.stay_id
    ) resp_inputs
),
resp_sofa AS (
    SELECT stay_id,
           CASE
               WHEN pf_ratio IS NULL THEN 0   -- not assessable -> 0 per project convention
               WHEN pf_ratio < 100 THEN 4
               WHEN pf_ratio < 200 THEN 3
               WHEN pf_ratio < 300 THEN 2
               WHEN pf_ratio < 400 THEN 1
               ELSE 0
           END AS sofa_resp
    FROM resp_ratio
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
--
-- Bug fix vs. original: creatinine (labevents) and urine output
-- (outputevents) were LEFT JOINed in a single CTE and then aggregated,
-- producing a cartesian product. SUM(urine) was multiplied by the number
-- of creatinine rows, badly inflating urine_24h and breaking the
-- urine-based thresholds. Each source is now aggregated to one row per
-- stay BEFORE being combined.
-- ----------------------------------------------------------------
creat_24h AS (
    SELECT c.stay_id,
           MAX(cr.valuenum) AS creatinine_max
    FROM cohort c
    JOIN mimiciv_hosp.labevents cr
      ON cr.hadm_id = c.hadm_id
     AND cr.itemid = 50912
     AND cr.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
     AND cr.valuenum BETWEEN 0.1 AND 20
    GROUP BY c.stay_id
),
urine_24h_cte AS (
    SELECT c.stay_id,
           SUM(oe.value) AS urine_24h
    FROM cohort c
    JOIN mimiciv_icu.outputevents oe
      ON oe.stay_id = c.stay_id
     AND oe.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
     AND oe.value IS NOT NULL
     AND oe.itemid IN (226559, 226560, 226561, 226584,
                       226563, 226564, 226565, 226567,
                       226557, 226558, 227488, 227489)
    GROUP BY c.stay_id
),
renal_sofa AS (
    SELECT c.stay_id,
           cr.creatinine_max,
           ur.urine_24h,
           CASE
               WHEN cr.creatinine_max >= 5.0
                 OR ur.urine_24h < 200 THEN 4
               WHEN cr.creatinine_max >= 3.5
                 OR ur.urine_24h < 500 THEN 3
               WHEN cr.creatinine_max >= 2.0 THEN 2
               WHEN cr.creatinine_max >= 1.2 THEN 1
               ELSE 0
           END AS sofa_renal
    FROM cohort c
    LEFT JOIN creat_24h     cr ON cr.stay_id = c.stay_id
    LEFT JOIN urine_24h_cte ur ON ur.stay_id = c.stay_id
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