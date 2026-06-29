-- Base ICU stay cohort.
-- Exclusions: age < 18, LOS < 4h, LOS > 30 days, non-first ICU stays.
WITH first_icu AS (
    SELECT
        ie.subject_id,
        ie.hadm_id,
        ie.stay_id,
        ie.first_careunit,
        ie.last_careunit,
        ie.intime,
        ie.outtime,
        EXTRACT(EPOCH FROM (ie.outtime - ie.intime)) / 3600.0 AS los_hours,
        EXTRACT(EPOCH FROM (ie.outtime - ie.intime)) / 86400.0 AS los_days,
        ROW_NUMBER() OVER (
            PARTITION BY ie.subject_id
            ORDER BY ie.intime ASC
        ) AS stay_rank
    FROM mimiciv_icu.icustays ie
),
patient_age AS (
    SELECT
        p.subject_id,
        p.gender,
        p.anchor_age AS age
    FROM mimiciv_hosp.patients p
)
SELECT
    f.subject_id,
    f.hadm_id,
    f.stay_id,
    f.first_careunit,
    f.last_careunit,
    f.intime,
    f.outtime,
    f.los_hours,
    f.los_days,
    pa.gender,
    pa.age
FROM first_icu f
INNER JOIN patient_age pa ON f.subject_id = pa.subject_id
WHERE
    f.stay_rank = 1
    AND f.los_hours >= 4.0
    AND f.los_days <= 30.0
    AND pa.age >= 18
ORDER BY f.subject_id, f.stay_id;
