-- Demographics are embedded in 01_extract_icustays.sql.
-- age: mimiciv_hosp.patients.anchor_age
-- gender: mimiciv_hosp.patients.gender
SELECT
    p.subject_id,
    p.anchor_age AS age,
    p.gender
FROM mimiciv_hosp.patients p;
