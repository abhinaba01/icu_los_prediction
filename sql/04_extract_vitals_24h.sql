-- Vital signs: mean of first 24h of ICU stay.
WITH vital_items AS (
    SELECT * FROM (VALUES
        (220045, 'heart_rate',       20,   300),
        (220277, 'spo2',             70,   100),
        (220210, 'resp_rate',         4,    60),
        (223761, 'temp_f',           85,   108),
        (223762, 'temp_c',           28,    42),
        (220739, 'gcs_eye',           1,     4),
        (223900, 'gcs_verbal',        1,     5),
        (223901, 'gcs_motor',         1,     6)
    ) AS v(itemid, feature_name, val_min, val_max)
),
raw_vitals AS (
    SELECT
        ce.subject_id,
        ce.hadm_id,
        ce.stay_id,
        vi.feature_name,
        ce.valuenum,
        ce.charttime,
        b.intime
    FROM mimiciv_icu.chartevents ce
    INNER JOIN vital_items vi ON ce.itemid = vi.itemid
    INNER JOIN base_cohort b ON ce.stay_id = b.stay_id
    WHERE
        ce.valuenum IS NOT NULL
        AND ce.valuenum BETWEEN vi.val_min AND vi.val_max
        AND ce.charttime >= b.intime
        AND ce.charttime <= b.intime + INTERVAL '24 hours'
),
temp_unified AS (
    SELECT
        stay_id,
        charttime,
        'temperature' AS feature_name,
        CASE
            WHEN feature_name = 'temp_f' THEN (valuenum - 32.0) * 5.0 / 9.0
            ELSE valuenum
        END AS valuenum
    FROM raw_vitals
    WHERE feature_name IN ('temp_f', 'temp_c')
),
non_temp AS (
    SELECT stay_id, charttime, feature_name, valuenum
    FROM raw_vitals
    WHERE feature_name NOT IN ('temp_f', 'temp_c')
),
all_vitals AS (
    SELECT * FROM non_temp
    UNION ALL
    SELECT * FROM temp_unified
)
SELECT
    stay_id,
    AVG(CASE WHEN feature_name = 'heart_rate' THEN valuenum END) AS heart_rate_mean,
    AVG(CASE WHEN feature_name = 'spo2' THEN valuenum END) AS spo2_mean,
    AVG(CASE WHEN feature_name = 'resp_rate' THEN valuenum END) AS resp_rate_mean,
    AVG(CASE WHEN feature_name = 'temperature' THEN valuenum END) AS temperature_mean,
    AVG(CASE WHEN feature_name = 'gcs_eye' THEN valuenum END) AS gcs_eye_mean,
    AVG(CASE WHEN feature_name = 'gcs_verbal' THEN valuenum END) AS gcs_verbal_mean,
    AVG(CASE WHEN feature_name = 'gcs_motor' THEN valuenum END) AS gcs_motor_mean
FROM all_vitals
GROUP BY stay_id;
