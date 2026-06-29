-- Lab values: mean of first 24h window anchored at ICU intime.
WITH lab_items AS (
    SELECT * FROM (VALUES
        (50868,  'anion_gap',     1,    40),
        (50882,  'bicarbonate',   5,    50),
        (50902,  'chloride',     70,   140),
        (50912,  'creatinine',  0.1,    20),
        (50931,  'glucose',      33,  1000),
        (50983,  'sodium',      100,   180),
        (50960,  'magnesium',   0.5,     5),
        (50971,  'potassium',   1.5,    10),
        (50970,  'phosphate',   0.5,    10),
        (51006,  'bun',           1,   150),
        (51221,  'hematocrit',    5,    65),
        (51222,  'hemoglobin',    2,    22),
        (51248,  'mch',          15,    45),
        (51249,  'mchc',         20,    45),
        (51250,  'mcv',          50,   130),
        (51277,  'rdw',           9,    35),
        (51279,  'rbc',           1,    10),
        (51301,  'wbc',         0.1,    80),
        (51265,  'platelets',     5,  1500)
    ) AS l(itemid, feature_name, val_min, val_max)
),
raw_labs AS (
    SELECT
        le.subject_id,
        le.hadm_id,
        li.feature_name,
        le.valuenum,
        le.charttime,
        b.intime
    FROM mimiciv_hosp.labevents le
    INNER JOIN lab_items li ON le.itemid = li.itemid
    INNER JOIN base_cohort b ON le.hadm_id = b.hadm_id
    WHERE
        le.valuenum IS NOT NULL
        AND le.valuenum BETWEEN li.val_min AND li.val_max
        AND le.charttime >= b.intime
        AND le.charttime <= b.intime + INTERVAL '24 hours'
)
SELECT
    hadm_id,
    AVG(CASE WHEN feature_name = 'anion_gap' THEN valuenum END) AS anion_gap_mean,
    AVG(CASE WHEN feature_name = 'bicarbonate' THEN valuenum END) AS bicarbonate_mean,
    AVG(CASE WHEN feature_name = 'chloride' THEN valuenum END) AS chloride_mean,
    AVG(CASE WHEN feature_name = 'creatinine' THEN valuenum END) AS creatinine_mean,
    AVG(CASE WHEN feature_name = 'glucose' THEN valuenum END) AS glucose_mean,
    AVG(CASE WHEN feature_name = 'sodium' THEN valuenum END) AS sodium_mean,
    AVG(CASE WHEN feature_name = 'magnesium' THEN valuenum END) AS magnesium_mean,
    AVG(CASE WHEN feature_name = 'potassium' THEN valuenum END) AS potassium_mean,
    AVG(CASE WHEN feature_name = 'phosphate' THEN valuenum END) AS phosphate_mean,
    AVG(CASE WHEN feature_name = 'bun' THEN valuenum END) AS bun_mean,
    AVG(CASE WHEN feature_name = 'hematocrit' THEN valuenum END) AS hematocrit_mean,
    AVG(CASE WHEN feature_name = 'hemoglobin' THEN valuenum END) AS hemoglobin_mean,
    AVG(CASE WHEN feature_name = 'mch' THEN valuenum END) AS mch_mean,
    AVG(CASE WHEN feature_name = 'mchc' THEN valuenum END) AS mchc_mean,
    AVG(CASE WHEN feature_name = 'mcv' THEN valuenum END) AS mcv_mean,
    AVG(CASE WHEN feature_name = 'rdw' THEN valuenum END) AS rdw_mean,
    AVG(CASE WHEN feature_name = 'rbc' THEN valuenum END) AS rbc_mean,
    AVG(CASE WHEN feature_name = 'wbc' THEN valuenum END) AS wbc_mean,
    AVG(CASE WHEN feature_name = 'platelets' THEN valuenum END) AS platelets_mean
FROM raw_labs
GROUP BY hadm_id;
