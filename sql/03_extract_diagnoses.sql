-- Extract top-2 ICD-10 diagnoses per admission, mapped to ICD-10 chapters.
WITH ranked_diag AS (
    SELECT
        d.subject_id,
        d.hadm_id,
        d.icd_code,
        d.icd_version,
        d.seq_num,
        ROW_NUMBER() OVER (
            PARTITION BY d.hadm_id, d.icd_version
            ORDER BY d.seq_num ASC
        ) AS diag_rank
    FROM mimiciv_hosp.diagnoses_icd d
    WHERE d.icd_version = 10
),
top2 AS (
    SELECT subject_id, hadm_id, icd_code, diag_rank
    FROM ranked_diag
    WHERE diag_rank <= 2
),
chapter_mapped AS (
    SELECT
        subject_id,
        hadm_id,
        diag_rank,
        icd_code,
        CASE
            WHEN icd_code ~ '^[AB]' THEN 1
            WHEN icd_code ~ '^C|^D[0-4]' THEN 2
            WHEN icd_code ~ '^D[5-8]' THEN 3
            WHEN icd_code ~ '^E' THEN 4
            WHEN icd_code ~ '^F' THEN 5
            WHEN icd_code ~ '^G' THEN 6
            WHEN icd_code ~ '^H[0-5]' THEN 7
            WHEN icd_code ~ '^H[6-9]' THEN 8
            WHEN icd_code ~ '^I' THEN 9
            WHEN icd_code ~ '^J' THEN 10
            WHEN icd_code ~ '^K' THEN 11
            WHEN icd_code ~ '^L' THEN 12
            WHEN icd_code ~ '^M' THEN 13
            WHEN icd_code ~ '^N' THEN 14
            WHEN icd_code ~ '^O' THEN 15
            WHEN icd_code ~ '^P' THEN 16
            WHEN icd_code ~ '^Q' THEN 17
            WHEN icd_code ~ '^R' THEN 18
            WHEN icd_code ~ '^[ST]' THEN 19
            WHEN icd_code ~ '^[VWX-Y]' THEN 20
            WHEN icd_code ~ '^Z' THEN 21
            WHEN icd_code ~ '^U' THEN 22
            ELSE 0
        END AS icd10_chapter
    FROM top2
)
SELECT
    subject_id,
    hadm_id,
    MAX(CASE WHEN diag_rank = 1 THEN icd10_chapter END) AS diag1_chapter,
    MAX(CASE WHEN diag_rank = 2 THEN icd10_chapter END) AS diag2_chapter
FROM chapter_mapped
GROUP BY subject_id, hadm_id;
