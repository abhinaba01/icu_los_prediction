#!/usr/bin/env bash
#
# Append the SOFA item ids that are MISSING from a subset-loaded Postgres
# (bilirubin/PaO2 in labevents; FiO2/SpO2/MAP in chartevents) by extracting
# just those rows from the full .csv.gz files. Existing rows (creatinine,
# platelets, GCS, ...) are left untouched.
#
# Run from Git Bash on Windows:
#   export PGPASSWORD='your_password'
#   bash scripts/append_sofa_items.sh
#
# Safe to re-run: each item set is DELETEd before it is re-loaded.
#
set -euo pipefail

# --- connection --------------------------------------------------------------
: "${PGHOST:=localhost}"
: "${PGPORT:=5432}"
: "${PGUSER:=postgres}"
: "${PGDATABASE:=postgres}"
export PGHOST PGPORT PGUSER PGDATABASE
SCHEMA_HOSP="${SCHEMA_HOSP:-mimiciv_hosp}"
SCHEMA_ICU="${SCHEMA_ICU:-mimiciv_icu}"

if [[ -z "${PGPASSWORD:-}" ]]; then
    echo "ERROR: set PGPASSWORD first, e.g.  export PGPASSWORD='your_password'" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAB_GZ="$ROOT/data/raw/hosp/labevents.csv.gz"
CHART_GZ="$ROOT/data/raw/icu/chartevents.csv.gz"

for f in "$LAB_GZ" "$CHART_GZ"; do
    [[ -f "$f" ]] || { echo "ERROR: source file not found: $f" >&2; exit 1; }
done

psql_q() { psql -v ON_ERROR_STOP=1 -qtA "$@"; }

# append_items <schema> <table> <gz> <itemid_column_index> <itemid...>
append_items () {
    local schema="$1" table="$2" gz="$3" col="$4"; shift 4
    local items=("$@")
    echo ""
    echo ">>> ${schema}.${table}  <-  ${gz}   items: ${items[*]} (itemid is column ${col})"

    # Header -> column list; auto-add any column the table lacks (as TEXT).
    local header c collist=""
    header="$( { gzip -dc "$gz" || true; } | head -1 | tr -d '\r"' )"
    local -a cols=()
    IFS=',' read -ra cols <<< "$header"
    for c in "${cols[@]}"; do
        c="$(echo "$c" | xargs)"
        [[ -z "$c" ]] && continue
        psql_q -c "ALTER TABLE ${schema}.${table} ADD COLUMN IF NOT EXISTS \"${c}\" TEXT;" >/dev/null
        collist+="${collist:+,}${c}"
    done

    # Build the awk filter (header + matching itemid rows) and the SQL IN-list.
    local awk_cond="NR==1" inlist=""
    for it in "${items[@]}"; do
        awk_cond="${awk_cond} || \$${col}==\"${it}\""
        inlist+="${inlist:+,}${it}"
    done

    # Idempotent: clear just these items, then stream-filter + append.
    echo "    deleting existing rows for items (${inlist})..."
    psql_q -c "DELETE FROM ${schema}.${table} WHERE itemid IN (${inlist});"
    echo "    scanning CSV and appending matching rows..."
    gzip -dc "$gz" | awk -F, "$awk_cond" \
        | psql -v ON_ERROR_STOP=1 \
            -c "\copy ${schema}.${table} (${collist}) FROM STDIN WITH (FORMAT csv, HEADER true)"

    local n
    n="$(psql_q -c "SELECT count(*) FROM ${schema}.${table} WHERE itemid IN (${inlist});")"
    echo "    now ${n} rows for items (${inlist})."
}

# Respiratory needs FiO2 + (SpO2 or PaO2); cardio's MAP also lives in chartevents.
append_items "$SCHEMA_HOSP" labevents   "$LAB_GZ"   5 50885 50821
append_items "$SCHEMA_ICU"  chartevents "$CHART_GZ" 7 223835 220277 220052 220181 225312

echo ""
echo "Refreshing planner stats..."
psql_q -c "ANALYZE ${SCHEMA_HOSP}.labevents;"
psql_q -c "ANALYZE ${SCHEMA_ICU}.chartevents;"

echo ""
echo "Verification (should all be > 0 now):"
psql -v ON_ERROR_STOP=1 -c "
SELECT 'fio2  chartevents 223835'  AS item, count(*) FROM ${SCHEMA_ICU}.chartevents  WHERE itemid=223835
UNION ALL SELECT 'spo2  chartevents 220277', count(*) FROM ${SCHEMA_ICU}.chartevents  WHERE itemid=220277
UNION ALL SELECT 'map   chartevents 220052', count(*) FROM ${SCHEMA_ICU}.chartevents  WHERE itemid IN (220052,220181,225312)
UNION ALL SELECT 'pao2  labevents  50821',   count(*) FROM ${SCHEMA_HOSP}.labevents   WHERE itemid=50821
UNION ALL SELECT 'bili  labevents  50885',   count(*) FROM ${SCHEMA_HOSP}.labevents   WHERE itemid=50885;"

echo ""
echo "Done. Next: python scripts/run_extended.py  (expect the build_sofa guard to pass for all 6 components)."
