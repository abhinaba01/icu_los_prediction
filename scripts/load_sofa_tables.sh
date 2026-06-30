#!/usr/bin/env bash
#
# Load MIMIC-IV inputevents + outputevents (the two tables the SOFA extension
# needs) from .csv.gz into PostgreSQL. Run from Git Bash on Windows.
#
# Why this is header-driven:
#   PostgreSQL COPY ... CSV HEADER maps columns by POSITION, not by name.
#   To stay robust across MIMIC-IV versions (e.g. the caregiver_id column was
#   added in v2.2) we read each file's own header and COPY with an explicit
#   NAMED column list, so the file maps to the table by name in file order.
#   Any header column not already in the table is auto-added as TEXT.
#
# Usage:
#   export PGPASSWORD='your_password'      # required
#   # optional overrides (these are the defaults):
#   #   export PGHOST=localhost PGPORT=5432 PGUSER=postgres PGDATABASE=postgres
#   #   export SCHEMA_ICU=mimiciv_icu
#   bash scripts/load_sofa_tables.sh
#
set -euo pipefail

# --- connection (libpq env vars; avoids arg-quoting headaches) ---------------
: "${PGHOST:=localhost}"
: "${PGPORT:=5432}"
: "${PGUSER:=postgres}"
: "${PGDATABASE:=postgres}"
export PGHOST PGPORT PGUSER PGDATABASE
SCHEMA_ICU="${SCHEMA_ICU:-mimiciv_icu}"

# --- paths (resolve project root from this script's location) ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ICU_DIR="$ROOT/data/raw/icu"

# --- pre-flight: both files must be fully downloaded before we touch the DB --
missing=0
for f in inputevents outputevents; do
    if [[ ! -f "$ICU_DIR/$f.csv.gz" ]]; then
        echo "Waiting on: $ICU_DIR/$f.csv.gz (not present yet)"
        missing=1
    fi
done
if (( missing )); then
    if compgen -G "$ICU_DIR/*.crdownload" >/dev/null; then
        echo "Downloads still in progress (.crdownload present). Re-run when BOTH finish."
    fi
    exit 1
fi

if [[ -z "${PGPASSWORD:-}" ]]; then
    echo "ERROR: set PGPASSWORD first, e.g.  export PGPASSWORD='your_password'" >&2
    exit 1
fi

psql_q() { psql -v ON_ERROR_STOP=1 -qtA "$@"; }

# --- canonical typed schemas (v2.2/v3.1). Extra header cols are added later. --
psql_q <<SQL
CREATE TABLE IF NOT EXISTS ${SCHEMA_ICU}.inputevents (
    subject_id INTEGER, hadm_id INTEGER, stay_id INTEGER, caregiver_id INTEGER,
    starttime TIMESTAMP, endtime TIMESTAMP, storetime TIMESTAMP, itemid INTEGER,
    amount DOUBLE PRECISION, amountuom VARCHAR(20),
    rate DOUBLE PRECISION, rateuom VARCHAR(20),
    orderid INTEGER, linkorderid INTEGER,
    ordercategoryname VARCHAR(100), secondaryordercategoryname VARCHAR(100),
    ordercomponenttypedescription VARCHAR(200), ordercategorydescription VARCHAR(100),
    patientweight DOUBLE PRECISION, totalamount DOUBLE PRECISION, totalamountuom VARCHAR(100),
    isopenbag SMALLINT, continueinnextdept SMALLINT, statusdescription VARCHAR(50),
    originalamount DOUBLE PRECISION, originalrate DOUBLE PRECISION
);
CREATE TABLE IF NOT EXISTS ${SCHEMA_ICU}.outputevents (
    subject_id INTEGER, hadm_id INTEGER, stay_id INTEGER, caregiver_id INTEGER,
    charttime TIMESTAMP, storetime TIMESTAMP, itemid INTEGER,
    value DOUBLE PRECISION, valueuom VARCHAR(20)
);
SQL
echo "Tables ensured in schema ${SCHEMA_ICU}."

# --- load one table from its gz, driven by the file header -------------------
load_table () {
    local table="$1" gz="$2"
    echo ""
    echo ">>> ${SCHEMA_ICU}.${table}  <-  ${gz}"
    if [[ ! -f "$gz" ]]; then
        if compgen -G "${gz%.gz}.gz.crdownload" >/dev/null || compgen -G "${ICU_DIR}/*.crdownload" >/dev/null; then
            echo "    NOT READY: download still in progress (.crdownload present). Re-run when finished." >&2
        else
            echo "    MISSING: $gz not found." >&2
        fi
        return 1
    fi

    # Read header -> comma-separated column names (strip CR/quotes/space).
    local header
    header="$( { gzip -dc "$gz" || true; } | head -1 | tr -d '\r"' )"
    local -a cols=()
    local c collist=""
    IFS=',' read -ra cols <<< "$header"
    for c in "${cols[@]}"; do
        c="$(echo "$c" | xargs)"                 # trim whitespace
        [[ -z "$c" ]] && continue
        # Add as TEXT if the table doesn't already have this column (no-op if it does).
        psql_q -c "ALTER TABLE ${SCHEMA_ICU}.${table} ADD COLUMN IF NOT EXISTS \"${c}\" TEXT;" >/dev/null
        collist+="${collist:+,}${c}"
    done
    echo "    file columns: ${collist}"

    # Fresh load (idempotent on re-run).
    psql_q -c "TRUNCATE ${SCHEMA_ICU}.${table};"
    echo "    loading (streaming, no temp file)..."
    gzip -dc "$gz" | psql -v ON_ERROR_STOP=1 \
        -c "\copy ${SCHEMA_ICU}.${table} (${collist}) FROM STDIN WITH (FORMAT csv, HEADER true)"

    local n
    n="$(psql_q -c "SELECT count(*) FROM ${SCHEMA_ICU}.${table};")"
    echo "    loaded ${n} rows."
}

load_table outputevents "$ICU_DIR/outputevents.csv.gz"
load_table inputevents  "$ICU_DIR/inputevents.csv.gz"

# --- indexes the SOFA query relies on, then refresh planner stats ------------
echo ""
echo "Creating indexes + ANALYZE (speeds up the SOFA join/filters)..."
psql_q <<SQL
CREATE INDEX IF NOT EXISTS ie_stay_item ON ${SCHEMA_ICU}.inputevents (stay_id, itemid);
CREATE INDEX IF NOT EXISTS oe_stay_item ON ${SCHEMA_ICU}.outputevents (stay_id, itemid);
ANALYZE ${SCHEMA_ICU}.inputevents;
ANALYZE ${SCHEMA_ICU}.outputevents;
SQL

# --- quick sanity check: are the SOFA-relevant items actually present? -------
echo ""
echo "Sanity check (SOFA-relevant rows):"
psql -v ON_ERROR_STOP=1 -c "
SELECT 'vasopressors (cardio)' AS signal,
       count(*) AS rows
FROM ${SCHEMA_ICU}.inputevents
WHERE itemid IN (221906,221289,221662,221653,222315)
UNION ALL
SELECT 'urine output (renal)', count(*)
FROM ${SCHEMA_ICU}.outputevents
WHERE itemid IN (226559,226560,226561,226584,226563,226564,
                 226565,226567,226557,226558,227488,227489);"

echo ""
echo "Done. Next: python scripts/run_extended.py"
echo "If build_sofa's guard passes silently, SOFA now carries signal."
