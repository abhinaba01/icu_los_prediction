"""Unified MIMIC-IV loading utilities for CSV and PostgreSQL sources."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)


SQL_EXTRACT_MAP = {
    "cohort": "01_extract_icustays.sql",
    "icustays": "01_extract_icustays.sql",
    "demographics": "02_extract_demographics.sql",
    "diagnoses": "03_extract_diagnoses.sql",
    "vitals": "04_extract_vitals_24h.sql",
    "labs": "05_extract_labs_24h.sql",
    "occupancy": "06_extract_icu_occupancy.sql",
}


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class MIMICLoader:
    """Unified loader for MIMIC-IV data via PostgreSQL or CSV files."""

    def __init__(self, config: dict[str, Any], project_root: str | Path | None = None):
        """Initialize the loader from a config dictionary.

        Args:
            config: Parsed project configuration.
            project_root: Project root containing `config.yaml`, `sql/`, and
                `data/`. Defaults to the current working directory.

        Raises:
            ValueError: If the configured source is unsupported.
        """
        self.config = config
        self.project_root = Path(project_root or ".").resolve()
        self.source = config["data"].get("source", "csv").lower()
        self.sql_dir = self.project_root / "sql"
        self._engine = None
        if self.source not in {"csv", "postgresql"}:
            raise ValueError("data.source must be either 'csv' or 'postgresql'")

    @property
    def engine(self):
        """Return a lazily created SQLAlchemy engine for PostgreSQL.

        Returns:
            SQLAlchemy engine connected to the configured PostgreSQL database.

        Raises:
            ImportError: If SQLAlchemy is not installed.
        """
        if self._engine is None:
            from sqlalchemy import create_engine

            pg = self.config["data"]["postgresql"]
            password = os.getenv("MIMIC_PG_PASSWORD", pg.get("password", ""))
            url = (
                f"postgresql+psycopg2://{pg['user']}:{password}"
                f"@{pg['host']}:{pg['port']}/{pg['dbname']}"
            )
            self._engine = create_engine(url)
        return self._engine

    def load_table(self, table_name: str) -> pd.DataFrame:
        """Load a named MIMIC-IV table.

        Args:
            table_name: Table name such as `icustays`, `admissions`,
                `patients`, `diagnoses_icd`, `d_icd_diagnoses`, `chartevents`,
                or `labevents`.

        Returns:
            Loaded table as a DataFrame.

        Raises:
            FileNotFoundError: If a configured CSV path is missing.
            ValueError: If the table cannot be mapped to a source.
        """
        source = self._source_for_table(table_name)
        if source == "csv":
            return self._load_csv_table(table_name)
        return self._load_postgresql_table(table_name)

    def iter_csv_table(
        self,
        table_name: str,
        chunksize: int | None = None,
        usecols: list[str] | None = None,
    ):
        """Iterate over a configured CSV or CSV.GZ table in chunks.

        Args:
            table_name: Key under `data.csv.files`.
            chunksize: Number of rows per chunk. Defaults to
                `data.csv_chunksize` or 1,000,000.
            usecols: Optional subset of columns to read.

        Yields:
            DataFrame chunks from the configured CSV file.
        """
        path = self._csv_path(table_name)
        row_count = chunksize or int(self.config["data"].get("csv_chunksize", 1_000_000))
        LOGGER.info("Streaming CSV table %s from %s in chunks of %s", table_name, path, row_count)
        yield from pd.read_csv(path, chunksize=row_count, usecols=usecols, low_memory=False)

    def run_sql(self, sql_path: str | Path, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Execute a SQL file and return results as a DataFrame.

        Args:
            sql_path: Path to a SQL file. Relative paths resolve against
                `sql/`.
            params: Optional query parameters for pandas/SQLAlchemy.

        Returns:
            Query result as a DataFrame.

        Raises:
            ValueError: If called while not using PostgreSQL source.
        """
        if self.source != "postgresql":
            raise ValueError("run_sql requires data.source='postgresql'")
        path = Path(sql_path)
        if not path.is_absolute():
            path = self.sql_dir / path
        query = self._render_sql(path.read_text(encoding="utf-8"))
        return pd.read_sql_query(query, self.engine, params=params)

    def run_sql_extract(self, extract_name: str) -> pd.DataFrame:
        """Run a named extraction query from the `sql/` directory.

        Args:
            extract_name: Logical extraction name such as `cohort`, `diagnoses`,
                `vitals`, `labs`, or `occupancy`.

        Returns:
            Extraction result as a DataFrame.

        Raises:
            KeyError: If the extraction name is unknown.
        """
        sql_file = SQL_EXTRACT_MAP[extract_name]
        if extract_name in {"vitals", "labs", "occupancy"}:
            return self.run_sql_with_base_cohort(self.sql_dir / sql_file)
        return self.run_sql(self.sql_dir / sql_file)

    def run_sql_with_base_cohort(self, sql_path: str | Path) -> pd.DataFrame:
        """Execute a SQL file that references a temporary `base_cohort` table.

        Args:
            sql_path: SQL file that expects `base_cohort` to exist.

        Returns:
            Query result as a DataFrame.
        """
        from sqlalchemy import text

        base_query = self._render_sql((self.sql_dir / SQL_EXTRACT_MAP["cohort"]).read_text(encoding="utf-8"))
        target_query = self._render_sql(Path(sql_path).read_text(encoding="utf-8"))
        base_query = base_query.rstrip().rstrip(";")
        with self.engine.begin() as connection:
            connection.execute(text("DROP TABLE IF EXISTS pg_temp.base_cohort"))
            connection.execute(text(f"CREATE TEMP TABLE base_cohort AS {base_query}"))
            return pd.read_sql_query(target_query, connection)

    def _load_csv_table(self, table_name: str) -> pd.DataFrame:
        """Load a configured CSV table.

        Args:
            table_name: Key under `data.csv.files`.

        Returns:
            CSV contents as a DataFrame.

        Raises:
            FileNotFoundError: If neither the plain nor gzipped path exists.
        """
        path = self._csv_path(table_name)
        LOGGER.info("Loading CSV table %s from %s", table_name, path)
        return pd.read_csv(path, low_memory=False)

    def _csv_path(self, table_name: str) -> Path:
        """Resolve a configured CSV path, accepting `.csv` or `.csv.gz`.

        Args:
            table_name: Key under `data.csv.files`.

        Returns:
            Existing CSV or CSV.GZ path.

        Raises:
            FileNotFoundError: If neither path exists.
        """
        csv_config = self.config["data"]["csv"]
        relative = Path(csv_config["files"][table_name])
        path = self.project_root / csv_config["base_path"] / relative
        if not path.exists() and path.with_suffix(path.suffix + ".gz").exists():
            path = path.with_suffix(path.suffix + ".gz")
        if not path.exists():
            raise FileNotFoundError(f"CSV table not found for {table_name}: {path}")
        return path

    def _source_for_table(self, table_name: str) -> str:
        """Return the configured source for a table.

        Args:
            table_name: MIMIC table name.

        Returns:
            Source name, either `csv` or `postgresql`.

        Raises:
            ValueError: If the override source is unsupported.
        """
        source = self.config["data"].get("table_sources", {}).get(table_name, self.source).lower()
        if source not in {"csv", "postgresql"}:
            raise ValueError(f"Unsupported source for {table_name}: {source}")
        return source

    def _load_postgresql_table(self, table_name: str) -> pd.DataFrame:
        """Load a raw PostgreSQL table from the appropriate MIMIC schema.

        Args:
            table_name: Raw MIMIC table name.

        Returns:
            Database table as a DataFrame.

        Raises:
            ValueError: If the table name cannot be mapped to a schema.
        """
        pg = self.config["data"]["postgresql"]
        if table_name in {"icustays", "chartevents", "d_items"}:
            schema = pg["schema_icu"]
        elif table_name in {"admissions", "patients", "diagnoses_icd", "d_icd_diagnoses", "labevents", "d_labitems"}:
            schema = pg["schema_hosp"]
        else:
            raise ValueError(f"Unknown PostgreSQL table: {table_name}")
        LOGGER.info("Loading PostgreSQL table %s.%s", schema, table_name)
        return pd.read_sql_table(table_name, self.engine, schema=schema)

    def _render_sql(self, sql: str) -> str:
        """Render SQL with configured schema names.

        Args:
            sql: SQL text that may contain default MIMIC schema names.

        Returns:
            SQL text with schema names replaced from config.
        """
        pg = self.config["data"]["postgresql"]
        rendered = sql.replace("mimiciv_hosp", pg["schema_hosp"])
        rendered = rendered.replace("mimiciv_icu", pg["schema_icu"])
        rendered = rendered.replace("mimiciv_derived", pg.get("schema_derived", "mimiciv_derived"))
        return rendered
