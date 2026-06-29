"""Extraction orchestration for MIMIC-IV source data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loader import MIMICLoader

LOGGER = logging.getLogger(__name__)


class DataExtractor:
    """Runs SQL queries or CSV-based extraction steps and saves interim files."""

    def __init__(self, loader: MIMICLoader, config: dict[str, Any], project_root: str | Path | None = None):
        """Initialize an extractor.

        Args:
            loader: Configured MIMIC loader.
            config: Parsed project configuration.
            project_root: Project root path.
        """
        self.loader = loader
        self.config = config
        self.project_root = Path(project_root or ".").resolve()
        self.interim_dir = self.project_root / "data" / "interim"
        self.interim_dir.mkdir(parents=True, exist_ok=True)

    def extract_all(self) -> dict[str, pd.DataFrame]:
        """Run all extraction steps for the configured source.

        Returns:
            Mapping of extraction names to DataFrames.
        """
        if self.loader.source == "postgresql":
            outputs = self._extract_postgresql()
        else:
            outputs = self._extract_csv()
        for name, frame in outputs.items():
            self._save_interim(name, frame)
            LOGGER.info("Completed extraction: %s (%s rows)", name, len(frame))
        return outputs

    def _extract_postgresql(self) -> dict[str, pd.DataFrame]:
        """Run SQL extraction files against PostgreSQL.

        Returns:
            Mapping of extraction names to DataFrames.
        """
        return {
            "cohort": self.loader.run_sql_extract("cohort"),
            "diagnoses": self.loader.run_sql_extract("diagnoses"),
            "vitals": self.loader.run_sql_extract("vitals"),
            "labs": self.loader.run_sql_extract("labs"),
            "occupancy": self.loader.run_sql_extract("occupancy"),
        }

    def _extract_csv(self) -> dict[str, pd.DataFrame]:
        """Load raw CSV tables needed for local feature engineering.

        Returns:
            Mapping of raw extraction names to DataFrames.
        """
        table_names = [
            "icustays",
            "admissions",
            "patients",
            "diagnoses_icd",
            "chartevents",
            "labevents",
        ]
        return {name: self.loader.load_table(name) for name in table_names}

    def _save_interim(self, name: str, df: pd.DataFrame) -> None:
        """Save an interim DataFrame to parquet.

        Args:
            name: Output stem.
            df: DataFrame to save.
        """
        path = self.interim_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
