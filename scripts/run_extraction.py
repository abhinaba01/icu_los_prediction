"""Run source-data extraction for the ICU LOS project."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.extractor import DataExtractor
from src.data.loader import MIMICLoader, load_config


def configure_logging(config: dict) -> None:
    """Configure project logging.

    Args:
        config: Parsed project configuration.
    """
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, config["project"].get("log_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_dir / "extraction.log"), logging.StreamHandler()],
    )


def main() -> None:
    """Run all configured extraction steps."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    configure_logging(config)
    loader = MIMICLoader(config, PROJECT_ROOT)
    extractor = DataExtractor(loader, config, PROJECT_ROOT)
    outputs = extractor.extract_all()
    logging.getLogger(__name__).info("Extraction completed for: %s", ", ".join(outputs))


if __name__ == "__main__":
    main()
