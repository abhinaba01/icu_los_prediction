# ICU LOS Prediction

End-to-end research pipeline for replicating the two-stage ICU length-of-stay
prediction workflow from Hempel et al. (2023), extending it with ICU bed
occupancy features, and running an ablation study.

The project supports both MIMIC-IV CSV files and PostgreSQL. The default
PostgreSQL configuration targets database `postgres` with schemas
`mimiciv_hosp`, `mimiciv_icu`, and `mimiciv_derived`.

## Setup

```powershell
cd los_prediction
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For PostgreSQL, set `data.source: postgresql` in `config.yaml`. Passwords can
be supplied through `MIMIC_PG_PASSWORD`.

For CSV, place files under `data/raw/` according to the paths in
`config.yaml`.

## Main Commands

```powershell
python scripts/run_extraction.py
python scripts/run_hempel.py
python scripts/run_extended.py
python scripts/run_ablation.py
```

Each script saves tables, figures, models, and predictions under `results/`.
The Hempel and extended runs share `results/split_indices.npz` so the ablation
uses the same patients in train, validation, and test sets.

## Tests

```powershell
pytest
```

The tests use synthetic data and check feature builders, preprocessing leakage
guards, model smoke tests, and the occupancy calculation.
