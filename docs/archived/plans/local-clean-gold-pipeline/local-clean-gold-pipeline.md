# Local Task Plan: Clean Gold & Self-Healing Weather Pipeline

## Plan / Requirements
- [x] Strict Gold data quality in `src/city_pulse/pipelines/add_live_datas.py`:
  - [x] Remove `fill_null(0.0)` on weather columns.
  - [x] Implement hourly tolerance: drop any row where traffic or weather is missing/null (inner join / null-drop).
  - [x] Never write rows with missing weather or missing traffic to Gold.
- [x] Self-healing weather ingestion (Option A):
  - [x] If required live weather data is missing in `live_weather_dir`, automatically attempt to fetch today's forecast via `OpenMeteoClient` and save to Silver.
  - [x] If weather fetch fails or no valid records match, abort without touching Gold or archiving Silver traffic files.
- [x] DAG orchestration in `dags/traffic_gold_update_dag.py`:
  - [x] Thin DAG structure calling `add_live_datas_to_gold` with proper error handling and logging.
- [x] Testing & Verification:
  - [x] Update `tests/unit/pipelines/test_add_live_datas.py` with tests for strict validation, hourly tolerance, and self-healing.
  - [x] Run `uv run pytest tests/unit/pipelines/test_add_live_datas.py` and full suite (41 passed).
  - [x] Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src` (All clean).
- [x] Data Cleanup (Live test data reset):
  - [x] Remove live test rows (> 2025-12-31) from `data/gold/city_pulse_hourly_2023-01-01_2025-12-31.parquet`.
  - [x] Clean live Bronze & Silver files (`data/bronze/traffic_live/`, `data/silver/traffic_live/`, `data/silver/weather_live/`).

## Expected Verification
- [x] `uv run pytest tests/unit/pipelines/test_add_live_datas.py` (6 passed)
- [x] `uv run pytest` (41 passed)
- [x] `uv run ruff check .` (All checks passed)
- [x] `uv run ruff format --check .` (62 files formatted)
- [x] `uv run mypy src` (Success: no issues in 39 source files)

## Progress
- [x] Initial requirement analysis and user options confirmed.
- [x] Implementation in `add_live_datas.py`.
- [x] DAG updates in `traffic_gold_update_dag.py`.
- [x] Unit tests updated.
- [x] Full checks passing.
- [x] Cleaned Gold and live directories.

## Blockers / Open Questions
- None. Everything verified and passing.
