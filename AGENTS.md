# AGENTS.md — Istanbul City Pulse

This document defines the core architecture, operational boundaries, working agreements, and verification standards for all AI agents and coding assistants working on **Istanbul City Pulse**. Explicit user instructions take precedence over repository guidance.

---

## Repository Instructions

**Istanbul City Pulse** is an end-to-end modern Data Engineering and MLOps platform. It ingests urban mobility and weather data (İBB Open Data Portal & weather APIs), stores and models data across Bronze/Silver/Gold layers (DuckDB, PostgreSQL, Parquet), transforms metrics via dbt, orchestrates workflows with Apache Airflow, and trains predictive machine learning models (LightGBM, MLflow) to analyze and forecast city dynamics.

Work normally starts at the Git root. The repository layout and subproject responsibilities are:
- `src/city_pulse/`: Core application and domain code (`domain`, `ingest`, `storage`, `pipelines`, `dashboard`).
- `dags/`: Apache Airflow DAG files (keep DAGs thin; orchestration only; business logic resides in `src/`).
- `dbt/`: dbt transformation project (`models/staging`, `models/intermediate`, `models/marts`, `seeds/`, `schema.yml`).
- `docker/`: Service Dockerfiles and runtime container definitions (`docker-compose.yml`).
- `data/`: Local raw and processed datasets (Bronze/Silver/Gold) — strictly gitignored, never committed to VCS.
- `tests/`: Automated test suite (`tests/unit`, `tests/integration`).
- `ZZ_NOTLAR/`: Developer's personal notes — read-only; never modify automatically.

---

## Rule Routing

The scoped rules below apply to specific subsystems across the repository. Read and apply them when touching corresponding modules:

- **For Core Python (`src/city_pulse/`):**
  - **SOLID & Pythonic:** Enforce single responsibility (separate I/O from pure logic), dependency inversion via `Protocol` interfaces, and composition over inheritance (`@dataclass(frozen=True, slots=True)`, `NamedTuple`, `Enum`).
  - **Layer Dependency Direction:** Dependencies flow strictly inward: `domain ← ingest / transform / storage ← pipelines ← dags / CLI`. Lower layers never import upper layers; circular imports are prohibited.
  - **Data Processing:** Default to `polars` or `duckdb`. Reserve `pandas` exclusively for ML model input boundaries where required.
  - **Type Safety & Validation:** Complete type hints (`typing` / built-in types). All external API responses validated via **Pydantic (v2)** models with explicit column types. Configurations loaded via `pydantic-settings` from `.env`.
  - **Storage & Ingestion:** Bronze layer is raw and immutable. Store all timestamps in **UTC**. Guarantee idempotency and atomic writes (write to temporary file, then move). Tag every record with `source` and `fetched_at_utc`. Never silently fill missing values.

- **For Airflow DAGs (`dags/`):**
  - Keep DAG definitions thin; no heavy computations, transforms, or network calls at the top-level module scope.
  - Never use `datetime.now()` inside DAGs; rely on Airflow's `logical_date` / `data_interval`.
  - Tasks must be idempotent and accept explicit execution dates. Keep XCom payloads minimal (file paths, run IDs).

- **For dbt Transformations (`dbt/`):**
  - Layer progression: `staging → intermediate → marts`.
  - Document the grain of each model. Enforce `unique` and `not_null` constraints on primary keys and `relationships` tests on foreign keys.
  - In mart models, reference upstream models using `ref()`, never `source()`.

- **For Machine Learning & Analytics:**
  - **No Data Leakage:** Train/validation/test splits must strictly follow chronological (time-based) order.
  - **Feature Integrity:** Separate observed weather from forecast weather; evaluate training-serving skew.
  - **Baselines & Reproducibility:** Always benchmark against a naive baseline (e.g. day-of-week × hour mean). Pin random seeds, store training snapshots, and log parameters/metrics to MLflow. Champion/challenger evaluation before promotion.

---

## Working Agreements

- **Scope Discipline (Highest Priority):**
  - Do only what is requested. Never add unrequested features, files, dependencies, directories, configurations, automations, or docs. Unsolicited additions are treated as bugs.
  - Suggest out-of-scope ideas at the end of the response; do not implement them without explicit user approval.
  - If requirements are ambiguous, ask a concise clarifying question rather than guessing.
  - Preserve existing working code. Do not reformat or reorganize unrelated files. Keep diffs focused and reviewable.
- **Honest Reporting & Grounded Facts:**
  - Never claim code or tests run if they were not executed. Explicitly state what was verified (`ruff`, `mypy`, `pytest`) and what could not be executed (e.g. live network, external APIs).
  - Never hallucinate API endpoints, column names, library methods, or dataset fields.
- **Domain Realities (Verified Facts):**
  - **İBB Traffic Index CSV:** Daily city-wide only (single row per day; columns: `minimum/maximum/average_traffic_index`). No spatial or hourly granularity. Dataset ends 2024-10-18; discard corrupted final row.
  - **Hourly/Spatial Traffic:** Requires verified monthly datasets; verify schemas before writing code.
  - **Live Traffic API:** No official live endpoint is documented. Use CKAN `package_search` to discover portal packages.
  - **Open-Meteo ERA5:** Reanalysis model (CC BY 4.0; attribution required in README). Meteosource free tier is capped at 400 calls/day.
- **Language & Communication:**
  - Identifiers (variables, functions, classes, columns), commit messages, and task plans must be in **English**.
  - Docstrings, inline code comments, and user explanations should be in **Turkish** (clear, concise, educational).
  - Use **Conventional Commits** (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`). Do not commit or push unless explicitly requested.

---

## Commands and Dependencies

- **Package Management:** Strictly use `uv`. Never use `pip`.
  - Synchronize environment: `uv sync` or `uv sync --all-groups` (or `make sync` / `make install`).
  - Add dependency: Only when explicitly requested and justified: `uv add <pkg>` or `uv add --group <dev|ml|dbt|quality> <pkg>`.
  - Lockfile integrity: Never edit `uv.lock` manually.
- **Command Execution (Host vs. Docker):**
  - Run Python scripts, tests, linters, and type checking via `uv run` or corresponding `Makefile` targets.
  - Run multi-service infrastructure (Airflow, PostgreSQL, containers) via Docker Compose (`make up`, `make down`, `docker compose run --rm <service> <command>`).
- **Standard Makefile Targets:**
  - Verification & Linting: `make check` (runs ruff check, ruff format check, mypy, pytest), `make lint`, `make format`.
  - Testing: `make test` (`uv run pytest`), `make test-cov`.
  - dbt: `make dbt-run`, `make dbt-test`, `make dbt-core`.
  - Services & ML: `make up`, `make down`, `make logs`, `make mlflow`, `make dashboard`.

---

## Task Plans

- Before editing project files on non-trivial tasks, create or update a task plan under `docs/plans/`:
  - Numbered tasks: `docs/plans/issue-<number>/issue-<number>.md`.
  - Local/ad-hoc tasks: `docs/plans/local-<short-name>/local-<short-name>.md`.
- Include Markdown checklists for:
  - Plan / Requirements
  - Expected Verification
  - Progress
  - Blockers / Open Questions
- Update completed items as work progresses and record reasons for any skipped checks.
- Put detailed workstream notes in `scratchpads/` or personal notes in `ZZ_NOTLAR/` only when needed. When the task is complete, move its plan directory to `docs/archived/plans/`.

---

## Verification

- **Core Python Code & Configuration:** Run `make check` (or `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src`, `uv run pytest`).
- **dbt Models & Data Pipelines:** Run `make dbt-test` (`uv run --group dbt dbt test --project-dir dbt --profiles-dir dbt`).
- **Airflow DAGs:** Run DAG integrity and syntax validation tests (`uv run pytest tests/unit/dags/` or Airflow DAG check).
- **Cross-Stack & Infrastructure Changes:** Run applicable tests for both Python/dbt layers and verify container health in Docker Compose.
- **Documentation-only Changes:** No tests required unless the documentation describes executable behavior.
- **Definition of Done:** A task is only complete when all applicable checks pass without warnings or errors. If a required check fails, fix and rerun it. Record unresolved external blockers in the task plan.
- Keep the root `AGENTS.md` and `CLAUDE.md` aligned when changing agent guidance.
