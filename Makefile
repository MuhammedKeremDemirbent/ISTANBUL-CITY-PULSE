.PHONY: help install sync test test-cov lint format check up down logs clean

help:
	@echo ========================================================
	@echo Istanbul City Pulse - Project Command Manager
	@echo ========================================================
	@echo make install    : Synchronize all dependency groups
	@echo make test       : Run pytest unit tests
	@echo make test-cov   : Run tests with coverage report
	@echo make lint       : Check code quality with Ruff and Mypy
	@echo make format     : Format and auto-fix code with Ruff
	@echo make check      : Run full verification (lint, format check, mypy, pytest)
	@echo make up         : Start Docker services in background
	@echo make down       : Stop Docker services
	@echo make logs       : Follow live Docker container logs
	@echo make clean      : Clean temporary cache directories

install:
	uv sync --all-groups

sync:
	uv sync

test:
	uv run pytest

test-cov:
	uv run pytest --cov=src --cov-report=term-missing

lint:
	uv run ruff check .
	uv run mypy src

format:
	uv run ruff format .
	uv run ruff check --fix .

check: lint test

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	uv run python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.ruff_cache')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.mypy_cache')]"

push:	
	git push -u origin main

shell:
	uv run python src/city_pulse/sql_shell.py

mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src
	uv run pytest

dbt-core:
	uv run --group dbt dbt docs generate --project-dir dbt --profiles-dir dbt
	uv run --group dbt dbt docs serve --project-dir dbt --profiles-dir dbt --port 8081

dbt-run:
	uv run --group dbt dbt run --project-dir dbt --profiles-dir dbt

dbt-test:
	uv run --group dbt dbt test --project-dir dbt --profiles-dir dbt

dashboard:
	uv run streamlit run src/city_pulse/dashboard/app.py
