.PHONY: help install sync test test-cov lint format check up down logs clean

help:
	@echo ========================================================
	@echo Istanbul City Pulse - Proje Komut Yoneticisi
	@echo ========================================================
	@echo make install    : Tum bagimlilik gruplarini senkronize eder
	@echo make test       : Pytest birim testlerini calistirir
	@echo make test-cov   : Testleri kapsam raporuyla calistirir
	@echo make lint       : Ruff ve Mypy ile kod kalitesini denetler
	@echo make format     : Kodu Ruff ile otomatik formatlar ve duzeltir
	@echo make check      : Hem lint denetimini hem testleri koşturur
	@echo make up         : Docker servislerini arka planda baslatir
	@echo make down       : Docker servislerini durdurur
	@echo make logs       : Docker konteyner loglarini canli takip eder
	@echo make clean      : Gecici onbellek ve cache dosyalarini temizler

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
	uv run python -m city_pulse.sql_shell
	