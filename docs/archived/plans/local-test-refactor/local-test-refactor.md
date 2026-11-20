# Task Plan: Test Mimarisi ve Dizin Refactoring

## Plan / Requirements
- [x] `tests/conftest.py` oluşturmak (global fixture'lar, sahte zaman, mock API verileri, DataFrame üreticileri)
- [x] `tests/fixtures/` klasörünü açıp örnek İBB ve Open-Meteo JSON yanıtlarını eklemek
- [x] `tests/unit/` altındaki testleri modüllere göre ayırmak:
  - [x] `tests/unit/domain/` (`test_domain.py`)
  - [x] `tests/unit/features/` (`test_builder.py`)
  - [x] `tests/unit/ml/` (`test_split.py`, `test_baseline.py`, `test_train.py`, `test_evaluate.py`, `test_lightgbm.py`)
  - [x] `tests/unit/pipelines/` (`test_predict_live.py`, `test_train_model.py`, `test_add_live_datas.py`)
  - [x] `tests/unit/ingest/` (`test_traffic_live_ibb.py`, `test_openmeteo.py`)
- [x] `tests/unit/storage/` altına temel storage atomik yazma / FileStore testleri eklemek
- [x] Testlerin tamamını (`uv run pytest`) çalıştırarak doğrulamak

## Expected Verification
- `uv run pytest` tüm testleri toplamalı ve hatasız geçmeli. (37 testin 37'si geçti)
- `uv run ruff check tests/` ve `uv run ruff format --check tests/` kontrolleri temiz geçmeli.

## Progress
- [x] `pyproject.toml` pytest konfigürasyonu güncellendi ve doğrulandı.
- [x] Test dizin hiyerarşisi `src/city_pulse/` mimarisine uygun şekilde refactor edildi.
- [x] Tekrarlayan/kök dizinde başıboş kalan testler temizlendi.

## Blockers / Open Questions
- Yok.
