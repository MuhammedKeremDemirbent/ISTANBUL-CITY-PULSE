# Istanbul City Pulse

Istanbul City Pulse, İstanbul Büyükşehir Belediyesi (İBB) Açık Veri Portalı ve şehir API'lerinden dinamik kentsel verileri toplayan, depolayan, analitik dönüşümlerden geçiren ve makine öğrenimi modelleriyle şehir dinamiklerini tahmin eden uçtan uca bir Veri Mühendisliği ve MLOps platformudur.

---

## Mimari ve Teknoloji Yığını

* **Paket ve Ortam Yönetimi:** uv, Python 3.12+
* **Veri Toplama (Ingestion):** Requests, Pydantic v2, Pydantic-Settings
* **Veri İşleme ve Analitik Depolama:** Polars, DuckDB, Apache Arrow, PostgreSQL (SQLAlchemy 2.0, Psycopg3, Alembic)
* **Veri Dönüşümü (ELT):** dbt-core, dbt-postgres
* **İş Akışı Orkestrasyonu:** Apache Airflow
* **Makine Öğrenimi ve MLOps:** Scikit-Learn, LightGBM, MLflow
* **Veri Kalitesi ve Test:** Great Expectations, Pytest, Pytest-Cov
* **Kod Kalitesi:** Ruff (linter & formatter), Mypy (statik tip denetimi)
* **Konteynerizasyon:** Docker, Docker Compose

---

## Dizin Yapısı

```text
Istanbul-City-Pulse/
├── src/istanbul_city_pulse/   # Çekirdek Python kaynak kodları ve veri akışları
├── dags/                      # Apache Airflow iş akışı (DAG) tanımları
├── dbt/                       # Veri ambarı dönüşüm (transform) modelleri
├── db/migrations/             # Veritabanı şema versiyonlama dosyaları (Alembic)
├── docker/                    # Airflow ve uygulama servis Dockerfile dosyaları
├── data/                      # Yerel ham ve işlenmiş veri dosyaları (Git dışı)
├── sql/                       # Tablo tanımları ve SQL analiz sorguları
├── tests/                     # Birim ve entegrasyon testleri
├── docker-compose.yml         # Konteyner orkestrasyon dosyası
├── pyproject.toml             # Proje bağımlılıkları ve araç ayarları
└── uv.lock                    # Deterministik bağımlılık kilit dosyası
```

---

## Hızlı Başlangıç

### 1. Gereksinimler
* Python 3.12+
* uv (Astral)
* Docker ve Docker Compose

### 2. Ortam Kurulumu
Depoyu klonladıktan sonra tüm bağımlılık gruplarıyla birlikte sanal ortamı kurun:

```bash
uv sync --all-groups
```

Ortam değişkenlerini tanımlayın:

```bash
cp .env-example .env
```

### 3. Test ve Kod Denetimi
Testleri ve kod kalite kontrollerini çalıştırmak için:

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

