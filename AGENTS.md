# AGENTS.md — Istanbul City Pulse

Bu dosya, projede çalışan tüm **Yapay Zeka Ajanları ve Kod Asistanları** için bağlamı, mimari prensipleri ve uyulması gereken zorunlu kuralları tanımlar.

---

## 🏙️ 1. Proje Özeti ve Amacı
* **Proje Adı:** Istanbul City Pulse
* **Tanım:** İstanbul Büyükşehir Belediyesi (İBB) Açık Veri Portalı ve şehir API'lerinden dinamik veri toplayan, depolayan, dbt ile dönüştüren ve makine öğrenimi modelleriyle (LightGBM/MLflow) şehir dinamiklerini analiz ve tahmin eden uçtan uca modern bir Veri Mühendisliği ve MLOps platformudur.
* **Teknoloji Yığını:** Python 3.12+, `uv`, Polars, DuckDB, PostgreSQL (SQLAlchemy + Alembic), dbt, Apache Airflow, Docker Compose, LightGBM, MLflow, Ruff, Pytest.

---

## ⚙️ 2. Paket Yönetimi ve Çalıştırma Kuralları
1. **Asla `pip` Kullanma:** Paket yönetimi için sadece `uv` kullanılmalıdır.
   * Paket ekleme: `uv add <paket>`
   * Gruba paket ekleme: `uv add --group <dev|ml|dbt|quality> <paket>`
   * Ortam eşitleme: `uv sync` veya `uv sync --all-groups`
2. **Komutları `uv run` ile Koştur:**
   * Testler: `uv run pytest`
   * Linter & Format kontrolü: `uv run ruff check .` / `uv run ruff format .`
   * Tip denetimi: `uv run mypy src`
   * Python scriptleri: `uv run python <dosya_yolu>`
3. **Kilit Dosyası Bütünlüğü:** `uv.lock` dosyasını asla elle düzenleme; `pyproject.toml` üzerinden `uv` aracılığıyla güncelle.

---

## 📂 3. Dizin Yapısı ve Sorumluluklar
* **`src/istanbul_city_pulse/`:** Tüm çekirdek Python kaynak kodları buraya yazılır (`src` layout). Kök dizine doğrudan `.py` kodu bırakma.
* **`dags/`:** Sadece Apache Airflow DAG (iş akışı) dosyaları barındırılır.
* **`dbt/`:** Veri ambarı dönüşüm (ELT) SQL modelleri ve `schema.yml` dosyaları burada yer alır.
* **`docker/`:** Servislere ait özel `Dockerfile`'lar tutulur.
* **`data/`:** Yerel ham ve işlenmiş veri dosyaları (CSV, Parquet) tutulur. Asla Git'e gönderilmez.
* **`tests/`:** Pytest birim ve entegrasyon testleri buraya yazılır.
* **`ZZ_NOTLAR/`:** Geliştiricinin kişisel notlarıdır. Kod bağımlılığı içermez, otomatik olarak değiştirilmemelidir.

---

## 💻 4. Kodlama ve Mimari Standartları
* **Veri İşleme Önceliği:** Büyük veri manipülasyonu ve ETL işlemlerinde `pandas` yerine **`polars`** veya **`duckdb`** tercih edilmelidir. Pandas sadece ML modellerinin doğrudan girdi beklediği noktalarda kullanılmalıdır.
* **Tip Güvenliği (Type Hints):** Tüm fonksiyon, metod ve sınıf değişkenlerinde Python tip belirteçleri (`typing` / built-in types) eksiksiz yazılmalıdır.
* **Veri Doğrulama:** Dış kaynaklardan gelen tüm JSON ve API yanıtları **Pydantic (v2)** modelleri ile şemaya bağlanmalı ve doğrulanmalıdır.
* **Konfigürasyon Yönetimi:** Gizli anahtarlar, veritabanı bağlantı URI'ları ve ayarlar **`pydantic-settings`** kullanılarak `.env` üzerinden okunmalıdır. Koda hardcode anahtar yazmak **kesinlikle yasaktır**.
* **Hata Yönetimi ve Loglama:** `print()` yerine standart Python `logging` veya yapılandırılmış logger kullanılmalıdır.

---

## 🛡️ 5. Güvenlik ve Git Kuralları
* `.env`, `*.duckdb`, `data/`, `.venv/` ve geçici önbellek dizinleri asla Git'e eklenmemelidir.
* Commit mesajları **Conventional Commits** standartlarına uygun olmalıdır (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`).

