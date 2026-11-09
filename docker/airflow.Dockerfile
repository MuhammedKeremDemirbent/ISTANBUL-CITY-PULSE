# Apache Airflow 2.10.4 Python 3.12 tabanlı özel imaj
FROM apache/airflow:2.10.4-python3.12

# LightGBM paralel işlem motoru için Linux OpenMP kütüphanesi
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# City Pulse canlı tahmin ve veri boru hattı bağımlılıkları
# NumPy 1.26.4 ile C-ABI ikili uyumluluğu garanti altına alınır (numpy.dtype size mismatch engellenir)
RUN pip install --no-cache-dir \
    "numpy>=1.26.4,<2.0.0" \
    "pandas>=2.2.0" \
    "polars>=1.20.0" \
    "pyarrow>=18.0.0" \
    "lightgbm>=4.5.0" \
    "scikit-learn>=1.5.0" \
    "holidays>=0.60" \
    "requests>=2.32.0" \
    "pydantic>=2.10.0" \
    "pydantic-settings>=2.7.0"
