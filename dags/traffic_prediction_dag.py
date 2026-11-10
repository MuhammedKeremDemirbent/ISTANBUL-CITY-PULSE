from datetime import datetime, timedelta
from pathlib import Path

# Airflow ortamında değilken yerel importların çökmemesi için try/except
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


def task_fetch_live_weather(**context: object) -> str:
    """Open-Meteo'dan anlık hava durumu ve 7 günlük tahmini çeker."""
    from city_pulse.domain import ISTANBUL_CENTER
    from city_pulse.ingest.openmeteo_client import OpenMeteoClient
    from city_pulse.pipelines.weather_live import run_live
    from city_pulse.storage import FileStore

    base_dir = Path(__file__).resolve().parents[1]
    store = FileStore(base_dir / "data")
    client = OpenMeteoClient()

    result = run_live(ISTANBUL_CENTER, source=client, store=store)
    return f"Weather fetched: {result.forecast.height} forecast hours."


def task_predict_live_traffic(**context: object) -> str:
    """Hava tahmini ve takvim verilerini kullanarak gelecek 168 saatin trafiğini tahmin eder."""
    from city_pulse.pipelines.predict_live import run_predict_live

    predictions = run_predict_live()
    return f"Predictions generated successfully: {predictions.height} hours."


if AIRFLOW_AVAILABLE:
    default_args = {
        "owner": "city_pulse",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=1),
    }

    with DAG(
        dag_id="istanbul_traffic_live_prediction",
        default_args=default_args,
        description="İstanbul Canlı Hava & Trafik Tahmin Hattı (168 Saat)",
        schedule="45 10 * * *",
        start_date=datetime(2025, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=["city_pulse", "ml", "traffic", "inference"],
    ) as dag:
        fetch_weather = PythonOperator(
            task_id="fetch_live_weather",
            python_callable=task_fetch_live_weather,
        )

        predict_traffic = PythonOperator(
            task_id="predict_live_traffic",
            python_callable=task_predict_live_traffic,
        )

        fetch_weather >> predict_traffic

