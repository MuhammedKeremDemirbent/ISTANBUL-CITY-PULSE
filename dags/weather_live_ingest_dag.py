from datetime import datetime, timedelta
from pathlib import Path

try:
    from airflow import DAG
    from airflow.datasets import Dataset
    from airflow.operators.python import PythonOperator

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


def task_fetch_live_weather(**context: object) -> str:
    """Fetch daily weather forecast from Open-Meteo and store in Silver.

    Runs every night at 00:00 Istanbul time (21:00 UTC).
    Prepares weather data for downstream prediction and gold_update DAGs.
    """
    from city_pulse.domain import ISTANBUL_CENTER
    from city_pulse.ingest.openmeteo_client import OpenMeteoClient
    from city_pulse.pipelines.weather_live import run_live
    from city_pulse.storage import FileStore

    base_dir = Path(__file__).resolve().parents[1]
    store = FileStore(base_dir / "data")
    client = OpenMeteoClient()

    result = run_live(ISTANBUL_CENTER, source=client, store=store, forecast_days=1)
    return f"Weather data saved: {result.forecast.height} hours forecast."


def task_verify_weather_silver(**context: object) -> str:
    """Validate integrity of the newly written weather Silver Parquet file."""
    import polars as pl

    base_dir = Path(__file__).resolve().parents[1]
    weather_dir = base_dir / "data" / "silver" / "weather_live" / "istanbul_center"
    files = sorted(weather_dir.glob("forecast_*.parquet"))
    if not files:
        return "No weather forecast files found to verify."
    latest = files[-1]
    df = pl.read_parquet(latest)
    if (
        "timestamp" not in df.columns
        or "temperature_2m" not in df.columns
        or df.height == 0
    ):
        raise ValueError(f"Invalid weather schema or empty file: {latest.name}")
    return f"Verified: {latest.name} ({df.height} hours forecast ready)"


if AIRFLOW_AVAILABLE:
    default_args = {
        "owner": "city_pulse",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 3,
        "retry_delay": timedelta(minutes=10),
    }

    with DAG(
        dag_id="istanbul_weather_live_ingest",
        default_args=default_args,
        description=(
            "Fetches daily weather forecast from Open-Meteo and stores to Silver "
            "every night at 00:00 Istanbul time (21:00 UTC)."
        ),
        schedule="0 21 * * *",  # 00:00 Istanbul = 21:00 UTC
        start_date=datetime(2025, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=["city_pulse", "weather", "ingestion", "live"],
    ) as dag:
        fetch_weather_task = PythonOperator(
            task_id="fetch_live_weather",
            python_callable=task_fetch_live_weather,
        )

        verify_weather_task = PythonOperator(
            task_id="verify_silver_weather",
            python_callable=task_verify_weather_silver,
            outlets=[Dataset("city_pulse://silver/weather_live")],
        )

        fetch_weather_task >> verify_weather_task
