from datetime import datetime, timedelta
from pathlib import Path

# Try/except to prevent import errors during local type checking or testing without Airflow
try:
    from airflow import DAG
    from airflow.datasets import Dataset
    from airflow.operators.python import PythonOperator

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


def task_fetch_live_weather(**context: object) -> str:
    """Fetch current weather forecast from Open-Meteo at prediction time."""
    from city_pulse.domain import ISTANBUL_CENTER
    from city_pulse.ingest.openmeteo_client import OpenMeteoClient
    from city_pulse.pipelines.weather_live import run_live
    from city_pulse.storage import FileStore

    base_dir = Path(__file__).resolve().parents[1]
    store = FileStore(base_dir / "data")
    client = OpenMeteoClient()

    result = run_live(ISTANBUL_CENTER, source=client, store=store)
    return f"Weather forecast fetched: {result.forecast.height} hours of data."


def task_fetch_live_traffic(**context: object) -> str:
    """Fetch latest IBB TKM traffic state as reference lag for prediction."""
    from city_pulse.ingest.ibb_client import IbbTrafficClient
    from city_pulse.pipelines.traffic_live import run_traffic_live
    from city_pulse.storage import FileStore

    base_dir = Path(__file__).resolve().parents[1]
    store = FileStore(base_dir / "data")
    client = IbbTrafficClient()

    result = run_traffic_live(source=client, store=store)
    df = result.frame
    return (
        f"IBB live traffic saved: "
        f"Overall: %{df['traffic_index'][0]:.0f}, "
        f"Europe: %{df['traffic_index_av'][0]:.0f}, "
        f"Anatolia: %{df['traffic_index_an'][0]:.0f}"
    )


def task_predict_live_traffic(**context: object) -> str:
    """Predict next 168 hours of traffic using current weather and reference traffic."""
    from city_pulse.pipelines.predict_live import run_predict_live

    predictions = run_predict_live()
    return f"Forecast successfully generated: {predictions.height} hours recorded."


if AIRFLOW_AVAILABLE:
    default_args = {
        "owner": "city_pulse",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
    }

    with DAG(
        dag_id="istanbul_traffic_live_prediction",
        default_args=default_args,
        description="Istanbul live weather and IBB traffic prediction pipeline (168 hours)",
        schedule="15 21 * * *",  # 00:15 Istanbul = 21:15 UTC
        start_date=datetime(2025, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=["city_pulse", "ml", "traffic", "inference", "ibb"],
    ) as dag:
        fetch_weather = PythonOperator(
            task_id="fetch_live_weather",
            python_callable=task_fetch_live_weather,
        )

        fetch_traffic = PythonOperator(
            task_id="fetch_live_traffic",
            python_callable=task_fetch_live_traffic,
        )

        predict_traffic = PythonOperator(
            task_id="predict_live_traffic",
            python_callable=task_predict_live_traffic,
            inlets=[
                Dataset("city_pulse://gold/city_pulse_hourly"),
                Dataset("city_pulse://silver/weather_live"),
            ],
            outlets=[Dataset("city_pulse://gold/predictions")],
        )

        # Parallel data fetch -> Prediction inference
        [fetch_weather, fetch_traffic] >> predict_traffic
