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


def task_validate_silver_inputs(**context: object) -> str:
    """Validate presence of today's Silver traffic and weather files."""
    base_dir = Path(__file__).resolve().parents[1]
    traffic_dir = base_dir / "data" / "silver" / "traffic_live"
    weather_dir = base_dir / "data" / "silver" / "weather_live" / "istanbul_center"

    traffic_count = (
        len(list(traffic_dir.glob("traffic_*.parquet"))) if traffic_dir.exists() else 0
    )
    weather_count = (
        len(list(weather_dir.glob("forecast_*.parquet"))) if weather_dir.exists() else 0
    )

    if traffic_count == 0:
        return "No live traffic files to process, waiting."

    return (
        f"Validation: {traffic_count} traffic files present. "
        f"{weather_count} weather forecasts present (self-healing will run if missing)."
    )


def task_add_live_datas_to_gold(**context: object) -> str:
    """Add accumulated daily IBB traffic and weather data into the Gold table.

    Runs every night at 00:05 Istanbul time (21:05 UTC).
    Validates weather data; only complete hours are added to Gold.
    """
    from city_pulse.pipelines.add_live_datas import add_live_datas_to_gold

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"

    added = add_live_datas_to_gold(
        live_traffic_dir=data_dir / "silver" / "traffic_live",
        live_weather_dir=data_dir / "silver" / "weather_live" / "istanbul_center",
        gold_file=data_dir / "gold" / "city_pulse_hourly_2023-01-01_2025-12-31.parquet",
        holidays_path=data_dir
        / "silver"
        / "holidays"
        / "turkey_holidays_2023_2025.parquet",
        school_path=data_dir
        / "silver"
        / "school"
        / "school_calendar_2023_2025.parquet",
        archive_dir=data_dir / "silver" / "traffic_live_archived",
    )

    if added == 0:
        return (
            "Gold table not updated: conditions not met, "
            "missing weather data, or no new clean hours to process."
        )

    return f"Gold table updated: +{added} new rows added."


if AIRFLOW_AVAILABLE:
    default_args = {
        "owner": "city_pulse",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    }

    with DAG(
        dag_id="istanbul_traffic_gold_update",
        default_args=default_args,
        description=(
            "Adds daily 24-hour IBB traffic and weather data to Gold table "
            "and archives Silver files every night at 00:05 Istanbul time (21:05 UTC)."
        ),
        schedule="5 21 * * *",  # 00:05 Istanbul = 21:05 UTC
        start_date=datetime(2025, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=["city_pulse", "traffic", "gold", "pipeline"],
    ) as dag:
        validate_inputs = PythonOperator(
            task_id="validate_silver_inputs",
            python_callable=task_validate_silver_inputs,
            inlets=[
                Dataset("city_pulse://silver/traffic_live"),
                Dataset("city_pulse://silver/weather_live"),
            ],
        )

        update_gold = PythonOperator(
            task_id="merge_and_update_gold",
            python_callable=task_add_live_datas_to_gold,
            outlets=[Dataset("city_pulse://gold/city_pulse_hourly")],
        )

        validate_inputs >> update_gold
