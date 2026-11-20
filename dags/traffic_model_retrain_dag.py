from datetime import UTC, datetime, timedelta
from pathlib import Path

try:
    from airflow import DAG
    from airflow.datasets import Dataset
    from airflow.operators.empty import EmptyOperator
    from airflow.operators.python import BranchPythonOperator, PythonOperator

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


def task_check_data_completeness(**context: object) -> str:
    """Check whether data is complete from 2023-01-01 to the 1st of current month UTC 00:00.

    Branches directly to training if complete, or to backfill task if missing.
    """
    import polars as pl

    now = datetime.now(UTC)
    first_of_this_month = datetime(now.year, now.month, 1, 0, 0, tzinfo=UTC)
    start_date = datetime(2023, 1, 1, 0, 0, tzinfo=UTC)

    # Expected total hours
    expected_hours = int((first_of_this_month - start_date).total_seconds() // 3600)

    # Check training table in Gold layer
    root_dir = Path(__file__).resolve().parents[1]
    gold_dir = root_dir / "data" / "gold"
    gold_files = list(gold_dir.glob("*.parquet"))

    is_complete = False
    if gold_files:
        latest_gold = max(gold_files)
        df = pl.read_parquet(latest_gold)
        # Check whether row count matches expected hours
        if df.height >= expected_hours:
            is_complete = True

    if is_complete:
        return "proceed_to_retrain"
    return "backfill_missing_data"


def task_backfill_missing_data(**context: object) -> str:
    """Fetch missing weather and calendar data to complete the pipeline."""
    from city_pulse.domain import ISTANBUL_CENTER
    from city_pulse.ingest.openmeteo_client import OpenMeteoClient
    from city_pulse.pipelines.weather_history import run_history
    from city_pulse.storage import FileStore

    root_dir = Path(__file__).resolve().parents[1]
    store = FileStore(root_dir / "data")
    client = OpenMeteoClient()

    now = datetime.now(UTC)
    first_of_this_month = datetime(now.year, now.month, 1, 0, 0, tzinfo=UTC)

    # Sync historical ERA5 weather for missing range
    run_history(
        ISTANBUL_CENTER,
        start="2023-01-01",
        end=first_of_this_month.strftime("%Y-%m-%d"),
        source=client,
        store=store,
    )
    return "Missing data backfill completed successfully."


def task_retrain_traffic_model(**context: object) -> str:
    """Train model on completed dataset and log to MLflow."""
    from city_pulse.pipelines.train_model import run_training_pipeline

    result = run_training_pipeline()
    return (
        f"Model successfully trained and logged to MLflow! "
        f"Run ID: {result.run_id}, Test MAE: {result.model_metrics.mae:.4f}, "
        f"Champion Promoted: {result.promoted}"
    )


if AIRFLOW_AVAILABLE:
    default_args = {
        "owner": "city_pulse",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    with DAG(
        dag_id="istanbul_traffic_model_retraining",
        default_args=default_args,
        description="Monthly model retraining and MLflow tracking pipeline",
        schedule="45 20 2 * *",  # 2nd day of each month at 23:45 Istanbul (20:45 UTC)
        start_date=datetime(2025, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=["city_pulse", "mlops", "retraining", "mlflow"],
    ) as dag:
        check_completeness = BranchPythonOperator(
            task_id="check_data_completeness",
            python_callable=task_check_data_completeness,
        )

        proceed_to_retrain = EmptyOperator(
            task_id="proceed_to_retrain",
        )

        backfill_data = PythonOperator(
            task_id="backfill_missing_data",
            python_callable=task_backfill_missing_data,
        )

        retrain_model = PythonOperator(
            task_id="retrain_and_log_model",
            python_callable=task_retrain_traffic_model,
            trigger_rule="none_failed_min_one_success",
            inlets=[Dataset("city_pulse://gold/city_pulse_hourly")],
            outlets=[Dataset("city_pulse://models/champion_model")],
        )

        # Decision flow (diamond branching pattern):
        # 1. If data complete -> proceed_to_retrain -> retrain_and_log_model
        # 2. If data missing  -> backfill_missing_data -> retrain_and_log_model
        check_completeness >> [proceed_to_retrain, backfill_data]
        proceed_to_retrain >> retrain_model
        backfill_data >> retrain_model
