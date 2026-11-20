from datetime import datetime, timedelta
from pathlib import Path

try:
    from airflow import DAG
    from airflow.datasets import Dataset
    from airflow.operators.python import PythonOperator

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


def task_fetch_live_traffic(**context: object) -> str:
    """Fetch IBB TKM live traffic index; store as bronze and silver."""
    from city_pulse.ingest.ibb_client import IbbTrafficClient
    from city_pulse.pipelines.traffic_live import run_traffic_live
    from city_pulse.storage import FileStore

    base_dir = Path(__file__).resolve().parents[1]
    store = FileStore(base_dir / "data")
    client = IbbTrafficClient()

    result = run_traffic_live(source=client, store=store)
    df = result.frame
    ti = df["traffic_index"][0]
    ti_av = df["traffic_index_av"][0]
    ti_an = df["traffic_index_an"][0]

    return (
        f"IBB live traffic successfully saved! "
        f"Overall: %{ti:.0f}, Europe: %{ti_av:.0f}, Anatolia: %{ti_an:.0f}"
    )


def task_verify_traffic_silver(**context: object) -> str:
    """Validate schema and data of the newly written Silver Parquet file."""
    import polars as pl

    base_dir = Path(__file__).resolve().parents[1]
    live_dir = base_dir / "data" / "silver" / "traffic_live"
    files = sorted(live_dir.glob("traffic_*.parquet"))
    if not files:
        return "No traffic files found to verify."
    latest = files[-1]
    df = pl.read_parquet(latest)
    if "timestamp" not in df.columns or "traffic_index" not in df.columns:
        raise ValueError(f"Invalid schema: {latest.name}")
    return (
        f"Verified: {latest.name} (Rows: {df.height}, Ti: {df['traffic_index'][0]:.0f})"
    )


if AIRFLOW_AVAILABLE:
    default_args = {
        "owner": "city_pulse",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 3,
        "retry_delay": timedelta(minutes=2),
    }

    with DAG(
        dag_id="istanbul_traffic_live_ingest",
        default_args=default_args,
        description="Hourly IBB TKM live traffic index collection pipeline (Bronze JSON & Silver Parquet)",
        schedule="0 * * * *",  # Every hour at minute 0
        start_date=datetime(2025, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=["city_pulse", "traffic", "ingestion", "live", "ibb"],
    ) as dag:
        fetch_live_traffic_task = PythonOperator(
            task_id="fetch_live_traffic",
            python_callable=task_fetch_live_traffic,
        )

        verify_traffic_task = PythonOperator(
            task_id="verify_silver_traffic",
            python_callable=task_verify_traffic_silver,
            outlets=[Dataset("city_pulse://silver/traffic_live")],
        )

        fetch_live_traffic_task >> verify_traffic_task
