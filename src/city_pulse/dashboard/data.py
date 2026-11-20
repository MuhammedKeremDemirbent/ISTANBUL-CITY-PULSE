from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import streamlit as st

DAY_NAMES: Final[dict[int, str]] = {
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
    7: "Sunday",
}

DEFAULT_PREDICTIONS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[3] / "data" / "gold" / "predictions"
)


@dataclass(frozen=True, slots=True)
class DashboardKPIs:
    """Summary indicators for dashboard metric cards."""

    max_traffic_val: float
    max_traffic_time: datetime
    avg_traffic_val: float
    rain_hours: int
    total_rain_mm: float
    critical_hours_count: int


@st.cache_data(ttl=300)
def load_latest_forecast(
    pred_dir: Path = DEFAULT_PREDICTIONS_DIR,
) -> tuple[pl.DataFrame, str]:
    """Load latest forecast Parquet file from disk."""
    files = list(pred_dir.glob("live_traffic_forecast_*.parquet"))
    if not files:
        raise FileNotFoundError(f"Forecast file not found in {pred_dir}")

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    df = pl.read_parquet(latest_file)
    return df, latest_file.name


def filter_by_day(df: pl.DataFrame, selected_day_name: str) -> pl.DataFrame:
    """Filter DataFrame by selected day name."""
    if selected_day_name == "All Week":
        return df

    day_num = next(k for k, v in DAY_NAMES.items() if v == selected_day_name)
    return df.filter(pl.col("day_of_week") == day_num)


def compute_kpis(df: pl.DataFrame, threshold: float) -> DashboardKPIs:
    """Compute KPI metrics for selected data."""
    max_row = df.sort("predicted_traffic_index_av", descending=True).head(1)
    raw_val = max_row.item(0, "predicted_traffic_index_av")
    max_traffic_val = float(raw_val) if isinstance(raw_val, (int, float)) else 0.0

    raw_time = max_row.item(0, "timestamp")
    max_traffic_time = raw_time if isinstance(raw_time, datetime) else datetime.now()

    avg_val = df["predicted_traffic_index_av"].mean()
    avg_traffic = float(avg_val) if isinstance(avg_val, (int, float)) else 0.0

    rain_sum = df["precipitation"].sum()
    total_rain = float(rain_sum) if isinstance(rain_sum, (int, float)) else 0.0

    return DashboardKPIs(
        max_traffic_val=max_traffic_val,
        max_traffic_time=max_traffic_time,
        avg_traffic_val=avg_traffic,
        rain_hours=df.filter(pl.col("is_raining") == 1).height,
        total_rain_mm=total_rain,
        critical_hours_count=df.filter(
            pl.col("predicted_traffic_index_av") >= threshold
        ).height,
    )


def compute_heatmap_matrix(df: pl.DataFrame) -> pd.DataFrame:
    """Compute day-and-hour intensity matrix."""
    pivot_df = (
        df.with_columns(
            pl.col("day_of_week")
            .replace_strict(DAY_NAMES, default="Unknown")
            .alias("day_name")
        )
        .group_by(["day_name", "hour"])
        .agg(pl.col("predicted_traffic_index_av").mean().alias("traffic"))
        .to_pandas()
    )

    day_order = list(DAY_NAMES.values())
    return (
        pivot_df.pivot(index="day_name", columns="hour", values="traffic")
        .reindex(day_order)
        .fillna(0)
    )


def compute_weather_impact(df: pl.DataFrame) -> pd.DataFrame:
    """Compare average traffic during rainy vs dry hours."""
    return (
        df.group_by("is_raining")
        .agg(
            pl.col("predicted_traffic_index_av").mean().alias("avg_traffic"),
            pl.col("predicted_traffic_index_av").count().alias("hours_count"),
        )
        .with_columns(
            pl.when(pl.col("is_raining") == 1)
            .then(pl.lit("Rainy Weather"))
            .otherwise(pl.lit("Clear/Dry Weather"))
            .alias("condition")
        )
        .to_pandas()
    )
