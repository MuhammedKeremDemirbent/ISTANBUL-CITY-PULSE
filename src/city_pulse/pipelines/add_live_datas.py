import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from city_pulse.pipelines.weather_live import LiveSource

log = logging.getLogger(__name__)

GOLD_COLUMNS_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "timestamp": pl.Datetime("us", None),
    "hour": pl.Int32,
    "day_of_week": pl.Int32,
    "month": pl.Int32,
    "is_weekend": pl.Int32,
    "is_holiday": pl.Int32,
    "is_school_term": pl.Int32,
    "is_school_morning_rush": pl.Int32,
    "is_school_afternoon_rush": pl.Int32,
    "traffic_index": pl.Int64,
    "traffic_index_an": pl.Int64,
    "traffic_index_av": pl.Int64,
    "temperature_2m": pl.Float64,
    "precipitation": pl.Float64,
    "wind_speed_10m": pl.Float64,
    "relative_humidity_2m": pl.Float64,
}

REQUIRED_WEATHER_COLS = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "relative_humidity_2m",
]

REQUIRED_TRAFFIC_COLS = [
    "traffic_index",
    "traffic_index_an",
    "traffic_index_av",
]


def _align_timestamp(series: pl.Series) -> pl.Expr:
    """Round timestamp to top of the hour and strip timezone info."""
    return pl.col("timestamp").dt.replace_time_zone(None)


def build_live_gold_rows(
    traffic_df: pl.DataFrame,
    weather_df: pl.DataFrame,
    *,
    holidays_df: pl.DataFrame | None = None,
    school_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Combine live IBB traffic and weather data with calendar features into Gold schema.

    Data Quality Rule (Hourly Tolerance):
    Only hours where both traffic and weather observations exist are included in Gold.
    Missing weather values are never filled with artificial zeros.
    """
    if traffic_df.height == 0 or weather_df.height == 0:
        return pl.DataFrame(schema=GOLD_COLUMNS_SCHEMA)

    # 1. Clean traffic data and drop nulls
    traffic_clean = (
        traffic_df.with_columns(
            _align_timestamp(traffic_df["timestamp"]).alias("timestamp")
        )
        .select(
            "timestamp",
            pl.col("traffic_index").round().cast(pl.Int64),
            pl.col("traffic_index_an").round().cast(pl.Int64),
            pl.col("traffic_index_av").round().cast(pl.Int64),
        )
        .drop_nulls(subset=["timestamp", *REQUIRED_TRAFFIC_COLS])
        .unique(subset=["timestamp"], keep="last")
    )

    # 2. Clean weather data and drop nulls
    weather_clean = (
        weather_df.with_columns(
            _align_timestamp(weather_df["timestamp"]).alias("timestamp")
        )
        .select(
            "timestamp",
            pl.col("temperature_2m").cast(pl.Float64),
            pl.col("precipitation").cast(pl.Float64),
            pl.col("wind_speed_10m").cast(pl.Float64),
            pl.col("relative_humidity_2m").cast(pl.Float64),
        )
        .drop_nulls(subset=["timestamp", *REQUIRED_WEATHER_COLS])
        .unique(subset=["timestamp"], keep="last")
    )

    # 3. Inner join on timestamp (Hourly tolerance)
    merged = traffic_clean.join(weather_clean, on="timestamp", how="inner")
    if merged.height == 0:
        return pl.DataFrame(schema=GOLD_COLUMNS_SCHEMA)

    # 4. Generate calendar features
    merged = merged.with_columns(
        pl.col("timestamp").dt.hour().cast(pl.Int32).alias("hour"),
        pl.col("timestamp").dt.weekday().cast(pl.Int32).alias("day_of_week"),
        pl.col("timestamp").dt.month().cast(pl.Int32).alias("month"),
        pl.when(pl.col("timestamp").dt.weekday().is_in([6, 7]))
        .then(1)
        .otherwise(0)
        .cast(pl.Int32)
        .alias("is_weekend"),
    )

    # 5. Match holidays
    if holidays_df is not None and holidays_df.height > 0:
        holiday_dates = holidays_df.select(pl.col("date").cast(pl.Date))[
            "date"
        ].to_list()
        merged = merged.with_columns(
            pl.col("timestamp")
            .dt.date()
            .is_in(holiday_dates)
            .cast(pl.Int32)
            .alias("is_holiday")
        )
    else:
        merged = merged.with_columns(pl.lit(0).cast(pl.Int32).alias("is_holiday"))

    # 6. Match school calendar
    school_cols = [
        "is_school_term",
        "is_school_morning_rush",
        "is_school_afternoon_rush",
    ]
    if school_df is not None and school_df.height > 0:
        school_clean = school_df.with_columns(
            _align_timestamp(school_df["timestamp"]).alias("timestamp")
        ).select("timestamp", *school_cols)
        merged = merged.join(school_clean, on="timestamp", how="left").with_columns(
            [pl.col(c).fill_null(0).cast(pl.Int32) for c in school_cols]
        )
    else:
        merged = merged.with_columns(
            [pl.lit(0).cast(pl.Int32).alias(c) for c in school_cols]
        )

    # 7. Final schema ordering
    return merged.sort("timestamp").select(list(GOLD_COLUMNS_SCHEMA.keys()))


def add_live_datas_to_gold(
    *,
    live_traffic_dir: Path,
    live_weather_dir: Path,
    gold_file: Path,
    holidays_path: Path | None = None,
    school_path: Path | None = None,
    archive_dir: Path | None = None,
    weather_source: "LiveSource | None" = None,
) -> int:
    """Atomically append accumulated live IBB traffic and weather data to Gold table.

    Self-Healing:
    If live weather forecast is missing or does not cover traffic dates,
    the pipeline fetches today's forecast from Open-Meteo. If unavailable, Gold is untouched.
    """
    if not live_traffic_dir.exists():
        log.info("Live traffic directory not found: %s", live_traffic_dir)
        return 0

    traffic_files = sorted(live_traffic_dir.glob("traffic_*.parquet"))
    if not traffic_files:
        log.info("No new live traffic files found to process.")
        return 0

    log.info("Found %d live traffic files, reading...", len(traffic_files))
    traffic_df = pl.concat([pl.read_parquet(f) for f in traffic_files])
    if traffic_df.height == 0:
        log.info("No rows found in live traffic files.")
        return 0

    traffic_timestamps = traffic_df.with_columns(
        _align_timestamp(traffic_df["timestamp"]).alias("aligned_ts")
    )["aligned_ts"]
    traffic_dates = set(traffic_timestamps.dt.date().to_list())

    weather_files = sorted(
        list(live_weather_dir.glob("forecast_*.parquet"))
        or list(live_weather_dir.glob("current_*.parquet"))
    )

    def _weather_covers_traffic(w_files: list[Path]) -> bool:
        if not w_files:
            return False
        latest_file = w_files[-1]
        try:
            w_df = pl.read_parquet(latest_file)
            if "timestamp" not in w_df.columns:
                return False
            w_dates = set(
                w_df.with_columns(
                    _align_timestamp(w_df["timestamp"]).alias("aligned_ts")
                )["aligned_ts"]
                .dt.date()
                .to_list()
            )
            return bool(traffic_dates.intersection(w_dates))
        except (OSError, ValueError, pl.exceptions.PolarsError) as err:
            log.warning("Error checking weather file (%s): %s", latest_file.name, err)
            return False

    if not _weather_covers_traffic(weather_files):
        log.info(
            "No matching weather forecast found for traffic dates. "
            "Triggering self-healing fetch from Open-Meteo..."
        )
        try:
            from city_pulse.domain import ISTANBUL_CENTER
            from city_pulse.ingest.openmeteo_client import OpenMeteoClient
            from city_pulse.pipelines.weather_live import run_live
            from city_pulse.storage import FileStore

            src = weather_source if weather_source is not None else OpenMeteoClient()
            data_dir = live_weather_dir.parents[2]
            store = FileStore(data_dir)
            run_live(ISTANBUL_CENTER, source=src, store=store, forecast_days=1)

            weather_files = sorted(
                list(live_weather_dir.glob("forecast_*.parquet"))
                or list(live_weather_dir.glob("current_*.parquet"))
            )
        except (OSError, ValueError, pl.exceptions.PolarsError) as err:
            log.error("Failed to automatically fetch weather forecast: %s", err)

    if not weather_files:
        log.warning(
            "Weather data unavailable. Aborting update to keep Gold dataset clean."
        )
        return 0

    latest_weather_file = weather_files[-1]
    log.info("Using weather file: %s", latest_weather_file.name)
    weather_df = pl.read_parquet(latest_weather_file)

    holidays_df = (
        pl.read_parquet(holidays_path)
        if holidays_path and holidays_path.exists()
        else None
    )
    school_df = (
        pl.read_parquet(school_path) if school_path and school_path.exists() else None
    )

    new_gold_rows = build_live_gold_rows(
        traffic_df,
        weather_df,
        holidays_df=holidays_df,
        school_df=school_df,
    )

    if new_gold_rows.height == 0:
        log.warning(
            "No matching complete hours found between traffic and weather. "
            "Gold table preserved; no rows added and files not archived."
        )
        return 0

    if not gold_file.exists():
        raise FileNotFoundError(f"Main Gold file not found: {gold_file}")

    gold_df = pl.read_parquet(gold_file)
    initial_count = gold_df.height

    combined = (
        pl.concat([gold_df, new_gold_rows])
        .sort("timestamp")
        .unique(subset=["timestamp"], keep="last")
    )

    # Atomic write
    tmp_path = gold_file.with_suffix(".tmp.parquet")
    combined.write_parquet(tmp_path)
    tmp_path.replace(gold_file)

    added_count = combined.height - initial_count
    log.info(
        "Gold table updated: %s (+%d new rows, total %d rows)",
        gold_file.name,
        added_count,
        combined.height,
    )

    # Only archive traffic files after Gold is safely updated
    if archive_dir:
        archive_dir.mkdir(parents=True, exist_ok=True)
        for tf in traffic_files:
            tf.replace(archive_dir / tf.name)
        log.info(
            "Archived %d live traffic files: %s",
            len(traffic_files),
            archive_dir,
        )

    return added_count


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append live data into main Gold table"
    )
    parser.add_argument(
        "--traffic-dir", type=Path, default=Path("data/silver/traffic_live")
    )
    parser.add_argument(
        "--weather-dir",
        type=Path,
        default=Path("data/silver/weather_live/istanbul_center"),
    )
    parser.add_argument(
        "--gold-file",
        type=Path,
        default=Path("data/gold/city_pulse_hourly_2023-01-01_2025-12-31.parquet"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    add_live_datas_to_gold(
        live_traffic_dir=args.traffic_dir,
        live_weather_dir=args.weather_dir,
        gold_file=args.gold_file,
        holidays_path=root / "data/silver/holidays/turkey_holidays_2023_2025.parquet",
        school_path=root / "data/silver/school/school_calendar_2023_2025.parquet",
        archive_dir=root / "data/silver/traffic_live_archived",
    )


if __name__ == "__main__":
    main()
