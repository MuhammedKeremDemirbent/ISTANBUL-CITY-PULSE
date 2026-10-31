#Client dosyasından alınan verileri tabloya çevirir. İnternetsiz çalışır.

from datetime import datetime

import polars as pl

from city_pulse.domain import ISTANBUL_TIMEZONE, VARIABLES

_TIME_FORMAT = "%Y-%m-%dT%H:%M"


def _to_istanbul_time(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .str.to_datetime(_TIME_FORMAT)
        .dt.replace_time_zone("UTC")               
        .dt.convert_time_zone(ISTANBUL_TIMEZONE)  
        .dt.replace_time_zone(None)               
    )


def _numeric_columns(available: dict) -> list[pl.Expr]:
    # API bazen tam sayı (82), bazen ondalık (82.5) döner; tipi Float64'e sabitle
    return [pl.col(v).cast(pl.Float64) for v in VARIABLES if v in available]


def _hourly_frame(payload: dict, point_id: str, source: str) -> pl.DataFrame:
    hourly = payload["hourly"]
    return (
        pl.DataFrame(hourly)
        .select(
            _to_istanbul_time("time").alias("timestamp"),
            *_numeric_columns(hourly),
        )
        .with_columns(
            pl.lit(point_id).alias("weather_point_id"),
            pl.lit(source).alias("source"),
        )
    )


def archive_to_frame(payload: dict, point_id: str) -> pl.DataFrame:
    return _hourly_frame(payload, point_id, source="era5")


def forecast_to_frame(
    payload: dict, point_id: str, fetched_at: datetime
) -> pl.DataFrame:
    return _hourly_frame(payload, point_id, source="forecast").with_columns(
        pl.lit(fetched_at).alias("fetched_at_utc"),
    )


def current_to_frame(
    payload: dict, point_id: str, fetched_at: datetime
) -> pl.DataFrame:
    current = payload["current"]
    row = {"observed_at_utc": current["time"], **{v: current[v] for v in VARIABLES}}
    return pl.DataFrame([row]).with_columns(
        _to_istanbul_time("observed_at_utc"),
        *_numeric_columns(row),
        pl.lit(point_id).alias("weather_point_id"),
        pl.lit(fetched_at).alias("fetched_at_utc"),
        pl.lit("current").alias("source"),
    )
