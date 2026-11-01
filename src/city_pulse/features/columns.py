"""Hangi kolon girdi hangisi çıktı olduğunu burada tanımlıyoruz."""

from collections.abc import Mapping
from typing import Final

import polars as pl

from city_pulse.domain import FEATURE_COLUMNS as DOMAIN_FEATURE_COLUMNS
from city_pulse.domain import TARGET_COLUMN as DOMAIN_TARGET_COLUMN

TIMESTAMP_COLUMN: Final = "timestamp"
TARGET_COLUMN: Final = DOMAIN_TARGET_COLUMN

BASELINE_KEYS: Final[tuple[str, ...]] = ("day_of_week", "hour")
FEATURE_COLUMNS: Final[tuple[str, ...]] = DOMAIN_FEATURE_COLUMNS

FRAME_COLUMNS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys((TIMESTAMP_COLUMN, *BASELINE_KEYS, *FEATURE_COLUMNS, TARGET_COLUMN))
)

COLUMN_DTYPES: Final[Mapping[str, type[pl.DataType]]] = {
    "day_of_week": pl.Int32,
    "hour": pl.Int32,
    "month": pl.Int32,
    "is_weekend": pl.Int32,
    "temperature_2m": pl.Float64,
    "precipitation": pl.Float64,
    "wind_speed_10m": pl.Float64,
    "relative_humidity_2m": pl.Float64,
    TARGET_COLUMN: pl.Float64,
}
