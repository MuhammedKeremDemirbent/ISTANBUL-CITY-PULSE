import logging
from collections.abc import Mapping
from typing import Final

import polars as pl

from city_pulse.features.columns import (
    COLUMN_DTYPES,
    FRAME_COLUMNS,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
)

log = logging.getLogger(__name__)

HEAVY_RAIN_THRESHOLD_MM: Final[float] = 3.0

# Map of lag column name to hour offset
LAG_HOURS: Final[Mapping[str, int]] = {
    "traffic_index_av_lag_1h": 1,
    "traffic_index_av_lag_24h": 24,
}


class FeatureSchemaError(ValueError):
    """The input frame does not match the feature contract."""


def build_training_frame(
    main: pl.DataFrame,
    holidays: pl.DataFrame | None = None,
    school_calendar: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build the final training frame from the analytical gold table.

    - Attaches holiday, rain, lag, and school calendar features.
    - Validates schema against FRAME_COLUMNS.
    - Enforces strict data types and drops incomplete rows.
    """
    main = _attach_holiday_flag(main, holidays)
    main = _attach_rain_flags(main)
    main = _attach_lag_features(main)
    main = _attach_school_flags(main, school_calendar)
    _validate_schema(main)
    frame = main.select(FRAME_COLUMNS).with_columns(
        [pl.col(name).cast(dtype) for name, dtype in COLUMN_DTYPES.items()]
    )
    return _drop_incomplete_rows(frame).sort(TIMESTAMP_COLUMN)


def _attach_holiday_flag(
    main: pl.DataFrame,
    holidays: pl.DataFrame | None,
) -> pl.DataFrame:
    """Attach is_holiday indicator from holidays table."""
    if holidays is not None and holidays.height > 0:
        holiday_dates = holidays.select(pl.col("date").cast(pl.Date))["date"]
        return main.with_columns(
            pl.col(TIMESTAMP_COLUMN)
            .dt.date()
            .is_in(holiday_dates)
            .cast(pl.Int32)
            .alias("is_holiday")
        )
    log.debug("Holidays table not provided; defaulting is_holiday to 0.")
    return main.with_columns(pl.lit(0).cast(pl.Int32).alias("is_holiday"))


def _attach_rain_flags(main: pl.DataFrame) -> pl.DataFrame:
    """Derive is_raining and heavy_rain flags from precipitation."""
    if "precipitation" not in main.columns:
        raise FeatureSchemaError("Column 'precipitation' not found for rain flags.")
    return main.with_columns(
        (pl.col("precipitation") > 0.0).cast(pl.Int32).alias("is_raining"),
        (pl.col("precipitation") >= HEAVY_RAIN_THRESHOLD_MM)
        .cast(pl.Int32)
        .alias("heavy_rain"),
    )


def _attach_lag_features(main: pl.DataFrame) -> pl.DataFrame:
    """Attach time-aligned lag values of the target variable using left joins.

    Prevents gap shifting issues in non-continuous time series.
    """
    if TARGET_COLUMN not in main.columns:
        raise FeatureSchemaError(
            f"Target column '{TARGET_COLUMN}' not found for lag features."
        )
    if main[TIMESTAMP_COLUMN].n_unique() != main.height:
        raise FeatureSchemaError(
            f"Duplicate values in '{TIMESTAMP_COLUMN}'; cannot perform lag join."
        )

    sorted_main = main.sort(TIMESTAMP_COLUMN)
    result = sorted_main
    for col_name, hours in LAG_HOURS.items():
        offset = pl.duration(hours=hours)
        lag_table = sorted_main.select(
            (pl.col(TIMESTAMP_COLUMN) + offset).alias(TIMESTAMP_COLUMN),
            pl.col(TARGET_COLUMN).alias(col_name),
        )
        result = result.join(lag_table, on=TIMESTAMP_COLUMN, how="left")

    return result


def _attach_school_flags(
    main: pl.DataFrame,
    school_calendar: pl.DataFrame | None,
) -> pl.DataFrame:
    """Attach school term and rush-hour flags from school calendar."""
    school_cols = (
        "is_school_term",
        "is_school_morning_rush",
        "is_school_afternoon_rush",
    )
    if all(c in main.columns for c in school_cols):
        log.debug("School flags already exist in table; skipping join.")
        return main
    if school_calendar is not None and school_calendar.height > 0:
        cal = school_calendar.select(TIMESTAMP_COLUMN, *school_cols)
        joined = main.join(cal, on=TIMESTAMP_COLUMN, how="left")
        return joined.with_columns(
            [pl.col(c).fill_null(0).cast(pl.Int32) for c in school_cols]
        )
    log.debug("School calendar not provided; defaulting school flags to 0.")
    return main.with_columns([pl.lit(0).cast(pl.Int32).alias(c) for c in school_cols])


def _validate_schema(main: pl.DataFrame) -> None:
    missing = [name for name in FRAME_COLUMNS if name not in main.columns]
    if missing:
        raise FeatureSchemaError(f"Missing required columns: {missing}")
    timestamp_dtype = main.schema[TIMESTAMP_COLUMN]
    if not isinstance(timestamp_dtype, pl.Datetime):
        raise FeatureSchemaError(
            f"'{TIMESTAMP_COLUMN}' must be of Datetime type, found: {timestamp_dtype}"
        )


def _drop_incomplete_rows(frame: pl.DataFrame) -> pl.DataFrame:
    complete = frame.drop_nulls()
    dropped = frame.height - complete.height
    if dropped:
        log.warning(
            "Dropped %d/%d rows due to missing values (%.2f%%)",
            dropped,
            frame.height,
            100 * dropped / frame.height,
        )
    return complete
