from datetime import datetime

import polars as pl
import pytest

from city_pulse.features.columns import TIMESTAMP_COLUMN
from city_pulse.ml.split import SplitConfig, time_based_split

CONFIG = SplitConfig(train_end=datetime(2025, 1, 6), valid_end=datetime(2025, 1, 8))


def _hourly_frame() -> pl.DataFrame:
    """Hourly data between 2025-01-01 00:00 and 2025-01-10 23:00, 240 rows."""
    timestamps = pl.datetime_range(
        datetime(2025, 1, 1), datetime(2025, 1, 10, 23), interval="1h", eager=True
    )
    return pl.DataFrame({TIMESTAMP_COLUMN: timestamps, "x": range(timestamps.len())})


@pytest.mark.unit
def test_parts_are_disjoint_ordered_and_complete():
    frame = _hourly_frame()
    split = time_based_split(frame, CONFIG)

    assert split.train.height + split.valid.height + split.test.height == frame.height
    assert split.train[TIMESTAMP_COLUMN].max() < CONFIG.train_end
    assert split.valid[TIMESTAMP_COLUMN].min() >= CONFIG.train_end
    assert split.valid[TIMESTAMP_COLUMN].max() < CONFIG.valid_end
    assert split.test[TIMESTAMP_COLUMN].min() >= CONFIG.valid_end


@pytest.mark.unit
def test_boundaries_are_half_open():
    split = time_based_split(_hourly_frame(), CONFIG)
    assert split.valid[TIMESTAMP_COLUMN].min() == CONFIG.train_end
    assert split.test[TIMESTAMP_COLUMN].min() == CONFIG.valid_end


@pytest.mark.unit
def test_row_order_is_preserved():
    split = time_based_split(_hourly_frame(), CONFIG)
    for part in split:
        assert part[TIMESTAMP_COLUMN].is_sorted()


@pytest.mark.unit
def test_empty_part_raises_with_its_name():
    config = SplitConfig(train_end=datetime(2024, 1, 1), valid_end=datetime(2025, 1, 5))
    with pytest.raises(ValueError, match="train"):
        time_based_split(_hourly_frame(), config)


@pytest.mark.unit
def test_missing_timestamp_column_raises():
    with pytest.raises(ValueError, match="not found"):
        time_based_split(_hourly_frame().drop(TIMESTAMP_COLUMN), CONFIG)


@pytest.mark.unit
def test_null_timestamp_raises():
    frame = _hourly_frame().with_columns(
        pl.when(pl.col("x") == 0)
        .then(None)
        .otherwise(pl.col(TIMESTAMP_COLUMN))
        .alias(TIMESTAMP_COLUMN)
    )
    with pytest.raises(ValueError, match="Null values"):
        time_based_split(frame, CONFIG)
