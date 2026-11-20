"""Unit tests for _attach_lag_features function.

Why timestamp-based matching is more accurate than row shifting:
shift(N) means 'take the value N rows prior'. If there are hour gaps,
the row after the gap carries data from the wrong time.
Timestamp-based join means 'if a row exists with timestamp - N hours, take its value;
otherwise leave NULL', which preserves true data integrity.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest

from city_pulse.features.builder import FeatureSchemaError, _attach_lag_features
from city_pulse.features.columns import TARGET_COLUMN, TIMESTAMP_COLUMN


def _make_frame(hours: list[int]) -> pl.DataFrame:
    """Generate hourly traffic table from a list of hour offsets."""
    base = datetime(2024, 1, 1, 0, 0)
    timestamps = [base + timedelta(hours=h) for h in hours]
    return pl.DataFrame(
        {
            TIMESTAMP_COLUMN: timestamps,
            TARGET_COLUMN: [float(h) for h in hours],
        }
    ).with_columns(pl.col(TIMESTAMP_COLUMN).cast(pl.Datetime))


@pytest.mark.unit
def test_lag_1h_is_null_when_previous_hour_missing() -> None:
    """When there is an hour gap, lag_1h must be NULL."""
    frame = _make_frame([0, 1, 2, 5])
    result = _attach_lag_features(frame).sort(TIMESTAMP_COLUMN)

    lag_1h = result["traffic_index_av_lag_1h"].to_list()
    assert lag_1h[0] is None, "hour 0: no prior hour -> expected None"
    assert lag_1h[1] == pytest.approx(0.0), "hour 1: prior is hour 0 -> 0.0"
    assert lag_1h[2] == pytest.approx(1.0), "hour 2: prior is hour 1 -> 1.0"
    assert lag_1h[3] is None, "hour 5: prior (hour 4) missing in table -> None"


@pytest.mark.unit
def test_lag_24h_uses_real_time_not_row_offset() -> None:
    """lag_24h must be computed from real time differences, not row offset."""
    all_hours = list(range(36))
    all_hours.remove(10)  # remove hour 10
    frame = _make_frame(all_hours)
    result = _attach_lag_features(frame).sort(TIMESTAMP_COLUMN)

    row_24 = result.filter(
        pl.col(TIMESTAMP_COLUMN) == datetime(2024, 1, 1, 0, 0) + timedelta(hours=24)
    )
    assert row_24["traffic_index_av_lag_24h"][0] == pytest.approx(0.0)

    row_25 = result.filter(
        pl.col(TIMESTAMP_COLUMN) == datetime(2024, 1, 1, 0, 0) + timedelta(hours=25)
    )
    assert row_25["traffic_index_av_lag_24h"][0] == pytest.approx(1.0)

    row_34 = result.filter(
        pl.col(TIMESTAMP_COLUMN) == datetime(2024, 1, 1, 0, 0) + timedelta(hours=34)
    )
    assert row_34["traffic_index_av_lag_24h"][0] is None


@pytest.mark.unit
def test_row_count_unchanged() -> None:
    """_attach_lag_features must not add or drop rows."""
    frame = _make_frame([0, 1, 2, 5, 10, 24, 25])
    result = _attach_lag_features(frame)
    assert result.height == frame.height


@pytest.mark.unit
def test_duplicate_timestamp_raises() -> None:
    """Duplicate timestamps cause join row multiplication; must raise error."""
    base = datetime(2024, 1, 1, 0, 0)
    frame = pl.DataFrame(
        {
            TIMESTAMP_COLUMN: [base, base, base + timedelta(hours=1)],
            TARGET_COLUMN: [1.0, 2.0, 3.0],
        }
    ).with_columns(pl.col(TIMESTAMP_COLUMN).cast(pl.Datetime))

    with pytest.raises(FeatureSchemaError, match="Duplicate values"):
        _attach_lag_features(frame)
