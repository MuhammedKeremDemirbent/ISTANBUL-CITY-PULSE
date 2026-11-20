from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from city_pulse.domain import WeatherPoint
from city_pulse.pipelines.add_live_datas import (
    GOLD_COLUMNS_SCHEMA,
    add_live_datas_to_gold,
    build_live_gold_rows,
)


def _make_dummy_traffic(hours: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": hours,
            "fetched_at_utc": hours,
            "source": ["ibb_tkm"] * len(hours),
            "traffic_index": [30.0] * len(hours),
            "traffic_index_an": [25.0] * len(hours),
            "traffic_index_av": [35.0] * len(hours),
        }
    )


def _make_dummy_weather(hours: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": hours,
            "temperature_2m": [18.5] * len(hours),
            "precipitation": [0.0] * len(hours),
            "wind_speed_10m": [12.0] * len(hours),
            "relative_humidity_2m": [65.0] * len(hours),
        }
    )


class DummyWeatherSource:
    """Mock Open-Meteo data source for self-healing tests."""

    def __init__(self, sample_payload: dict[str, Any]) -> None:
        self._payload = sample_payload

    def fetch_forecast(self, point: WeatherPoint, forecast_days: int = 1) -> dict:
        return self._payload


@pytest.mark.unit
def test_build_live_gold_rows_schema() -> None:
    dt1 = datetime(2026, 11, 14, 10, 0, 0, tzinfo=UTC)
    dt2 = datetime(2026, 11, 14, 11, 0, 0, tzinfo=UTC)
    hours = [dt1, dt2]

    traffic = _make_dummy_traffic(hours)
    weather = _make_dummy_weather(hours)

    rows = build_live_gold_rows(traffic, weather)

    assert rows.height == 2
    assert list(rows.columns) == list(GOLD_COLUMNS_SCHEMA.keys())
    assert rows["hour"].to_list() == [10, 11]
    assert rows["traffic_index_av"].to_list() == [35, 35]
    assert rows["is_weekend"].to_list() == [1, 1]


@pytest.mark.unit
def test_build_live_gold_rows_drops_missing_weather() -> None:
    """Hourly tolerance: hours without matching weather data cannot enter Gold."""
    dt1 = datetime(2026, 11, 14, 10, 0, 0, tzinfo=UTC)
    dt2 = datetime(2026, 11, 14, 11, 0, 0, tzinfo=UTC)  # No weather for this hour

    traffic = _make_dummy_traffic([dt1, dt2])
    weather = _make_dummy_weather([dt1])  # Only dt1 exists

    rows = build_live_gold_rows(traffic, weather)

    # dt2 must be dropped, only dt1 retained
    assert rows.height == 1
    assert rows["hour"].to_list() == [10]
    assert rows["temperature_2m"][0] == 18.5


@pytest.mark.unit
def test_build_live_gold_rows_drops_null_weather() -> None:
    """Records with null weather columns cannot be filled with 0.0 and must be dropped."""
    dt1 = datetime(2026, 11, 14, 10, 0, 0, tzinfo=UTC)
    traffic = _make_dummy_traffic([dt1])
    weather_with_null = pl.DataFrame(
        {
            "timestamp": [dt1],
            "temperature_2m": [None],
            "precipitation": [0.0],
            "wind_speed_10m": [12.0],
            "relative_humidity_2m": [65.0],
        }
    )

    rows = build_live_gold_rows(traffic, weather_with_null)
    assert rows.height == 0


@pytest.mark.unit
def test_add_live_datas_to_gold_flow(tmp_path: Path) -> None:
    traffic_dir = tmp_path / "silver" / "traffic_live"
    weather_dir = tmp_path / "silver" / "weather_live"
    gold_dir = tmp_path / "gold"
    archive_dir = tmp_path / "silver" / "traffic_live_archived"

    traffic_dir.mkdir(parents=True)
    weather_dir.mkdir(parents=True)
    gold_dir.mkdir(parents=True)

    # 1. Initial Gold file
    gold_file = gold_dir / "city_pulse_hourly.parquet"
    past_dt = datetime(2026, 11, 14, 8, 0, 0)
    initial_gold = pl.DataFrame(
        {
            "timestamp": [past_dt],
            "hour": [8],
            "day_of_week": [6],
            "month": [11],
            "is_weekend": [1],
            "is_holiday": [0],
            "is_school_term": [1],
            "is_school_morning_rush": [0],
            "is_school_afternoon_rush": [0],
            "traffic_index": [40],
            "traffic_index_an": [35],
            "traffic_index_av": [45],
            "temperature_2m": [15.0],
            "precipitation": [0.0],
            "wind_speed_10m": [10.0],
            "relative_humidity_2m": [70.0],
        },
        schema=GOLD_COLUMNS_SCHEMA,
    )
    initial_gold.write_parquet(gold_file)

    # 2. Live weather forecast file
    dt9 = datetime(2026, 11, 14, 9, 0, 0, tzinfo=UTC)
    dt10 = datetime(2026, 11, 14, 10, 0, 0, tzinfo=UTC)
    weather_df = _make_dummy_weather([dt9, dt10])
    weather_df.write_parquet(weather_dir / "forecast_20261114T090000Z.parquet")

    # 3. Two hourly live IBB traffic files
    tf9 = _make_dummy_traffic([dt9])
    tf10 = _make_dummy_traffic([dt10])
    tf9.write_parquet(traffic_dir / "traffic_20261114T090000Z.parquet")
    tf10.write_parquet(traffic_dir / "traffic_20261114T100000Z.parquet")

    # 4. Run pipeline
    added = add_live_datas_to_gold(
        live_traffic_dir=traffic_dir,
        live_weather_dir=weather_dir,
        gold_file=gold_file,
        archive_dir=archive_dir,
    )

    assert added == 2

    # Check updated Gold file
    updated_gold = pl.read_parquet(gold_file)
    assert updated_gold.height == 3
    assert updated_gold["hour"].to_list() == [8, 9, 10]

    # Verify live files were moved to archive
    assert len(list(traffic_dir.glob("traffic_*.parquet"))) == 0
    assert len(list(archive_dir.glob("traffic_*.parquet"))) == 2


@pytest.mark.unit
def test_add_live_datas_to_gold_no_match_keeps_clean_and_no_archive(
    tmp_path: Path,
) -> None:
    """If weather does not match, Gold must not be touched and traffic files must not be archived."""
    traffic_dir = tmp_path / "silver" / "traffic_live"
    weather_dir = tmp_path / "silver" / "weather_live"
    gold_dir = tmp_path / "gold"
    archive_dir = tmp_path / "silver" / "traffic_live_archived"

    traffic_dir.mkdir(parents=True)
    weather_dir.mkdir(parents=True)
    gold_dir.mkdir(parents=True)

    gold_file = gold_dir / "city_pulse_hourly.parquet"
    past_dt = datetime(2026, 11, 14, 8, 0, 0)
    initial_gold = pl.DataFrame(
        {
            "timestamp": [past_dt],
            "hour": [8],
            "day_of_week": [6],
            "month": [11],
            "is_weekend": [1],
            "is_holiday": [0],
            "is_school_term": [1],
            "is_school_morning_rush": [0],
            "is_school_afternoon_rush": [0],
            "traffic_index": [40],
            "traffic_index_an": [35],
            "traffic_index_av": [45],
            "temperature_2m": [15.0],
            "precipitation": [0.0],
            "wind_speed_10m": [10.0],
            "relative_humidity_2m": [70.0],
        },
        schema=GOLD_COLUMNS_SCHEMA,
    )
    initial_gold.write_parquet(gold_file)

    # Traffic: 14 November
    dt_traffic = datetime(2026, 11, 14, 12, 0, 0, tzinfo=UTC)
    tf = _make_dummy_traffic([dt_traffic])
    tf.write_parquet(traffic_dir / "traffic_20261114T120000Z.parquet")

    # Weather: 10 November (mismatched)
    dt_old_weather = datetime(2026, 11, 10, 12, 0, 0, tzinfo=UTC)
    wf = _make_dummy_weather([dt_old_weather])
    wf.write_parquet(weather_dir / "forecast_20261110T120000Z.parquet")

    added = add_live_datas_to_gold(
        live_traffic_dir=traffic_dir,
        live_weather_dir=weather_dir,
        gold_file=gold_file,
        archive_dir=archive_dir,
    )

    assert added == 0
    # Gold remains unchanged
    assert pl.read_parquet(gold_file).height == 1
    # Traffic files are not deleted/archived
    assert len(list(traffic_dir.glob("traffic_*.parquet"))) == 1
    assert (
        not archive_dir.exists()
        or len(list(archive_dir.glob("traffic_*.parquet"))) == 0
    )


@pytest.mark.unit
def test_add_live_datas_to_gold_self_healing(tmp_path: Path) -> None:
    """Missing weather is automatically fetched via self-healing and appended to Gold."""
    traffic_dir = tmp_path / "silver" / "traffic_live"
    weather_dir = tmp_path / "silver" / "weather_live" / "istanbul_center"
    gold_dir = tmp_path / "gold"
    archive_dir = tmp_path / "silver" / "traffic_live_archived"

    traffic_dir.mkdir(parents=True)
    weather_dir.mkdir(parents=True)
    gold_dir.mkdir(parents=True)

    gold_file = gold_dir / "city_pulse_hourly.parquet"
    past_dt = datetime(2026, 11, 14, 8, 0, 0)
    initial_gold = pl.DataFrame(
        {
            "timestamp": [past_dt],
            "hour": [8],
            "day_of_week": [6],
            "month": [11],
            "is_weekend": [1],
            "is_holiday": [0],
            "is_school_term": [1],
            "is_school_morning_rush": [0],
            "is_school_afternoon_rush": [0],
            "traffic_index": [40],
            "traffic_index_an": [35],
            "traffic_index_av": [45],
            "temperature_2m": [15.0],
            "precipitation": [0.0],
            "wind_speed_10m": [10.0],
            "relative_humidity_2m": [70.0],
        },
        schema=GOLD_COLUMNS_SCHEMA,
    )
    initial_gold.write_parquet(gold_file)

    # Write traffic for 14 November 10:00
    # Open-Meteo frames converts UTC to Istanbul time (+3): 07:00 UTC -> 10:00 TRT
    target_dt = datetime(2026, 11, 14, 10, 0, 0)
    tf = _make_dummy_traffic([target_dt])
    tf.write_parquet(traffic_dir / "traffic_target.parquet")

    # Weather directory is initially empty
    assert len(list(weather_dir.glob("*.parquet"))) == 0

    complete_payload = {
        "current": {
            "time": "2026-11-14T07:00",
            "temperature_2m": 16.0,
            "relative_humidity_2m": 60.0,
            "precipitation": 0.0,
            "wind_speed_10m": 12.0,
        },
        "hourly": {
            "time": ["2026-11-14T07:00", "2026-11-14T08:00"],
            "temperature_2m": [16.0, 17.0],
            "relative_humidity_2m": [60.0, 58.0],
            "precipitation": [0.0, 0.0],
            "wind_speed_10m": [12.0, 11.0],
        },
    }
    mock_source = DummyWeatherSource(complete_payload)

    added = add_live_datas_to_gold(
        live_traffic_dir=traffic_dir,
        live_weather_dir=weather_dir,
        gold_file=gold_file,
        archive_dir=archive_dir,
        weather_source=mock_source,
    )

    assert added == 1
    # Weather saved via self-healing
    assert len(list(weather_dir.glob("forecast_*.parquet"))) >= 1
    # Gold updated
    updated_gold = pl.read_parquet(gold_file)
    assert updated_gold.height == 2
    # Traffic archived
    assert len(list(traffic_dir.glob("traffic_*.parquet"))) == 0
    assert len(list(archive_dir.glob("traffic_*.parquet"))) == 1
