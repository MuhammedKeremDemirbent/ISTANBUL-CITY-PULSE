"""Open-Meteo veri donusumleri (frames) icin birim testler."""

from datetime import UTC, datetime

import polars as pl
import pytest

from city_pulse.ingest.openmeteo_frames import archive_to_frame, forecast_to_frame


@pytest.mark.unit
def test_openmeteo_forecast_to_frame(sample_openmeteo_payload: dict) -> None:
    fetched_at = datetime(2026, 11, 16, 12, 0, 0, tzinfo=UTC)
    frame = forecast_to_frame(
        sample_openmeteo_payload, point_id="kadikoy", fetched_at=fetched_at
    )

    assert isinstance(frame, pl.DataFrame)
    assert frame.height == len(sample_openmeteo_payload["hourly"]["time"])
    assert "timestamp" in frame.columns
    assert "weather_point_id" in frame.columns
    assert "source" in frame.columns
    assert "temperature_2m" in frame.columns
    assert frame["source"][0] == "forecast"
    assert frame["weather_point_id"][0] == "kadikoy"


@pytest.mark.unit
def test_openmeteo_archive_to_frame(sample_openmeteo_payload: dict) -> None:
    frame = archive_to_frame(sample_openmeteo_payload, point_id="besiktas")

    assert isinstance(frame, pl.DataFrame)
    assert frame["source"][0] == "era5"
    assert frame["weather_point_id"][0] == "besiktas"
