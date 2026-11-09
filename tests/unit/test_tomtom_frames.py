"""TomTom frames dönüştürücüleri için birim testler. Ağdan bağımsızdır."""

from datetime import UTC, datetime

import polars as pl

from city_pulse.ingest.tomtom_client import TomTomFlowResponse, TomTomFlowSegment
from city_pulse.ingest.tomtom_frames import flow_to_frame


def _make_segment() -> TomTomFlowSegment:
    return TomTomFlowSegment(
        current_speed_kmh=45.5,
        free_flow_speed_kmh=60.0,
        current_travel_time_s=120.0,
        free_flow_travel_time_s=90.0,
        confidence=0.9,
        road_closure=False,
    )


def _make_response() -> TomTomFlowResponse:
    return TomTomFlowResponse(flow_segment_data=_make_segment())


def test_flow_to_frame_returns_expected_columns() -> None:
    response = _make_response()
    fetched_at = datetime(2026, 11, 4, 10, 0, 0, tzinfo=UTC)

    df = flow_to_frame(
        response,
        point_id="istanbul_center",
        lat=41.01,
        lon=28.98,
        fetched_at=fetched_at,
    )

    expected_columns = {
        "point_id",
        "latitude",
        "longitude",
        "fetched_at_utc",
        "source",
        "current_speed_kmh",
        "free_flow_speed_kmh",
        "current_travel_time_s",
        "free_flow_travel_time_s",
        "confidence",
        "road_closure",
    }
    assert set(df.columns) == expected_columns


def test_flow_to_frame_single_row() -> None:
    response = _make_response()
    fetched_at = datetime(2026, 11, 4, 10, 0, 0, tzinfo=UTC)

    df = flow_to_frame(
        response,
        point_id="istanbul_center",
        lat=41.01,
        lon=28.98,
        fetched_at=fetched_at,
    )

    assert df.height == 1


def test_flow_to_frame_values_match() -> None:
    response = _make_response()
    fetched_at = datetime(2026, 11, 4, 10, 0, 0, tzinfo=UTC)

    df = flow_to_frame(
        response,
        point_id="istanbul_center",
        lat=41.01,
        lon=28.98,
        fetched_at=fetched_at,
    )

    row = df.to_dicts()[0]
    assert row["point_id"] == "istanbul_center"
    assert row["latitude"] == 41.01
    assert row["longitude"] == 28.98
    assert row["source"] == "tomtom"
    assert row["current_speed_kmh"] == 45.5
    assert row["free_flow_speed_kmh"] == 60.0
    assert row["current_travel_time_s"] == 120.0
    assert row["free_flow_travel_time_s"] == 90.0
    assert row["confidence"] == 0.9
    assert row["road_closure"] is False


def test_flow_to_frame_fetched_at_is_set() -> None:
    response = _make_response()
    fetched_at = datetime(2026, 11, 4, 10, 0, 0, tzinfo=UTC)

    df = flow_to_frame(
        response,
        point_id="istanbul_center",
        lat=41.01,
        lon=28.98,
        fetched_at=fetched_at,
    )

    assert df["fetched_at_utc"][0] == fetched_at


def test_flow_to_frame_numeric_columns_are_float64() -> None:
    response = _make_response()
    fetched_at = datetime(2026, 11, 4, 10, 0, 0, tzinfo=UTC)

    df = flow_to_frame(
        response,
        point_id="istanbul_center",
        lat=41.01,
        lon=28.98,
        fetched_at=fetched_at,
    )

    numeric_cols = [
        "current_speed_kmh",
        "free_flow_speed_kmh",
        "current_travel_time_s",
        "free_flow_travel_time_s",
        "confidence",
    ]
    for col in numeric_cols:
        assert df[col].dtype == pl.Float64


def test_flow_to_frame_with_none_segment() -> None:
    response = TomTomFlowResponse(flow_segment_data=None)
    fetched_at = datetime(2026, 11, 4, 10, 0, 0, tzinfo=UTC)

    df = flow_to_frame(
        response,
        point_id="istanbul_center",
        lat=41.01,
        lon=28.98,
        fetched_at=fetched_at,
    )

    row = df.to_dicts()[0]
    assert row["current_speed_kmh"] is None
    assert row["confidence"] is None
    assert row["road_closure"] is None
    assert row["source"] == "tomtom"
