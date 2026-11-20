"""Global pytest configuration and centralized test fixtures.

Shared mock datasets, fixed timestamps, and sample DataFrame generators
for the test suite.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from city_pulse.domain import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixed_now_utc() -> datetime:
    """Return fixed UTC datetime for reproducible tests."""
    return datetime(2026, 11, 16, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def sample_ibb_traffic_payload() -> dict[str, Any]:
    """Sample API response for IBB Traffic Index."""
    file_path = FIXTURES_DIR / "ibb_traffic_response.json"
    if file_path.exists():
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    return {"TI": 20, "TI_An": 13, "TI_Av": 26}


@pytest.fixture
def sample_openmeteo_payload() -> dict[str, Any]:
    """Sample API response for Open-Meteo hourly weather."""
    file_path = FIXTURES_DIR / "openmeteo_response.json"
    if file_path.exists():
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    return {
        "hourly": {
            "time": ["2026-11-16T00:00", "2026-11-16T01:00"],
            "temperature_2m": [15.0, 14.5],
            "relative_humidity_2m": [75, 78],
            "precipitation": [0.0, 0.0],
            "wind_speed_10m": [12.0, 11.5],
        }
    }


@pytest.fixture
def sample_traffic_df() -> pl.DataFrame:
    """Sample Polars DataFrame containing basic traffic measurements."""
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 11, 16, 10, 0, 0),
                datetime(2026, 11, 16, 11, 0, 0),
                datetime(2026, 11, 16, 12, 0, 0),
            ],
            "traffic_index": [35, 42, 50],
            "traffic_index_an": [30, 38, 45],
            "traffic_index_av": [40, 46, 55],
        }
    )


@pytest.fixture
def sample_features_df() -> pl.DataFrame:
    """Synthetic features DataFrame for model training and evaluation tests."""
    n_rows = 100
    dates = [
        datetime(2026, 1, 1, 0, 0, tzinfo=UTC) + pl.duration(hours=i)
        for i in range(n_rows)
    ]
    data: dict[str, Any] = {
        "timestamp": dates,
        TARGET_COLUMN: [float(30 + (i % 20)) for i in range(n_rows)],
    }
    for col in FEATURE_COLUMNS:
        data[col] = [float(10 + (j % 15)) for j in range(n_rows)]

    return pl.DataFrame(data)
