"""Anlık (current) ve saatlik tahmin (forecast): bronze (ham JSON) + silver (Parquet).

Kullanım:
    uv run python -m city_pulse.pipelines.weather_live
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple, Protocol

import polars as pl

from city_pulse.domain import ISTANBUL_CENTER, WeatherPoint
from city_pulse.ingest.openmeteo_client import OpenMeteoClient
from city_pulse.ingest.openmeteo_frames import current_to_frame, forecast_to_frame
from city_pulse.storage import FileStore, Store

log = logging.getLogger(__name__)

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


class LiveSource(Protocol):
    def fetch_forecast(self, point: WeatherPoint) -> dict: ...


class LiveResult(NamedTuple):
    current: pl.DataFrame
    forecast: pl.DataFrame


def run_live(
    point: WeatherPoint, *, source: LiveSource, store: Store, clock: Clock = utc_now
) -> LiveResult:
    fetched_at = clock()
    stamp = fetched_at.strftime("%Y%m%dT%H%M%SZ")
    pid = point.point_id

    payload = source.fetch_forecast(point)
    store.write_json(f"bronze/openmeteo_live/{pid}/{stamp}.json", payload)  # ham veri

    current = current_to_frame(payload, pid, fetched_at)
    forecast = forecast_to_frame(payload, pid, fetched_at)
    store.write_parquet(f"silver/weather_live/{pid}/current_{stamp}.parquet", current)
    store.write_parquet(f"silver/weather_live/{pid}/forecast_{stamp}.parquet", forecast)

    log.info("%s: anlık 1 satır, tahmin %d satır (%s)", pid, forecast.height, stamp)
    return LiveResult(current, forecast)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    result = run_live(
        ISTANBUL_CENTER, source=OpenMeteoClient(), store=FileStore(Path("data"))
    )
    log.info("anlık:\n%s", result.current)
    log.info("tahmin (ilk 3 satır):\n%s", result.forecast.head(3))


if __name__ == "__main__":
    main()
