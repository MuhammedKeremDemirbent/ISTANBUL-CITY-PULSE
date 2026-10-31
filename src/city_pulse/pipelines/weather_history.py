"""Geçmiş (ERA5) saatlik hava verisi: bronze (ham JSON) + silver (Parquet).

Kullanım:
    uv run python -m city_pulse.pipelines.weather_history 2024-01-01 2024-12-31
"""

import argparse
import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Protocol

import polars as pl

from city_pulse.domain import ISTANBUL_CENTER, WeatherPoint
from city_pulse.ingest.openmeteo_client import OpenMeteoClient
from city_pulse.ingest.openmeteo_frames import archive_to_frame
from city_pulse.storage import FileStore, Store

log = logging.getLogger(__name__)

DEFAULT_START = "2023-01-01"
DEFAULT_END = "2025-12-31"


class ArchiveSource(Protocol):
    def fetch_archive(self, point: WeatherPoint, start: str, end: str) -> dict: ...


def run_history(
    point: WeatherPoint, start: str, end: str, *, source: ArchiveSource, store: Store
) -> pl.DataFrame:
    payload = source.fetch_archive(point, start, end)

    key = f"{point.point_id}_{start}_{end}"
    store.write_json(f"bronze/openmeteo_era5/{key}.json", payload)  # ham veri

    df = archive_to_frame(payload, point.point_id)
    store.write_parquet(f"silver/weather_hourly/{key}.parquet", df)  # tablo

    log.info("%s: %d satır yazıldı (%s - %s)", point.point_id, df.height, start, end)
    return df


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ERA5 geçmiş hava verisi")
    parser.add_argument(
        "start", nargs="?", default=DEFAULT_START, type=date.fromisoformat
    )
    parser.add_argument("end", nargs="?", default=DEFAULT_END, type=date.fromisoformat)
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("başlangıç tarihi bitişten sonra olamaz")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args(argv)
    df = run_history(
        ISTANBUL_CENTER,
        args.start.isoformat(),
        args.end.isoformat(),
        source=OpenMeteoClient(),
        store=FileStore(Path("data")),
    )
    log.info("şema: %s", dict(df.schema))


if __name__ == "__main__":
    main()
