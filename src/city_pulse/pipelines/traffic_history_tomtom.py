import argparse
import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import polars as pl

from city_pulse.domain import ISTANBUL_CENTER, WeatherPoint
from city_pulse.ingest.tomtom_client import TomTomClient, TomTomFlowResponse
from city_pulse.ingest.tomtom_frames import flow_to_frame
from city_pulse.storage import FileStore, Store

log = logging.getLogger(__name__)

DEFAULT_START = "2026-11-03"
DEFAULT_END = "2026-11-03"


class TrafficHistorySource(Protocol):
    def fetch_flow_at(
        self, lat: float, lon: float, at: datetime
    ) -> TomTomFlowResponse: ...


def _hour_range(start: str, end: str) -> list[datetime]:
    """start ve end (dahil) arasında saatlik tarih listesi döndürür (UTC)."""
    start_dt = datetime.fromisoformat(start).replace(tzinfo=UTC)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=UTC)
    hours: list[datetime] = []
    current = start_dt
    while current <= end_dt:
        hours.append(current)
        current += timedelta(hours=1)
    return hours


def run_traffic_history_tomtom(
    point: WeatherPoint,
    start: str,
    end: str,
    *,
    source: TrafficHistorySource,
    store: Store,
) -> pl.DataFrame:
    """Belirtilen tarih aralığında saatlik TomTom trafik verisini çeker."""
    hours = _hour_range(start, end)
    frames: list[pl.DataFrame] = []

    for at in hours:
        pid = point.point_id
        stamp = at.strftime("%Y%m%dT%H%M%SZ")
        log.info("%s: trafik akışı çekiliyor %s", pid, stamp)

        response = source.fetch_flow_at(point.lat, point.lon, at)
        store.write_json(
            f"bronze/tomtom_history/{pid}/{stamp}.json",
            response.model_dump(),
        )

        frame = flow_to_frame(response, pid, point.lat, point.lon, at)
        frames.append(frame)

    if not frames:
        raise ValueError("Hiç saat bulunamadı.")

    result = pl.concat(frames)
    key = f"{point.point_id}_{start}_{end}"
    store.write_parquet(
        f"silver/traffic_history_tomtom/{pid}/traffic_{key}.parquet",
        result,
    )

    log.info(
        "%s: %d saatlik trafik satırı yazıldı (%s - %s)",
        pid,
        result.height,
        start,
        end,
    )
    return result


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TomTom saatlik trafik akışı geçmiş verisi"
    )
    parser.add_argument("start", nargs="?", default=DEFAULT_START, type=lambda s: s)
    parser.add_argument("end", nargs="?", default=DEFAULT_END, type=lambda s: s)
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("Başlangıç tarihi bitiş tarihinden sonra olamaz")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args(argv)
    df = run_traffic_history_tomtom(
        ISTANBUL_CENTER,
        args.start,
        args.end,
        source=TomTomClient(),
        store=FileStore(Path("data")),
    )
    log.info("şema: %s", dict(df.schema))


if __name__ == "__main__":
    main()
