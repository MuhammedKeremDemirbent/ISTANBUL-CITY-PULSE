import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple, Protocol

import polars as pl

from city_pulse.domain import ISTANBUL_CENTER, WeatherPoint
from city_pulse.ingest.tomtom_client import TomTomClient, TomTomFlowResponse
from city_pulse.ingest.tomtom_frames import flow_to_frame
from city_pulse.storage import FileStore, Store

log = logging.getLogger(__name__)

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


class TrafficSource(Protocol):
    def fetch_flow(self, lat: float, lon: float) -> TomTomFlowResponse: ...


class TrafficResult(NamedTuple):
    frame: pl.DataFrame


def run_traffic_live(
    point: WeatherPoint,
    *,
    source: TrafficSource,
    store: Store,
    clock: Clock = utc_now,
) -> TrafficResult:
    """Anlık trafik verisini çeker, bronze JSON ve silver Parquet olarak kaydeder."""
    fetched_at = clock()
    stamp = fetched_at.strftime("%Y%m%dT%H%M%SZ")
    pid = point.point_id

    response = source.fetch_flow(point.lat, point.lon)
    store.write_json(f"bronze/tomtom_live/{pid}/{stamp}.json", response.model_dump())

    frame = flow_to_frame(response, pid, point.lat, point.lon, fetched_at)
    store.write_parquet(f"silver/traffic_live/{pid}/traffic_{stamp}.parquet", frame)

    log.info("%s: anlık trafik 1 satır (%s)", pid, stamp)
    return TrafficResult(frame)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    result = run_traffic_live(
        ISTANBUL_CENTER,
        source=TomTomClient(),
        store=FileStore(Path("data")),
    )
    log.info("anlık trafik:\n%s", result.frame)


if __name__ == "__main__":
    main()
