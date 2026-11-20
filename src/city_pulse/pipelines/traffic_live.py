import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple, Protocol

import polars as pl

from city_pulse.ingest.ibb_client import IbbTrafficClient, IbbTrafficResponse
from city_pulse.ingest.ibb_frames import ibb_traffic_to_frame
from city_pulse.storage import FileStore, Store

log = logging.getLogger(__name__)

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


class TrafficSource(Protocol):
    """Traffic data provider protocol."""

    def fetch_traffic_index(self) -> IbbTrafficResponse: ...


class TrafficResult(NamedTuple):
    frame: pl.DataFrame


def run_traffic_live(
    *,
    source: TrafficSource,
    store: Store,
    clock: Clock = utc_now,
) -> TrafficResult:
    """Fetch live IBB traffic index and store as Bronze JSON and Silver Parquet."""
    fetched_at = clock()
    stamp = fetched_at.strftime("%Y%m%dT%H%M%SZ")

    # 1. Fetch raw data from API
    response = source.fetch_traffic_index()

    # 2. Bronze: Store raw JSON
    store.write_json(
        f"bronze/ibb_traffic_live/{stamp}.json",
        response.model_dump(by_alias=True),
    )

    # 3. Silver: Clean single-row dataframe
    frame = ibb_traffic_to_frame(response, fetched_at)
    store.write_parquet(f"silver/traffic_live/traffic_{stamp}.parquet", frame)

    log.info(
        "Saved live IBB traffic (%s) -> Total: %%%d, Asian: %%%d, European: %%%d",
        stamp,
        response.traffic_index,
        response.traffic_index_an,
        response.traffic_index_av,
    )
    return TrafficResult(frame)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    result = run_traffic_live(
        source=IbbTrafficClient(),
        store=FileStore(Path("data")),
    )
    log.info("Live IBB traffic frame:\n%s", result.frame)


if __name__ == "__main__":
    main()
