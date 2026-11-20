from datetime import UTC, datetime

import polars as pl
import pytest

from city_pulse.ingest.ibb_client import IbbTrafficResponse
from city_pulse.ingest.ibb_frames import ibb_traffic_to_frame
from city_pulse.pipelines.traffic_live import run_traffic_live


class FakeTrafficSource:
    def __init__(self, response: IbbTrafficResponse) -> None:
        self.response = response
        self.calls = 0

    def fetch_traffic_index(self) -> IbbTrafficResponse:
        self.calls += 1
        return self.response


class MemoryStore:
    def __init__(self) -> None:
        self.json_writes: dict[str, dict] = {}
        self.parquet_writes: dict[str, pl.DataFrame] = {}

    def write_json(self, key: str, payload: dict) -> None:
        self.json_writes[key] = payload

    def write_parquet(self, key: str, frame: pl.DataFrame) -> None:
        self.parquet_writes[key] = frame

    def read_parquet(self, key: str) -> pl.DataFrame:
        if key not in self.parquet_writes:
            raise FileNotFoundError(key)
        return self.parquet_writes[key]


@pytest.mark.unit
def test_ibb_traffic_response_model() -> None:
    data = {"TI": 25, "TI_An": 18, "TI_Av": 32}
    resp = IbbTrafficResponse.model_validate(data)
    assert resp.traffic_index == 25
    assert resp.traffic_index_an == 18
    assert resp.traffic_index_av == 32
    assert resp.model_dump(by_alias=True) == data


@pytest.mark.unit
def test_ibb_traffic_to_frame() -> None:
    resp = IbbTrafficResponse(
        traffic_index=20,
        traffic_index_an=13,
        traffic_index_av=26,
    )
    test_dt = datetime(2026, 11, 14, 15, 42, 10, tzinfo=UTC)
    frame = ibb_traffic_to_frame(resp, test_dt)

    assert frame.height == 1
    assert set(frame.columns) == {
        "timestamp",
        "fetched_at_utc",
        "source",
        "traffic_index",
        "traffic_index_an",
        "traffic_index_av",
    }
    row = frame.to_dicts()[0]
    assert row["traffic_index"] == 20.0
    assert row["traffic_index_an"] == 13.0
    assert row["traffic_index_av"] == 26.0
    assert row["source"] == "ibb_tkm"
    assert row["timestamp"] == datetime(2026, 11, 14, 18, 0, 0)


@pytest.mark.unit
def test_run_traffic_live_orchestration() -> None:
    resp = IbbTrafficResponse(
        traffic_index=45,
        traffic_index_an=40,
        traffic_index_av=50,
    )
    source = FakeTrafficSource(resp)
    store = MemoryStore()
    fixed_time = datetime(2026, 11, 14, 18, 0, 0, tzinfo=UTC)

    result = run_traffic_live(
        source=source,
        store=store,  # type: ignore[arg-type]
        clock=lambda: fixed_time,
    )

    assert source.calls == 1
    assert result.frame.height == 1

    # Verify Bronze JSON write
    expected_stamp = "20261114T180000Z"
    json_key = f"bronze/ibb_traffic_live/{expected_stamp}.json"
    assert json_key in store.json_writes
    assert store.json_writes[json_key] == {"TI": 45, "TI_An": 40, "TI_Av": 50}

    # Verify Silver Parquet write
    parquet_key = f"silver/traffic_live/traffic_{expected_stamp}.parquet"
    assert parquet_key in store.parquet_writes
    written_df = store.parquet_writes[parquet_key]
    assert written_df["traffic_index_av"][0] == 50.0
