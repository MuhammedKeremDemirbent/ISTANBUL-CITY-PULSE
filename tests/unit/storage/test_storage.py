"""storage.py FileStore atomik yazma davranisi icin birim testler."""

import json
from pathlib import Path

import polars as pl
import pytest

from city_pulse.storage import FileStore


@pytest.mark.unit
def test_filestore_atomic_write_json(tmp_path: Path) -> None:
    store = FileStore(root=tmp_path)
    payload = {"city": "Istanbul", "metric": 42}
    rel_path = "bronze/test_data.json"

    saved_path = store.write_json(rel_path, payload)

    assert saved_path.exists()
    assert not saved_path.with_name(saved_path.name + ".tmp").exists()
    with open(saved_path, encoding="utf-8") as f:
        assert json.load(f) == payload


@pytest.mark.unit
def test_filestore_atomic_write_parquet(tmp_path: Path) -> None:
    store = FileStore(root=tmp_path)
    df = pl.DataFrame({"timestamp": [1, 2], "val": [10.5, 20.5]})
    rel_path = "silver/test_data.parquet"

    saved_path = store.write_parquet(rel_path, df)

    assert saved_path.exists()
    assert not saved_path.with_name(saved_path.name + ".tmp").exists()
    read_df = pl.read_parquet(saved_path)
    assert read_df.shape == (2, 2)
