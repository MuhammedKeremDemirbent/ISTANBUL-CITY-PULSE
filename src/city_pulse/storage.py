"""Dosya yazma (I/O) tek yerde. Boru hatları `Store` arayüzüne bağımlıdır, diske değil."""

import json
from pathlib import Path
from typing import Protocol

import polars as pl


class Store(Protocol):
    def write_json(self, rel: str, payload: dict) -> Path: ...

    def write_parquet(self, rel: str, df: pl.DataFrame) -> Path: ...


class FileStore:
    """Yerel dosya sistemi. Önce geçici dosyaya yazar, sonra yerine koyar (yarım dosya kalmaz)."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def write_json(self, rel: str, payload: dict) -> Path:
        path = self._prepare(rel)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
        return path

    def write_parquet(self, rel: str, df: pl.DataFrame) -> Path:
        path = self._prepare(rel)
        tmp = path.with_name(path.name + ".tmp")
        df.write_parquet(tmp)
        tmp.replace(path)
        return path

    def _prepare(self, rel: str) -> Path:
        path = self._root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
