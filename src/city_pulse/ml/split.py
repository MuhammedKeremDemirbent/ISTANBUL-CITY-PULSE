"""Zamana göre eğitim/doğrulama/test ayrımı. Rastgele ayrım yasaktır (sızıntı)."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import NamedTuple

import polars as pl

from city_pulse.features.columns import TIMESTAMP_COLUMN

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SplitConfig:
    train_end: datetime
    valid_end: datetime

    def __post_init__(self) -> None:
        if self.train_end >= self.valid_end:
            raise ValueError("train_end, valid_end'den önce olmalı")


class Split(NamedTuple):
    train: pl.DataFrame
    valid: pl.DataFrame
    test: pl.DataFrame


def time_based_split(frame: pl.DataFrame, config: SplitConfig) -> Split:
    """`TIMESTAMP_COLUMN`'a göre üç ayrık, zaman sıralı parçaya böler.

    Parçalar çakışmaz ve birleşimleri `frame`'in tamamıdır; gelecek verisi eğitime girmez.
    Satır sırası korunur.

    Raises:
        ValueError: Zaman kolonu yoksa, zaman kolonunda boş değer varsa veya
            herhangi bir parça boş kalırsa.
    """
    if TIMESTAMP_COLUMN not in frame.columns:
        raise ValueError(f"'{TIMESTAMP_COLUMN}' kolonu bulunamadı")
    if frame[TIMESTAMP_COLUMN].null_count():
        raise ValueError(f"'{TIMESTAMP_COLUMN}' kolonunda boş değer var")

    ts = pl.col(TIMESTAMP_COLUMN)
    split = Split(
        train=frame.filter(ts < config.train_end),
        valid=frame.filter((ts >= config.train_end) & (ts < config.valid_end)),
        test=frame.filter(ts >= config.valid_end),
    )
    for name, part in split._asdict().items():
        if part.is_empty():
            raise ValueError(
                f"'{name}' parçası boş kaldı; sınır tarihlerini veri aralığına göre ayarla"
            )

    log.info(
        "zamana göre ayrım: train=%d valid=%d test=%d",
        split.train.height,
        split.valid.height,
        split.test.height,
    )
    return split
