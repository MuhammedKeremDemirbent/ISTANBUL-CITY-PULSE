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
            raise ValueError("train_end must be earlier than valid_end")


class Split(NamedTuple):
    train: pl.DataFrame
    valid: pl.DataFrame
    test: pl.DataFrame


def time_based_split(frame: pl.DataFrame, config: SplitConfig) -> Split:
    """Split frame into three disjoint time-ordered partitions based on TIMESTAMP_COLUMN.

    Prevents future data leakage by strictly partitioning chronologically.
    """
    if TIMESTAMP_COLUMN not in frame.columns:
        raise ValueError(f"Column '{TIMESTAMP_COLUMN}' not found")
    if frame[TIMESTAMP_COLUMN].null_count():
        raise ValueError(f"Null values present in '{TIMESTAMP_COLUMN}'")

    ts = pl.col(TIMESTAMP_COLUMN)
    split = Split(
        train=frame.filter(ts < config.train_end),
        valid=frame.filter((ts >= config.train_end) & (ts < config.valid_end)),
        test=frame.filter(ts >= config.valid_end),
    )
    for name, part in split._asdict().items():
        if part.is_empty():
            raise ValueError(
                f"Split partition '{name}' is empty; adjust split boundaries."
            )

    log.info(
        "Time-based split: train=%d valid=%d test=%d",
        split.train.height,
        split.valid.height,
        split.test.height,
    )
    return split
