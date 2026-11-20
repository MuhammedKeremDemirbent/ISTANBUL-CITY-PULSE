from typing import Final, Protocol

import polars as pl

from city_pulse.features.columns import BASELINE_KEYS, FEATURE_COLUMNS, TARGET_COLUMN

INPUT_COLUMNS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys((*BASELINE_KEYS, *FEATURE_COLUMNS))
)


class Regressor(Protocol):
    """Regressor interface implemented by baseline and ML models."""

    def fit(self, features: pl.DataFrame, target: pl.Series) -> None: ...

    def predict(self, features: pl.DataFrame) -> pl.Series: ...


def split_inputs_target(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.Series]:
    """Split training frame into input features and target series."""
    return frame.select(INPUT_COLUMNS), frame.get_column(TARGET_COLUMN)
