from typing import Final

import polars as pl

from city_pulse.features.columns import BASELINE_KEYS

_ROW_INDEX: Final = "_row"
_PREDICTION: Final = "prediction"
_TARGET: Final = "_target"


class BaselineNotFittedError(RuntimeError):
    """The baseline model must be fitted before predict is called."""


class HourDayBaseline:
    """Predicts target group mean based on BASELINE_KEYS (day_of_week, hour)."""

    def __init__(self) -> None:
        self._group_means: pl.DataFrame | None = None
        self._global_mean: float | None = None

    def fit(self, features: pl.DataFrame, target: pl.Series) -> None:
        """Learn group means and global mean from training data."""
        if target.is_empty():
            raise ValueError("Target series is empty; cannot fit baseline.")
        global_mean = target.mean()
        if not isinstance(global_mean, int | float):
            raise TypeError(f"Target mean is not numeric: {type(global_mean).__name__}")
        self._global_mean = float(global_mean)
        self._group_means = (
            features.select(BASELINE_KEYS)
            .with_columns(target.alias(_TARGET))
            .group_by(BASELINE_KEYS)
            .agg(pl.col(_TARGET).mean().alias(_PREDICTION))
        )

    def predict(self, features: pl.DataFrame) -> pl.Series:
        """Generate predictions preserving row order.

        Falls back to global mean for unseen groups.
        """
        if self._group_means is None or self._global_mean is None:
            raise BaselineNotFittedError("Must call `fit` before `predict`.")
        joined = (
            features.select(BASELINE_KEYS)
            .with_row_index(_ROW_INDEX)
            .join(self._group_means, on=list(BASELINE_KEYS), how="left")
            .sort(_ROW_INDEX)
        )
        return joined.get_column(_PREDICTION).fill_null(self._global_mean)
