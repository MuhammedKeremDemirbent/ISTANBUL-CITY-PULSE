"""Baz model: haftanın günü x saat ortalaması. Her gerçek model bunu geçmek zorundadır."""

from typing import Final

import polars as pl

from city_pulse.features.columns import BASELINE_KEYS

_ROW_INDEX: Final = "_row"
_PREDICTION: Final = "prediction"
_TARGET: Final = "_target"


class BaselineNotFittedError(RuntimeError):
    """`predict` çağrılmadan önce `fit` çağrılmadı."""


class HourDayBaseline:
    """`Regressor` arayüzünü karşılar; `BASELINE_KEYS` kolonlarına göre hedef ortalamasını tahmin eder."""

    def __init__(self) -> None:
        self._group_means: pl.DataFrame | None = None
        self._global_mean: float | None = None

    def fit(self, features: pl.DataFrame, target: pl.Series) -> None:
        """Her (gün, saat) grubunun hedef ortalamasını ve genel ortalamayı öğrenir."""
        if target.is_empty():
            raise ValueError("Hedef serisi boş; baz model eğitilemez.")
        global_mean = target.mean()
        if not isinstance(global_mean, int | float):
            raise TypeError(
                f"Hedef ortalaması sayısal değil: {type(global_mean).__name__}"
            )
        self._global_mean = float(global_mean)
        self._group_means = (
            features.select(BASELINE_KEYS)
            .with_columns(target.alias(_TARGET))
            .group_by(BASELINE_KEYS)
            .agg(pl.col(_TARGET).mean().alias(_PREDICTION))
        )

    def predict(self, features: pl.DataFrame) -> pl.Series:
        """Girdi sırasını koruyarak tahmin üretir.

        Eğitimde görülmeyen gruplar için genel ortalama kullanılır.
        """
        if self._group_means is None or self._global_mean is None:
            raise BaselineNotFittedError("Önce `fit` çağrılmalı.")
        joined = (
            features.select(BASELINE_KEYS)
            .with_row_index(_ROW_INDEX)
            .join(self._group_means, on=list(BASELINE_KEYS), how="left")
            .sort(_ROW_INDEX)
        )
        return joined.get_column(_PREDICTION).fill_null(self._global_mean)
