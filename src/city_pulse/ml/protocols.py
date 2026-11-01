"""Modelleme katmanının ortak sözleşmesi: somut kütüphaneye değil arayüze bağımlılık."""

from typing import Final, Protocol

import polars as pl

from city_pulse.features.columns import BASELINE_KEYS, FEATURE_COLUMNS, TARGET_COLUMN

INPUT_COLUMNS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys((*BASELINE_KEYS, *FEATURE_COLUMNS))
)


class Regressor(Protocol):
    """Hem baz model hem LightGBM gibi gerçek modeller bu arayüzü karşılar."""

    def fit(self, features: pl.DataFrame, target: pl.Series) -> None: ...

    def predict(self, features: pl.DataFrame) -> pl.Series: ...


def split_inputs_target(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.Series]:
    """Eğitim çerçevesini girdi kolonları ve hedef seri olarak ayırır."""
    return frame.select(INPUT_COLUMNS), frame.get_column(TARGET_COLUMN)
