"""Model değerlendirmesi."""

import math
from dataclasses import dataclass

import polars as pl

from city_pulse.ml.protocols import Regressor, split_inputs_target


@dataclass(frozen=True, slots=True)
class Metrics:
    mae: float
    rmse: float

    def as_dict(self) -> dict[str, float]:
        return {"mae": self.mae, "rmse": self.rmse}


def evaluate(model: Regressor, frame: pl.DataFrame) -> Metrics:
    """Modelin `frame` üzerindeki MAE ve RMSE değerlerini hesaplar."""
    if frame.is_empty():
        raise ValueError("Boş çerçeve üzerinde değerlendirme yapılamaz.")
    features, target = split_inputs_target(frame)
    predictions = model.predict(features)
    if len(predictions) != len(target):
        raise ValueError(
            f"Tahmin sayısı ({len(predictions)}) satır sayısıyla ({len(target)}) uyuşmuyor."
        )
    errors = predictions - target
    return Metrics(
        mae=_mean(errors.abs()),
        rmse=math.sqrt(_mean(errors.pow(2))),
    )


def _mean(series: pl.Series) -> float:
    if series.is_empty() or series.null_count():
        raise ValueError("Ortalama hesaplanamadı: seri boş veya eksik değer içeriyor.")
    value = series.mean()
    if not isinstance(value, int | float):
        raise TypeError(f"Ortalama sayısal değil: {type(value).__name__}")
    return float(value)
