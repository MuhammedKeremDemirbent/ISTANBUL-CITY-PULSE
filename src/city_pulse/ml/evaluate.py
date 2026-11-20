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
    """Calculate MAE and RMSE metrics for a model on the given frame."""
    if frame.is_empty():
        raise ValueError("Cannot evaluate on an empty frame.")
    features, target = split_inputs_target(frame)
    predictions = model.predict(features)
    if len(predictions) != len(target):
        raise ValueError(
            f"Prediction count ({len(predictions)}) does not match row count ({len(target)})."
        )
    errors = predictions - target
    return Metrics(
        mae=_mean(errors.abs()),
        rmse=math.sqrt(_mean(errors.pow(2))),
    )


def _mean(series: pl.Series) -> float:
    if series.is_empty() or series.null_count():
        raise ValueError("Cannot compute mean: series is empty or contains nulls.")
    value = series.mean()
    if not isinstance(value, int | float):
        raise TypeError(f"Mean is not numeric: {type(value).__name__}")
    return float(value)
