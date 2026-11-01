# tests/unit/test_ml_evaluate.py
import math

import polars as pl
import pytest

from city_pulse.ml.evaluate import evaluate


class _FixedModel:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def fit(self, features: pl.DataFrame, target: pl.Series) -> None:
        return None

    def predict(self, features: pl.DataFrame) -> pl.Series:
        return pl.Series(self._values)


def _frame(target: list[float]) -> pl.DataFrame:
    n = len(target)
    return pl.DataFrame(
        {
            "day_of_week": [1] * n,
            "hour": [8] * n,
            "month": [1] * n,
            "is_weekend": [0] * n,
            "temperature_2m": [15.0] * n,
            "precipitation": [0.0] * n,
            "wind_speed_10m": [10.0] * n,
            "relative_humidity_2m": [60.0] * n,
            "traffic_index_av": target,
        }
    )


def test_mae_and_rmse() -> None:
    metrics = evaluate(_FixedModel([12.0, 16.0]), _frame([10.0, 20.0]))

    assert metrics.mae == pytest.approx(3.0)
    assert metrics.rmse == pytest.approx(math.sqrt(10.0))


def test_empty_frame_raises() -> None:
    with pytest.raises(ValueError):
        evaluate(_FixedModel([]), _frame([]))


def test_prediction_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        evaluate(_FixedModel([1.0]), _frame([1.0, 2.0]))
