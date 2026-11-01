import polars as pl

from city_pulse.ml.split import Split
from city_pulse.ml.train import train_model


class _SpyModel:
    def __init__(self) -> None:
        self.seen_rows = 0

    def fit(self, features: pl.DataFrame, target: pl.Series) -> None:
        self.seen_rows = features.height

    def predict(self, features: pl.DataFrame) -> pl.Series:
        return pl.Series([0.0] * features.height)


def _frame(n: int) -> pl.DataFrame:
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
            "traffic_index_av": [1.0] * n,
        }
    )


def test_trains_only_on_train_part() -> None:
    split = Split(train=_frame(5), valid=_frame(3), test=_frame(2))
    model = _SpyModel()

    train_model(model, split)

    assert model.seen_rows == 5
