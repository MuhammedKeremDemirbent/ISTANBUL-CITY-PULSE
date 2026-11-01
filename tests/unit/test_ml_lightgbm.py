import polars as pl
import pytest

from city_pulse.ml.lightgbm import LightGbmRegressor


def _sample_data() -> tuple[pl.DataFrame, pl.Series]:
    features = pl.DataFrame(
        {
            "hour": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            "day_of_week": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        }
    )
    target = pl.Series(
        "target", [20.0, 25.0, 30.0, 35.0, 40.0, 22.0, 27.0, 32.0, 37.0, 42.0]
    )
    return features, target


def test_lightgbm_fit_and_predict() -> None:
    features, target = _sample_data()
    model = LightGbmRegressor(params={"n_estimators": 20, "learning_rate": 0.1})

    model.fit(features, target)
    preds = model.predict(features)

    assert len(preds) == len(target)
    assert preds.dtype == pl.Float64


def test_predict_before_fit_raises() -> None:
    model = LightGbmRegressor()
    features, _ = _sample_data()

    with pytest.raises(RuntimeError, match="Model eğitilmeden"):
        model.predict(features)


def test_fit_empty_raises() -> None:
    model = LightGbmRegressor()
    with pytest.raises(ValueError, match="boş olamaz"):
        model.fit(pl.DataFrame(), pl.Series([]))
