# tests/unit/test_ml_baseline.py
import polars as pl
import pytest

from city_pulse.ml.baseline import BaselineNotFittedError, HourDayBaseline


def _features(days: list[int], hours: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {"day_of_week": days, "hour": hours},
        schema={"day_of_week": pl.Int32, "hour": pl.Int32},
    )


def test_predicts_group_mean_and_keeps_row_order() -> None:
    model = HourDayBaseline()
    model.fit(_features([1, 1, 2], [8, 8, 9]), pl.Series([10.0, 20.0, 40.0]))

    result = model.predict(_features([2, 1], [9, 8]))

    assert result.to_list() == [40.0, 15.0]


def test_unseen_group_falls_back_to_global_mean() -> None:
    model = HourDayBaseline()
    model.fit(_features([1, 2], [8, 9]), pl.Series([10.0, 30.0]))

    assert model.predict(_features([5], [3])).to_list() == [20.0]


def test_predict_before_fit_raises() -> None:
    with pytest.raises(BaselineNotFittedError):
        HourDayBaseline().predict(_features([1], [1]))
