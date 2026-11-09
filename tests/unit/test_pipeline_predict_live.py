"""predict_live pipeline'ı için birim testleri (Ağsız ve diskten bağımsız)."""

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from city_pulse.features.columns import TIMESTAMP_COLUMN
from city_pulse.pipelines.predict_live import (
    predict_recursive,
    prepare_forecast_feature_matrix,
    save_predictions_atomically,
)


class _DummyModel:
    def fit(self, features: pl.DataFrame, target: pl.Series) -> None:
        pass

    def predict(self, features: pl.DataFrame) -> pl.Series:
        # lag_1h'nin 1.05 katını tahmin eden basit sahte model
        lags = features["traffic_index_av_lag_1h"]
        return lags * 1.05


def _sample_weather_forecast(n: int = 5) -> pl.DataFrame:
    timestamps = [datetime(2026, 11, 4, h, 0) for h in range(n)]
    return pl.DataFrame(
        {
            TIMESTAMP_COLUMN: timestamps,
            "temperature_2m": [15.0] * n,
            "relative_humidity_2m": [70.0] * n,
            "precipitation": [0.0] * n,
            "wind_speed_10m": [5.0] * n,
        }
    )


def test_prepare_forecast_feature_matrix_shapes_and_types() -> None:
    weather = _sample_weather_forecast(3)
    matrix = prepare_forecast_feature_matrix(weather)

    assert matrix.height == 3
    assert "hour" in matrix.columns
    assert "day_of_week" in matrix.columns
    assert "is_weekend" in matrix.columns
    assert "is_raining" in matrix.columns
    assert "heavy_rain" in matrix.columns
    assert "is_holiday" in matrix.columns
    assert "is_school_term" in matrix.columns


def test_predict_recursive_feeds_previous_step() -> None:
    weather = _sample_weather_forecast(3)
    matrix = prepare_forecast_feature_matrix(weather)
    model = _DummyModel()

    result = predict_recursive(
        model,
        matrix,
        initial_lag_1h=50.0,
        initial_lag_24h=50.0,
    )

    assert result.height == 3
    assert "predicted_traffic_index_av" in result.columns

    preds = result["predicted_traffic_index_av"].to_list()
    # 1. adım: 50 * 1.05 = 52.5
    assert preds[0] == pytest.approx(52.5, rel=1e-2)
    # 2. adım: 52.5 * 1.05 = 55.125
    assert preds[1] == pytest.approx(55.125, rel=1e-2)


def test_save_predictions_atomically(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    target_file = tmp_path / "sub" / "output.parquet"

    save_predictions_atomically(df, target_file)

    assert target_file.exists()
    loaded = pl.read_parquet(target_file)
    assert loaded.height == 3
