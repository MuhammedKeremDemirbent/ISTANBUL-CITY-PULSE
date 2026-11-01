"""LightGBM regresyon modeli adaptörü. `Regressor` protokolünü karşılar."""

from collections.abc import Callable, Mapping
from typing import Any

import lightgbm as lgb
import polars as pl


class LightGbmRegressor:
    """LightGBM algoritması ile trafik indeksi tahmini."""

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        default_params: dict[str, Any] = {
            "objective": "regression",
            "metric": "mae",
            "random_state": 42,
            "verbose": -1,
            "n_estimators": 100,
            "learning_rate": 0.05,
        }
        if params:
            default_params.update(params)
        self._params = default_params
        self._model = lgb.LGBMRegressor(**self._params)
        self._is_fitted = False

    def fit(self, features: pl.DataFrame, target: pl.Series) -> None:
        """Polars DataFrame ve Series verilerini kullanarak modeli eğitir."""
        if features.is_empty() or target.is_empty():
            raise ValueError("Eğitim verisi boş olamaz.")
        x_train = features.to_pandas()
        y_train = target.to_numpy()
        self._model.fit(x_train, y_train)
        self._is_fitted = True

    def fit_eval(
        self,
        features: pl.DataFrame,
        target: pl.Series,
        valid_features: pl.DataFrame,
        valid_target: pl.Series,
    ) -> None:
        """Eğitim ve validasyon setleriyle erken durdurma (early stopping) kullanarak eğitir."""
        if features.is_empty() or target.is_empty():
            raise ValueError("Eğitim verisi boş olamaz.")
        x_train = features.to_pandas()
        y_train = target.to_numpy()
        x_valid = valid_features.to_pandas()
        y_valid = valid_target.to_numpy()

        callbacks: list[Callable[..., Any]] = [
            lgb.early_stopping(stopping_rounds=15, verbose=False)
        ]
        self._model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            callbacks=callbacks,
        )
        self._is_fitted = True

    def predict(self, features: pl.DataFrame) -> pl.Series:
        """Yeni girdiler üzerinde tahmin üretip Polars Series olarak döndürür."""
        if not self._is_fitted:
            raise RuntimeError("Model eğitilmeden `predict` çağrılamaz.")
        x_pred = features.to_pandas()
        preds = self._model.predict(x_pred)
        return pl.Series("prediction", preds, dtype=pl.Float64)
