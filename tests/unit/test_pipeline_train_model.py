from collections.abc import Mapping
from datetime import datetime

import polars as pl
import pytest

from city_pulse.features.columns import TIMESTAMP_COLUMN
from city_pulse.ml.baseline import HourDayBaseline
from city_pulse.ml.evaluate import Metrics
from city_pulse.ml.protocols import Regressor
from city_pulse.ml.split import SplitConfig
from city_pulse.pipelines.train_model import run_training

CONFIG = SplitConfig(train_end=datetime(2025, 1, 6), valid_end=datetime(2025, 1, 8))


class _FakeRegistry:
    """MLflow'a dokunmadan çağrıları kaydeden sahte kayıt defteri."""

    def __init__(self, *, promote: bool) -> None:
        self._promote = promote
        self.logged_metrics: list[Metrics] = []

    def log_run(
        self,
        *,
        run_name: str,
        params: Mapping[str, object],
        metrics: Metrics,
        model: Regressor,
    ) -> str:
        self.logged_metrics.append(metrics)
        return "run-1"

    def promote_if_better(self, run_id: str, metrics: Metrics) -> bool:
        return self._promote


def _main_table() -> pl.DataFrame:
    """2025-01-01 ile 2025-01-10 arası saatlik; hedef yalnızca saate bağlı."""
    timestamps = pl.datetime_range(
        datetime(2025, 1, 1), datetime(2025, 1, 10, 23), interval="1h", eager=True
    )
    return (
        pl.DataFrame({TIMESTAMP_COLUMN: timestamps})
        .with_columns(
            pl.col(TIMESTAMP_COLUMN).dt.weekday().alias("day_of_week"),
            pl.col(TIMESTAMP_COLUMN).dt.hour().alias("hour"),
            pl.col(TIMESTAMP_COLUMN).dt.month().alias("month"),
            pl.when(pl.col(TIMESTAMP_COLUMN).dt.weekday().is_in([6, 7]))
            .then(1)
            .otherwise(0)
            .alias("is_weekend"),
            pl.lit(15.0).alias("temperature_2m"),
            pl.lit(0.0).alias("precipitation"),
            pl.lit(10.0).alias("wind_speed_10m"),
            pl.lit(60.0).alias("relative_humidity_2m"),
        )
        .with_columns((pl.col("hour") * 1.0).alias("traffic_index_av"))
    )


@pytest.mark.parametrize("promote", [True, False])
def test_run_training_logs_once_and_returns_registry_decision(promote: bool) -> None:
    registry = _FakeRegistry(promote=promote)

    result = run_training(
        _main_table(),
        CONFIG,
        model=HourDayBaseline(),
        baseline=HourDayBaseline(),
        registry=registry,
    )

    assert result.run_id == "run-1"
    assert result.promoted is promote
    assert registry.logged_metrics == [result.model_metrics]
    assert result.model_metrics.mae == pytest.approx(0.0)
