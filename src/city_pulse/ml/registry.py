from collections.abc import Mapping
from typing import Any, Final, Protocol

import polars as pl

from city_pulse.ml.evaluate import Metrics
from city_pulse.ml.protocols import Regressor

MODEL_ARTIFACT_PATH: Final = "model"
CHAMPION_TAG: Final = "champion_run_id"
PRIMARY_METRIC: Final = "mae"


class MlflowRegistryError(RuntimeError):
    """MLflow kaydı beklenen durumda değil (örn. experiment bulunamadı)."""


class ModelRegistry(Protocol):
    def log_run(
        self,
        *,
        run_name: str,
        params: Mapping[str, object],
        metrics: Metrics,
        model: Regressor,
    ) -> str:
        """Eğitimi kaydeder ve `run_id` döndürür."""
        ...

    def promote_if_better(self, run_id: str, metrics: Metrics) -> bool:
        """Champion/challenger: yeni model mevcut champion'dan iyiyse yükseltir."""
        ...


def _make_pyfunc_model(model: Regressor) -> Any:
    """`Regressor`'ı MLflow'un pyfunc model yapısına uyarlar."""
    import mlflow.pyfunc  # type: ignore[import-untyped]  # MLflow'un resmi tip belirteçleri yok

    class _PolarsRegressorModel(mlflow.pyfunc.PythonModel):
        def __init__(self, regressor: Regressor) -> None:
            self._model = regressor

        def predict(self, context: Any, model_input: Any, params: Any = None) -> Any:
            """pandas girdiyi polars'a çevirip tahmini numpy olarak döndürür."""
            return self._model.predict(pl.from_pandas(model_input)).to_numpy()

    return _PolarsRegressorModel(model)


class MlflowRegistry:
    """`ModelRegistry`'nin MLflow gerçeklemesi.

    `tracking_uri` ayarlardan (pydantic-settings) dışarıdan verilir; burada sabit yazılmaz.
    """

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name

    def log_run(
        self,
        *,
        run_name: str,
        params: Mapping[str, object],
        metrics: Metrics,
        model: Regressor,
    ) -> str:
        """Parametreleri, metrikleri ve modeli yeni bir MLflow run'ına yazar."""
        try:
            import mlflow  # type: ignore[import-untyped]  # MLflow'un resmi tip belirteçleri yok
            import mlflow.pyfunc  # type: ignore[import-untyped]
        except ImportError as e:
            raise MlflowRegistryError(
                f"MLflow kütüphanesi yüklenemedi: {e}. Lütfen MLflow sürümünü güncelleyin."
            ) from e

        mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(self._experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(dict(params))
            mlflow.log_metrics(metrics.as_dict())
            mlflow.pyfunc.log_model(
                artifact_path=MODEL_ARTIFACT_PATH,
                python_model=_make_pyfunc_model(model),
            )
            return run.info.run_id

    def promote_if_better(self, run_id: str, metrics: Metrics) -> bool:
        """Yeni run, mevcut champion'dan kesin olarak daha düşük MAE'ye sahipse champion yapar."""
        try:
            from mlflow.tracking import MlflowClient  # type: ignore[import-untyped]
        except ImportError as e:
            raise MlflowRegistryError(f"MLflow client yüklenemedi: {e}") from e

        client = MlflowClient(tracking_uri=self._tracking_uri)
        experiment = client.get_experiment_by_name(self._experiment_name)
        if experiment is None:
            raise MlflowRegistryError(
                f"Experiment bulunamadı: {self._experiment_name!r}; önce `log_run` çağrılmalı."
            )
        champion_id = experiment.tags.get(CHAMPION_TAG)
        if champion_id is not None:
            champion_value = client.get_run(champion_id).data.metrics[PRIMARY_METRIC]
            if metrics.mae >= champion_value:
                return False
        client.set_experiment_tag(experiment.experiment_id, CHAMPION_TAG, run_id)
        return True
