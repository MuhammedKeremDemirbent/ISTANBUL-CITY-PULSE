from collections.abc import Mapping
from typing import Any, Final, Protocol

import polars as pl

from city_pulse.ml.evaluate import Metrics
from city_pulse.ml.protocols import Regressor

MODEL_ARTIFACT_PATH: Final = "model"
CHAMPION_TAG: Final = "champion_run_id"
PRIMARY_METRIC: Final = "mae"


class MlflowRegistryError(RuntimeError):
    """MLflow registry state error (e.g. experiment not found)."""


class ModelRegistry(Protocol):
    def log_run(
        self,
        *,
        run_name: str,
        params: Mapping[str, object],
        metrics: Metrics,
        model: Regressor,
    ) -> str:
        """Log training run and return run_id."""
        ...

    def promote_if_better(self, run_id: str, metrics: Metrics) -> bool:
        """Promote model to champion if its metric improves upon the current champion."""
        ...


def _make_pyfunc_model(model: Regressor) -> Any:
    """Adapt Regressor to MLflow PythonModel format."""
    from mlflow.pyfunc import PythonModel  # type: ignore[import-untyped]

    class _PolarsRegressorModel(PythonModel):  # type: ignore[misc]
        def __init__(self, regressor: Regressor) -> None:
            self._model = regressor

        def predict(self, context: Any, model_input: Any, params: Any = None) -> Any:
            """Convert pandas input to polars and return prediction as numpy array."""
            return self._model.predict(pl.from_pandas(model_input)).to_numpy()

    return _PolarsRegressorModel(model)


class MlflowRegistry:
    """MLflow implementation of ModelRegistry."""

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
        """Log parameters, metrics, and pyfunc model to MLflow run."""
        try:
            import mlflow  # type: ignore[import-untyped]
            import mlflow.pyfunc  # type: ignore[import-untyped]
        except ImportError as e:
            raise MlflowRegistryError(
                f"Failed to import MLflow: {e}. Please ensure mlflow is installed."
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
        """Promote to champion if new run has lower MAE than the current champion."""
        try:
            from mlflow.tracking import MlflowClient  # type: ignore[import-untyped]
        except ImportError as e:
            raise MlflowRegistryError(f"Failed to load MLflow client: {e}") from e

        client = MlflowClient(tracking_uri=self._tracking_uri)
        experiment = client.get_experiment_by_name(self._experiment_name)
        if experiment is None:
            raise MlflowRegistryError(
                f"Experiment not found: {self._experiment_name!r}; call log_run first."
            )

        champion_id = experiment.tags.get(CHAMPION_TAG)
        if champion_id is not None:
            champion_value = client.get_run(champion_id).data.metrics[PRIMARY_METRIC]
            if metrics.mae >= champion_value:
                return False
        client.set_experiment_tag(experiment.experiment_id, CHAMPION_TAG, run_id)
        return True
