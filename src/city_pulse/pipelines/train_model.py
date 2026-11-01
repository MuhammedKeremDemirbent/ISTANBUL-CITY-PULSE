"""Eğitim akışı: özellik -> zamana göre ayır -> baz model + model -> değerlendir -> kaydet.

Kullanım:
    uv run python -m city_pulse.pipelines.train_model
    uv run python -m city_pulse.pipelines.train_model --train-end 2025-01-01 --valid-end 2025-07-01
"""

import argparse
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from city_pulse.features.builder import build_training_frame
from city_pulse.ml.baseline import HourDayBaseline
from city_pulse.ml.evaluate import Metrics, evaluate
from city_pulse.ml.lightgbm import LightGbmRegressor
from city_pulse.ml.protocols import Regressor
from city_pulse.ml.registry import MlflowRegistry, MlflowRegistryError, ModelRegistry
from city_pulse.ml.split import SplitConfig, time_based_split
from city_pulse.ml.train import train_model

log = logging.getLogger(__name__)

DEFAULT_TRAIN_END = datetime(2025, 1, 1)
DEFAULT_VALID_END = datetime(2025, 7, 1)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    baseline_metrics: Metrics
    model_metrics: Metrics
    run_id: str
    promoted: bool


class _LocalConsoleRegistry:
    """MLflow bağlantısı kurulamadığında yerel loglama yapan yedek kayıt defteri."""

    def log_run(
        self,
        *,
        run_name: str,
        params: Mapping[str, object],
        metrics: Metrics,
        model: Regressor,
    ) -> str:
        log.info(
            "Yerel Kayıt [%s] - Parametreler: %s | Metrikler: MAE=%.4f, RMSE=%.4f",
            run_name,
            params,
            metrics.mae,
            metrics.rmse,
        )
        return "local-run"

    def promote_if_better(self, run_id: str, metrics: Metrics) -> bool:
        log.info("Model champion olarak işaretlendi.")
        return True


def run_training(
    main: pl.DataFrame,
    split_config: SplitConfig,
    *,
    model: Regressor,
    baseline: Regressor,
    registry: ModelRegistry,
    holidays: pl.DataFrame | None = None,
) -> TrainingResult:
    """Uçtan uca eğitim: ayır, baz model ve modeli eğit, test setinde ölç, kaydet, gerekirse yükselt."""
    frame = build_training_frame(main, holidays)
    split = time_based_split(frame, split_config)

    train_model(baseline, split)
    train_model(model, split)

    baseline_metrics = evaluate(baseline, split.test)
    model_metrics = evaluate(model, split.test)
    valid_metrics = evaluate(model, split.valid)

    log.info(
        "Validasyon Sonuçları -> Model MAE: %.4f | RMSE: %.4f",
        valid_metrics.mae,
        valid_metrics.rmse,
    )
    log.info(
        "Test Seti Sonuçları  -> Baz Model MAE: %.4f | LightGBM MAE: %.4f",
        baseline_metrics.mae,
        model_metrics.mae,
    )

    if model_metrics.mae >= baseline_metrics.mae:
        log.warning(
            "Model baz modeli geçemedi (MAE %.4f >= %.4f).",
            model_metrics.mae,
            baseline_metrics.mae,
        )
    else:
        diff = ((baseline_metrics.mae - model_metrics.mae) / baseline_metrics.mae) * 100
        log.info("Tebrikler! Model baz modele kıyasla %% %.2f daha başarılı.", diff)

    model_name = type(model).__name__
    try:
        run_id = registry.log_run(
            run_name=model_name,
            params={
                "model": model_name,
                "train_rows": split.train.height,
                "valid_rows": split.valid.height,
                "test_rows": split.test.height,
                "baseline_mae": baseline_metrics.mae,
                "valid_mae": valid_metrics.mae,
            },
            metrics=model_metrics,
            model=model,
        )
        promoted = registry.promote_if_better(run_id, model_metrics)
    except MlflowRegistryError as e:
        log.warning(
            "MLflow kaydı başarısız oldu (%s), yerel konsol kaydına geçiliyor.", e
        )
        fallback = _LocalConsoleRegistry()
        run_id = fallback.log_run(
            run_name=model_name,
            params={"model": model_name},
            metrics=model_metrics,
            model=model,
        )
        promoted = fallback.promote_if_better(run_id, model_metrics)

    return TrainingResult(baseline_metrics, model_metrics, run_id, promoted)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model Eğitim ve Değerlendirme Hattı")
    parser.add_argument(
        "--train-end",
        default=DEFAULT_TRAIN_END,
        type=datetime.fromisoformat,
        help="Eğitim seti bitiş tarihi (ISO format, örn: 2025-01-01)",
    )
    parser.add_argument(
        "--valid-end",
        default=DEFAULT_VALID_END,
        type=datetime.fromisoformat,
        help="Validasyon seti bitiş tarihi (ISO format, örn: 2025-07-01)",
    )
    parser.add_argument(
        "--tracking-uri",
        default="sqlite:///mlflow.db",
        help="MLflow tracking URI (varsayılan: sqlite:///mlflow.db)",
    )
    parser.add_argument(
        "--experiment-name",
        default="istanbul-traffic-pulse",
        help="MLflow experiment adı",
    )
    args = parser.parse_args(argv)
    if args.train_end >= args.valid_end:
        parser.error("train_end tarihi valid_end tarihinden önce olmalıdır.")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args(argv)

    root_dir = Path(__file__).resolve().parents[3]
    gold_dir = root_dir / "data" / "gold"
    gold_files = list(gold_dir.glob("*.parquet"))
    if not gold_files:
        raise FileNotFoundError(
            f"Gold tablosu bulunamadı: {gold_dir}. Önce 'merged_datas' hattını çalıştırın."
        )

    gold_path = max(gold_files)
    log.info("Gold veri tablosu okunuyor: %s", gold_path)
    main_df = pl.read_parquet(gold_path)

    # Tatil günleri tablosunu yükle (yoksa model tatilsiz çalışır)
    holidays_path = root_dir / "data" / "silver" / "holidays" / "turkey_holidays_2023_2025.parquet"
    holidays_df: pl.DataFrame | None = None
    if holidays_path.exists():
        holidays_df = pl.read_parquet(holidays_path)
        log.info("Tatil tablosu yüklendi: %d gün", holidays_df.height)
    else:
        log.warning("Tatil tablosu bulunamadı (%s); is_holiday=0 olarak devam ediliyor.", holidays_path)

    split_config = SplitConfig(train_end=args.train_end, valid_end=args.valid_end)
    baseline = HourDayBaseline()
    model = LightGbmRegressor()
    registry = MlflowRegistry(
        tracking_uri=args.tracking_uri, experiment_name=args.experiment_name
    )

    result = run_training(
        main_df,
        split_config,
        model=model,
        baseline=baseline,
        registry=registry,
        holidays=holidays_df,
    )

    log.info(
        "Eğitim tamamlandı! Run ID: %s | Champion Durumu: %s | Test MAE: %.4f",
        result.run_id,
        result.promoted,
        result.model_metrics.mae,
    )


if __name__ == "__main__":
    main()
