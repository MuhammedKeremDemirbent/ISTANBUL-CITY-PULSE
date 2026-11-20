import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl

from city_pulse.features.builder import HEAVY_RAIN_THRESHOLD_MM, build_training_frame
from city_pulse.features.columns import FEATURE_COLUMNS, TIMESTAMP_COLUMN
from city_pulse.ml.lightgbm import LightGbmRegressor
from city_pulse.ml.protocols import Regressor
from city_pulse.ml.split import SplitConfig, time_based_split
from city_pulse.ml.train import train_model

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[3] / "data" / "gold" / "predictions"
)


def get_latest_forecast_file(weather_live_dir: Path) -> Path:
    """Return latest weather forecast Parquet file from Silver layer."""
    forecast_files = list(weather_live_dir.glob("**/forecast_*.parquet"))
    if not forecast_files:
        raise FileNotFoundError(f"No forecast files found in {weather_live_dir}.")
    return max(forecast_files)


def prepare_forecast_feature_matrix(
    weather_forecast: pl.DataFrame,
    *,
    holidays: pl.DataFrame | None = None,
    school_calendar: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Generate external feature matrix from weather forecast and calendar."""
    time_col = "timestamp" if "timestamp" in weather_forecast.columns else "hour_utc"

    matrix = weather_forecast.select(
        pl.col(time_col).alias(TIMESTAMP_COLUMN),
        pl.col("temperature_2m").cast(pl.Float64),
        pl.col("precipitation").cast(pl.Float64),
        pl.col("wind_speed_10m").cast(pl.Float64),
        pl.col("relative_humidity_2m").cast(pl.Float64),
    ).sort(TIMESTAMP_COLUMN)

    matrix = matrix.with_columns(
        pl.col(TIMESTAMP_COLUMN).dt.hour().cast(pl.Int32).alias("hour"),
        pl.col(TIMESTAMP_COLUMN).dt.weekday().cast(pl.Int32).alias("day_of_week"),
        pl.col(TIMESTAMP_COLUMN).dt.month().cast(pl.Int32).alias("month"),
        pl.when(pl.col(TIMESTAMP_COLUMN).dt.weekday().is_in([6, 7]))
        .then(1)
        .otherwise(0)
        .cast(pl.Int32)
        .alias("is_weekend"),
        (pl.col("precipitation") > 0.0).cast(pl.Int32).alias("is_raining"),
        (pl.col("precipitation") >= HEAVY_RAIN_THRESHOLD_MM)
        .cast(pl.Int32)
        .alias("heavy_rain"),
    )

    if holidays is not None and holidays.height > 0:
        holiday_dates = holidays.select(pl.col("date").cast(pl.Date))["date"].to_list()
        matrix = matrix.with_columns(
            pl.col(TIMESTAMP_COLUMN)
            .dt.date()
            .is_in(holiday_dates)
            .cast(pl.Int32)
            .alias("is_holiday")
        )
    else:
        matrix = matrix.with_columns(pl.lit(0).cast(pl.Int32).alias("is_holiday"))

    school_cols = (
        "is_school_term",
        "is_school_morning_rush",
        "is_school_afternoon_rush",
    )
    if school_calendar is not None and school_calendar.height > 0:
        cal = school_calendar.select(TIMESTAMP_COLUMN, *school_cols)
        matrix = matrix.join(cal, on=TIMESTAMP_COLUMN, how="left").with_columns(
            [pl.col(c).fill_null(0).cast(pl.Int32) for c in school_cols]
        )
    else:
        matrix = matrix.with_columns(
            [pl.lit(0).cast(pl.Int32).alias(c) for c in school_cols]
        )

    return matrix


def predict_recursive(
    model: Regressor,
    feature_matrix: pl.DataFrame,
    *,
    initial_lag_1h: float,
    initial_lag_24h: float,
) -> pl.DataFrame:
    """Recursive forecasting: Feeds previous step predictions as autoregressive lags."""
    n_steps = feature_matrix.height
    rows = feature_matrix.to_dicts()
    predictions: list[float] = []

    for i in range(n_steps):
        row = rows[i]

        current_lag_1h = initial_lag_1h if i == 0 else predictions[i - 1]
        current_lag_24h = initial_lag_24h if i < 24 else predictions[i - 24]

        row["traffic_index_av_lag_1h"] = float(current_lag_1h)
        row["traffic_index_av_lag_24h"] = float(current_lag_24h)

        step_df = pl.DataFrame([row]).select(FEATURE_COLUMNS)
        pred_val = float(model.predict(step_df)[0])

        bounded_pred = max(0.0, min(100.0, pred_val))
        predictions.append(round(bounded_pred, 2))

    return feature_matrix.with_columns(
        pl.Series("predicted_traffic_index_av", predictions, dtype=pl.Float64)
    )


def train_production_model(
    gold: pl.DataFrame,
    *,
    holidays: pl.DataFrame | None = None,
    school_calendar: pl.DataFrame | None = None,
) -> Regressor:
    """Train production model using all available historical data."""
    frame = build_training_frame(gold, holidays, school_calendar)
    split_config = SplitConfig(
        train_end=datetime(2025, 1, 1), valid_end=datetime(2025, 7, 1)
    )
    split = time_based_split(frame, split_config)

    log.info("Training production model for live prediction...")
    model = LightGbmRegressor()
    train_model(model, split)
    return model


def save_predictions_atomically(df: pl.DataFrame, output_path: Path) -> None:
    """Save prediction results atomically to Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.parquet")
    df.write_parquet(tmp_path)
    tmp_path.rename(output_path)
    log.info(
        "Saved live traffic predictions: %s (%d hours)",
        output_path,
        df.height,
    )


def run_predict_live(
    *,
    forecast_file: Path | None = None,
    output_path: Path | None = None,
) -> pl.DataFrame:
    """Run end-to-end live prediction pipeline."""
    root_dir = Path(__file__).resolve().parents[3]
    gold_path = (
        root_dir / "data" / "gold" / "city_pulse_hourly_2023-01-01_2025-12-31.parquet"
    )
    holidays_path = (
        root_dir / "data" / "silver" / "holidays" / "turkey_holidays_2023_2025.parquet"
    )
    school_path = (
        root_dir / "data" / "silver" / "school" / "school_calendar_2023_2025.parquet"
    )
    live_weather_dir = root_dir / "data" / "silver" / "weather_live"

    target_forecast = forecast_file or get_latest_forecast_file(live_weather_dir)
    log.info("Reading weather forecast: %s", target_forecast.name)
    weather_df = pl.read_parquet(target_forecast)

    gold = pl.read_parquet(gold_path)
    holidays = pl.read_parquet(holidays_path) if holidays_path.exists() else None
    school = pl.read_parquet(school_path) if school_path.exists() else None

    last_known_traffic = float(gold.sort("timestamp")["traffic_index_av"][-1])
    log.info("Initial reference traffic index: %.2f", last_known_traffic)

    model = train_production_model(gold, holidays=holidays, school_calendar=school)

    feature_matrix = prepare_forecast_feature_matrix(
        weather_df, holidays=holidays, school_calendar=school
    )

    predictions_df = predict_recursive(
        model,
        feature_matrix,
        initial_lag_1h=last_known_traffic,
        initial_lag_24h=last_known_traffic,
    )

    if output_path is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_file = DEFAULT_OUTPUT_DIR / f"live_traffic_forecast_{stamp}.parquet"
    else:
        out_file = output_path

    save_predictions_atomically(predictions_df, out_file)
    return predictions_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Traffic Prediction Pipeline")
    parser.add_argument(
        "--forecast-file",
        type=Path,
        default=None,
        help="Custom weather forecast Parquet file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    predictions = run_predict_live(
        forecast_file=args.forecast_file,
        output_path=args.output,
    )

    summary_cols = [
        TIMESTAMP_COLUMN,
        "temperature_2m",
        "precipitation",
        "is_school_morning_rush",
        "predicted_traffic_index_av",
    ]
    print("\n" + "=" * 70)
    print("LIVE TRAFFIC PREDICTION (FIRST 5 HOURS):")
    print("=" * 70)
    print(predictions.select(summary_cols).head(5))

    print("\n" + "=" * 70)
    print("TOP 5 CONGESTED HOURS NEXT WEEK (PEAK TRAFFIC):")
    print("=" * 70)
    print(
        predictions.select(summary_cols)
        .sort("predicted_traffic_index_av", descending=True)
        .head(5)
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
