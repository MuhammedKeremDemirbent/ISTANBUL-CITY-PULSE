import argparse
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from city_pulse.features.builder import build_training_frame
from city_pulse.features.columns import TIMESTAMP_COLUMN
from city_pulse.ml.evaluate import evaluate
from city_pulse.ml.lightgbm import LightGbmRegressor
from city_pulse.ml.protocols import split_inputs_target
from city_pulse.ml.split import SplitConfig, time_based_split
from city_pulse.ml.train import train_model

log = logging.getLogger(__name__)


def get_live_forecast_range(weather_live_dir: Path) -> tuple[datetime, datetime]:
    """Retrieve start and end timestamps from the latest weather forecast file."""
    forecast_files = list(weather_live_dir.glob("**/forecast_*.parquet"))
    if not forecast_files:
        raise FileNotFoundError(
            f"No weather forecast files found in {weather_live_dir}."
        )

    latest_file = max(forecast_files)
    df = pl.read_parquet(latest_file)

    time_col = "timestamp" if "timestamp" in df.columns else "hour_utc"
    start_val = df[time_col].min()
    end_val = df[time_col].max()
    if not isinstance(start_val, datetime) or not isinstance(end_val, datetime):
        raise TypeError("Timestamp column in weather file is not datetime type.")

    start_dt: datetime = start_val
    end_dt: datetime = end_val

    log.info(
        "Dynamic date range from '%s': %s -> %s",
        latest_file.name,
        start_dt,
        end_dt,
    )
    return start_dt, end_dt


def plot_comparison(
    timestamps: list[datetime],
    actuals: list[float],
    predictions: list[float],
    mae: float,
    rmse: float,
    output_path: Path,
) -> None:
    """Save actual vs predicted traffic index comparison plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 6))
    plt.plot(
        timestamps,  # type: ignore[arg-type]
        actuals,
        label="Actual Traffic Index (y)",
        color="#1f77b4",
        linewidth=2,
    )
    plt.plot(
        timestamps,  # type: ignore[arg-type]
        predictions,
        label=f"LightGBM Forecast (ŷ) - MAE: {mae:.2f}",
        color="#d62728",
        linestyle="--",
        linewidth=2,
    )

    plt.title(
        f"Weekly Traffic Simulation (Backtest)\nMAE: {mae:.4f} | RMSE: {rmse:.4f}",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Date & Time", fontsize=12)
    plt.ylabel("Traffic Index (0 - 100)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=12, loc="upper right")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    log.info("Comparison plot saved: %s", output_path)


def run_backtest(
    gold: pl.DataFrame,
    start_dt: datetime,
    end_dt: datetime,
    *,
    holidays: pl.DataFrame | None = None,
    school_calendar: pl.DataFrame | None = None,
    output_image_path: Path,
) -> None:
    """Run backtest simulation on chosen date range using trained model."""
    frame = build_training_frame(gold, holidays, school_calendar)

    split_config = SplitConfig(
        train_end=datetime(2025, 1, 1), valid_end=datetime(2025, 7, 1)
    )
    split = time_based_split(frame, split_config)

    log.info("Training model...")
    model = LightGbmRegressor()
    train_model(model, split)

    test_week = frame.filter(
        (pl.col(TIMESTAMP_COLUMN) >= start_dt) & (pl.col(TIMESTAMP_COLUMN) <= end_dt)
    ).sort(TIMESTAMP_COLUMN)

    if test_week.is_empty():
        raise ValueError(f"No Gold data found for date range: {start_dt} -> {end_dt}.")

    log.info(
        "Simulation period filtered: %d hours found.",
        test_week.height,
    )

    inputs, actual_series = split_inputs_target(test_week)
    pred_series = model.predict(inputs)
    metrics = evaluate(model, test_week)

    log.info("==========================================")
    log.info("BACKTEST RESULTS (%s -> %s)", start_dt, end_dt)
    log.info("  Evaluated Hours : %d hours", test_week.height)
    log.info("  Mean Absolute Error (MAE) : %.4f", metrics.mae)
    log.info("  Root Mean Squared Error (RMSE) : %.4f", metrics.rmse)
    log.info("==========================================")

    plot_comparison(
        timestamps=test_week[TIMESTAMP_COLUMN].to_list(),
        actuals=actual_series.to_list(),
        predictions=pred_series.to_list(),
        mae=metrics.mae,
        rmse=metrics.rmse,
        output_path=output_image_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly Backtest Pipeline")
    parser.add_argument(
        "--start",
        type=datetime.fromisoformat,
        default=None,
        help="Start date (ISO format, e.g. 2025-10-06T00:00:00)",
    )
    parser.add_argument(
        "--end",
        type=datetime.fromisoformat,
        default=None,
        help="End date (ISO format, e.g. 2025-10-12T23:00:00)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()

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
    live_dir = root_dir / "data" / "silver" / "weather_live"
    output_img = root_dir / "reports" / "figures" / "backtest_weekly_comparison.png"

    if args.start and args.end:
        start_dt, end_dt = args.start, args.end
    else:
        start_dt, end_dt = get_live_forecast_range(live_dir)

    log.info("Simulation range: %s -> %s", start_dt, end_dt)

    gold = pl.read_parquet(gold_path)
    holidays = pl.read_parquet(holidays_path) if holidays_path.exists() else None
    school = pl.read_parquet(school_path) if school_path.exists() else None

    run_backtest(
        gold,
        start_dt,
        end_dt,
        holidays=holidays,
        school_calendar=school,
        output_image_path=output_img,
    )


if __name__ == "__main__":
    main()
