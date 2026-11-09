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
    """Silver katmanındaki en güncel hava tahmini Parquet dosyasını döner."""
    forecast_files = list(weather_live_dir.glob("**/forecast_*.parquet"))
    if not forecast_files:
        raise FileNotFoundError(
            f"Hava tahmin dosyası bulunamadı: {weather_live_dir}. "
            "Önce 'weather_live' pipeline'ını çalıştırın."
        )
    return max(forecast_files)


def prepare_forecast_feature_matrix(
    weather_forecast: pl.DataFrame,
    *,
    holidays: pl.DataFrame | None = None,
    school_calendar: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Hava tahmininden ve takvimden modelin beklediği dışsal öznitelikleri üretir.

    (Lag öznitelikleri hariç tüm dışsal sinyalleri hazırlar).
    """
    time_col = "timestamp" if "timestamp" in weather_forecast.columns else "hour_utc"

    # 1. Temel zaman ve hava kolonlarını standartlaştır
    matrix = weather_forecast.select(
        pl.col(time_col).alias(TIMESTAMP_COLUMN),
        pl.col("temperature_2m").cast(pl.Float64),
        pl.col("precipitation").cast(pl.Float64),
        pl.col("wind_speed_10m").cast(pl.Float64),
        pl.col("relative_humidity_2m").cast(pl.Float64),
    ).sort(TIMESTAMP_COLUMN)

    # 2. Takvim değişkenleri
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

    # 3. Tatil eşleştirmesi
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

    # 4. Okul takvimi eşleştirmesi
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
    """Recursive Forecasting: Her saatin tahminini sonraki adımlara lag olarak besler.

    Model gelecek 168 saati adım adım tahmin eder; t anındaki tahmin,
    t+1 anında lag_1h ve t+24 anında lag_24h olarak kullanılır.
    """
    n_steps = feature_matrix.height

    # Model girdilerini hazırlamak için sözlük listesi
    rows = feature_matrix.to_dicts()

    predictions: list[float] = []

    for i in range(n_steps):
        row = rows[i]

        # 1. lag_1h belirle
        if i == 0:
            current_lag_1h = initial_lag_1h
        else:
            current_lag_1h = predictions[i - 1]

        # 2. lag_24h belirle
        if i < 24:
            current_lag_24h = initial_lag_24h
        else:
            current_lag_24h = predictions[i - 24]

        # Satıra lag değerlerini ekle
        row["traffic_index_av_lag_1h"] = float(current_lag_1h)
        row["traffic_index_av_lag_24h"] = float(current_lag_24h)

        # Tek satırlık tahmin için dataframe kur
        step_df = pl.DataFrame([row]).select(FEATURE_COLUMNS)
        pred_val = float(model.predict(step_df)[0])

        # Trafik indeksi 0 ile 100 arasında sınırlandırılır
        bounded_pred = max(0.0, min(100.0, pred_val))
        predictions.append(round(bounded_pred, 2))

    # Sonuç DataFrame'i oluştur
    result = feature_matrix.with_columns(
        pl.Series("predicted_traffic_index_av", predictions, dtype=pl.Float64)
    )
    return result


def train_production_model(
    gold: pl.DataFrame,
    *,
    holidays: pl.DataFrame | None = None,
    school_calendar: pl.DataFrame | None = None,
) -> Regressor:
    """Tüm geçmiş veriyi kullanarak canlı tahmin modelini eğitir."""
    frame = build_training_frame(gold, holidays, school_calendar)
    split_config = SplitConfig(
        train_end=datetime(2025, 1, 1), valid_end=datetime(2025, 7, 1)
    )
    split = time_based_split(frame, split_config)

    log.info("Canlı tahmin için üretim modeli eğitiliyor...")
    model = LightGbmRegressor()
    train_model(model, split)
    return model


def save_predictions_atomically(df: pl.DataFrame, output_path: Path) -> None:
    """Tahmin sonuçlarını atomik yazma (tmp -> rename) ile Parquet olarak kaydeder."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.parquet")
    df.write_parquet(tmp_path)
    tmp_path.rename(output_path)
    log.info(
        "Canlı trafik tahminleri başarıyla kaydedildi: %s (%d saat)",
        output_path,
        df.height,
    )


def run_predict_live(
    *,
    forecast_file: Path | None = None,
    output_path: Path | None = None,
) -> pl.DataFrame:
    """Canlı tahmin hattını uçtan uca çalıştırır. Airflow'dan doğrudan çağrılabilir."""
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

    # 1. En güncel hava tahmin dosyasını seç
    target_forecast = forecast_file or get_latest_forecast_file(live_weather_dir)
    log.info("Hava tahmini okunuyor: %s", target_forecast.name)
    weather_df = pl.read_parquet(target_forecast)

    # 2. Yardımcı tabloları oku
    gold = pl.read_parquet(gold_path)
    holidays = pl.read_parquet(holidays_path) if holidays_path.exists() else None
    school = pl.read_parquet(school_path) if school_path.exists() else None

    # 3. Modelin başlangıç referans lag'lerini Gold'un son değerlerinden al
    last_known_traffic = float(gold.sort("timestamp")["traffic_index_av"][-1])
    log.info(
        "Başlangıç referans trafik indeksi: %.2f",
        last_known_traffic,
    )

    # 4. Modeli eğit
    model = train_production_model(gold, holidays=holidays, school_calendar=school)

    # 5. Gelecek 168 saatin öznitelik matrisini hazırla
    feature_matrix = prepare_forecast_feature_matrix(
        weather_df, holidays=holidays, school_calendar=school
    )

    # 6. Recursive olarak tahmin et
    predictions_df = predict_recursive(
        model,
        feature_matrix,
        initial_lag_1h=last_known_traffic,
        initial_lag_24h=last_known_traffic,
    )

    # 7. Çıktı dosyasını belirle ve atomik kaydet
    if output_path is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_file = DEFAULT_OUTPUT_DIR / f"live_traffic_forecast_{stamp}.parquet"
    else:
        out_file = output_path

    save_predictions_atomically(predictions_df, out_file)
    return predictions_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canlı Trafik Tahmin Hattı")
    parser.add_argument(
        "--forecast-file",
        type=Path,
        default=None,
        help="Özel hava tahmin Parquet dosyası (Boşsa en günceli seçilir)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Çıktı Parquet dosya yolu",
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

    # Örnek ilk 5 saati ve en yoğun 3 saati konsola bas
    summary_cols = [
        TIMESTAMP_COLUMN,
        "temperature_2m",
        "precipitation",
        "is_school_morning_rush",
        "predicted_traffic_index_av",
    ]
    print("\n" + "=" * 70)
    print("CANLI TRAFİK TAHMİNİ (İLK 5 SAAT):")
    print("=" * 70)
    print(predictions.select(summary_cols).head(5))

    print("\n" + "=" * 70)
    print("GELECEK HAFTANIN EN YOĞUN BEKLENEN 5 SAATİ (PİK TRAFİK):")
    print("=" * 70)
    print(
        predictions.select(summary_cols)
        .sort("predicted_traffic_index_av", descending=True)
        .head(5)
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
