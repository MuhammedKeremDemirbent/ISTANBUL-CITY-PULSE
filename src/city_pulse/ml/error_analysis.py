from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from city_pulse.features.builder import build_training_frame
from city_pulse.ml.lightgbm import LightGbmRegressor
from city_pulse.ml.split import SplitConfig, time_based_split


def generate_error_plots(output_dir: Path = Path("reports/figures")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gold verisini oku ve eğitim çerçevesini kur
    gold_path = max(Path("data/gold").glob("*.parquet"))
    main_df = pl.read_parquet(gold_path)
    holidays_path = Path("data/silver/holidays/turkey_holidays_2023_2025.parquet")
    holidays_df = pl.read_parquet(holidays_path) if holidays_path.exists() else None

    frame = build_training_frame(main_df, holidays_df)
    split = time_based_split(
        frame,
        SplitConfig(train_end=datetime(2025, 1, 1), valid_end=datetime(2025, 7, 1)),
    )

    # 2. Modeli eğit ve test setinde tahmin üret
    model = LightGbmRegressor()
    model.fit_eval(
        features=split.train.drop(["timestamp", "traffic_index_av"]),
        target=split.train["traffic_index_av"],
        valid_features=split.valid.drop(["timestamp", "traffic_index_av"]),
        valid_target=split.valid["traffic_index_av"],
    )

    preds = model.predict(split.test.drop(["timestamp", "traffic_index_av"]))
    eval_df = split.test.with_columns(
        preds.alias("pred"),
        (preds - pl.col("traffic_index_av")).abs().alias("abs_error"),
    )

    # 3. Saat Bazlı Ortalama Hata Grafiği
    hourly_mae = (
        eval_df.group_by("hour")
        .agg(pl.col("abs_error").mean())
        .sort("hour")
        .to_pandas()
    )

    plt.figure(figsize=(10, 4))
    plt.bar(hourly_mae["hour"], hourly_mae["abs_error"], color="steelblue")
    plt.title("Saat Bazında Ortalama Mutlak Hata (MAE) - Pik Saatler Analizi")
    plt.xlabel("Günün Saati (0-23)")
    plt.ylabel("Test MAE")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_dir / "mae_by_hour.png", bbox_inches="tight")
    plt.close()

    # 4. Haftanın Günü Bazlı Ortalama Hata Grafiği
    day_labels = {
        1: "Pzt",
        2: "Sal",
        3: "Çar",
        4: "Per",
        5: "Cum",
        6: "Cmt",
        7: "Paz",
    }
    daily_mae = (
        eval_df.group_by("day_of_week")
        .agg(pl.col("abs_error").mean())
        .sort("day_of_week")
        .to_pandas()
    )
    daily_mae["day_name"] = daily_mae["day_of_week"].map(day_labels)

    plt.figure(figsize=(8, 4))
    plt.bar(
        daily_mae["day_name"],
        daily_mae["abs_error"],
        color="coral",
    )
    plt.title("Haftanın Gününe Göre Ortalama Hata (MAE)")
    plt.xlabel("Gün")
    plt.ylabel("Test MAE")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_dir / "mae_by_day.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    generate_error_plots()
