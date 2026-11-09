import io
import sys
from pathlib import Path

import numpy as np
import polars as pl

from city_pulse.features.builder import build_training_frame
from city_pulse.features.columns import FEATURE_COLUMNS, TARGET_COLUMN

# Windows terminalinde Türkçe karakter ve Polars tablo çizgilerinin bozulmasını engeller
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except io.UnsupportedOperation:
        pass


def run_eda() -> None:
    root_dir = Path(__file__).resolve().parents[3]
    gold_dir = root_dir / "data" / "gold"
    gold_files = list(gold_dir.glob("*.parquet"))
    if not gold_files:
        raise FileNotFoundError(f"Gold tablosu bulunamadı: {gold_dir}")

    main_df = pl.read_parquet(max(gold_files))

    holidays_path = (
        root_dir / "data" / "silver" / "holidays" / "turkey_holidays_2023_2025.parquet"
    )
    holidays_df = pl.read_parquet(holidays_path) if holidays_path.exists() else None

    # Dinamik eğitim çerçevesi (Lag kolonları ve tatil bayrakları dahil)
    frame = build_training_frame(main_df, holidays_df)

    features = list(FEATURE_COLUMNS)
    target = TARGET_COLUMN

    df = frame.select(features + [target]).drop_nulls()
    print("Toplam analiz satırı:", df.height)

    pl.Config.set_tbl_cols(25)
    pl.Config.set_tbl_rows(25)

    print("\nHedefle Korelasyon (Pearson / Spearman):")
    print(
        pl.DataFrame(
            {
                "feature": features,
                "pearson": [df.select(pl.corr(c, target)).item() for c in features],
                "spearman": [
                    df.select(pl.corr(c, target, method="spearman")).item()
                    for c in features
                ],
            }
        ).with_columns(pl.col(["pearson", "spearman"]).round(3))
    )

    print("\nVIF (Variance Inflation Factor):")
    x_matrix = df.select(features).to_numpy().astype(float)
    vif_rows = []
    for j, name in enumerate(features):
        y_col = x_matrix[:, j]
        a_matrix = np.column_stack(
            [np.ones(len(x_matrix)), np.delete(x_matrix, j, axis=1)]
        )
        coef, *_ = np.linalg.lstsq(a_matrix, y_col, rcond=None)
        r2 = 1 - (y_col - a_matrix @ coef).var() / y_col.var()
        vif_val = round(1 / (1 - r2), 2) if r2 < 1 else float("inf")
        vif_rows.append((name, vif_val))

    print(pl.DataFrame(vif_rows, schema=["feature", "vif"], orient="row"))


if __name__ == "__main__":
    run_eda()
