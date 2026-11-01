import numpy as np
import polars as pl

PATH = "data/gold/city_pulse_hourly_2023-01-01_2025-12-31.parquet"
FEATURES = [
    "hour", "day_of_week", "month", "is_weekend", "is_holiday",
    "temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m",
]
TARGET = "traffic_index_av"

pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_rows(20)
df = pl.read_parquet(PATH).select(FEATURES + [TARGET]).drop_nulls()
print("satır:", df.height)

print("\nHedefle korelasyon (pearson / spearman)")
print(pl.DataFrame({
    "feature": FEATURES,
    "pearson": [df.select(pl.corr(c, TARGET)).item() for c in FEATURES],
    "spearman": [df.select(pl.corr(c, TARGET, method="spearman")).item() for c in FEATURES],
}).with_columns(pl.col(["pearson", "spearman"]).round(3)))

print("\nKolonlar arası korelasyon (satır/sütun sırası FEATURES ile aynı)")
print(df.select(FEATURES).corr().with_columns(pl.all().round(2)))

X = df.select(FEATURES).to_numpy().astype(float)
rows = []
for j, name in enumerate(FEATURES):
    y = X[:, j]
    A = np.column_stack([np.ones(len(X)), np.delete(X, j, axis=1)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - (y - A @ coef).var() / y.var()
    rows.append((name, round(1 / (1 - r2), 2) if r2 < 1 else float("inf")))
print("\nVIF")
print(pl.DataFrame(rows, schema=["feature", "vif"], orient="row"))