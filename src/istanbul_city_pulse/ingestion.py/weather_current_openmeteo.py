import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import requests

URL = "https://api.open-meteo.com/v1/forecast"
VARS = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]
POINT = {"point_id": "istanbul_center", "lat": 41.01, "lon": 28.98}


def fetch(lat: float, lon: float) -> dict:
    r = requests.get(
        URL,
        params={
            "latitude": lat, "longitude": lon,
            "current": ",".join(VARS),
            "hourly": ",".join(VARS),
            "forecast_days": 2,
            "timezone": "UTC",
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _to_utc(col: str) -> pl.Expr:
    return pl.col(col).str.to_datetime("%Y-%m-%dT%H:%M").dt.replace_time_zone("UTC")


def current_frame(payload: dict, point_id: str, fetched_at: datetime) -> pl.DataFrame:
    cur = payload["current"]
    row = {"observed_at_utc": cur["time"], **{v: cur[v] for v in VARS}}
    return pl.DataFrame([row]).with_columns(
        _to_utc("observed_at_utc"),
        pl.lit(point_id).alias("weather_point_id"),
        pl.lit(fetched_at).alias("fetched_at_utc"),
        pl.lit("current").alias("source"),
    )


def forecast_frame(payload: dict, point_id: str, fetched_at: datetime) -> pl.DataFrame:
    return pl.DataFrame(payload["hourly"]).with_columns(
        _to_utc("time").alias("hour_utc"),
        pl.lit(point_id).alias("weather_point_id"),
        pl.lit(fetched_at).alias("fetched_at_utc"),
        pl.lit("forecast").alias("source"),
    ).drop("time")


def main() -> None:
    fetched_at = datetime.now(timezone.utc)
    stamp = fetched_at.strftime("%Y%m%dT%H%M%SZ")
    pid = POINT["point_id"]

    payload = fetch(POINT["lat"], POINT["lon"])

    bronze = Path(f"data/bronze/openmeteo_live/{pid}/{stamp}.json")
    bronze.parent.mkdir(parents=True, exist_ok=True)
    bronze.write_text(json.dumps(payload))

    # SILVER: tablo halinde, iki ayrı dosya
    silver_dir = Path(f"data/silver/weather_live/{pid}")
    silver_dir.mkdir(parents=True, exist_ok=True)
    cur = current_frame(payload, pid, fetched_at)
    fc = forecast_frame(payload, pid, fetched_at)
    cur.write_parquet(silver_dir / f"current_{stamp}.parquet")
    fc.write_parquet(silver_dir / f"forecast_{stamp}.parquet")

    print(cur)
    print(fc.shape)
    print(fc.head(3))


if __name__ == "__main__":
    main()