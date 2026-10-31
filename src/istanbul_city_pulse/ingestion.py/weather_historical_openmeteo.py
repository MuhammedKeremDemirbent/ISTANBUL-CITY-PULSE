import json
from pathlib import Path

import polars as pl
import requests

URL = "https://archive-api.open-meteo.com/v1/era5"
HOURLY = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]
POINT = {"point_id": "istanbul_center", "lat": 41.01, "lon": 28.98}


def fetch(lat: float, lon: float, start: str, end: str) -> dict:
    r = requests.get(
        URL,
        params={
            "latitude": lat, "longitude": lon,
            "start_date": start, "end_date": end,
            "hourly": ",".join(HOURLY), "timezone": "UTC",
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def to_frame(payload: dict, point_id: str) -> pl.DataFrame:
    df = pl.DataFrame(payload["hourly"])          
    return df.with_columns(
        pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M")
          .dt.replace_time_zone("UTC").alias("hour_utc"),
        pl.lit(point_id).alias("weather_point_id"),
    ).drop("time")


def main(start: str = "2023-01-01", end: str = "2025-12-31") -> None:
    payload = fetch(POINT["lat"], POINT["lon"], start, end)

    bronze = Path(f"data/bronze/openmeteo_era5/{POINT['point_id']}_{start}_{end}.json")
    bronze.parent.mkdir(parents=True, exist_ok=True)
    bronze.write_text(json.dumps(payload))        

    df = to_frame(payload, POINT["point_id"])
    silver = Path(f"data/silver/weather_hourly/{POINT['point_id']}_{start}_{end}.parquet")
    silver.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(silver)


if __name__ == "__main__":
    main()