"""Gold katmanı boru hattı: Silver (Trafik + Hava Durumu + Tatil) -> Gold (ML Özellik Tablosu).

Kullanım:
    uv run python -m city_pulse.pipelines.merged_datas 2023-01-01 2025-12-31
"""

import argparse
import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)

DEFAULT_START = "2023-01-01"
DEFAULT_END = "2025-12-31"


def run_gold_pipeline(
    start: str,
    end: str,
    *,
    silver_traffic_path: Path,
    silver_weather_path: Path,
    silver_holidays_path: Path,
    gold_output_path: Path,
) -> int:
    """Silver tablolarını birleştirir, zaman özelliklerini ekler ve Gold Parquet olarak kaydeder.

    Tatil tablosu günlüktür (saat yok); `timestamp`'in tarihine göre birleştirilir,
    böylece tatil gününün tüm saatleri `is_holiday = 1` olur.
    """
    gold_output_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    # Tatil tablosu tarih bazında tekilleştirilir: aynı güne iki kayıt düşse bile
    # LEFT JOIN saat satırlarını çoğaltmaz (fan-out koruması).
    query = f"""
    COPY (
        SELECT
            t.timestamp,

            -- Zaman Özellikleri (Model Girdileri)
            EXTRACT(hour FROM t.timestamp)::INT AS hour,
            EXTRACT(isodow FROM t.timestamp)::INT AS day_of_week,  -- 1: Pazartesi, 7: Pazar
            EXTRACT(month FROM t.timestamp)::INT AS month,
            CASE WHEN EXTRACT(isodow FROM t.timestamp) IN (6, 7) THEN 1 ELSE 0 END AS is_weekend,
            CASE WHEN h.date IS NOT NULL THEN 1 ELSE 0 END AS is_holiday,

            -- Hedef Değişkenler (Tahmin edilecek trafik indeksleri)
            t.traffic_index,
            t.traffic_index_an,
            t.traffic_index_av,

            -- Hava Durumu Etkenleri (Saf sayısal ölçümler)
            w.temperature_2m,
            w.precipitation,
            w.wind_speed_10m,
            w.relative_humidity_2m

        FROM read_parquet('{silver_traffic_path.as_posix()}') t
        JOIN read_parquet('{silver_weather_path.as_posix()}') w ON t.timestamp = w.timestamp
        LEFT JOIN (
            SELECT DISTINCT date
            FROM read_parquet('{silver_holidays_path.as_posix()}')
        ) h ON CAST(t.timestamp AS DATE) = h.date
        WHERE t.timestamp >= '{start} 00:00:00'
          AND t.timestamp <= '{end} 23:59:59'
        ORDER BY t.timestamp
    ) TO '{gold_output_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """

    con.execute(query)

    res = con.execute(
        f"SELECT count(*) FROM read_parquet('{gold_output_path.as_posix()}')"
    ).fetchone()
    if res is None:
        raise RuntimeError("Satır sayısı okunamadı.")
    row_count: int = int(res[0])

    log.info(
        "Gold tablosu hazırlandı: %d satır yazıldı -> %s (%s - %s)",
        row_count,
        gold_output_path,
        start,
        end,
    )
    return row_count


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="City Pulse Gold ML Tablo Üreticisi")
    parser.add_argument(
        "start", nargs="?", default=DEFAULT_START, type=date.fromisoformat
    )
    parser.add_argument("end", nargs="?", default=DEFAULT_END, type=date.fromisoformat)
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("Başlangıç tarihi bitiş tarihinden sonra olamaz")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args(argv)

    root_dir = Path(__file__).resolve().parents[3]
    silver_traffic = root_dir / "data" / "silver" / "traffic_hourly" / "*.parquet"
    silver_weather = root_dir / "data" / "silver" / "weather_hourly" / "*.parquet"
    silver_holidays = (
        root_dir / "data" / "silver" / "holidays" / "turkey_holidays_2023_2025.parquet"
    )
    gold_output = (
        root_dir
        / "data"
        / "gold"
        / f"city_pulse_hourly_{args.start.isoformat()}_{args.end.isoformat()}.parquet"
    )

    run_gold_pipeline(
        args.start.isoformat(),
        args.end.isoformat(),
        silver_traffic_path=silver_traffic,
        silver_weather_path=silver_weather,
        silver_holidays_path=silver_holidays,
        gold_output_path=gold_output,
    )


if __name__ == "__main__":
    main()