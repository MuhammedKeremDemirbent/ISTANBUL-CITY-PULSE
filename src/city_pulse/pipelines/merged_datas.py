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
    silver_school_path: Path,
    gold_output_path: Path,
) -> int:
    """Merge Silver tables, attach calendar features, and write Gold analytical Parquet.

    - De-duplicates traffic to the latest observation per hour.
    - Joins weather, holidays, and school calendar features.
    """
    gold_output_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    query = f"""
    COPY (
        WITH traffic_hourly AS (
            SELECT
                date_trunc('hour', timestamp) AS timestamp,
                traffic_index,
                traffic_index_an,
                traffic_index_av,
                ROW_NUMBER() OVER (
                    PARTITION BY date_trunc('hour', timestamp)
                    ORDER BY timestamp DESC
                ) AS rn
            FROM read_parquet('{silver_traffic_path.as_posix()}')
        )
        SELECT
            t.timestamp,

            -- Time Features
            EXTRACT(hour FROM t.timestamp)::INT AS hour,
            EXTRACT(isodow FROM t.timestamp)::INT AS day_of_week,  -- 1: Mon, 7: Sun
            EXTRACT(month FROM t.timestamp)::INT AS month,
            CASE WHEN EXTRACT(isodow FROM t.timestamp) IN (6, 7) THEN 1 ELSE 0 END AS is_weekend,
            CASE WHEN h.date IS NOT NULL THEN 1 ELSE 0 END AS is_holiday,

            -- School Calendar Flags
            COALESCE(sc.is_school_term, 0)::INT           AS is_school_term,
            COALESCE(sc.is_school_morning_rush, 0)::INT   AS is_school_morning_rush,
            COALESCE(sc.is_school_afternoon_rush, 0)::INT AS is_school_afternoon_rush,

            -- Target Traffic Variables
            t.traffic_index,
            t.traffic_index_an,
            t.traffic_index_av,

            -- Weather Factors
            w.temperature_2m,
            w.precipitation,
            w.wind_speed_10m,
            w.relative_humidity_2m

        FROM (
            SELECT timestamp, traffic_index, traffic_index_an, traffic_index_av
            FROM traffic_hourly
            WHERE rn = 1
        ) t
        JOIN read_parquet('{silver_weather_path.as_posix()}') w ON t.timestamp = w.timestamp
        LEFT JOIN (
            SELECT DISTINCT date
            FROM read_parquet('{silver_holidays_path.as_posix()}')
        ) h ON CAST(t.timestamp AS DATE) = h.date
        LEFT JOIN read_parquet('{silver_school_path.as_posix()}') sc ON t.timestamp = sc.timestamp
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
        raise RuntimeError("Failed to read row count from output file.")
    row_count: int = int(res[0])

    log.info(
        "Gold table created: %d rows written -> %s (%s to %s)",
        row_count,
        gold_output_path,
        start,
        end,
    )
    return row_count


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gold Table Builder")
    parser.add_argument(
        "start", nargs="?", default=DEFAULT_START, type=date.fromisoformat
    )
    parser.add_argument("end", nargs="?", default=DEFAULT_END, type=date.fromisoformat)
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("Start date must be earlier than or equal to end date.")
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
    silver_school = (
        root_dir / "data" / "silver" / "school" / "school_calendar_2023_2025.parquet"
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
        silver_school_path=silver_school,
        gold_output_path=gold_output,
    )


if __name__ == "__main__":
    main()
