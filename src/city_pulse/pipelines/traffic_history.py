import argparse
import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)

DEFAULT_START = "2023-01-01"
DEFAULT_END = "2025-12-31"


def run_traffic_history(
    start: str,
    end: str,
    *,
    bronze_csv: Path,
    silver_parquet: Path,
) -> int:
    """Bronze CSV'den dakikalık veriyi okur, saatlik tekil kayıtları seçip silver Parquet olarak kaydeder."""
    if not bronze_csv.exists():
        raise FileNotFoundError(f"Ham veri dosyası bulunamadı: {bronze_csv}")

    silver_parquet.parent.mkdir(parents=True, exist_ok=True)

    # DuckDB bellek içi motoruyla CSV -> Filtreleme & Dedup -> Parquet dönüşümü
    con = duckdb.connect()

    query = f"""
    COPY (
        SELECT 
            datetime::TIMESTAMP AS timestamp,
            TI AS traffic_index,
            TI_An AS traffic_index_an,
            TI_Av AS traffic_index_av
        FROM read_csv_auto('{bronze_csv.as_posix()}')
        WHERE datetime >= '{start} 00:00:00'
          AND datetime <= '{end} 23:59:59'
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY date_trunc('hour', datetime::TIMESTAMP)
            ORDER BY datetime::TIMESTAMP ASC
        ) = 1
        ORDER BY timestamp
    ) TO '{silver_parquet.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """

    con.execute(query)

    res = con.execute(
        f"SELECT count(*) FROM read_parquet('{silver_parquet.as_posix()}')"
    ).fetchone()
    if res is None:
        raise RuntimeError("Satır sayısı okunamadı.")
    row_count: int = int(res[0])

    log.info(
        "Trafik verisi dönüştürüldü: %d satır yazıldı -> %s (%s - %s)",
        row_count,
        silver_parquet,
        start,
        end,
    )
    return row_count


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="İBB Trafik İndeksi saatlik dönüştürücü"
    )
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
    bronze_csv = root_dir / "data" / "bronze" / "traffic_pulse" / "istanbul_traffic.csv"
    silver_parquet = (
        root_dir
        / "data"
        / "silver"
        / "traffic_hourly"
        / f"istanbul_{args.start.isoformat()}_{args.end.isoformat()}.parquet"
    )

    run_traffic_history(
        args.start.isoformat(),
        args.end.isoformat(),
        bronze_csv=bronze_csv,
        silver_parquet=silver_parquet,
    )


if __name__ == "__main__":
    main()
