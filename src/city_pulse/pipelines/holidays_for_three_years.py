"""Türkiye resmi tatil günleri için Parquet veri seti üretir.

2023-2025 yıllarını kapsar; hem resmi hem dini tatilleri içerir.
Çıktı: data/silver/holidays/turkey_holidays_2023_2025.parquet

Kullanım:
    PYTHONPATH=src uv run python -m city_pulse.pipelines.holidays_for_three_years
"""

import logging
from pathlib import Path

import holidays
import polars as pl

log = logging.getLogger(__name__)

_YEARS = [2023, 2024, 2025]
_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "silver" / "holidays"
_OUTPUT_FILE = _OUTPUT_DIR / "turkey_holidays_2023_2025.parquet"


def build_holidays_frame(years: list[int]) -> pl.DataFrame:
    """Verilen yıllar için Türkiye tatil günlerini Polars DataFrame'e dönüştürür.

    Dini tatiller (Ramazan/Kurban Bayramı) dahil tüm resmi tatilleri kapsar.
    Her satır bir tatil gününü temsil eder.
    """
    tr = holidays.Turkey(years=years)
    rows = [
        {"date": date, "holiday_name": name}
        for date, name in sorted(tr.items())
    ]
    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("holiday_name").cast(pl.Utf8),
    )


def save_holidays(frame: pl.DataFrame, output_path: Path) -> None:
    """DataFrame'i atomik yazma ile Parquet dosyasına kaydeder."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.parquet")
    frame.write_parquet(tmp_path)
    tmp_path.rename(output_path)
    log.info("Tatil verisi kaydedildi: %s (%d satır)", output_path, len(frame))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("Türkiye tatil günleri oluşturuluyor: %s", _YEARS)

    frame = build_holidays_frame(_YEARS)

    log.info("Tatil günü sayısı: %d", len(frame))
    log.info("\n%s", frame)

    save_holidays(frame, _OUTPUT_FILE)


if __name__ == "__main__":
    main()
