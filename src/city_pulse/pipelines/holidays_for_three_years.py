import logging
from pathlib import Path

import holidays
import polars as pl

log = logging.getLogger(__name__)

_YEARS = [2023, 2024, 2025]
_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "silver" / "holidays"
_OUTPUT_FILE = _OUTPUT_DIR / "turkey_holidays_2023_2025.parquet"


def build_holidays_frame(years: list[int]) -> pl.DataFrame:
    """Build a Polars DataFrame of Turkey official holidays for given years."""
    tr = holidays.Turkey(years=years)
    rows = [{"date": date, "holiday_name": name} for date, name in sorted(tr.items())]
    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("holiday_name").cast(pl.Utf8),
    )


def save_holidays(frame: pl.DataFrame, output_path: Path) -> None:
    """Save DataFrame to Parquet using atomic write."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.parquet")
    frame.write_parquet(tmp_path)
    tmp_path.rename(output_path)
    log.info("Saved holiday data: %s (%d rows)", output_path, len(frame))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    log.info("Generating Turkey holidays for years: %s", _YEARS)

    frame = build_holidays_frame(_YEARS)

    log.info("Total holidays: %d", len(frame))
    log.info("\n%s", frame)

    save_holidays(frame, _OUTPUT_FILE)


if __name__ == "__main__":
    main()
