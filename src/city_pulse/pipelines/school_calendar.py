import logging
from datetime import date, datetime
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)

# Active school term date ranges (inclusive)
_PERIODS_2022_2023 = [
    (date(2023, 1, 2), date(2023, 1, 20)),
    (date(2023, 2, 6), date(2023, 4, 14)),
    (date(2023, 4, 24), date(2023, 6, 16)),
]

_PERIODS_2023_2024 = [
    (date(2023, 9, 11), date(2023, 11, 10)),
    (date(2023, 11, 20), date(2024, 1, 19)),
    (date(2024, 2, 5), date(2024, 4, 7)),
    (date(2024, 4, 15), date(2024, 6, 14)),
]

_PERIODS_2024_2025 = [
    (date(2024, 9, 9), date(2024, 11, 10)),
    (date(2024, 11, 18), date(2025, 1, 17)),
    (date(2025, 2, 3), date(2025, 3, 30)),
    (date(2025, 4, 7), date(2025, 6, 20)),
]

_PERIODS_2025_2026 = [
    (date(2025, 9, 8), date(2025, 11, 9)),
    (date(2025, 11, 17), date(2025, 12, 31)),
]

ACTIVE_SCHOOL_PERIODS: list[tuple[date, date]] = (
    _PERIODS_2022_2023 + _PERIODS_2023_2024 + _PERIODS_2024_2025 + _PERIODS_2025_2026
)

# Morning rush: 06:00 - 08:59
MORNING_RUSH_HOURS: frozenset[int] = frozenset({6, 7, 8})
# Afternoon rush: 15:00 - 17:59
AFTERNOON_RUSH_HOURS: frozenset[int] = frozenset({15, 16, 17})

_START_DT = datetime(2023, 1, 1)
_END_DT = datetime(2025, 12, 31, 23)
_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "silver" / "school"
_OUTPUT_FILE = _OUTPUT_DIR / "school_calendar_2023_2025.parquet"
_HOLIDAYS_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "silver"
    / "holidays"
    / "turkey_holidays_2023_2025.parquet"
)


def _load_holiday_dates(holidays_path: Path) -> frozenset[date]:
    """Load holiday dates as a frozenset from holidays parquet."""
    if not holidays_path.exists():
        log.warning(
            "Holidays file not found (%s); holidays will not be excluded.",
            holidays_path,
        )
        return frozenset()
    df = pl.read_parquet(holidays_path)
    return frozenset(df["date"].cast(pl.Date).to_list())


def _is_active_school_date(d: date, holiday_dates: frozenset[date]) -> bool:
    """Check if date is a weekday during an active school period and not a holiday."""
    if d.weekday() >= 5:
        return False
    if d in holiday_dates:
        return False
    return any(start <= d <= end for start, end in ACTIVE_SCHOOL_PERIODS)


def build_school_calendar(holiday_dates: frozenset[date]) -> pl.DataFrame:
    """Build hourly school calendar DataFrame."""
    timestamps = pl.datetime_range(_START_DT, _END_DT, interval="1h", eager=True)

    df = pl.DataFrame({"timestamp": timestamps}).with_columns(
        pl.col("timestamp").dt.date().alias("_date"),
        pl.col("timestamp").dt.hour().alias("_hour"),
    )

    unique_dates = df["_date"].unique().to_list()
    school_day_map = {d: _is_active_school_date(d, holiday_dates) for d in unique_dates}

    term_series = pl.Series(
        "is_school_term",
        [int(school_day_map[d]) for d in df["_date"].to_list()],
        dtype=pl.Int32,
    )

    hour_list = df["_hour"].to_list()
    term_list = term_series.to_list()

    morning_series = pl.Series(
        "is_school_morning_rush",
        [
            int(term == 1 and h in MORNING_RUSH_HOURS)
            for term, h in zip(term_list, hour_list)
        ],
        dtype=pl.Int32,
    )
    afternoon_series = pl.Series(
        "is_school_afternoon_rush",
        [
            int(term == 1 and h in AFTERNOON_RUSH_HOURS)
            for term, h in zip(term_list, hour_list)
        ],
        dtype=pl.Int32,
    )

    return df.drop("_date", "_hour").with_columns(
        term_series,
        morning_series,
        afternoon_series,
    )


def save_school_calendar(frame: pl.DataFrame, output_path: Path) -> None:
    """Save DataFrame to Parquet using atomic write."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.parquet")
    frame.write_parquet(tmp_path)
    tmp_path.rename(output_path)
    log.info(
        "Saved school calendar: %s (%d rows)",
        output_path,
        len(frame),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    log.info("Building school calendar: %s to %s", _START_DT.date(), _END_DT.date())

    holiday_dates = _load_holiday_dates(_HOLIDAYS_PATH)
    log.info("Total holiday days: %d", len(holiday_dates))

    frame = build_school_calendar(holiday_dates)

    school_days = frame["is_school_term"].sum()
    morning_rush = frame["is_school_morning_rush"].sum()
    afternoon_rush = frame["is_school_afternoon_rush"].sum()
    log.info(
        "School term hours: %d | Morning rush: %d | Afternoon rush: %d",
        school_days,
        morning_rush,
        afternoon_rush,
    )

    save_school_calendar(frame, _OUTPUT_FILE)


if __name__ == "__main__":
    main()
