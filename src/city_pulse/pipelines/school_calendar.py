import logging
from datetime import date, datetime
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MEB Aktif Okul Dönemleri (başlangıç ve bitiş tarihleri dahil)
# ---------------------------------------------------------------------------
# Her tuple (başlangıç, bitiş) şeklinde; tarihler dahildir.
# Araları (ara tatil, yarıyıl tatili) liste dışıdır — otomatik olarak
# is_school_term = 0 kalır.
# ---------------------------------------------------------------------------

# 2022-2023 öğretim yılı (2023 Ocak-Haziran kısmı; İstanbul bölgesi)
# Kaynak: MEB 2022-2023 takvimi (kullanıcı tarafından sağlanmadı;
# İstanbul için standart MEB bitiş tarihleri kullanıldı).
_PERIODS_2022_2023 = [
    (date(2023, 1, 2), date(2023, 1, 20)),  # 1. dönem sonu
    (date(2023, 2, 6), date(2023, 4, 14)),  # 2. dönem 1. kısım
    (date(2023, 4, 24), date(2023, 6, 16)),  # 2. dönem 2. kısım (Nisan tatili sonrası)
]

# 2023-2024 öğretim yılı (MEB resmi genelge)
# 1. Dönem: 11 Eylül 2023 - 19 Ocak 2024
#   Ara Tatil: 13-17 Kasım 2023
# Yarıyıl Tatili: 22 Ocak - 2 Şubat 2024
# 2. Dönem: 5 Şubat 2024 - 14 Haziran 2024
#   Ara Tatil: 8-12 Nisan 2024
_PERIODS_2023_2024 = [
    (date(2023, 9, 11), date(2023, 11, 10)),  # 1. dönem 1. kısım
    (date(2023, 11, 20), date(2024, 1, 19)),  # 1. dönem 2. kısım (ara tatil sonrası)
    (date(2024, 2, 5), date(2024, 4, 7)),  # 2. dönem 1. kısım
    (date(2024, 4, 15), date(2024, 6, 14)),  # 2. dönem 2. kısım (ara tatil sonrası)
]

# 2024-2025 öğretim yılı (MEB resmi genelge)
# 1. Dönem: 9 Eylül 2024 - 17 Ocak 2025
#   Ara Tatil: 11-15 Kasım 2024
# Yarıyıl Tatili: 20 Ocak - 31 Ocak 2025
# 2. Dönem: 3 Şubat 2025 - 20 Haziran 2025
#   Ara Tatil: 31 Mart - 4 Nisan 2025
_PERIODS_2024_2025 = [
    (date(2024, 9, 9), date(2024, 11, 10)),  # 1. dönem 1. kısım
    (date(2024, 11, 18), date(2025, 1, 17)),  # 1. dönem 2. kısım (ara tatil sonrası)
    (date(2025, 2, 3), date(2025, 3, 30)),  # 2. dönem 1. kısım
    (date(2025, 4, 7), date(2025, 6, 20)),  # 2. dönem 2. kısım (ara tatil sonrası)
]

# 2025-2026 öğretim yılı (MEB resmi genelge) — 2025 sonuna kadar olan kısım
# 1. Dönem: 8 Eylül 2025 - 16 Ocak 2026
#   Ara Tatil: 10-14 Kasım 2025
# Verisetimizdeki son tarih 2025-12-31 olduğu için 1. dönem bu tarihte kesilir.
_PERIODS_2025_2026 = [
    (date(2025, 9, 8), date(2025, 11, 9)),  # 1. dönem 1. kısım
    (date(2025, 11, 17), date(2025, 12, 31)),  # 1. dönem 2. kısım (veri seti sonu)
]

# Tüm aktif okul dönemleri tek listede
ACTIVE_SCHOOL_PERIODS: list[tuple[date, date]] = (
    _PERIODS_2022_2023 + _PERIODS_2023_2024 + _PERIODS_2024_2025 + _PERIODS_2025_2026
)

# Sabah rush: 06:00 - 08:59
MORNING_RUSH_HOURS: frozenset[int] = frozenset({6, 7, 8})
# Öğleden sonra rush: 15:00 - 17:59
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
    """Tatil parquet dosyasından tatil günlerini set olarak yükler.

    Dosya yoksa boş set döner; eksik tatil bilgisi is_school_term'i
    bozmamalıdır (okul yanlışlıkla açık görünür, kapalı değil).
    """
    if not holidays_path.exists():
        log.warning(
            "Tatil tablosu bulunamadı (%s); tatiller hariç tutulmayacak.",
            holidays_path,
        )
        return frozenset()
    df = pl.read_parquet(holidays_path)
    return frozenset(df["date"].cast(pl.Date).to_list())


def _is_active_school_date(d: date, holiday_dates: frozenset[date]) -> bool:
    """Verilen gün aktif okul dönemi içinde ve çalışma günü mü?

    Üç koşulun tümü sağlanmalıdır:
    1. Tarihin MEB aktif dönem aralıklarından birine denk gelmesi.
    2. Hafta içi olması (Pazartesi=0 ... Cuma=4).
    3. Resmi/dini tatil günü olmaması.
    """
    if d.weekday() >= 5:  # Cumartesi (5) veya Pazar (6)
        return False
    if d in holiday_dates:
        return False
    return any(start <= d <= end for start, end in ACTIVE_SCHOOL_PERIODS)


def build_school_calendar(holiday_dates: frozenset[date]) -> pl.DataFrame:
    """Saatlik okul takvimi DataFrame'ini üretir.

    Her satır bir saat dilimine karşılık gelir:
    - `timestamp`: Saatlik UTC zaman damgası
    - `is_school_term`: Aktif okul döneminde hafta içi, tatil olmayan gün → 1
    - `is_school_morning_rush`: Okul günü + saat ∈ {6,7,8} → 1
    - `is_school_afternoon_rush`: Okul günü + saat ∈ {15,16,17} → 1
    """
    timestamps = pl.datetime_range(_START_DT, _END_DT, interval="1h", eager=True)

    df = pl.DataFrame({"timestamp": timestamps}).with_columns(
        pl.col("timestamp").dt.date().alias("_date"),
        pl.col("timestamp").dt.hour().alias("_hour"),
    )

    # Gün bazlı okul durumu: her benzersiz gün için bir kez hesapla
    unique_dates = df["_date"].unique().to_list()
    school_day_map = {d: _is_active_school_date(d, holiday_dates) for d in unique_dates}

    # is_school_term: gün aktif okul günü mü?
    term_series = pl.Series(
        "is_school_term",
        [int(school_day_map[d]) for d in df["_date"].to_list()],
        dtype=pl.Int32,
    )

    hour_list = df["_hour"].to_list()
    term_list = term_series.to_list()

    # Rush bayrakları: okul günü AND ilgili saatte olmak
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
    """DataFrame'i atomik yazma ile Parquet dosyasına kaydeder."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.parquet")
    frame.write_parquet(tmp_path)
    tmp_path.rename(output_path)
    log.info(
        "Okul takvimi kaydedildi: %s (%d satır)",
        output_path,
        len(frame),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    log.info("Okul takvimi oluşturuluyor: %s → %s", _START_DT.date(), _END_DT.date())

    holiday_dates = _load_holiday_dates(_HOLIDAYS_PATH)
    log.info("Tatil günü sayısı: %d", len(holiday_dates))

    frame = build_school_calendar(holiday_dates)

    school_days = frame["is_school_term"].sum()
    morning_rush = frame["is_school_morning_rush"].sum()
    afternoon_rush = frame["is_school_afternoon_rush"].sum()
    log.info(
        "Okul saati: %d | Sabah rush: %d | Öğleden sonra rush: %d",
        school_days,
        morning_rush,
        afternoon_rush,
    )

    save_school_calendar(frame, _OUTPUT_FILE)


if __name__ == "__main__":
    main()
