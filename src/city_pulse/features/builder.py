import logging
from collections.abc import Mapping
from typing import Final

import polars as pl

from city_pulse.features.columns import (
    COLUMN_DTYPES,
    FRAME_COLUMNS,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
)

log = logging.getLogger(__name__)

HEAVY_RAIN_THRESHOLD_MM: Final[float] = 3.0

# Lag adı → kaç saat geriye bakılacağı.
# Sihirli sayıları kaldırmak ve kolon adını tek yerde tutmak için sabit harita.
LAG_HOURS: Final[Mapping[str, int]] = {
    "traffic_index_av_lag_1h": 1,
    "traffic_index_av_lag_24h": 24,
}


class FeatureSchemaError(ValueError):
    """`main` tablosu eğitim çerçevesi sözleşmesine uymuyor."""


def build_training_frame(
    main: pl.DataFrame,
    holidays: pl.DataFrame | None = None,
    school_calendar: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """`main` tablosunu `FRAME_COLUMNS` sözleşmesine uyan eğitim çerçevesine çevirir.

    * `holidays`: `date` (pl.Date) kolonuna sahip tatil günleri tablosu.
      Verilirse her satıra `is_holiday` (0/1) etiketi eklenir.
      Verilmezse tüm satırlar 0 alır — tatil bilgisi olmadan da model çalışır.
    * `school_calendar`: `timestamp` (pl.Datetime) ve okul bayrak kolonlarına
      sahip saatlik okul takvimi tablosu.
      Verilirse `is_school_term`, `is_school_morning_rush`, `is_school_afternoon_rush`
      timestamp üzerinden join ile eklenir.
      Verilmezse tüm satırlar 0 alır.
    * Yalnızca sözleşmedeki kolonlar alınır; diğerleri (örn. `traffic_index`,
      `traffic_index_an`) çıktıya girmez, böylece hedef sızıntısı engellenir.
    * Sayısal kolonlar `COLUMN_DTYPES` tiplerine sabitlenir.
    * Eksik değer içeren satırlar **doldurulmaz**, çıkarılır ve sayısı loglanır.
    * Satırlar `TIMESTAMP_COLUMN`'a göre sıralanır.

    Raises:
        FeatureSchemaError: Sözleşme kolonlarından biri yoksa veya zaman damgası
            kolonu tarih-saat tipinde değilse.
    """
    main = _attach_holiday_flag(main, holidays)
    main = _attach_rain_flags(main)
    main = _attach_lag_features(main)
    main = _attach_school_flags(main, school_calendar)
    _validate_schema(main)
    frame = main.select(FRAME_COLUMNS).with_columns(
        [pl.col(name).cast(dtype) for name, dtype in COLUMN_DTYPES.items()]
    )
    return _drop_incomplete_rows(frame).sort(TIMESTAMP_COLUMN)


def _attach_holiday_flag(
    main: pl.DataFrame,
    holidays: pl.DataFrame | None,
) -> pl.DataFrame:
    """Timestamp'ten tarih çıkarır ve tatil tablosuyla eşleştirerek `is_holiday` ekler.

    Tatil tablosu yoksa veya boşsa tüm satırlar 0 (tatil değil) alır.
    """
    if holidays is not None and holidays.height > 0:
        holiday_dates = holidays.select(pl.col("date").cast(pl.Date))["date"]
        return main.with_columns(
            pl.col(TIMESTAMP_COLUMN)
            .dt.date()
            .is_in(holiday_dates)
            .cast(pl.Int32)
            .alias("is_holiday")
        )
    log.debug("Tatil tablosu sağlanmadı; is_holiday=0 olarak işaretlendi.")
    return main.with_columns(pl.lit(0).cast(pl.Int32).alias("is_holiday"))


def _attach_rain_flags(main: pl.DataFrame) -> pl.DataFrame:
    """Precipitation kolonundan `is_raining` (yağış var mı) ve `heavy_rain` (sağanak mı) bayraklarını üretir."""
    if "precipitation" not in main.columns:
        raise FeatureSchemaError(
            "Yağış bayrakları için 'precipitation' kolonu bulunamadı."
        )
    return main.with_columns(
        (pl.col("precipitation") > 0.0).cast(pl.Int32).alias("is_raining"),
        (pl.col("precipitation") >= HEAVY_RAIN_THRESHOLD_MM)
        .cast(pl.Int32)
        .alias("heavy_rain"),
    )


def _attach_lag_features(main: pl.DataFrame) -> pl.DataFrame:
    """Hedef değişkenin gecikmeli değerlerini zaman anahtarlı join ile ekler.

    Her lag kolonu için `timestamp - N saat` değerine sahip satır tabloda
    varsa o satırın hedef değeri alınır, yoksa NULL bırakılır.
    Bu yaklaşım satır kaydırmanın (shift) aksine tablo boşluklarına karşı
    doğrudur: boşluktan sonraki satır yanlış zamandaki değeri taşımaz.

    Raises:
        FeatureSchemaError: Hedef kolon yoksa veya aynı timestamp birden fazla
            geçiyorsa (join satır çoğaltmasını önlemek için).
    """
    if TARGET_COLUMN not in main.columns:
        raise FeatureSchemaError(
            f"Lag öznitelikleri için '{TARGET_COLUMN}' kolonu bulunamadı."
        )
    # Tekrar eden timestamp join'de satır çoğalmasına yol açar; erken yakala
    if main[TIMESTAMP_COLUMN].n_unique() != main.height:
        raise FeatureSchemaError(
            f"'{TIMESTAMP_COLUMN}' kolonunda tekrar eden değerler var; "
            "lag join'i satır çoğaltır."
        )

    sorted_main = main.sort(TIMESTAMP_COLUMN)
    # Her lag için: (timestamp + offset, hedef_değer) tablosu kur, left join yap
    result = sorted_main
    for col_name, hours in LAG_HOURS.items():
        offset = pl.duration(hours=hours)
        lag_table = sorted_main.select(
            # Lag tablosunda zaman damgasını N saat ilerlet; böylece ana tablodaki
            # gerçek saatiyle eşleşir (ör. 1h lag: saat 5'in değeri → join'de saat 6'ya karşılık gelir)
            (pl.col(TIMESTAMP_COLUMN) + offset).alias(TIMESTAMP_COLUMN),
            pl.col(TARGET_COLUMN).alias(col_name),
        )
        result = result.join(lag_table, on=TIMESTAMP_COLUMN, how="left")

    return result


def _attach_school_flags(
    main: pl.DataFrame,
    school_calendar: pl.DataFrame | None,
) -> pl.DataFrame:
    """Okul takvimi tablosuyla timestamp üzerinden join ederek okul bayraklarını ekler.

    Kolonlar gold tablosundan geliyorsa (merged_datas pipeline'ı tarafından
    önceden eklenmiş) tekrar join yapılmaz; mevcut değerler korunur.
    `school_calendar` sağlanmazsa ve kolonlar yoksa tüm satırlar 0 alır.
    """
    school_cols = (
        "is_school_term",
        "is_school_morning_rush",
        "is_school_afternoon_rush",
    )
    # Gold tablosu bu kolonları zaten taşıyorsa join gereksizdir
    if all(c in main.columns for c in school_cols):
        log.debug("Okul bayrakları gold tablosundan geliyor; join atlandı.")
        return main
    if school_calendar is not None and school_calendar.height > 0:
        # Sadece gerekli kolonları al; timestamp üzerinden sol join
        cal = school_calendar.select(TIMESTAMP_COLUMN, *school_cols)
        joined = main.join(cal, on=TIMESTAMP_COLUMN, how="left")
        return joined.with_columns(
            [pl.col(c).fill_null(0).cast(pl.Int32) for c in school_cols]
        )
    log.debug("Okul takvimi sağlanmadı; okul bayrakları=0 olarak işaretlendi.")
    return main.with_columns([pl.lit(0).cast(pl.Int32).alias(c) for c in school_cols])


def _validate_schema(main: pl.DataFrame) -> None:
    missing = [name for name in FRAME_COLUMNS if name not in main.columns]
    if missing:
        raise FeatureSchemaError(f"Eksik kolonlar: {missing}")
    timestamp_dtype = main.schema[TIMESTAMP_COLUMN]
    if not isinstance(timestamp_dtype, pl.Datetime):
        raise FeatureSchemaError(
            f"'{TIMESTAMP_COLUMN}' tarih-saat tipinde olmalı, bulunan tip: {timestamp_dtype}"
        )


def _drop_incomplete_rows(frame: pl.DataFrame) -> pl.DataFrame:
    complete = frame.drop_nulls()
    dropped = frame.height - complete.height
    if dropped:
        log.warning(
            "%d/%d satır eksik değer nedeniyle çıkarıldı (%.2f%%)",
            dropped,
            frame.height,
            100 * dropped / frame.height,
        )
    return complete
