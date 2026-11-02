"""`main` analitik tablosundan model eğitim çerçevesi üretir. Genel düzeltme sağlar data leakgeyi engelleme konusunda çok önemlidir."""

import logging
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


class FeatureSchemaError(ValueError):
    """`main` tablosu eğitim çerçevesi sözleşmesine uymuyor."""


def build_training_frame(
    main: pl.DataFrame,
    holidays: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """`main` tablosunu `FRAME_COLUMNS` sözleşmesine uyan eğitim çerçevesine çevirir.

    * `holidays`: `date` (pl.Date) kolonuna sahip tatil günleri tablosu.
      Verilirse her satıra `is_holiday` (0/1) etiketi eklenir.
      Verilmezse tüm satırlar 0 alır — tatil bilgisi olmadan da model çalışır.
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
    """Zamana göre sıralı tablodan hedef değişkenin 1 saatlik ve 24 saatlik gecikmelerini üretir."""
    if TARGET_COLUMN not in main.columns:
        raise FeatureSchemaError(
            f"Lag öznitelikleri için '{TARGET_COLUMN}' kolonu bulunamadı."
        )
    return main.sort(TIMESTAMP_COLUMN).with_columns(
        pl.col(TARGET_COLUMN).shift(1).alias("traffic_index_av_lag_1h"),
        pl.col(TARGET_COLUMN).shift(24).alias("traffic_index_av_lag_24h"),
    )


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
