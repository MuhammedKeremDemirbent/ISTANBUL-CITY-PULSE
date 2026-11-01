"""`main` analitik tablosundan model eğitim çerçevesi üretir. Genel düzeltme sağlar data leakgeyi engelleme konusunda çok önemlidir."""

import logging

import polars as pl

from city_pulse.features.columns import (
    COLUMN_DTYPES,
    FRAME_COLUMNS,
    TIMESTAMP_COLUMN,
)

log = logging.getLogger(__name__)


class FeatureSchemaError(ValueError):
    """`main` tablosu eğitim çerçevesi sözleşmesine uymuyor."""


def build_training_frame(main: pl.DataFrame) -> pl.DataFrame:
    """`main` tablosunu `FRAME_COLUMNS` sözleşmesine uyan eğitim çerçevesine çevirir.

    * Yalnızca sözleşmedeki kolonlar alınır; diğerleri (örn. `traffic_index`,
      `traffic_index_an`) çıktıya girmez, böylece hedef sızıntısı engellenir.
    * Sayısal kolonlar `COLUMN_DTYPES` tiplerine sabitlenir.
    * Eksik değer içeren satırlar **doldurulmaz**, çıkarılır ve sayısı loglanır.
    * Satırlar `TIMESTAMP_COLUMN`'a göre sıralanır.

    Raises:
        FeatureSchemaError: Sözleşme kolonlarından biri yoksa veya zaman damgası
            kolonu tarih-saat tipinde değilse.
    """
    _validate_schema(main)
    frame = main.select(FRAME_COLUMNS).with_columns(
        [pl.col(name).cast(dtype) for name, dtype in COLUMN_DTYPES.items()]
    )
    return _drop_incomplete_rows(frame).sort(TIMESTAMP_COLUMN)


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
