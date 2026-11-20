from datetime import UTC, datetime, timedelta

import polars as pl

from city_pulse.ingest.ibb_client import IbbTrafficResponse

# Istanbul UTC offset (+03:00, no DST)
_ISTANBUL_OFFSET = timedelta(hours=3)


def ibb_traffic_to_frame(
    response: IbbTrafficResponse,
    fetched_at: datetime,
) -> pl.DataFrame:
    """Convert live IBB traffic response to a clean Silver Polars DataFrame.

    Timestamp is stored as tz-naive Istanbul local time.
    fetched_at_utc preserves original UTC fetch time for lineage.
    """
    if fetched_at.tzinfo is not None:
        local_dt = fetched_at.astimezone(UTC) + _ISTANBUL_OFFSET
    else:
        local_dt = fetched_at + _ISTANBUL_OFFSET

    # Round to top of the hour and strip tzinfo
    hour_timestamp = datetime(
        local_dt.year,
        local_dt.month,
        local_dt.day,
        local_dt.hour,
    )

    row = {
        "timestamp": hour_timestamp,
        "fetched_at_utc": fetched_at,
        "source": "ibb_tkm",
        "traffic_index": float(response.traffic_index),
        "traffic_index_an": float(response.traffic_index_an),
        "traffic_index_av": float(response.traffic_index_av),
    }

    df = pl.DataFrame([row])
    return df.with_columns(
        pl.col("traffic_index").cast(pl.Float64),
        pl.col("traffic_index_an").cast(pl.Float64),
        pl.col("traffic_index_av").cast(pl.Float64),
    )
