from datetime import datetime

import polars as pl

from city_pulse.ingest.tomtom_client import TomTomFlowResponse


def flow_to_frame(
    response: TomTomFlowResponse,
    point_id: str,
    lat: float,
    lon: float,
    fetched_at: datetime,
) -> pl.DataFrame:
    """TomTom flowSegmentData yanıtını tek-satırlık silver çerçevesine dönüştürür."""
    seg = response.flow_segment_data
    row = {
        "point_id": point_id,
        "latitude": lat,
        "longitude": lon,
        "fetched_at_utc": fetched_at,
        "source": "tomtom",
        "current_speed_kmh": seg.current_speed_kmh if seg else None,
        "free_flow_speed_kmh": seg.free_flow_speed_kmh if seg else None,
        "current_travel_time_s": seg.current_travel_time_s if seg else None,
        "free_flow_travel_time_s": seg.free_flow_travel_time_s if seg else None,
        "confidence": seg.confidence if seg else None,
        "road_closure": seg.road_closure if seg else None,
    }
    df = pl.DataFrame([row])
    numeric_cols = [
        "current_speed_kmh",
        "free_flow_speed_kmh",
        "current_travel_time_s",
        "free_flow_travel_time_s",
        "confidence",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64))
    return df
