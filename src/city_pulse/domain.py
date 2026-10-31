"""Alan modelleri ve sabitler: bağımlılığı olmayan en alt katman."""

from dataclasses import dataclass

VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
)


@dataclass(frozen=True, slots=True)
class WeatherPoint:
    point_id: str
    lat: float
    lon: float


ISTANBUL_CENTER = WeatherPoint("istanbul_center", 41.01, 28.98)
