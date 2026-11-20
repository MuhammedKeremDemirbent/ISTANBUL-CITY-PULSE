from dataclasses import dataclass

# Timezone
ISTANBUL_TIMEZONE: str = "Europe/Istanbul"

# Weather Variables
WEATHER_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
)
# Backward compatibility alias
VARIABLES = WEATHER_VARIABLES


@dataclass(frozen=True, slots=True)
class WeatherPoint:
    point_id: str
    lat: float
    lon: float


ISTANBUL_CENTER = WeatherPoint("istanbul_center", 41.01, 28.98)


# Traffic Index (TI) Column Mappings and Constants
TRAFFIC_COLUMNS: dict[str, str] = {
    "TI": "traffic_index",  # City-wide Traffic Index (0-100)
    "TI_An": "traffic_index_an",  # Asian Side Traffic Index
    "TI_Av": "traffic_index_av",  # European Side Traffic Index
}

TRAFFIC_VARIABLES: tuple[str, ...] = tuple(TRAFFIC_COLUMNS.values())


@dataclass(frozen=True, slots=True)
class TrafficRegion:
    code: str
    name: str
    column: str


TRAFFIC_REGIONS: tuple[TrafficRegion, ...] = (
    TrafficRegion("all", "Istanbul (All)", "traffic_index"),
    TrafficRegion("anadolu", "Asian Side", "traffic_index_an"),
    TrafficRegion("avrupa", "European Side", "traffic_index_av"),
)

# Machine Learning Input Features and Target
FEATURE_COLUMNS: tuple[str, ...] = (
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "temperature_2m",
    "precipitation",
    "is_raining",
    "heavy_rain",
    "wind_speed_10m",
    "relative_humidity_2m",
    "traffic_index_av_lag_1h",
    "traffic_index_av_lag_24h",
    "is_school_term",
    "is_school_morning_rush",
    "is_school_afternoon_rush",
)

TARGET_COLUMN: str = "traffic_index_av"
