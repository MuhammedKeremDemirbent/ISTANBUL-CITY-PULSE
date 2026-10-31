"""Alan modelleri ve sabitler: bağımlılığı olmayan en alt katman."""

from dataclasses import dataclass

# Zaman Dilimi
ISTANBUL_TIMEZONE: str = "Europe/Istanbul"

# Hava Durumu Değişkenleri
WEATHER_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
)
# Geriye dönük uyumluluk için alias
VARIABLES = WEATHER_VARIABLES


@dataclass(frozen=True, slots=True)
class WeatherPoint:
    point_id: str
    lat: float
    lon: float


ISTANBUL_CENTER = WeatherPoint("istanbul_center", 41.01, 28.98)


# Trafik İndeksi (TI) Kolon Eşlemeleri ve Sabitleri
TRAFFIC_COLUMNS: dict[str, str] = {
    "TI": "traffic_index",  # İstanbul Geneli Trafik İndeksi (0-100)
    "TI_An": "traffic_index_an",  # Anadolu Yakası Trafik İndeksi
    "TI_Av": "traffic_index_av",  # Avrupa Yakası Trafik İndeksi
}

TRAFFIC_VARIABLES: tuple[str, ...] = tuple(TRAFFIC_COLUMNS.values())


@dataclass(frozen=True, slots=True)
class TrafficRegion:
    code: str
    name: str
    column: str


TRAFFIC_REGIONS: tuple[TrafficRegion, ...] = (
    TrafficRegion("all", "İstanbul Geneli", "traffic_index"),
    TrafficRegion("anadolu", "Anadolu Yakası", "traffic_index_an"),
    TrafficRegion("avrupa", "Avrupa Yakası", "traffic_index_av"),
)

# Makine Öğrenimi için Girdi Özellikleri ve Target
FEATURE_COLUMNS: tuple[str, ...] = (
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "relative_humidity_2m",
)

TARGET_COLUMN: str = "traffic_index_av"