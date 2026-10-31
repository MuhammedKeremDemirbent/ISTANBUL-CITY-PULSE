#İnternetten gelen veriyi alır.

import requests #Veri Çeker
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from city_pulse.domain import VARIABLES, WeatherPoint

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/era5" #Geçmiş
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"    #Anlık ve Gelecek


def make_session(retries: int = 3) -> requests.Session:  #Bağlantı ve Hata Tölare etme
    retry = Retry(
        total=retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class OpenMeteoClient:
    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or make_session()

    def fetch_archive(self, point: WeatherPoint, start: str, end: str) -> dict:
        params = {
            "latitude": point.lat,
            "longitude": point.lon,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(VARIABLES),
            "timezone": "UTC",
        }
        return self._get(ARCHIVE_URL, params, timeout=60)

    def fetch_forecast(self, point: WeatherPoint, forecast_days: int = 2) -> dict:
        params = {
            "latitude": point.lat,
            "longitude": point.lon,
            "current": ",".join(VARIABLES),
            "hourly": ",".join(VARIABLES),
            "forecast_days": forecast_days,
            "timezone": "UTC",
        }
        return self._get(FORECAST_URL, params, timeout=30)

    def _get(self, url: str, params: dict, timeout: int) -> dict:
        response = self._session.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
