import logging
from datetime import datetime

import requests
from pydantic import BaseModel, Field
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from city_pulse.config import settings

log = logging.getLogger(__name__)


class TomTomFlowSegment(BaseModel):
    """Tek trafik akış parçasının (flow segment) yanıtı."""

    current_speed_kmh: float | None = Field(default=None, alias="currentSpeed")
    free_flow_speed_kmh: float | None = Field(default=None, alias="freeFlowSpeed")
    current_travel_time_s: float | None = Field(default=None, alias="currentTravelTime")
    free_flow_travel_time_s: float | None = Field(
        default=None, alias="freeFlowTravelTime"
    )
    confidence: float | None = Field(default=None, alias="confidence")
    road_closure: bool | None = Field(default=None, alias="roadClosure")

    model_config = {"populate_by_name": True}


class TomTomFlowResponse(BaseModel):
    flow_segment_data: TomTomFlowSegment | None = Field(
        default=None, alias="flowSegmentData"
    )

    model_config = {"populate_by_name": True}


def make_session(retries: int = 3) -> requests.Session:
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


class TomTomClient:
    def __init__(
        self,
        session: requests.Session | None = None,
        flow_url: str | None = None,
        flow_history_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._session = session or make_session()
        self._flow_url = flow_url or settings.tomtom_flow_url
        self._flow_history_url = flow_history_url or settings.tomtom_flow_history_url
        self._api_key = api_key or settings.tomtom_api_key

    def fetch_flow(
        self,
        lat: float,
        lon: float,
        *,
        timeout: int = 30,
    ) -> TomTomFlowResponse:
        """Belirtilen koordinat için anlık trafik akışını döndürür (current flow)."""
        params = {
            "point": f"{lat},{lon}",
            "key": self._api_key,
            "unit": "KMTH",
            "openLr": "false",
        }
        response = self._session.get(self._flow_url, params=params, timeout=timeout)
        response.raise_for_status()
        return TomTomFlowResponse.model_validate(response.json())

    def fetch_flow_at(
        self,
        lat: float,
        lon: float,
        at: datetime,
        *,
        timeout: int = 30,
    ) -> TomTomFlowResponse:
        """Belirtilen tarihteki trafik akışını döndürür.

        Geçmiş veri için ``TOMTOM_FLOW_HISTORY_URL`` .env'de tanımlanmalıdır.
        Tanımlanmamışsa anlık flow endpoint'i çağrılır (sonuç o anki trafik olur).
        TomTom'ın geçmiş flow endpoint'i timestamp parametresini destekliyorsa,
        bu metot o URL'ye yönlendirilmeli ve ``at`` parametresi eklenmeli.
        """
        url = self._flow_history_url or self._flow_url
        params = {
            "point": f"{lat},{lon}",
            "key": self._api_key,
            "unit": "KMTH",
            "openLr": "false",
        }
        response = self._session.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return TomTomFlowResponse.model_validate(response.json())
