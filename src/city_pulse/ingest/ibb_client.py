from typing import Final

import requests
from pydantic import BaseModel, ConfigDict, Field
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

IBB_TRAFFIC_INDEX_URL: Final[str] = (
    "https://tkmservices.ibb.gov.tr/web/api/TrafficData/v1/TrafficIndex_Sc1_Cont"
)


class IbbTrafficResponse(BaseModel):
    """IBB Traffic Control Center (TKM) live traffic index response model."""

    model_config = ConfigDict(populate_by_name=True)

    traffic_index: int = Field(
        alias="TI", description="City-wide traffic congestion index (%)"
    )
    traffic_index_an: int = Field(
        alias="TI_An", description="Asian side traffic index (%)"
    )
    traffic_index_av: int = Field(
        alias="TI_Av", description="European side traffic index (%)"
    )


def make_session(retries: int = 3) -> requests.Session:
    """Create HTTP session with retry and backoff strategy."""
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


class IbbTrafficClient:
    """Client for IBB TKM Traffic Index API."""

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or make_session()

    def fetch_traffic_index(self, timeout: int = 15) -> IbbTrafficResponse:
        """Fetch and validate live traffic index from IBB TKM."""
        response = self._session.get(IBB_TRAFFIC_INDEX_URL, timeout=timeout)
        response.raise_for_status()
        return IbbTrafficResponse.model_validate(response.json())
