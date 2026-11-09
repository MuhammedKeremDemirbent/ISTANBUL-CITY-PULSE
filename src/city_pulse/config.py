from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # TomTom API anahtarı (.env: IBB-TRAFFIC-KEY)
    tomtom_api_key: str = Field(default="", validation_alias="IBB-TRAFFIC-KEY")
    tomtom_base_url: str = "https://api.tomtom.com"

    # TomTom trafik akış (flow) uç noktaları — JSON döndüren, sayı içeren endpointler.
    # Üretimde TomTom geliştirici dökümantasyonundan doğru path onaylanmalı.
    tomtom_flow_url: str = Field(
        default="https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json",
        validation_alias="TOMTOM_FLOW_URL",
    )
    tomtom_flow_history_url: str | None = Field(
        default=None,
        validation_alias="TOMTOM_FLOW_HISTORY_URL",
    )


settings = AppSettings()
