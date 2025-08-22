from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    alpaca_api_key: SecretStr
    alpaca_api_secret: SecretStr
    paper_trading: bool = True
    max_position_size: float = Field(..., gt=0, lt=1)
    baseline_exposure: float = Field(..., ge=0, le=1)
    max_exposure: float = Field(..., ge=0, le=1)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


__all__ = ["Settings"]
