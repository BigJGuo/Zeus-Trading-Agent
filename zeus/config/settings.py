from functools import lru_cache
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Alpaca
    alpaca_api_key: str = Field(..., description="Alpaca API key")
    alpaca_secret_key: str = Field(..., description="Alpaca secret key")
    alpaca_paper: bool = Field(True, description="Use paper trading environment")

    # Database
    db_password: str = Field(..., description="PostgreSQL password")
    database_url: str = Field(..., description="Full PostgreSQL connection URL")

    # Redis
    redis_url: str = Field("redis://localhost:6379/0", description="Redis URL")

    # Telegram (optional — system runs in no-op observability mode without these)
    telegram_bot_token: Optional[str] = Field(None, description="Telegram bot token")
    telegram_chat_id: Optional[str] = Field(None, description="Telegram chat ID for alerts")

    # FRED
    fred_api_key: Optional[str] = Field(None, description="FRED API key for macro data")

    # System
    environment: str = Field("paper", description="paper or live")
    log_level: str = Field("INFO", description="Logging level")
    artifacts_path: str = Field("./artifacts", description="Path to artifacts directory")
    logs_path: str = Field("./logs", description="Path to logs directory")

    # Anthropic (LLM reasoning for 7-agent system). Optional — agents only
    # spin up when this is present; otherwise Zeus runs in its single-model path.
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    llm_model_opus: str = Field("claude-opus-4-7", description="Opus tier model id")
    llm_model_sonnet: str = Field("claude-sonnet-4-6", description="Sonnet tier model id")
    llm_model_haiku: str = Field("claude-haiku-4-5-20251001", description="Haiku tier model id")
    llm_daily_budget_usd_per_agent: float = Field(15.0, description="Hard daily LLM-spend cap per agent")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("environment")
    def validate_environment(cls, v: str) -> str:
        allowed = {"paper", "live", "backtest"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v

    def is_paper(self) -> bool:
        return self.alpaca_paper or self.environment == "paper"

    def is_live(self) -> bool:
        return not self.alpaca_paper and self.environment == "live"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Required fields are populated by pydantic-settings from .env / env vars
    # at runtime — Pylance can't model that, so silence the "missing args".
    return Settings()  # type: ignore[call-arg]
