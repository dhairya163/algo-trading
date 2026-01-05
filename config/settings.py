"""
Configuration management for Sentiment Algo Trader.

Uses Pydantic for type-safe configuration with environment variable support.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingMode(str, Enum):
    """Trading execution mode."""

    PAPER = "paper"  # Simulated trading (no real money)
    LIVE = "live"  # Real money trading
    BACKTEST = "backtest"  # Historical backtesting


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Groww API Configuration
    # -------------------------------------------------------------------------
    groww_api_key: Optional[SecretStr] = Field(default=None, description="Groww API Key")
    groww_api_secret: Optional[SecretStr] = Field(default=None, description="Groww API Secret")

    # -------------------------------------------------------------------------
    # Sentiment Data Source API Keys
    # -------------------------------------------------------------------------
    alpha_vantage_api_key: Optional[SecretStr] = Field(
        default=None, description="Alpha Vantage API Key for news sentiment"
    )
    finnhub_api_key: Optional[SecretStr] = Field(
        default=None, description="Finnhub API Key for social sentiment"
    )
    news_api_key: Optional[SecretStr] = Field(
        default=None, description="NewsAPI Key for news articles"
    )
    eodhd_api_key: Optional[SecretStr] = Field(
        default=None, description="EODHD API Key for premium sentiment data"
    )

    # -------------------------------------------------------------------------
    # Trading Configuration
    # -------------------------------------------------------------------------
    trading_mode: TradingMode = Field(
        default=TradingMode.PAPER,
        description="Trading mode: paper, live, or backtest",
    )
    initial_capital: float = Field(
        default=100000.0,
        ge=1000.0,
        description="Initial trading capital in INR",
    )

    # -------------------------------------------------------------------------
    # Risk Management Parameters
    # -------------------------------------------------------------------------
    max_position_size_percent: float = Field(
        default=10.0,
        ge=1.0,
        le=100.0,
        description="Maximum single position size as % of portfolio",
    )
    max_portfolio_risk_percent: float = Field(
        default=2.0,
        ge=0.1,
        le=10.0,
        description="Maximum portfolio risk per trade as % of capital",
    )
    stop_loss_percent: float = Field(
        default=3.0,
        ge=0.5,
        le=20.0,
        description="Default stop loss percentage",
    )
    take_profit_percent: float = Field(
        default=6.0,
        ge=1.0,
        le=50.0,
        description="Default take profit percentage",
    )
    max_daily_loss_percent: float = Field(
        default=5.0,
        ge=1.0,
        le=20.0,
        description="Maximum daily loss as % of capital (circuit breaker)",
    )
    max_open_positions: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of concurrent open positions",
    )

    # -------------------------------------------------------------------------
    # Sentiment Analysis Configuration
    # -------------------------------------------------------------------------
    sentiment_lookback_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Hours of sentiment data to consider",
    )
    min_sentiment_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum sentiment score to trigger buy signal",
    )
    max_sentiment_score: float = Field(
        default=-0.3,
        ge=-1.0,
        le=0.0,
        description="Maximum (negative) sentiment score to trigger sell signal",
    )
    sentiment_smoothing_window: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Moving average window for sentiment smoothing",
    )

    # -------------------------------------------------------------------------
    # Stock Universe Configuration
    # -------------------------------------------------------------------------
    default_watchlist: list[str] = Field(
        default=[
            "RELIANCE",
            "TCS",
            "HDFCBANK",
            "INFY",
            "ICICIBANK",
            "HINDUNILVR",
            "SBIN",
            "BHARTIARTL",
            "KOTAKBANK",
            "ITC",
            "LT",
            "AXISBANK",
            "ASIANPAINT",
            "MARUTI",
            "BAJFINANCE",
            "WIPRO",
            "HCLTECH",
            "TATAMOTORS",
            "SUNPHARMA",
            "TITAN",
        ],
        description="Default list of stocks to monitor (NSE symbols)",
    )

    # -------------------------------------------------------------------------
    # Database Configuration
    # -------------------------------------------------------------------------
    database_url: str = Field(
        default="sqlite:///./data/trading.db",
        description="Database connection URL",
    )

    # -------------------------------------------------------------------------
    # Notification Configuration
    # -------------------------------------------------------------------------
    telegram_bot_token: Optional[SecretStr] = Field(
        default=None, description="Telegram bot token for notifications"
    )
    telegram_chat_id: Optional[str] = Field(
        default=None, description="Telegram chat ID for notifications"
    )
    smtp_host: str = Field(default="smtp.gmail.com", description="SMTP server host")
    smtp_port: int = Field(default=587, description="SMTP server port")
    smtp_user: Optional[str] = Field(default=None, description="SMTP username")
    smtp_password: Optional[SecretStr] = Field(default=None, description="SMTP password")
    notification_email: Optional[str] = Field(
        default=None, description="Email for notifications"
    )

    # -------------------------------------------------------------------------
    # Dashboard Configuration
    # -------------------------------------------------------------------------
    dashboard_host: str = Field(default="0.0.0.0", description="Dashboard server host")
    dashboard_port: int = Field(default=8000, description="Dashboard server port")
    secret_key: SecretStr = Field(
        default=SecretStr("change-me-in-production"),
        description="Secret key for JWT tokens",
    )

    # -------------------------------------------------------------------------
    # Logging Configuration
    # -------------------------------------------------------------------------
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    log_file: Path = Field(default=Path("logs/trading.log"), description="Log file path")

    # -------------------------------------------------------------------------
    # Scheduling Configuration
    # -------------------------------------------------------------------------
    market_open_time: str = Field(
        default="09:15", description="Market opening time (IST)"
    )
    market_close_time: str = Field(
        default="15:30", description="Market closing time (IST)"
    )
    sentiment_fetch_interval_minutes: int = Field(
        default=15,
        ge=5,
        le=60,
        description="Interval for fetching sentiment data",
    )
    signal_check_interval_minutes: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Interval for checking trading signals",
    )

    @field_validator("default_watchlist", mode="before")
    @classmethod
    def parse_watchlist(cls, v):
        """Parse watchlist from comma-separated string if needed."""
        if isinstance(v, str):
            return [s.strip().upper() for s in v.split(",") if s.strip()]
        return [s.upper() for s in v] if v else []

    @property
    def is_live_trading(self) -> bool:
        """Check if live trading is enabled."""
        return self.trading_mode == TradingMode.LIVE

    @property
    def is_paper_trading(self) -> bool:
        """Check if paper trading is enabled."""
        return self.trading_mode == TradingMode.PAPER

    def validate_live_trading_requirements(self) -> list[str]:
        """Validate that all required settings for live trading are configured."""
        errors = []
        if not self.groww_api_key:
            errors.append("GROWW_API_KEY is required for live trading")
        if not self.groww_api_secret:
            errors.append("GROWW_API_SECRET is required for live trading")
        return errors


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience function for accessing settings
settings = get_settings()
