"""Data models for news, sentiment, and market data."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SentimentLabel(str, Enum):
    """Sentiment classification labels."""

    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class DataSource(str, Enum):
    """Data source identifiers."""

    ALPHA_VANTAGE = "alpha_vantage"
    FINNHUB = "finnhub"
    NEWS_API = "newsapi"
    EODHD = "eodhd"
    CUSTOM_NLP = "custom_nlp"
    GROWW = "groww"


class NewsArticle(BaseModel):
    """Represents a news article with metadata."""

    id: str = Field(description="Unique identifier for the article")
    title: str = Field(description="Article headline")
    content: Optional[str] = Field(default=None, description="Full article text")
    summary: Optional[str] = Field(default=None, description="Article summary")
    url: str = Field(description="Source URL")
    source: str = Field(description="News source name")
    published_at: datetime = Field(description="Publication timestamp")
    symbols: list[str] = Field(default_factory=list, description="Related stock symbols")
    source_api: DataSource = Field(description="API that provided this article")

    # Sentiment scores (may be pre-computed by API or calculated later)
    sentiment_score: Optional[float] = Field(
        default=None, ge=-1.0, le=1.0, description="Overall sentiment score (-1 to 1)"
    )
    sentiment_label: Optional[SentimentLabel] = Field(
        default=None, description="Categorical sentiment label"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class SentimentData(BaseModel):
    """Aggregated sentiment data for a stock."""

    symbol: str = Field(description="Stock symbol (e.g., RELIANCE)")
    timestamp: datetime = Field(description="Timestamp of sentiment calculation")

    # Sentiment scores from different sources
    news_sentiment: Optional[float] = Field(
        default=None, ge=-1.0, le=1.0, description="News-based sentiment"
    )
    social_sentiment: Optional[float] = Field(
        default=None, ge=-1.0, le=1.0, description="Social media sentiment"
    )
    combined_sentiment: float = Field(
        ge=-1.0, le=1.0, description="Weighted combined sentiment score"
    )

    # Confidence and metadata
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence in sentiment score"
    )
    article_count: int = Field(default=0, ge=0, description="Number of articles analyzed")
    social_mention_count: int = Field(default=0, ge=0, description="Number of social mentions")

    # Sentiment change metrics
    sentiment_change_1h: Optional[float] = Field(
        default=None, description="Sentiment change in last hour"
    )
    sentiment_change_24h: Optional[float] = Field(
        default=None, description="Sentiment change in last 24 hours"
    )

    # Source breakdown
    sources: list[DataSource] = Field(
        default_factory=list, description="Data sources used"
    )

    @property
    def sentiment_label(self) -> SentimentLabel:
        """Get categorical label based on combined sentiment."""
        score = self.combined_sentiment
        if score <= -0.6:
            return SentimentLabel.VERY_NEGATIVE
        elif score <= -0.2:
            return SentimentLabel.NEGATIVE
        elif score <= 0.2:
            return SentimentLabel.NEUTRAL
        elif score <= 0.6:
            return SentimentLabel.POSITIVE
        else:
            return SentimentLabel.VERY_POSITIVE

    @property
    def is_bullish(self) -> bool:
        """Check if sentiment is bullish."""
        return self.combined_sentiment > 0.2

    @property
    def is_bearish(self) -> bool:
        """Check if sentiment is bearish."""
        return self.combined_sentiment < -0.2


class StockPrice(BaseModel):
    """Stock price data."""

    symbol: str = Field(description="Stock symbol")
    timestamp: datetime = Field(description="Price timestamp")
    open: float = Field(ge=0, description="Opening price")
    high: float = Field(ge=0, description="High price")
    low: float = Field(ge=0, description="Low price")
    close: float = Field(ge=0, description="Closing price")
    volume: int = Field(ge=0, description="Trading volume")
    vwap: Optional[float] = Field(default=None, ge=0, description="Volume weighted avg price")

    # Calculated fields
    change: Optional[float] = Field(default=None, description="Price change")
    change_percent: Optional[float] = Field(default=None, description="Price change percentage")

    @property
    def is_green(self) -> bool:
        """Check if price closed higher than open."""
        return self.close > self.open

    @property
    def range(self) -> float:
        """Get price range (high - low)."""
        return self.high - self.low


class StockQuote(BaseModel):
    """Real-time stock quote."""

    symbol: str
    last_price: float
    change: float
    change_percent: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    volume: int
    timestamp: datetime


class OrderSide(str, Enum):
    """Order side (buy or sell)."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LOSS_MARKET = "stop_loss_market"


class OrderStatus(str, Enum):
    """Order status."""

    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Order(BaseModel):
    """Trading order."""

    id: str = Field(description="Order ID")
    symbol: str = Field(description="Stock symbol")
    side: OrderSide = Field(description="Buy or sell")
    order_type: OrderType = Field(description="Order type")
    quantity: int = Field(ge=1, description="Number of shares")
    price: Optional[float] = Field(default=None, ge=0, description="Limit price")
    stop_price: Optional[float] = Field(default=None, ge=0, description="Stop price")
    status: OrderStatus = Field(default=OrderStatus.PENDING, description="Order status")
    filled_quantity: int = Field(default=0, ge=0, description="Filled quantity")
    average_price: Optional[float] = Field(default=None, ge=0, description="Average fill price")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Sentiment context
    sentiment_score: Optional[float] = Field(
        default=None, description="Sentiment score at time of order"
    )
    signal_reason: Optional[str] = Field(
        default=None, description="Reason for the trading signal"
    )


class Position(BaseModel):
    """Open position in a stock."""

    symbol: str = Field(description="Stock symbol")
    quantity: int = Field(description="Number of shares (positive=long, negative=short)")
    average_cost: float = Field(ge=0, description="Average entry price")
    current_price: float = Field(ge=0, description="Current market price")
    unrealized_pnl: float = Field(description="Unrealized profit/loss")
    unrealized_pnl_percent: float = Field(description="Unrealized P&L percentage")
    market_value: float = Field(ge=0, description="Current market value")
    opened_at: datetime = Field(description="When position was opened")

    # Risk management
    stop_loss: Optional[float] = Field(default=None, ge=0, description="Stop loss price")
    take_profit: Optional[float] = Field(default=None, ge=0, description="Take profit price")

    @property
    def is_long(self) -> bool:
        """Check if this is a long position."""
        return self.quantity > 0

    @property
    def is_profitable(self) -> bool:
        """Check if position is currently profitable."""
        return self.unrealized_pnl > 0
