"""Data ingestion module for fetching news, sentiment, and market data."""

from src.data.models import NewsArticle, SentimentData, StockPrice
from src.data.news_fetcher import NewsFetcher
from src.data.sentiment_apis import SentimentAPIClient
from src.data.market_data import MarketDataClient

__all__ = [
    "NewsArticle",
    "SentimentData",
    "StockPrice",
    "NewsFetcher",
    "SentimentAPIClient",
    "MarketDataClient",
]
