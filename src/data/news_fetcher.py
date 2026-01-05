"""
News fetching module for collecting financial news from multiple sources.

Supports:
- NewsAPI (newsapi.org)
- Alpha Vantage News Sentiment
- Finnhub Company News
"""

import hashlib
from datetime import datetime, timedelta
from typing import Optional

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from src.data.models import DataSource, NewsArticle


class NewsFetcher:
    """Fetches news articles from multiple sources."""

    BASE_URLS = {
        "newsapi": "https://newsapi.org/v2",
        "alpha_vantage": "https://www.alphavantage.co/query",
        "finnhub": "https://finnhub.io/api/v1",
    }

    # Map of Indian company names to help with news search
    COMPANY_NAMES = {
        "RELIANCE": "Reliance Industries",
        "TCS": "Tata Consultancy Services",
        "HDFCBANK": "HDFC Bank",
        "INFY": "Infosys",
        "ICICIBANK": "ICICI Bank",
        "HINDUNILVR": "Hindustan Unilever",
        "SBIN": "State Bank of India",
        "BHARTIARTL": "Bharti Airtel",
        "KOTAKBANK": "Kotak Mahindra Bank",
        "ITC": "ITC Limited",
        "LT": "Larsen & Toubro",
        "AXISBANK": "Axis Bank",
        "ASIANPAINT": "Asian Paints",
        "MARUTI": "Maruti Suzuki",
        "BAJFINANCE": "Bajaj Finance",
        "WIPRO": "Wipro",
        "HCLTECH": "HCL Technologies",
        "TATAMOTORS": "Tata Motors",
        "SUNPHARMA": "Sun Pharma",
        "TITAN": "Titan Company",
    }

    def __init__(self):
        """Initialize news fetcher with API clients."""
        self._client = httpx.AsyncClient(timeout=30.0)
        self._cache: dict[str, tuple[datetime, list[NewsArticle]]] = {}
        self._cache_ttl = timedelta(minutes=15)

    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()

    def _generate_article_id(self, url: str, title: str) -> str:
        """Generate unique ID for an article."""
        content = f"{url}{title}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _get_company_name(self, symbol: str) -> str:
        """Get company name for a symbol."""
        return self.COMPANY_NAMES.get(symbol.upper(), symbol)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False
        cached_time, _ = self._cache[cache_key]
        return datetime.now() - cached_time < self._cache_ttl

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def fetch_from_newsapi(
        self,
        symbol: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> list[NewsArticle]:
        """
        Fetch news from NewsAPI.

        Args:
            symbol: Stock symbol to search for
            from_date: Start date for news search
            to_date: End date for news search

        Returns:
            List of NewsArticle objects
        """
        api_key = settings.news_api_key
        if not api_key:
            logger.warning("NewsAPI key not configured, skipping")
            return []

        # Build search query with company name
        company_name = self._get_company_name(symbol)
        query = f'"{company_name}" OR "{symbol}" stock'

        # Default to last 24 hours
        if not from_date:
            from_date = datetime.now() - timedelta(hours=24)
        if not to_date:
            to_date = datetime.now()

        cache_key = f"newsapi_{symbol}_{from_date.date()}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key][1]

        params = {
            "q": query,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,
            "apiKey": api_key.get_secret_value(),
        }

        try:
            response = await self._client.get(
                f"{self.BASE_URLS['newsapi']}/everything", params=params
            )
            response.raise_for_status()
            data = response.json()

            articles = []
            for article in data.get("articles", []):
                if not article.get("title") or article["title"] == "[Removed]":
                    continue

                news_article = NewsArticle(
                    id=self._generate_article_id(
                        article.get("url", ""), article.get("title", "")
                    ),
                    title=article.get("title", ""),
                    content=article.get("content"),
                    summary=article.get("description"),
                    url=article.get("url", ""),
                    source=article.get("source", {}).get("name", "Unknown"),
                    published_at=datetime.fromisoformat(
                        article["publishedAt"].replace("Z", "+00:00")
                    ),
                    symbols=[symbol],
                    source_api=DataSource.NEWS_API,
                )
                articles.append(news_article)

            self._cache[cache_key] = (datetime.now(), articles)
            logger.info(f"Fetched {len(articles)} articles from NewsAPI for {symbol}")
            return articles

        except httpx.HTTPError as e:
            logger.error(f"NewsAPI request failed: {e}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def fetch_from_alpha_vantage(
        self,
        symbol: str,
        limit: int = 50,
    ) -> list[NewsArticle]:
        """
        Fetch news with sentiment from Alpha Vantage.

        Args:
            symbol: Stock symbol (will search for related tickers)
            limit: Maximum number of articles to fetch

        Returns:
            List of NewsArticle objects with pre-computed sentiment
        """
        api_key = settings.alpha_vantage_api_key
        if not api_key:
            logger.warning("Alpha Vantage API key not configured, skipping")
            return []

        cache_key = f"alphavantage_{symbol}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key][1]

        # Alpha Vantage uses different ticker format
        # For Indian stocks, we search by company name
        company_name = self._get_company_name(symbol)

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,  # Try direct symbol first
            "topics": "earnings,financial_markets",
            "limit": limit,
            "apikey": api_key.get_secret_value(),
        }

        try:
            response = await self._client.get(
                self.BASE_URLS["alpha_vantage"], params=params
            )
            response.raise_for_status()
            data = response.json()

            # Check for API limit message
            if "Note" in data or "Information" in data:
                logger.warning(f"Alpha Vantage API limit: {data.get('Note', data.get('Information'))}")
                return []

            articles = []
            for item in data.get("feed", []):
                # Parse sentiment score
                sentiment_score = None
                if "overall_sentiment_score" in item:
                    sentiment_score = float(item["overall_sentiment_score"])

                # Get ticker-specific sentiment if available
                ticker_sentiment = item.get("ticker_sentiment", [])
                for ts in ticker_sentiment:
                    if ts.get("ticker", "").upper() == symbol.upper():
                        sentiment_score = float(ts.get("ticker_sentiment_score", sentiment_score or 0))
                        break

                news_article = NewsArticle(
                    id=self._generate_article_id(
                        item.get("url", ""), item.get("title", "")
                    ),
                    title=item.get("title", ""),
                    summary=item.get("summary"),
                    url=item.get("url", ""),
                    source=item.get("source", "Unknown"),
                    published_at=datetime.strptime(
                        item["time_published"], "%Y%m%dT%H%M%S"
                    ),
                    symbols=[symbol],
                    source_api=DataSource.ALPHA_VANTAGE,
                    sentiment_score=sentiment_score,
                )
                articles.append(news_article)

            self._cache[cache_key] = (datetime.now(), articles)
            logger.info(f"Fetched {len(articles)} articles from Alpha Vantage for {symbol}")
            return articles

        except httpx.HTTPError as e:
            logger.error(f"Alpha Vantage request failed: {e}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def fetch_from_finnhub(
        self,
        symbol: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> list[NewsArticle]:
        """
        Fetch company news from Finnhub.

        Args:
            symbol: Stock symbol
            from_date: Start date
            to_date: End date

        Returns:
            List of NewsArticle objects
        """
        api_key = settings.finnhub_api_key
        if not api_key:
            logger.warning("Finnhub API key not configured, skipping")
            return []

        if not from_date:
            from_date = datetime.now() - timedelta(days=7)
        if not to_date:
            to_date = datetime.now()

        cache_key = f"finnhub_{symbol}_{from_date.date()}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key][1]

        # Finnhub primarily supports US stocks, but we can try
        # For Indian stocks, general market news might be available
        params = {
            "symbol": symbol,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
            "token": api_key.get_secret_value(),
        }

        try:
            response = await self._client.get(
                f"{self.BASE_URLS['finnhub']}/company-news", params=params
            )
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data:
                news_article = NewsArticle(
                    id=self._generate_article_id(
                        item.get("url", ""), item.get("headline", "")
                    ),
                    title=item.get("headline", ""),
                    summary=item.get("summary"),
                    url=item.get("url", ""),
                    source=item.get("source", "Unknown"),
                    published_at=datetime.fromtimestamp(item.get("datetime", 0)),
                    symbols=[symbol],
                    source_api=DataSource.FINNHUB,
                )
                articles.append(news_article)

            self._cache[cache_key] = (datetime.now(), articles)
            logger.info(f"Fetched {len(articles)} articles from Finnhub for {symbol}")
            return articles

        except httpx.HTTPError as e:
            logger.error(f"Finnhub request failed: {e}")
            return []

    async def fetch_all_sources(
        self,
        symbol: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> list[NewsArticle]:
        """
        Fetch news from all configured sources.

        Args:
            symbol: Stock symbol
            from_date: Start date
            to_date: End date

        Returns:
            Combined list of NewsArticle objects from all sources
        """
        all_articles = []

        # Fetch from all sources concurrently would be better,
        # but sequential is safer for rate limits
        try:
            articles = await self.fetch_from_newsapi(symbol, from_date, to_date)
            all_articles.extend(articles)
        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {e}")

        try:
            articles = await self.fetch_from_alpha_vantage(symbol)
            all_articles.extend(articles)
        except Exception as e:
            logger.error(f"Error fetching from Alpha Vantage: {e}")

        try:
            articles = await self.fetch_from_finnhub(symbol, from_date, to_date)
            all_articles.extend(articles)
        except Exception as e:
            logger.error(f"Error fetching from Finnhub: {e}")

        # Remove duplicates based on title similarity
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title_key = article.title.lower()[:50]  # First 50 chars
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)

        # Sort by published date (newest first)
        unique_articles.sort(key=lambda x: x.published_at, reverse=True)

        logger.info(
            f"Total {len(unique_articles)} unique articles for {symbol} "
            f"from {len(all_articles)} total fetched"
        )

        return unique_articles
