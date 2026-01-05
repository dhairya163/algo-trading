"""
Sentiment API client for fetching pre-computed sentiment scores.

Supports:
- Alpha Vantage News Sentiment
- Finnhub Social Sentiment (Reddit, Twitter)
- EODHD Sentiment Data
"""

from datetime import datetime, timedelta
from typing import Optional

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from src.data.models import DataSource, SentimentData


class SentimentAPIClient:
    """Client for fetching sentiment data from various APIs."""

    BASE_URLS = {
        "alpha_vantage": "https://www.alphavantage.co/query",
        "finnhub": "https://finnhub.io/api/v1",
        "eodhd": "https://eodhd.com/api",
    }

    def __init__(self):
        """Initialize sentiment API client."""
        self._client = httpx.AsyncClient(timeout=30.0)
        self._cache: dict[str, tuple[datetime, SentimentData]] = {}
        self._cache_ttl = timedelta(minutes=10)

    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False
        cached_time, _ = self._cache[cache_key]
        return datetime.now() - cached_time < self._cache_ttl

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_alpha_vantage_sentiment(
        self,
        symbol: str,
    ) -> Optional[dict]:
        """
        Get sentiment data from Alpha Vantage.

        Returns aggregated sentiment from recent news articles.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with sentiment scores and metadata
        """
        api_key = settings.alpha_vantage_api_key
        if not api_key:
            logger.warning("Alpha Vantage API key not configured")
            return None

        cache_key = f"av_sentiment_{symbol}"
        if self._is_cache_valid(cache_key):
            _, cached = self._cache[cache_key]
            return {
                "score": cached.news_sentiment,
                "article_count": cached.article_count,
            }

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "limit": 50,
            "apikey": api_key.get_secret_value(),
        }

        try:
            response = await self._client.get(
                self.BASE_URLS["alpha_vantage"], params=params
            )
            response.raise_for_status()
            data = response.json()

            if "Note" in data or "Information" in data:
                logger.warning(f"Alpha Vantage API limit reached")
                return None

            # Calculate average sentiment from articles
            feed = data.get("feed", [])
            if not feed:
                return {"score": 0.0, "article_count": 0}

            total_sentiment = 0.0
            count = 0

            for article in feed:
                # Get overall sentiment
                if "overall_sentiment_score" in article:
                    total_sentiment += float(article["overall_sentiment_score"])
                    count += 1

                # Or get ticker-specific sentiment
                for ts in article.get("ticker_sentiment", []):
                    if ts.get("ticker", "").upper() == symbol.upper():
                        total_sentiment += float(ts.get("ticker_sentiment_score", 0))
                        count += 1
                        break

            avg_sentiment = total_sentiment / count if count > 0 else 0.0

            result = {
                "score": avg_sentiment,
                "article_count": len(feed),
                "source": DataSource.ALPHA_VANTAGE,
            }

            logger.info(
                f"Alpha Vantage sentiment for {symbol}: {avg_sentiment:.3f} "
                f"from {len(feed)} articles"
            )

            return result

        except httpx.HTTPError as e:
            logger.error(f"Alpha Vantage sentiment request failed: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_finnhub_social_sentiment(
        self,
        symbol: str,
    ) -> Optional[dict]:
        """
        Get social media sentiment from Finnhub.

        Aggregates sentiment from Reddit and Twitter.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with social sentiment scores
        """
        api_key = settings.finnhub_api_key
        if not api_key:
            logger.warning("Finnhub API key not configured")
            return None

        cache_key = f"finnhub_social_{symbol}"
        if self._is_cache_valid(cache_key):
            _, cached = self._cache[cache_key]
            return {
                "score": cached.social_sentiment,
                "mention_count": cached.social_mention_count,
            }

        # Finnhub social sentiment endpoint
        params = {
            "symbol": symbol,
            "token": api_key.get_secret_value(),
        }

        try:
            response = await self._client.get(
                f"{self.BASE_URLS['finnhub']}/stock/social-sentiment", params=params
            )
            response.raise_for_status()
            data = response.json()

            # Aggregate Reddit and Twitter sentiment
            reddit_data = data.get("reddit", [])
            twitter_data = data.get("twitter", [])

            total_score = 0.0
            total_mentions = 0

            # Process Reddit sentiment
            for item in reddit_data:
                score = item.get("score", 0)
                mentions = item.get("mention", 0)
                if mentions > 0:
                    total_score += score * mentions
                    total_mentions += mentions

            # Process Twitter sentiment
            for item in twitter_data:
                score = item.get("score", 0)
                mentions = item.get("mention", 0)
                if mentions > 0:
                    total_score += score * mentions
                    total_mentions += mentions

            avg_score = total_score / total_mentions if total_mentions > 0 else 0.0

            result = {
                "score": avg_score,
                "mention_count": total_mentions,
                "reddit_mentions": sum(r.get("mention", 0) for r in reddit_data),
                "twitter_mentions": sum(t.get("mention", 0) for t in twitter_data),
                "source": DataSource.FINNHUB,
            }

            logger.info(
                f"Finnhub social sentiment for {symbol}: {avg_score:.3f} "
                f"from {total_mentions} mentions"
            )

            return result

        except httpx.HTTPError as e:
            logger.error(f"Finnhub social sentiment request failed: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_eodhd_sentiment(
        self,
        symbol: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> Optional[dict]:
        """
        Get sentiment data from EODHD.

        Args:
            symbol: Stock symbol
            from_date: Start date
            to_date: End date

        Returns:
            Dictionary with sentiment scores
        """
        api_key = settings.eodhd_api_key
        if not api_key:
            logger.debug("EODHD API key not configured")
            return None

        if not from_date:
            from_date = datetime.now() - timedelta(days=7)
        if not to_date:
            to_date = datetime.now()

        # EODHD uses exchange suffix for Indian stocks
        eodhd_symbol = f"{symbol}.NSE"

        params = {
            "s": eodhd_symbol,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
            "api_token": api_key.get_secret_value(),
        }

        try:
            response = await self._client.get(
                f"{self.BASE_URLS['eodhd']}/sentiments", params=params
            )
            response.raise_for_status()
            data = response.json()

            if not data or eodhd_symbol not in data:
                return None

            sentiment_data = data[eodhd_symbol]

            # Calculate average sentiment
            total_sentiment = 0.0
            count = 0
            for item in sentiment_data:
                if "normalized" in item:
                    total_sentiment += item["normalized"]
                    count += 1

            avg_sentiment = total_sentiment / count if count > 0 else 0.0

            result = {
                "score": avg_sentiment,
                "data_points": count,
                "source": DataSource.EODHD,
            }

            logger.info(f"EODHD sentiment for {symbol}: {avg_sentiment:.3f}")

            return result

        except httpx.HTTPError as e:
            logger.error(f"EODHD sentiment request failed: {e}")
            return None

    async def get_combined_sentiment(
        self,
        symbol: str,
        news_weight: float = 0.6,
        social_weight: float = 0.4,
    ) -> SentimentData:
        """
        Get combined sentiment from all available sources.

        Args:
            symbol: Stock symbol
            news_weight: Weight for news sentiment (0-1)
            social_weight: Weight for social sentiment (0-1)

        Returns:
            SentimentData with aggregated sentiment scores
        """
        # Normalize weights
        total_weight = news_weight + social_weight
        news_weight = news_weight / total_weight
        social_weight = social_weight / total_weight

        # Fetch from all sources
        av_sentiment = await self.get_alpha_vantage_sentiment(symbol)
        finnhub_sentiment = await self.get_finnhub_social_sentiment(symbol)
        eodhd_sentiment = await self.get_eodhd_sentiment(symbol)

        # Calculate weighted sentiment
        news_score = 0.0
        social_score = 0.0
        article_count = 0
        mention_count = 0
        sources = []

        # News sentiment (Alpha Vantage + EODHD)
        news_scores = []
        if av_sentiment and av_sentiment.get("score") is not None:
            news_scores.append(av_sentiment["score"])
            article_count += av_sentiment.get("article_count", 0)
            sources.append(DataSource.ALPHA_VANTAGE)

        if eodhd_sentiment and eodhd_sentiment.get("score") is not None:
            news_scores.append(eodhd_sentiment["score"])
            sources.append(DataSource.EODHD)

        if news_scores:
            news_score = sum(news_scores) / len(news_scores)

        # Social sentiment (Finnhub)
        if finnhub_sentiment and finnhub_sentiment.get("score") is not None:
            social_score = finnhub_sentiment["score"]
            mention_count = finnhub_sentiment.get("mention_count", 0)
            sources.append(DataSource.FINNHUB)

        # Calculate combined score
        if news_scores and finnhub_sentiment:
            combined_score = (news_score * news_weight) + (social_score * social_weight)
        elif news_scores:
            combined_score = news_score
        elif finnhub_sentiment:
            combined_score = social_score
        else:
            combined_score = 0.0

        # Calculate confidence based on data availability
        confidence = 0.0
        if news_scores:
            confidence += 0.4 * min(article_count / 10, 1.0)  # Max 0.4 for news
        if finnhub_sentiment:
            confidence += 0.3 * min(mention_count / 50, 1.0)  # Max 0.3 for social
        if len(sources) > 1:
            confidence += 0.3  # Bonus for multiple sources

        sentiment_data = SentimentData(
            symbol=symbol,
            timestamp=datetime.now(),
            news_sentiment=news_score if news_scores else None,
            social_sentiment=social_score if finnhub_sentiment else None,
            combined_sentiment=combined_score,
            confidence=min(confidence, 1.0),
            article_count=article_count,
            social_mention_count=mention_count,
            sources=sources,
        )

        # Cache the result
        cache_key = f"combined_{symbol}"
        self._cache[cache_key] = (datetime.now(), sentiment_data)

        logger.info(
            f"Combined sentiment for {symbol}: {combined_score:.3f} "
            f"(news: {news_score:.3f}, social: {social_score:.3f}, confidence: {confidence:.2f})"
        )

        return sentiment_data
