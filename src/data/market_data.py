"""
Market data client for fetching stock prices and quotes.

Supports:
- Groww API (primary for Indian markets)
- Yahoo Finance (backup/historical data)
"""

from datetime import datetime, timedelta
from typing import Optional

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from src.data.models import StockPrice, StockQuote


class MarketDataClient:
    """Client for fetching market data from various sources."""

    # NSE stock symbols with exchange suffix for different APIs
    EXCHANGE_SUFFIXES = {
        "yahoo": ".NS",  # Yahoo Finance NSE suffix
        "groww": "",  # Groww uses plain symbols
    }

    def __init__(self):
        """Initialize market data client."""
        self._client = httpx.AsyncClient(timeout=30.0)
        self._quote_cache: dict[str, tuple[datetime, StockQuote]] = {}
        self._cache_ttl = timedelta(seconds=30)  # Short TTL for quotes
        self._groww_api = None

    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._quote_cache:
            return False
        cached_time, _ = self._quote_cache[cache_key]
        return datetime.now() - cached_time < self._cache_ttl

    async def _init_groww_api(self):
        """Initialize Groww API client if credentials available."""
        if self._groww_api is not None:
            return

        api_key = settings.groww_api_key
        if not api_key:
            logger.warning("Groww API key not configured")
            return

        try:
            # Import growwapi dynamically
            from growwapi import GrowwAPI

            self._groww_api = GrowwAPI(api_key=api_key.get_secret_value())
            logger.info("Groww API client initialized")
        except ImportError:
            logger.warning("growwapi package not installed")
        except Exception as e:
            logger.error(f"Failed to initialize Groww API: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_quote_yahoo(self, symbol: str) -> Optional[StockQuote]:
        """
        Get real-time quote from Yahoo Finance.

        Args:
            symbol: Stock symbol (NSE)

        Returns:
            StockQuote object or None
        """
        yahoo_symbol = f"{symbol}{self.EXCHANGE_SUFFIXES['yahoo']}"

        cache_key = f"yahoo_quote_{symbol}"
        if self._is_cache_valid(cache_key):
            return self._quote_cache[cache_key][1]

        # Yahoo Finance API endpoint
        url = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

        try:
            response = await self._client.get(
                url.format(symbol=yahoo_symbol),
                params={"interval": "1m", "range": "1d"},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            data = response.json()

            result = data.get("chart", {}).get("result", [])
            if not result:
                return None

            meta = result[0].get("meta", {})
            quote_data = result[0].get("indicators", {}).get("quote", [{}])[0]

            # Get latest values
            timestamps = result[0].get("timestamp", [])
            if not timestamps:
                return None

            last_idx = -1
            last_price = meta.get("regularMarketPrice", 0)
            prev_close = meta.get("previousClose", last_price)

            quote = StockQuote(
                symbol=symbol,
                last_price=last_price,
                change=last_price - prev_close,
                change_percent=((last_price - prev_close) / prev_close * 100)
                if prev_close
                else 0,
                volume=int(meta.get("regularMarketVolume", 0)),
                timestamp=datetime.now(),
            )

            self._quote_cache[cache_key] = (datetime.now(), quote)
            return quote

        except httpx.HTTPError as e:
            logger.error(f"Yahoo Finance quote request failed for {symbol}: {e}")
            return None

    async def get_quote_groww(self, symbol: str) -> Optional[StockQuote]:
        """
        Get real-time quote from Groww API.

        Args:
            symbol: Stock symbol (NSE)

        Returns:
            StockQuote object or None
        """
        await self._init_groww_api()

        if not self._groww_api:
            return None

        cache_key = f"groww_quote_{symbol}"
        if self._is_cache_valid(cache_key):
            return self._quote_cache[cache_key][1]

        try:
            # Use Groww API to get quote
            # Note: Actual API method may differ based on growwapi version
            quote_data = self._groww_api.get_quote(symbol)

            if not quote_data:
                return None

            quote = StockQuote(
                symbol=symbol,
                last_price=quote_data.get("ltp", 0),
                change=quote_data.get("change", 0),
                change_percent=quote_data.get("change_percent", 0),
                bid=quote_data.get("bid"),
                ask=quote_data.get("ask"),
                bid_size=quote_data.get("bid_size"),
                ask_size=quote_data.get("ask_size"),
                volume=quote_data.get("volume", 0),
                timestamp=datetime.now(),
            )

            self._quote_cache[cache_key] = (datetime.now(), quote)
            return quote

        except Exception as e:
            logger.error(f"Groww API quote request failed for {symbol}: {e}")
            return None

    async def get_quote(self, symbol: str) -> Optional[StockQuote]:
        """
        Get real-time quote from best available source.

        Tries Groww first, falls back to Yahoo Finance.

        Args:
            symbol: Stock symbol

        Returns:
            StockQuote object or None
        """
        # Try Groww first if configured
        if settings.groww_api_key:
            quote = await self.get_quote_groww(symbol)
            if quote:
                return quote

        # Fallback to Yahoo Finance
        return await self.get_quote_yahoo(symbol)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_historical_prices(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d",
    ) -> list[StockPrice]:
        """
        Get historical price data from Yahoo Finance.

        Args:
            symbol: Stock symbol
            start_date: Start date (default: 1 year ago)
            end_date: End date (default: today)
            interval: Data interval (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)

        Returns:
            List of StockPrice objects
        """
        yahoo_symbol = f"{symbol}{self.EXCHANGE_SUFFIXES['yahoo']}"

        if not start_date:
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()

        # Convert to Unix timestamps
        period1 = int(start_date.timestamp())
        period2 = int(end_date.timestamp())

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"

        try:
            response = await self._client.get(
                url,
                params={
                    "period1": period1,
                    "period2": period2,
                    "interval": interval,
                    "events": "history",
                },
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            data = response.json()

            result = data.get("chart", {}).get("result", [])
            if not result:
                return []

            timestamps = result[0].get("timestamp", [])
            quote = result[0].get("indicators", {}).get("quote", [{}])[0]

            prices = []
            prev_close = None

            for i, ts in enumerate(timestamps):
                open_price = quote.get("open", [])[i]
                high = quote.get("high", [])[i]
                low = quote.get("low", [])[i]
                close = quote.get("close", [])[i]
                volume = quote.get("volume", [])[i]

                # Skip if any value is None
                if any(v is None for v in [open_price, high, low, close, volume]):
                    continue

                change = close - prev_close if prev_close else 0
                change_percent = (change / prev_close * 100) if prev_close else 0

                price = StockPrice(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(ts),
                    open=open_price,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    change=change,
                    change_percent=change_percent,
                )
                prices.append(price)
                prev_close = close

            logger.info(f"Fetched {len(prices)} historical prices for {symbol}")
            return prices

        except httpx.HTTPError as e:
            logger.error(f"Historical price request failed for {symbol}: {e}")
            return []

    async def get_multiple_quotes(self, symbols: list[str]) -> dict[str, StockQuote]:
        """
        Get quotes for multiple symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbols to StockQuote objects
        """
        quotes = {}

        for symbol in symbols:
            quote = await self.get_quote(symbol)
            if quote:
                quotes[symbol] = quote

        return quotes

    def is_market_open(self) -> bool:
        """
        Check if Indian stock market is currently open.

        Returns:
            True if market is open, False otherwise
        """
        now = datetime.now()

        # Check if it's a weekday (Monday = 0, Sunday = 6)
        if now.weekday() >= 5:
            return False

        # Market hours: 9:15 AM to 3:30 PM IST
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

        return market_open <= now <= market_close
