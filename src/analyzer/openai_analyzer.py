"""
OpenAI-powered stock analysis.

Uses GPT models to analyze stocks based on:
- Recent news and market sentiment
- Technical indicators
- Fundamental data
- Market conditions
"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import httpx
from loguru import logger

from config.settings import settings


class Recommendation(str, Enum):
    """Stock recommendation types."""

    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class StockAnalysis:
    """Analysis result for a stock."""

    symbol: str
    company_name: str
    recommendation: Recommendation
    confidence: float  # 0-100%
    current_price: Optional[float]
    target_price: Optional[float]
    stop_loss: Optional[float]

    # Analysis details
    sentiment_summary: str
    technical_summary: str
    fundamental_summary: str
    risks: list[str]
    catalysts: list[str]

    # Overall reasoning
    reasoning: str
    timestamp: datetime

    def __str__(self) -> str:
        emoji = {
            Recommendation.STRONG_BUY: "🟢🟢",
            Recommendation.BUY: "🟢",
            Recommendation.HOLD: "🟡",
            Recommendation.SELL: "🔴",
            Recommendation.STRONG_SELL: "🔴🔴",
        }.get(self.recommendation, "⚪")

        return f"{emoji} {self.symbol}: {self.recommendation.value} ({self.confidence:.0f}% confidence)"


class OpenAIStockAnalyzer:
    """
    Analyzes stocks using OpenAI GPT models.

    Fetches news, analyzes sentiment, and provides recommendations.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
    ):
        """
        Initialize OpenAI analyzer.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            model: Model to use (gpt-4o, gpt-4o-mini, etc.)
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = model
        self._client = httpx.AsyncClient(timeout=60.0)

        if not self.api_key:
            logger.warning("OpenAI API key not configured")

    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()

    async def _fetch_news(self, symbol: str, company_name: str) -> list[dict]:
        """Fetch recent news for a stock."""
        # Try to get news from free sources
        news = []

        # Use a simple news search
        try:
            # Google News RSS (free, no API key needed)
            search_query = f"{company_name} stock"
            url = f"https://news.google.com/rss/search?q={search_query}&hl=en-IN&gl=IN&ceid=IN:en"

            response = await self._client.get(url)
            if response.status_code == 200:
                # Parse RSS (simplified)
                import re
                titles = re.findall(r'<title>(.*?)</title>', response.text)
                for title in titles[1:6]:  # Skip first (feed title), get 5 news
                    news.append({"title": title, "source": "Google News"})
        except Exception as e:
            logger.debug(f"Could not fetch news: {e}")

        return news

    async def _call_openai(self, messages: list[dict]) -> str:
        """Make a call to OpenAI API."""
        if not self.api_key:
            raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY in .env")

        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value() if hasattr(self.api_key, 'get_secret_value') else self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,  # Lower temperature for more consistent analysis
            "max_tokens": 2000,
        }

        response = await self._client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
        )

        if response.status_code != 200:
            error = response.json()
            raise Exception(f"OpenAI API error: {error}")

        result = response.json()
        return result["choices"][0]["message"]["content"]

    async def analyze_stock(
        self,
        symbol: str,
        company_name: str,
        current_price: Optional[float] = None,
        avg_cost: Optional[float] = None,
        quantity: Optional[int] = None,
        additional_context: str = "",
    ) -> StockAnalysis:
        """
        Analyze a stock and provide recommendation.

        Args:
            symbol: Stock symbol (e.g., RELIANCE)
            company_name: Full company name
            current_price: Current stock price
            avg_cost: Average purchase cost (if held)
            quantity: Number of shares held
            additional_context: Any additional context

        Returns:
            StockAnalysis with recommendation
        """
        logger.info(f"Analyzing {symbol} ({company_name})...")

        # Fetch recent news
        news = await self._fetch_news(symbol, company_name)
        news_text = "\n".join([f"- {n['title']}" for n in news]) if news else "No recent news available"

        # Build context
        position_context = ""
        if avg_cost and quantity:
            pnl = (current_price - avg_cost) * quantity if current_price else 0
            pnl_pct = ((current_price - avg_cost) / avg_cost * 100) if current_price and avg_cost else 0
            position_context = f"""
Current Position:
- Shares held: {quantity}
- Average cost: ₹{avg_cost:,.2f}
- Current price: ₹{current_price:,.2f}
- Unrealized P&L: ₹{pnl:,.2f} ({pnl_pct:+.1f}%)
"""

        # Create prompt
        system_prompt = """You are an expert Indian stock market analyst. Analyze stocks and provide clear recommendations.

Your analysis should consider:
1. Recent news and sentiment
2. Technical factors (price trends, support/resistance)
3. Fundamental factors (sector outlook, company fundamentals)
4. Risk factors
5. Market conditions

Provide your response in the following JSON format:
{
    "recommendation": "STRONG_BUY" | "BUY" | "HOLD" | "SELL" | "STRONG_SELL",
    "confidence": <number 0-100>,
    "target_price": <number or null>,
    "stop_loss": <number or null>,
    "sentiment_summary": "<brief sentiment analysis>",
    "technical_summary": "<brief technical analysis>",
    "fundamental_summary": "<brief fundamental view>",
    "risks": ["risk1", "risk2"],
    "catalysts": ["catalyst1", "catalyst2"],
    "reasoning": "<detailed reasoning for the recommendation>"
}

Be specific to Indian market context (NSE/BSE). Consider the investor's current position if provided."""

        user_prompt = f"""Analyze this stock:

Stock: {symbol} - {company_name}
Current Price: ₹{current_price:,.2f} (NSE)
{position_context}

Recent News:
{news_text}

{additional_context}

Provide your analysis and recommendation in JSON format."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Call OpenAI
        response = await self._call_openai(messages)

        # Parse response
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            data = json.loads(json_str.strip())

            return StockAnalysis(
                symbol=symbol,
                company_name=company_name,
                recommendation=Recommendation(data["recommendation"]),
                confidence=data["confidence"],
                current_price=current_price,
                target_price=data.get("target_price"),
                stop_loss=data.get("stop_loss"),
                sentiment_summary=data.get("sentiment_summary", ""),
                technical_summary=data.get("technical_summary", ""),
                fundamental_summary=data.get("fundamental_summary", ""),
                risks=data.get("risks", []),
                catalysts=data.get("catalysts", []),
                reasoning=data.get("reasoning", ""),
                timestamp=datetime.now(),
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse OpenAI response: {e}")
            logger.debug(f"Response was: {response}")

            # Return a default analysis
            return StockAnalysis(
                symbol=symbol,
                company_name=company_name,
                recommendation=Recommendation.HOLD,
                confidence=30,
                current_price=current_price,
                target_price=None,
                stop_loss=None,
                sentiment_summary="Analysis failed",
                technical_summary="",
                fundamental_summary="",
                risks=["Analysis could not be completed"],
                catalysts=[],
                reasoning=f"OpenAI analysis failed to parse: {response[:500]}",
                timestamp=datetime.now(),
            )

    async def analyze_stocks_batch(
        self,
        stocks: list[dict],
    ) -> list[StockAnalysis]:
        """
        Analyze multiple stocks.

        Args:
            stocks: List of stock dicts with keys: symbol, company_name, current_price, avg_cost, quantity

        Returns:
            List of StockAnalysis objects
        """
        analyses = []

        for stock in stocks:
            try:
                analysis = await self.analyze_stock(
                    symbol=stock["symbol"],
                    company_name=stock.get("company_name", stock["symbol"]),
                    current_price=stock.get("current_price"),
                    avg_cost=stock.get("avg_cost"),
                    quantity=stock.get("quantity"),
                )
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"Failed to analyze {stock['symbol']}: {e}")

        return analyses
