"""
OpenAI Web Search Analyzer for real-time stock research.

Uses OpenAI's GPT-4o with web search capability to fetch latest
news, sentiment, and market data for stocks.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import httpx
from loguru import logger


class Recommendation(str, Enum):
    """Stock recommendation levels."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class StockResearch:
    """Research result for a stock."""
    symbol: str
    company_name: str
    recommendation: Recommendation
    confidence: float  # 0-100
    current_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    upside_potential: Optional[float] = None  # percentage

    # Analysis components
    news_sentiment: str = ""
    technical_outlook: str = ""
    fundamental_view: str = ""

    # Key points
    bull_case: list[str] = None
    bear_case: list[str] = None
    catalysts: list[str] = None
    risks: list[str] = None

    # Full reasoning
    reasoning: str = ""
    sources: list[str] = None

    # Metadata
    analyzed_at: datetime = None

    def __post_init__(self):
        self.bull_case = self.bull_case or []
        self.bear_case = self.bear_case or []
        self.catalysts = self.catalysts or []
        self.risks = self.risks or []
        self.sources = self.sources or []
        self.analyzed_at = self.analyzed_at or datetime.now()

    def __str__(self):
        emoji = {
            Recommendation.STRONG_BUY: "🟢🟢",
            Recommendation.BUY: "🟢",
            Recommendation.HOLD: "🟡",
            Recommendation.SELL: "🔴",
            Recommendation.STRONG_SELL: "🔴🔴",
        }
        price_str = f"₹{self.current_price:,.2f}" if self.current_price else "N/A"
        return f"{emoji.get(self.recommendation, '')} {self.symbol} ({self.company_name}) - {self.recommendation.value} @ {price_str} ({self.confidence:.0f}% confidence)"


class WebSearchAnalyzer:
    """
    OpenAI-powered stock analyzer with web search capability.

    Uses OpenAI's Responses API with web_search tool for real-time
    market data, news, and sentiment analysis.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize the web search analyzer.

        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-4o recommended for web search)
        """
        self.api_key = api_key
        self.model = model
        self._client = httpx.AsyncClient(
            timeout=120.0,  # Longer timeout for web search
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    async def research_stock(
        self,
        symbol: str,
        company_name: str,
        current_price: Optional[float] = None,
        avg_cost: Optional[float] = None,
        quantity: Optional[int] = None,
        is_existing_holding: bool = False,
    ) -> StockResearch:
        """
        Research a stock using web search for latest data.

        Args:
            symbol: Stock symbol (NSE)
            company_name: Full company name
            current_price: Current market price (if known)
            avg_cost: Average purchase cost (for holdings)
            quantity: Number of shares held
            is_existing_holding: Whether this is an existing holding

        Returns:
            StockResearch with comprehensive analysis
        """

        context = ""
        if is_existing_holding and avg_cost and quantity:
            investment = avg_cost * quantity
            context = f"""
This is an EXISTING HOLDING in the user's portfolio:
- Quantity: {quantity} shares
- Average Cost: ₹{avg_cost:,.2f}
- Investment Value: ₹{investment:,.2f}
- Current Price: ₹{current_price:,.2f} (if available)
"""

        prompt = f"""Research and analyze {company_name} ({symbol}.NS) - an Indian stock listed on NSE.

{context}

SEARCH FOR AND ANALYZE:
1. **Latest News** (last 7 days): Any major announcements, results, deals, or events
2. **Current Price & Technicals**: Live price, 52-week range, moving averages, RSI, support/resistance
3. **Recent Quarterly Results**: Latest earnings, revenue growth, profit margins
4. **Analyst Ratings**: Recent broker upgrades/downgrades, target prices
5. **Sector Trends**: How the sector is performing, any tailwinds/headwinds
6. **FII/DII Activity**: Recent institutional buying/selling patterns
7. **Market Sentiment**: Social media buzz, retail investor sentiment

Based on your research, provide a comprehensive analysis in the following JSON format:
{{
    "recommendation": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
    "confidence": <0-100>,
    "current_price": <current trading price or null>,
    "target_price": <12-month target price>,
    "stop_loss": <suggested stop loss price>,
    "upside_potential": <percentage upside to target>,
    "news_sentiment": "<1-2 sentence summary of recent news sentiment>",
    "technical_outlook": "<1-2 sentence technical analysis summary>",
    "fundamental_view": "<1-2 sentence fundamental outlook>",
    "bull_case": ["point 1", "point 2", "point 3"],
    "bear_case": ["point 1", "point 2"],
    "catalysts": ["upcoming catalyst 1", "catalyst 2"],
    "risks": ["key risk 1", "risk 2"],
    "reasoning": "<2-3 paragraph detailed reasoning for recommendation>",
    "sources": ["source 1", "source 2"]
}}

IMPORTANT GUIDELINES:
- Use REAL, CURRENT data from your web search - do not hallucinate prices or news
- Be specific about dates and figures from news articles
- For existing holdings, factor in the user's cost basis when making recommendations
- Be conservative with STRONG_BUY/STRONG_SELL - reserve for clear opportunities
- Include specific price targets and stop losses in INR
- Cite your sources for key data points"""

        try:
            # Use OpenAI Responses API with web search
            response = await self._client.post(
                "https://api.openai.com/v1/responses",
                json={
                    "model": self.model,
                    "tools": [{"type": "web_search"}],
                    "input": prompt,
                }
            )

            if response.status_code == 404:
                # Fallback to chat completions if Responses API not available
                return await self._research_with_chat(
                    symbol, company_name, current_price, avg_cost, quantity, is_existing_holding
                )

            response.raise_for_status()
            data = response.json()

            # Extract the text response
            output_text = ""
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            output_text = content.get("text", "")
                            break

            return self._parse_research_response(symbol, company_name, output_text)

        except Exception as e:
            logger.warning(f"Web search API failed for {symbol}, using chat fallback: {e}")
            return await self._research_with_chat(
                symbol, company_name, current_price, avg_cost, quantity, is_existing_holding
            )

    async def _research_with_chat(
        self,
        symbol: str,
        company_name: str,
        current_price: Optional[float] = None,
        avg_cost: Optional[float] = None,
        quantity: Optional[int] = None,
        is_existing_holding: bool = False,
    ) -> StockResearch:
        """Fallback to chat completions API."""

        context = ""
        if is_existing_holding and avg_cost and quantity:
            investment = avg_cost * quantity
            pnl = ((current_price or avg_cost) - avg_cost) * quantity
            pnl_pct = ((current_price or avg_cost) / avg_cost - 1) * 100
            context = f"""
This is an EXISTING HOLDING:
- Quantity: {quantity} shares
- Average Cost: ₹{avg_cost:,.2f}
- Investment: ₹{investment:,.2f}
- Current Price: ₹{current_price:,.2f}
- P&L: ₹{pnl:,.2f} ({pnl_pct:+.1f}%)
"""

        system_prompt = """You are an expert Indian stock market analyst. Analyze stocks with comprehensive research.
Your knowledge is current as of January 2025. For recent data, make reasonable assumptions based on trends.
Always return analysis in valid JSON format."""

        user_prompt = f"""Analyze {company_name} ({symbol}) - NSE listed stock.
{context}

Provide analysis in this JSON format:
{{
    "recommendation": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
    "confidence": <0-100>,
    "current_price": <estimated current price or null>,
    "target_price": <12-month target>,
    "stop_loss": <stop loss level>,
    "upside_potential": <percentage>,
    "news_sentiment": "<recent news summary>",
    "technical_outlook": "<technical view>",
    "fundamental_view": "<fundamental outlook>",
    "bull_case": ["point 1", "point 2"],
    "bear_case": ["point 1", "point 2"],
    "catalysts": ["catalyst 1"],
    "risks": ["risk 1"],
    "reasoning": "<detailed reasoning>",
    "sources": ["General market knowledge"]
}}

Be specific with price targets in INR. For holdings, consider the cost basis."""

        try:
            response = await self._client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3,
                }
            )

            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            return self._parse_research_response(symbol, company_name, content)

        except Exception as e:
            logger.error(f"Chat API also failed for {symbol}: {e}")
            return StockResearch(
                symbol=symbol,
                company_name=company_name,
                recommendation=Recommendation.HOLD,
                confidence=30,
                reasoning=f"Unable to complete analysis: {str(e)}",
            )

    def _parse_research_response(
        self,
        symbol: str,
        company_name: str,
        response_text: str
    ) -> StockResearch:
        """Parse the JSON response into StockResearch."""

        try:
            # Extract JSON from response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)

                return StockResearch(
                    symbol=symbol,
                    company_name=company_name,
                    recommendation=Recommendation(data.get("recommendation", "HOLD")),
                    confidence=float(data.get("confidence", 50)),
                    current_price=data.get("current_price"),
                    target_price=data.get("target_price"),
                    stop_loss=data.get("stop_loss"),
                    upside_potential=data.get("upside_potential"),
                    news_sentiment=data.get("news_sentiment", ""),
                    technical_outlook=data.get("technical_outlook", ""),
                    fundamental_view=data.get("fundamental_view", ""),
                    bull_case=data.get("bull_case", []),
                    bear_case=data.get("bear_case", []),
                    catalysts=data.get("catalysts", []),
                    risks=data.get("risks", []),
                    reasoning=data.get("reasoning", ""),
                    sources=data.get("sources", []),
                )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse response for {symbol}: {e}")

        # Return a basic hold recommendation if parsing fails
        return StockResearch(
            symbol=symbol,
            company_name=company_name,
            recommendation=Recommendation.HOLD,
            confidence=40,
            reasoning=response_text[:500] if response_text else "Analysis parsing failed",
        )

    async def find_opportunities(
        self,
        sector: Optional[str] = None,
        budget: float = 100000,
        existing_symbols: list[str] = None,
    ) -> list[StockResearch]:
        """
        Find new stock opportunities in the market.

        Args:
            sector: Specific sector to focus on (optional)
            budget: Investment budget in INR
            existing_symbols: Symbols already in portfolio (to avoid)

        Returns:
            List of StockResearch for recommended buys
        """
        existing_symbols = existing_symbols or []
        exclude_str = f"Exclude these stocks (already held): {', '.join(existing_symbols)}" if existing_symbols else ""

        sector_focus = f"Focus on the {sector} sector." if sector else "Consider all sectors."

        prompt = f"""Find the TOP 5 Indian stocks to BUY right now for a ₹{budget:,.0f} investment.

{sector_focus}
{exclude_str}

SEARCH FOR:
1. Stocks with recent positive momentum and good fundamentals
2. Recent breakouts or accumulation patterns
3. Upcoming catalysts (earnings, deals, launches)
4. Stocks with institutional buying
5. Undervalued opportunities in growth sectors

For each stock, provide comprehensive research including:
- Why it's a good buy NOW
- Recent news and developments
- Technical levels (entry, target, stop loss)
- Risk/reward ratio

Return as a JSON array:
[
    {{
        "symbol": "SYMBOL",
        "company_name": "Full Company Name",
        "recommendation": "BUY|STRONG_BUY",
        "confidence": <0-100>,
        "current_price": <price>,
        "target_price": <target>,
        "stop_loss": <stop loss>,
        "upside_potential": <percentage>,
        "suggested_allocation": <percentage of budget>,
        "news_sentiment": "<recent news>",
        "technical_outlook": "<technicals>",
        "fundamental_view": "<fundamentals>",
        "catalysts": ["catalyst 1", "catalyst 2"],
        "risks": ["risk 1"],
        "reasoning": "<why to buy now>"
    }}
]

Focus on liquid, well-known NSE stocks. Prioritize quality over speculation."""

        try:
            response = await self._client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert Indian stock market analyst. Find the best investment opportunities. Return valid JSON."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.4,
                }
            )

            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Parse JSON array
            json_start = content.find("[")
            json_end = content.rfind("]") + 1

            if json_start >= 0 and json_end > json_start:
                stocks_data = json.loads(content[json_start:json_end])

                opportunities = []
                for stock in stocks_data:
                    opportunities.append(StockResearch(
                        symbol=stock.get("symbol", ""),
                        company_name=stock.get("company_name", ""),
                        recommendation=Recommendation(stock.get("recommendation", "BUY")),
                        confidence=float(stock.get("confidence", 60)),
                        current_price=stock.get("current_price"),
                        target_price=stock.get("target_price"),
                        stop_loss=stock.get("stop_loss"),
                        upside_potential=stock.get("upside_potential"),
                        news_sentiment=stock.get("news_sentiment", ""),
                        technical_outlook=stock.get("technical_outlook", ""),
                        fundamental_view=stock.get("fundamental_view", ""),
                        catalysts=stock.get("catalysts", []),
                        risks=stock.get("risks", []),
                        reasoning=stock.get("reasoning", ""),
                    ))

                return opportunities

        except Exception as e:
            logger.error(f"Failed to find opportunities: {e}")
            return []

        return []

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
