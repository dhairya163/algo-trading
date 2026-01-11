"""
OpenAI Web Search Analyzer for real-time stock research.

Uses OpenAI's Responses API with web_search tool to fetch latest
news, sentiment, and market data for stocks.

Incorporates investment wisdom from legendary investors:
- Warren Buffett: Value investing, economic moats, ROE > 20%, low debt
- Peter Lynch: Growth at reasonable price, invest in what you know
- Rakesh Jhunjhunwala: Buy right sit tight, contrarian approach
- Radhakishan Damani: Cash flow focus, conservative approach
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from openai import OpenAI
from loguru import logger


class Recommendation(str, Enum):
    """Stock recommendation levels."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class RiskLevel(str, Enum):
    """Risk level categories."""
    LOW = "LOW"        # Blue chips, consistent performers, low volatility
    MEDIUM = "MEDIUM"  # Mid-caps, moderate volatility, growing companies
    HIGH = "HIGH"      # Small caps, high volatility, speculative


@dataclass
class StockResearch:
    """Research result for a stock."""
    symbol: str
    company_name: str
    recommendation: Recommendation
    confidence: float  # 0-100
    risk_level: RiskLevel = RiskLevel.MEDIUM

    current_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    upside_potential: Optional[float] = None  # percentage

    # Analysis components
    news_sentiment: str = ""
    technical_outlook: str = ""
    fundamental_view: str = ""

    # Key points
    bull_case: list[str] = field(default_factory=list)
    bear_case: list[str] = field(default_factory=list)
    catalysts: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    # Investor framework scores
    buffett_score: int = 0      # Value investing score (0-100)
    lynch_score: int = 0        # Growth at reasonable price (0-100)
    jhunjhunwala_score: int = 0 # Contrarian opportunity (0-100)

    # Full reasoning
    reasoning: str = ""
    sources: list[str] = field(default_factory=list)

    # Metadata
    analyzed_at: datetime = field(default_factory=datetime.now)

    def __str__(self):
        emoji = {
            Recommendation.STRONG_BUY: "🟢🟢",
            Recommendation.BUY: "🟢",
            Recommendation.HOLD: "🟡",
            Recommendation.SELL: "🔴",
            Recommendation.STRONG_SELL: "🔴🔴",
        }
        risk_emoji = {
            RiskLevel.LOW: "🛡️",
            RiskLevel.MEDIUM: "⚖️",
            RiskLevel.HIGH: "🎲",
        }
        price_str = f"₹{self.current_price:,.2f}" if self.current_price else "N/A"
        return f"{emoji.get(self.recommendation, '')} {self.symbol} ({self.company_name}) - {self.recommendation.value} @ {price_str} | Risk: {risk_emoji.get(self.risk_level, '')} {self.risk_level.value} ({self.confidence:.0f}% confidence)"


# Legendary investor criteria embedded in prompts
INVESTOR_WISDOM = """
## LEGENDARY INVESTOR FRAMEWORKS - Apply These When Analyzing:

### Warren Buffett's Criteria (Value Investing):
- Look for companies with ROE > 15-20% consistently
- Prefer low debt-to-equity ratio (< 0.5 ideal)
- Seek "economic moats" - competitive advantages like strong brands, patents, network effects
- Focus on predictable earnings and cash flows
- Buy when price is below intrinsic value (margin of safety)
- "Be fearful when others are greedy, greedy when others are fearful"

### Peter Lynch's Criteria (Growth at Reasonable Price):
- PEG ratio < 1 is attractive (P/E divided by earnings growth rate)
- Look for companies you understand - "invest in what you know"
- Categorize: Slow Growers, Stalwarts, Fast Growers, Cyclicals, Turnarounds, Asset Plays
- Fast growers (20-50% growth) in non-hot industries are ideal
- Avoid hot stocks in hot industries

### Rakesh Jhunjhunwala's Criteria (Indian Market Focus):
- "Buy right and sit tight" - patience is key
- Contrarian investing - buy when others panic sell
- Focus on India growth story - consumption, infrastructure, financials
- Look for companies with strong promoters and management
- Diversify across sectors but concentrate in high-conviction bets

### Radhakishan Damani's Criteria (Conservative Value):
- Cash flow is more important than earnings
- Low debt is crucial - avoid highly leveraged companies
- Simple business models that are easy to understand
- Strong competitive advantages in their niche
- Management integrity and track record

## RISK LEVEL CLASSIFICATION:

### LOW RISK (🛡️):
- Large-cap stocks (market cap > ₹50,000 Cr)
- Consistent dividend payers
- Beta < 1, low volatility
- Strong balance sheet, minimal debt
- Examples: HDFC Bank, TCS, Reliance, ITC, HUL

### MEDIUM RISK (⚖️):
- Mid-cap stocks (₹10,000 - ₹50,000 Cr market cap)
- Growing companies with moderate debt
- Beta around 1, moderate volatility
- Good fundamentals but less proven track record
- Examples: Most Nifty Next 50 stocks

### HIGH RISK (🎲):
- Small-cap stocks (< ₹10,000 Cr market cap)
- High growth but unproven business models
- Beta > 1.5, high volatility
- May have high debt or negative cash flows
- New IPOs, turnaround stories, speculative plays
- Examples: Recent IPOs, penny stocks, highly leveraged companies
"""


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
            model: Model to use (gpt-4o for web search via Responses API)
        """
        self.api_key = api_key
        self.model = model
        self._client = OpenAI(api_key=api_key)

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
            pnl = ((current_price or avg_cost) - avg_cost) * quantity
            pnl_pct = ((current_price or avg_cost) / avg_cost - 1) * 100 if avg_cost > 0 else 0
            context = f"""
This is an EXISTING HOLDING in the user's portfolio:
- Quantity: {quantity} shares
- Average Cost: ₹{avg_cost:,.2f}
- Investment Value: ₹{investment:,.2f}
- Current Price: ₹{current_price:,.2f} (if known)
- P&L: ₹{pnl:,.2f} ({pnl_pct:+.1f}%)
"""

        prompt = f"""Research and analyze {company_name} ({symbol}.NS) - an Indian stock listed on NSE.

{context}

{INVESTOR_WISDOM}

SEARCH FOR AND ANALYZE:
1. **Latest News** (last 7 days): Any major announcements, results, deals, or events
2. **Current Price & Technicals**: Live price, 52-week range, moving averages, RSI, support/resistance
3. **Recent Quarterly Results**: Latest earnings, revenue growth, profit margins, ROE
4. **Debt & Balance Sheet**: Debt-to-equity ratio, cash flows, financial health
5. **Analyst Ratings**: Recent broker upgrades/downgrades, target prices
6. **Sector Trends**: How the sector is performing, any tailwinds/headwinds
7. **FII/DII Activity**: Recent institutional buying/selling patterns
8. **Competitive Position**: Market share, economic moat, competitive advantages

Apply the legendary investor frameworks above to score this stock.

Based on your research, provide analysis in the following JSON format:
{{
    "recommendation": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
    "confidence": <0-100>,
    "risk_level": "LOW|MEDIUM|HIGH",
    "current_price": <current trading price>,
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
    "buffett_score": <0-100 based on value investing criteria>,
    "lynch_score": <0-100 based on GARP criteria>,
    "jhunjhunwala_score": <0-100 based on contrarian/India growth criteria>,
    "reasoning": "<2-3 paragraph detailed reasoning including which investor framework supports this>",
    "sources": ["source 1 URL", "source 2 URL"]
}}

IMPORTANT GUIDELINES:
- Use REAL, CURRENT data from your web search - cite actual news and prices
- Be specific about dates and figures from news articles
- For existing holdings, factor in the user's cost basis when making recommendations
- Apply investor frameworks: High Buffett score = strong fundamentals, High Lynch score = good growth value, High Jhunjhunwala score = contrarian opportunity
- Classify risk level based on market cap, volatility, debt levels
- Include specific price targets and stop losses in INR"""

        try:
            # Use OpenAI Responses API with web_search tool
            response = self._client.responses.create(
                model=self.model,
                tools=[{"type": "web_search"}],
                input=prompt,
            )

            # Extract the output text from response
            output_text = ""
            for item in response.output:
                if item.type == "message":
                    for content in item.content:
                        if content.type == "output_text":
                            output_text = content.text
                            break

            if output_text:
                return self._parse_research_response(symbol, company_name, output_text)

        except Exception as e:
            logger.warning(f"Responses API failed for {symbol}: {e}, using chat fallback")

        # Fallback to chat completions
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
            pnl_pct = ((current_price or avg_cost) / avg_cost - 1) * 100 if avg_cost > 0 else 0
            context = f"""
This is an EXISTING HOLDING:
- Quantity: {quantity} shares
- Average Cost: ₹{avg_cost:,.2f}
- Investment: ₹{investment:,.2f}
- Current Price: ₹{current_price:,.2f}
- P&L: ₹{pnl:,.2f} ({pnl_pct:+.1f}%)
"""

        system_prompt = f"""You are an expert Indian stock market analyst who follows the investment philosophies of Warren Buffett, Peter Lynch, Rakesh Jhunjhunwala, and Radhakishan Damani.

{INVESTOR_WISDOM}

Analyze stocks comprehensively and return analysis in valid JSON format."""

        user_prompt = f"""Analyze {company_name} ({symbol}) - NSE listed stock.
{context}

Provide analysis in this JSON format:
{{
    "recommendation": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
    "confidence": <0-100>,
    "risk_level": "LOW|MEDIUM|HIGH",
    "current_price": <estimated current price>,
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
    "buffett_score": <0-100>,
    "lynch_score": <0-100>,
    "jhunjhunwala_score": <0-100>,
    "reasoning": "<detailed reasoning with investor framework analysis>",
    "sources": ["General market knowledge"]
}}

Be specific with price targets in INR. Apply investor frameworks to score the stock."""

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
            )

            content = response.choices[0].message.content
            return self._parse_research_response(symbol, company_name, content)

        except Exception as e:
            logger.error(f"Chat API also failed for {symbol}: {e}")
            return StockResearch(
                symbol=symbol,
                company_name=company_name,
                recommendation=Recommendation.HOLD,
                confidence=30,
                risk_level=RiskLevel.MEDIUM,
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

                # Parse risk level
                risk_str = data.get("risk_level", "MEDIUM").upper()
                try:
                    risk_level = RiskLevel(risk_str)
                except ValueError:
                    risk_level = RiskLevel.MEDIUM

                return StockResearch(
                    symbol=symbol,
                    company_name=company_name,
                    recommendation=Recommendation(data.get("recommendation", "HOLD")),
                    confidence=float(data.get("confidence", 50)),
                    risk_level=risk_level,
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
                    buffett_score=int(data.get("buffett_score", 0)),
                    lynch_score=int(data.get("lynch_score", 0)),
                    jhunjhunwala_score=int(data.get("jhunjhunwala_score", 0)),
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
            risk_level=RiskLevel.MEDIUM,
            reasoning=response_text[:500] if response_text else "Analysis parsing failed",
        )

    async def find_opportunities(
        self,
        sector: Optional[str] = None,
        budget: float = 100000,
        existing_symbols: list[str] = None,
        risk_preference: Optional[str] = None,  # "LOW", "MEDIUM", "HIGH", or None for all
    ) -> list[StockResearch]:
        """
        Find new stock opportunities in the market using legendary investor wisdom.

        Args:
            sector: Specific sector to focus on (optional)
            budget: Investment budget in INR
            existing_symbols: Symbols already in portfolio (to avoid)
            risk_preference: Filter by risk level (optional)

        Returns:
            List of StockResearch for recommended buys categorized by risk
        """
        existing_symbols = existing_symbols or []
        exclude_str = f"Exclude these stocks (already held): {', '.join(existing_symbols)}" if existing_symbols else ""

        sector_focus = f"Focus on the {sector} sector." if sector else "Consider all sectors."

        risk_filter = ""
        if risk_preference:
            risk_filter = f"Focus primarily on {risk_preference} RISK stocks as per user preference."

        prompt = f"""Find the TOP 5 Indian stocks to BUY right now for a ₹{budget:,.0f} investment.
Search for the latest market news, stock performance, and analyst recommendations.

{sector_focus}
{exclude_str}
{risk_filter}

{INVESTOR_WISDOM}

SEARCH FOR STOCKS THAT MEET THESE CRITERIA:

**For LOW RISK recommendations:**
- Large-cap Nifty 50 stocks with consistent dividends
- Strong balance sheets, ROE > 15%, low debt
- Stocks Warren Buffett would approve of

**For MEDIUM RISK recommendations:**
- Mid-cap growth stocks with good fundamentals
- Companies Peter Lynch would call "Stalwarts" or "Fast Growers"
- Reasonable valuations (PEG < 1.5)

**For HIGH RISK recommendations:**
- Small-cap stocks with high growth potential
- Turnaround stories or contrarian plays Rakesh Jhunjhunwala would like
- Recent IPOs with strong business models

SEARCH FOR:
1. Stocks with recent positive momentum and strong fundamentals
2. Recent breakouts or accumulation patterns
3. Upcoming catalysts (earnings, deals, launches)
4. Stocks with institutional buying (FII/DII)
5. Undervalued opportunities based on legendary investor criteria

Return as a JSON array with stocks across ALL THREE risk categories:
[
    {{
        "symbol": "SYMBOL",
        "company_name": "Full Company Name",
        "recommendation": "BUY|STRONG_BUY",
        "confidence": <0-100>,
        "risk_level": "LOW|MEDIUM|HIGH",
        "current_price": <price>,
        "target_price": <target>,
        "stop_loss": <stop loss>,
        "upside_potential": <percentage>,
        "suggested_allocation": <percentage of budget>,
        "news_sentiment": "<recent news from web search>",
        "technical_outlook": "<technicals>",
        "fundamental_view": "<fundamentals>",
        "catalysts": ["catalyst 1", "catalyst 2"],
        "risks": ["risk 1"],
        "buffett_score": <0-100>,
        "lynch_score": <0-100>,
        "jhunjhunwala_score": <0-100>,
        "reasoning": "<why to buy now, which investor framework supports this>"
    }}
]

IMPORTANT:
- Include at least 1-2 stocks from each risk category (LOW, MEDIUM, HIGH)
- Use REAL data from web search - cite actual news and current prices
- Focus on liquid, NSE-listed stocks
- Higher Buffett score = value play, Higher Lynch score = growth play, Higher Jhunjhunwala score = contrarian play"""

        try:
            # Use Responses API with web search
            response = self._client.responses.create(
                model=self.model,
                tools=[{"type": "web_search"}],
                input=prompt,
            )

            # Extract output text
            output_text = ""
            for item in response.output:
                if item.type == "message":
                    for content in item.content:
                        if content.type == "output_text":
                            output_text = content.text
                            break

            if output_text:
                return self._parse_opportunities_response(output_text)

        except Exception as e:
            logger.warning(f"Responses API failed for opportunities: {e}, using chat fallback")

        # Fallback to chat completions
        return await self._find_opportunities_with_chat(
            sector, budget, existing_symbols, risk_preference
        )

    async def _find_opportunities_with_chat(
        self,
        sector: Optional[str],
        budget: float,
        existing_symbols: list[str],
        risk_preference: Optional[str],
    ) -> list[StockResearch]:
        """Fallback to chat completions for finding opportunities."""

        existing_symbols = existing_symbols or []
        exclude_str = f"Exclude: {', '.join(existing_symbols)}" if existing_symbols else ""
        sector_focus = f"Focus on {sector}." if sector else ""

        system_prompt = f"""You are an expert Indian stock market analyst combining the wisdom of Warren Buffett, Peter Lynch, Rakesh Jhunjhunwala, and Radhakishan Damani.

{INVESTOR_WISDOM}

Find the best investment opportunities. Return valid JSON array."""

        user_prompt = f"""Find TOP 5 Indian stocks to BUY for ₹{budget:,.0f} budget.
{sector_focus}
{exclude_str}

Return a JSON object with a "stocks" array containing stocks across LOW, MEDIUM, and HIGH risk categories:
{{
    "stocks": [
        {{
            "symbol": "SYMBOL",
            "company_name": "Full Company Name",
            "recommendation": "BUY",
            "confidence": 75,
            "risk_level": "LOW",
            "current_price": 1500.00,
            "target_price": 1800.00,
            "stop_loss": 1400.00,
            "upside_potential": 20.0,
            "news_sentiment": "Recent positive news summary",
            "technical_outlook": "Technical analysis summary",
            "fundamental_view": "Fundamental outlook",
            "catalysts": ["Upcoming catalyst 1", "Catalyst 2"],
            "risks": ["Key risk 1", "Risk 2"],
            "buffett_score": 80,
            "lynch_score": 70,
            "jhunjhunwala_score": 65,
            "reasoning": "Detailed 2-3 sentence reasoning explaining WHY to buy this stock now, which investor framework supports it, and what makes it attractive at current levels."
        }}
    ]
}}

IMPORTANT:
- Include 1-2 stocks from each risk category (LOW, MEDIUM, HIGH)
- Use current January 2025 market knowledge
- Provide DETAILED reasoning for each stock explaining WHY to buy
- All prices in INR
- reasoning field should be comprehensive (2-3 sentences minimum)"""

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            # Wrap in array if it's an object with stocks key
            if content.strip().startswith("{"):
                try:
                    data = json.loads(content)
                    if "stocks" in data:
                        content = json.dumps(data["stocks"])
                    elif "recommendations" in data:
                        content = json.dumps(data["recommendations"])
                except:
                    pass
            return self._parse_opportunities_response(content)

        except Exception as e:
            logger.error(f"Failed to find opportunities: {e}")
            return []

    def _parse_opportunities_response(self, response_text: str) -> list[StockResearch]:
        """Parse opportunities response into list of StockResearch."""
        import re

        try:
            # First try to find a JSON object with "stocks" key
            if '"stocks"' in response_text:
                obj_start = response_text.find("{")
                # Find matching closing brace
                brace_count = 0
                obj_end = -1
                for i, char in enumerate(response_text[obj_start:], obj_start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            obj_end = i + 1
                            break
                if obj_end > obj_start:
                    json_str = response_text[obj_start:obj_end]
                    json_str = re.sub(r',\s*]', ']', json_str)
                    json_str = re.sub(r',\s*}', '}', json_str)
                    data = json.loads(json_str)
                    if "stocks" in data:
                        stocks_data = data["stocks"]
                    elif "recommendations" in data:
                        stocks_data = data["recommendations"]
                    else:
                        stocks_data = list(data.values())[0] if data else []
            else:
                # Parse JSON array directly
                json_start = response_text.find("[")
                # Find matching closing bracket
                bracket_count = 0
                json_end = -1
                for i, char in enumerate(response_text[json_start:], json_start):
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            json_end = i + 1
                            break

                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    # Clean up common JSON issues
                    json_str = re.sub(r',\s*]', ']', json_str)
                    json_str = re.sub(r',\s*}', '}', json_str)
                    stocks_data = json.loads(json_str)
                else:
                    logger.error("No valid JSON array found in response")
                    return []

            if not isinstance(stocks_data, list):
                stocks_data = [stocks_data]

            opportunities = []
            for stock in stocks_data:
                # Parse risk level
                risk_str = stock.get("risk_level", "MEDIUM").upper()
                try:
                    risk_level = RiskLevel(risk_str)
                except ValueError:
                    risk_level = RiskLevel.MEDIUM

                opportunities.append(StockResearch(
                    symbol=stock.get("symbol", ""),
                    company_name=stock.get("company_name", ""),
                    recommendation=Recommendation(stock.get("recommendation", "BUY")),
                    confidence=float(stock.get("confidence", 60)),
                    risk_level=risk_level,
                    current_price=stock.get("current_price"),
                    target_price=stock.get("target_price"),
                    stop_loss=stock.get("stop_loss"),
                    upside_potential=stock.get("upside_potential"),
                    news_sentiment=stock.get("news_sentiment", ""),
                    technical_outlook=stock.get("technical_outlook", ""),
                    fundamental_view=stock.get("fundamental_view", ""),
                    catalysts=stock.get("catalysts", []),
                    risks=stock.get("risks", []),
                    buffett_score=int(stock.get("buffett_score", 0)),
                    lynch_score=int(stock.get("lynch_score", 0)),
                    jhunjhunwala_score=int(stock.get("jhunjhunwala_score", 0)),
                    reasoning=stock.get("reasoning", ""),
                ))

            return opportunities

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse opportunities: {e}")

        return []

    async def close(self):
        """Close the client (no-op for sync client)."""
        pass
