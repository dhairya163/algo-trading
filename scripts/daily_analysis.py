#!/usr/bin/env python3
"""
Daily Stock Analysis Script - AI-Powered Portfolio & Market Research

This script provides comprehensive daily analysis:
1. Analyzes your current Groww holdings (HOLD/SELL recommendations)
2. Researches the market for new opportunities (BUY recommendations)
3. Provides actionable insights with specific price targets

Uses:
- Groww API for portfolio holdings and prices (or manual input/file)
- OpenAI GPT-4o for AI-powered analysis with market knowledge

Usage:
    python scripts/daily_analysis.py                        # Full analysis with ₹1L budget
    python scripts/daily_analysis.py --budget 50000         # Custom budget
    python scripts/daily_analysis.py --holdings-only        # Only analyze holdings
    python scripts/daily_analysis.py --opportunities        # Only find new opportunities
    python scripts/daily_analysis.py --holdings-file h.json # Load holdings from file
    python scripts/daily_analysis.py --manual               # Enter holdings manually
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

# Add project root to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from config.settings import settings
from src.analyzer.web_search_analyzer import WebSearchAnalyzer, StockResearch, Recommendation
from src.analyzer.groww_direct import GrowwDirectClient


# Company name mappings for NSE stocks
COMPANY_NAMES = {
    # Nifty 50 + Popular Stocks
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
    "SUNPHARMA": "Sun Pharmaceutical",
    "TITAN": "Titan Company",
    "TATASTEEL": "Tata Steel",
    "JSWSTEEL": "JSW Steel",
    "ADANIENT": "Adani Enterprises",
    "ADANIPORTS": "Adani Ports",
    "POWERGRID": "Power Grid Corporation",
    "NTPC": "NTPC Limited",
    "ONGC": "Oil and Natural Gas Corporation",
    "COALINDIA": "Coal India",
    "BPCL": "Bharat Petroleum",
    "IOC": "Indian Oil Corporation",
    # User's holdings
    "ORIANA": "Oriana Power",
    "ETERNAL": "Eternal Limited (Zomato)",
    "IREDA": "Indian Renewable Energy Development Agency",
    "SWIGGY": "Swiggy",
    "JSWENERGY": "JSW Energy",
    "FORTIS": "Fortis Healthcare",
    "ESAFSFB": "ESAF Small Finance Bank",
    "SUZLON": "Suzlon Energy",
    "YESBANK": "Yes Bank",
    "BAJAJHFL": "Bajaj Housing Finance",
    "BHEL": "Bharat Heavy Electricals",
    "GOLDBEES": "Nippon India ETF Gold BeES",
    "ADANIENSOL": "Adani Energy Solutions",
    "ITBEES": "Nippon India ETF Nifty BeES",
    "EQUITASBNK": "Equitas Small Finance Bank",
    "MAFANG": "Mirae Asset NYSE FANG+ ETF",
    # Additional popular stocks
    "ZOMATO": "Zomato",
    "PAYTM": "One97 Communications (Paytm)",
    "NYKAA": "FSN E-Commerce (Nykaa)",
    "DELHIVERY": "Delhivery",
    "POLICYBZR": "PB Fintech (Policybazaar)",
    "TATAPOWER": "Tata Power",
    "ADANIGREEN": "Adani Green Energy",
    "VEDL": "Vedanta",
    "HINDALCO": "Hindalco Industries",
    "GRASIM": "Grasim Industries",
    "ULTRACEMCO": "UltraTech Cement",
    "SHREECEM": "Shree Cement",
    "DRREDDY": "Dr. Reddy's Laboratories",
    "CIPLA": "Cipla",
    "DIVISLAB": "Divi's Laboratories",
    "APOLLOHOSP": "Apollo Hospitals",
    "MAXHEALTH": "Max Healthcare",
    "HDFCLIFE": "HDFC Life Insurance",
    "SBILIFE": "SBI Life Insurance",
    "ICICIPRULI": "ICICI Prudential Life",
    "BAJAJFINSV": "Bajaj Finserv",
    "CHOLAFIN": "Cholamandalam Investment",
    "M&M": "Mahindra & Mahindra",
    "EICHERMOT": "Eicher Motors",
    "HEROMOTOCO": "Hero MotoCorp",
    "BAJAJ-AUTO": "Bajaj Auto",
    "TVSMOTOR": "TVS Motor",
    "HAL": "Hindustan Aeronautics",
    "BEL": "Bharat Electronics",
    "IRFC": "Indian Railway Finance Corporation",
    "PFC": "Power Finance Corporation",
    "RECLTD": "REC Limited",
    "NHPC": "NHPC Limited",
    "SJVN": "SJVN Limited",
}


class DailyAnalyzer:
    """
    Comprehensive daily stock analyzer.

    Combines Groww API (portfolio data) with OpenAI (AI analysis)
    to provide actionable trading recommendations.
    """

    def __init__(
        self,
        groww_api_key: Optional[str] = None,
        groww_api_secret: Optional[str] = None,
        openai_api_key: str = None,
        openai_model: str = "gpt-4o",
        investment_budget: float = 100000,
    ):
        self.groww_api_key = groww_api_key
        self.groww_api_secret = groww_api_secret
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.investment_budget = investment_budget

        self.groww_client: Optional[GrowwDirectClient] = None
        self.ai_analyzer: Optional[WebSearchAnalyzer] = None

        self.holdings: list[dict] = []
        self.holding_analyses: list[StockResearch] = []
        self.opportunities: list[StockResearch] = []

    async def initialize(self) -> bool:
        """Initialize API connections."""
        # Initialize OpenAI analyzer
        if self.openai_api_key:
            self.ai_analyzer = WebSearchAnalyzer(
                api_key=self.openai_api_key,
                model=self.openai_model,
            )
            logger.info("OpenAI analyzer initialized")

        # Initialize Groww client
        if self.groww_api_key:
            self.groww_client = GrowwDirectClient(
                api_key=self.groww_api_key,
                api_secret=self.groww_api_secret,
            )
            logger.info("Groww client initialized")
            return True

        return False

    async def fetch_holdings(self) -> list[dict]:
        """Fetch current holdings from Groww."""
        if not self.groww_client:
            logger.warning("Groww client not available")
            return []

        try:
            self.holdings = self.groww_client.get_holdings()
            logger.info(f"Fetched {len(self.holdings)} holdings from Groww")
            return self.holdings
        except Exception as e:
            logger.error(f"Failed to fetch holdings: {e}")
            return []

    def load_holdings_from_file(self, file_path: str) -> list[dict]:
        """
        Load holdings from a JSON file.

        Expected format:
        [
            {"symbol": "RELIANCE", "quantity": 10, "average_price": 2400, "last_price": 2500},
            {"symbol": "TCS", "quantity": 5, "average_price": 3500, "last_price": 3600}
        ]
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"Holdings file not found: {file_path}")
                return []

            with open(path) as f:
                self.holdings = json.load(f)
                logger.info(f"Loaded {len(self.holdings)} holdings from {file_path}")
                return self.holdings
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in holdings file: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to load holdings file: {e}")
            return []

    def input_holdings_manually(self) -> list[dict]:
        """Interactive mode to enter holdings manually."""
        print("\n📝 MANUAL HOLDINGS ENTRY")
        print("-" * 50)
        print("Enter your holdings (type 'done' when finished):")
        print("Format: SYMBOL QUANTITY AVG_COST [CURRENT_PRICE]")
        print("Example: RELIANCE 10 2400 2500")
        print()

        holdings = []
        while True:
            try:
                entry = input("  > ").strip()
                if entry.lower() == 'done':
                    break
                if not entry:
                    continue

                parts = entry.split()
                if len(parts) < 3:
                    print("    ⚠️  Invalid format. Use: SYMBOL QUANTITY AVG_COST [CURRENT_PRICE]")
                    continue

                symbol = parts[0].upper()
                quantity = int(parts[1])
                avg_price = float(parts[2])
                current_price = float(parts[3]) if len(parts) > 3 else avg_price

                holdings.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "average_price": avg_price,
                    "last_price": current_price,
                })
                print(f"    ✓ Added {symbol}: {quantity} shares @ ₹{avg_price:,.2f}")

            except ValueError as e:
                print(f"    ⚠️  Invalid input: {e}")
            except KeyboardInterrupt:
                print("\n    Cancelled.")
                break

        self.holdings = holdings
        logger.info(f"Entered {len(holdings)} holdings manually")
        return holdings

    async def analyze_holdings(self) -> list[StockResearch]:
        """Analyze all current holdings."""
        if not self.ai_analyzer:
            logger.error("OpenAI analyzer not initialized")
            return []

        if not self.holdings:
            await self.fetch_holdings()

        if not self.holdings:
            logger.warning("No holdings to analyze")
            return []

        print(f"\n📊 Analyzing {len(self.holdings)} holdings...")
        print("-" * 50)

        analyses = []
        for i, holding in enumerate(self.holdings, 1):
            symbol = holding.get("symbol", holding.get("tradingsymbol", "UNKNOWN"))
            quantity = holding.get("quantity", 0)
            avg_cost = holding.get("average_price", holding.get("averagePrice", 0))
            current_price = holding.get("last_price", holding.get("ltp", avg_cost))
            company_name = COMPANY_NAMES.get(symbol.upper(), symbol)

            print(f"  [{i}/{len(self.holdings)}] Researching {symbol}...", end=" ", flush=True)

            try:
                research = await self.ai_analyzer.research_stock(
                    symbol=symbol,
                    company_name=company_name,
                    current_price=current_price,
                    avg_cost=avg_cost,
                    quantity=quantity,
                    is_existing_holding=True,
                )
                analyses.append(research)
                print(f"✓ {research.recommendation.value}")
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")
                print(f"✗ Error")

            # Small delay to avoid rate limits
            await asyncio.sleep(1)

        self.holding_analyses = analyses
        return analyses

    async def find_opportunities(self) -> list[StockResearch]:
        """Find new stock opportunities."""
        if not self.ai_analyzer:
            logger.error("OpenAI analyzer not initialized")
            return []

        # Get list of existing holdings to exclude
        existing_symbols = []
        if self.holdings:
            existing_symbols = [
                h.get("symbol", h.get("tradingsymbol", "")).upper()
                for h in self.holdings
            ]

        print(f"\n🔍 Searching for new opportunities (Budget: ₹{self.investment_budget:,.0f})...")
        print("-" * 50)

        try:
            opportunities = await self.ai_analyzer.find_opportunities(
                budget=self.investment_budget,
                existing_symbols=existing_symbols,
            )
            self.opportunities = opportunities
            print(f"  Found {len(opportunities)} potential opportunities")
            return opportunities
        except Exception as e:
            logger.error(f"Failed to find opportunities: {e}")
            return []

    async def run_full_analysis(self) -> dict:
        """Run complete daily analysis."""

        print("\n" + "=" * 70)
        print(f"📈 DAILY STOCK ANALYSIS - {datetime.now().strftime('%B %d, %Y %I:%M %p')}")
        print("=" * 70)

        results = {
            "timestamp": datetime.now().isoformat(),
            "budget": self.investment_budget,
            "holdings_count": 0,
            "sell_recommendations": [],
            "hold_recommendations": [],
            "buy_recommendations": [],
            "total_portfolio_value": 0,
        }

        # Fetch and analyze holdings (skip fetch if already loaded)
        if not self.holdings:
            await self.fetch_holdings()
        if self.holdings:
            results["holdings_count"] = len(self.holdings)

            # Calculate portfolio value
            total_value = sum(
                h.get("quantity", 0) * h.get("last_price", h.get("ltp", h.get("average_price", 0)))
                for h in self.holdings
            )
            results["total_portfolio_value"] = total_value

            # Analyze each holding
            await self.analyze_holdings()

            # Categorize recommendations
            for analysis in self.holding_analyses:
                if analysis.recommendation in [Recommendation.SELL, Recommendation.STRONG_SELL]:
                    results["sell_recommendations"].append(analysis)
                else:
                    results["hold_recommendations"].append(analysis)

        # Find new opportunities
        await self.find_opportunities()
        results["buy_recommendations"] = self.opportunities

        return results

    def print_report(self, results: dict):
        """Print comprehensive analysis report."""

        print("\n")
        print("=" * 70)
        print("📊 DAILY ANALYSIS REPORT")
        print("=" * 70)

        # Portfolio Summary
        print(f"\n💼 PORTFOLIO SUMMARY")
        print("-" * 50)
        print(f"   Total Holdings: {results['holdings_count']} stocks")
        print(f"   Portfolio Value: ₹{results['total_portfolio_value']:,.2f}")
        print(f"   Investment Budget: ₹{results['budget']:,.0f}")

        # SELL Recommendations
        sells = results["sell_recommendations"]
        if sells:
            print(f"\n🔴 SELL RECOMMENDATIONS ({len(sells)} stocks)")
            print("-" * 50)
            for analysis in sorted(sells, key=lambda x: x.confidence, reverse=True):
                self._print_stock_card(analysis, show_action="SELL")
        else:
            print(f"\n🟢 No SELL recommendations - portfolio looks healthy!")

        # HOLD Recommendations
        holds = results["hold_recommendations"]
        if holds:
            # Sort by recommendation strength and confidence
            holds_sorted = sorted(
                holds,
                key=lambda x: (
                    0 if x.recommendation == Recommendation.STRONG_BUY else
                    1 if x.recommendation == Recommendation.BUY else
                    2 if x.recommendation == Recommendation.HOLD else 3,
                    -x.confidence
                )
            )

            print(f"\n🟡 HOLD/KEEP RECOMMENDATIONS ({len(holds)} stocks)")
            print("-" * 50)
            for analysis in holds_sorted:
                self._print_stock_summary(analysis)

        # BUY Recommendations (New Opportunities)
        buys = results["buy_recommendations"]
        if buys:
            print(f"\n🟢 BUY RECOMMENDATIONS - New Opportunities ({len(buys)} stocks)")
            print("-" * 50)
            for analysis in sorted(buys, key=lambda x: x.confidence, reverse=True):
                self._print_stock_card(analysis, show_action="BUY")

            # Suggested allocation
            print(f"\n💰 SUGGESTED ALLOCATION (Budget: ₹{results['budget']:,.0f})")
            print("-" * 50)
            allocation_per_stock = results['budget'] / len(buys)
            for analysis in buys:
                if analysis.current_price:
                    shares = int(allocation_per_stock / analysis.current_price)
                    investment = shares * analysis.current_price
                    print(f"   {analysis.symbol}: {shares} shares @ ₹{analysis.current_price:,.2f} = ₹{investment:,.2f}")
                else:
                    print(f"   {analysis.symbol}: ~₹{allocation_per_stock:,.0f} (price TBD)")
        else:
            print(f"\n📝 No strong BUY recommendations at this time")

        # Action Summary
        print("\n" + "=" * 70)
        print("📋 ACTION SUMMARY")
        print("=" * 70)

        if sells:
            print(f"\n⚡ IMMEDIATE ACTIONS:")
            for s in sells:
                print(f"   • SELL {s.symbol} - {s.reasoning[:100]}...")

        if buys:
            print(f"\n🎯 BUYING OPPORTUNITIES:")
            for b in buys:
                price_str = f"@ ₹{b.current_price:,.2f}" if b.current_price else ""
                target_str = f"(Target: ₹{b.target_price:,.2f})" if b.target_price else ""
                print(f"   • BUY {b.symbol} {price_str} {target_str}")

        print("\n" + "=" * 70)
        print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

    def _print_stock_card(self, analysis: StockResearch, show_action: str = ""):
        """Print detailed stock analysis card."""
        emoji = {
            Recommendation.STRONG_BUY: "🟢🟢",
            Recommendation.BUY: "🟢",
            Recommendation.HOLD: "🟡",
            Recommendation.SELL: "🔴",
            Recommendation.STRONG_SELL: "🔴🔴",
        }

        print(f"\n   {emoji.get(analysis.recommendation, '')} {analysis.symbol} - {analysis.company_name}")
        print(f"   {'─' * 45}")

        if analysis.current_price:
            print(f"   Current: ₹{analysis.current_price:,.2f}", end="")
            if analysis.target_price:
                upside = (analysis.target_price / analysis.current_price - 1) * 100
                print(f" → Target: ₹{analysis.target_price:,.2f} ({upside:+.1f}%)", end="")
            if analysis.stop_loss:
                print(f" | SL: ₹{analysis.stop_loss:,.2f}", end="")
            print()

        print(f"   Confidence: {analysis.confidence:.0f}%")

        if analysis.news_sentiment:
            print(f"   📰 News: {analysis.news_sentiment[:80]}...")

        if analysis.catalysts:
            print(f"   ✨ Catalysts: {', '.join(analysis.catalysts[:2])}")

        if analysis.risks:
            print(f"   ⚠️  Risks: {', '.join(analysis.risks[:2])}")

        # Truncate reasoning for readability
        if analysis.reasoning:
            reasoning = analysis.reasoning.replace("\n", " ")[:200]
            print(f"   💡 {reasoning}...")

    def _print_stock_summary(self, analysis: StockResearch):
        """Print brief stock summary for holdings."""
        emoji = {
            Recommendation.STRONG_BUY: "🟢🟢",
            Recommendation.BUY: "🟢",
            Recommendation.HOLD: "🟡",
            Recommendation.SELL: "🔴",
            Recommendation.STRONG_SELL: "🔴🔴",
        }

        rec_text = analysis.recommendation.value.replace("_", " ")
        price_str = f"₹{analysis.current_price:,.2f}" if analysis.current_price else ""

        print(f"   {emoji.get(analysis.recommendation, '')} {analysis.symbol:12} {rec_text:12} {price_str:>12}  ({analysis.confidence:.0f}% conf)")

    async def close(self):
        """Cleanup resources."""
        if self.ai_analyzer:
            await self.ai_analyzer.close()
        if self.groww_client:
            self.groww_client.close()


async def main():
    """Main entry point."""

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<dim>{time:HH:mm:ss}</dim> | {message}",
        level="INFO",
    )

    parser = argparse.ArgumentParser(
        description="Daily AI-powered stock analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/daily_analysis.py                        # Full analysis
    python scripts/daily_analysis.py --budget 50000         # ₹50K budget
    python scripts/daily_analysis.py --holdings-only        # Only analyze holdings
    python scripts/daily_analysis.py --opportunities        # Only find opportunities
    python scripts/daily_analysis.py --holdings-file h.json # Load holdings from file
    python scripts/daily_analysis.py --manual               # Enter holdings manually
        """,
    )

    parser.add_argument(
        "--budget",
        type=float,
        default=100000,
        help="Investment budget in INR (default: ₹1,00,000)",
    )
    parser.add_argument(
        "--holdings-only",
        action="store_true",
        help="Only analyze current holdings, skip opportunity search",
    )
    parser.add_argument(
        "--opportunities",
        action="store_true",
        help="Only search for new opportunities",
    )
    parser.add_argument(
        "--holdings-file",
        type=str,
        help="Load holdings from a JSON file instead of Groww API",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Enter holdings manually via interactive prompt",
    )
    parser.add_argument(
        "--groww-key",
        type=str,
        help="Groww API key (or set GROWW_API_KEY in .env)",
    )
    parser.add_argument(
        "--openai-key",
        type=str,
        help="OpenAI API key (or set OPENAI_API_KEY in .env)",
    )

    args = parser.parse_args()

    # Get API keys
    groww_api_key = args.groww_key or (
        settings.groww_api_key.get_secret_value() if settings.groww_api_key else None
    )
    groww_api_secret = (
        settings.groww_api_secret.get_secret_value() if settings.groww_api_secret else None
    )
    openai_api_key = args.openai_key or (
        settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    )

    if not openai_api_key:
        print("❌ OpenAI API key required!")
        print("   Set OPENAI_API_KEY in .env or pass via --openai-key")
        sys.exit(1)

    # Initialize analyzer
    analyzer = DailyAnalyzer(
        groww_api_key=groww_api_key,
        groww_api_secret=groww_api_secret,
        openai_api_key=openai_api_key,
        openai_model=settings.openai_model,
        investment_budget=args.budget,
    )

    try:
        await analyzer.initialize()

        # Handle holdings source
        if args.holdings_file:
            analyzer.load_holdings_from_file(args.holdings_file)
        elif args.manual:
            analyzer.input_holdings_manually()

        if args.opportunities:
            # Only find opportunities
            print("\n🔍 Searching for investment opportunities...")
            opportunities = await analyzer.find_opportunities()
            results = {
                "timestamp": datetime.now().isoformat(),
                "budget": args.budget,
                "holdings_count": 0,
                "sell_recommendations": [],
                "hold_recommendations": [],
                "buy_recommendations": opportunities,
                "total_portfolio_value": 0,
            }
        elif args.holdings_only or args.holdings_file or args.manual:
            # Analyze holdings (from file, manual, or Groww)
            if not analyzer.holdings:
                await analyzer.fetch_holdings()
            await analyzer.analyze_holdings()

            sell_recs = [a for a in analyzer.holding_analyses
                        if a.recommendation in [Recommendation.SELL, Recommendation.STRONG_SELL]]
            hold_recs = [a for a in analyzer.holding_analyses
                        if a.recommendation not in [Recommendation.SELL, Recommendation.STRONG_SELL]]

            total_value = sum(
                h.get("quantity", 0) * h.get("last_price", h.get("ltp", h.get("average_price", 0)))
                for h in analyzer.holdings
            )

            results = {
                "timestamp": datetime.now().isoformat(),
                "budget": args.budget,
                "holdings_count": len(analyzer.holdings),
                "sell_recommendations": sell_recs,
                "hold_recommendations": hold_recs,
                "buy_recommendations": [],
                "total_portfolio_value": total_value,
            }
        else:
            # Full analysis
            results = await analyzer.run_full_analysis()

        # Print report
        analyzer.print_report(results)

    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise
    finally:
        await analyzer.close()


if __name__ == "__main__":
    asyncio.run(main())
