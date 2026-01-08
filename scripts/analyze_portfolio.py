#!/usr/bin/env python3
"""
Portfolio Analyzer - Analyze your Groww holdings with AI.

This script connects to your Groww account, fetches your holdings,
and uses OpenAI to analyze each stock and provide recommendations.

Usage:
    python scripts/analyze_portfolio.py                    # Interactive mode
    python scripts/analyze_portfolio.py --watchlist        # Analyze watchlist for new buys
    python scripts/analyze_portfolio.py --symbol RELIANCE  # Analyze specific stock
"""

import argparse
import asyncio
import sys

from loguru import logger

# Add project root to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from config.settings import settings
from src.analyzer.portfolio_analyzer import PortfolioAnalyzer


async def analyze_portfolio(groww_key: str = None, openai_key: str = None):
    """Analyze your Groww portfolio."""

    print("\n" + "=" * 60)
    print("📊 PORTFOLIO ANALYZER - AI-Powered Stock Recommendations")
    print("=" * 60)

    # Get API keys
    groww_api_key = groww_key or (
        settings.groww_api_key.get_secret_value()
        if settings.groww_api_key else None
    )
    openai_api_key = openai_key or (
        settings.openai_api_key.get_secret_value()
        if settings.openai_api_key else None
    )

    if not openai_api_key:
        print("\n❌ OpenAI API key required!")
        print("   Set OPENAI_API_KEY in .env or pass via --openai-key")
        return

    # Initialize analyzer
    analyzer = PortfolioAnalyzer(
        groww_api_key=groww_api_key,
        openai_api_key=openai_api_key,
        openai_model=settings.openai_model,
    )

    try:
        # Try to connect to Groww
        connected = await analyzer.initialize()

        if connected:
            print("✓ Connected to Groww API")
        else:
            print("ℹ Groww API not available - using manual input")

        # Analyze portfolio
        summary = await analyzer.analyze_portfolio()

        # Print report
        analyzer.print_report(summary)

    finally:
        await analyzer.close()


async def analyze_watchlist(openai_key: str = None):
    """Analyze watchlist for potential buys."""

    print("\n" + "=" * 60)
    print("🔍 WATCHLIST ANALYZER - Find Best Stocks to Buy")
    print("=" * 60)

    openai_api_key = openai_key or (
        settings.openai_api_key.get_secret_value()
        if settings.openai_api_key else None
    )

    if not openai_api_key:
        print("\n❌ OpenAI API key required!")
        return

    analyzer = PortfolioAnalyzer(
        openai_api_key=openai_api_key,
        openai_model=settings.openai_model,
    )

    try:
        watchlist = settings.default_watchlist[:10]  # Top 10
        print(f"\nAnalyzing {len(watchlist)} stocks: {', '.join(watchlist)}")

        analyses = await analyzer.analyze_watchlist(watchlist)

        # Print results
        print("\n" + "=" * 60)
        print("📈 WATCHLIST RECOMMENDATIONS (Sorted by Strength)")
        print("=" * 60)

        for analysis in analyses:
            print(f"\n{analysis}")
            print(f"   {analysis.reasoning[:100]}...")

    finally:
        await analyzer.close()


async def analyze_single_stock(symbol: str, openai_key: str = None):
    """Analyze a single stock."""

    print(f"\n🔍 Analyzing {symbol.upper()}...")

    openai_api_key = openai_key or (
        settings.openai_api_key.get_secret_value()
        if settings.openai_api_key else None
    )

    if not openai_api_key:
        print("\n❌ OpenAI API key required!")
        return

    from src.analyzer.openai_analyzer import OpenAIStockAnalyzer
    from src.analyzer.portfolio_analyzer import COMPANY_NAMES

    analyzer = OpenAIStockAnalyzer(
        api_key=openai_api_key,
        model=settings.openai_model,
    )

    try:
        company_name = COMPANY_NAMES.get(symbol.upper(), symbol.upper())

        # Get current price (optional)
        price = None
        try:
            price_input = input(f"Current price of {symbol} (press Enter to skip): ").strip()
            if price_input:
                price = float(price_input)
        except ValueError:
            pass

        analysis = await analyzer.analyze_stock(
            symbol=symbol.upper(),
            company_name=company_name,
            current_price=price,
        )

        # Print detailed analysis
        print("\n" + "=" * 60)
        print(f"📊 {analysis.symbol} - {analysis.company_name}")
        print("=" * 60)

        rec_emoji = {
            "STRONG_BUY": "🟢🟢",
            "BUY": "🟢",
            "HOLD": "🟡",
            "SELL": "🔴",
            "STRONG_SELL": "🔴🔴",
        }

        print(f"\n{rec_emoji.get(analysis.recommendation.value, '')} Recommendation: {analysis.recommendation.value}")
        print(f"Confidence: {analysis.confidence:.0f}%")

        if analysis.target_price:
            print(f"Target Price: ₹{analysis.target_price:,.2f}")
        if analysis.stop_loss:
            print(f"Stop Loss: ₹{analysis.stop_loss:,.2f}")

        print(f"\n📰 Sentiment: {analysis.sentiment_summary}")
        print(f"📈 Technical: {analysis.technical_summary}")
        print(f"📊 Fundamental: {analysis.fundamental_summary}")

        if analysis.risks:
            print(f"\n⚠️  Risks: {', '.join(analysis.risks)}")
        if analysis.catalysts:
            print(f"✨ Catalysts: {', '.join(analysis.catalysts)}")

        print(f"\n💡 Analysis:\n{analysis.reasoning}")

    finally:
        await analyzer.close()


def main():
    """Main entry point."""

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<dim>{time:HH:mm:ss}</dim> | {message}",
        level="INFO",
    )

    parser = argparse.ArgumentParser(
        description="AI-powered stock portfolio analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/analyze_portfolio.py                     # Analyze your portfolio
    python scripts/analyze_portfolio.py --watchlist         # Find stocks to buy
    python scripts/analyze_portfolio.py --symbol RELIANCE   # Analyze one stock
    python scripts/analyze_portfolio.py --openai-key sk-... # Pass API key directly
        """,
    )

    parser.add_argument(
        "--watchlist",
        action="store_true",
        help="Analyze watchlist for potential buys instead of portfolio",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Analyze a specific stock symbol",
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

    if args.symbol:
        asyncio.run(analyze_single_stock(args.symbol, args.openai_key))
    elif args.watchlist:
        asyncio.run(analyze_watchlist(args.openai_key))
    else:
        asyncio.run(analyze_portfolio(args.groww_key, args.openai_key))


if __name__ == "__main__":
    main()
