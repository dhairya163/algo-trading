#!/usr/bin/env python3
"""
Command-line interface for Sentiment Algo Trader.
"""

import argparse
import asyncio
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sentiment-based Algorithmic Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  algo-trade paper          Run paper trading
  algo-trade backtest       Run backtest on historical data
  algo-trade dashboard      Start the monitoring dashboard
  algo-trade signal RELIANCE  Generate signal for a stock
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Paper trading
    paper_parser = subparsers.add_parser("paper", help="Run paper trading")
    paper_parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Minutes between signal checks",
    )

    # Backtest
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument(
        "--strategy",
        choices=["momentum", "mean_reversion"],
        default="momentum",
        help="Strategy to test",
    )
    backtest_parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of historical data",
    )
    backtest_parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to backtest",
    )

    # Dashboard
    dashboard_parser = subparsers.add_parser("dashboard", help="Start dashboard")
    dashboard_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Dashboard port",
    )

    # Signal generation
    signal_parser = subparsers.add_parser("signal", help="Generate signal")
    signal_parser.add_argument("symbol", help="Stock symbol")

    args = parser.parse_args()

    if args.command == "paper":
        from scripts.run_paper import main as run_paper

        asyncio.run(run_paper())

    elif args.command == "backtest":
        from scripts.run_backtest import run_backtest

        asyncio.run(
            run_backtest(
                strategy_name=args.strategy,
                symbols=args.symbols,
                days=args.days,
            )
        )

    elif args.command == "dashboard":
        from scripts.run_dashboard import main as run_dashboard

        run_dashboard()

    elif args.command == "signal":
        asyncio.run(generate_signal(args.symbol))

    else:
        parser.print_help()


async def generate_signal(symbol: str):
    """Generate a signal for a single symbol."""
    from loguru import logger

    from src.strategy.signal_generator import SignalGenerator

    logger.remove()
    logger.add(sys.stderr, format="{message}", level="INFO")

    print(f"\nGenerating signal for {symbol.upper()}...")

    generator = SignalGenerator()

    try:
        signal = await generator.generate_signal(symbol.upper())

        if signal:
            print(f"\n{'=' * 50}")
            print(f"SIGNAL: {signal.side.value.upper()} {symbol.upper()}")
            print(f"{'=' * 50}")
            print(f"Action:     {signal.action}")
            print(f"Quantity:   {signal.quantity}")
            print(f"Price:      ₹{signal.price:,.2f}" if signal.price else "Price:      N/A")
            print(f"Confidence: {signal.confidence:.2%}")
            print(f"Stop Loss:  ₹{signal.stop_loss:,.2f}" if signal.stop_loss else "")
            print(f"Strategy:   {signal.strategy}")
            print(f"Sentiment:  {signal.sentiment_score:+.2f}")
            print(f"\nReason: {signal.reason}")
        else:
            print(f"\nNo actionable signal for {symbol.upper()}")
            print("(Sentiment may be neutral or data unavailable)")

    finally:
        await generator.close()


if __name__ == "__main__":
    main()
