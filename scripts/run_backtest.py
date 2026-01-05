#!/usr/bin/env python3
"""
Backtesting script for strategy evaluation.

Run with: python scripts/run_backtest.py
"""

import asyncio
import sys
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from config.settings import settings
from src.backtest.backtester import Backtester
from src.data.market_data import MarketDataClient
from src.strategy.momentum import MeanReversionStrategy, MomentumStrategy


async def fetch_historical_data(
    symbols: list[str],
    days: int = 365,
) -> dict[str, pd.DataFrame]:
    """
    Fetch historical data for backtesting.

    Args:
        symbols: List of stock symbols
        days: Number of days of history

    Returns:
        Dictionary of symbol -> DataFrame
    """
    market_data = MarketDataClient()
    data = {}

    start_date = datetime.now() - timedelta(days=days)

    for symbol in symbols:
        logger.info(f"Fetching historical data for {symbol}...")
        try:
            prices = await market_data.get_historical_prices(
                symbol=symbol,
                start_date=start_date,
                interval="1d",
            )

            if prices:
                df = pd.DataFrame([
                    {
                        "timestamp": p.timestamp,
                        "open": p.open,
                        "high": p.high,
                        "low": p.low,
                        "close": p.close,
                        "volume": p.volume,
                    }
                    for p in prices
                ])
                df = df.sort_values("timestamp").reset_index(drop=True)
                data[symbol] = df
                logger.info(f"  ✓ {len(df)} bars for {symbol}")

        except Exception as e:
            logger.error(f"  ✗ Failed to fetch {symbol}: {e}")

    await market_data.close()
    return data


async def run_backtest(
    strategy_name: str = "momentum",
    symbols: list[str] = None,
    days: int = 365,
    initial_capital: float = None,
):
    """
    Run a backtest.

    Args:
        strategy_name: Strategy to test ("momentum" or "mean_reversion")
        symbols: Symbols to backtest
        days: Days of historical data
        initial_capital: Starting capital
    """
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )

    logger.info("=" * 60)
    logger.info(f"BACKTEST: {strategy_name.upper()} STRATEGY")
    logger.info("=" * 60)

    # Get symbols
    symbols = symbols or settings.default_watchlist[:10]  # Limit for demo
    capital = initial_capital or settings.initial_capital

    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Period: {days} days")
    logger.info(f"Capital: ₹{capital:,.2f}")

    # Fetch historical data
    logger.info("\nFetching historical data...")
    price_data = await fetch_historical_data(symbols, days)

    if not price_data:
        logger.error("No historical data available")
        return

    # Select strategy
    if strategy_name == "momentum":
        strategy = MomentumStrategy(
            entry_threshold=0.25,
            exit_threshold=0.1,
            min_confidence=0.4,
        )
    elif strategy_name == "mean_reversion":
        strategy = MeanReversionStrategy(
            extreme_threshold=0.6,
            reversion_target=0.2,
            min_confidence=0.5,
        )
    else:
        logger.error(f"Unknown strategy: {strategy_name}")
        return

    # Run backtest
    logger.info(f"\nRunning backtest with {strategy.name}...")
    backtester = Backtester(
        strategy=strategy,
        initial_capital=capital,
        commission_percent=0.03,
        slippage_percent=0.05,
    )

    result = await backtester.run(price_data)

    # Print results
    print(result.summary())

    # Save detailed results
    if result.trades:
        trades_df = pd.DataFrame([
            {
                "symbol": t.symbol,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl": t.pnl,
                "pnl_percent": t.pnl_percent,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
            }
            for t in result.trades
        ])
        trades_file = f"backtest_trades_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(trades_file, index=False)
        logger.info(f"Trade details saved to: {trades_file}")

    return result


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument(
        "--strategy",
        choices=["momentum", "mean_reversion"],
        default="momentum",
        help="Strategy to test",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of historical data",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Initial capital",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to backtest",
    )

    args = parser.parse_args()

    await run_backtest(
        strategy_name=args.strategy,
        symbols=args.symbols,
        days=args.days,
        initial_capital=args.capital,
    )


if __name__ == "__main__":
    asyncio.run(main())
