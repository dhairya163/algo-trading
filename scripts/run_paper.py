#!/usr/bin/env python3
"""
Paper trading script for testing strategies.

Run with: python scripts/run_paper.py
"""

import asyncio
import signal
import sys
from datetime import datetime

from loguru import logger

# Add project root to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from config.settings import TradingMode, settings
from src.execution.order_manager import OrderManager
from src.risk.risk_manager import RiskManager
from src.strategy.signal_generator import SignalGenerator


async def main():
    """Main paper trading loop."""

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )
    logger.add(
        "logs/paper_trading.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
    )

    logger.info("=" * 60)
    logger.info("SENTIMENT ALGO TRADER - PAPER TRADING MODE")
    logger.info("=" * 60)

    # Force paper trading mode
    if settings.trading_mode != TradingMode.PAPER:
        logger.warning("Forcing paper trading mode")

    # Initialize components
    signal_generator = SignalGenerator()
    order_manager = OrderManager(mode=TradingMode.PAPER)
    risk_manager = RiskManager()

    # Shutdown handler
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await order_manager.initialize()

        watchlist = settings.default_watchlist
        logger.info(f"Monitoring {len(watchlist)} symbols: {', '.join(watchlist[:5])}...")

        iteration = 0
        while not shutdown_event.is_set():
            iteration += 1
            logger.info(f"\n{'─' * 40}")
            logger.info(f"Iteration {iteration} at {datetime.now().strftime('%H:%M:%S')}")

            # Generate signals for watchlist
            for symbol in watchlist:
                try:
                    # Generate signal
                    trading_signal = await signal_generator.generate_signal(symbol)

                    if trading_signal:
                        logger.info(
                            f"Signal: {trading_signal.side.value.upper()} {symbol} "
                            f"(confidence: {trading_signal.confidence:.2f})"
                        )

                        # Validate against risk manager
                        is_valid, reason = risk_manager.validate_trade(trading_signal)

                        if is_valid:
                            # Execute signal
                            result = await order_manager.execute_signal(trading_signal)
                            if result.success:
                                logger.info(f"✓ Executed: {result.order.id}")
                            else:
                                logger.warning(f"✗ Failed: {result.message}")
                        else:
                            logger.warning(f"✗ Risk check failed: {reason}")

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

            # Update risk manager with current state
            positions = await order_manager.get_positions()
            summary = order_manager.paper_trader.get_portfolio_summary()
            risk_manager.update_state(
                cash=summary["cash"],
                positions=positions,
            )

            # Print portfolio summary
            logger.info(f"\n📊 Portfolio Summary:")
            logger.info(f"   Cash: ₹{summary['cash']:,.2f}")
            logger.info(f"   Positions: ₹{summary['positions_value']:,.2f}")
            logger.info(f"   Total: ₹{summary['total_value']:,.2f}")
            logger.info(f"   P&L: ₹{summary['total_pnl']:,.2f} ({summary['total_pnl_percent']:+.2f}%)")

            # Check risk status
            risk_report = risk_manager.get_status_report()
            if risk_report["breached_limits"]:
                logger.warning(f"⚠️  Risk limits breached: {risk_report['breached_limits']}")

            # Wait before next iteration
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=settings.signal_check_interval_minutes * 60,
                )
            except asyncio.TimeoutError:
                pass  # Continue to next iteration

    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        raise

    finally:
        logger.info("Shutting down...")
        await signal_generator.close()
        await order_manager.close()

        # Print final summary
        summary = order_manager.paper_trader.get_portfolio_summary()
        logger.info("\n" + "=" * 60)
        logger.info("FINAL PAPER TRADING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Final Capital: ₹{summary['total_value']:,.2f}")
        logger.info(f"Total P&L: ₹{summary['total_pnl']:,.2f} ({summary['total_pnl_percent']:+.2f}%)")
        logger.info(f"Total Trades: {summary['trades_count']}")


if __name__ == "__main__":
    asyncio.run(main())
