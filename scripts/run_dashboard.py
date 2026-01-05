#!/usr/bin/env python3
"""
Dashboard server script.

Run with: python scripts/run_dashboard.py
"""

import sys

import uvicorn

# Add project root to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from config.settings import TradingMode, settings
from src.dashboard.api import create_app
from src.execution.order_manager import OrderManager
from src.risk.risk_manager import RiskManager
from src.strategy.signal_generator import SignalGenerator


def main():
    """Run dashboard server."""
    print("=" * 60)
    print("SENTIMENT ALGO TRADER - DASHBOARD")
    print("=" * 60)

    # Initialize components
    signal_generator = SignalGenerator()
    order_manager = OrderManager(mode=TradingMode.PAPER)
    risk_manager = RiskManager()

    # Create app with components
    app = create_app(
        signal_generator=signal_generator,
        order_manager=order_manager,
        risk_manager=risk_manager,
    )

    print(f"\nStarting dashboard at http://{settings.dashboard_host}:{settings.dashboard_port}")
    print("API docs available at /docs")
    print("\nPress Ctrl+C to stop\n")

    uvicorn.run(
        app,
        host=settings.dashboard_host,
        port=settings.dashboard_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
