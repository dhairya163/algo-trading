"""
Backtesting engine for evaluating trading strategies.

Features:
- Historical data simulation
- Realistic execution (slippage, commissions)
- Performance metrics calculation
- Trade-by-trade analysis
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

from config.settings import settings
from src.backtest.metrics import PerformanceMetrics, calculate_metrics
from src.data.models import OrderSide, StockPrice
from src.execution.paper_trader import PaperTrader
from src.sentiment.aggregator import SentimentAggregator, SentimentSignal
from src.strategy.base_strategy import BaseStrategy, SignalAction
from src.strategy.momentum import MomentumStrategy


@dataclass
class BacktestTrade:
    """Record of a trade during backtesting."""

    symbol: str
    side: str
    quantity: int
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: datetime = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_percent: float = 0.0
    sentiment_at_entry: float = 0.0
    reason: str = ""

    @property
    def is_closed(self) -> bool:
        return self.exit_price is not None

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_percent: float
    metrics: PerformanceMetrics
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        """Generate summary report."""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                    BACKTEST RESULTS                          ║
╠══════════════════════════════════════════════════════════════╣
║  Period: {self.start_date.date()} to {self.end_date.date()}
║  Initial Capital: ₹{self.initial_capital:,.2f}
║  Final Capital:   ₹{self.final_capital:,.2f}
║  Total Return:    ₹{self.total_return:,.2f} ({self.total_return_percent:+.2f}%)
╠══════════════════════════════════════════════════════════════╣
║  PERFORMANCE METRICS
║  ─────────────────────────────────────────────────────────────
║  Total Trades:      {self.metrics.total_trades}
║  Win Rate:          {self.metrics.win_rate:.1%}
║  Profit Factor:     {self.metrics.profit_factor:.2f}
║  Sharpe Ratio:      {self.metrics.sharpe_ratio:.2f}
║  Sortino Ratio:     {self.metrics.sortino_ratio:.2f}
║  Max Drawdown:      {self.metrics.max_drawdown:.2f}%
║  Avg Trade:         ₹{self.metrics.avg_trade:.2f}
║  Best Trade:        ₹{self.metrics.best_trade:.2f}
║  Worst Trade:       ₹{self.metrics.worst_trade:.2f}
╚══════════════════════════════════════════════════════════════╝
"""


class Backtester:
    """
    Backtesting engine for sentiment trading strategies.

    Simulates trading based on historical price data and
    sentiment signals.
    """

    def __init__(
        self,
        strategy: Optional[BaseStrategy] = None,
        initial_capital: Optional[float] = None,
        commission_percent: float = 0.03,
        slippage_percent: float = 0.05,
    ):
        """
        Initialize backtester.

        Args:
            strategy: Trading strategy to test
            initial_capital: Starting capital
            commission_percent: Commission per trade
            slippage_percent: Slippage per trade
        """
        self.strategy = strategy or MomentumStrategy()
        self.initial_capital = initial_capital or settings.initial_capital
        self.commission_percent = commission_percent
        self.slippage_percent = slippage_percent

        # Simulation state
        self._paper_trader: Optional[PaperTrader] = None
        self._aggregator = SentimentAggregator()
        self._trades: list[BacktestTrade] = []
        self._equity_curve: list[dict] = []
        self._open_positions: dict[str, BacktestTrade] = {}

    def _reset(self):
        """Reset backtester state."""
        self._paper_trader = PaperTrader(
            initial_capital=self.initial_capital,
            slippage_percent=self.slippage_percent,
            commission_percent=self.commission_percent,
        )
        self._trades = []
        self._equity_curve = []
        self._open_positions = {}
        self._aggregator.clear_history()

    def _simulate_sentiment(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        idx: int,
        lookback: int = 5,
    ) -> SentimentSignal:
        """
        Simulate sentiment based on price momentum.

        In real backtesting, you'd use historical sentiment data.
        This is a simplified simulation for demonstration.

        Args:
            symbol: Stock symbol
            price_data: Historical price DataFrame
            idx: Current index
            lookback: Days to look back

        Returns:
            Simulated SentimentSignal
        """
        # Calculate price momentum as sentiment proxy
        if idx < lookback:
            return SentimentSignal(
                symbol=symbol,
                timestamp=price_data.iloc[idx]["timestamp"],
                sentiment_score=0.0,
                confidence=0.3,
                signal_strength=0.2,
                trend="stable",
                sources_count=1,
                recommendation="hold",
            )

        prices = price_data.iloc[idx - lookback : idx + 1]["close"].values
        returns = (prices[-1] - prices[0]) / prices[0]

        # Normalize to -1 to 1
        sentiment_score = max(-1, min(1, returns * 10))

        # Calculate trend
        recent_returns = (prices[-1] - prices[-3]) / prices[-3] if len(prices) >= 3 else 0
        if recent_returns > 0.01:
            trend = "improving"
        elif recent_returns < -0.01:
            trend = "deteriorating"
        else:
            trend = "stable"

        # Confidence based on momentum strength
        confidence = min(0.8, 0.4 + abs(sentiment_score) * 0.5)

        # Signal strength
        strength = abs(sentiment_score) * confidence

        # Recommendation
        if sentiment_score >= 0.5:
            recommendation = "strong_buy"
        elif sentiment_score >= 0.2:
            recommendation = "buy"
        elif sentiment_score <= -0.5:
            recommendation = "strong_sell"
        elif sentiment_score <= -0.2:
            recommendation = "sell"
        else:
            recommendation = "hold"

        return SentimentSignal(
            symbol=symbol,
            timestamp=price_data.iloc[idx]["timestamp"],
            sentiment_score=sentiment_score,
            confidence=confidence,
            signal_strength=strength,
            trend=trend,
            sources_count=1,
            recommendation=recommendation,
        )

    async def _process_bar(
        self,
        symbol: str,
        bar: dict,
        sentiment: SentimentSignal,
    ):
        """
        Process a single price bar.

        Args:
            symbol: Stock symbol
            bar: Price bar data
            sentiment: Sentiment signal
        """
        price = bar["close"]
        timestamp = bar["timestamp"]

        # Update paper trader prices
        self._paper_trader.set_price(symbol, price)

        # Get current position
        position = self._paper_trader.get_position(symbol)
        current_qty = position.quantity if position else 0

        # Run strategy
        signal = self.strategy.analyze(
            symbol=symbol,
            sentiment=sentiment,
            quote=None,
            current_position=current_qty,
        )

        if signal:
            if signal.action == SignalAction.BUY and symbol not in self._open_positions:
                # Open long position
                qty = int((self.initial_capital * 0.1) / price)  # 10% position
                if qty > 0:
                    order = await self._paper_trader.place_order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=qty,
                    )

                    if order and order.status.value == "filled":
                        self._open_positions[symbol] = BacktestTrade(
                            symbol=symbol,
                            side="long",
                            quantity=qty,
                            entry_price=order.average_price,
                            entry_time=timestamp,
                            sentiment_at_entry=sentiment.sentiment_score,
                            reason=signal.reason,
                        )

            elif signal.action == SignalAction.CLOSE_LONG and symbol in self._open_positions:
                # Close long position
                trade = self._open_positions[symbol]
                order = await self._paper_trader.place_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=trade.quantity,
                )

                if order and order.status.value == "filled":
                    trade.exit_price = order.average_price
                    trade.exit_time = timestamp
                    trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity
                    trade.pnl_percent = (
                        (trade.exit_price - trade.entry_price) / trade.entry_price
                    ) * 100

                    self._trades.append(trade)
                    del self._open_positions[symbol]

        # Record equity
        summary = self._paper_trader.get_portfolio_summary()
        self._equity_curve.append({
            "timestamp": timestamp,
            "equity": summary["total_value"],
            "cash": summary["cash"],
            "positions_value": summary["positions_value"],
        })

    async def run(
        self,
        price_data: dict[str, pd.DataFrame],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            price_data: Dictionary of symbol -> price DataFrame
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            BacktestResult with performance metrics
        """
        self._reset()

        logger.info(f"Starting backtest with {len(price_data)} symbols")

        # Get date range
        all_dates = set()
        for df in price_data.values():
            all_dates.update(df["timestamp"].dt.date.tolist())

        sorted_dates = sorted(all_dates)

        if start_date:
            sorted_dates = [d for d in sorted_dates if d >= start_date.date()]
        if end_date:
            sorted_dates = [d for d in sorted_dates if d <= end_date.date()]

        if not sorted_dates:
            raise ValueError("No data in specified date range")

        actual_start = datetime.combine(sorted_dates[0], datetime.min.time())
        actual_end = datetime.combine(sorted_dates[-1], datetime.min.time())

        # Process each date
        for date in sorted_dates:
            for symbol, df in price_data.items():
                # Get bar for this date
                day_data = df[df["timestamp"].dt.date == date]
                if day_data.empty:
                    continue

                bar = day_data.iloc[0].to_dict()
                idx = df[df["timestamp"].dt.date == date].index[0]

                # Simulate sentiment
                sentiment = self._simulate_sentiment(symbol, df, idx)

                # Process bar
                await self._process_bar(symbol, bar, sentiment)

        # Close any remaining positions at last price
        for symbol, trade in list(self._open_positions.items()):
            price = self._paper_trader.get_price(symbol)
            if price:
                trade.exit_price = price
                trade.exit_time = actual_end
                trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity
                trade.pnl_percent = (
                    (trade.exit_price - trade.entry_price) / trade.entry_price
                ) * 100
                self._trades.append(trade)

        # Calculate final metrics
        summary = self._paper_trader.get_portfolio_summary()
        final_capital = summary["total_value"]
        total_return = final_capital - self.initial_capital
        total_return_percent = (total_return / self.initial_capital) * 100

        # Calculate performance metrics
        metrics = calculate_metrics(
            trades=self._trades,
            equity_curve=self._equity_curve,
            initial_capital=self.initial_capital,
        )

        result = BacktestResult(
            start_date=actual_start,
            end_date=actual_end,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_percent=total_return_percent,
            metrics=metrics,
            trades=self._trades,
            equity_curve=self._equity_curve,
        )

        logger.info(f"Backtest complete: {total_return_percent:+.2f}% return")
        print(result.summary())

        return result
