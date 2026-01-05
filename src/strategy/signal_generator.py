"""
Signal generator that orchestrates multiple strategies and data sources.

This is the main entry point for generating trading signals.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from loguru import logger

from config.settings import settings
from src.data.market_data import MarketDataClient
from src.data.models import OrderSide, StockQuote
from src.data.news_fetcher import NewsFetcher
from src.data.sentiment_apis import SentimentAPIClient
from src.sentiment.aggregator import SentimentAggregator, SentimentSignal
from src.strategy.base_strategy import BaseStrategy, SignalAction, StrategySignal
from src.strategy.momentum import MeanReversionStrategy, MomentumStrategy


@dataclass
class TradingSignal:
    """
    Final trading signal ready for execution.

    Combines strategy signals with market data and risk parameters.
    """

    symbol: str
    side: OrderSide
    action: str  # "open", "close", "add", "reduce"
    quantity: Optional[int] = None  # Shares to trade
    price: Optional[float] = None  # Current price
    stop_loss: Optional[float] = None  # Stop loss price
    take_profit: Optional[float] = None  # Take profit price
    confidence: float = 0.5
    reason: str = ""
    strategy: str = ""
    sentiment_score: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class SignalGenerator:
    """
    Orchestrates the full signal generation pipeline.

    1. Fetches news and sentiment data
    2. Aggregates sentiment into signals
    3. Runs trading strategies
    4. Produces final trading signals
    """

    def __init__(
        self,
        strategies: Optional[list[BaseStrategy]] = None,
        news_fetcher: Optional[NewsFetcher] = None,
        sentiment_client: Optional[SentimentAPIClient] = None,
        market_data: Optional[MarketDataClient] = None,
        aggregator: Optional[SentimentAggregator] = None,
    ):
        """
        Initialize signal generator.

        Args:
            strategies: List of trading strategies (defaults to momentum + mean reversion)
            news_fetcher: NewsFetcher instance
            sentiment_client: SentimentAPIClient instance
            market_data: MarketDataClient instance
            aggregator: SentimentAggregator instance
        """
        # Initialize strategies
        if strategies is None:
            self.strategies = [
                MomentumStrategy(),
                MeanReversionStrategy(),
            ]
        else:
            self.strategies = strategies

        # Initialize data clients
        self.news_fetcher = news_fetcher or NewsFetcher()
        self.sentiment_client = sentiment_client or SentimentAPIClient()
        self.market_data = market_data or MarketDataClient()
        self.aggregator = aggregator or SentimentAggregator()

        # Position tracking (should be synchronized with execution module)
        self._positions: dict[str, float] = {}  # symbol -> quantity

        # Cache for recent signals
        self._last_signals: dict[str, TradingSignal] = {}

    async def close(self):
        """Close all client connections."""
        await self.news_fetcher.close()
        await self.sentiment_client.close()
        await self.market_data.close()

    def set_position(self, symbol: str, quantity: float):
        """Update position tracking."""
        self._positions[symbol] = quantity

    def get_position(self, symbol: str) -> float:
        """Get current position for a symbol."""
        return self._positions.get(symbol, 0)

    async def _fetch_sentiment_data(
        self,
        symbol: str,
    ) -> SentimentSignal:
        """
        Fetch and aggregate sentiment data for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            SentimentSignal with aggregated sentiment
        """
        # Fetch news articles
        articles = await self.news_fetcher.fetch_all_sources(symbol)

        # Fetch API sentiment
        api_sentiment = await self.sentiment_client.get_combined_sentiment(symbol)

        # Generate aggregated sentiment signal
        sentiment_signal = self.aggregator.generate_signal(
            symbol=symbol,
            articles=articles,
            api_sentiment=api_sentiment,
        )

        return sentiment_signal

    def _run_strategies(
        self,
        symbol: str,
        sentiment: SentimentSignal,
        quote: Optional[StockQuote],
        current_position: float,
    ) -> list[StrategySignal]:
        """
        Run all enabled strategies and collect signals.

        Args:
            symbol: Stock symbol
            sentiment: Aggregated sentiment signal
            quote: Current price quote
            current_position: Current position size

        Returns:
            List of strategy signals
        """
        signals = []

        for strategy in self.strategies:
            if not strategy.enabled:
                continue

            try:
                signal = strategy.analyze(
                    symbol=symbol,
                    sentiment=sentiment,
                    quote=quote,
                    current_position=current_position,
                )
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed for {symbol}: {e}")

        return signals

    def _resolve_conflicts(
        self,
        signals: list[StrategySignal],
    ) -> Optional[StrategySignal]:
        """
        Resolve conflicting signals from multiple strategies.

        Uses confidence-weighted voting.

        Args:
            signals: List of strategy signals

        Returns:
            Best signal or None if no consensus
        """
        if not signals:
            return None

        if len(signals) == 1:
            return signals[0]

        # Group signals by action type
        buy_signals = [s for s in signals if s.action in (SignalAction.BUY,)]
        sell_signals = [s for s in signals if s.action in (SignalAction.SELL,)]
        exit_signals = [
            s for s in signals if s.action in (SignalAction.CLOSE_LONG, SignalAction.CLOSE_SHORT)
        ]

        # Exit signals take priority
        if exit_signals:
            return max(exit_signals, key=lambda s: s.confidence)

        # Check for conflicting buy/sell signals
        if buy_signals and sell_signals:
            buy_confidence = sum(s.confidence for s in buy_signals)
            sell_confidence = sum(s.confidence for s in sell_signals)

            # Need clear majority (60%+) to act
            total = buy_confidence + sell_confidence
            if buy_confidence / total > 0.6:
                return max(buy_signals, key=lambda s: s.confidence)
            elif sell_confidence / total > 0.6:
                return max(sell_signals, key=lambda s: s.confidence)
            else:
                logger.info("Conflicting signals with no clear majority, holding")
                return None

        # Return highest confidence signal
        return max(signals, key=lambda s: s.confidence)

    def _create_trading_signal(
        self,
        strategy_signal: StrategySignal,
        quote: Optional[StockQuote],
        capital: float,
    ) -> TradingSignal:
        """
        Convert strategy signal to executable trading signal.

        Args:
            strategy_signal: Signal from strategy
            quote: Current price quote
            capital: Available capital

        Returns:
            TradingSignal ready for execution
        """
        # Determine side and action
        if strategy_signal.action == SignalAction.BUY:
            side = OrderSide.BUY
            action = "open"
        elif strategy_signal.action == SignalAction.SELL:
            side = OrderSide.SELL
            action = "open"
        elif strategy_signal.action == SignalAction.CLOSE_LONG:
            side = OrderSide.SELL
            action = "close"
        elif strategy_signal.action == SignalAction.CLOSE_SHORT:
            side = OrderSide.BUY
            action = "close"
        else:
            side = OrderSide.BUY
            action = "hold"

        # Calculate quantity based on position size
        quantity = None
        stop_loss = None
        take_profit = None
        price = quote.last_price if quote else None

        if price and strategy_signal.suggested_size_percent:
            # Calculate position value
            position_value = capital * (strategy_signal.suggested_size_percent / 100)
            quantity = int(position_value / price)

            # Calculate stop loss and take profit prices
            if strategy_signal.stop_loss_percent:
                if side == OrderSide.BUY:
                    stop_loss = price * (1 - strategy_signal.stop_loss_percent / 100)
                else:
                    stop_loss = price * (1 + strategy_signal.stop_loss_percent / 100)

            if strategy_signal.take_profit_percent:
                if side == OrderSide.BUY:
                    take_profit = price * (1 + strategy_signal.take_profit_percent / 100)
                else:
                    take_profit = price * (1 - strategy_signal.take_profit_percent / 100)

        return TradingSignal(
            symbol=strategy_signal.symbol,
            side=side,
            action=action,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=strategy_signal.confidence,
            reason=strategy_signal.reason,
            strategy=strategy_signal.strategy_type.value,
            sentiment_score=strategy_signal.sentiment_score or 0.0,
        )

    async def generate_signal(
        self,
        symbol: str,
        capital: Optional[float] = None,
    ) -> Optional[TradingSignal]:
        """
        Generate a trading signal for a symbol.

        Full pipeline:
        1. Fetch news and sentiment data
        2. Aggregate into sentiment signal
        3. Run trading strategies
        4. Resolve conflicts
        5. Create executable trading signal

        Args:
            symbol: Stock symbol
            capital: Available capital for position sizing

        Returns:
            TradingSignal or None if no action recommended
        """
        capital = capital or settings.initial_capital

        logger.info(f"Generating signal for {symbol}")

        try:
            # Step 1 & 2: Fetch and aggregate sentiment
            sentiment = await self._fetch_sentiment_data(symbol)

            if not sentiment.is_actionable:
                logger.debug(f"{symbol}: Sentiment not actionable")
                return None

            # Get current price
            quote = await self.market_data.get_quote(symbol)

            # Get current position
            current_position = self.get_position(symbol)

            # Step 3: Run strategies
            strategy_signals = self._run_strategies(
                symbol=symbol,
                sentiment=sentiment,
                quote=quote,
                current_position=current_position,
            )

            if not strategy_signals:
                logger.debug(f"{symbol}: No strategy signals generated")
                return None

            # Step 4: Resolve conflicts
            best_signal = self._resolve_conflicts(strategy_signals)

            if not best_signal:
                logger.debug(f"{symbol}: No consensus signal")
                return None

            # Step 5: Create trading signal
            trading_signal = self._create_trading_signal(
                strategy_signal=best_signal,
                quote=quote,
                capital=capital,
            )

            # Cache the signal
            self._last_signals[symbol] = trading_signal

            logger.info(
                f"Generated trading signal for {symbol}: "
                f"{trading_signal.side.value} {trading_signal.quantity} shares "
                f"@ {trading_signal.price} ({trading_signal.reason})"
            )

            return trading_signal

        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            return None

    async def generate_signals_batch(
        self,
        symbols: list[str],
        capital: Optional[float] = None,
    ) -> dict[str, TradingSignal]:
        """
        Generate signals for multiple symbols.

        Args:
            symbols: List of stock symbols
            capital: Available capital

        Returns:
            Dictionary mapping symbols to trading signals
        """
        signals = {}

        for symbol in symbols:
            signal = await self.generate_signal(symbol, capital)
            if signal:
                signals[symbol] = signal

        return signals

    def get_last_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Get the most recent signal for a symbol."""
        return self._last_signals.get(symbol)

    def get_all_signals(self) -> dict[str, TradingSignal]:
        """Get all cached signals."""
        return self._last_signals.copy()

    def enable_strategy(self, strategy_type: str):
        """Enable a strategy by type name."""
        for strategy in self.strategies:
            if strategy.strategy_type.value == strategy_type:
                strategy.enable()
                logger.info(f"Enabled strategy: {strategy.name}")

    def disable_strategy(self, strategy_type: str):
        """Disable a strategy by type name."""
        for strategy in self.strategies:
            if strategy.strategy_type.value == strategy_type:
                strategy.disable()
                logger.info(f"Disabled strategy: {strategy.name}")
