"""
Momentum-based sentiment trading strategy.

This strategy follows the direction of sentiment:
- Buy when sentiment is positive and improving
- Sell when sentiment is negative and deteriorating
"""

from datetime import datetime
from typing import Optional

from loguru import logger

from config.settings import settings
from src.data.models import StockQuote
from src.sentiment.aggregator import SentimentSignal
from src.strategy.base_strategy import (
    BaseStrategy,
    SignalAction,
    StrategySignal,
    StrategyType,
)


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy based on sentiment direction.

    Entry Conditions:
    - Long: Sentiment > threshold AND (improving OR stable)
    - Short: Sentiment < -threshold AND (deteriorating OR stable)

    Exit Conditions:
    - Close Long: Sentiment drops below exit threshold OR deteriorating trend
    - Close Short: Sentiment rises above exit threshold OR improving trend
    """

    def __init__(
        self,
        entry_threshold: float = 0.25,
        exit_threshold: float = 0.1,
        min_confidence: float = 0.4,
        require_trend_alignment: bool = True,
        position_size_percent: float = 5.0,
        stop_loss_percent: float = 3.0,
        take_profit_percent: float = 6.0,
    ):
        """
        Initialize momentum strategy.

        Args:
            entry_threshold: Minimum sentiment score to enter position
            exit_threshold: Sentiment threshold to exit position
            min_confidence: Minimum confidence for signals
            require_trend_alignment: Require trend to match sentiment direction
            position_size_percent: Default position size as % of portfolio
            stop_loss_percent: Default stop loss percentage
            take_profit_percent: Default take profit percentage
        """
        super().__init__(
            name="Sentiment Momentum",
            strategy_type=StrategyType.MOMENTUM,
            min_confidence=min_confidence,
        )

        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.require_trend_alignment = require_trend_alignment
        self.position_size_percent = position_size_percent
        self.stop_loss_percent = stop_loss_percent
        self.take_profit_percent = take_profit_percent

    def _is_trend_aligned(self, sentiment: SentimentSignal, direction: str) -> bool:
        """Check if trend aligns with desired direction."""
        if not self.require_trend_alignment:
            return True

        if direction == "long":
            return sentiment.trend in ("improving", "stable")
        else:  # short
            return sentiment.trend in ("deteriorating", "stable")

    def _calculate_position_size(self, sentiment: SentimentSignal) -> float:
        """
        Calculate position size based on signal strength.

        Stronger signals get larger positions (up to max).
        """
        base_size = self.position_size_percent

        # Scale by signal strength (0.5x to 1.5x)
        strength_multiplier = 0.5 + sentiment.signal_strength

        # Scale by confidence
        confidence_multiplier = 0.7 + (sentiment.confidence * 0.3)

        adjusted_size = base_size * strength_multiplier * confidence_multiplier

        # Cap at max position size from settings
        max_size = settings.max_position_size_percent
        return min(adjusted_size, max_size)

    def _generate_reason(
        self,
        action: SignalAction,
        sentiment: SentimentSignal,
    ) -> str:
        """Generate human-readable reason for the signal."""
        score = sentiment.sentiment_score
        trend = sentiment.trend
        strength = sentiment.signal_strength

        if action == SignalAction.BUY:
            return (
                f"Bullish momentum: sentiment={score:.2f} ({trend}), "
                f"strength={strength:.2f}, recommendation={sentiment.recommendation}"
            )
        elif action == SignalAction.SELL:
            return (
                f"Bearish momentum: sentiment={score:.2f} ({trend}), "
                f"strength={strength:.2f}, recommendation={sentiment.recommendation}"
            )
        elif action == SignalAction.CLOSE_LONG:
            return f"Exit long: sentiment weakened to {score:.2f}, trend={trend}"
        elif action == SignalAction.CLOSE_SHORT:
            return f"Exit short: sentiment strengthened to {score:.2f}, trend={trend}"
        else:
            return f"Hold: sentiment={score:.2f}, no clear signal"

    def analyze(
        self,
        symbol: str,
        sentiment: SentimentSignal,
        quote: Optional[StockQuote] = None,
        current_position: Optional[float] = None,
    ) -> Optional[StrategySignal]:
        """
        Analyze sentiment and generate momentum-based trading signal.

        Args:
            symbol: Stock symbol
            sentiment: Current sentiment signal
            quote: Optional current price quote
            current_position: Current position size (+ long, - short, 0 or None = flat)

        Returns:
            StrategySignal or None
        """
        if not self.enabled:
            return None

        # Check confidence threshold
        if sentiment.confidence < self.min_confidence:
            logger.debug(
                f"{symbol}: Sentiment confidence {sentiment.confidence:.2f} "
                f"below threshold {self.min_confidence}"
            )
            return None

        score = sentiment.sentiment_score
        has_position = current_position and current_position != 0
        is_long = current_position and current_position > 0
        is_short = current_position and current_position < 0

        action = None
        confidence = sentiment.confidence

        # Check for exit signals first (if we have a position)
        if is_long:
            # Check if we should close long
            if score < self.exit_threshold or sentiment.trend == "deteriorating":
                action = SignalAction.CLOSE_LONG
                confidence *= 0.9  # Slightly lower confidence for exits
        elif is_short:
            # Check if we should close short
            if score > -self.exit_threshold or sentiment.trend == "improving":
                action = SignalAction.CLOSE_SHORT
                confidence *= 0.9

        # Check for entry signals (if no position or no exit triggered)
        if action is None:
            if score >= self.entry_threshold and self._is_trend_aligned(sentiment, "long"):
                if not is_long:  # Don't add to existing long
                    action = SignalAction.BUY
            elif score <= -self.entry_threshold and self._is_trend_aligned(sentiment, "short"):
                if not is_short:  # Don't add to existing short
                    action = SignalAction.SELL

        # No action needed
        if action is None:
            return None

        # Calculate position size for entries
        suggested_size = None
        stop_loss = None
        take_profit = None

        if action in (SignalAction.BUY, SignalAction.SELL):
            suggested_size = self._calculate_position_size(sentiment)
            stop_loss = self.stop_loss_percent
            take_profit = self.take_profit_percent

            # Adjust stop/take profit based on volatility if we have price data
            if quote:
                # Could adjust based on ATR or recent price range
                pass

        signal = StrategySignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            reason=self._generate_reason(action, sentiment),
            timestamp=datetime.now(),
            strategy_type=self.strategy_type,
            suggested_size_percent=suggested_size,
            stop_loss_percent=stop_loss,
            take_profit_percent=take_profit,
            sentiment_score=score,
            sentiment_trend=sentiment.trend,
        )

        self._cache_signal(signal)

        logger.info(
            f"Momentum signal for {symbol}: {action.value} "
            f"(confidence={confidence:.2f}, sentiment={score:.2f})"
        )

        return signal


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy based on extreme sentiment.

    Fades extreme sentiment readings assuming reversion to mean.
    - Buy when sentiment is extremely negative (oversold)
    - Sell when sentiment is extremely positive (overbought)
    """

    def __init__(
        self,
        extreme_threshold: float = 0.6,
        reversion_target: float = 0.2,
        min_confidence: float = 0.5,
        position_size_percent: float = 3.0,
    ):
        """
        Initialize mean reversion strategy.

        Args:
            extreme_threshold: Sentiment level considered extreme
            reversion_target: Target sentiment for exit
            min_confidence: Minimum confidence threshold
            position_size_percent: Default position size
        """
        super().__init__(
            name="Sentiment Mean Reversion",
            strategy_type=StrategyType.MEAN_REVERSION,
            min_confidence=min_confidence,
        )

        self.extreme_threshold = extreme_threshold
        self.reversion_target = reversion_target
        self.position_size_percent = position_size_percent

    def analyze(
        self,
        symbol: str,
        sentiment: SentimentSignal,
        quote: Optional[StockQuote] = None,
        current_position: Optional[float] = None,
    ) -> Optional[StrategySignal]:
        """
        Analyze sentiment for mean reversion opportunities.

        Args:
            symbol: Stock symbol
            sentiment: Current sentiment signal
            quote: Optional current price quote
            current_position: Current position size

        Returns:
            StrategySignal or None
        """
        if not self.enabled:
            return None

        if sentiment.confidence < self.min_confidence:
            return None

        score = sentiment.sentiment_score
        is_long = current_position and current_position > 0
        is_short = current_position and current_position < 0

        action = None
        confidence = sentiment.confidence

        # Check for extreme negative (buy opportunity)
        if score <= -self.extreme_threshold and not is_long:
            action = SignalAction.BUY
            reason = f"Mean reversion buy: extreme negative sentiment ({score:.2f})"
        # Check for extreme positive (sell opportunity)
        elif score >= self.extreme_threshold and not is_short:
            action = SignalAction.SELL
            reason = f"Mean reversion sell: extreme positive sentiment ({score:.2f})"
        # Check for exit on mean reversion
        elif is_long and score >= -self.reversion_target:
            action = SignalAction.CLOSE_LONG
            reason = f"Mean reversion complete: sentiment normalized to {score:.2f}"
        elif is_short and score <= self.reversion_target:
            action = SignalAction.CLOSE_SHORT
            reason = f"Mean reversion complete: sentiment normalized to {score:.2f}"

        if action is None:
            return None

        signal = StrategySignal(
            symbol=symbol,
            action=action,
            confidence=confidence * 0.9,  # Lower confidence for contrarian
            reason=reason,
            timestamp=datetime.now(),
            strategy_type=self.strategy_type,
            suggested_size_percent=self.position_size_percent,
            sentiment_score=score,
            sentiment_trend=sentiment.trend,
        )

        self._cache_signal(signal)
        return signal
