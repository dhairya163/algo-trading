"""
Sentiment aggregation module for combining multiple sentiment signals.

Aggregates sentiment from:
- News articles (analyzed with NLP)
- Pre-computed API sentiment scores
- Social media sentiment
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from loguru import logger

from config.settings import settings
from src.data.models import DataSource, NewsArticle, SentimentData
from src.sentiment.analyzer import SentimentAnalyzer, SentimentResult


@dataclass
class SentimentSignal:
    """A single sentiment signal for trading decisions."""

    symbol: str
    timestamp: datetime
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    signal_strength: float  # 0 to 1 (how actionable)
    trend: str  # "improving", "stable", "deteriorating"
    sources_count: int
    recommendation: str  # "strong_buy", "buy", "hold", "sell", "strong_sell"

    @property
    def is_actionable(self) -> bool:
        """Check if signal is strong enough to act on."""
        return self.signal_strength >= 0.3 and self.confidence >= 0.4


@dataclass
class SentimentHistory:
    """Historical sentiment data for a symbol."""

    symbol: str
    scores: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    max_history: int = 100

    def add(self, score: float, timestamp: Optional[datetime] = None):
        """Add a sentiment score to history."""
        self.scores.append(score)
        self.timestamps.append(timestamp or datetime.now())

        # Trim to max history
        if len(self.scores) > self.max_history:
            self.scores = self.scores[-self.max_history :]
            self.timestamps = self.timestamps[-self.max_history :]

    def get_recent(self, hours: int = 24) -> list[float]:
        """Get scores from the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [s for s, t in zip(self.scores, self.timestamps) if t >= cutoff]

    def get_trend(self) -> str:
        """Calculate trend direction."""
        recent = self.get_recent(hours=6)
        if len(recent) < 2:
            return "stable"

        # Compare recent average to older average
        mid = len(recent) // 2
        recent_avg = np.mean(recent[mid:]) if recent[mid:] else 0
        older_avg = np.mean(recent[:mid]) if recent[:mid] else 0

        diff = recent_avg - older_avg
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "deteriorating"
        return "stable"

    def get_moving_average(self, window: int = 3) -> float:
        """Get simple moving average of recent scores."""
        recent = self.scores[-window:] if self.scores else [0]
        return np.mean(recent)


class SentimentAggregator:
    """
    Aggregates sentiment from multiple sources into actionable signals.

    Combines:
    - NLP-analyzed news articles
    - Pre-computed API sentiment
    - Historical sentiment trends
    """

    def __init__(self, analyzer: Optional[SentimentAnalyzer] = None):
        """
        Initialize aggregator.

        Args:
            analyzer: SentimentAnalyzer instance (creates one if not provided)
        """
        self.analyzer = analyzer or SentimentAnalyzer()
        self._history: dict[str, SentimentHistory] = defaultdict(SentimentHistory)
        self._last_signals: dict[str, SentimentSignal] = {}

    def _calculate_signal_strength(
        self,
        score: float,
        confidence: float,
        sources_count: int,
        trend: str,
    ) -> float:
        """
        Calculate how strong/actionable a signal is.

        Factors:
        - Magnitude of sentiment score
        - Confidence in the score
        - Number of data sources
        - Trend alignment
        """
        # Base strength from score magnitude
        magnitude = abs(score)

        # Confidence factor
        confidence_factor = confidence

        # Source diversity factor (more sources = more reliable)
        source_factor = min(sources_count / 3, 1.0)  # Max out at 3 sources

        # Trend alignment factor
        trend_factor = 1.0
        if trend == "improving" and score > 0:
            trend_factor = 1.2  # Bonus for aligned positive trend
        elif trend == "deteriorating" and score < 0:
            trend_factor = 1.2  # Bonus for aligned negative trend
        elif trend == "improving" and score < 0:
            trend_factor = 0.8  # Penalty for conflicting signals
        elif trend == "deteriorating" and score > 0:
            trend_factor = 0.8

        # Combined strength
        strength = magnitude * confidence_factor * source_factor * trend_factor

        return min(strength, 1.0)

    def _get_recommendation(self, score: float, strength: float) -> str:
        """Get trading recommendation based on sentiment."""
        if strength < 0.3:
            return "hold"  # Signal too weak

        if score >= 0.5:
            return "strong_buy"
        elif score >= 0.2:
            return "buy"
        elif score <= -0.5:
            return "strong_sell"
        elif score <= -0.2:
            return "sell"
        else:
            return "hold"

    def aggregate_articles(
        self,
        articles: list[NewsArticle],
        symbol: str,
    ) -> tuple[float, float, int]:
        """
        Aggregate sentiment from multiple news articles.

        Args:
            articles: List of NewsArticle objects
            symbol: Stock symbol

        Returns:
            Tuple of (average_score, confidence, article_count)
        """
        if not articles:
            return 0.0, 0.0, 0

        scores = []
        confidences = []

        for article in articles:
            # Use pre-computed sentiment if available
            if article.sentiment_score is not None:
                scores.append(article.sentiment_score)
                confidences.append(0.7)  # API scores have moderate confidence
            else:
                # Analyze with NLP
                result = self.analyzer.analyze_article(article)
                scores.append(result.score)
                confidences.append(result.confidence)

        # Weighted average by confidence
        total_confidence = sum(confidences)
        if total_confidence > 0:
            weighted_score = sum(s * c for s, c in zip(scores, confidences)) / total_confidence
            avg_confidence = total_confidence / len(confidences)
        else:
            weighted_score = np.mean(scores)
            avg_confidence = 0.5

        return weighted_score, avg_confidence, len(articles)

    def aggregate_api_sentiment(
        self,
        api_data: SentimentData,
    ) -> tuple[float, float]:
        """
        Process pre-computed API sentiment data.

        Args:
            api_data: SentimentData from API

        Returns:
            Tuple of (score, confidence)
        """
        return api_data.combined_sentiment, api_data.confidence

    def generate_signal(
        self,
        symbol: str,
        articles: Optional[list[NewsArticle]] = None,
        api_sentiment: Optional[SentimentData] = None,
    ) -> SentimentSignal:
        """
        Generate a trading signal from all available sentiment data.

        Args:
            symbol: Stock symbol
            articles: Optional list of news articles
            api_sentiment: Optional pre-computed API sentiment

        Returns:
            SentimentSignal with aggregated sentiment and recommendation
        """
        scores = []
        weights = []
        sources_count = 0

        # Process news articles
        if articles:
            article_score, article_conf, count = self.aggregate_articles(articles, symbol)
            if count > 0:
                scores.append(article_score)
                weights.append(article_conf * 0.6)  # 60% weight for news
                sources_count += 1

        # Process API sentiment
        if api_sentiment:
            api_score, api_conf = self.aggregate_api_sentiment(api_sentiment)
            if api_conf > 0:
                scores.append(api_score)
                weights.append(api_conf * 0.4)  # 40% weight for API
                sources_count += 1

        # Calculate weighted average
        if scores and sum(weights) > 0:
            total_weight = sum(weights)
            final_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            final_confidence = total_weight / len(weights)
        else:
            final_score = 0.0
            final_confidence = 0.0

        # Update history
        history = self._history[symbol]
        history.symbol = symbol
        history.add(final_score)

        # Apply smoothing
        smoothed_score = history.get_moving_average(settings.sentiment_smoothing_window)

        # Get trend
        trend = history.get_trend()

        # Calculate signal strength
        strength = self._calculate_signal_strength(
            smoothed_score, final_confidence, sources_count, trend
        )

        # Get recommendation
        recommendation = self._get_recommendation(smoothed_score, strength)

        signal = SentimentSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            sentiment_score=smoothed_score,
            confidence=final_confidence,
            signal_strength=strength,
            trend=trend,
            sources_count=sources_count,
            recommendation=recommendation,
        )

        # Cache signal
        self._last_signals[symbol] = signal

        logger.info(
            f"Generated signal for {symbol}: score={smoothed_score:.3f}, "
            f"strength={strength:.3f}, recommendation={recommendation}"
        )

        return signal

    def get_last_signal(self, symbol: str) -> Optional[SentimentSignal]:
        """Get the most recent signal for a symbol."""
        return self._last_signals.get(symbol)

    def get_history(self, symbol: str) -> SentimentHistory:
        """Get sentiment history for a symbol."""
        return self._history.get(symbol, SentimentHistory(symbol=symbol))

    def get_all_signals(self) -> dict[str, SentimentSignal]:
        """Get all cached signals."""
        return self._last_signals.copy()

    def clear_history(self, symbol: Optional[str] = None):
        """Clear sentiment history."""
        if symbol:
            if symbol in self._history:
                del self._history[symbol]
        else:
            self._history.clear()
