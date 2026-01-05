"""
Risk manager for portfolio protection.

Implements risk controls including:
- Position limits
- Drawdown monitoring
- Correlation-based diversification
- Circuit breakers
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from config.settings import settings
from src.data.models import Position
from src.strategy.signal_generator import TradingSignal


@dataclass
class RiskMetrics:
    """Current risk metrics for the portfolio."""

    total_exposure: float  # Total position value
    exposure_percent: float  # Exposure as % of capital
    largest_position_percent: float  # Largest single position %
    position_count: int
    unrealized_pnl: float
    unrealized_pnl_percent: float
    daily_pnl: float
    daily_pnl_percent: float
    max_drawdown: float
    current_drawdown: float
    risk_score: float  # 0-100, higher = more risky


@dataclass
class RiskLimit:
    """A risk limit with current value and threshold."""

    name: str
    current_value: float
    limit_value: float
    is_breached: bool = False
    severity: str = "warning"  # "warning", "critical"

    @property
    def utilization(self) -> float:
        """Get utilization as percentage of limit."""
        if self.limit_value == 0:
            return 0
        return (self.current_value / self.limit_value) * 100


@dataclass
class PortfolioState:
    """Current state of the portfolio for risk tracking."""

    capital: float
    cash: float
    positions: list[Position] = field(default_factory=list)
    peak_value: float = 0.0
    daily_high: float = 0.0
    daily_low: float = 0.0
    daily_start_value: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)

    @property
    def total_value(self) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(p.market_value for p in self.positions)
        return self.cash + positions_value

    @property
    def total_exposure(self) -> float:
        """Calculate total market exposure."""
        return sum(abs(p.market_value) for p in self.positions)


class RiskManager:
    """
    Manages portfolio risk and enforces risk limits.

    Features:
    - Pre-trade risk validation
    - Real-time position monitoring
    - Drawdown tracking and alerts
    - Correlation-based position limits
    - Circuit breakers for extreme losses
    """

    def __init__(
        self,
        capital: Optional[float] = None,
        max_position_percent: Optional[float] = None,
        max_portfolio_risk: Optional[float] = None,
        max_daily_loss_percent: Optional[float] = None,
        max_drawdown_percent: float = 15.0,
    ):
        """
        Initialize risk manager.

        Args:
            capital: Portfolio capital
            max_position_percent: Max single position size
            max_portfolio_risk: Max portfolio risk per trade
            max_daily_loss_percent: Max daily loss before circuit breaker
            max_drawdown_percent: Max drawdown before alerts
        """
        self.capital = capital or settings.initial_capital
        self.max_position_percent = max_position_percent or settings.max_position_size_percent
        self.max_portfolio_risk = max_portfolio_risk or settings.max_portfolio_risk_percent
        self.max_daily_loss_percent = max_daily_loss_percent or settings.max_daily_loss_percent
        self.max_drawdown_percent = max_drawdown_percent

        # Portfolio state tracking
        self.state = PortfolioState(
            capital=self.capital,
            cash=self.capital,
            peak_value=self.capital,
            daily_start_value=self.capital,
            daily_high=self.capital,
            daily_low=self.capital,
        )

        # Track blocked symbols (circuit breakers)
        self._blocked_symbols: dict[str, datetime] = {}
        self._trading_halted = False
        self._halt_reason = ""

    def update_state(
        self,
        cash: float,
        positions: list[Position],
    ):
        """
        Update portfolio state with current values.

        Args:
            cash: Current cash balance
            positions: Current positions
        """
        self.state.cash = cash
        self.state.positions = positions
        self.state.last_update = datetime.now()

        total_value = self.state.total_value

        # Update peak (for drawdown calculation)
        if total_value > self.state.peak_value:
            self.state.peak_value = total_value

        # Update daily high/low
        if total_value > self.state.daily_high:
            self.state.daily_high = total_value
        if total_value < self.state.daily_low:
            self.state.daily_low = total_value

        # Check for new trading day
        now = datetime.now()
        if now.date() != self.state.last_update.date():
            self._reset_daily_metrics()

    def _reset_daily_metrics(self):
        """Reset daily tracking metrics."""
        self.state.daily_start_value = self.state.total_value
        self.state.daily_high = self.state.total_value
        self.state.daily_low = self.state.total_value
        self._blocked_symbols.clear()

        if self._trading_halted:
            logger.info("New trading day - resetting circuit breakers")
            self._trading_halted = False
            self._halt_reason = ""

    def get_risk_metrics(self) -> RiskMetrics:
        """
        Calculate current risk metrics.

        Returns:
            RiskMetrics with current portfolio risk state
        """
        total_value = self.state.total_value
        exposure = self.state.total_exposure

        # Calculate position percentages
        position_percents = []
        for pos in self.state.positions:
            if total_value > 0:
                pct = (abs(pos.market_value) / total_value) * 100
                position_percents.append(pct)

        largest_position = max(position_percents) if position_percents else 0

        # Calculate P&L
        unrealized_pnl = sum(p.unrealized_pnl for p in self.state.positions)
        unrealized_pnl_pct = (unrealized_pnl / self.capital) * 100 if self.capital > 0 else 0

        daily_pnl = total_value - self.state.daily_start_value
        daily_pnl_pct = (daily_pnl / self.state.daily_start_value) * 100 if self.state.daily_start_value > 0 else 0

        # Calculate drawdowns
        max_drawdown = 0
        if self.state.peak_value > 0:
            max_drawdown = ((self.state.peak_value - total_value) / self.state.peak_value) * 100

        current_drawdown = 0
        if self.state.daily_high > 0:
            current_drawdown = ((self.state.daily_high - total_value) / self.state.daily_high) * 100

        # Calculate risk score (0-100)
        risk_factors = [
            min(exposure / self.capital * 50, 30),  # Exposure risk (max 30)
            min(largest_position / self.max_position_percent * 20, 20),  # Concentration (max 20)
            min(abs(daily_pnl_pct) / self.max_daily_loss_percent * 25, 25),  # Daily loss (max 25)
            min(max_drawdown / self.max_drawdown_percent * 25, 25),  # Drawdown (max 25)
        ]
        risk_score = sum(risk_factors)

        return RiskMetrics(
            total_exposure=exposure,
            exposure_percent=(exposure / self.capital) * 100 if self.capital > 0 else 0,
            largest_position_percent=largest_position,
            position_count=len(self.state.positions),
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_percent=unrealized_pnl_pct,
            daily_pnl=daily_pnl,
            daily_pnl_percent=daily_pnl_pct,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            risk_score=risk_score,
        )

    def check_risk_limits(self) -> list[RiskLimit]:
        """
        Check all risk limits and return status.

        Returns:
            List of RiskLimit objects with current status
        """
        metrics = self.get_risk_metrics()
        limits = []

        # Position count limit
        limits.append(RiskLimit(
            name="Open Positions",
            current_value=metrics.position_count,
            limit_value=settings.max_open_positions,
            is_breached=metrics.position_count >= settings.max_open_positions,
            severity="warning",
        ))

        # Largest position limit
        limits.append(RiskLimit(
            name="Largest Position",
            current_value=metrics.largest_position_percent,
            limit_value=self.max_position_percent,
            is_breached=metrics.largest_position_percent > self.max_position_percent,
            severity="warning",
        ))

        # Daily loss limit
        daily_loss_limit = -self.max_daily_loss_percent
        limits.append(RiskLimit(
            name="Daily Loss",
            current_value=metrics.daily_pnl_percent,
            limit_value=daily_loss_limit,
            is_breached=metrics.daily_pnl_percent <= daily_loss_limit,
            severity="critical",
        ))

        # Drawdown limit
        limits.append(RiskLimit(
            name="Max Drawdown",
            current_value=metrics.max_drawdown,
            limit_value=self.max_drawdown_percent,
            is_breached=metrics.max_drawdown >= self.max_drawdown_percent,
            severity="critical",
        ))

        # Exposure limit (150% of capital)
        max_exposure = 150
        limits.append(RiskLimit(
            name="Total Exposure",
            current_value=metrics.exposure_percent,
            limit_value=max_exposure,
            is_breached=metrics.exposure_percent > max_exposure,
            severity="warning",
        ))

        return limits

    def validate_trade(
        self,
        signal: TradingSignal,
    ) -> tuple[bool, str]:
        """
        Validate a trade against risk limits.

        Args:
            signal: Trading signal to validate

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check if trading is halted
        if self._trading_halted:
            return False, f"Trading halted: {self._halt_reason}"

        # Check if symbol is blocked
        if signal.symbol in self._blocked_symbols:
            block_until = self._blocked_symbols[signal.symbol]
            if datetime.now() < block_until:
                return False, f"Symbol {signal.symbol} blocked until {block_until}"
            else:
                del self._blocked_symbols[signal.symbol]

        # Get current metrics
        metrics = self.get_risk_metrics()

        # Check daily loss limit
        if metrics.daily_pnl_percent <= -self.max_daily_loss_percent:
            self._trading_halted = True
            self._halt_reason = f"Daily loss limit ({self.max_daily_loss_percent}%) breached"
            logger.critical(self._halt_reason)
            return False, self._halt_reason

        # Check max drawdown
        if metrics.max_drawdown >= self.max_drawdown_percent:
            return False, f"Max drawdown ({self.max_drawdown_percent}%) breached"

        # Check position count (only for new positions)
        if signal.action == "open":
            if metrics.position_count >= settings.max_open_positions:
                return False, f"Max positions ({settings.max_open_positions}) reached"

        # Check position size
        if signal.price and signal.quantity:
            position_value = signal.price * signal.quantity
            position_percent = (position_value / self.capital) * 100

            if position_percent > self.max_position_percent:
                return (
                    False,
                    f"Position size ({position_percent:.1f}%) exceeds limit "
                    f"({self.max_position_percent}%)",
                )

        # Check risk amount
        if signal.stop_loss and signal.price and signal.quantity:
            risk_per_share = abs(signal.price - signal.stop_loss)
            total_risk = risk_per_share * signal.quantity
            risk_percent = (total_risk / self.capital) * 100

            if risk_percent > self.max_portfolio_risk:
                return (
                    False,
                    f"Trade risk ({risk_percent:.1f}%) exceeds limit "
                    f"({self.max_portfolio_risk}%)",
                )

        return True, "Trade validated"

    def block_symbol(self, symbol: str, duration_minutes: int = 60):
        """
        Block trading for a symbol temporarily.

        Args:
            symbol: Symbol to block
            duration_minutes: How long to block
        """
        block_until = datetime.now() + timedelta(minutes=duration_minutes)
        self._blocked_symbols[symbol] = block_until
        logger.warning(f"Symbol {symbol} blocked until {block_until}")

    def halt_trading(self, reason: str):
        """Halt all trading (circuit breaker)."""
        self._trading_halted = True
        self._halt_reason = reason
        logger.critical(f"TRADING HALTED: {reason}")

    def resume_trading(self):
        """Resume trading after halt."""
        self._trading_halted = False
        self._halt_reason = ""
        logger.info("Trading resumed")

    @property
    def is_trading_halted(self) -> bool:
        """Check if trading is halted."""
        return self._trading_halted

    def get_status_report(self) -> dict:
        """Get comprehensive risk status report."""
        metrics = self.get_risk_metrics()
        limits = self.check_risk_limits()

        breached_limits = [l for l in limits if l.is_breached]
        warning_limits = [l for l in limits if l.utilization > 80 and not l.is_breached]

        return {
            "trading_halted": self._trading_halted,
            "halt_reason": self._halt_reason,
            "risk_score": metrics.risk_score,
            "risk_level": (
                "critical" if metrics.risk_score > 75
                else "high" if metrics.risk_score > 50
                else "medium" if metrics.risk_score > 25
                else "low"
            ),
            "metrics": {
                "exposure_percent": metrics.exposure_percent,
                "largest_position": metrics.largest_position_percent,
                "position_count": metrics.position_count,
                "daily_pnl_percent": metrics.daily_pnl_percent,
                "max_drawdown": metrics.max_drawdown,
            },
            "breached_limits": [
                {"name": l.name, "value": l.current_value, "limit": l.limit_value}
                for l in breached_limits
            ],
            "warnings": [
                {"name": l.name, "utilization": l.utilization}
                for l in warning_limits
            ],
            "blocked_symbols": list(self._blocked_symbols.keys()),
        }
