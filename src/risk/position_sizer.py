"""
Position sizing algorithms for optimal capital allocation.

Implements various position sizing strategies:
- Fixed percentage
- Kelly Criterion
- Volatility-based
- Risk parity
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from loguru import logger

from config.settings import settings


class SizingMethod(str, Enum):
    """Position sizing methods."""

    FIXED_PERCENT = "fixed_percent"
    KELLY = "kelly"
    VOLATILITY = "volatility"
    RISK_PARITY = "risk_parity"


@dataclass
class SizingResult:
    """Result of position size calculation."""

    shares: int
    position_value: float
    position_percent: float
    risk_amount: float
    method: SizingMethod
    notes: str = ""


class PositionSizer:
    """
    Calculate optimal position sizes based on various methods.

    Ensures positions stay within risk limits while maximizing
    potential returns.
    """

    def __init__(
        self,
        default_method: SizingMethod = SizingMethod.FIXED_PERCENT,
        max_position_percent: Optional[float] = None,
        max_risk_percent: Optional[float] = None,
    ):
        """
        Initialize position sizer.

        Args:
            default_method: Default sizing method
            max_position_percent: Maximum position size as % of portfolio
            max_risk_percent: Maximum risk per trade as % of portfolio
        """
        self.default_method = default_method
        self.max_position_percent = max_position_percent or settings.max_position_size_percent
        self.max_risk_percent = max_risk_percent or settings.max_portfolio_risk_percent

    def calculate_fixed_percent(
        self,
        capital: float,
        price: float,
        target_percent: float,
    ) -> SizingResult:
        """
        Calculate position size as fixed percentage of capital.

        Args:
            capital: Available capital
            price: Stock price
            target_percent: Target position size as % of capital

        Returns:
            SizingResult with calculated shares
        """
        # Cap at maximum allowed
        target_percent = min(target_percent, self.max_position_percent)

        # Calculate position value
        position_value = capital * (target_percent / 100)

        # Calculate shares (round down to avoid over-allocation)
        shares = int(position_value / price)

        # Recalculate actual values based on whole shares
        actual_value = shares * price
        actual_percent = (actual_value / capital) * 100 if capital > 0 else 0

        return SizingResult(
            shares=shares,
            position_value=actual_value,
            position_percent=actual_percent,
            risk_amount=actual_value * (settings.stop_loss_percent / 100),
            method=SizingMethod.FIXED_PERCENT,
            notes=f"Target: {target_percent:.1f}%, Actual: {actual_percent:.1f}%",
        )

    def calculate_kelly(
        self,
        capital: float,
        price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.25,  # Use quarter Kelly for safety
    ) -> SizingResult:
        """
        Calculate position size using Kelly Criterion.

        Kelly formula: f* = (p * b - q) / b
        where:
        - p = probability of winning
        - q = probability of losing (1 - p)
        - b = win/loss ratio

        Args:
            capital: Available capital
            price: Stock price
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)
            fraction: Fraction of Kelly to use (0.25 = quarter Kelly)

        Returns:
            SizingResult with calculated shares
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            # Invalid inputs, fall back to fixed percent
            return self.calculate_fixed_percent(capital, price, 2.0)

        # Calculate Kelly fraction
        b = avg_win / avg_loss  # Win/loss ratio
        p = win_rate
        q = 1 - p

        kelly_percent = ((p * b) - q) / b

        # Apply fraction (quarter Kelly is safer)
        kelly_percent = kelly_percent * fraction * 100

        # Ensure within bounds
        kelly_percent = max(0, min(kelly_percent, self.max_position_percent))

        if kelly_percent <= 0:
            return SizingResult(
                shares=0,
                position_value=0,
                position_percent=0,
                risk_amount=0,
                method=SizingMethod.KELLY,
                notes="Kelly suggests no position (negative edge)",
            )

        # Calculate position
        position_value = capital * (kelly_percent / 100)
        shares = int(position_value / price)

        actual_value = shares * price
        actual_percent = (actual_value / capital) * 100 if capital > 0 else 0

        return SizingResult(
            shares=shares,
            position_value=actual_value,
            position_percent=actual_percent,
            risk_amount=actual_value * (avg_loss / 100),
            method=SizingMethod.KELLY,
            notes=f"Kelly: {kelly_percent:.1f}% (win_rate={win_rate:.0%}, b={b:.2f})",
        )

    def calculate_volatility_based(
        self,
        capital: float,
        price: float,
        volatility: float,  # Standard deviation of returns
        target_risk: Optional[float] = None,
    ) -> SizingResult:
        """
        Calculate position size based on volatility.

        Allocates more capital to less volatile stocks.

        Args:
            capital: Available capital
            price: Stock price
            volatility: Stock volatility (std dev of returns, e.g., 0.02 = 2%)
            target_risk: Target portfolio risk contribution (uses max_risk_percent if None)

        Returns:
            SizingResult with calculated shares
        """
        target_risk = target_risk or self.max_risk_percent

        if volatility <= 0:
            # No volatility data, use fixed percent
            return self.calculate_fixed_percent(capital, price, 3.0)

        # Position size = Target Risk / Stock Volatility
        # This ensures each position contributes similar risk
        position_percent = target_risk / volatility

        # Cap at maximum
        position_percent = min(position_percent, self.max_position_percent)

        # Calculate position
        position_value = capital * (position_percent / 100)
        shares = int(position_value / price)

        actual_value = shares * price
        actual_percent = (actual_value / capital) * 100 if capital > 0 else 0
        risk_contribution = actual_percent * volatility

        return SizingResult(
            shares=shares,
            position_value=actual_value,
            position_percent=actual_percent,
            risk_amount=actual_value * volatility,
            method=SizingMethod.VOLATILITY,
            notes=f"Volatility: {volatility:.1%}, Risk contribution: {risk_contribution:.2f}%",
        )

    def calculate_risk_based(
        self,
        capital: float,
        price: float,
        stop_loss_price: float,
        risk_per_trade: Optional[float] = None,
    ) -> SizingResult:
        """
        Calculate position size based on risk per trade.

        Position size is determined by how much we're willing to lose
        if stop loss is hit.

        Args:
            capital: Available capital
            price: Stock price
            stop_loss_price: Stop loss price
            risk_per_trade: Maximum risk per trade as % of capital

        Returns:
            SizingResult with calculated shares
        """
        risk_per_trade = risk_per_trade or self.max_risk_percent

        # Calculate risk per share
        risk_per_share = abs(price - stop_loss_price)

        if risk_per_share <= 0:
            return self.calculate_fixed_percent(capital, price, 3.0)

        # Calculate maximum risk amount
        max_risk_amount = capital * (risk_per_trade / 100)

        # Calculate shares based on risk
        shares = int(max_risk_amount / risk_per_share)

        # Check position size limit
        position_value = shares * price
        position_percent = (position_value / capital) * 100

        if position_percent > self.max_position_percent:
            # Reduce to meet position limit
            position_value = capital * (self.max_position_percent / 100)
            shares = int(position_value / price)
            position_percent = self.max_position_percent

        actual_value = shares * price
        actual_risk = shares * risk_per_share

        return SizingResult(
            shares=shares,
            position_value=actual_value,
            position_percent=position_percent,
            risk_amount=actual_risk,
            method=SizingMethod.FIXED_PERCENT,  # Risk-based variant
            notes=f"Risk/trade: ₹{actual_risk:,.2f} ({actual_risk/capital*100:.2f}%)",
        )

    def calculate(
        self,
        capital: float,
        price: float,
        method: Optional[SizingMethod] = None,
        confidence: float = 0.5,
        volatility: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        win_rate: float = 0.5,
        avg_win: float = 5.0,
        avg_loss: float = 3.0,
    ) -> SizingResult:
        """
        Calculate position size using specified method.

        Args:
            capital: Available capital
            price: Stock price
            method: Sizing method to use
            confidence: Signal confidence (0-1)
            volatility: Stock volatility
            stop_loss_price: Stop loss price
            win_rate: Historical win rate
            avg_win: Average winning trade %
            avg_loss: Average losing trade %

        Returns:
            SizingResult
        """
        method = method or self.default_method

        # Base position size adjusted by confidence
        base_percent = settings.max_position_size_percent * confidence

        if method == SizingMethod.FIXED_PERCENT:
            result = self.calculate_fixed_percent(capital, price, base_percent)

        elif method == SizingMethod.KELLY:
            result = self.calculate_kelly(
                capital, price, win_rate, avg_win, avg_loss
            )

        elif method == SizingMethod.VOLATILITY:
            if volatility:
                result = self.calculate_volatility_based(
                    capital, price, volatility
                )
            else:
                result = self.calculate_fixed_percent(capital, price, base_percent)

        elif method == SizingMethod.RISK_PARITY and stop_loss_price:
            result = self.calculate_risk_based(
                capital, price, stop_loss_price
            )

        else:
            result = self.calculate_fixed_percent(capital, price, base_percent)

        # Final safety check
        if result.position_percent > self.max_position_percent:
            logger.warning(
                f"Position size {result.position_percent:.1f}% exceeds max "
                f"{self.max_position_percent}%, reducing"
            )
            result = self.calculate_fixed_percent(
                capital, price, self.max_position_percent
            )

        return result
