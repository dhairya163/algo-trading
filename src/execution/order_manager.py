"""
Order manager for executing trading signals.

Handles the full order lifecycle:
- Signal validation
- Risk checks
- Order placement
- Order tracking
- Position updates
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from loguru import logger

from config.settings import TradingMode, settings
from src.data.models import Order, OrderSide, OrderStatus, OrderType, Position
from src.execution.groww_client import GrowwClient
from src.execution.paper_trader import PaperTrader
from src.strategy.signal_generator import TradingSignal


@dataclass
class ExecutionResult:
    """Result of order execution."""

    success: bool
    order: Optional[Order] = None
    message: str = ""
    signal: Optional[TradingSignal] = None


class OrderManager:
    """
    Manages order execution for both paper and live trading.

    Features:
    - Unified interface for paper/live trading
    - Pre-trade risk validation
    - Order tracking and status updates
    - Position synchronization
    """

    def __init__(
        self,
        mode: Optional[TradingMode] = None,
        groww_client: Optional[GrowwClient] = None,
        paper_trader: Optional[PaperTrader] = None,
    ):
        """
        Initialize order manager.

        Args:
            mode: Trading mode (paper or live)
            groww_client: GrowwClient for live trading
            paper_trader: PaperTrader for paper trading
        """
        self.mode = mode or settings.trading_mode
        self.groww_client = groww_client or GrowwClient()
        self.paper_trader = paper_trader or PaperTrader()

        # Track active orders
        self._active_orders: dict[str, Order] = {}

        # Daily P&L tracking for circuit breaker
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._trading_day: str = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Order manager initialized in {self.mode.value} mode")

    async def initialize(self) -> bool:
        """Initialize trading connections."""
        if self.mode == TradingMode.LIVE:
            return await self.groww_client.initialize()
        return True

    async def close(self):
        """Close connections."""
        if self.groww_client:
            await self.groww_client.close()

    def _check_daily_reset(self):
        """Reset daily counters if new trading day."""
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._trading_day:
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._trading_day = today
            logger.info("Daily counters reset for new trading day")

    def _validate_signal(self, signal: TradingSignal) -> tuple[bool, str]:
        """
        Validate a trading signal before execution.

        Args:
            signal: Trading signal to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if we have minimum required data
        if not signal.symbol:
            return False, "Signal missing symbol"

        if signal.quantity is None or signal.quantity <= 0:
            return False, f"Invalid quantity: {signal.quantity}"

        # Check confidence threshold
        min_confidence = 0.3
        if signal.confidence < min_confidence:
            return False, f"Confidence {signal.confidence:.2f} below threshold {min_confidence}"

        # Check daily loss limit (circuit breaker)
        self._check_daily_reset()
        max_daily_loss = settings.initial_capital * (settings.max_daily_loss_percent / 100)
        if self._daily_pnl < -max_daily_loss:
            return False, f"Daily loss limit reached: ₹{self._daily_pnl:,.2f}"

        return True, ""

    async def _get_available_capital(self) -> float:
        """Get available capital for trading."""
        if self.mode == TradingMode.LIVE:
            funds = await self.groww_client.get_funds()
            return funds.get("available_cash", 0)
        else:
            return self.paper_trader.portfolio.cash

    async def _check_position_limits(
        self,
        signal: TradingSignal,
    ) -> tuple[bool, str]:
        """
        Check if signal would exceed position limits.

        Args:
            signal: Trading signal

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Get current positions
        if self.mode == TradingMode.LIVE:
            positions = await self.groww_client.get_positions()
        else:
            positions = self.paper_trader.get_all_positions()

        # Check max open positions
        if len(positions) >= settings.max_open_positions:
            if signal.action == "open":
                return False, f"Maximum open positions ({settings.max_open_positions}) reached"

        # Check position size limit
        available_capital = await self._get_available_capital()
        if signal.price and signal.quantity:
            position_value = signal.price * signal.quantity
            position_percent = (position_value / settings.initial_capital) * 100

            if position_percent > settings.max_position_size_percent:
                return (
                    False,
                    f"Position size {position_percent:.1f}% exceeds limit "
                    f"{settings.max_position_size_percent}%",
                )

        return True, ""

    async def execute_signal(self, signal: TradingSignal) -> ExecutionResult:
        """
        Execute a trading signal.

        Args:
            signal: Trading signal to execute

        Returns:
            ExecutionResult with order details
        """
        # Validate signal
        is_valid, error_msg = self._validate_signal(signal)
        if not is_valid:
            logger.warning(f"Signal validation failed: {error_msg}")
            return ExecutionResult(
                success=False,
                message=error_msg,
                signal=signal,
            )

        # Check position limits
        is_valid, error_msg = await self._check_position_limits(signal)
        if not is_valid:
            logger.warning(f"Position limit check failed: {error_msg}")
            return ExecutionResult(
                success=False,
                message=error_msg,
                signal=signal,
            )

        # Place order
        try:
            if self.mode == TradingMode.LIVE:
                order = await self.groww_client.place_order(
                    symbol=signal.symbol,
                    side=signal.side,
                    quantity=signal.quantity,
                    order_type=OrderType.MARKET,
                )
            else:
                # Paper trading - set price first
                if signal.price:
                    self.paper_trader.set_price(signal.symbol, signal.price)

                order = await self.paper_trader.place_order(
                    symbol=signal.symbol,
                    side=signal.side,
                    quantity=signal.quantity,
                    order_type=OrderType.MARKET,
                )

            if order and order.status != OrderStatus.REJECTED:
                self._active_orders[order.id] = order
                self._daily_trades += 1

                # Place stop loss order if specified
                if signal.stop_loss and order.status == OrderStatus.FILLED:
                    await self._place_stop_loss(signal, order)

                logger.info(
                    f"Signal executed: {signal.side.value} {signal.quantity} "
                    f"{signal.symbol} @ ₹{signal.price:,.2f}"
                )

                return ExecutionResult(
                    success=True,
                    order=order,
                    message="Order placed successfully",
                    signal=signal,
                )
            else:
                return ExecutionResult(
                    success=False,
                    order=order,
                    message="Order rejected",
                    signal=signal,
                )

        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return ExecutionResult(
                success=False,
                message=str(e),
                signal=signal,
            )

    async def _place_stop_loss(self, signal: TradingSignal, filled_order: Order):
        """Place a stop loss order for a filled position."""
        stop_side = OrderSide.SELL if signal.side == OrderSide.BUY else OrderSide.BUY

        try:
            if self.mode == TradingMode.LIVE:
                await self.groww_client.place_order(
                    symbol=signal.symbol,
                    side=stop_side,
                    quantity=signal.quantity,
                    order_type=OrderType.STOP_LOSS_MARKET,
                    stop_price=signal.stop_loss,
                )
            else:
                await self.paper_trader.place_order(
                    symbol=signal.symbol,
                    side=stop_side,
                    quantity=signal.quantity,
                    order_type=OrderType.STOP_LOSS_MARKET,
                    stop_price=signal.stop_loss,
                )

            logger.info(f"Stop loss placed for {signal.symbol} @ ₹{signal.stop_loss:,.2f}")

        except Exception as e:
            logger.error(f"Failed to place stop loss: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if self.mode == TradingMode.LIVE:
            success = await self.groww_client.cancel_order(order_id)
        else:
            success = await self.paper_trader.cancel_order(order_id)

        if success and order_id in self._active_orders:
            del self._active_orders[order_id]

        return success

    async def get_positions(self) -> list[Position]:
        """Get all current positions."""
        if self.mode == TradingMode.LIVE:
            return await self.groww_client.get_positions()
        else:
            return self.paper_trader.get_all_positions()

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    def get_active_orders(self) -> list[Order]:
        """Get all active orders."""
        return list(self._active_orders.values())

    def get_daily_stats(self) -> dict:
        """Get daily trading statistics."""
        self._check_daily_reset()
        return {
            "date": self._trading_day,
            "trades": self._daily_trades,
            "pnl": self._daily_pnl,
            "mode": self.mode.value,
        }

    async def close_all_positions(self) -> list[ExecutionResult]:
        """
        Close all open positions (emergency exit).

        Returns:
            List of execution results
        """
        positions = await self.get_positions()
        results = []

        for position in positions:
            if position.quantity == 0:
                continue

            side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
            quantity = abs(position.quantity)

            signal = TradingSignal(
                symbol=position.symbol,
                side=side,
                action="close",
                quantity=quantity,
                price=position.current_price,
                confidence=1.0,
                reason="Emergency close all positions",
            )

            result = await self.execute_signal(signal)
            results.append(result)

        return results
