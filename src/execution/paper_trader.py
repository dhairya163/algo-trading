"""
Paper trading simulator for testing strategies without real money.

Simulates order execution with realistic fills and slippage.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger

from config.settings import settings
from src.data.models import Order, OrderSide, OrderStatus, OrderType, Position


@dataclass
class PaperPortfolio:
    """Simulated portfolio for paper trading."""

    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    orders: dict[str, Order] = field(default_factory=dict)
    trade_history: list[dict] = field(default_factory=list)
    initial_capital: float = 0.0

    def __post_init__(self):
        if self.initial_capital == 0:
            self.initial_capital = self.cash

    @property
    def total_value(self) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value

    @property
    def total_pnl(self) -> float:
        """Calculate total profit/loss."""
        return self.total_value - self.initial_capital

    @property
    def total_pnl_percent(self) -> float:
        """Calculate total P&L percentage."""
        if self.initial_capital == 0:
            return 0
        return (self.total_pnl / self.initial_capital) * 100


class PaperTrader:
    """
    Paper trading simulator.

    Features:
    - Realistic order simulation with slippage
    - Position tracking
    - P&L calculation
    - Trade history logging
    """

    def __init__(
        self,
        initial_capital: Optional[float] = None,
        slippage_percent: float = 0.05,  # 0.05% slippage
        commission_percent: float = 0.03,  # 0.03% commission
    ):
        """
        Initialize paper trader.

        Args:
            initial_capital: Starting capital (uses settings if not provided)
            slippage_percent: Simulated slippage as percentage
            commission_percent: Commission per trade as percentage
        """
        capital = initial_capital or settings.initial_capital

        self.portfolio = PaperPortfolio(
            cash=capital,
            initial_capital=capital,
        )
        self.slippage_percent = slippage_percent
        self.commission_percent = commission_percent

        # Price cache for simulation
        self._price_cache: dict[str, float] = {}

        logger.info(f"Paper trader initialized with ₹{capital:,.2f}")

    def set_price(self, symbol: str, price: float):
        """Set current price for a symbol (for simulation)."""
        self._price_cache[symbol] = price

    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        return self._price_cache.get(symbol)

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply slippage to price based on order side."""
        slippage = price * (self.slippage_percent / 100)

        if side == OrderSide.BUY:
            return price + slippage  # Pay more when buying
        else:
            return price - slippage  # Get less when selling

    def _calculate_commission(self, value: float) -> float:
        """Calculate commission for a trade."""
        return value * (self.commission_percent / 100)

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        return f"PAPER_{uuid.uuid4().hex[:8].upper()}"

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Optional[Order]:
        """
        Place a simulated order.

        Market orders are filled immediately.
        Limit/stop orders are tracked for later fill.

        Args:
            symbol: Stock symbol
            side: BUY or SELL
            quantity: Number of shares
            order_type: Order type
            price: Limit price
            stop_price: Stop/trigger price

        Returns:
            Order object
        """
        order_id = self._generate_order_id()

        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            status=OrderStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # For market orders, try to fill immediately
        if order_type == OrderType.MARKET:
            current_price = self.get_price(symbol)
            if current_price:
                filled = await self._fill_order(order, current_price)
                if filled:
                    return order

            # No price available, reject order
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order {order_id} rejected: no price available for {symbol}")
            return order

        # Store pending limit/stop orders
        self.portfolio.orders[order_id] = order
        logger.info(f"Paper order placed: {order_id} - {side.value} {quantity} {symbol}")

        return order

    async def _fill_order(self, order: Order, market_price: float) -> bool:
        """
        Fill an order at the given market price.

        Args:
            order: Order to fill
            market_price: Current market price

        Returns:
            True if order was filled
        """
        # Apply slippage
        fill_price = self._apply_slippage(market_price, order.side)

        # Calculate trade value and commission
        trade_value = fill_price * order.quantity
        commission = self._calculate_commission(trade_value)

        # Check if we have enough cash for buy orders
        if order.side == OrderSide.BUY:
            total_cost = trade_value + commission
            if total_cost > self.portfolio.cash:
                order.status = OrderStatus.REJECTED
                logger.warning(
                    f"Order {order.id} rejected: insufficient funds "
                    f"(need ₹{total_cost:,.2f}, have ₹{self.portfolio.cash:,.2f})"
                )
                return False

            # Deduct cash
            self.portfolio.cash -= total_cost

            # Update or create position
            if order.symbol in self.portfolio.positions:
                pos = self.portfolio.positions[order.symbol]
                # Calculate new average cost
                total_quantity = pos.quantity + order.quantity
                total_cost_basis = (pos.average_cost * pos.quantity) + (
                    fill_price * order.quantity
                )
                pos.quantity = total_quantity
                pos.average_cost = total_cost_basis / total_quantity
                pos.current_price = fill_price
                pos.market_value = fill_price * total_quantity
                pos.unrealized_pnl = (fill_price - pos.average_cost) * total_quantity
                pos.unrealized_pnl_percent = (
                    (pos.unrealized_pnl / (pos.average_cost * total_quantity)) * 100
                    if pos.average_cost
                    else 0
                )
            else:
                self.portfolio.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    average_cost=fill_price,
                    current_price=fill_price,
                    unrealized_pnl=0,
                    unrealized_pnl_percent=0,
                    market_value=trade_value,
                    opened_at=datetime.now(),
                )

        else:  # SELL
            # Check if we have the position
            if order.symbol not in self.portfolio.positions:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Order {order.id} rejected: no position in {order.symbol}")
                return False

            pos = self.portfolio.positions[order.symbol]
            if pos.quantity < order.quantity:
                order.status = OrderStatus.REJECTED
                logger.warning(
                    f"Order {order.id} rejected: insufficient shares "
                    f"(need {order.quantity}, have {pos.quantity})"
                )
                return False

            # Add cash (minus commission)
            self.portfolio.cash += trade_value - commission

            # Calculate realized P&L
            realized_pnl = (fill_price - pos.average_cost) * order.quantity

            # Update position
            pos.quantity -= order.quantity
            pos.current_price = fill_price
            pos.market_value = fill_price * pos.quantity

            if pos.quantity == 0:
                # Close position
                del self.portfolio.positions[order.symbol]
            else:
                # Update unrealized P&L
                pos.unrealized_pnl = (fill_price - pos.average_cost) * pos.quantity
                pos.unrealized_pnl_percent = (
                    (pos.unrealized_pnl / (pos.average_cost * pos.quantity)) * 100
                    if pos.average_cost
                    else 0
                )

        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_price = fill_price
        order.updated_at = datetime.now()

        # Record trade
        self.portfolio.trade_history.append(
            {
                "order_id": order.id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
                "price": fill_price,
                "commission": commission,
                "timestamp": datetime.now().isoformat(),
            }
        )

        logger.info(
            f"Paper order filled: {order.id} - {order.side.value} {order.quantity} "
            f"{order.symbol} @ ₹{fill_price:,.2f} (commission: ₹{commission:,.2f})"
        )

        return True

    async def update_prices(self, prices: dict[str, float]):
        """
        Update prices and check pending orders.

        Args:
            prices: Dictionary of symbol -> price
        """
        # Update price cache
        self._price_cache.update(prices)

        # Update position values
        for symbol, price in prices.items():
            if symbol in self.portfolio.positions:
                pos = self.portfolio.positions[symbol]
                pos.current_price = price
                pos.market_value = price * pos.quantity
                pos.unrealized_pnl = (price - pos.average_cost) * pos.quantity
                pos.unrealized_pnl_percent = (
                    (pos.unrealized_pnl / (pos.average_cost * pos.quantity)) * 100
                    if pos.average_cost
                    else 0
                )

        # Check pending orders for fills
        pending_orders = [
            o for o in self.portfolio.orders.values() if o.status == OrderStatus.OPEN
        ]

        for order in pending_orders:
            if order.symbol not in prices:
                continue

            current_price = prices[order.symbol]
            should_fill = False

            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and current_price <= order.price:
                    should_fill = True
                elif order.side == OrderSide.SELL and current_price >= order.price:
                    should_fill = True

            elif order.order_type in (OrderType.STOP_LOSS, OrderType.STOP_LOSS_MARKET):
                if order.side == OrderSide.SELL and current_price <= order.stop_price:
                    should_fill = True
                elif order.side == OrderSide.BUY and current_price >= order.stop_price:
                    should_fill = True

            if should_fill:
                await self._fill_order(order, current_price)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id not in self.portfolio.orders:
            return False

        order = self.portfolio.orders[order_id]
        if order.status not in (OrderStatus.PENDING, OrderStatus.OPEN):
            return False

        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()
        logger.info(f"Paper order cancelled: {order_id}")
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID."""
        return self.portfolio.orders.get(order_id)

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.portfolio.positions.get(symbol)

    def get_all_positions(self) -> list[Position]:
        """Get all open positions."""
        return list(self.portfolio.positions.values())

    def get_portfolio_summary(self) -> dict:
        """Get portfolio summary."""
        return {
            "cash": self.portfolio.cash,
            "positions_value": sum(p.market_value for p in self.portfolio.positions.values()),
            "total_value": self.portfolio.total_value,
            "total_pnl": self.portfolio.total_pnl,
            "total_pnl_percent": self.portfolio.total_pnl_percent,
            "initial_capital": self.portfolio.initial_capital,
            "positions_count": len(self.portfolio.positions),
            "trades_count": len(self.portfolio.trade_history),
        }

    def get_trade_history(self) -> list[dict]:
        """Get trade history."""
        return self.portfolio.trade_history.copy()

    def reset(self, initial_capital: Optional[float] = None):
        """Reset paper trader to initial state."""
        capital = initial_capital or self.portfolio.initial_capital

        self.portfolio = PaperPortfolio(
            cash=capital,
            initial_capital=capital,
        )
        self._price_cache.clear()
        logger.info(f"Paper trader reset with ₹{capital:,.2f}")
