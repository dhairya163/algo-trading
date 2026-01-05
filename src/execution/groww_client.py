"""
Groww API client wrapper for order execution.

Provides a clean interface to the Groww Trading API for:
- Placing orders (market, limit, stop loss)
- Modifying/cancelling orders
- Getting order status
- Fetching positions and holdings
"""

from datetime import datetime
from typing import Optional

from loguru import logger

from config.settings import settings
from src.data.models import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    StockQuote,
)


class GrowwClient:
    """
    Wrapper around Groww Trading API.

    Handles authentication and provides high-level methods for trading.
    """

    def __init__(self):
        """Initialize Groww client."""
        self._api = None
        self._feed = None
        self._initialized = False
        self._session_token = None

    async def initialize(self) -> bool:
        """
        Initialize connection to Groww API.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        api_key = settings.groww_api_key
        if not api_key:
            logger.warning("Groww API key not configured")
            return False

        try:
            from growwapi import GrowwAPI, GrowwFeed

            self._api = GrowwAPI(api_key=api_key.get_secret_value())

            # Initialize feed for real-time data
            self._feed = GrowwFeed(api_key=api_key.get_secret_value())

            self._initialized = True
            logger.info("Groww API client initialized successfully")
            return True

        except ImportError:
            logger.error("growwapi package not installed. Run: pip install growwapi")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Groww API: {e}")
            return False

    async def close(self):
        """Close API connections."""
        if self._feed:
            try:
                self._feed.close()
            except Exception:
                pass
        self._initialized = False

    def _ensure_initialized(self):
        """Ensure client is initialized."""
        if not self._initialized:
            raise RuntimeError("Groww client not initialized. Call initialize() first.")

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        product_type: str = "CNC",  # CNC for delivery, MIS for intraday
    ) -> Optional[Order]:
        """
        Place an order on Groww.

        Args:
            symbol: Stock symbol (NSE)
            side: BUY or SELL
            quantity: Number of shares
            order_type: MARKET, LIMIT, STOP_LOSS, or STOP_LOSS_MARKET
            price: Limit price (required for LIMIT orders)
            stop_price: Stop/trigger price (required for stop orders)
            product_type: CNC (delivery) or MIS (intraday)

        Returns:
            Order object with order ID, or None on failure
        """
        self._ensure_initialized()

        try:
            # Map our order types to Groww API format
            groww_order_type = {
                OrderType.MARKET: "MARKET",
                OrderType.LIMIT: "LIMIT",
                OrderType.STOP_LOSS: "SL",
                OrderType.STOP_LOSS_MARKET: "SL-M",
            }.get(order_type, "MARKET")

            # Prepare order parameters
            order_params = {
                "symbol": symbol,
                "exchange": "NSE",
                "transaction_type": side.value.upper(),
                "quantity": quantity,
                "order_type": groww_order_type,
                "product": product_type,
            }

            if price and order_type in (OrderType.LIMIT, OrderType.STOP_LOSS):
                order_params["price"] = price

            if stop_price and order_type in (OrderType.STOP_LOSS, OrderType.STOP_LOSS_MARKET):
                order_params["trigger_price"] = stop_price

            # Place order via API
            response = self._api.place_order(**order_params)

            if not response or "order_id" not in response:
                logger.error(f"Order placement failed: {response}")
                return None

            order = Order(
                id=response["order_id"],
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                status=OrderStatus.OPEN,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

            logger.info(
                f"Order placed: {order.id} - {side.value} {quantity} {symbol} "
                f"@ {price or 'MARKET'}"
            )

            return order

        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return None

    async def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        order_type: Optional[OrderType] = None,
    ) -> bool:
        """
        Modify an existing order.

        Args:
            order_id: Order ID to modify
            quantity: New quantity
            price: New price
            order_type: New order type

        Returns:
            True if modification successful
        """
        self._ensure_initialized()

        try:
            modify_params = {"order_id": order_id}

            if quantity:
                modify_params["quantity"] = quantity
            if price:
                modify_params["price"] = price
            if order_type:
                modify_params["order_type"] = order_type.value

            response = self._api.modify_order(**modify_params)

            if response and response.get("status") == "success":
                logger.info(f"Order {order_id} modified successfully")
                return True

            logger.error(f"Order modification failed: {response}")
            return False

        except Exception as e:
            logger.error(f"Order modification failed: {e}")
            return False

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation successful
        """
        self._ensure_initialized()

        try:
            response = self._api.cancel_order(order_id=order_id)

            if response and response.get("status") == "success":
                logger.info(f"Order {order_id} cancelled successfully")
                return True

            logger.error(f"Order cancellation failed: {response}")
            return False

        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False

    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get current status of an order.

        Args:
            order_id: Order ID

        Returns:
            Order object with current status
        """
        self._ensure_initialized()

        try:
            response = self._api.get_order_history(order_id=order_id)

            if not response:
                return None

            # Map Groww status to our OrderStatus
            status_map = {
                "PENDING": OrderStatus.PENDING,
                "OPEN": OrderStatus.OPEN,
                "COMPLETE": OrderStatus.FILLED,
                "CANCELLED": OrderStatus.CANCELLED,
                "REJECTED": OrderStatus.REJECTED,
                "TRIGGER_PENDING": OrderStatus.OPEN,
            }

            order_data = response[0] if isinstance(response, list) else response

            return Order(
                id=order_id,
                symbol=order_data.get("symbol", ""),
                side=OrderSide(order_data.get("transaction_type", "buy").lower()),
                order_type=OrderType.MARKET,  # Simplified
                quantity=order_data.get("quantity", 0),
                price=order_data.get("price"),
                status=status_map.get(order_data.get("status", ""), OrderStatus.PENDING),
                filled_quantity=order_data.get("filled_quantity", 0),
                average_price=order_data.get("average_price"),
                updated_at=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return None

    async def get_positions(self) -> list[Position]:
        """
        Get all current positions.

        Returns:
            List of Position objects
        """
        self._ensure_initialized()

        try:
            response = self._api.get_positions()

            if not response:
                return []

            positions = []
            for pos in response:
                quantity = pos.get("quantity", 0)
                avg_cost = pos.get("average_price", 0)
                current_price = pos.get("last_price", avg_cost)

                unrealized_pnl = (current_price - avg_cost) * quantity
                unrealized_pnl_percent = (
                    (unrealized_pnl / (avg_cost * quantity) * 100)
                    if avg_cost and quantity
                    else 0
                )

                position = Position(
                    symbol=pos.get("symbol", ""),
                    quantity=quantity,
                    average_cost=avg_cost,
                    current_price=current_price,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_percent=unrealized_pnl_percent,
                    market_value=current_price * quantity,
                    opened_at=datetime.now(),  # Not available from API
                )
                positions.append(position)

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    async def get_holdings(self) -> list[Position]:
        """
        Get all holdings (delivery positions).

        Returns:
            List of Position objects
        """
        self._ensure_initialized()

        try:
            response = self._api.get_holdings()

            if not response:
                return []

            holdings = []
            for holding in response:
                quantity = holding.get("quantity", 0)
                avg_cost = holding.get("average_price", 0)
                current_price = holding.get("last_price", avg_cost)

                unrealized_pnl = (current_price - avg_cost) * quantity
                unrealized_pnl_percent = (
                    (unrealized_pnl / (avg_cost * quantity) * 100)
                    if avg_cost and quantity
                    else 0
                )

                position = Position(
                    symbol=holding.get("symbol", ""),
                    quantity=quantity,
                    average_cost=avg_cost,
                    current_price=current_price,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_percent=unrealized_pnl_percent,
                    market_value=current_price * quantity,
                    opened_at=datetime.now(),
                )
                holdings.append(position)

            return holdings

        except Exception as e:
            logger.error(f"Failed to get holdings: {e}")
            return []

    async def get_quote(self, symbol: str) -> Optional[StockQuote]:
        """
        Get real-time quote for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            StockQuote object
        """
        self._ensure_initialized()

        try:
            response = self._api.get_quote(symbol=symbol, exchange="NSE")

            if not response:
                return None

            return StockQuote(
                symbol=symbol,
                last_price=response.get("last_price", 0),
                change=response.get("change", 0),
                change_percent=response.get("change_percent", 0),
                bid=response.get("bid_price"),
                ask=response.get("ask_price"),
                bid_size=response.get("bid_quantity"),
                ask_size=response.get("ask_quantity"),
                volume=response.get("volume", 0),
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Failed to get quote: {e}")
            return None

    async def get_funds(self) -> dict:
        """
        Get available funds/margins.

        Returns:
            Dictionary with fund details
        """
        self._ensure_initialized()

        try:
            response = self._api.get_funds()

            return {
                "available_cash": response.get("available_cash", 0),
                "used_margin": response.get("used_margin", 0),
                "available_margin": response.get("available_margin", 0),
                "total_balance": response.get("total_balance", 0),
            }

        except Exception as e:
            logger.error(f"Failed to get funds: {e}")
            return {"available_cash": 0, "available_margin": 0}

    def subscribe_quotes(self, symbols: list[str], callback):
        """
        Subscribe to real-time quote updates.

        Args:
            symbols: List of symbols to subscribe
            callback: Function to call with quote updates
        """
        self._ensure_initialized()

        try:
            for symbol in symbols:
                self._feed.subscribe(
                    symbol=symbol,
                    exchange="NSE",
                    callback=callback,
                )
            logger.info(f"Subscribed to quotes for: {symbols}")
        except Exception as e:
            logger.error(f"Failed to subscribe to quotes: {e}")

    def unsubscribe_quotes(self, symbols: list[str]):
        """
        Unsubscribe from quote updates.

        Args:
            symbols: List of symbols to unsubscribe
        """
        if self._feed:
            try:
                for symbol in symbols:
                    self._feed.unsubscribe(symbol=symbol, exchange="NSE")
            except Exception:
                pass
