"""
FastAPI dashboard for monitoring trading system.

Provides REST API endpoints for:
- Portfolio status
- Signal history
- Risk metrics
- Trade history
"""

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import settings


class PortfolioSummary(BaseModel):
    """Portfolio summary response."""

    cash: float
    positions_value: float
    total_value: float
    total_pnl: float
    total_pnl_percent: float
    positions_count: int


class SignalResponse(BaseModel):
    """Trading signal response."""

    symbol: str
    action: str
    side: str
    quantity: Optional[int]
    price: Optional[float]
    confidence: float
    reason: str
    timestamp: datetime


class RiskResponse(BaseModel):
    """Risk metrics response."""

    risk_score: float
    risk_level: str
    exposure_percent: float
    daily_pnl_percent: float
    max_drawdown: float
    trading_halted: bool
    breached_limits: list[dict]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    trading_mode: str
    market_open: bool
    timestamp: datetime


def create_app(
    signal_generator=None,
    order_manager=None,
    risk_manager=None,
) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        signal_generator: SignalGenerator instance
        order_manager: OrderManager instance
        risk_manager: RiskManager instance

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="Sentiment Algo Trader",
        description="API for sentiment-based algorithmic trading system",
        version="0.1.0",
    )

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store components
    app.state.signal_generator = signal_generator
    app.state.order_manager = order_manager
    app.state.risk_manager = risk_manager

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Check system health."""
        from src.data.market_data import MarketDataClient

        market = MarketDataClient()

        return HealthResponse(
            status="healthy",
            trading_mode=settings.trading_mode.value,
            market_open=market.is_market_open(),
            timestamp=datetime.now(),
        )

    @app.get("/portfolio", response_model=PortfolioSummary)
    async def get_portfolio():
        """Get current portfolio summary."""
        if not app.state.order_manager:
            raise HTTPException(status_code=503, detail="Order manager not initialized")

        if settings.trading_mode.value == "paper":
            summary = app.state.order_manager.paper_trader.get_portfolio_summary()
        else:
            positions = await app.state.order_manager.get_positions()
            funds = await app.state.order_manager.groww_client.get_funds()

            positions_value = sum(p.market_value for p in positions)
            cash = funds.get("available_cash", 0)
            total_value = cash + positions_value
            initial = settings.initial_capital
            pnl = total_value - initial

            summary = {
                "cash": cash,
                "positions_value": positions_value,
                "total_value": total_value,
                "total_pnl": pnl,
                "total_pnl_percent": (pnl / initial) * 100 if initial > 0 else 0,
                "positions_count": len(positions),
            }

        return PortfolioSummary(**summary)

    @app.get("/positions")
    async def get_positions():
        """Get all current positions."""
        if not app.state.order_manager:
            raise HTTPException(status_code=503, detail="Order manager not initialized")

        positions = await app.state.order_manager.get_positions()

        return [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "average_cost": p.average_cost,
                "current_price": p.current_price,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pnl,
                "unrealized_pnl_percent": p.unrealized_pnl_percent,
            }
            for p in positions
        ]

    @app.get("/signals")
    async def get_signals():
        """Get recent trading signals."""
        if not app.state.signal_generator:
            raise HTTPException(status_code=503, detail="Signal generator not initialized")

        signals = app.state.signal_generator.get_all_signals()

        return [
            SignalResponse(
                symbol=s.symbol,
                action=s.action,
                side=s.side.value,
                quantity=s.quantity,
                price=s.price,
                confidence=s.confidence,
                reason=s.reason,
                timestamp=s.timestamp,
            )
            for s in signals.values()
        ]

    @app.get("/signals/{symbol}", response_model=SignalResponse)
    async def get_signal(symbol: str):
        """Get latest signal for a symbol."""
        if not app.state.signal_generator:
            raise HTTPException(status_code=503, detail="Signal generator not initialized")

        signal = app.state.signal_generator.get_last_signal(symbol.upper())
        if not signal:
            raise HTTPException(status_code=404, detail=f"No signal for {symbol}")

        return SignalResponse(
            symbol=signal.symbol,
            action=signal.action,
            side=signal.side.value,
            quantity=signal.quantity,
            price=signal.price,
            confidence=signal.confidence,
            reason=signal.reason,
            timestamp=signal.timestamp,
        )

    @app.get("/risk", response_model=RiskResponse)
    async def get_risk_metrics():
        """Get current risk metrics."""
        if not app.state.risk_manager:
            raise HTTPException(status_code=503, detail="Risk manager not initialized")

        report = app.state.risk_manager.get_status_report()
        metrics = report["metrics"]

        return RiskResponse(
            risk_score=report["risk_score"],
            risk_level=report["risk_level"],
            exposure_percent=metrics["exposure_percent"],
            daily_pnl_percent=metrics["daily_pnl_percent"],
            max_drawdown=metrics["max_drawdown"],
            trading_halted=report["trading_halted"],
            breached_limits=report["breached_limits"],
        )

    @app.get("/trades")
    async def get_trade_history():
        """Get trade history."""
        if not app.state.order_manager:
            raise HTTPException(status_code=503, detail="Order manager not initialized")

        if settings.trading_mode.value == "paper":
            trades = app.state.order_manager.paper_trader.get_trade_history()
            return trades
        else:
            # For live trading, would fetch from database
            return []

    @app.post("/signals/{symbol}/generate")
    async def generate_signal(symbol: str):
        """Generate a new signal for a symbol."""
        if not app.state.signal_generator:
            raise HTTPException(status_code=503, detail="Signal generator not initialized")

        signal = await app.state.signal_generator.generate_signal(symbol.upper())

        if not signal:
            return {"message": f"No actionable signal for {symbol}"}

        return SignalResponse(
            symbol=signal.symbol,
            action=signal.action,
            side=signal.side.value,
            quantity=signal.quantity,
            price=signal.price,
            confidence=signal.confidence,
            reason=signal.reason,
            timestamp=signal.timestamp,
        )

    @app.post("/trading/halt")
    async def halt_trading(reason: str = "Manual halt"):
        """Halt all trading."""
        if not app.state.risk_manager:
            raise HTTPException(status_code=503, detail="Risk manager not initialized")

        app.state.risk_manager.halt_trading(reason)
        return {"message": "Trading halted", "reason": reason}

    @app.post("/trading/resume")
    async def resume_trading():
        """Resume trading after halt."""
        if not app.state.risk_manager:
            raise HTTPException(status_code=503, detail="Risk manager not initialized")

        app.state.risk_manager.resume_trading()
        return {"message": "Trading resumed"}

    @app.post("/positions/close-all")
    async def close_all_positions():
        """Close all open positions."""
        if not app.state.order_manager:
            raise HTTPException(status_code=503, detail="Order manager not initialized")

        results = await app.state.order_manager.close_all_positions()

        return {
            "message": f"Closed {len(results)} positions",
            "results": [
                {"symbol": r.signal.symbol if r.signal else "unknown", "success": r.success}
                for r in results
            ],
        }

    @app.get("/config")
    async def get_config():
        """Get current configuration (non-sensitive)."""
        return {
            "trading_mode": settings.trading_mode.value,
            "initial_capital": settings.initial_capital,
            "max_position_size_percent": settings.max_position_size_percent,
            "max_portfolio_risk_percent": settings.max_portfolio_risk_percent,
            "stop_loss_percent": settings.stop_loss_percent,
            "take_profit_percent": settings.take_profit_percent,
            "max_daily_loss_percent": settings.max_daily_loss_percent,
            "max_open_positions": settings.max_open_positions,
            "watchlist": settings.default_watchlist,
        }

    return app
