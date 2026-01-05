"""
Performance metrics calculation for backtesting.

Calculates standard trading metrics including:
- Returns and P&L
- Risk-adjusted returns (Sharpe, Sortino)
- Drawdown analysis
- Trade statistics
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.backtest.backtester import BacktestTrade


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    # Returns
    total_return: float
    total_return_percent: float
    annualized_return: float
    cagr: float  # Compound Annual Growth Rate

    # Risk metrics
    volatility: float  # Annualized volatility
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float  # Return / Max Drawdown

    # Drawdown
    max_drawdown: float
    max_drawdown_duration: int  # Days
    avg_drawdown: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float  # Gross profit / Gross loss
    avg_trade: float
    avg_winner: float
    avg_loser: float
    best_trade: float
    worst_trade: float
    avg_holding_period: float  # Days

    # Streaks
    max_consecutive_wins: int
    max_consecutive_losses: int

    # Expectancy
    expectancy: float  # Expected value per trade
    expectancy_ratio: float  # Expectancy / Avg loser


def calculate_metrics(
    trades: list["BacktestTrade"],
    equity_curve: list[dict],
    initial_capital: float,
    risk_free_rate: float = 0.05,  # 5% annual risk-free rate
    trading_days_per_year: int = 252,
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.

    Args:
        trades: List of completed trades
        equity_curve: List of equity snapshots
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate for Sharpe calculation
        trading_days_per_year: Number of trading days per year

    Returns:
        PerformanceMetrics object
    """
    # Handle empty cases
    if not equity_curve:
        return _empty_metrics()

    # Extract equity values
    equities = [e["equity"] for e in equity_curve]
    final_equity = equities[-1] if equities else initial_capital

    # Calculate returns
    total_return = final_equity - initial_capital
    total_return_percent = (total_return / initial_capital) * 100

    # Daily returns
    daily_returns = []
    for i in range(1, len(equities)):
        if equities[i - 1] > 0:
            ret = (equities[i] - equities[i - 1]) / equities[i - 1]
            daily_returns.append(ret)

    daily_returns = np.array(daily_returns) if daily_returns else np.array([0])

    # Annualized return
    n_days = len(equity_curve)
    years = n_days / trading_days_per_year if n_days > 0 else 1

    if years > 0 and final_equity > 0 and initial_capital > 0:
        cagr = ((final_equity / initial_capital) ** (1 / years) - 1) * 100
    else:
        cagr = 0

    annualized_return = np.mean(daily_returns) * trading_days_per_year * 100 if len(daily_returns) > 0 else 0

    # Volatility
    volatility = np.std(daily_returns) * np.sqrt(trading_days_per_year) * 100 if len(daily_returns) > 1 else 0

    # Sharpe Ratio
    excess_return = annualized_return - risk_free_rate * 100
    sharpe_ratio = excess_return / volatility if volatility > 0 else 0

    # Sortino Ratio (only considers downside volatility)
    negative_returns = daily_returns[daily_returns < 0]
    downside_volatility = np.std(negative_returns) * np.sqrt(trading_days_per_year) * 100 if len(negative_returns) > 1 else 0
    sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else 0

    # Drawdown analysis
    max_drawdown, max_dd_duration, avg_drawdown = _calculate_drawdowns(equities)

    # Calmar Ratio
    calmar_ratio = cagr / max_drawdown if max_drawdown > 0 else 0

    # Trade statistics
    trade_stats = _calculate_trade_statistics(trades)

    return PerformanceMetrics(
        total_return=total_return,
        total_return_percent=total_return_percent,
        annualized_return=annualized_return,
        cagr=cagr,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_dd_duration,
        avg_drawdown=avg_drawdown,
        **trade_stats,
    )


def _calculate_drawdowns(equities: list[float]) -> tuple[float, int, float]:
    """
    Calculate drawdown metrics.

    Returns:
        Tuple of (max_drawdown_percent, max_duration_days, avg_drawdown_percent)
    """
    if not equities:
        return 0, 0, 0

    peak = equities[0]
    max_drawdown = 0
    max_duration = 0
    current_duration = 0
    drawdowns = []
    in_drawdown = False

    for equity in equities:
        if equity > peak:
            peak = equity
            if in_drawdown:
                in_drawdown = False
                current_duration = 0
        else:
            drawdown = ((peak - equity) / peak) * 100 if peak > 0 else 0
            drawdowns.append(drawdown)

            if drawdown > max_drawdown:
                max_drawdown = drawdown

            if not in_drawdown:
                in_drawdown = True
                current_duration = 1
            else:
                current_duration += 1

            if current_duration > max_duration:
                max_duration = current_duration

    avg_drawdown = np.mean(drawdowns) if drawdowns else 0

    return max_drawdown, max_duration, avg_drawdown


def _calculate_trade_statistics(trades: list["BacktestTrade"]) -> dict:
    """Calculate trade-level statistics."""
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "avg_trade": 0,
            "avg_winner": 0,
            "avg_loser": 0,
            "best_trade": 0,
            "worst_trade": 0,
            "avg_holding_period": 0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "expectancy": 0,
            "expectancy_ratio": 0,
        }

    pnls = [t.pnl for t in trades]
    winners = [t.pnl for t in trades if t.pnl > 0]
    losers = [t.pnl for t in trades if t.pnl <= 0]

    total_trades = len(trades)
    winning_trades = len(winners)
    losing_trades = len(losers)

    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

    avg_trade = np.mean(pnls) if pnls else 0
    avg_winner = np.mean(winners) if winners else 0
    avg_loser = np.mean(losers) if losers else 0

    best_trade = max(pnls) if pnls else 0
    worst_trade = min(pnls) if pnls else 0

    # Holding period
    holding_periods = []
    for t in trades:
        if t.entry_time and t.exit_time:
            days = (t.exit_time - t.entry_time).days
            holding_periods.append(max(1, days))
    avg_holding_period = np.mean(holding_periods) if holding_periods else 0

    # Consecutive wins/losses
    max_wins, max_losses = _calculate_streaks(trades)

    # Expectancy
    expectancy = (win_rate * avg_winner) + ((1 - win_rate) * avg_loser)
    expectancy_ratio = abs(expectancy / avg_loser) if avg_loser != 0 else 0

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade": avg_trade,
        "avg_winner": avg_winner,
        "avg_loser": avg_loser,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "avg_holding_period": avg_holding_period,
        "max_consecutive_wins": max_wins,
        "max_consecutive_losses": max_losses,
        "expectancy": expectancy,
        "expectancy_ratio": expectancy_ratio,
    }


def _calculate_streaks(trades: list["BacktestTrade"]) -> tuple[int, int]:
    """Calculate maximum consecutive wins and losses."""
    if not trades:
        return 0, 0

    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0

    for trade in trades:
        if trade.pnl > 0:
            current_wins += 1
            current_losses = 0
            if current_wins > max_wins:
                max_wins = current_wins
        else:
            current_losses += 1
            current_wins = 0
            if current_losses > max_losses:
                max_losses = current_losses

    return max_wins, max_losses


def _empty_metrics() -> PerformanceMetrics:
    """Return empty metrics object."""
    return PerformanceMetrics(
        total_return=0,
        total_return_percent=0,
        annualized_return=0,
        cagr=0,
        volatility=0,
        sharpe_ratio=0,
        sortino_ratio=0,
        calmar_ratio=0,
        max_drawdown=0,
        max_drawdown_duration=0,
        avg_drawdown=0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0,
        profit_factor=0,
        avg_trade=0,
        avg_winner=0,
        avg_loser=0,
        best_trade=0,
        worst_trade=0,
        avg_holding_period=0,
        max_consecutive_wins=0,
        max_consecutive_losses=0,
        expectancy=0,
        expectancy_ratio=0,
    )
