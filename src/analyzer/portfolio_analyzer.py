"""
Portfolio analyzer that connects to Groww and provides recommendations.

Main entry point for analyzing your portfolio.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.analyzer.openai_analyzer import OpenAIStockAnalyzer, Recommendation, StockAnalysis


# Company name mapping for NSE stocks
COMPANY_NAMES = {
    "RELIANCE": "Reliance Industries Limited",
    "TCS": "Tata Consultancy Services",
    "HDFCBANK": "HDFC Bank Limited",
    "INFY": "Infosys Limited",
    "ICICIBANK": "ICICI Bank Limited",
    "HINDUNILVR": "Hindustan Unilever Limited",
    "SBIN": "State Bank of India",
    "BHARTIARTL": "Bharti Airtel Limited",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "ITC": "ITC Limited",
    "LT": "Larsen & Toubro Limited",
    "AXISBANK": "Axis Bank Limited",
    "ASIANPAINT": "Asian Paints Limited",
    "MARUTI": "Maruti Suzuki India Limited",
    "BAJFINANCE": "Bajaj Finance Limited",
    "WIPRO": "Wipro Limited",
    "HCLTECH": "HCL Technologies Limited",
    "TATAMOTORS": "Tata Motors Limited",
    "SUNPHARMA": "Sun Pharmaceutical Industries",
    "TITAN": "Titan Company Limited",
    "TATASTEEL": "Tata Steel Limited",
    "POWERGRID": "Power Grid Corporation",
    "NTPC": "NTPC Limited",
    "ONGC": "Oil and Natural Gas Corporation",
    "COALINDIA": "Coal India Limited",
    "ADANIENT": "Adani Enterprises Limited",
    "ADANIPORTS": "Adani Ports and SEZ",
    "BAJAJFINSV": "Bajaj Finserv Limited",
    "NESTLEIND": "Nestle India Limited",
    "ULTRACEMCO": "UltraTech Cement Limited",
    "JSWSTEEL": "JSW Steel Limited",
    "TECHM": "Tech Mahindra Limited",
    "HDFCLIFE": "HDFC Life Insurance",
    "SBILIFE": "SBI Life Insurance",
    "DIVISLAB": "Divi's Laboratories",
    "DRREDDY": "Dr. Reddy's Laboratories",
    "CIPLA": "Cipla Limited",
    "APOLLOHOSP": "Apollo Hospitals",
    "EICHERMOT": "Eicher Motors Limited",
    "HEROMOTOCO": "Hero MotoCorp Limited",
    "M&M": "Mahindra & Mahindra Limited",
    "BAJAJ-AUTO": "Bajaj Auto Limited",
    "BRITANNIA": "Britannia Industries",
    "GRASIM": "Grasim Industries Limited",
    "INDUSINDBK": "IndusInd Bank Limited",
    "HINDALCO": "Hindalco Industries",
    "BPCL": "Bharat Petroleum Corporation",
    "TATACONSUM": "Tata Consumer Products",
    "UPL": "UPL Limited",
}


@dataclass
class PortfolioSummary:
    """Summary of portfolio analysis."""

    total_value: float
    total_invested: float
    total_pnl: float
    total_pnl_percent: float
    holdings_count: int

    strong_buys: list[StockAnalysis]
    buys: list[StockAnalysis]
    holds: list[StockAnalysis]
    sells: list[StockAnalysis]
    strong_sells: list[StockAnalysis]

    timestamp: datetime


class PortfolioAnalyzer:
    """
    Analyzes your Groww portfolio using OpenAI.

    Usage:
        analyzer = PortfolioAnalyzer(groww_api_key, openai_api_key)
        await analyzer.initialize()
        summary = await analyzer.analyze_portfolio()
        analyzer.print_report(summary)
    """

    def __init__(
        self,
        groww_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o",
    ):
        """
        Initialize portfolio analyzer.

        Args:
            groww_api_key: Groww API key
            openai_api_key: OpenAI API key
            openai_model: OpenAI model to use
        """
        self.groww_api_key = groww_api_key
        self.openai_analyzer = OpenAIStockAnalyzer(
            api_key=openai_api_key,
            model=openai_model,
        )
        self._groww_api = None
        self._console = Console()

    async def initialize(self) -> bool:
        """Initialize Groww API connection."""
        if not self.groww_api_key:
            logger.warning("Groww API key not provided - will use manual input")
            return False

        try:
            from growwapi import GrowwAPI
            self._groww_api = GrowwAPI(api_key=self.groww_api_key)
            logger.info("Connected to Groww API")
            return True
        except ImportError:
            logger.warning("growwapi not installed - will use manual input")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Groww: {e}")
            return False

    async def close(self):
        """Close connections."""
        await self.openai_analyzer.close()

    async def get_holdings_from_groww(self) -> list[dict]:
        """Fetch holdings from Groww API."""
        if not self._groww_api:
            raise RuntimeError("Groww API not initialized")

        try:
            holdings = self._groww_api.get_holdings()

            return [
                {
                    "symbol": h.get("symbol", "").replace("-EQ", ""),
                    "company_name": COMPANY_NAMES.get(
                        h.get("symbol", "").replace("-EQ", ""),
                        h.get("symbol", "")
                    ),
                    "quantity": h.get("quantity", 0),
                    "avg_cost": h.get("average_price", 0),
                    "current_price": h.get("last_price", 0),
                }
                for h in holdings
                if h.get("quantity", 0) > 0
            ]
        except Exception as e:
            logger.error(f"Failed to fetch holdings: {e}")
            return []

    def get_holdings_manual(self) -> list[dict]:
        """Get holdings from manual input."""
        print("\n" + "=" * 60)
        print("ENTER YOUR HOLDINGS")
        print("=" * 60)
        print("Enter each holding in format: SYMBOL,QUANTITY,AVG_COST,CURRENT_PRICE")
        print("Example: RELIANCE,10,2500,2650")
        print("Type 'done' when finished")
        print("-" * 60)

        holdings = []
        while True:
            try:
                line = input("> ").strip()
                if line.lower() == "done":
                    break
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) < 4:
                    print("Invalid format. Use: SYMBOL,QUANTITY,AVG_COST,CURRENT_PRICE")
                    continue

                symbol = parts[0].strip().upper()
                holdings.append({
                    "symbol": symbol,
                    "company_name": COMPANY_NAMES.get(symbol, symbol),
                    "quantity": int(parts[1].strip()),
                    "avg_cost": float(parts[2].strip()),
                    "current_price": float(parts[3].strip()),
                })
                print(f"  ✓ Added {symbol}")

            except ValueError as e:
                print(f"Error parsing input: {e}")
            except KeyboardInterrupt:
                break

        return holdings

    async def analyze_portfolio(
        self,
        holdings: Optional[list[dict]] = None,
    ) -> PortfolioSummary:
        """
        Analyze entire portfolio.

        Args:
            holdings: List of holdings (fetches from Groww if not provided)

        Returns:
            PortfolioSummary with recommendations
        """
        # Get holdings
        if holdings is None:
            if self._groww_api:
                holdings = await self.get_holdings_from_groww()
            else:
                holdings = self.get_holdings_manual()

        if not holdings:
            raise ValueError("No holdings to analyze")

        logger.info(f"Analyzing {len(holdings)} holdings...")

        # Calculate portfolio totals
        total_value = sum(h["current_price"] * h["quantity"] for h in holdings)
        total_invested = sum(h["avg_cost"] * h["quantity"] for h in holdings)
        total_pnl = total_value - total_invested
        total_pnl_percent = (total_pnl / total_invested * 100) if total_invested > 0 else 0

        # Analyze each stock
        analyses = await self.openai_analyzer.analyze_stocks_batch(holdings)

        # Categorize by recommendation
        strong_buys = [a for a in analyses if a.recommendation == Recommendation.STRONG_BUY]
        buys = [a for a in analyses if a.recommendation == Recommendation.BUY]
        holds = [a for a in analyses if a.recommendation == Recommendation.HOLD]
        sells = [a for a in analyses if a.recommendation == Recommendation.SELL]
        strong_sells = [a for a in analyses if a.recommendation == Recommendation.STRONG_SELL]

        return PortfolioSummary(
            total_value=total_value,
            total_invested=total_invested,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            holdings_count=len(holdings),
            strong_buys=strong_buys,
            buys=buys,
            holds=holds,
            sells=sells,
            strong_sells=strong_sells,
            timestamp=datetime.now(),
        )

    async def analyze_watchlist(
        self,
        symbols: list[str],
        prices: Optional[dict[str, float]] = None,
    ) -> list[StockAnalysis]:
        """
        Analyze a watchlist for potential buys.

        Args:
            symbols: List of stock symbols
            prices: Optional dict of symbol -> current price

        Returns:
            List of StockAnalysis sorted by recommendation
        """
        stocks = []
        for symbol in symbols:
            stocks.append({
                "symbol": symbol,
                "company_name": COMPANY_NAMES.get(symbol, symbol),
                "current_price": prices.get(symbol) if prices else None,
            })

        analyses = await self.openai_analyzer.analyze_stocks_batch(stocks)

        # Sort by recommendation strength
        order = {
            Recommendation.STRONG_BUY: 0,
            Recommendation.BUY: 1,
            Recommendation.HOLD: 2,
            Recommendation.SELL: 3,
            Recommendation.STRONG_SELL: 4,
        }

        return sorted(analyses, key=lambda a: (order.get(a.recommendation, 2), -a.confidence))

    def print_report(self, summary: PortfolioSummary):
        """Print formatted analysis report."""
        console = self._console

        # Header
        console.print("\n")
        console.print(Panel.fit(
            "[bold blue]📊 PORTFOLIO ANALYSIS REPORT[/bold blue]",
            border_style="blue",
        ))

        # Portfolio Summary
        pnl_color = "green" if summary.total_pnl >= 0 else "red"
        summary_table = Table(show_header=False, box=None)
        summary_table.add_column("Metric", style="dim")
        summary_table.add_column("Value", justify="right")

        summary_table.add_row("Total Value", f"₹{summary.total_value:,.2f}")
        summary_table.add_row("Total Invested", f"₹{summary.total_invested:,.2f}")
        summary_table.add_row(
            "Total P&L",
            f"[{pnl_color}]₹{summary.total_pnl:,.2f} ({summary.total_pnl_percent:+.1f}%)[/{pnl_color}]"
        )
        summary_table.add_row("Holdings", str(summary.holdings_count))

        console.print(Panel(summary_table, title="Portfolio Summary", border_style="cyan"))

        # Recommendations
        def print_category(title: str, analyses: list[StockAnalysis], color: str, emoji: str):
            if not analyses:
                return

            console.print(f"\n{emoji} [bold {color}]{title}[/bold {color}]")

            table = Table(show_header=True, header_style="bold")
            table.add_column("Stock", style="cyan")
            table.add_column("Confidence", justify="center")
            table.add_column("Target", justify="right")
            table.add_column("Reasoning", max_width=50)

            for a in analyses:
                target = f"₹{a.target_price:,.0f}" if a.target_price else "-"
                table.add_row(
                    f"{a.symbol}",
                    f"{a.confidence:.0f}%",
                    target,
                    a.reasoning[:100] + "..." if len(a.reasoning) > 100 else a.reasoning,
                )

            console.print(table)

        # Print each category
        print_category("🚀 STRONG BUY - Add More!", summary.strong_buys, "green", "🟢🟢")
        print_category("📈 BUY - Good to Add", summary.buys, "green", "🟢")
        print_category("📊 HOLD - Keep Position", summary.holds, "yellow", "🟡")
        print_category("📉 SELL - Consider Exiting", summary.sells, "red", "🔴")
        print_category("⚠️  STRONG SELL - Exit Now!", summary.strong_sells, "red", "🔴🔴")

        # Detailed Analysis
        console.print("\n")
        console.print(Panel.fit("[bold]DETAILED ANALYSIS[/bold]", border_style="blue"))

        all_analyses = (
            summary.strong_sells + summary.sells +
            summary.holds +
            summary.buys + summary.strong_buys
        )

        for analysis in all_analyses:
            self._print_stock_detail(analysis)

        # Timestamp
        console.print(f"\n[dim]Analysis generated at: {summary.timestamp.strftime('%Y-%m-%d %H:%M:%S')}[/dim]")

    def _print_stock_detail(self, analysis: StockAnalysis):
        """Print detailed analysis for a single stock."""
        console = self._console

        rec_colors = {
            Recommendation.STRONG_BUY: "bold green",
            Recommendation.BUY: "green",
            Recommendation.HOLD: "yellow",
            Recommendation.SELL: "red",
            Recommendation.STRONG_SELL: "bold red",
        }
        color = rec_colors.get(analysis.recommendation, "white")

        # Stock header
        console.print(f"\n[bold cyan]━━━ {analysis.symbol} - {analysis.company_name} ━━━[/bold cyan]")
        console.print(f"[{color}]Recommendation: {analysis.recommendation.value} ({analysis.confidence:.0f}% confidence)[/{color}]")

        if analysis.current_price:
            console.print(f"Current Price: ₹{analysis.current_price:,.2f}")
        if analysis.target_price:
            console.print(f"Target Price: ₹{analysis.target_price:,.2f}")
        if analysis.stop_loss:
            console.print(f"Stop Loss: ₹{analysis.stop_loss:,.2f}")

        # Analysis sections
        if analysis.sentiment_summary:
            console.print(f"\n[dim]Sentiment:[/dim] {analysis.sentiment_summary}")
        if analysis.technical_summary:
            console.print(f"[dim]Technical:[/dim] {analysis.technical_summary}")
        if analysis.fundamental_summary:
            console.print(f"[dim]Fundamental:[/dim] {analysis.fundamental_summary}")

        if analysis.risks:
            console.print(f"\n[red]Risks:[/red] {', '.join(analysis.risks)}")
        if analysis.catalysts:
            console.print(f"[green]Catalysts:[/green] {', '.join(analysis.catalysts)}")

        console.print(f"\n[italic]{analysis.reasoning}[/italic]")
