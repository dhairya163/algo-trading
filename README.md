# Sentiment Algo Trader 📈

A sentiment-based algorithmic trading system for Indian stock markets (NSE/BSE) using the Groww API.

## Overview

This system analyzes news and social media sentiment to generate trading signals for stocks. It combines multiple data sources and NLP models to make informed trading decisions.

### Key Features

- **Multi-source Sentiment Analysis**: Aggregates sentiment from news APIs (Alpha Vantage, NewsAPI, Finnhub) and social media
- **Advanced NLP**: Uses FinBERT (financial domain BERT) for accurate sentiment classification
- **Multiple Trading Strategies**: Momentum and Mean Reversion strategies based on sentiment
- **Risk Management**: Position sizing, stop losses, daily loss limits, and circuit breakers
- **Paper Trading**: Test strategies without risking real money
- **Backtesting**: Evaluate strategies on historical data
- **Live Trading**: Connect to Groww API for real execution
- **REST API Dashboard**: Monitor portfolio, signals, and risk metrics

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SENTIMENT ALGO TRADING SYSTEM                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐     │
│  │  DATA LAYER   │   │ ANALYSIS LAYER│   │EXECUTION LAYER│     │
│  ├───────────────┤   ├───────────────┤   ├───────────────┤     │
│  │ • News APIs   │──▶│ • FinBERT NLP │──▶│ • Groww API   │     │
│  │ • Social APIs │   │ • Aggregator  │   │ • Paper Trader│     │
│  │ • Market Data │   │ • Strategies  │   │ • Risk Checks │     │
│  └───────────────┘   └───────────────┘   └───────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repo-url>
cd algo-trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 2. Configuration

```bash
# Copy example config
cp .env.example .env

# Edit .env with your API keys
# Required for basic functionality:
# - ALPHA_VANTAGE_API_KEY (free at alphavantage.co)
# - FINNHUB_API_KEY (free at finnhub.io)
#
# Required for live trading:
# - GROWW_API_KEY (from groww.in/trade-api)
```

### 3. Run Paper Trading

```bash
python scripts/run_paper.py
```

### 4. Run Backtest

```bash
python scripts/run_backtest.py --strategy momentum --days 365
```

### 5. Start Dashboard

```bash
python scripts/run_dashboard.py
# Open http://localhost:8000/docs for API documentation
```

## Project Structure

```
algo-trading/
├── config/
│   └── settings.py          # Configuration management
├── src/
│   ├── data/                 # Data ingestion
│   │   ├── news_fetcher.py   # News API clients
│   │   ├── sentiment_apis.py # Sentiment data APIs
│   │   └── market_data.py    # Price data
│   ├── sentiment/            # Sentiment analysis
│   │   ├── analyzer.py       # NLP models (FinBERT, VADER)
│   │   └── aggregator.py     # Multi-source aggregation
│   ├── strategy/             # Trading strategies
│   │   ├── momentum.py       # Momentum strategy
│   │   └── signal_generator.py
│   ├── execution/            # Order execution
│   │   ├── groww_client.py   # Groww API wrapper
│   │   ├── paper_trader.py   # Paper trading simulator
│   │   └── order_manager.py  # Order lifecycle
│   ├── risk/                 # Risk management
│   │   ├── position_sizer.py # Position sizing
│   │   └── risk_manager.py   # Risk controls
│   ├── backtest/             # Backtesting
│   │   ├── backtester.py     # Backtest engine
│   │   └── metrics.py        # Performance metrics
│   └── dashboard/            # REST API
│       └── api.py            # FastAPI endpoints
├── scripts/
│   ├── run_paper.py          # Paper trading
│   ├── run_backtest.py       # Backtesting
│   └── run_dashboard.py      # Dashboard server
└── tests/
```

## Trading Strategies

### Momentum Strategy
- **Entry**: Buy when sentiment > 0.25 and trend is improving
- **Exit**: Sell when sentiment drops below 0.1 or trend deteriorates
- **Best for**: Trending markets, news-driven moves

### Mean Reversion Strategy
- **Entry**: Buy when sentiment is extremely negative (< -0.6)
- **Exit**: Close when sentiment normalizes to -0.2
- **Best for**: Oversold bounces, contrarian plays

## Risk Management

| Feature | Description | Default |
|---------|-------------|---------|
| Position Sizing | Max single position | 10% of capital |
| Stop Loss | Auto stop loss | 3% |
| Take Profit | Auto take profit | 6% |
| Daily Loss Limit | Circuit breaker | 5% |
| Max Positions | Maximum open positions | 10 |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/portfolio` | GET | Portfolio summary |
| `/positions` | GET | Current positions |
| `/signals` | GET | Recent trading signals |
| `/signals/{symbol}` | GET | Signal for specific symbol |
| `/risk` | GET | Risk metrics |
| `/trades` | GET | Trade history |
| `/trading/halt` | POST | Halt trading |
| `/trading/resume` | POST | Resume trading |

## Data Sources

### Sentiment Data
- **Alpha Vantage**: News sentiment with scores
- **Finnhub**: Social media sentiment (Reddit, Twitter)
- **NewsAPI**: General financial news

### Market Data
- **Groww API**: Real-time prices, order execution
- **Yahoo Finance**: Historical data (backup)

## Performance Metrics

The backtester calculates comprehensive metrics:
- Total Return & CAGR
- Sharpe Ratio & Sortino Ratio
- Maximum Drawdown
- Win Rate & Profit Factor
- Average Trade & Expectancy

## Configuration Options

Key settings in `.env`:

```bash
# Trading mode
TRADING_MODE=paper  # paper, live, backtest

# Capital
INITIAL_CAPITAL=100000

# Risk limits
MAX_POSITION_SIZE_PERCENT=10
STOP_LOSS_PERCENT=3
MAX_DAILY_LOSS_PERCENT=5

# Sentiment thresholds
MIN_SENTIMENT_SCORE=0.3
SENTIMENT_SMOOTHING_WINDOW=3
```

## Disclaimer

⚠️ **This software is for educational purposes only.**

- Trading involves substantial risk of loss
- Past performance does not guarantee future results
- Always test with paper trading before using real money
- The authors are not responsible for any financial losses

## License

MIT License - See LICENSE file for details

## Sources

This project uses the following APIs:
- [Groww Trading API](https://groww.in/trade-api)
- [Alpha Vantage](https://www.alphavantage.co/)
- [Finnhub](https://finnhub.io/)
- [NewsAPI](https://newsapi.org/)
