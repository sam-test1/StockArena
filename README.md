# StockArena

Multi-stock intelligent trading simulator — compete against rule-based AI and Q-Learning AI across 6 stocks with distinct characteristics. Supports technical indicators, portfolio risk control, and backtest analysis.

## Quick Start

```bash
# Zero-config — just one command
./run.sh
```

Then open **http://localhost:8000** in your browser.

The script auto-creates a virtual environment and installs dependencies on first run. The only prerequisite is **Python 3.8+**.

## Features

- **6 Stocks with Distinct Characteristics** — Tech Blue Chip, Steady Growth, High Volatility, Hot Sector, Crypto Concept, Defensive Consumer
- **Three-Way Competition** — You (manual trading) vs. Rule AI vs. ML AI (Q-Learning)
- **Real-Time Technical Indicators** — MA, RSI, MACD, Bollinger Bands, ATR
- **Portfolio Management** — Multi-stock holding, risk control, auto-trading
- **ML Training Dashboard** — Adjustable learning rate, exploration rate, training data size, pre-training
- **Backtest Analysis** — Sharpe ratio, decision accuracy, max drawdown, win rate, profit factor
- **Asset Comparison** — Real-time chart comparing your assets vs. AI opponents

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Browser (Frontend)                │
│  index.html  +  app.js  +  Chart.js  +  Tailwind CSS │
│         │                                           │
│         │  HTTP REST API (JSON)                     │
│         ▼                                           │
│  ml_trading_api.py (Python HTTP Server)             │
│  ┌───────────────────────────────────────────────┐  │
│  │  QLearningAgent     TechnicalIndicators        │  │
│  │  (Double DQN)       (MA, RSI, MACD, BB, ATR)  │  │
│  │  MLTradingBot       RuleBasedBot               │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, JavaScript, Chart.js, Tailwind CSS, Font Awesome |
| Backend | Python 3 (pure standard library — no dependencies required) |
| ML Engine | Q-Learning with Double DQN, epsilon-greedy exploration |
| Optional | Flask, NumPy, Pandas, Matplotlib (for extended analysis) |

## How to Play

1. **Select a stock** from the stock pool (top panel)
2. **Buy/Sell** shares manually using the trading panel
3. **Enable auto-trading** to let the Rule AI trade for you
4. **Control the ML AI** via the learning panel — adjust learning rate, exploration rate, and pre-train
5. **Monitor performance** in real-time charts and backtest metrics
6. **Compare your assets** against both AI opponents in the asset comparison chart

### Stock Characteristics

| Stock | Sector | Volatility | Trend Bias |
|-------|--------|-----------|-------------|
| AAPL | Tech | Low | Slight upward |
| TSLA | Auto | Medium | Neutral |
| NVDA | Chip | High | Strong upward |
| JPM | Finance | Medium | Mild upward |
| AMZN | Consumer | Medium | Upward |
| BTC | Crypto | Extreme | Random walk |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/ml/status` | Get ML model status |
| `POST` | `/api/ml/action` | Get ML trading decision |
| `POST` | `/api/ml/train` | Train ML model on historical data |
| `POST` | `/api/ml/feedback` | Send reward feedback to ML |
| `POST` | `/api/ml/reset` | Reset ML model |
| `POST` | `/api/ml/save` | Save ML model to disk |
| `POST` | `/api/ml/load` | Load ML model from disk |
| `POST` | `/api/ml/indicators` | Calculate technical indicators |
| `POST` | `/api/ml/backtest` | Get backtest metrics |

## Project Structure

```
stock-trading-simulator/
├── index.html          # Frontend page
├── app.js              # Frontend logic
├── ml_trading_api.py   # Python ML backend
├── requirements_ml.txt # Python dependencies
├── run.sh              # One-click startup script
└── .gitignore
```

## License

MIT