# Roostoo Quant Trading Hackathon 2026

Autonomous quantitative trading system developed for the Roostoo Quant Trading Hackathon 2026.

## Strategy Overview

The strategy uses a regime-aware cross-sectional momentum framework across:

- BTC
- ETH
- SOL
- BNB
- XRP

The system combines medium- and long-horizon momentum with trend and volatility filters to construct the portfolio.

### 1. Market Regime

BTC is used as the market regime indicator.

- BTC above its 200-day moving average: risk-on
- BTC below its 200-day moving average: risk-off

During risk-off periods, the strategy moves to cash.

### 2. Asset Selection

During risk-on periods, each asset receives a momentum score:

`Momentum Score = 0.3 × 30-day momentum + 0.7 × 90-day momentum`

Assets must also:

- have positive momentum;
- trade above their 30-day moving average.

The two highest-scoring eligible assets are selected.

### 3. Portfolio Construction

Selected assets are weighted using inverse 30-day realised volatility.

This gives lower-volatility assets a larger allocation and reduces concentration in more volatile assets.

The strategy maintains a small cash buffer for fees and execution.

### 4. Autonomous Rebalancing

The bot recalculates the strategy using the latest completed market data.

A rebalance is generated when the difference between the strategy target and current portfolio exceeds the configured portfolio-weight threshold.

All trades are generated programmatically by the strategy. No manual trading is required.

### 5. Risk Management

Risk controls include:

- BTC 200-day moving-average regime filter
- individual asset trend filters
- inverse-volatility position sizing
- diversification across the two strongest eligible assets
- cash buffer for trading costs
- minimum rebalance threshold to reduce unnecessary turnover
- risk-off transition to cash

## Data

Historical market data is sourced from Binance.

Only completed candles are used for signal generation to avoid making decisions using incomplete price data.

Roostoo market prices and account information are retrieved through the official Roostoo API.

## Execution

The execution engine:

1. retrieves the current Roostoo portfolio;
2. calculates target portfolio weights;
3. compares current and target positions;
4. executes required sells before buys;
5. respects Roostoo quantity precision and minimum-order requirements;
6. submits market orders through the Roostoo API;
7. verifies submitted orders using `/v3/query_order`;
8. records strategy decisions and executed trades.

The bot runs autonomously on AWS EC2.

## Repository Structure

```text
roostoo-bot/
├── data/
│   └── market_data.py
├── execution/
│   └── roostoo_client.py
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
