import sys
import os
import numpy as np
import pandas as pd

# Allow imports from project root
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from data.market_data import get_historical_ohlcv


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ASSETS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "BNB": "BNBUSDT",
    "XRP": "XRPUSDT"
}

HISTORY_DAYS = 365

# Momentum horizons
MOMENTUM_SHORT = 24 * 7       # 7 days
MOMENTUM_LONG = 24 * 30       # 30 days

# Volatility lookback
VOLATILITY_WINDOW = 24 * 14   # 14 days

# Number of assets we are willing to hold
TOP_N = 2


# --------------------------------------------------
# Download market data
# --------------------------------------------------

def download_assets():
    """
    Download one year of hourly data for all assets.

    Returns
    -------
    dict
        Example:
        {
            "BTC": DataFrame,
            "ETH": DataFrame,
            ...
        }
    """

    market_data = {}

    for name, symbol in ASSETS.items():

        print()
        print(f"Downloading {name}...")

        df = get_historical_ohlcv(
            symbol=symbol,
            interval="1h",
            days=HISTORY_DAYS
        )

        market_data[name] = df

    return market_data


# --------------------------------------------------
# Calculate features
# --------------------------------------------------

def calculate_features(df):
    """
    Calculate momentum and volatility features.

    Returns
    -------
    pandas.DataFrame
    """

    df = df.copy()

    # Hourly returns
    df["hourly_return"] = (
        df["close"].pct_change()
    )

    # 7-day momentum
    df["momentum_7d"] = (
        df["close"].pct_change(
            MOMENTUM_SHORT
        )
    )

    # 30-day momentum
    df["momentum_30d"] = (
        df["close"].pct_change(
            MOMENTUM_LONG
        )
    )

    # Annualised volatility based on hourly returns
    df["volatility"] = (
        df["hourly_return"]
        .rolling(VOLATILITY_WINDOW)
        .std()
        * np.sqrt(24 * 365)
    )

    # Long-term trend filter:
    # 200-hour moving average
    df["ma_200"] = (
        df["close"]
        .rolling(200)
        .mean()
    )

    df["above_ma_200"] = (
        df["close"] > df["ma_200"]
    )

    return df


# --------------------------------------------------
# Create latest portfolio
# --------------------------------------------------

def build_portfolio(market_data):
    """
    Rank assets using momentum and construct
    an inverse-volatility portfolio.

    Returns
    -------
    pandas.DataFrame
        Current portfolio decision.
    """

    rows = []

    for asset, df in market_data.items():

        featured = calculate_features(df)

        latest = featured.iloc[-1]

        rows.append({
            "asset": asset,
            "price": latest["close"],
            "momentum_7d": latest["momentum_7d"],
            "momentum_30d": latest["momentum_30d"],
            "volatility": latest["volatility"],
            "above_ma_200": latest["above_ma_200"]
        })

    portfolio = pd.DataFrame(rows)

    # --------------------------------------------------
    # Combined momentum score
    # --------------------------------------------------

    portfolio["momentum_score"] = (
        0.4 * portfolio["momentum_7d"]
        +
        0.6 * portfolio["momentum_30d"]
    )

    # --------------------------------------------------
    # Eligibility filter
    # --------------------------------------------------

    portfolio["eligible"] = (
        (portfolio["momentum_score"] > 0)
        &
        (portfolio["above_ma_200"])
    )

    # Rank strongest momentum first
    portfolio = portfolio.sort_values(
        "momentum_score",
        ascending=False
    )

    # --------------------------------------------------
    # Select top N eligible assets
    # --------------------------------------------------

    eligible_assets = portfolio[
        portfolio["eligible"]
    ].head(TOP_N)

    portfolio["selected"] = False

    portfolio.loc[
        eligible_assets.index,
        "selected"
    ] = True

    # --------------------------------------------------
    # Inverse-volatility weights
    # --------------------------------------------------

    portfolio["raw_weight"] = 0.0
    portfolio["weight"] = 0.0

    selected_mask = portfolio["selected"]

    if selected_mask.any():

        portfolio.loc[
            selected_mask,
            "raw_weight"
        ] = (
            1
            /
            portfolio.loc[
                selected_mask,
                "volatility"
            ]
        )

        total_raw_weight = (
            portfolio.loc[
                selected_mask,
                "raw_weight"
            ].sum()
        )

        if total_raw_weight > 0:

            portfolio.loc[
                selected_mask,
                "weight"
            ] = (
                portfolio.loc[
                    selected_mask,
                    "raw_weight"
                ]
                /
                total_raw_weight
            )

    return portfolio


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":

    market_data = download_assets()

    portfolio = build_portfolio(
        market_data
    )

    print()
    print("MULTI-ASSET MOMENTUM PORTFOLIO")
    print("=" * 75)

    display_columns = [
        "asset",
        "price",
        "momentum_7d",
        "momentum_30d",
        "momentum_score",
        "volatility",
        "above_ma_200",
        "eligible",
        "selected",
        "weight"
    ]

    print(
        portfolio[
            display_columns
        ].to_string(
            index=False,
            formatters={
                "price":
                    lambda x: f"{x:,.4f}",

                "momentum_7d":
                    lambda x: f"{x:.2%}",

                "momentum_30d":
                    lambda x: f"{x:.2%}",

                "momentum_score":
                    lambda x: f"{x:.2%}",

                "volatility":
                    lambda x: f"{x:.2%}",

                "weight":
                    lambda x: f"{x:.2%}"
            }
        )
    )
