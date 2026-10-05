import os
import sys
import math
import numpy as np
import pandas as pd

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from data.market_data import get_historical_ohlcv
from execution.roostoo_client import RoostooClient


# ==================================================
# CONFIGURATION
# ==================================================

ASSETS = {
    "BTC": {
        "binance": "BTCUSDT",
        "roostoo": "BTC/USD"
    },
    "ETH": {
        "binance": "ETHUSDT",
        "roostoo": "ETH/USD"
    },
    "SOL": {
        "binance": "SOLUSDT",
        "roostoo": "SOL/USD"
    },
    "BNB": {
        "binance": "BNBUSDT",
        "roostoo": "BNB/USD"
    },
    "XRP": {
        "binance": "XRPUSDT",
        "roostoo": "XRP/USD"
    }
}


# ==================================================
# FROZEN V1 STRATEGY
# ==================================================

MOMENTUM_SHORT = 24 * 30
MOMENTUM_LONG = 24 * 90

VOL_WINDOW = 24 * 30
ASSET_MA_WINDOW = 24 * 30

REGIME_MA_WINDOW = 24 * 200

TOP_N = 2

HOURS_PER_YEAR = 24 * 365

HISTORY_DAYS = 230


# ==================================================
# EXECUTION SETTINGS
# ==================================================

# KEEP FALSE FOR NOW.
LIVE_TRADING = False

# Keep 1% of the portfolio in USD.
CASH_BUFFER = 0.01

# Ignore tiny rebalance differences.
MIN_TRADE_USD = 5.00


# ==================================================
# DOWNLOAD DATA
# ==================================================

def load_market_data():

    data = {}

    print()
    print("DOWNLOADING MARKET DATA")
    print("=" * 60)

    for asset, config in ASSETS.items():

        print(f"Downloading {asset}...")

        df = get_historical_ohlcv(
            symbol=config["binance"],
            interval="1h",
            days=HISTORY_DAYS
        )

        # Drop the newest candle because it may
        # still be forming.
        if len(df) > 1:
            df = df.iloc[:-1].copy()

        df = df.set_index("open_time")

        data[asset] = df

    return data


# ==================================================
# PRICE MATRIX
# ==================================================

def build_prices(data):

    prices = pd.DataFrame()

    for asset, df in data.items():
        prices[asset] = df["close"]

    prices = (
        prices
        .dropna()
        .sort_index()
    )

    return prices


# ==================================================
# STRATEGY
# ==================================================

def calculate_target_weights(prices):

    returns = prices.pct_change()

    momentum_30 = prices.pct_change(
        MOMENTUM_SHORT
    )

    momentum_90 = prices.pct_change(
        MOMENTUM_LONG
    )

    volatility = (
        returns
        .rolling(VOL_WINDOW)
        .std()
        * np.sqrt(HOURS_PER_YEAR)
    )

    asset_ma = (
        prices
        .rolling(ASSET_MA_WINDOW)
        .mean()
    )

    above_asset_ma = (
        prices > asset_ma
    )

    btc_200d_ma = (
        prices["BTC"]
        .rolling(REGIME_MA_WINDOW)
        .mean()
    )

    latest_btc_price = (
        prices["BTC"].iloc[-1]
    )

    latest_btc_ma = (
        btc_200d_ma.iloc[-1]
    )

    risk_on = (
        latest_btc_price
        > latest_btc_ma
    )

    target_weights = pd.Series(
        0.0,
        index=prices.columns
    )

    # ----------------------------------------------
    # RISK-OFF
    # ----------------------------------------------

    if not risk_on:

        return {
            "weights": target_weights,
            "risk_on": False,
            "btc_price": latest_btc_price,
            "btc_200d_ma": latest_btc_ma
        }

    # ----------------------------------------------
    # RISK-ON
    # ----------------------------------------------

    mom_30 = momentum_30.iloc[-1]
    mom_90 = momentum_90.iloc[-1]
    vol = volatility.iloc[-1]
    trend = above_asset_ma.iloc[-1]

    # Frozen V1 score
    score = (
        0.3 * mom_30
        +
        0.7 * mom_90
    )

    eligible = (
        (score > 0)
        &
        trend
        &
        vol.notna()
    )

    ranked = (
        score[eligible]
        .sort_values(
            ascending=False
        )
    )

    selected = (
        ranked
        .head(TOP_N)
        .index
    )

    if len(selected) == 0:

        return {
            "weights": target_weights,
            "risk_on": True,
            "btc_price": latest_btc_price,
            "btc_200d_ma": latest_btc_ma
        }

    # ----------------------------------------------
    # INVERSE VOLATILITY
    # ----------------------------------------------

    inverse_vol = (
        1 / vol[selected]
    )

    inverse_vol = (
        inverse_vol
        /
        inverse_vol.sum()
    )

    target_weights[selected] = (
        inverse_vol
    )

    # ----------------------------------------------
    # 1% EXECUTION CASH BUFFER
    # ----------------------------------------------

    target_weights = (
        target_weights
        *
        (1 - CASH_BUFFER)
    )

    return {
        "weights": target_weights,
        "risk_on": True,
        "btc_price": latest_btc_price,
        "btc_200d_ma": latest_btc_ma
    }


# ==================================================
# EXCHANGE RULES
# ==================================================

def get_exchange_rules(client):

    response = (
        client.get_exchange_info()
    )

    # IMPORTANT FIX:
    # exchangeInfo does not necessarily return
    # {"Success": True}.
    #
    # Therefore we validate the field we actually
    # require instead.
    if "TradePairs" not in response:

        raise RuntimeError(
            "Could not retrieve exchange info: "
            + str(response)
        )

    trade_pairs = response["TradePairs"]

    rules = {}

    for asset, config in ASSETS.items():

        pair = config["roostoo"]

        pair_info = (
            trade_pairs.get(pair)
        )

        if pair_info is None:

            raise RuntimeError(
                f"Missing exchange rules "
                f"for {pair}"
            )

        rules[asset] = {
            "pair": pair,

            "amount_precision": int(
                pair_info[
                    "AmountPrecision"
                ]
            ),

            "price_precision": int(
                pair_info[
                    "PricePrecision"
                ]
            ),

            "mini_order": float(
                pair_info[
                    "MiniOrder"
                ]
            )
        }

    return rules


# ==================================================
# ROUND DOWN
# ==================================================

def round_down(
    value,
    precision
):

    """
    Round a number DOWN to a specified
    number of decimal places.

    Input
    -----
    value : float
    precision : int

    Output
    ------
    float
    """

    factor = (
        10 ** precision
    )

    return (
        math.floor(
            value * factor
        )
        /
        factor
    )


# ==================================================
# ROOSTOO PRICES
# ==================================================

def get_roostoo_prices(client):

    prices = {}

    for asset, config in ASSETS.items():

        pair = config["roostoo"]

        response = (
            client.get_ticker(pair)
        )

        if not response.get(
            "Success",
            False
        ):

            raise RuntimeError(
                f"Ticker failed for "
                f"{pair}: {response}"
            )

        pair_data = (
            response
            .get("Data", {})
            .get(pair, {})
        )

        price = (
            pair_data.get(
                "LastPrice"
            )
        )

        if price is None:

            raise RuntimeError(
                f"No price returned "
                f"for {pair}"
            )

        prices[asset] = float(
            price
        )

    return prices


# ==================================================
# CURRENT PORTFOLIO
# ==================================================

def get_current_portfolio(
    client,
    prices
):

    response = (
        client.get_balance()
    )

    if not response.get(
        "Success",
        False
    ):

        raise RuntimeError(
            "Could not retrieve balance: "
            + str(response)
        )

    wallet = response.get(
        "SpotWallet",
        {}
    )

    usd = float(
        wallet
        .get("USD", {})
        .get("Free", 0)
    )

    holdings = {}

    portfolio_value = usd

    for asset in ASSETS:

        quantity = float(
            wallet
            .get(asset, {})
            .get("Free", 0)
        )

        holdings[asset] = quantity

        portfolio_value += (
            quantity
            *
            prices[asset]
        )

    return {
        "usd": usd,
        "holdings": holdings,
        "portfolio_value": portfolio_value
    }


# ==================================================
# CALCULATE TRADES
# ==================================================

def calculate_trades(
    target_weights,
    portfolio,
    prices,
    rules
):

    total_value = (
        portfolio[
            "portfolio_value"
        ]
    )

    holdings = (
        portfolio[
            "holdings"
        ]
    )

    trades = []

    for asset in ASSETS:

        price = prices[asset]

        current_quantity = (
            holdings[asset]
        )

        current_value = (
            current_quantity
            *
            price
        )

        target_weight = float(
            target_weights[asset]
        )

        target_value = (
            total_value
            *
            target_weight
        )

        difference_usd = (
            target_value
            -
            current_value
        )

        # Ignore tiny differences
        if (
            abs(difference_usd)
            <
            MIN_TRADE_USD
        ):
            continue

        if difference_usd > 0:
            side = "BUY"
        else:
            side = "SELL"

        raw_quantity = (
            abs(difference_usd)
            /
            price
        )

        precision = (
            rules[asset][
                "amount_precision"
            ]
        )

        quantity = (
            round_down(
                raw_quantity,
                precision
            )
        )

        # ------------------------------------------
        # SELL SAFETY
        # ------------------------------------------

        if side == "SELL":

            max_sell = (
                round_down(
                    current_quantity,
                    precision
                )
            )

            quantity = min(
                quantity,
                max_sell
            )

        trade_value = (
            quantity
            *
            price
        )

        mini_order = (
            rules[asset][
                "mini_order"
            ]
        )

        minimum_required = max(
            MIN_TRADE_USD,
            mini_order
        )

        if (
            trade_value
            <
            minimum_required
        ):
            continue

        if quantity <= 0:
            continue

        trades.append(
            {
                "asset": asset,
                "pair": rules[
                    asset
                ]["pair"],
                "side": side,
                "price": price,
                "current_value":
                    current_value,
                "target_value":
                    target_value,
                "difference_usd":
                    difference_usd,
                "quantity":
                    quantity,
                "trade_value":
                    trade_value,
                "precision":
                    precision,
                "mini_order":
                    mini_order
            }
        )

    # SELL orders first.
    trades.sort(
        key=lambda trade:
            (
                0
                if trade["side"]
                == "SELL"
                else 1
            )
    )

    return trades


# ==================================================
# PRINT STRATEGY
# ==================================================

def print_strategy(result):

    print()
    print("STRATEGY SIGNAL")
    print("=" * 60)

    print(
        f"BTC Price:       "
        f"${result['btc_price']:,.2f}"
    )

    print(
        f"BTC 200d MA:     "
        f"${result['btc_200d_ma']:,.2f}"
    )

    print(
        "Regime:          "
        +
        (
            "RISK-ON"
            if result["risk_on"]
            else
            "RISK-OFF"
        )
    )

    print()
    print("Execution target:")

    weights = (
        result["weights"]
        .sort_values(
            ascending=False
        )
    )

    for asset, weight in (
        weights.items()
    ):

        print(
            f"  {asset}: "
            f"{weight:.2%}"
        )

    cash = (
        1
        -
        weights.sum()
    )

    print(
        f"  CASH: "
        f"{cash:.2%}"
    )


# ==================================================
# PRINT EXCHANGE RULES
# ==================================================

def print_exchange_rules(rules):

    print()
    print("ROOSTOO EXCHANGE RULES")
    print("=" * 60)

    for asset in ASSETS:

        rule = rules[asset]

        print(
            f"{rule['pair']}: "
            f"amount_precision="
            f"{rule['amount_precision']}, "
            f"min_order="
            f"${rule['mini_order']:.2f}"
        )


# ==================================================
# PRINT PORTFOLIO
# ==================================================

def print_portfolio(
    portfolio,
    prices
):

    print()
    print("CURRENT TEST PORTFOLIO")
    print("=" * 60)

    print(
        f"USD Cash: "
        f"${portfolio['usd']:,.2f}"
    )

    print()

    for asset in ASSETS:

        quantity = (
            portfolio[
                "holdings"
            ][asset]
        )

        value = (
            quantity
            *
            prices[asset]
        )

        print(
            f"{asset}: "
            f"{quantity:.8f} "
            f"(${value:,.2f})"
        )

    print()

    print(
        f"Total Portfolio Value: "
        f"${portfolio['portfolio_value']:,.2f}"
    )


# ==================================================
# PRINT TRADES
# ==================================================

def print_trades(trades):

    print()
    print("PROPOSED EXECUTION PLAN")
    print("=" * 60)

    if len(trades) == 0:

        print(
            "No trades required."
        )

        return

    for number, trade in enumerate(
        trades,
        start=1
    ):

        print()

        print(
            f"{number}. "
            f"{trade['side']} "
            f"{trade['asset']}"
        )

        print(
            f"   Pair:          "
            f"{trade['pair']}"
        )

        print(
            f"   Price:         "
            f"${trade['price']:,.4f}"
        )

        print(
            f"   Current value: "
            f"${trade['current_value']:,.2f}"
        )

        print(
            f"   Target value:  "
            f"${trade['target_value']:,.2f}"
        )

        print(
            f"   Difference:    "
            f"${trade['difference_usd']:,.2f}"
        )

        print(
            f"   Quantity:      "
            f"{trade['quantity']}"
        )

        print(
            f"   Trade value:   "
            f"${trade['trade_value']:,.2f}"
        )

        print(
            f"   Precision:     "
            f"{trade['precision']} "
            f"decimals"
        )


# ==================================================
# EXECUTION
# ==================================================

def execute_trades(
    client,
    trades
):

    # ----------------------------------------------
    # MASTER SAFETY SWITCH
    # ----------------------------------------------

    if not LIVE_TRADING:

        print()
        print(
            "LIVE_TRADING=False"
        )

        print(
            "Execution skipped."
        )

        return []

    results = []

    for trade in trades:

        print()

        print(
            f"EXECUTING: "
            f"{trade['side']} "
            f"{trade['quantity']} "
            f"{trade['asset']}"
        )

        result = (
            client.place_order(
                pair=trade["pair"],
                side=trade["side"],
                quantity=trade[
                    "quantity"
                ],
                order_type="MARKET"
            )
        )

        results.append(result)

        print(result)

    return results


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print(
        "ROOSTOO SAFE PORTFOLIO REBALANCER"
    )
    print("=" * 60)

    print(
        f"LIVE_TRADING = "
        f"{LIVE_TRADING}"
    )

    # ----------------------------------------------
    # API CLIENT
    # ----------------------------------------------

    client = RoostooClient()

    # ----------------------------------------------
    # EXCHANGE RULES
    # ----------------------------------------------

    rules = (
        get_exchange_rules(
            client
        )
    )

    print_exchange_rules(
        rules
    )

    # ----------------------------------------------
    # BINANCE DATA
    # ----------------------------------------------

    data = (
        load_market_data()
    )

    prices = (
        build_prices(
            data
        )
    )

    # ----------------------------------------------
    # STRATEGY
    # ----------------------------------------------

    strategy = (
        calculate_target_weights(
            prices
        )
    )

    print_strategy(
        strategy
    )

    # ----------------------------------------------
    # ROOSTOO PRICES
    # ----------------------------------------------

    print()
    print(
        "Reading Roostoo prices..."
    )

    roostoo_prices = (
        get_roostoo_prices(
            client
        )
    )

    # ----------------------------------------------
    # CURRENT PORTFOLIO
    # ----------------------------------------------

    portfolio = (
        get_current_portfolio(
            client,
            roostoo_prices
        )
    )

    print_portfolio(
        portfolio,
        roostoo_prices
    )

    # ----------------------------------------------
    # CALCULATE TRADES
    # ----------------------------------------------

    trades = (
        calculate_trades(
            strategy["weights"],
            portfolio,
            roostoo_prices,
            rules
        )
    )

    print_trades(trades)

    # ----------------------------------------------
    # EXECUTION
    # ----------------------------------------------

    execute_trades(
        client,
        trades
    )

    print()
    print("=" * 60)

    if LIVE_TRADING:

        print(
            "LIVE EXECUTION COMPLETE"
        )

    else:

        print(
            "DRY RUN COMPLETE — "
            "NO ORDERS WERE SENT"
        )

    print("=" * 60)
