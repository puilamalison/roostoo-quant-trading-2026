import os
import json
import math
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from data.market_data import get_historical_ohlcv
from execution.roostoo_client import RoostooClient


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

# IMPORTANT:
# Keep False until dry run is validated.
LIVE_TRADING = False

ASSETS = {
    "BTC": {
        "binance": "BTCUSDT",
        "roostoo": "BTC/USD",
    },
    "ETH": {
        "binance": "ETHUSDT",
        "roostoo": "ETH/USD",
    },
    "SOL": {
        "binance": "SOLUSDT",
        "roostoo": "SOL/USD",
    },
    "BNB": {
        "binance": "BNBUSDT",
        "roostoo": "BNB/USD",
    },
    "XRP": {
        "binance": "XRPUSDT",
        "roostoo": "XRP/USD",
    },
}

HISTORY_DAYS = 230

MOM_30_HOURS = 24 * 30
MOM_90_HOURS = 24 * 90
VOL_30_HOURS = 24 * 30
MA_30_HOURS = 24 * 30
BTC_200D_HOURS = 24 * 200

TOP_N = 2

# 98.8% invested = 1.2% cash buffer.
INVESTED_FRACTION = 0.988

CORE_REBALANCE_DAYS = 7

# Daily risk-layer trigger.
DRIFT_THRESHOLD = 0.03

MIN_TRADE_USD = 5.0

# Prevent accidental repeated execution.
MIN_HOURS_BETWEEN_EXECUTIONS = 20

STATE_DIR = Path("state")
LOG_DIR = Path("logs")

STATE_FILE = STATE_DIR / "bot_state.json"
TRADE_LOG = LOG_DIR / "trades.csv"
BOT_LOG = LOG_DIR / "bot.log"


# ============================================================
# LOGGING
# ============================================================

def setup_logging():

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
        handlers=[
            logging.FileHandler(BOT_LOG),
            logging.StreamHandler(),
        ],
    )


# ============================================================
# STATE
# ============================================================

def load_state():

    STATE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not STATE_FILE.exists():

        return {
            "last_execution_time": None,
            "last_core_rebalance": None,
        }

    with open(
        STATE_FILE,
        "r"
    ) as f:

        return json.load(f)


def save_state(state):

    STATE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        STATE_FILE,
        "w"
    ) as f:

        json.dump(
            state,
            f,
            indent=2
        )


# ============================================================
# MARKET DATA
# ============================================================

def load_market_data():

    prices = pd.DataFrame()

    for asset, info in ASSETS.items():

        logging.info(
            "Downloading %s data...",
            asset
        )

        df = get_historical_ohlcv(
            symbol=info["binance"],
            interval="1h",
            days=HISTORY_DAYS
        )

        # Remove potentially incomplete candle.
        if len(df) > 1:
            df = df.iloc[:-1].copy()

        df = df.set_index(
            "open_time"
        )

        prices[asset] = (
            df["close"]
        )

    prices = (
        prices
        .dropna()
        .sort_index()
    )

    if len(prices) < BTC_200D_HOURS:

        raise RuntimeError(
            "Insufficient historical data."
        )

    return prices


# ============================================================
# FROZEN V1 STRATEGY
# ============================================================

def calculate_target(prices):

    returns = (
        prices.pct_change()
    )

    mom_30 = (
        prices.pct_change(
            MOM_30_HOURS
        )
    )

    mom_90 = (
        prices.pct_change(
            MOM_90_HOURS
        )
    )

    vol_30 = (
        returns
        .rolling(
            VOL_30_HOURS
        )
        .std()
        *
        np.sqrt(
            24 * 365
        )
    )

    ma_30 = (
        prices
        .rolling(
            MA_30_HOURS
        )
        .mean()
    )

    btc_200 = (
        prices["BTC"]
        .rolling(
            BTC_200D_HOURS
        )
        .mean()
    )

    btc_price = float(
        prices["BTC"].iloc[-1]
    )

    btc_ma = float(
        btc_200.iloc[-1]
    )

    target = pd.Series(
        0.0,
        index=prices.columns
    )

    # --------------------------------------------------------
    # BTC REGIME FILTER
    # --------------------------------------------------------

    risk_on = (
        btc_price > btc_ma
    )

    if not risk_on:

        return {
            "risk_on": False,
            "btc_price": btc_price,
            "btc_ma": btc_ma,
            "weights": target,
        }

    # --------------------------------------------------------
    # MOMENTUM SCORE
    # --------------------------------------------------------

    score = (
        0.3
        *
        mom_30.iloc[-1]
        +
        0.7
        *
        mom_90.iloc[-1]
    )

    current_vol = (
        vol_30.iloc[-1]
    )

    trend = (
        prices.iloc[-1]
        >
        ma_30.iloc[-1]
    )

    eligible = (
        (score > 0)
        &
        trend
        &
        current_vol.notna()
        &
        (current_vol > 0)
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

    if len(selected) > 0:

        inverse_vol = (
            1.0
            /
            current_vol[selected]
        )

        inverse_vol = (
            inverse_vol
            /
            inverse_vol.sum()
        )

        target[selected] = (
            inverse_vol
            *
            INVESTED_FRACTION
        )

    return {
        "risk_on": True,
        "btc_price": btc_price,
        "btc_ma": btc_ma,
        "weights": target,
    }


# ============================================================
# EXCHANGE RULES
# ============================================================

def get_exchange_rules(client):

    response = (
        client.get_exchange_info()
    )

    if "TradePairs" not in response:

        raise RuntimeError(
            "Invalid exchangeInfo response: "
            + str(response)
        )

    rules = {}

    for asset, info in ASSETS.items():

        pair = info["roostoo"]

        if pair not in response["TradePairs"]:

            raise RuntimeError(
                f"{pair} missing from exchangeInfo."
            )

        pair_info = (
            response["TradePairs"][pair]
        )

        rules[asset] = {
            "amount_precision":
                int(
                    pair_info[
                        "AmountPrecision"
                    ]
                ),

            "minimum_order":
                float(
                    pair_info[
                        "MiniOrder"
                    ]
                ),
        }

    return rules


# ============================================================
# ROOSTOO PRICE EXTRACTION
# ============================================================

def extract_ticker_price(
    response,
    pair
):

    """
    Safely extract LastPrice from different
    Roostoo ticker response structures.
    """

    if not isinstance(
        response,
        dict
    ):

        raise RuntimeError(
            f"Unexpected ticker response "
            f"for {pair}: {response}"
        )

    # --------------------------------------------------------
    # STRUCTURE 1
    #
    # {
    #   "Success": True,
    #   "BTC/USD": {
    #       "LastPrice": 86000
    #   }
    # }
    # --------------------------------------------------------

    if pair in response:

        ticker = response[pair]

        if (
            isinstance(
                ticker,
                dict
            )
            and
            "LastPrice" in ticker
        ):

            return float(
                ticker["LastPrice"]
            )

    # --------------------------------------------------------
    # STRUCTURE 2
    #
    # {
    #   "LastPrice": 86000
    # }
    # --------------------------------------------------------

    if "LastPrice" in response:

        return float(
            response["LastPrice"]
        )

    # --------------------------------------------------------
    # STRUCTURE 3
    #
    # {
    #   "Data": {
    #       "BTC/USD": {
    #           "LastPrice": ...
    #       }
    #   }
    # }
    # --------------------------------------------------------

    data = response.get(
        "Data"
    )

    if isinstance(
        data,
        dict
    ):

        if pair in data:

            ticker = data[pair]

            if (
                isinstance(
                    ticker,
                    dict
                )
                and
                "LastPrice" in ticker
            ):

                return float(
                    ticker[
                        "LastPrice"
                    ]
                )

        if "LastPrice" in data:

            return float(
                data["LastPrice"]
            )

    # --------------------------------------------------------
    # UNKNOWN STRUCTURE
    # --------------------------------------------------------

    raise RuntimeError(
        f"Could not extract price for {pair}. "
        f"Full response: {response}"
    )


# ============================================================
# ROOSTOO PRICES
# ============================================================

def get_roostoo_prices(client):

    prices = {}

    for asset, info in ASSETS.items():

        pair = (
            info["roostoo"]
        )

        response = (
            client.get_ticker(
                pair
            )
        )

        price = (
            extract_ticker_price(
                response,
                pair
            )
        )

        if price <= 0:

            raise RuntimeError(
                f"Invalid {pair} price: "
                f"{price}"
            )

        prices[asset] = price

        logging.info(
            "Roostoo %s price: %.8f",
            asset,
            price
        )

        time.sleep(0.1)

    return prices


# ============================================================
# CURRENT PORTFOLIO
# ============================================================

def get_portfolio(
    client,
    market_prices
):

    response = (
        client.get_balance()
    )

    if not response.get(
        "Success",
        False
    ):

        raise RuntimeError(
            "Balance request failed: "
            + str(response)
        )

    wallet = (
        response["SpotWallet"]
    )

    usd = float(
        wallet
        .get("USD", {})
        .get("Free", 0)
    )

    quantities = {}
    asset_values = {}

    total_value = usd

    for asset in ASSETS:

        quantity = float(
            wallet
            .get(asset, {})
            .get("Free", 0)
        )

        value = (
            quantity
            *
            market_prices[asset]
        )

        quantities[asset] = (
            quantity
        )

        asset_values[asset] = (
            value
        )

        total_value += value

    weights = {}

    for asset in ASSETS:

        if total_value > 0:

            weights[asset] = (
                asset_values[asset]
                /
                total_value
            )

        else:

            weights[asset] = 0.0

    return {
        "usd": usd,
        "quantities": quantities,
        "asset_values": asset_values,
        "weights": weights,
        "total_value": total_value,
    }


# ============================================================
# STRATEGY REBALANCE DECISION
# ============================================================

def should_rebalance(
    target_weights,
    portfolio,
    state
):

    now = datetime.now(
        timezone.utc
    )

    last_core = (
        state.get(
            "last_core_rebalance"
        )
    )

    # First execution.
    if last_core is None:

        return (
            True,
            "INITIAL_CORE_REBALANCE"
        )

    last_core_dt = (
        datetime.fromisoformat(
            last_core
        )
    )

    elapsed = (
        now
        -
        last_core_dt
    )

    # Weekly alpha rebalance.
    if elapsed.total_seconds() >= (
        CORE_REBALANCE_DAYS
        *
        24
        *
        3600
    ):

        return (
            True,
            "WEEKLY_CORE_REBALANCE"
        )

    # --------------------------------------------------------
    # DAILY RISK LAYER
    # --------------------------------------------------------

    maximum_drift = 0.0

    for asset in ASSETS:

        actual = (
            portfolio[
                "weights"
            ][asset]
        )

        target = float(
            target_weights[
                asset
            ]
        )

        drift = abs(
            actual
            -
            target
        )

        maximum_drift = max(
            maximum_drift,
            drift
        )

    if maximum_drift >= (
        DRIFT_THRESHOLD
    ):

        return (
            True,
            f"RISK_DRIFT_{maximum_drift:.4f}"
        )

    return (
        False,
        f"NO_REBALANCE_MAX_DRIFT_"
        f"{maximum_drift:.4f}"
    )


# ============================================================
# EXECUTION COOLDOWN
# ============================================================

def cooldown_ok(state):

    last_execution = (
        state.get(
            "last_execution_time"
        )
    )

    if last_execution is None:

        return True

    last_dt = (
        datetime.fromisoformat(
            last_execution
        )
    )

    elapsed_hours = (
        datetime.now(
            timezone.utc
        )
        -
        last_dt
    ).total_seconds() / 3600

    return (
        elapsed_hours
        >=
        MIN_HOURS_BETWEEN_EXECUTIONS
    )


# ============================================================
# ROUND QUANTITY DOWN
# ============================================================

def round_down(
    value,
    precision
):

    multiplier = (
        10 ** precision
    )

    return (
        math.floor(
            value
            *
            multiplier
            +
            1e-12
        )
        /
        multiplier
    )


# ============================================================
# BUILD ORDER PLAN
# ============================================================

def build_orders(
    target_weights,
    portfolio,
    market_prices,
    rules
):

    total_value = (
        portfolio[
            "total_value"
        ]
    )

    orders = []

    for asset, info in ASSETS.items():

        target_value = (
            total_value
            *
            float(
                target_weights[asset]
            )
        )

        current_value = (
            portfolio[
                "asset_values"
            ][asset]
        )

        difference = (
            target_value
            -
            current_value
        )

        minimum = max(
            MIN_TRADE_USD,
            rules[
                asset
            ][
                "minimum_order"
            ]
        )

        if abs(
            difference
        ) < minimum:

            continue

        if difference > 0:

            side = "BUY"

        else:

            side = "SELL"

        quantity = (
            abs(difference)
            /
            market_prices[asset]
        )

        # Never sell more than we own.
        if side == "SELL":

            quantity = min(
                quantity,
                portfolio[
                    "quantities"
                ][asset]
            )

        quantity = round_down(
            quantity,
            rules[
                asset
            ][
                "amount_precision"
            ]
        )

        estimated_value = (
            quantity
            *
            market_prices[asset]
        )

        if quantity <= 0:
            continue

        if estimated_value < minimum:
            continue

        orders.append({
            "asset": asset,
            "pair": info["roostoo"],
            "side": side,
            "quantity": quantity,
            "estimated_value":
                estimated_value,
        })

    # SELL before BUY.
    orders.sort(
        key=lambda order:
        0
        if order["side"] == "SELL"
        else 1
    )

    return orders


# ============================================================
# TRADE LOG
# ============================================================

def log_trade(
    reason,
    order,
    response
):

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    row = {
        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "reason":
            reason,

        "pair":
            order["pair"],

        "side":
            order["side"],

        "quantity":
            order["quantity"],

        "estimated_value":
            order[
                "estimated_value"
            ],

        "success":
            response.get(
                "Success",
                False
            ),

        "response":
            json.dumps(
                response
            ),
    }

    df = pd.DataFrame(
        [row]
    )

    df.to_csv(
        TRADE_LOG,
        mode="a",
        header=(
            not TRADE_LOG.exists()
        ),
        index=False
    )


# ============================================================
# EXECUTE
# ============================================================

def execute_orders(
    client,
    orders,
    reason
):

    if not orders:

        logging.info(
            "No executable orders."
        )

        return False

    # --------------------------------------------------------
    # DRY RUN
    # --------------------------------------------------------

    if not LIVE_TRADING:

        logging.info(
            "DRY RUN — no orders sent."
        )

        for order in orders:

            logging.info(
                "PLAN %s %s "
                "%.8f ~$%.2f",
                order["side"],
                order["pair"],
                order["quantity"],
                order[
                    "estimated_value"
                ]
            )

        return False

    # --------------------------------------------------------
    # LIVE EXECUTION
    # --------------------------------------------------------

    executed_any = False

    for order in orders:

        # Refresh available USD before
        # every BUY.
        if order["side"] == "BUY":

            balance = (
                client.get_balance()
            )

            available_usd = float(
                balance[
                    "SpotWallet"
                ][
                    "USD"
                ][
                    "Free"
                ]
            )

            # 0.1% taker fee plus
            # extra execution cushion.
            required_usd = (
                order[
                    "estimated_value"
                ]
                *
                1.002
            )

            if required_usd > available_usd:

                logging.warning(
                    "Skipping %s: "
                    "requires $%.2f, "
                    "available $%.2f",
                    order["pair"],
                    required_usd,
                    available_usd
                )

                continue

        logging.info(
            "EXECUTING %s %s %.8f",
            order["side"],
            order["pair"],
            order["quantity"]
        )

        response = (
            client.place_order(
                pair=order["pair"],
                side=order["side"],
                quantity=order[
                    "quantity"
                ],
                order_type="MARKET"
            )
        )

        log_trade(
            reason,
            order,
            response
        )

        if response.get(
            "Success",
            False
        ):

            executed_any = True

            logging.info(
                "Order successful."
            )

        else:

            logging.error(
                "Order failed: %s",
                response
            )

        time.sleep(1)

    return executed_any


# ============================================================
# MAIN CYCLE
# ============================================================

def run():

    setup_logging()

    logging.info(
        "=" * 60
    )

    logging.info(
        "ROOSTOO PRODUCTION BOT START"
    )

    logging.info(
        "LIVE_TRADING=%s",
        LIVE_TRADING
    )

    client = (
        RoostooClient()
    )

    state = (
        load_state()
    )

    # --------------------------------------------------------
    # STRATEGY
    # --------------------------------------------------------

    prices = (
        load_market_data()
    )

    signal = (
        calculate_target(
            prices
        )
    )

    target = (
        signal["weights"]
    )

    logging.info(
        "BTC price: %.2f",
        signal["btc_price"]
    )

    logging.info(
        "BTC 200d MA: %.2f",
        signal["btc_ma"]
    )

    logging.info(
        "Regime: %s",
        (
            "RISK-ON"
            if signal["risk_on"]
            else
            "RISK-OFF"
        )
    )

    for asset in ASSETS:

        logging.info(
            "Target %-3s: %.2f%%",
            asset,
            target[asset] * 100
        )

    target_cash = (
        1
        -
        target.sum()
    )

    logging.info(
        "Target cash: %.2f%%",
        target_cash * 100
    )

    # --------------------------------------------------------
    # ROOSTOO MARKET
    # --------------------------------------------------------

    market_prices = (
        get_roostoo_prices(
            client
        )
    )

    portfolio = (
        get_portfolio(
            client,
            market_prices
        )
    )

    logging.info(
        "Portfolio value: $%.2f",
        portfolio[
            "total_value"
        ]
    )

    logging.info(
        "USD cash: $%.2f",
        portfolio["usd"]
    )

    for asset in ASSETS:

        logging.info(
            "Current %-3s: %.2f%%",
            asset,
            portfolio[
                "weights"
            ][asset]
            *
            100
        )

    # --------------------------------------------------------
    # REBALANCE DECISION
    # --------------------------------------------------------

    rebalance, reason = (
        should_rebalance(
            target,
            portfolio,
            state
        )
    )

    logging.info(
        "Decision: %s",
        reason
    )

    if not rebalance:

        logging.info(
            "No strategy-driven "
            "rebalance required."
        )

        return

    if not cooldown_ok(
        state
    ):

        logging.info(
            "Execution cooldown active."
        )

        return

    # --------------------------------------------------------
    # ORDER PLAN
    # --------------------------------------------------------

    rules = (
        get_exchange_rules(
            client
        )
    )

    orders = (
        build_orders(
            target,
            portfolio,
            market_prices,
            rules
        )
    )

    if not orders:

        logging.info(
            "No executable portfolio "
            "adjustments required."
        )

        return

    logging.info(
        "Execution reason: %s",
        reason
    )

    for order in orders:

        logging.info(
            "PLAN %s %s "
            "%.8f ~$%.2f",
            order["side"],
            order["pair"],
            order["quantity"],
            order[
                "estimated_value"
            ]
        )

    # --------------------------------------------------------
    # EXECUTION
    # --------------------------------------------------------

    executed = (
        execute_orders(
            client,
            orders,
            reason
        )
    )

    # Only modify state after REAL execution.
    if executed:

        now = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        state[
            "last_execution_time"
        ] = now

        if reason in {
            "INITIAL_CORE_REBALANCE",
            "WEEKLY_CORE_REBALANCE",
        }:

            state[
                "last_core_rebalance"
            ] = now

        save_state(
            state
        )

        logging.info(
            "Execution state saved."
        )

    logging.info(
        "BOT CYCLE COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        run()

    except Exception:

        logging.exception(
            "BOT CYCLE FAILED"
        )

        raise
