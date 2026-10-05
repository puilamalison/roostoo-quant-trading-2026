import json
import logging
import math
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from data.market_data import get_historical_ohlcv
from execution.roostoo_client import RoostooClient


# ============================================================
# CONFIG
# ============================================================

# IMPORTANT:
# Keep False until the final competition-account dry run passes.
LIVE_TRADING = False

ASSETS = {
    "BTC": {"binance": "BTCUSDT", "roostoo": "BTC/USD"},
    "ETH": {"binance": "ETHUSDT", "roostoo": "ETH/USD"},
    "SOL": {"binance": "SOLUSDT", "roostoo": "SOL/USD"},
    "BNB": {"binance": "BNBUSDT", "roostoo": "BNB/USD"},
    "XRP": {"binance": "XRPUSDT", "roostoo": "XRP/USD"},
}

HISTORY_DAYS = 230

# 1-hour candles
MOM_30 = 24 * 30
MOM_90 = 24 * 90
VOL_30 = 24 * 30
MA_30 = 24 * 30
BTC_MA_200 = 24 * 200

TOP_N = 2

# 98.8% invested -> 1.2% nominal cash buffer.
INVESTED_FRACTION = 0.988

# Daily strategy-driven rebalance threshold.
# 0.002 = 0.20 percentage points of portfolio weight.
#
# This does NOT manufacture a trade.
# A trade only occurs when the strategy target and actual portfolio
# differ sufficiently to generate an executable order.
DAILY_WEIGHT_DRIFT = 0.002

MIN_TRADE_USD = 5.0

# Main strategy evaluation once per UTC day.
DAILY_RUN_HOUR = 0
DAILY_RUN_MINUTE = 15

STATE_DIR = Path("state")
LOG_DIR = Path("logs")

STATE_FILE = STATE_DIR / "production_state.json"

TRADE_LOG = LOG_DIR / "production_trades.csv"
DECISION_LOG = LOG_DIR / "production_decisions.csv"
ACTIVITY_LOG = LOG_DIR / "active_days.csv"
BOT_LOG = LOG_DIR / "production_bot.log"


# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(BOT_LOG),
            logging.StreamHandler(),
        ],
    )


# ============================================================
# STATE
# ============================================================

def default_state():
    return {
        "last_daily_cycle": None,
        "last_regime": None,
    }


def load_state():
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    if not STATE_FILE.exists():
        return default_state()

    with open(STATE_FILE, "r") as f:
        state = json.load(f)

    for key, value in default_state().items():
        if key not in state:
            state[key] = value

    return state


def save_state(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    temporary = STATE_FILE.with_suffix(".tmp")

    with open(temporary, "w") as f:
        json.dump(state, f, indent=2)

    temporary.replace(STATE_FILE)


# ============================================================
# MARKET DATA
# ============================================================

def load_market_data():
    prices = pd.DataFrame()

    for asset, info in ASSETS.items():
        logging.info("Downloading %s data...", asset)

        df = get_historical_ohlcv(
            symbol=info["binance"],
            interval="1h",
            days=HISTORY_DAYS,
        )

        if len(df) < 2:
            raise RuntimeError(
                f"Insufficient data for {asset}."
            )

        # The newest Binance candle may still be forming.
        # Never use it to generate a signal.
        df = df.iloc[:-1].copy()

        df = df.set_index("open_time")

        prices[asset] = df["close"]

    prices = prices.dropna().sort_index()

    if len(prices) < BTC_MA_200:
        raise RuntimeError(
            "Insufficient aligned history."
        )

    logging.info(
        "Latest closed signal candle: %s",
        prices.index[-1],
    )

    return prices


# ============================================================
# STRATEGY
# ============================================================

def calculate_target(prices):
    returns = prices.pct_change()

    mom_30 = prices.pct_change(MOM_30)
    mom_90 = prices.pct_change(MOM_90)

    vol_30 = (
        returns
        .rolling(VOL_30)
        .std()
        * np.sqrt(24 * 365)
    )

    ma_30 = prices.rolling(MA_30).mean()

    btc_ma = (
        prices["BTC"]
        .rolling(BTC_MA_200)
        .mean()
    )

    btc_price = float(
        prices["BTC"].iloc[-1]
    )

    btc_200 = float(
        btc_ma.iloc[-1]
    )

    target = pd.Series(
        0.0,
        index=prices.columns,
    )

    risk_on = btc_price > btc_200

    # Regime filter:
    # BTC below 200-day MA -> 100% cash.
    if not risk_on:
        return {
            "risk_on": False,
            "btc_price": btc_price,
            "btc_ma": btc_200,
            "weights": target,
        }

    score = (
        0.3 * mom_30.iloc[-1]
        + 0.7 * mom_90.iloc[-1]
    )

    current_vol = vol_30.iloc[-1]

    trend = (
        prices.iloc[-1]
        > ma_30.iloc[-1]
    )

    eligible = (
        (score > 0)
        & trend
        & current_vol.notna()
        & (current_vol > 0)
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
            / current_vol[selected]
        )

        inverse_vol = (
            inverse_vol
            / inverse_vol.sum()
        )

        target[selected] = (
            inverse_vol
            * INVESTED_FRACTION
        )

    return {
        "risk_on": True,
        "btc_price": btc_price,
        "btc_ma": btc_200,
        "weights": target,
    }


# ============================================================
# API RETRY
# ============================================================

def api_call(
    function,
    *args,
    attempts=3,
    **kwargs,
):
    last_error = None

    for attempt in range(
        1,
        attempts + 1,
    ):
        try:
            return function(
                *args,
                **kwargs,
            )

        except Exception as error:
            last_error = error

            logging.warning(
                "API attempt %d/%d failed: %s",
                attempt,
                attempts,
                error,
            )

            if attempt < attempts:
                time.sleep(2 * attempt)

    raise RuntimeError(
        "API call failed after "
        f"{attempts} attempts: "
        f"{last_error}"
    )


# ============================================================
# TICKER PARSING
# ============================================================

def extract_ticker_price(
    response,
    pair,
):
    if not isinstance(response, dict):
        raise RuntimeError(
            f"Unexpected ticker response "
            f"for {pair}: {response}"
        )

    # Possible format:
    # {"BTC/USD": {"LastPrice": ...}}
    if pair in response:
        ticker = response[pair]

        if (
            isinstance(ticker, dict)
            and "LastPrice" in ticker
        ):
            return float(
                ticker["LastPrice"]
            )

    # Possible format:
    # {"LastPrice": ...}
    if "LastPrice" in response:
        return float(
            response["LastPrice"]
        )

    # Possible format:
    # {"Data": {...}}
    data = response.get("Data")

    if isinstance(data, dict):
        if pair in data:
            ticker = data[pair]

            if (
                isinstance(ticker, dict)
                and "LastPrice" in ticker
            ):
                return float(
                    ticker["LastPrice"]
                )

        if "LastPrice" in data:
            return float(
                data["LastPrice"]
            )

    raise RuntimeError(
        f"Could not extract price "
        f"for {pair}. "
        f"Response: {response}"
    )


# ============================================================
# ROOSTOO PRICES
# ============================================================

def get_roostoo_prices(client):
    prices = {}

    for asset, info in ASSETS.items():
        pair = info["roostoo"]

        response = api_call(
            client.get_ticker,
            pair,
        )

        price = extract_ticker_price(
            response,
            pair,
        )

        if (
            not math.isfinite(price)
            or price <= 0
        ):
            raise RuntimeError(
                f"Invalid price for "
                f"{pair}: {price}"
            )

        prices[asset] = price

        logging.info(
            "Roostoo %-3s: %.8f",
            asset,
            price,
        )

        # Stay comfortably below API rate limits.
        time.sleep(0.15)

    return prices


# ============================================================
# EXCHANGE RULES
# ============================================================

def get_exchange_rules(client):
    response = api_call(
        client.get_exchange_info
    )

    if "TradePairs" not in response:
        raise RuntimeError(
            "Invalid exchange info: "
            + str(response)
        )

    rules = {}

    for asset, info in ASSETS.items():
        pair = info["roostoo"]

        if pair not in response["TradePairs"]:
            raise RuntimeError(
                f"{pair} missing from "
                f"exchangeInfo."
            )

        pair_info = (
            response["TradePairs"][pair]
        )

        rules[asset] = {
            "amount_precision": int(
                pair_info["AmountPrecision"]
            ),
            "minimum_order": float(
                pair_info["MiniOrder"]
            ),
        }

    return rules


# ============================================================
# PORTFOLIO
# ============================================================

def get_portfolio(
    client,
    market_prices,
):
    response = api_call(
        client.get_balance
    )

    if not response.get(
        "Success",
        False,
    ):
        raise RuntimeError(
            "Balance request failed: "
            + str(response)
        )

    wallet = response["SpotWallet"]

    usd = float(
        wallet
        .get("USD", {})
        .get("Free", 0)
    )

    quantities = {}
    asset_values = {}

    total = usd

    for asset in ASSETS:
        quantity = float(
            wallet
            .get(asset, {})
            .get("Free", 0)
        )

        value = (
            quantity
            * market_prices[asset]
        )

        quantities[asset] = quantity
        asset_values[asset] = value

        total += value

    weights = {}

    for asset in ASSETS:
        if total > 0:
            weights[asset] = (
                asset_values[asset]
                / total
            )
        else:
            weights[asset] = 0.0

    return {
        "usd": usd,
        "quantities": quantities,
        "asset_values": asset_values,
        "weights": weights,
        "total_value": total,
    }


# ============================================================
# ROUNDING
# ============================================================

def round_down(
    value,
    precision,
):
    multiplier = 10 ** precision

    return (
        math.floor(
            value
            * multiplier
            + 1e-12
        )
        / multiplier
    )


# ============================================================
# REBALANCE DECISION
# ============================================================

def calculate_maximum_drift(
    target,
    portfolio,
):
    maximum_drift = 0.0

    for asset in ASSETS:
        drift = abs(
            float(target[asset])
            - portfolio["weights"][asset]
        )

        maximum_drift = max(
            maximum_drift,
            drift,
        )

    return maximum_drift


def needs_daily_rebalance(
    target,
    portfolio,
):
    maximum_drift = (
        calculate_maximum_drift(
            target,
            portfolio,
        )
    )

    return (
        maximum_drift
        >= DAILY_WEIGHT_DRIFT,
        maximum_drift,
    )


# ============================================================
# BUILD ORDERS
# ============================================================

def build_orders(
    target,
    portfolio,
    prices,
    rules,
):
    total = portfolio["total_value"]

    orders = []

    for asset, info in ASSETS.items():
        target_value = (
            total
            * float(target[asset])
        )

        current_value = (
            portfolio[
                "asset_values"
            ][asset]
        )

        difference = (
            target_value
            - current_value
        )

        minimum = max(
            MIN_TRADE_USD,
            rules[
                asset
            ]["minimum_order"],
        )

        if abs(difference) < minimum:
            continue

        side = (
            "BUY"
            if difference > 0
            else "SELL"
        )

        quantity = (
            abs(difference)
            / prices[asset]
        )

        if side == "SELL":
            quantity = min(
                quantity,
                portfolio[
                    "quantities"
                ][asset],
            )

        quantity = round_down(
            quantity,
            rules[
                asset
            ][
                "amount_precision"
            ],
        )

        estimated_value = (
            quantity
            * prices[asset]
        )

        if quantity <= 0:
            continue

        if estimated_value < minimum:
            continue

        orders.append(
            {
                "asset": asset,
                "pair": info["roostoo"],
                "side": side,
                "quantity": quantity,
                "estimated_value":
                    estimated_value,
            }
        )

    # Always sell before buying.
    orders.sort(
        key=lambda order: (
            0
            if order["side"] == "SELL"
            else 1
        )
    )

    return orders


# ============================================================
# DECISION LOG
# ============================================================

def log_decision(
    reason,
    signal,
    portfolio,
    orders,
    maximum_drift,
):
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    row = {
        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "reason":
            reason,

        "risk_on":
            signal["risk_on"],

        "btc_price":
            signal["btc_price"],

        "btc_200d_ma":
            signal["btc_ma"],

        "portfolio_value":
            portfolio[
                "total_value"
            ],

        "cash":
            portfolio["usd"],

        "maximum_weight_drift":
            maximum_drift,

        "target_weights":
            json.dumps(
                signal[
                    "weights"
                ].to_dict()
            ),

        "current_weights":
            json.dumps(
                portfolio[
                    "weights"
                ]
            ),

        "planned_orders":
            json.dumps(orders),
    }

    pd.DataFrame(
        [row]
    ).to_csv(
        DECISION_LOG,
        mode="a",
        header=(
            not DECISION_LOG.exists()
        ),
        index=False,
    )


# ============================================================
# TRADE LOG
# ============================================================

def log_trade(
    reason,
    order,
    response,
    verification,
):
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
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
                False,
            ),

        "order_response":
            json.dumps(response),

        "query_verification":
            json.dumps(
                verification
            ),
    }

    pd.DataFrame(
        [row]
    ).to_csv(
        TRADE_LOG,
        mode="a",
        header=(
            not TRADE_LOG.exists()
        ),
        index=False,
    )


# ============================================================
# ACTIVE DAY LOG
# ============================================================

def record_active_day(
    date_string,
    trade_count,
):
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    row = {
        "date": date_string,
        "trade_count": trade_count,
        "active": trade_count >= 1,
    }

    pd.DataFrame(
        [row]
    ).to_csv(
        ACTIVITY_LOG,
        mode="a",
        header=(
            not ACTIVITY_LOG.exists()
        ),
        index=False,
    )


# ============================================================
# ORDER VERIFICATION
# ============================================================

def extract_order_id(response):
    if not response.get(
        "Success",
        False,
    ):
        return None

    detail = response.get(
        "OrderDetail",
        {},
    )

    return detail.get("OrderID")


def verify_order(
    client,
    response,
):
    order_id = extract_order_id(
        response
    )

    if order_id is None:
        raise RuntimeError(
            "Successful order response "
            "did not contain OrderID."
        )

    # Give Roostoo a moment before
    # querying the submitted order.
    time.sleep(0.5)

    verification = api_call(
        client.query_order,
        order_id=order_id,
    )

    if not verification.get(
        "Success",
        False,
    ):
        raise RuntimeError(
            "query_order verification "
            f"failed for {order_id}: "
            f"{verification}"
        )

    return verification


# ============================================================
# EXECUTE ONE ORDER
# ============================================================

def execute_single_order(
    client,
    order,
    reason,
):
    logging.info(
        "SUBMIT %s %s %.10f "
        "(estimated $%.2f)",
        order["side"],
        order["pair"],
        order["quantity"],
        order["estimated_value"],
    )

    response = api_call(
        client.place_order,
        pair=order["pair"],
        side=order["side"],
        quantity=order["quantity"],
        order_type="MARKET",
    )

    if not response.get(
        "Success",
        False,
    ):
        log_trade(
            reason,
            order,
            response,
            {},
        )

        raise RuntimeError(
            f"Order failed: {response}"
        )

    verification = verify_order(
        client,
        response,
    )

    log_trade(
        reason,
        order,
        response,
        verification,
    )

    order_id = extract_order_id(
        response
    )

    logging.info(
        "VERIFIED order %s",
        order_id,
    )

    return response


# ============================================================
# EXECUTE REBALANCE
# ============================================================

def execute_rebalance(
    client,
    signal,
    rules,
    reason,
):
    trade_count = 0

    # ----------------------------
    # STEP 1: current prices
    # ----------------------------

    prices = get_roostoo_prices(
        client
    )

    portfolio = get_portfolio(
        client,
        prices,
    )

    # ----------------------------
    # STEP 2: build initial orders
    # ----------------------------

    orders = build_orders(
        signal["weights"],
        portfolio,
        prices,
        rules,
    )

    sell_orders = [
        order
        for order in orders
        if order["side"] == "SELL"
    ]

    # ----------------------------
    # STEP 3: execute sells first
    # ----------------------------

    for order in sell_orders:
        execute_single_order(
            client,
            order,
            reason,
        )

        trade_count += 1

        time.sleep(0.5)

    # ----------------------------
    # STEP 4:
    # refresh after sells
    # ----------------------------

    prices = get_roostoo_prices(
        client
    )

    portfolio = get_portfolio(
        client,
        prices,
    )

    # Recalculate remaining orders
    # from actual post-sell portfolio.
    remaining_orders = build_orders(
        signal["weights"],
        portfolio,
        prices,
        rules,
    )

    buy_orders = [
        order
        for order in remaining_orders
        if order["side"] == "BUY"
    ]

    # ----------------------------
    # STEP 5: execute buys
    # ----------------------------

    for order in buy_orders:
        # Refresh cash immediately
        # before each buy.
        latest_prices = (
            get_roostoo_prices(
                client
            )
        )

        latest_portfolio = (
            get_portfolio(
                client,
                latest_prices,
            )
        )

        current_price = (
            latest_prices[
                order["asset"]
            ]
        )

        # Recalculate the required
        # quantity using the latest
        # portfolio value and price.
        target_value = (
            latest_portfolio[
                "total_value"
            ]
            * float(
                signal[
                    "weights"
                ][
                    order["asset"]
                ]
            )
        )

        current_value = (
            latest_portfolio[
                "asset_values"
            ][
                order["asset"]
            ]
        )

        required_value = (
            target_value
            - current_value
        )

        if required_value <= 0:
            continue

        minimum = max(
            MIN_TRADE_USD,
            rules[
                order["asset"]
            ]["minimum_order"],
        )

        if required_value < minimum:
            continue

        # 0.2% cash cushion.
        # This covers the 0.1% taker fee
        # plus small price movement.
        maximum_spend = (
            latest_portfolio["usd"]
            / 1.002
        )

        spend = min(
            required_value,
            maximum_spend,
        )

        if spend < minimum:
            logging.warning(
                "Skipping %s buy: "
                "insufficient free USD.",
                order["asset"],
            )
            continue

        quantity = round_down(
            spend / current_price,
            rules[
                order["asset"]
            ]["amount_precision"],
        )

        estimated_value = (
            quantity
            * current_price
        )

        if (
            quantity <= 0
            or estimated_value < minimum
        ):
            continue

        refreshed_order = {
            "asset":
                order["asset"],

            "pair":
                order["pair"],

            "side":
                "BUY",

            "quantity":
                quantity,

            "estimated_value":
                estimated_value,
        }

        execute_single_order(
            client,
            refreshed_order,
            reason,
        )

        trade_count += 1

        time.sleep(0.5)

    return trade_count


# ============================================================
# DRY-RUN DISPLAY
# ============================================================

def display_dry_run_orders(orders):
    if not orders:
        print(
            "No executable orders "
            "generated."
        )
        return

    print("\nDRY RUN - PLANNED ORDERS")

    for order in orders:
        print(
            f"{order['side']:4s} "
            f"{order['pair']:8s} "
            f"{order['quantity']:.10f} "
            f"~${order['estimated_value']:,.2f}"
        )


# ============================================================
# RUN ONE DAILY STRATEGY CYCLE
# ============================================================

def run_cycle(client):
    now = datetime.now(
        timezone.utc
    )

    today = now.date().isoformat()

    state = load_state()

    # Prevent duplicate strategy cycles
    # after a restart on the same UTC day.
    if (
        LIVE_TRADING
        and state[
            "last_daily_cycle"
        ] == today
    ):
        logging.info(
            "Today's strategy cycle "
            "has already completed."
        )

        return True

    # ----------------------------
    # 1. Market data + signal
    # ----------------------------

    prices = load_market_data()

    signal = calculate_target(
        prices
    )

    regime = (
        "RISK_ON"
        if signal["risk_on"]
        else "RISK_OFF"
    )

    print()
    print(
        "BTC:",
        f"{signal['btc_price']:.2f}",
    )

    print(
        "BTC 200d MA:",
        f"{signal['btc_ma']:.2f}",
    )

    print(
        "REGIME:",
        regime,
    )

    print()
    print("TARGET WEIGHTS")

    for asset in ASSETS:
        print(
            f"{asset}: "
            f"{signal['weights'][asset] * 100:.2f}%"
        )

    target_cash = (
        1.0
        - signal["weights"].sum()
    )

    print(
        "Cash:",
        f"{target_cash * 100:.2f}%",
    )

    # ----------------------------
    # 2. Roostoo account state
    # ----------------------------

    roostoo_prices = (
        get_roostoo_prices(
            client
        )
    )

    rules = get_exchange_rules(
        client
    )

    portfolio = get_portfolio(
        client,
        roostoo_prices,
    )

    print()
    print(
        "PORTFOLIO VALUE:",
        f"${portfolio['total_value']:,.2f}",
    )

    print(
        "FREE USD:",
        f"${portfolio['usd']:,.2f}",
    )

    print()
    print("CURRENT WEIGHTS")

    for asset in ASSETS:
        print(
            f"{asset}: "
            f"{portfolio['weights'][asset] * 100:.2f}%"
        )

    # ----------------------------
    # 3. Determine whether the
    # strategy requires trading
    # ----------------------------

    should_rebalance, maximum_drift = (
        needs_daily_rebalance(
            signal["weights"],
            portfolio,
        )
    )

    previous_regime = (
        state.get("last_regime")
    )

    # Risk-off transition gets an
    # explicit reason in the logs.
    if (
        previous_regime == "RISK_ON"
        and not signal["risk_on"]
    ):
        reason = "REGIME_EXIT"

    elif should_rebalance:
        reason = (
            "DAILY_STRATEGY_REBALANCE"
        )

    else:
        reason = (
            "DAILY_NO_REBALANCE"
        )

    orders = build_orders(
        signal["weights"],
        portfolio,
        roostoo_prices,
        rules,
    )

    # If the drift threshold says no
    # rebalance, do not send small
    # incidental orders.
    if not should_rebalance:
        orders = []

    print()
    print("DECISION:", reason)

    print(
        "MAX WEIGHT DRIFT:",
        f"{maximum_drift * 100:.4f}%",
    )

    log_decision(
        reason,
        signal,
        portfolio,
        orders,
        maximum_drift,
    )

    # ----------------------------
    # 4. DRY RUN
    # ----------------------------

    if not LIVE_TRADING:
        display_dry_run_orders(
            orders
        )

        print()
        print(
            "LIVE_TRADING=False"
        )

        print(
            "No orders were sent."
        )

        return True

    # ----------------------------
    # 5. LIVE EXECUTION
    # ----------------------------

    trade_count = 0

    if should_rebalance:
        trade_count = (
            execute_rebalance(
                client,
                signal,
                rules,
                reason,
            )
        )

    record_active_day(
        today,
        trade_count,
    )

    # State is updated only after the
    # cycle completes successfully.
    state[
        "last_daily_cycle"
    ] = today

    state[
        "last_regime"
    ] = regime

    save_state(state)

    logging.info(
        "Cycle complete. "
        "Trades today: %d | "
        "Active trading day: %s",
        trade_count,
        trade_count >= 1,
    )

    return True


# ============================================================
# SCHEDULING
# ============================================================

def scheduled_time_reached(now):
    if now.hour > DAILY_RUN_HOUR:
        return True

    if (
        now.hour == DAILY_RUN_HOUR
        and
        now.minute
        >= DAILY_RUN_MINUTE
    ):
        return True

    return False


def run_forever(client):
    logging.info(
        "Production bot started."
    )

    logging.info(
        "LIVE_TRADING=%s",
        LIVE_TRADING,
    )

    while True:
        now = datetime.now(
            timezone.utc
        )

        today = (
            now.date().isoformat()
        )

        try:
            state = load_state()

            already_completed = (
                state[
                    "last_daily_cycle"
                ]
                == today
            )

            if (
                scheduled_time_reached(
                    now
                )
                and
                not already_completed
            ):
                logging.info(
                    "Starting UTC daily "
                    "strategy cycle."
                )

                run_cycle(client)

        except Exception:
            logging.exception(
                "Strategy cycle failed. "
                "Bot will retry."
            )

        # Check once per minute.
        time.sleep(60)


# ============================================================
# MAIN
# ============================================================

def main():
    setup_logging()

    client = RoostooClient()

    if LIVE_TRADING:
        print()
        print(
            "=================================="
        )
        print(
            " LIVE COMPETITION TRADING ENABLED "
        )
        print(
            "=================================="
        )
        print()

        run_forever(client)

    else:
        print()
        print(
            "=================================="
        )
        print(
            " DRY RUN - NO ORDERS WILL BE SENT "
        )
        print(
            "=================================="
        )
        print()

        run_cycle(client)


if __name__ == "__main__":
    main()
