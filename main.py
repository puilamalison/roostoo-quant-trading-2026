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

# Keep False for the first TEST-account validation.
LIVE_TRADING = False

ASSETS = {
    "BTC": {"binance": "BTCUSDT", "roostoo": "BTC/USD"},
    "ETH": {"binance": "ETHUSDT", "roostoo": "ETH/USD"},
    "SOL": {"binance": "SOLUSDT", "roostoo": "SOL/USD"},
    "BNB": {"binance": "BNBUSDT", "roostoo": "BNB/USD"},
    "XRP": {"binance": "XRPUSDT", "roostoo": "XRP/USD"},
}

HISTORY_DAYS = 230

MOM_30 = 24 * 30
MOM_90 = 24 * 90
VOL_30 = 24 * 30
MA_30 = 24 * 30
BTC_MA_200 = 24 * 200

TOP_N = 2

# 98.8% invested = 1.2% cash reserve.
INVESTED_FRACTION = 0.988

MIN_TRADE_USD = 5.0

# Deterministic core schedule:
# Monday 00:00 UTC.
CORE_WEEKDAY = 0

# Daily bot evaluation time:
# 00:15 UTC.
DAILY_RUN_HOUR = 0
DAILY_RUN_MINUTE = 15

# Risk exits can occur outside the weekly rebalance.
#
# If BTC regime changes from risk-on to risk-off,
# liquidate the strategy portfolio.
#
# This is a genuine risk-control action rather than
# target chasing.
ENABLE_REGIME_EXIT = True

STATE_DIR = Path("state")
LOG_DIR = Path("logs")

STATE_FILE = STATE_DIR / "production_state.json"
TRADE_LOG = LOG_DIR / "production_trades.csv"
DECISION_LOG = LOG_DIR / "production_decisions.csv"
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
        "last_core_week": None,
        "last_regime": None,
    }


def load_state():

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    if not STATE_FILE.exists():
        return default_state()

    try:

        with open(STATE_FILE, "r") as f:
            state = json.load(f)

        for key, value in default_state().items():

            if key not in state:
                state[key] = value

        return state

    except Exception:

        logging.exception(
            "Could not read state file."
        )

        raise


def save_state(state):

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    temporary = STATE_FILE.with_suffix(".tmp")

    with open(temporary, "w") as f:

        json.dump(
            state,
            f,
            indent=2
        )

    temporary.replace(STATE_FILE)


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
            days=HISTORY_DAYS,
        )

        if len(df) < 2:

            raise RuntimeError(
                f"Insufficient data for {asset}."
            )

        # The final Binance candle may still be open.
        # Never generate signals from an incomplete candle.
        df = df.iloc[:-1].copy()

        df = df.set_index("open_time")

        prices[asset] = df["close"]

    prices = (
        prices
        .dropna()
        .sort_index()
    )

    if len(prices) < BTC_MA_200:

        raise RuntimeError(
            "Insufficient aligned market history."
        )

    logging.info(
        "Latest closed signal candle: %s",
        prices.index[-1],
    )

    return prices


# ============================================================
# FROZEN V1 ALPHA STRATEGY
# ============================================================

def calculate_target(prices):

    returns = prices.pct_change()

    mom_30 = prices.pct_change(MOM_30)
    mom_90 = prices.pct_change(MOM_90)

    vol_30 = (
        returns
        .rolling(VOL_30)
        .std()
        *
        np.sqrt(24 * 365)
    )

    ma_30 = (
        prices
        .rolling(MA_30)
        .mean()
    )

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

    if not risk_on:

        return {
            "risk_on": False,
            "btc_price": btc_price,
            "btc_ma": btc_200,
            "weights": target,
        }

    score = (
        0.3 * mom_30.iloc[-1]
        +
        0.7 * mom_90.iloc[-1]
    )

    current_vol = vol_30.iloc[-1]

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
        "btc_ma": btc_200,
        "weights": target,
    }


# ============================================================
# API RETRY WRAPPER
# ============================================================

def api_call(
    function,
    *args,
    attempts=3,
    **kwargs,
):

    last_error = None

    for attempt in range(1, attempts + 1):

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
        f"API call failed after "
        f"{attempts} attempts: "
        f"{last_error}"
    )


# ============================================================
# TICKER EXTRACTION
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

    if pair in response:

        ticker = response[pair]

        if (
            isinstance(ticker, dict)
            and
            "LastPrice" in ticker
        ):

            return float(
                ticker["LastPrice"]
            )

    if "LastPrice" in response:

        return float(
            response["LastPrice"]
        )

    data = response.get("Data")

    if isinstance(data, dict):

        if pair in data:

            ticker = data[pair]

            if (
                isinstance(ticker, dict)
                and
                "LastPrice" in ticker
            ):

                return float(
                    ticker["LastPrice"]
                )

        if "LastPrice" in data:

            return float(
                data["LastPrice"]
            )

    raise RuntimeError(
        f"Could not extract {pair} price. "
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

        if not math.isfinite(price) or price <= 0:

            raise RuntimeError(
                f"Invalid price for {pair}: "
                f"{price}"
            )

        prices[asset] = price

        logging.info(
            "Roostoo %-3s: %.8f",
            asset,
            price,
        )

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
            "Invalid exchangeInfo response: "
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
                pair_info[
                    "AmountPrecision"
                ]
            ),
            "minimum_order": float(
                pair_info[
                    "MiniOrder"
                ]
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
            *
            market_prices[asset]
        )

        quantities[asset] = quantity
        asset_values[asset] = value

        total += value

    weights = {}

    for asset in ASSETS:

        if total > 0:

            weights[asset] = (
                asset_values[asset]
                /
                total
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
# QUANTITY ROUNDING
# ============================================================

def round_down(
    value,
    precision,
):

    multiplier = 10 ** precision

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
# ORDER PLAN
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
            *
            float(target[asset])
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
            rules[asset][
                "minimum_order"
            ],
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
            /
            prices[asset]
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
            rules[asset][
                "amount_precision"
            ],
        )

        value = (
            quantity
            *
            prices[asset]
        )

        if quantity <= 0:
            continue

        if value < minimum:
            continue

        orders.append({
            "asset": asset,
            "pair": info["roostoo"],
            "side": side,
            "quantity": quantity,
            "estimated_value": value,
        })

    orders.sort(
        key=lambda order:
        0
        if order["side"] == "SELL"
        else 1
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
            portfolio["total_value"],

        "cash":
            portfolio["usd"],

        "target_weights":
            json.dumps(
                signal[
                    "weights"
                ].to_dict()
            ),

        "current_weights":
            json.dumps(
                portfolio["weights"]
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

        "response":
            json.dumps(response),
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
# SCHEDULE
# ============================================================

def current_week_id(now):

    iso = now.isocalendar()

    return (
        f"{iso.year}-W"
        f"{iso.week:02d}"
    )


def determine_action(
    signal,
    state,
    now,
):

    today = now.date().isoformat()

    if state.get(
        "last_daily_cycle"
    ) == today:

        return (
            False,
            "ALREADY_CHECKED_TODAY",
        )

    current_regime = (
        "RISK_ON"
        if signal["risk_on"]
        else
        "RISK_OFF"
    )

    previous_regime = (
        state.get(
            "last_regime"
        )
    )

    # Immediate defensive liquidation if
    # regime changes from ON -> OFF.
    if (
        ENABLE_REGIME_EXIT
        and
        previous_regime == "RISK_ON"
        and
        current_regime == "RISK_OFF"
    ):

        return (
            True,
            "REGIME_EXIT",
        )

    week_id = current_week_id(now)

    # Core portfolio is intentionally only
    # rebalanced once per week.
    if (
        now.weekday() == CORE_WEEKDAY
        and
        state.get(
            "last_core_week"
        ) != week_id
    ):

        return (
            True,
            "WEEKLY_CORE_REBALANCE",
        )

    return (
        False,
        "DAILY_RISK_CHECK_ONLY",
    )


# ============================================================
# ORDER EXECUTION
# ============================================================

def execute_single_order(
    client,
    order,
    reason,
):

    response = api_call(
        client.place_order,
        pair=order["pair"],
        side=order["side"],
        quantity=order["quantity"],
        order_type="MARKET",
    )

    log_trade(
        reason,
        order,
        response,
    )

    if not response.get(
        "Success",
        False,
    ):

        raise RuntimeError(
            f"Order failed: {response}"
        )

    logging.info(
        "FILLED/SUBMITTED %s %s %.8f",
        order["side"],
        order["pair"],
        order["quantity"],
    )

    return response


def execute_rebalance(
    client,
    target,
    reason,
):

    prices = get_roostoo_prices(
        client
    )

    portfolio = get_portfolio(
        client,
        prices,
    )

    rules = get_exchange_rules(
        client
    )

    orders = build_orders(
        target,
        portfolio,
        prices,
        rules,
    )

    if not orders:

        logging.info(
            "No executable trades required."
        )

        return False

    # --------------------------------------------------------
    # SELL PHASE
    # --------------------------------------------------------

    sells = [
        order
        for order in orders
        if order["side"] == "SELL"
    ]

    buys = [
        order
        for order in orders
        if order["side"] == "BUY"
    ]

    if not LIVE_TRADING:

        logging.info(
            "DRY RUN — no orders sent."
        )

        for order in orders:

            logging.info(
                "PLAN %s %s %.8f ~$%.2f",
                order["side"],
                order["pair"],
                order["quantity"],
                order["estimated_value"],
            )

        return False

    executed = False

    for order in sells:

        execute_single_order(
            client,
            order,
            reason,
        )

        executed = True

        time.sleep(1)

    # --------------------------------------------------------
    # REFRESH AFTER SELLS
    # --------------------------------------------------------

    if sells:

        prices = get_roostoo_prices(
            client
        )

        portfolio = get_portfolio(
            client,
            prices,
        )

        orders = build_orders(
            target,
            portfolio,
            prices,
            rules,
        )

        buys = [
            order
            for order in orders
            if order["side"] == "BUY"
        ]

    # --------------------------------------------------------
    # BUY PHASE
    # --------------------------------------------------------

    for order in buys:

        balance = api_call(
            client.get_balance
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

        # 0.1% taker fee + extra cushion.
        required = (
            order[
                "estimated_value"
            ]
            *
            1.002
        )

        if required > available_usd:

            logging.warning(
                "Skipping %s: "
                "requires $%.2f, "
                "available $%.2f",
                order["pair"],
                required,
                available_usd,
            )

            continue

        execute_single_order(
            client,
            order,
            reason,
        )

        executed = True

        time.sleep(1)

    return executed


# ============================================================
# ONE AUTONOMOUS DAILY CYCLE
# ============================================================

def run_cycle():

    client = RoostooClient()

    state = load_state()

    now = datetime.now(
        timezone.utc
    )

    logging.info(
        "=" * 65
    )

    logging.info(
        "BOT CYCLE | %s",
        now.isoformat(),
    )

    logging.info(
        "LIVE_TRADING=%s",
        LIVE_TRADING,
    )

    prices = load_market_data()

    signal = calculate_target(
        prices
    )

    target = signal["weights"]

    logging.info(
        "BTC: %.2f | 200d MA: %.2f",
        signal["btc_price"],
        signal["btc_ma"],
    )

    logging.info(
        "REGIME: %s",
        (
            "RISK-ON"
            if signal["risk_on"]
            else
            "RISK-OFF"
        ),
    )

    for asset in ASSETS:

        logging.info(
            "TARGET %-3s %.2f%%",
            asset,
            target[asset] * 100,
        )

    logging.info(
        "TARGET CASH %.2f%%",
        (
            1
            -
            target.sum()
        )
        *
        100,
    )

    roostoo_prices = (
        get_roostoo_prices(
            client
        )
    )

    portfolio = get_portfolio(
        client,
        roostoo_prices,
    )

    logging.info(
        "PORTFOLIO $%.2f | CASH $%.2f",
        portfolio["total_value"],
        portfolio["usd"],
    )

    for asset in ASSETS:

        logging.info(
            "CURRENT %-3s %.2f%%",
            asset,
            portfolio[
                "weights"
            ][asset]
            *
            100,
        )

    should_trade, reason = (
        determine_action(
            signal,
            state,
            now,
        )
    )

    logging.info(
        "DECISION: %s",
        reason,
    )

    rules = get_exchange_rules(
        client
    )

    planned_orders = build_orders(
        target,
        portfolio,
        roostoo_prices,
        rules,
    )

    log_decision(
        reason,
        signal,
        portfolio,
        planned_orders,
    )

    if should_trade:

        execute_rebalance(
            client,
            target,
            reason,
        )

        if (
            reason
            ==
            "WEEKLY_CORE_REBALANCE"
            and
            LIVE_TRADING
        ):

            state[
                "last_core_week"
            ] = current_week_id(
                now
            )

    # Mark daily cycle only in live autonomous mode.
    # Dry-run testing therefore does not contaminate
    # production state.
    if LIVE_TRADING:

        state[
            "last_daily_cycle"
        ] = (
            now.date()
            .isoformat()
        )

        state[
            "last_regime"
        ] = (
            "RISK_ON"
            if signal["risk_on"]
            else
            "RISK_OFF"
        )

        save_state(state)

    logging.info(
        "CYCLE COMPLETE"
    )


# ============================================================
# AUTONOMOUS LOOP
# ============================================================

def run_forever():

    logging.info(
        "Autonomous bot started."
    )

    logging.info(
        "Scheduled daily evaluation: "
        "%02d:%02d UTC",
        DAILY_RUN_HOUR,
        DAILY_RUN_MINUTE,
    )

    last_attempt_date = None

    while True:

        now = datetime.now(
            timezone.utc
        )

        today = (
            now.date()
            .isoformat()
        )

        scheduled_time_reached = (
            now.hour > DAILY_RUN_HOUR
            or
            (
                now.hour
                ==
                DAILY_RUN_HOUR
                and
                now.minute
                >=
                DAILY_RUN_MINUTE
            )
        )

        if (
            scheduled_time_reached
            and
            last_attempt_date
            !=
            today
        ):

            try:

                run_cycle()

                last_attempt_date = today

            except Exception:

                logging.exception(
                    "Daily cycle failed."
                )

                # Do not mark today complete.
                # Retry after the sleep period.

        time.sleep(60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    setup_logging()

    try:

        if LIVE_TRADING:

            run_forever()

        else:

            # Dry-run mode executes exactly one
            # diagnostic cycle and exits.
            run_cycle()

    except KeyboardInterrupt:

        logging.info(
            "Bot stopped."
        )

    except Exception:

        logging.exception(
            "BOT FAILED"
        )

        raise
