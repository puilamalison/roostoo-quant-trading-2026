import os
import sys
import math
import time

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from execution.roostoo_client import RoostooClient


# ==================================================
# TEST CONFIGURATION
# ==================================================

PAIR = "ETH/USD"

# Deliberately tiny TEST-account order.
TARGET_USD = 10.00

# Explicit safety switch.
EXECUTE_TEST_ORDER = True


# ==================================================
# ROUND DOWN
# ==================================================

def round_down(value, precision):
    """
    Round a float downward to a specified
    number of decimal places.

    Inputs
    ------
    value : float
    precision : int

    Output
    ------
    float
    """

    factor = 10 ** precision

    return (
        math.floor(
            value * factor
        )
        /
        factor
    )


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("ROOSTOO CONTROLLED LIVE TEST")
    print("=" * 60)

    client = RoostooClient()

    # ------------------------------------------------
    # 1. Confirm account balance
    # ------------------------------------------------

    print()
    print("1. BALANCE BEFORE ORDER")
    print("-" * 60)

    balance_before = (
        client.get_balance()
    )

    print(balance_before)

    if not balance_before.get(
        "Success",
        False
    ):
        raise RuntimeError(
            "Could not retrieve account balance."
        )

    usd_before = float(
        balance_before
        .get("SpotWallet", {})
        .get("USD", {})
        .get("Free", 0)
    )

    print()
    print(
        f"Available USD: "
        f"${usd_before:,.2f}"
    )

    # ------------------------------------------------
    # 2. Exchange rules
    # ------------------------------------------------

    print()
    print("2. EXCHANGE RULES")
    print("-" * 60)

    exchange_info = (
        client.get_exchange_info()
    )

    trade_pairs = (
        exchange_info.get(
            "TradePairs",
            {}
        )
    )

    if PAIR not in trade_pairs:
        raise RuntimeError(
            f"{PAIR} not found in exchange info."
        )

    pair_rules = (
        trade_pairs[PAIR]
    )

    amount_precision = int(
        pair_rules[
            "AmountPrecision"
        ]
    )

    mini_order = float(
        pair_rules[
            "MiniOrder"
        ]
    )

    print(
        f"Pair:             {PAIR}"
    )

    print(
        f"Amount precision: "
        f"{amount_precision}"
    )

    print(
        f"Minimum order:    "
        f"${mini_order:.2f}"
    )

    # ------------------------------------------------
    # 3. Current ETH price
    # ------------------------------------------------

    print()
    print("3. CURRENT PRICE")
    print("-" * 60)

    ticker = (
        client.get_ticker(
            PAIR
        )
    )

    if not ticker.get(
        "Success",
        False
    ):
        raise RuntimeError(
            "Could not retrieve ticker."
        )

    ticker_data = (
        ticker
        .get("Data", {})
        .get(PAIR, {})
    )

    ask_price = float(
        ticker_data[
            "MinAsk"
        ]
    )

    print(
        f"ETH Ask: "
        f"${ask_price:,.2f}"
    )

    # ------------------------------------------------
    # 4. Calculate safe quantity
    # ------------------------------------------------

    raw_quantity = (
        TARGET_USD
        /
        ask_price
    )

    quantity = (
        round_down(
            raw_quantity,
            amount_precision
        )
    )

    estimated_value = (
        quantity
        *
        ask_price
    )

    print()
    print("4. TEST ORDER")
    print("-" * 60)

    print(
        f"Target USD:       "
        f"${TARGET_USD:.2f}"
    )

    print(
        f"Raw quantity:     "
        f"{raw_quantity:.8f}"
    )

    print(
        f"Order quantity:   "
        f"{quantity}"
    )

    print(
        f"Estimated value:  "
        f"${estimated_value:.2f}"
    )

    # ------------------------------------------------
    # Safety checks
    # ------------------------------------------------

    if quantity <= 0:
        raise RuntimeError(
            "Calculated quantity is zero."
        )

    if estimated_value < mini_order:
        raise RuntimeError(
            "Test order is below "
            "Roostoo minimum order size."
        )

    if estimated_value > 15:
        raise RuntimeError(
            "SAFETY STOP: test order "
            "exceeds $15."
        )

    if usd_before < 20:
        raise RuntimeError(
            "SAFETY STOP: insufficient "
            "USD balance."
        )

    if not EXECUTE_TEST_ORDER:
        raise RuntimeError(
            "EXECUTE_TEST_ORDER=False. "
            "Order not sent."
        )

    # ------------------------------------------------
    # 5. PLACE ORDER
    # ------------------------------------------------

    print()
    print("5. SENDING TEST ORDER")
    print("-" * 60)

    order_response = (
        client.place_order(
            pair=PAIR,
            side="BUY",
            quantity=quantity,
            order_type="MARKET"
        )
    )

    print(order_response)

    if not order_response.get(
        "Success",
        False
    ):
        raise RuntimeError(
            "TEST ORDER FAILED: "
            +
            str(
                order_response.get(
                    "ErrMsg"
                )
            )
        )

    # ------------------------------------------------
    # Extract OrderID
    # ------------------------------------------------

    order_detail = (
        order_response.get(
            "OrderDetail",
            {}
        )
    )

    order_id = (
        order_detail.get(
            "OrderID"
        )
    )

    if order_id is None:
        raise RuntimeError(
            "Order succeeded but "
            "OrderID was not returned."
        )

    print()
    print(
        f"Order ID: {order_id}"
    )

    # ------------------------------------------------
    # 6. Query order
    # ------------------------------------------------

    # Short pause gives the API a moment to
    # finish updating account state.
    time.sleep(1)

    print()
    print("6. VERIFY ORDER")
    print("-" * 60)

    query_response = (
        client.query_order(
            order_id=order_id
        )
    )

    print(query_response)

    # ------------------------------------------------
    # 7. Updated balance
    # ------------------------------------------------

    time.sleep(1)

    print()
    print("7. BALANCE AFTER ORDER")
    print("-" * 60)

    balance_after = (
        client.get_balance()
    )

    print(balance_after)

    print()
    print("=" * 60)
    print(
        "CONTROLLED TEST FINISHED"
    )
    print("=" * 60)
