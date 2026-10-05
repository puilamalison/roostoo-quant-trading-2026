import os
import sys
import time

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from execution.roostoo_client import RoostooClient

from execution.portfolio_rebalancer import (
    load_market_data,
    build_prices,
    calculate_target_weights,
    get_exchange_rules,
    get_roostoo_prices,
    get_current_portfolio,
    calculate_trades,
    print_strategy,
    print_portfolio,
    print_trades
)


# ==================================================
# SAFETY CONFIGURATION
# ==================================================

# This script WILL place real orders on the
# currently connected Roostoo account.
EXECUTE_FULL_TEST_REBALANCE = True

# We expect the general TEST account to be
# approximately $50,000.
#
# This prevents accidental execution on the
# ~$100,000 competition account.
MAX_TEST_ACCOUNT_VALUE = 60000

# Give Roostoo a moment to update state
# after each market order.
ORDER_WAIT_SECONDS = 1


# ==================================================
# VERIFY FILLED ORDER
# ==================================================

def verify_order(
    client,
    order_response
):

    """
    Verify that a submitted order was filled.

    Input
    -----
    client : RoostooClient

    order_response : dict

    Output
    ------
    dict
        Confirmed order information.
    """

    if not order_response.get(
        "Success",
        False
    ):

        raise RuntimeError(
            "Order submission failed: "
            + str(order_response)
        )

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
            "Order response did not "
            "contain OrderID."
        )

    print(
        f"Order ID: {order_id}"
    )

    time.sleep(
        ORDER_WAIT_SECONDS
    )

    query = (
        client.query_order(
            order_id=order_id
        )
    )

    if not query.get(
        "Success",
        False
    ):

        raise RuntimeError(
            "Could not verify order "
            f"{order_id}: {query}"
        )

    matched = (
        query.get(
            "OrderMatched",
            []
        )
    )

    if not matched:

        raise RuntimeError(
            f"No matched order found "
            f"for {order_id}"
        )

    confirmed = matched[0]

    status = (
        confirmed.get(
            "Status"
        )
    )

    if status != "FILLED":

        raise RuntimeError(
            f"Order {order_id} "
            f"was not FILLED. "
            f"Status = {status}"
        )

    print(
        f"Confirmed FILLED: "
        f"{confirmed['Side']} "
        f"{confirmed['FilledQuantity']} "
        f"{confirmed['Pair']} "
        f"@ "
        f"${confirmed['FilledAverPrice']:,.4f}"
    )

    print(
        f"Commission: "
        f"${confirmed['CommissionChargeValue']:,.4f}"
    )

    return confirmed


# ==================================================
# EXECUTE ONE TRADE
# ==================================================

def execute_trade(
    client,
    trade
):

    print()
    print("-" * 65)

    print(
        f"EXECUTING "
        f"{trade['side']} "
        f"{trade['quantity']} "
        f"{trade['asset']}"
    )

    print(
        f"Estimated value: "
        f"${trade['trade_value']:,.2f}"
    )

    response = (
        client.place_order(
            pair=trade["pair"],
            side=trade["side"],
            quantity=trade["quantity"],
            order_type="MARKET"
        )
    )

    print(
        "Submission response:"
    )

    print(response)

    return verify_order(
        client,
        response
    )


# ==================================================
# TEST ACCOUNT SAFETY CHECK
# ==================================================

def verify_test_account(
    portfolio
):

    value = (
        portfolio[
            "portfolio_value"
        ]
    )

    print()
    print("ACCOUNT SAFETY CHECK")
    print("=" * 65)

    print(
        f"Portfolio value: "
        f"${value:,.2f}"
    )

    if value > MAX_TEST_ACCOUNT_VALUE:

        raise RuntimeError(
            "SAFETY STOP: account value "
            f"${value:,.2f} exceeds "
            f"${MAX_TEST_ACCOUNT_VALUE:,.2f}. "
            "This may be the competition account."
        )

    print(
        "Account value is within "
        "TEST-account safety range."
    )


# ==================================================
# FINAL ALLOCATION
# ==================================================

def print_final_allocation(
    portfolio,
    prices
):

    total = (
        portfolio[
            "portfolio_value"
        ]
    )

    print()
    print("FINAL TEST PORTFOLIO")
    print("=" * 65)

    usd = (
        portfolio[
            "usd"
        ]
    )

    cash_weight = (
        usd / total
        if total > 0
        else 0
    )

    print(
        f"USD: "
        f"${usd:,.2f} "
        f"({cash_weight:.2%})"
    )

    for asset, quantity in (
        portfolio[
            "holdings"
        ].items()
    ):

        value = (
            quantity
            *
            prices[asset]
        )

        weight = (
            value / total
            if total > 0
            else 0
        )

        print(
            f"{asset}: "
            f"{quantity:.8f} "
            f"= ${value:,.2f} "
            f"({weight:.2%})"
        )

    print()

    print(
        f"Total: "
        f"${total:,.2f}"
    )


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    print()
    print("=" * 65)
    print(
        "ROOSTOO FULL TEST-ACCOUNT REBALANCE"
    )
    print("=" * 65)

    print(
        f"EXECUTE_FULL_TEST_REBALANCE = "
        f"{EXECUTE_FULL_TEST_REBALANCE}"
    )

    if not EXECUTE_FULL_TEST_REBALANCE:

        raise RuntimeError(
            "Execution switch is False."
        )

    client = (
        RoostooClient()
    )

    # ==================================================
    # EXCHANGE RULES
    # ==================================================

    rules = (
        get_exchange_rules(
            client
        )
    )

    # ==================================================
    # STRATEGY
    # ==================================================

    data = (
        load_market_data()
    )

    prices = (
        build_prices(
            data
        )
    )

    strategy = (
        calculate_target_weights(
            prices
        )
    )

    print_strategy(
        strategy
    )

    # ==================================================
    # CURRENT ROOSTOO PORTFOLIO
    # ==================================================

    roostoo_prices = (
        get_roostoo_prices(
            client
        )
    )

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

    # ==================================================
    # CRITICAL SAFETY CHECK
    # ==================================================

    verify_test_account(
        portfolio
    )

    # ==================================================
    # INITIAL TRADE PLAN
    # ==================================================

    trades = (
        calculate_trades(
            strategy["weights"],
            portfolio,
            roostoo_prices,
            rules
        )
    )

    print_trades(
        trades
    )

    # ==================================================
    # PHASE 1 — SELLS
    # ==================================================

    sell_trades = [
        trade
        for trade in trades
        if trade["side"] == "SELL"
    ]

    print()
    print("=" * 65)
    print("PHASE 1 — SELL ORDERS")
    print("=" * 65)

    if not sell_trades:

        print(
            "No sell orders required."
        )

    else:

        for trade in sell_trades:

            execute_trade(
                client,
                trade
            )

    # ==================================================
    # REFRESH ACCOUNT AFTER SELLS
    # ==================================================

    time.sleep(
        ORDER_WAIT_SECONDS
    )

    print()
    print("=" * 65)
    print(
        "REFRESHING PORTFOLIO AFTER SELLS"
    )
    print("=" * 65)

    roostoo_prices = (
        get_roostoo_prices(
            client
        )
    )

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

    # Recalculate trades using fresh balances
    # and fresh Roostoo prices.
    refreshed_trades = (
        calculate_trades(
            strategy["weights"],
            portfolio,
            roostoo_prices,
            rules
        )
    )

    buy_trades = [
        trade
        for trade in refreshed_trades
        if trade["side"] == "BUY"
    ]

    # ==================================================
    # PHASE 2 — BUYS
    # ==================================================

    print()
    print("=" * 65)
    print("PHASE 2 — BUY ORDERS")
    print("=" * 65)

    if not buy_trades:

        print(
            "No buy orders required."
        )

    else:

        print_trades(
            buy_trades
        )

        for trade in buy_trades:

            # ------------------------------------------
            # Refresh USD before every BUY.
            # ------------------------------------------

            current_prices = (
                get_roostoo_prices(
                    client
                )
            )

            current_portfolio = (
                get_current_portfolio(
                    client,
                    current_prices
                )
            )

            available_usd = (
                current_portfolio[
                    "usd"
                ]
            )

            # Include 0.1% taker fee estimate.
            estimated_required = (
                trade[
                    "trade_value"
                ]
                *
                1.001
            )

            print()
            print(
                f"Available USD: "
                f"${available_usd:,.2f}"
            )

            print(
                f"Estimated cash required: "
                f"${estimated_required:,.2f}"
            )

            if (
                estimated_required
                >
                available_usd
            ):

                raise RuntimeError(
                    "SAFETY STOP: insufficient "
                    "USD for proposed BUY."
                )

            execute_trade(
                client,
                trade
            )

    # ==================================================
    # FINAL VERIFICATION
    # ==================================================

    time.sleep(
        ORDER_WAIT_SECONDS
    )

    final_prices = (
        get_roostoo_prices(
            client
        )
    )

    final_portfolio = (
        get_current_portfolio(
            client,
            final_prices
        )
    )

    print_final_allocation(
        final_portfolio,
        final_prices
    )

    print()
    print("=" * 65)
    print(
        "FULL TEST-ACCOUNT REBALANCE COMPLETE"
    )
    print("=" * 65)
