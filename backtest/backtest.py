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

# ---------------------------------------------
# Competition / strategy settings
# ---------------------------------------------

INITIAL_CAPITAL = 100000
TRADING_FEE = 0.001       # 0.1% taker fee
HOURS_PER_YEAR = 24 * 365


def run_backtest(df):
    """
    Backtest our first BTC momentum strategy.

    Strategy:
        BUY when:
            MA20 > MA50
            AND 24h return > 0

        Otherwise:
            stay in cash

    Returns:
        results : dict
        df      : pandas.DataFrame
    """

    df = df.copy()

    # -----------------------------------------
    # 1. Calculate indicators
    # -----------------------------------------

    df["ma_20"] = df["close"].rolling(20).mean()

    df["ma_50"] = df["close"].rolling(50).mean()

    df["return_24h"] = df["close"].pct_change(24)

    # Hourly BTC return
    df["market_return"] = df["close"].pct_change()


    # -----------------------------------------
    # 2. Generate signal
    # -----------------------------------------

    df["signal"] = 0

    buy_condition = (
        (df["ma_20"] > df["ma_50"])
        &
        (df["return_24h"] > 0)
    )

    df.loc[buy_condition, "signal"] = 1


    # -----------------------------------------
    # 3. Prevent look-ahead bias
    # -----------------------------------------

    df["position"] = df["signal"].shift(1)

    df["position"] = df["position"].fillna(0)


    # -----------------------------------------
    # 4. Calculate trading activity
    # -----------------------------------------

    df["trade"] = df["position"].diff().abs()

    df["trade"] = df["trade"].fillna(0)


    # -----------------------------------------
    # 5. Apply transaction costs
    # -----------------------------------------

    df["transaction_cost"] = (
        df["trade"] * TRADING_FEE
    )


    # -----------------------------------------
    # 6. Strategy return
    # -----------------------------------------

    df["strategy_return"] = (
        df["position"] * df["market_return"]
        - df["transaction_cost"]
    )

    df["strategy_return"] = (
        df["strategy_return"].fillna(0)
    )


    # -----------------------------------------
    # 7. Portfolio value
    # -----------------------------------------

    df["portfolio"] = (
        INITIAL_CAPITAL
        *
        (1 + df["strategy_return"]).cumprod()
    )


    # -----------------------------------------
    # 8. Buy-and-hold benchmark
    # -----------------------------------------

    df["buy_hold"] = (
        INITIAL_CAPITAL
        *
        (1 + df["market_return"].fillna(0)).cumprod()
    )


    # -----------------------------------------
    # 9. Total returns
    # -----------------------------------------

    strategy_total_return = (
        df["portfolio"].iloc[-1]
        / INITIAL_CAPITAL
        - 1
    )

    buy_hold_return = (
        df["buy_hold"].iloc[-1]
        / INITIAL_CAPITAL
        - 1
    )


    # -----------------------------------------
    # 10. Number of transactions
    # -----------------------------------------

    number_of_trades = int(df["trade"].sum())


    # -----------------------------------------
    # 11. Maximum drawdown
    # -----------------------------------------

    running_max = df["portfolio"].cummax()

    drawdown = (
        df["portfolio"] / running_max
        - 1
    )

    max_drawdown = abs(drawdown.min())


    # -----------------------------------------
    # 12. Sharpe ratio
    # -----------------------------------------

    returns = df["strategy_return"]

    if returns.std() > 0:

        sharpe = (
            returns.mean()
            / returns.std()
            * np.sqrt(HOURS_PER_YEAR)
        )

    else:

        sharpe = 0


    # -----------------------------------------
    # 13. Sortino ratio
    # -----------------------------------------

    downside_returns = returns[
        returns < 0
    ]

    if downside_returns.std() > 0:

        sortino = (
            returns.mean()
            / downside_returns.std()
            * np.sqrt(HOURS_PER_YEAR)
        )

    else:

        sortino = 0


    # -----------------------------------------
    # 14. Annualised return
    # -----------------------------------------

    number_of_hours = len(df)

    years = number_of_hours / HOURS_PER_YEAR

    if years > 0 and df["portfolio"].iloc[-1] > 0:

        annualised_return = (
            (
                df["portfolio"].iloc[-1]
                / INITIAL_CAPITAL
            ) ** (1 / years)
            - 1
        )

    else:

        annualised_return = 0


    # -----------------------------------------
    # 15. Calmar ratio
    # -----------------------------------------

    if max_drawdown > 0:

        calmar = (
            annualised_return
            / max_drawdown
        )

    else:

        calmar = 0


    # -----------------------------------------
    # 16. Competition score
    # -----------------------------------------

    composite_score = (
        0.4 * sortino
        + 0.3 * sharpe
        + 0.3 * calmar
    )


    results = {

        "Final Portfolio":
            df["portfolio"].iloc[-1],

        "Strategy Return":
            strategy_total_return,

        "Buy & Hold Return":
            buy_hold_return,

        "Trades":
            number_of_trades,

        "Max Drawdown":
            max_drawdown,

        "Sharpe":
            sharpe,

        "Sortino":
            sortino,

        "Calmar":
            calmar,

        "Composite Score":
            composite_score
    }

    return results, df


if __name__ == "__main__":

    # Binance permits up to 1000 candles per request.
    # Start with ~41 days of hourly observations.
    btc = get_historical_ohlcv(
    symbol="BTCUSDT",
    interval="1h",
    days=365
)

    results, backtest_data = run_backtest(btc)

    print()
    print("BTC MOMENTUM BACKTEST")
    print("=" * 35)

    print(
        f"Final Portfolio:  "
        f"${results['Final Portfolio']:,.2f}"
    )

    print(
        f"Strategy Return:  "
        f"{results['Strategy Return']:.2%}"
    )

    print(
        f"Buy & Hold:       "
        f"{results['Buy & Hold Return']:.2%}"
    )

    print(
        f"Trades:           "
        f"{results['Trades']}"
    )

    print(
        f"Max Drawdown:     "
        f"{results['Max Drawdown']:.2%}"
    )

    print(
        f"Sharpe Ratio:     "
        f"{results['Sharpe']:.2f}"
    )

    print(
        f"Sortino Ratio:    "
        f"{results['Sortino']:.2f}"
    )

    print(
        f"Calmar Ratio:     "
        f"{results['Calmar']:.2f}"
    )

    print(
        f"Composite Score:  "
        f"{results['Composite Score']:.2f}"
    )
