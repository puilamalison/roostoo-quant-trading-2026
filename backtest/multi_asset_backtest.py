import sys
import os
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


# ==================================================
# CONFIGURATION
# ==================================================

ASSETS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "BNB": "BNBUSDT",
    "XRP": "XRPUSDT"
}

INITIAL_CAPITAL = 100000

TRADING_FEE = 0.001

# --------------------------------------------------
# DATA
# --------------------------------------------------

# We want 3 years for evaluation plus enough
# additional history to calculate the 200-day MA.
HISTORY_DAYS = 1300

EVALUATION_DAYS = 365 * 3


# --------------------------------------------------
# STRATEGY
# --------------------------------------------------

# IMPORTANT:
# These parameters are FROZEN from V1.
# We are NOT optimising them using older data.

MOMENTUM_SHORT = 24 * 30
MOMENTUM_LONG = 24 * 90

VOL_WINDOW = 24 * 30

MA_WINDOW = 24 * 30

REGIME_MA_WINDOW = 24 * 200

TOP_N = 2

REBALANCE_HOURS = 24 * 7

HOURS_PER_YEAR = 24 * 365


# ==================================================
# DOWNLOAD DATA
# ==================================================

def load_data():

    data = {}

    for asset, symbol in ASSETS.items():

        print()
        print("=" * 50)
        print(f"Downloading {asset}")
        print("=" * 50)

        df = get_historical_ohlcv(
            symbol=symbol,
            interval="1h",
            days=HISTORY_DAYS
        )

        df = df.set_index(
            "open_time"
        )

        data[asset] = df

        print(
            f"{asset}: "
            f"{len(df):,} candles downloaded"
        )

    return data


# ==================================================
# PREPARE FEATURES
# ==================================================

def prepare_features(data):

    prices = pd.DataFrame()

    for asset, df in data.items():

        prices[asset] = (
            df["close"]
        )

    prices = prices.dropna()

    # ------------------------------------------------
    # Returns
    # ------------------------------------------------

    hourly_returns = (
        prices.pct_change()
    )

    # ------------------------------------------------
    # Momentum
    # ------------------------------------------------

    momentum_short = (
        prices.pct_change(
            MOMENTUM_SHORT
        )
    )

    momentum_long = (
        prices.pct_change(
            MOMENTUM_LONG
        )
    )

    # ------------------------------------------------
    # Volatility
    # ------------------------------------------------

    volatility = (
        hourly_returns
        .rolling(VOL_WINDOW)
        .std()
        *
        np.sqrt(HOURS_PER_YEAR)
    )

    # ------------------------------------------------
    # Individual asset trend
    # ------------------------------------------------

    moving_average = (
        prices
        .rolling(MA_WINDOW)
        .mean()
    )

    above_ma = (
        prices
        >
        moving_average
    )

    # ------------------------------------------------
    # BTC regime filter
    # ------------------------------------------------

    btc_regime_ma = (
        prices["BTC"]
        .rolling(REGIME_MA_WINDOW)
        .mean()
    )

    risk_on = (
        prices["BTC"]
        >
        btc_regime_ma
    )

    return {
        "prices": prices,
        "returns": hourly_returns,
        "momentum_short": momentum_short,
        "momentum_long": momentum_long,
        "volatility": volatility,
        "above_ma": above_ma,
        "btc_regime_ma": btc_regime_ma,
        "risk_on": risk_on
    }


# ==================================================
# TARGET WEIGHTS
# ==================================================

def calculate_target_weights(
    features,
    i
):

    mom_30 = (
        features[
            "momentum_short"
        ]
        .iloc[i]
    )

    mom_90 = (
        features[
            "momentum_long"
        ]
        .iloc[i]
    )

    vol = (
        features[
            "volatility"
        ]
        .iloc[i]
    )

    trend = (
        features[
            "above_ma"
        ]
        .iloc[i]
    )

    risk_on = (
        features[
            "risk_on"
        ]
        .iloc[i]
    )

    # Start in cash
    weights = pd.Series(
        0.0,
        index=mom_30.index
    )

    # ------------------------------------------------
    # BTC REGIME FILTER
    # ------------------------------------------------

    if not risk_on:

        return weights

    # ------------------------------------------------
    # MOMENTUM SCORE
    # ------------------------------------------------

    score = (
        0.3 * mom_30
        +
        0.7 * mom_90
    )

    # ------------------------------------------------
    # ELIGIBILITY
    # ------------------------------------------------

    eligible = (
        (score > 0)
        &
        trend
        &
        vol.notna()
    )

    eligible_scores = (
        score[
            eligible
        ]
        .sort_values(
            ascending=False
        )
    )

    selected = (
        eligible_scores
        .head(TOP_N)
        .index
    )

    if len(selected) == 0:

        return weights

    # ------------------------------------------------
    # INVERSE-VOLATILITY WEIGHTS
    # ------------------------------------------------

    inverse_vol = (
        1
        /
        vol[selected]
    )

    inverse_vol = (
        inverse_vol
        /
        inverse_vol.sum()
    )

    weights[selected] = (
        inverse_vol
    )

    return weights


# ==================================================
# RUN BACKTEST
# ==================================================

def run_backtest(features):

    returns = (
        features["returns"]
    )

    weights = pd.DataFrame(
        0.0,
        index=returns.index,
        columns=returns.columns
    )

    current_weights = pd.Series(
        0.0,
        index=returns.columns
    )

    strategy_returns = []

    transaction_costs = []

    turnover_series = []

    minimum_history = max(
        MOMENTUM_LONG,
        VOL_WINDOW,
        MA_WINDOW,
        REGIME_MA_WINDOW
    )

    for i in range(
        len(returns)
    ):

        cost = 0.0
        turnover = 0.0

        # ------------------------------------------------
        # WEEKLY REBALANCE
        # ------------------------------------------------

        if (
            i >= minimum_history
            and
            i % REBALANCE_HOURS == 0
        ):

            new_weights = (
                calculate_target_weights(
                    features,
                    i
                )
            )

            turnover = (
                new_weights
                -
                current_weights
            ).abs().sum()

            cost = (
                turnover
                *
                TRADING_FEE
            )

            current_weights = (
                new_weights.copy()
            )

        weights.iloc[i] = (
            current_weights
        )

        # ------------------------------------------------
        # PORTFOLIO RETURN
        # ------------------------------------------------

        if i == 0:

            portfolio_return = 0.0

        else:

            market_return = (
                returns.iloc[i]
            )

            previous_weights = (
                weights.iloc[
                    i - 1
                ]
            )

            portfolio_return = (
                previous_weights
                *
                market_return
            ).sum()

        portfolio_return -= cost

        strategy_returns.append(
            portfolio_return
        )

        transaction_costs.append(
            cost
        )

        turnover_series.append(
            turnover
        )

    strategy_returns = pd.Series(
        strategy_returns,
        index=returns.index
    )

    transaction_costs = pd.Series(
        transaction_costs,
        index=returns.index
    )

    turnover_series = pd.Series(
        turnover_series,
        index=returns.index
    )

    return (
        strategy_returns,
        weights,
        transaction_costs,
        turnover_series
    )


# ==================================================
# PERFORMANCE METRICS
# ==================================================

def calculate_metrics(
    strategy_returns,
    transaction_costs,
    turnover
):

    portfolio = (
        INITIAL_CAPITAL
        *
        (
            1
            +
            strategy_returns
        ).cumprod()
    )

    total_return = (
        portfolio.iloc[-1]
        /
        INITIAL_CAPITAL
        -
        1
    )

    # ------------------------------------------------
    # DRAWDOWN
    # ------------------------------------------------

    running_max = (
        portfolio.cummax()
    )

    drawdown = (
        portfolio
        /
        running_max
        -
        1
    )

    max_drawdown = abs(
        drawdown.min()
    )

    # ------------------------------------------------
    # SHARPE
    # ------------------------------------------------

    std = (
        strategy_returns.std()
    )

    if std > 0:

        sharpe = (
            strategy_returns.mean()
            /
            std
            *
            np.sqrt(
                HOURS_PER_YEAR
            )
        )

    else:

        sharpe = 0.0

    # ------------------------------------------------
    # SORTINO
    # ------------------------------------------------

    downside = (
        strategy_returns[
            strategy_returns < 0
        ]
    )

    downside_std = (
        downside.std()
    )

    if (
        len(downside) > 1
        and
        downside_std > 0
    ):

        sortino = (
            strategy_returns.mean()
            /
            downside_std
            *
            np.sqrt(
                HOURS_PER_YEAR
            )
        )

    else:

        sortino = 0.0

    # ------------------------------------------------
    # ANNUALISED RETURN
    # ------------------------------------------------

    years = (
        len(
            strategy_returns
        )
        /
        HOURS_PER_YEAR
    )

    if (
        years > 0
        and
        portfolio.iloc[-1] > 0
    ):

        annualised_return = (
            (
                portfolio.iloc[-1]
                /
                INITIAL_CAPITAL
            )
            **
            (1 / years)
            -
            1
        )

    else:

        annualised_return = 0.0

    # ------------------------------------------------
    # CALMAR
    # ------------------------------------------------

    if max_drawdown > 0:

        calmar = (
            annualised_return
            /
            max_drawdown
        )

    else:

        calmar = 0.0

    # ------------------------------------------------
    # COMPOSITE
    # ------------------------------------------------

    composite = (
        0.4 * sortino
        +
        0.3 * sharpe
        +
        0.3 * calmar
    )

    return {
        "Final Portfolio":
            portfolio.iloc[-1],

        "Total Return":
            total_return,

        "Max Drawdown":
            max_drawdown,

        "Sharpe":
            sharpe,

        "Sortino":
            sortino,

        "Calmar":
            calmar,

        "Composite":
            composite,

        "Trading Costs":
            transaction_costs.sum(),

        "Turnover":
            turnover.sum()
    }


# ==================================================
# PRINT METRICS
# ==================================================

def print_metrics(
    title,
    returns,
    costs,
    turnover
):

    metrics = (
        calculate_metrics(
            returns,
            costs,
            turnover
        )
    )

    print()
    print(title)
    print("=" * 55)

    print(
        f"Period:          "
        f"{returns.index[0]}"
        f" -> "
        f"{returns.index[-1]}"
    )

    print(
        f"Final Portfolio: "
        f"${metrics['Final Portfolio']:,.2f}"
    )

    print(
        f"Total Return:    "
        f"{metrics['Total Return']:.2%}"
    )

    print(
        f"Max Drawdown:    "
        f"{metrics['Max Drawdown']:.2%}"
    )

    print(
        f"Sharpe Ratio:    "
        f"{metrics['Sharpe']:.2f}"
    )

    print(
        f"Sortino Ratio:   "
        f"{metrics['Sortino']:.2f}"
    )

    print(
        f"Calmar Ratio:    "
        f"{metrics['Calmar']:.2f}"
    )

    print(
        f"Composite Score: "
        f"{metrics['Composite']:.2f}"
    )

    print(
        f"Trading Costs:   "
        f"{metrics['Trading Costs']:.2%}"
    )

    print(
        f"Total Turnover:  "
        f"{metrics['Turnover']:.2f}"
    )

    return metrics


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    # ------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------

    data = load_data()

    features = (
        prepare_features(
            data
        )
    )

    # ------------------------------------------------
    # RUN STRATEGY
    # ------------------------------------------------

    (
        all_returns,
        all_weights,
        all_costs,
        all_turnover
    ) = run_backtest(
        features
    )

    # ------------------------------------------------
    # KEEP LATEST 3 YEARS
    # ------------------------------------------------

    evaluation_hours = (
        EVALUATION_DAYS
        *
        24
    )

    returns = (
        all_returns
        .iloc[
            -evaluation_hours:
        ]
    )

    costs = (
        all_costs
        .loc[
            returns.index
        ]
    )

    turnover = (
        all_turnover
        .loc[
            returns.index
        ]
    )

    weights = (
        all_weights
        .loc[
            returns.index
        ]
    )

    risk_on = (
        features[
            "risk_on"
        ]
        .loc[
            returns.index
        ]
    )

    # ------------------------------------------------
    # FULL 3-YEAR RESULT
    # ------------------------------------------------

    print_metrics(
        "FULL 3-YEAR RESULT",
        returns,
        costs,
        turnover
    )

    # ------------------------------------------------
    # YEAR-BY-YEAR RESULTS
    # ------------------------------------------------

    print()
    print()
    print(
        "YEAR-BY-YEAR ROBUSTNESS"
    )
    print(
        "=" * 55
    )

    # Split the 3-year period into three
    # consecutive 365-day blocks.
    for year_number in range(3):

        start = (
            year_number
            *
            HOURS_PER_YEAR
        )

        end = (
            start
            +
            HOURS_PER_YEAR
        )

        year_returns = (
            returns.iloc[
                start:end
            ]
        )

        year_costs = (
            costs.iloc[
                start:end
            ]
        )

        year_turnover = (
            turnover.iloc[
                start:end
            ]
        )

        print_metrics(
            f"YEAR {year_number + 1}",
            year_returns,
            year_costs,
            year_turnover
        )

    # ------------------------------------------------
    # REGIME STATISTICS
    # ------------------------------------------------

    print()
    print(
        "REGIME STATISTICS"
    )
    print(
        "=" * 55
    )

    print(
        f"Risk-on:  "
        f"{risk_on.mean():.2%}"
    )

    print(
        f"Risk-off: "
        f"{1 - risk_on.mean():.2%}"
    )

    # ------------------------------------------------
    # CURRENT SIGNAL
    # ------------------------------------------------

    print()
    print(
        "LATEST PORTFOLIO"
    )
    print(
        "=" * 55
    )

    latest_risk_on = (
        features[
            "risk_on"
        ]
        .iloc[-1]
    )

    btc_price = (
        features[
            "prices"
        ]["BTC"]
        .iloc[-1]
    )

    btc_ma = (
        features[
            "btc_regime_ma"
        ]
        .iloc[-1]
    )

    print(
        f"BTC Price:      "
        f"${btc_price:,.2f}"
    )

    print(
        f"BTC 200d MA:    "
        f"${btc_ma:,.2f}"
    )

    print(
        f"Market Regime:  "
        f"{'RISK-ON' if latest_risk_on else 'RISK-OFF'}"
    )

    print()

    latest_weights = (
        weights
        .iloc[-1]
        .sort_values(
            ascending=False
        )
    )

    for asset, weight in (
        latest_weights.items()
    ):

        print(
            f"{asset}: "
            f"{weight:.2%}"
        )
