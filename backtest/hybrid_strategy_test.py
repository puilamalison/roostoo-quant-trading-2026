import os
import sys
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


# ============================================================
# CONFIGURATION
# ============================================================

ASSETS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "BNB": "BNBUSDT",
    "XRP": "XRPUSDT",
}

INITIAL_CAPITAL = 100000.0

DOWNLOAD_DAYS = 1300
EVALUATION_DAYS = 1095

FEE_RATE = 0.001
HOURS_PER_YEAR = 24 * 365


# ============================================================
# PORTFOLIO SPLIT
# ============================================================

# Current experiment:
# 95% weekly core
# 5% daily tactical sleeve

CORE_WEIGHT = 0.95
TACTICAL_WEIGHT = 0.05


# ============================================================
# CORE STRATEGY — FROZEN V1
# ============================================================

CORE_MOM_SHORT = 24 * 30
CORE_MOM_LONG = 24 * 90

CORE_VOL_WINDOW = 24 * 30
CORE_ASSET_MA = 24 * 30

REGIME_MA = 24 * 200

CORE_TOP_N = 2

CORE_REBALANCE_HOURS = 24 * 7


# ============================================================
# TACTICAL STRATEGY
# ============================================================

TACTICAL_MOM_SHORT = 24 * 7
TACTICAL_MOM_LONG = 24 * 30

TACTICAL_VOL_WINDOW = 24 * 14
TACTICAL_ASSET_MA = 24 * 14

TACTICAL_TOP_N = 1

TACTICAL_REBALANCE_HOURS = 24


# ============================================================
# LOAD DATA
# ============================================================

def load_prices():

    print()
    print("=" * 70)
    print("DOWNLOADING MARKET DATA")
    print("=" * 70)

    prices = pd.DataFrame()

    for asset, symbol in ASSETS.items():

        print()
        print(f"Downloading {asset}...")

        df = get_historical_ohlcv(
            symbol=symbol,
            interval="1h",
            days=DOWNLOAD_DAYS
        )

        # Remove newest candle because it may
        # still be incomplete.
        if len(df) > 1:
            df = df.iloc[:-1].copy()

        df = df.set_index("open_time")

        prices[asset] = df["close"]

    prices = (
        prices
        .dropna()
        .sort_index()
    )

    print()
    print(
        f"Combined candles: {len(prices):,}"
    )

    print(
        f"Data start: {prices.index[0]}"
    )

    print(
        f"Data end:   {prices.index[-1]}"
    )

    return prices


# ============================================================
# CALCULATE FEATURES
# ============================================================

def calculate_features(prices):

    returns = prices.pct_change()

    features = {}

    # --------------------------------------------------------
    # CORE FEATURES
    # --------------------------------------------------------

    features["core_mom_short"] = (
        prices.pct_change(
            CORE_MOM_SHORT
        )
    )

    features["core_mom_long"] = (
        prices.pct_change(
            CORE_MOM_LONG
        )
    )

    features["core_vol"] = (
        returns
        .rolling(
            CORE_VOL_WINDOW
        )
        .std()
        *
        np.sqrt(
            HOURS_PER_YEAR
        )
    )

    features["core_ma"] = (
        prices
        .rolling(
            CORE_ASSET_MA
        )
        .mean()
    )

    # --------------------------------------------------------
    # TACTICAL FEATURES
    # --------------------------------------------------------

    features["tactical_mom_short"] = (
        prices.pct_change(
            TACTICAL_MOM_SHORT
        )
    )

    features["tactical_mom_long"] = (
        prices.pct_change(
            TACTICAL_MOM_LONG
        )
    )

    features["tactical_vol"] = (
        returns
        .rolling(
            TACTICAL_VOL_WINDOW
        )
        .std()
        *
        np.sqrt(
            HOURS_PER_YEAR
        )
    )

    features["tactical_ma"] = (
        prices
        .rolling(
            TACTICAL_ASSET_MA
        )
        .mean()
    )

    # --------------------------------------------------------
    # SHARED BTC REGIME FILTER
    # --------------------------------------------------------

    features["btc_regime_ma"] = (
        prices["BTC"]
        .rolling(
            REGIME_MA
        )
        .mean()
    )

    return features


# ============================================================
# BTC REGIME
# ============================================================

def is_risk_on(
    i,
    prices,
    features
):

    btc_price = (
        prices["BTC"]
        .iloc[i]
    )

    btc_ma = (
        features[
            "btc_regime_ma"
        ]
        .iloc[i]
    )

    if pd.isna(btc_ma):
        return False

    return (
        btc_price
        >
        btc_ma
    )


# ============================================================
# CORE TARGET
# ============================================================

def get_core_target(
    i,
    prices,
    features
):

    weights = pd.Series(
        0.0,
        index=prices.columns
    )

    if not is_risk_on(
        i,
        prices,
        features
    ):
        return weights

    mom_short = (
        features[
            "core_mom_short"
        ]
        .iloc[i]
    )

    mom_long = (
        features[
            "core_mom_long"
        ]
        .iloc[i]
    )

    vol = (
        features[
            "core_vol"
        ]
        .iloc[i]
    )

    ma = (
        features[
            "core_ma"
        ]
        .iloc[i]
    )

    # Frozen V1 momentum score
    score = (
        0.3 * mom_short
        +
        0.7 * mom_long
    )

    trend = (
        prices.iloc[i]
        >
        ma
    )

    eligible = (
        (score > 0)
        &
        trend
        &
        vol.notna()
        &
        (vol > 0)
    )

    ranked = (
        score[
            eligible
        ]
        .sort_values(
            ascending=False
        )
    )

    selected = (
        ranked
        .head(
            CORE_TOP_N
        )
        .index
    )

    if len(selected) == 0:
        return weights

    inverse_vol = (
        1.0
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


# ============================================================
# TACTICAL TARGET
# ============================================================

def get_tactical_target(
    i,
    prices,
    features
):

    weights = pd.Series(
        0.0,
        index=prices.columns
    )

    if not is_risk_on(
        i,
        prices,
        features
    ):
        return weights

    mom_short = (
        features[
            "tactical_mom_short"
        ]
        .iloc[i]
    )

    mom_long = (
        features[
            "tactical_mom_long"
        ]
        .iloc[i]
    )

    vol = (
        features[
            "tactical_vol"
        ]
        .iloc[i]
    )

    ma = (
        features[
            "tactical_ma"
        ]
        .iloc[i]
    )

    # Faster tactical momentum score
    score = (
        0.6 * mom_short
        +
        0.4 * mom_long
    )

    trend = (
        prices.iloc[i]
        >
        ma
    )

    eligible = (
        (score > 0)
        &
        trend
        &
        vol.notna()
        &
        (vol > 0)
    )

    ranked = (
        score[
            eligible
        ]
        .sort_values(
            ascending=False
        )
    )

    selected = (
        ranked
        .head(
            TACTICAL_TOP_N
        )
        .index
    )

    if len(selected) == 0:
        return weights

    # Tactical sleeve holds only the
    # strongest eligible asset.
    weights[
        selected[0]
    ] = 1.0

    return weights


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    prices,
    features
):

    evaluation_hours = (
        EVALUATION_DAYS
        *
        24
    )

    start_i = (
        len(prices)
        -
        evaluation_hours
    )

    if start_i <= 0:

        raise RuntimeError(
            "Not enough historical data."
        )

    print()
    print("=" * 70)
    print("HYBRID 95/5 BACKTEST")
    print("=" * 70)

    print(
        f"Evaluation start: "
        f"{prices.index[start_i]}"
    )

    print(
        f"Evaluation end:   "
        f"{prices.index[-1]}"
    )

    print(
        f"Core allocation:      "
        f"{CORE_WEIGHT:.0%}"
    )

    print(
        f"Tactical allocation:  "
        f"{TACTICAL_WEIGHT:.0%}"
    )

    print(
        f"Core rebalance:       "
        f"{CORE_REBALANCE_HOURS // 24} days"
    )

    print(
        f"Tactical rebalance:   "
        f"{TACTICAL_REBALANCE_HOURS // 24} day"
    )

    # --------------------------------------------------------
    # INITIAL WEIGHTS
    # --------------------------------------------------------

    core_weights = pd.Series(
        0.0,
        index=prices.columns
    )

    tactical_weights = pd.Series(
        0.0,
        index=prices.columns
    )

    combined_weights = pd.Series(
        0.0,
        index=prices.columns
    )

    portfolio_value = (
        INITIAL_CAPITAL
    )

    equity_values = []
    equity_times = []

    total_turnover = 0.0
    total_cost_fraction = 0.0

    core_rebalances = 0
    tactical_rebalances = 0

    trade_events = 0

    # IMPORTANT:
    # Store the actual dates, not just
    # the number of active days.
    active_trade_days = set()

    risk_on_hours = 0
    risk_off_hours = 0

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    for i in range(
        start_i,
        len(prices)
    ):

        # ----------------------------------------------------
        # APPLY PREVIOUS POSITION'S HOURLY RETURN
        # ----------------------------------------------------

        if i > start_i:

            asset_returns = (
                prices.iloc[i]
                /
                prices.iloc[i - 1]
                -
                1.0
            )

            portfolio_return = float(
                (
                    combined_weights
                    *
                    asset_returns
                ).sum()
            )

            portfolio_value *= (
                1.0
                +
                portfolio_return
            )

        # ----------------------------------------------------
        # REGIME TRACKING
        # ----------------------------------------------------

        if is_risk_on(
            i,
            prices,
            features
        ):

            risk_on_hours += 1

        else:

            risk_off_hours += 1

        hours_since_start = (
            i - start_i
        )

        previous_combined = (
            combined_weights.copy()
        )

        changed = False

        # ----------------------------------------------------
        # WEEKLY CORE
        # ----------------------------------------------------

        if (
            hours_since_start
            %
            CORE_REBALANCE_HOURS
            ==
            0
        ):

            core_weights = (
                get_core_target(
                    i,
                    prices,
                    features
                )
            )

            core_rebalances += 1

            changed = True

        # ----------------------------------------------------
        # DAILY TACTICAL
        # ----------------------------------------------------

        if (
            hours_since_start
            %
            TACTICAL_REBALANCE_HOURS
            ==
            0
        ):

            tactical_weights = (
                get_tactical_target(
                    i,
                    prices,
                    features
                )
            )

            tactical_rebalances += 1

            changed = True

        # ----------------------------------------------------
        # COMBINE CORE + TACTICAL
        # ----------------------------------------------------

        if changed:

            new_combined = (
                CORE_WEIGHT
                *
                core_weights
                +
                TACTICAL_WEIGHT
                *
                tactical_weights
            )

            turnover = float(
                (
                    new_combined
                    -
                    previous_combined
                )
                .abs()
                .sum()
            )

            if turnover > 0:

                cost_fraction = (
                    turnover
                    *
                    FEE_RATE
                )

                portfolio_value *= (
                    1.0
                    -
                    cost_fraction
                )

                total_turnover += (
                    turnover
                )

                total_cost_fraction += (
                    cost_fraction
                )

                trade_events += 1

                active_trade_days.add(
                    prices.index[i].date()
                )

            combined_weights = (
                new_combined.copy()
            )

        equity_values.append(
            portfolio_value
        )

        equity_times.append(
            prices.index[i]
        )

    equity = pd.Series(
        equity_values,
        index=equity_times,
        name="equity"
    )

    regime_total = (
        risk_on_hours
        +
        risk_off_hours
    )

    risk_on_pct = (
        risk_on_hours
        /
        regime_total
        if regime_total > 0
        else 0
    )

    risk_off_pct = (
        risk_off_hours
        /
        regime_total
        if regime_total > 0
        else 0
    )

    return {
        "equity": equity,
        "turnover": total_turnover,
        "cost_fraction": total_cost_fraction,
        "core_rebalances": core_rebalances,
        "tactical_rebalances": tactical_rebalances,
        "trade_events": trade_events,

        # Number of active days
        "active_trade_days": len(
            active_trade_days
        ),

        # Actual dates
        "active_trade_dates":
            active_trade_days,

        "risk_on_pct": risk_on_pct,
        "risk_off_pct": risk_off_pct,
    }


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    equity,
    turnover,
    cost_fraction
):

    returns = (
        equity
        .pct_change()
        .dropna()
    )

    final_value = float(
        equity.iloc[-1]
    )

    total_return = (
        final_value
        /
        INITIAL_CAPITAL
        -
        1.0
    )

    running_max = (
        equity.cummax()
    )

    drawdown = (
        equity
        /
        running_max
        -
        1.0
    )

    max_drawdown = abs(
        float(
            drawdown.min()
        )
    )

    years = (
        len(returns)
        /
        HOURS_PER_YEAR
    )

    if (
        years > 0
        and
        final_value > 0
    ):

        annual_return = (
            (
                final_value
                /
                INITIAL_CAPITAL
            )
            **
            (
                1.0
                /
                years
            )
            -
            1.0
        )

    else:

        annual_return = 0.0

    # --------------------------------------------------------
    # SHARPE
    # --------------------------------------------------------

    std = (
        returns.std()
    )

    if (
        std > 0
        and
        not pd.isna(std)
    ):

        sharpe = (
            returns.mean()
            /
            std
            *
            np.sqrt(
                HOURS_PER_YEAR
            )
        )

    else:

        sharpe = 0.0

    # --------------------------------------------------------
    # SORTINO
    # --------------------------------------------------------

    downside = (
        returns[
            returns < 0
        ]
    )

    downside_std = (
        downside.std()
    )

    if (
        downside_std > 0
        and
        not pd.isna(
            downside_std
        )
    ):

        sortino = (
            returns.mean()
            /
            downside_std
            *
            np.sqrt(
                HOURS_PER_YEAR
            )
        )

    else:

        sortino = 0.0

    # --------------------------------------------------------
    # CALMAR
    # --------------------------------------------------------

    if max_drawdown > 0:

        calmar = (
            annual_return
            /
            max_drawdown
        )

    else:

        calmar = 0.0

    # --------------------------------------------------------
    # HACKATHON COMPOSITE
    # --------------------------------------------------------

    composite = (
        0.4 * sortino
        +
        0.3 * sharpe
        +
        0.3 * calmar
    )

    return {
        "final_value": final_value,
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "composite": composite,
        "turnover": turnover,
        "cost_fraction": cost_fraction,
    }


# ============================================================
# YEAR-BY-YEAR METRICS
# ============================================================

def calculate_period_metrics(
    equity,
    start,
    end
):

    period = (
        equity.loc[
            (equity.index >= start)
            &
            (equity.index < end)
        ]
    )

    if len(period) < 2:
        return None

    rebased = (
        period
        /
        period.iloc[0]
        *
        INITIAL_CAPITAL
    )

    returns = (
        rebased
        .pct_change()
        .dropna()
    )

    final_value = float(
        rebased.iloc[-1]
    )

    total_return = (
        final_value
        /
        INITIAL_CAPITAL
        -
        1
    )

    running_max = (
        rebased.cummax()
    )

    drawdown = (
        rebased
        /
        running_max
        -
        1
    )

    max_drawdown = abs(
        float(
            drawdown.min()
        )
    )

    std = (
        returns.std()
    )

    if (
        std > 0
        and
        not pd.isna(std)
    ):

        sharpe = (
            returns.mean()
            /
            std
            *
            np.sqrt(
                HOURS_PER_YEAR
            )
        )

    else:

        sharpe = 0.0

    downside = (
        returns[
            returns < 0
        ]
    )

    downside_std = (
        downside.std()
    )

    if (
        downside_std > 0
        and
        not pd.isna(
            downside_std
        )
    ):

        sortino = (
            returns.mean()
            /
            downside_std
            *
            np.sqrt(
                HOURS_PER_YEAR
            )
        )

    else:

        sortino = 0.0

    years = (
        len(returns)
        /
        HOURS_PER_YEAR
    )

    if years > 0:

        annual_return = (
            (
                final_value
                /
                INITIAL_CAPITAL
            )
            **
            (
                1.0
                /
                years
            )
            -
            1
        )

    else:

        annual_return = 0.0

    if max_drawdown > 0:

        calmar = (
            annual_return
            /
            max_drawdown
        )

    else:

        calmar = 0.0

    composite = (
        0.4 * sortino
        +
        0.3 * sharpe
        +
        0.3 * calmar
    )

    return {
        "final_value": final_value,
        "return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "composite": composite,
    }


# ============================================================
# 14-DAY COMPLIANCE ANALYSIS
# ============================================================

def analyse_compliance(
    equity,
    active_trade_dates
):

    evaluation_start = (
        equity.index[0]
        .normalize()
    )

    evaluation_end = (
        equity.index[-1]
        .normalize()
    )

    # pd.date_range(...)
    #
    # Input:
    # start timestamp,
    # end timestamp,
    # frequency string.
    #
    # Output:
    # pandas DatetimeIndex.
    all_days = pd.date_range(
        start=evaluation_start,
        end=evaluation_end,
        freq="D",
        tz="UTC"
    )

    active_dates = {
        pd.Timestamp(
            day,
            tz="UTC"
        ).normalize()
        for day in active_trade_dates
    }

    # pd.Series(...)
    #
    # Input:
    # sequence of values + index.
    #
    # Output:
    # pandas Series.
    daily_activity = pd.Series(
        [
            1
            if day in active_dates
            else 0
            for day in all_days
        ],
        index=all_days,
        dtype=float
    )

    # rolling(window=14)
    #
    # Input:
    # pandas Series.
    #
    # Output:
    # Rolling object.
    #
    # .sum()
    # Output:
    # pandas Series containing the sum
    # for each rolling 14-day window.
    rolling_14 = (
        daily_activity
        .rolling(
            window=14
        )
        .sum()
        .dropna()
    )

    minimum_active = int(
        rolling_14.min()
    )

    maximum_active = int(
        rolling_14.max()
    )

    average_active = float(
        rolling_14.mean()
    )

    median_active = float(
        rolling_14.median()
    )

    compliant_windows = (
        rolling_14 >= 8
    )

    total_windows = len(
        rolling_14
    )

    compliant_count = int(
        compliant_windows.sum()
    )

    compliance_rate = float(
        compliant_windows.mean()
    )

    # Also calculate more useful
    # threshold frequencies.
    at_least_1 = float(
        (
            rolling_14 >= 1
        ).mean()
    )

    at_least_4 = float(
        (
            rolling_14 >= 4
        ).mean()
    )

    at_least_6 = float(
        (
            rolling_14 >= 6
        ).mean()
    )

    at_least_8 = float(
        (
            rolling_14 >= 8
        ).mean()
    )

    return {
        "total_windows":
            total_windows,

        "minimum_active":
            minimum_active,

        "maximum_active":
            maximum_active,

        "average_active":
            average_active,

        "median_active":
            median_active,

        "compliant_count":
            compliant_count,

        "compliance_rate":
            compliance_rate,

        "at_least_1":
            at_least_1,

        "at_least_4":
            at_least_4,

        "at_least_6":
            at_least_6,

        "at_least_8":
            at_least_8,
    }


# ============================================================
# PRINT PERFORMANCE
# ============================================================

def print_metrics(
    title,
    metrics
):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"Final Portfolio: "
        f"${metrics['final_value']:,.2f}"
    )

    print(
        f"Total Return:     "
        f"{metrics['total_return']:.2%}"
    )

    print(
        f"Annual Return:    "
        f"{metrics['annual_return']:.2%}"
    )

    print(
        f"Max Drawdown:     "
        f"{metrics['max_drawdown']:.2%}"
    )

    print(
        f"Sharpe:           "
        f"{metrics['sharpe']:.2f}"
    )

    print(
        f"Sortino:          "
        f"{metrics['sortino']:.2f}"
    )

    print(
        f"Calmar:           "
        f"{metrics['calmar']:.2f}"
    )

    print(
        f"Composite:        "
        f"{metrics['composite']:.2f}"
    )

    print(
        f"Trading Costs:    "
        f"{metrics['cost_fraction']:.2%}"
    )

    print(
        f"Turnover:         "
        f"{metrics['turnover']:.2f}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    prices = (
        load_prices()
    )

    features = (
        calculate_features(
            prices
        )
    )

    results = (
        run_backtest(
            prices,
            features
        )
    )

    metrics = (
        calculate_metrics(
            results["equity"],
            results["turnover"],
            results["cost_fraction"]
        )
    )

    print_metrics(
        "FULL 3-YEAR — HYBRID 95/5",
        metrics
    )

    print()

    print(
        f"Core rebalance checks:     "
        f"{results['core_rebalances']}"
    )

    print(
        f"Tactical rebalance checks: "
        f"{results['tactical_rebalances']}"
    )

    print(
        f"Rebalances with trades:    "
        f"{results['trade_events']}"
    )

    print(
        f"Active trading days:       "
        f"{results['active_trade_days']}"
    )

    print(
        f"Risk-on:                   "
        f"{results['risk_on_pct']:.2%}"
    )

    print(
        f"Risk-off:                  "
        f"{results['risk_off_pct']:.2%}"
    )

    # ========================================================
    # YEAR-BY-YEAR
    # ========================================================

    equity = (
        results["equity"]
    )

    start = (
        equity.index[0]
    )

    for year_number in range(3):

        period_start = (
            start
            +
            pd.Timedelta(
                days=365
                *
                year_number
            )
        )

        period_end = (
            period_start
            +
            pd.Timedelta(
                days=365
            )
        )

        year_metrics = (
            calculate_period_metrics(
                equity,
                period_start,
                period_end
            )
        )

        if year_metrics is None:
            continue

        print()
        print("=" * 70)

        print(
            f"YEAR {year_number + 1}"
        )

        print("=" * 70)

        print(
            f"Period: "
            f"{period_start} "
            f"-> "
            f"{period_end}"
        )

        print(
            f"Final Portfolio: "
            f"${year_metrics['final_value']:,.2f}"
        )

        print(
            f"Return:           "
            f"{year_metrics['return']:.2%}"
        )

        print(
            f"Max Drawdown:     "
            f"{year_metrics['max_drawdown']:.2%}"
        )

        print(
            f"Sharpe:           "
            f"{year_metrics['sharpe']:.2f}"
        )

        print(
            f"Sortino:          "
            f"{year_metrics['sortino']:.2f}"
        )

        print(
            f"Calmar:           "
            f"{year_metrics['calmar']:.2f}"
        )

        print(
            f"Composite:        "
            f"{year_metrics['composite']:.2f}"
        )

    # ========================================================
    # COMPLIANCE STRESS TEST
    # ========================================================

    compliance = (
        analyse_compliance(
            equity,
            results[
                "active_trade_dates"
            ]
        )
    )

    print()
    print("=" * 70)
    print(
        "14-DAY ACTIVE-TRADING COMPLIANCE TEST"
    )
    print("=" * 70)

    print(
        f"Total 14-day windows:       "
        f"{compliance['total_windows']}"
    )

    print(
        f"Minimum active days:        "
        f"{compliance['minimum_active']}"
    )

    print(
        f"Maximum active days:        "
        f"{compliance['maximum_active']}"
    )

    print(
        f"Average active days:        "
        f"{compliance['average_active']:.2f}"
    )

    print(
        f"Median active days:         "
        f"{compliance['median_active']:.2f}"
    )

    print(
        f"Windows with >=8 days:      "
        f"{compliance['compliant_count']}"
    )

    print(
        f"Compliance rate:            "
        f"{compliance['compliance_rate']:.2%}"
    )

    print()
    print(
        "ROLLING 14-DAY DISTRIBUTION"
    )
    print("-" * 70)

    print(
        f"At least 1 active day:      "
        f"{compliance['at_least_1']:.2%}"
    )

    print(
        f"At least 4 active days:     "
        f"{compliance['at_least_4']:.2%}"
    )

    print(
        f"At least 6 active days:     "
        f"{compliance['at_least_6']:.2%}"
    )

    print(
        f"At least 8 active days:     "
        f"{compliance['at_least_8']:.2%}"
    )

    print()
    print("INTERPRETATION")
    print("-" * 70)

    if (
        compliance[
            "compliance_rate"
        ]
        >=
        0.90
    ):

        print(
            "Historical compliance "
            "reliability: HIGH"
        )

    elif (
        compliance[
            "compliance_rate"
        ]
        >=
        0.70
    ):

        print(
            "Historical compliance "
            "reliability: MODERATE"
        )

    else:

        print(
            "Historical compliance "
            "reliability: LOW"
        )

    if (
        compliance[
            "minimum_active"
        ]
        <
        8
    ):

        print(
            "WARNING: Some historical "
            "14-day periods contain "
            "fewer than 8 active "
            "trading days."
        )

    else:

        print(
            "Every historical 14-day "
            "window contains at least "
            "8 active trading days."
        )

    # ========================================================
    # BENCHMARKS
    # ========================================================

    print()
    print("=" * 70)
    print(
        "WEEKLY FROZEN-V1 BENCHMARK"
    )
    print("=" * 70)

    print(
        "Return:       +619.34%"
    )

    print(
        "Max Drawdown:   38.17%"
    )

    print(
        "Sharpe:          1.62"
    )

    print(
        "Sortino:         1.54"
    )

    print(
        "Calmar:          2.44"
    )

    print(
        "Composite:       1.83"
    )

    print(
        "Trading Costs:   6.39%"
    )

    print(
        "Turnover:       63.90"
    )

    print()
    print("=" * 70)
    print(
        "HYBRID 90/10 BENCHMARK"
    )
    print("=" * 70)

    print(
        "Return:       +439.05%"
    )

    print(
        "Max Drawdown:   48.21%"
    )

    print(
        "Sharpe:          1.38"
    )

    print(
        "Sortino:         1.35"
    )

    print(
        "Calmar:          1.56"
    )

    print(
        "Composite:       1.42"
    )

    print(
        "Trading Costs:   8.70%"
    )

    print(
        "Turnover:       87.01"
    )

    print()
    print("=" * 70)
    print(
        "HYBRID EXPERIMENT COMPLETE"
    )
    print("=" * 70)
