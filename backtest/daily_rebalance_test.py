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
# CONFIGURATION — FROZEN STRATEGY
# ============================================================

ASSETS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "BNB": "BNBUSDT",
    "XRP": "XRPUSDT",
}

INITIAL_CAPITAL = 100000.0

# Download enough history for:
# 3-year evaluation + 200-day MA warm-up.
DOWNLOAD_DAYS = 1300

# Evaluate exactly the latest 3 years.
EVALUATION_DAYS = 1095

# Frozen strategy parameters.
MOMENTUM_SHORT = 24 * 30
MOMENTUM_LONG = 24 * 90

VOL_WINDOW = 24 * 30
ASSET_MA_WINDOW = 24 * 30
REGIME_MA_WINDOW = 24 * 200

TOP_N = 2

# Roostoo market taker fee.
FEE_RATE = 0.001

HOURS_PER_YEAR = 24 * 365

# ============================================================
# ONLY EXPERIMENTAL CHANGE
# ============================================================

# Weekly version effectively rebalanced every 7 days.
# Here we rebalance every 24 hours.
REBALANCE_HOURS = 24


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

        # Remove newest candle because it may still
        # be incomplete.
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
# CALCULATE STRATEGY INPUTS
# ============================================================

def calculate_features(prices):

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

    btc_200d_ma = (
        prices["BTC"]
        .rolling(REGIME_MA_WINDOW)
        .mean()
    )

    return {
        "returns": returns,
        "momentum_30": momentum_30,
        "momentum_90": momentum_90,
        "volatility": volatility,
        "asset_ma": asset_ma,
        "btc_200d_ma": btc_200d_ma,
    }


# ============================================================
# TARGET WEIGHTS
# ============================================================

def get_target_weights(
    i,
    prices,
    features
):

    assets = prices.columns

    weights = pd.Series(
        0.0,
        index=assets
    )

    btc_price = (
        prices["BTC"].iloc[i]
    )

    btc_ma = (
        features[
            "btc_200d_ma"
        ].iloc[i]
    )

    # --------------------------------------------
    # BTC REGIME FILTER
    # --------------------------------------------

    if pd.isna(btc_ma):
        return weights

    if btc_price <= btc_ma:
        return weights

    # --------------------------------------------
    # MOMENTUM SCORE
    # --------------------------------------------

    mom_30 = (
        features[
            "momentum_30"
        ].iloc[i]
    )

    mom_90 = (
        features[
            "momentum_90"
        ].iloc[i]
    )

    vol = (
        features[
            "volatility"
        ].iloc[i]
    )

    asset_ma = (
        features[
            "asset_ma"
        ].iloc[i]
    )

    score = (
        0.3 * mom_30
        +
        0.7 * mom_90
    )

    # --------------------------------------------
    # INDIVIDUAL ASSET TREND FILTER
    # --------------------------------------------

    trend = (
        prices.iloc[i]
        >
        asset_ma
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
        return weights

    # --------------------------------------------
    # INVERSE VOLATILITY WEIGHTING
    # --------------------------------------------

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
# BACKTEST
# ============================================================

def run_backtest(
    prices,
    features
):

    evaluation_hours = (
        EVALUATION_DAYS * 24
    )

    start_i = (
        len(prices)
        -
        evaluation_hours
    )

    if start_i <= 0:

        raise RuntimeError(
            "Not enough data for "
            "requested evaluation period."
        )

    print()
    print("=" * 70)
    print("DAILY REBALANCE BACKTEST")
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
        f"Rebalance every:  "
        f"{REBALANCE_HOURS} hours"
    )

    current_weights = pd.Series(
        0.0,
        index=prices.columns
    )

    portfolio_value = (
        INITIAL_CAPITAL
    )

    portfolio_values = []

    portfolio_times = []

    total_turnover = 0.0
    total_cost_fraction = 0.0

    rebalance_count = 0
    trade_events = 0

    risk_on_hours = 0
    risk_off_hours = 0

    # --------------------------------------------
    # LOOP THROUGH EVALUATION PERIOD
    # --------------------------------------------

    for i in range(
        start_i,
        len(prices)
    ):

        # ----------------------------------------
        # APPLY RETURN FROM PREVIOUS HOUR
        # ----------------------------------------

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
                    current_weights
                    *
                    asset_returns
                ).sum()
            )

            portfolio_value *= (
                1.0
                +
                portfolio_return
            )

        # ----------------------------------------
        # TRACK REGIME
        # ----------------------------------------

        btc_ma = (
            features[
                "btc_200d_ma"
            ].iloc[i]
        )

        if (
            not pd.isna(btc_ma)
            and
            prices["BTC"].iloc[i]
            >
            btc_ma
        ):
            risk_on_hours += 1

        else:
            risk_off_hours += 1

        # ----------------------------------------
        # DAILY REBALANCE
        # ----------------------------------------

        hours_since_start = (
            i - start_i
        )

        should_rebalance = (
            hours_since_start
            %
            REBALANCE_HOURS
            ==
            0
        )

        if should_rebalance:

            target_weights = (
                get_target_weights(
                    i,
                    prices,
                    features
                )
            )

            turnover = float(
                (
                    target_weights
                    -
                    current_weights
                )
                .abs()
                .sum()
            )

            # Cost approximation:
            # fee rate × dollar turnover.
            cost_fraction = (
                turnover
                *
                FEE_RATE
            )

            if turnover > 0:

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

            current_weights = (
                target_weights.copy()
            )

            rebalance_count += 1

        portfolio_values.append(
            portfolio_value
        )

        portfolio_times.append(
            prices.index[i]
        )

    equity = pd.Series(
        portfolio_values,
        index=portfolio_times,
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
        "rebalance_count": rebalance_count,
        "trade_events": trade_events,
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

    hourly_returns = (
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

    # --------------------------------------------
    # ANNUALISED RETURN
    # --------------------------------------------

    years = (
        len(hourly_returns)
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

    # --------------------------------------------
    # SHARPE
    # --------------------------------------------

    return_std = (
        hourly_returns.std()
    )

    if (
        return_std > 0
        and
        not pd.isna(return_std)
    ):

        sharpe = (
            hourly_returns.mean()
            /
            return_std
            *
            np.sqrt(
                HOURS_PER_YEAR
            )
        )

    else:

        sharpe = 0.0

    # --------------------------------------------
    # SORTINO
    # --------------------------------------------

    downside_returns = (
        hourly_returns[
            hourly_returns < 0
        ]
    )

    downside_std = (
        downside_returns.std()
    )

    if (
        downside_std > 0
        and
        not pd.isna(
            downside_std
        )
    ):

        sortino = (
            hourly_returns.mean()
            /
            downside_std
            *
            np.sqrt(
                HOURS_PER_YEAR
            )
        )

    else:

        sortino = 0.0

    # --------------------------------------------
    # CALMAR
    # --------------------------------------------

    if max_drawdown > 0:

        calmar = (
            annual_return
            /
            max_drawdown
        )

    else:

        calmar = 0.0

    # --------------------------------------------
    # HACKATHON COMPOSITE
    # --------------------------------------------

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
# YEAR-BY-YEAR ANALYSIS
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

    # Rebase to $100,000 so each year is
    # independently comparable.
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

    std = returns.std()

    if std > 0:

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

    if downside_std > 0:

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

    hours = len(returns)

    years = (
        hours
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
                1 / years
            )
            -
            1
        )

    else:
        annual_return = 0

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
# PRINT RESULTS
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
        "FULL 3-YEAR — DAILY REBALANCE",
        metrics
    )

    print()
    print(
        f"Rebalance checks: "
        f"{results['rebalance_count']}"
    )

    print(
        f"Rebalances with trades: "
        f"{results['trade_events']}"
    )

    print(
        f"Risk-on:  "
        f"{results['risk_on_pct']:.2%}"
    )

    print(
        f"Risk-off: "
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
    # DIRECT WEEKLY BENCHMARK
    # ========================================================

    print()
    print("=" * 70)
    print("WEEKLY FROZEN-V1 BENCHMARK")
    print("=" * 70)

    print("Return:       +619.34%")
    print("Max Drawdown:   38.17%")
    print("Sharpe:          1.62")
    print("Sortino:         1.54")
    print("Calmar:          2.44")
    print("Composite:       1.83")
    print("Trading Costs:   6.39%")
    print("Turnover:       63.90")

    print()
    print("=" * 70)
    print(
        "EXPERIMENT COMPLETE"
    )
    print("=" * 70)
