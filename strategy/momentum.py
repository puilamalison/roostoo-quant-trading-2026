import sys
import os

# Allow Python to import from the project root
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from data.market_data import get_ohlcv


def calculate_signal(df):
    """
    Generate a simple momentum/trend signal.

    Returns
    -------
    str
        "BUY", "SELL", or "HOLD"
    """

    df = df.copy()

    # Calculate moving averages
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_50"] = df["close"].rolling(50).mean()

    # Calculate 24-hour price momentum
    df["return_24h"] = df["close"].pct_change(24)

    latest = df.iloc[-1]

    price = latest["close"]
    ma_20 = latest["ma_20"]
    ma_50 = latest["ma_50"]
    return_24h = latest["return_24h"]

    # BUY:
    # short-term trend > long-term trend
    # AND price has positive 24h momentum
    if ma_20 > ma_50 and return_24h > 0:
        signal = "BUY"

    # SELL:
    # short-term trend < long-term trend
    # AND price has negative 24h momentum
    elif ma_20 < ma_50 and return_24h < 0:
        signal = "SELL"

    else:
        signal = "HOLD"

    return {
        "signal": signal,
        "price": price,
        "ma_20": ma_20,
        "ma_50": ma_50,
        "return_24h": return_24h
    }


if __name__ == "__main__":

    btc = get_ohlcv(
        symbol="BTCUSDT",
        interval="1h",
        limit=200
    )

    result = calculate_signal(btc)

    print("BTC Strategy")
    print("---------------------")
    print(f"Price:      ${result['price']:,.2f}")
    print(f"20h MA:     ${result['ma_20']:,.2f}")
    print(f"50h MA:     ${result['ma_50']:,.2f}")
    print(f"24h Return: {result['return_24h']:.2%}")
    print(f"Signal:     {result['signal']}")
