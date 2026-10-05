import requests
import pandas as pd


BINANCE_URL = "https://api.binance.com/api/v3/klines"


def get_ohlcv(symbol="BTCUSDT", interval="1h", limit=200):
    """
    Download historical candlestick data from Binance.

    Parameters
    ----------
    symbol : str
        Example: "BTCUSDT"

    interval : str
        Example: "1h"

    limit : int
        Number of candles to retrieve.

    Returns
    -------
    pandas.DataFrame
        Historical OHLCV data.
    """

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    response = requests.get(
        BINANCE_URL,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(
        data,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore"
        ]
    )

    # Keep only what our strategy currently needs
    df = df[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ].copy()

    # Convert prices/volume from strings to numbers
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    df[numeric_columns] = df[numeric_columns].astype(float)

    # Convert Binance millisecond timestamps into readable UTC time
    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True
    )

    return df


def get_historical_ohlcv(
    symbol="BTCUSDT",
    interval="1h",
    days=365
):
    """
    Download a longer history from Binance by making
    multiple requests.

    Parameters
    ----------
    symbol : str
        Binance trading pair, e.g. "BTCUSDT"

    interval : str
        Candle interval, e.g. "1h"

    days : int
        Number of historical days to download.

    Returns
    -------
    pandas.DataFrame
        Historical OHLCV data sorted from oldest to newest.
    """

    milliseconds_per_hour = 60 * 60 * 1000

    end_time = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

    start_time = (
        end_time
        - days * 24 * milliseconds_per_hour
    )

    all_data = []

    current_start = start_time

    while current_start < end_time:

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000
        }

        response = requests.get(
            BINANCE_URL,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        # No more candles available
        if not data:
            break

        all_data.extend(data)

        # Last candle's opening timestamp
        last_open_time = data[-1][0]

        # Start next request immediately after it
        current_start = (
            last_open_time
            + milliseconds_per_hour
        )

        print(
            f"Downloaded {len(all_data)} candles..."
        )

    df = pd.DataFrame(
        all_data,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore"
        ]
    )

    df = df[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ].copy()

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    df[numeric_columns] = (
        df[numeric_columns].astype(float)
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True
    )

    # Safety: remove any accidental duplicate candles
    df = df.drop_duplicates(
        subset="open_time"
    )

    df = df.sort_values(
        "open_time"
    )

    df = df.reset_index(drop=True)

    return df

if __name__ == "__main__":

    btc = get_historical_ohlcv(
        symbol="BTCUSDT",
        interval="1h",
        days=365
    )

    print()
    print("DATA SUMMARY")
    print("=" * 40)

    print(f"Candles: {len(btc)}")
    print(f"Start:   {btc['open_time'].iloc[0]}")
    print(f"End:     {btc['open_time'].iloc[-1]}")

    print()
    print("Latest candles:")
    print(btc.tail())
