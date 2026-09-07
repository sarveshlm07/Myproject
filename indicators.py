import pandas as pd

def calculate_indicators(df):

    df = df.copy()

    # EMA
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))


    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()


   


    # Supertrend
    high_low = df["High"] - df["Low"]
    high_close = abs(df["High"] - df["Close"].shift())
    low_close = abs(df["Low"] - df["Close"].shift())

    tr = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    # ATR
    atr = tr.rolling(14).mean()

    df["ATR"] = atr

# Supertrend
    df["Supertrend"] = df["Close"] - (3 * atr)

    # VWAP
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum()

    # EMA Slopes
    df["EMA20_Slope"] = df["EMA20"].diff()
    df["EMA50_Slope"] = df["EMA50"].diff()

# VWAP Distance
    df["VWAP_Distance"] = df["Close"] - df["VWAP"]

# Volume Average
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()

# Volume Strength
    df["Volume_Strength"] = (
    df["Volume"] / df["Volume_MA20"]
).fillna(0)

    # ADX

    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    tr = pd.concat([
        df["High"] - df["Low"],
        abs(df["High"] - df["Close"].shift()),
        abs(df["Low"] - df["Close"].shift())
], axis=1).max(axis=1)

    atr14 = tr.rolling(14).mean()

    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df["ADX"] = dx.rolling(14).mean()

    # EMA200
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

# Candle Body
    df["Body"] = abs(df["Close"] - df["Open"])

# Upper Wick
    df["Upper_Wick"] = df["High"] - df[["Open", "Close"]].max(axis=1)

# Lower Wick
    df["Lower_Wick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]

    return df