# market_state.py
# Market-state classifier for NiftyAI.
# Uses completed candles only.

import math
import numpy as np
import pandas as pd


def _safe(v, default=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _atr(df, period=10):
    if df is None or df.empty:
        return 5.0

    h = pd.to_numeric(df["High"], errors="coerce")
    l = pd.to_numeric(df["Low"], errors="coerce")
    c = pd.to_numeric(df["Close"], errors="coerce")
    pc = c.shift(1)

    tr = pd.concat(
        [h-l, (h-pc).abs(), (l-pc).abs()],
        axis=1
    ).max(axis=1)

    return max(_safe(tr.tail(period).mean(), 5.0), 1.0)


def classify_market_state(df, support=None, resistance=None):
    """
    Returns:
        state, bullish_score, bearish_score, reasons, features

    States:
        TREND_UP
        TREND_DOWN
        REVERSAL_UP
        REVERSAL_DOWN
        COMPRESSION
        RANGE
        BREAKOUT_UP
        BREAKOUT_DOWN
        MIXED
    """

    if df is None or len(df) < 2:
        return "MIXED", 0.0, 0.0, ["Not enough completed candles"], {}

    # Last row is running; classifier uses completed candles.
    base = df.iloc[:-1].copy()

    if base.empty:
        return "MIXED", 0.0, 0.0, ["No completed candle yet"], {}

    for c in ["Open", "High", "Low", "Close"]:
        base[c] = pd.to_numeric(base[c], errors="coerce")

    base.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

    if base.empty:
        return "MIXED", 0.0, 0.0, ["No usable completed candles"], {}

    atr = _atr(base)

    recent = base.tail(min(8, len(base)))
    closes = recent["Close"].to_numpy(float)
    highs = recent["High"].to_numpy(float)
    lows = recent["Low"].to_numpy(float)

    bullish = 0.0
    bearish = 0.0
    reasons = []

    up_closes = sum(closes[i] > closes[i-1] for i in range(1, len(closes)))
    down_closes = sum(closes[i] < closes[i-1] for i in range(1, len(closes)))

    higher_highs = sum(highs[i] > highs[i-1] for i in range(1, len(highs)))
    lower_highs = sum(highs[i] < highs[i-1] for i in range(1, len(highs)))
    higher_lows = sum(lows[i] > lows[i-1] for i in range(1, len(lows)))
    lower_lows = sum(lows[i] < lows[i-1] for i in range(1, len(lows)))

    recent_move = closes[-1] - closes[0]

    if up_closes >= 4:
        bullish += 5
        reasons.append("Recent closes mostly rising")

    if down_closes >= 4:
        bearish += 5
        reasons.append("Recent closes mostly falling")

    if higher_highs + higher_lows >= 7:
        bullish += 7
        reasons.append("Recent structure favors HH/HL")

    if lower_highs + lower_lows >= 7:
        bearish += 7
        reasons.append("Recent structure favors LH/LL")

    if recent_move >= max(5.0, atr*0.8):
        bullish += 6
        reasons.append("Recent completed move is strong upward")

    if recent_move <= -max(5.0, atr*0.8):
        bearish += 6
        reasons.append("Recent completed move is strong downward")

    # -----------------------------------------------------
    # Compression / wedge detection
    # -----------------------------------------------------
    ranges = (recent["High"] - recent["Low"]).to_numpy(float)

    compression = False
    shrinking_ranges = False
    converging_swings = False

    if len(ranges) >= 5:
        first_half = float(np.mean(ranges[:max(2, len(ranges)//2)]))
        second_half = float(np.mean(ranges[-max(2, len(ranges)//2):]))

        shrinking_ranges = (
            first_half > 0
            and second_half <= first_half * 0.72
        )

        # Swing envelope shrinking: recent max-high - min-low narrows.
        k = max(3, len(recent)//2)

        earlier = recent.iloc[:k]
        later = recent.iloc[-k:]

        earlier_width = _safe(
            earlier["High"].max() - earlier["Low"].min(),
            0
        )

        later_width = _safe(
            later["High"].max() - later["Low"].min(),
            0
        )

        converging_swings = (
            earlier_width > 0
            and later_width <= earlier_width * 0.70
        )

        compression = shrinking_ranges and converging_swings

    # -----------------------------------------------------
    # Breakout / false-break context
    # -----------------------------------------------------
    breakout_up = False
    breakout_down = False

    if len(base) >= 4:
        prior = base.iloc[:-1]
        last = base.iloc[-1]

        prior_high = _safe(prior["High"].tail(12).max())
        prior_low = _safe(prior["Low"].tail(12).min())
        last_close = _safe(last["Close"])

        breakout_up = last_close > prior_high + atr*0.10
        breakout_down = last_close < prior_low - atr*0.10

    # -----------------------------------------------------
    # Reversal detection
    # -----------------------------------------------------
    reversal_up = False
    reversal_down = False

    if len(base) >= 6:
        old = base.iloc[-6:-3]
        new = base.iloc[-3:]

        old_move = _safe(old["Close"].iloc[-1] - old["Close"].iloc[0])
        new_move = _safe(new["Close"].iloc[-1] - new["Close"].iloc[0])

        reversal_up = (
            old_move < -max(3.0, atr*0.35)
            and new_move > max(3.0, atr*0.35)
        )

        reversal_down = (
            old_move > max(3.0, atr*0.35)
            and new_move < -max(3.0, atr*0.35)
        )

    if breakout_up:
        state = "BREAKOUT_UP"
        bullish += 10
        reasons.append("Completed breakout above recent structure")

    elif breakout_down:
        state = "BREAKOUT_DOWN"
        bearish += 10
        reasons.append("Completed breakdown below recent structure")

    elif compression:
        state = "COMPRESSION"
        reasons.append("Shrinking candle ranges + converging swings")

    elif reversal_up:
        state = "REVERSAL_UP"
        bullish += 8
        reasons.append("Recent completed candles show bullish reversal")

    elif reversal_down:
        state = "REVERSAL_DOWN"
        bearish += 8
        reasons.append("Recent completed candles show bearish reversal")

    elif bullish >= bearish + 6:
        state = "TREND_UP"

    elif bearish >= bullish + 6:
        state = "TREND_DOWN"

    else:
        recent_width = _safe(recent["High"].max() - recent["Low"].min(), 0)

        if recent_width <= atr * 2.2:
            state = "RANGE"
            reasons.append("Recent price is moving in a tight range")
        else:
            state = "MIXED"

    features = {
        "atr": atr,
        "recent_move": recent_move,
        "up_closes": up_closes,
        "down_closes": down_closes,
        "higher_highs": higher_highs,
        "higher_lows": higher_lows,
        "lower_highs": lower_highs,
        "lower_lows": lower_lows,
        "shrinking_ranges": shrinking_ranges,
        "converging_swings": converging_swings,
        "compression": compression,
    }

    return state, bullish, bearish, reasons, features
