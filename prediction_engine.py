import math
import numpy as np
import pandas as pd

from market_state import classify_market_state
from prediction_logger import get_state_reliability

__all__ = ["predict_market"]


# =========================================================
# HELPERS
# =========================================================

def _safe(v, default=0.0):
    try:
        x = float(v)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return float(default)


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _clean(df):
    if df is None:
        return pd.DataFrame()

    try:
        x = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    except Exception:
        return pd.DataFrame()

    if x.empty:
        return x

    rename = {}
    for c in x.columns:
        k = str(c).strip().lower()
        if k == "open":
            rename[c] = "Open"
        elif k == "high":
            rename[c] = "High"
        elif k == "low":
            rename[c] = "Low"
        elif k == "close":
            rename[c] = "Close"
        elif k == "volume":
            rename[c] = "Volume"

    x.rename(columns=rename, inplace=True)

    required = ["Open", "High", "Low", "Close"]
    if not all(c in x.columns for c in required):
        return pd.DataFrame()

    for c in required:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    if "Volume" in x.columns:
        x["Volume"] = pd.to_numeric(x["Volume"], errors="coerce").fillna(0.0)

    x.dropna(subset=required, inplace=True)

    if isinstance(x.index, pd.DatetimeIndex):
        x.sort_index(inplace=True)

    return x


def _completed(df):
    return df.iloc[:-1].copy() if len(df) >= 2 else df.iloc[0:0].copy()


def _atr(df, period=10):
    if df.empty:
        return 5.0

    if len(df) == 1:
        return max(_safe(df.iloc[0]["High"] - df.iloc[0]["Low"], 5.0), 1.0)

    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    prev = c.shift(1)

    tr = pd.concat(
        [
            h - l,
            (h - prev).abs(),
            (l - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return max(_safe(tr.tail(period).mean(), 5.0), 1.0)


# =========================================================
# 1) FULL SESSION CONTEXT
# =========================================================

def _session_context(df):
    """
    Uses ALL completed candles available from market open.
    This is the broad context, not the final next-candle trigger.
    """
    base = _completed(df)

    if base.empty:
        return 0.0, ["Session context is still forming"], {}

    o = base["Open"].to_numpy(float)
    h = base["High"].to_numpy(float)
    l = base["Low"].to_numpy(float)
    c = base["Close"].to_numpy(float)

    n = len(base)

    day_open = float(o[0])
    day_high = float(np.max(h))
    day_low = float(np.min(l))
    last_close = float(c[-1])
    day_range = max(day_high - day_low, 1.0)

    score = 0.0
    reasons = []

    # Open-to-now direction.
    move = last_close - day_open
    score += _clip(move / day_range, -1.0, 1.0) * 15.0

    if move > 0:
        reasons.append(f"Session is +{move:.1f} pts above open")
    elif move < 0:
        reasons.append(f"Session is {move:.1f} pts below open")

    # Position in full day range.
    location = _clip((last_close - day_low) / day_range, 0.0, 1.0)

    if location >= 0.72:
        score += 7.0
        reasons.append("Price is holding upper part of day range")
    elif location <= 0.28:
        score -= 7.0
        reasons.append("Price is holding lower part of day range")

    # HH/HL vs LH/LL across the whole completed session.
    if n >= 2:
        hh = hl = lh = ll = 0

        for i in range(1, n):
            if h[i] > h[i - 1]:
                hh += 1
            elif h[i] < h[i - 1]:
                lh += 1

            if l[i] > l[i - 1]:
                hl += 1
            elif l[i] < l[i - 1]:
                ll += 1

        bull_structure = hh + hl
        bear_structure = lh + ll
        total = max(bull_structure + bear_structure, 1)

        balance = (bull_structure - bear_structure) / total
        score += balance * 15.0

        if balance >= 0.12:
            reasons.append("Full chart structure favors HH/HL")
        elif balance <= -0.12:
            reasons.append("Full chart structure favors LH/LL")

    # Morning -> current progression using all completed candles.
    if n >= 3:
        groups = np.array_split(np.arange(n), min(5, n))
        anchors = [float(c[g[-1]]) for g in groups if len(g)]

        if len(anchors) >= 2:
            changes = np.diff(anchors)
            ups = int(np.sum(changes > 0))
            downs = int(np.sum(changes < 0))

            if ups > downs:
                score += min(9.0, (ups - downs) * 3.0)
                reasons.append("Morning-to-current progression is upward")
            elif downs > ups:
                score -= min(9.0, (downs - ups) * 3.0)
                reasons.append("Morning-to-current progression is downward")

    return _clip(score, -45.0, 45.0), reasons, {
        "open": day_open,
        "high": day_high,
        "low": day_low,
        "range": day_range,
        "location": location,
    }


# =========================================================
# 2) RECENT COMPLETED PATH / REVERSAL
# =========================================================

def _recent_path(df):
    """
    Immediate next-candle path detector.
    Uses only COMPLETED candles, so the running candle cannot flip the signal alone.
    """
    base = _completed(df)

    if len(base) < 2:
        return 0.0, ["Recent path is still forming"]

    recent = base.tail(min(6, len(base)))

    c = recent["Close"].to_numpy(float)
    h = recent["High"].to_numpy(float)
    l = recent["Low"].to_numpy(float)

    atr = _atr(base)
    score = 0.0
    reasons = []

    up_closes = sum(c[i] > c[i - 1] for i in range(1, len(c)))
    down_closes = sum(c[i] < c[i - 1] for i in range(1, len(c)))
    higher_lows = sum(l[i] > l[i - 1] for i in range(1, len(l)))
    lower_highs = sum(h[i] < h[i - 1] for i in range(1, len(h)))

    if up_closes >= 3:
        score += 7.0
        reasons.append("Recent completed closes are rising")

    if down_closes >= 3:
        score -= 7.0
        reasons.append("Recent completed closes are falling")

    if higher_lows >= 3:
        score += 6.0
        reasons.append("Recent higher lows show recovery")

    if lower_highs >= 3:
        score -= 6.0
        reasons.append("Recent lower highs show selling pressure")

    recent_move = c[-1] - c[0]

    if recent_move >= max(4.0, atr * 0.60):
        score += 7.0
        reasons.append("Recent completed move is strongly upward")

    elif recent_move <= -max(4.0, atr * 0.60):
        score -= 7.0
        reasons.append("Recent completed move is strongly downward")

    # Last completed candle rejection.
    row = base.iloc[-1]

    o = _safe(row["Open"])
    hh = _safe(row["High"])
    ll = _safe(row["Low"])
    cc = _safe(row["Close"])

    rng = max(hh - ll, 0.01)
    upper = hh - max(o, cc)
    lower = min(o, cc) - ll

    if lower / rng >= 0.42 and cc >= ll + rng * 0.55:
        score += 4.0
        reasons.append("Last completed candle rejected lower prices")

    if upper / rng >= 0.42 and cc <= ll + rng * 0.45:
        score -= 4.0
        reasons.append("Last completed candle rejected higher prices")

    return _clip(score, -25.0, 25.0), reasons


# =========================================================
# 3) CHART SUPPORT / RESISTANCE
# =========================================================

def _cluster_levels(values, tolerance):
    values = sorted(float(v) for v in values)

    groups = []

    for v in values:
        if not groups or abs(v - float(np.mean(groups[-1]))) > tolerance:
            groups.append([v])
        else:
            groups[-1].append(v)

    return [(float(np.mean(g)), len(g)) for g in groups]


def _chart_levels(df, supplied_support, supplied_resistance):
    base = _completed(df)
    current = _safe(df.iloc[-1]["Close"])

    if base.empty:
        return (
            _safe(supplied_support, current),
            _safe(supplied_resistance, current),
            0.0,
            ["Chart S/R is still forming"],
        )

    atr = _atr(base)
    tolerance = max(2.0, atr * 0.42)

    sup_groups = [
        (p, n)
        for p, n in _cluster_levels(base["Low"].to_numpy(float), tolerance)
        if p <= current + tolerance
    ]

    res_groups = [
        (p, n)
        for p, n in _cluster_levels(base["High"].to_numpy(float), tolerance)
        if p >= current - tolerance
    ]

    chart_support = max(
        (p for p, n in sup_groups),
        default=_safe(supplied_support, current - atr),
    )

    chart_resistance = min(
        (p for p, n in res_groups),
        default=_safe(supplied_resistance, current + atr),
    )

    ext_support = _safe(supplied_support, chart_support)
    ext_resistance = _safe(supplied_resistance, chart_resistance)

    if 0 < ext_support < current:
        chart_support = max(chart_support, ext_support)

    if ext_resistance > current:
        chart_resistance = min(chart_resistance, ext_resistance)

    score = 0.0
    reasons = []

    sup_tests = max(
        (n for p, n in sup_groups if abs(p - chart_support) <= tolerance),
        default=0,
    )

    res_tests = max(
        (n for p, n in res_groups if abs(p - chart_resistance) <= tolerance),
        default=0,
    )

    if sup_tests >= 2:
        score += min(10.0, 4.0 + sup_tests)
        reasons.append(f"Repeated chart support reaction ({sup_tests} touches)")

    if res_tests >= 2:
        score -= min(10.0, 4.0 + res_tests)
        reasons.append(f"Repeated chart resistance rejection ({res_tests} touches)")

    # False breakout / breakdown based on last completed candle.
    if len(base) >= 3:
        prior = base.iloc[:-1]
        last = base.iloc[-1]

        prior_high = _safe(prior["High"].max())
        prior_low = _safe(prior["Low"].min())

        lc = _safe(last["Close"])
        lh = _safe(last["High"])
        ll = _safe(last["Low"])

        if lc > prior_high + atr * 0.08:
            score += 10.0
            reasons.append("Completed breakout confirmed")

        elif lh > prior_high and lc <= prior_high:
            score -= 7.0
            reasons.append("False upside breakout rejected")

        if lc < prior_low - atr * 0.08:
            score -= 10.0
            reasons.append("Completed breakdown confirmed")

        elif ll < prior_low and lc >= prior_low:
            score += 7.0
            reasons.append("False downside breakdown reclaimed")

    return (
        float(chart_support),
        float(chart_resistance),
        _clip(score, -25.0, 25.0),
        reasons,
    )


# =========================================================
# 4) OPTION CHAIN / OI
# =========================================================

def _option_score(bull_oc, bear_oc):
    bull_oc = max(_safe(bull_oc), 0.0)
    bear_oc = max(_safe(bear_oc), 0.0)

    total = bull_oc + bear_oc

    if total <= 0:
        return 0.0

    return _clip(((bull_oc - bear_oc) / total) * 25.0, -25.0, 25.0)


def _effective_oi_levels(current, chart_support, chart_resistance, oc_details):
    oc_details = oc_details or {}

    oi_support = _safe(oc_details.get("oi_support"), 0.0)
    oi_resistance = _safe(oc_details.get("oi_resistance"), 0.0)

    support = chart_support
    resistance = chart_resistance

    reasons = []

    # Nearest OI wall becomes part of the path only if it is on the correct side.
    if 0 < oi_support < current:
        if support <= 0 or oi_support > support:
            support = oi_support

        reasons.append(f"OI support wall near {oi_support:.0f}")

    if oi_resistance > current:
        if resistance <= current or oi_resistance < resistance:
            resistance = oi_resistance

        reasons.append(f"OI resistance wall near {oi_resistance:.0f}")

    down_room = current - support if 0 < support < current else 999.0
    up_room = resistance - current if resistance > current else 999.0

    return support, resistance, down_room, up_room, reasons


# =========================================================
# 5) LOW-WEIGHT CONFIRMATION
# =========================================================

def _indicator_score(df):
    row = df.iloc[-1]

    close = _safe(row.get("Close"))
    ema20 = _safe(row.get("EMA20"), close)
    ema50 = _safe(row.get("EMA50"), close)
    vwap = _safe(row.get("VWAP"), close)
    rsi = _safe(row.get("RSI"), 50.0)
    macd = _safe(row.get("MACD"), 0.0)
    signal_line = _safe(row.get("Signal_Line"), 0.0)

    score = 0.0

    if ema20 > ema50:
        score += 1.0
    elif ema20 < ema50:
        score -= 1.0

    if close > vwap:
        score += 1.0
    elif close < vwap:
        score -= 1.0

    if rsi >= 53:
        score += 0.8
    elif rsi <= 47:
        score -= 0.8

    if macd > signal_line:
        score += 0.8
    elif macd < signal_line:
        score -= 0.8

    return _clip(score, -4.0, 4.0)


def _running_score(df):
    """
    Early live-candle path detector.

    The running candle is still NOT allowed to dominate full-session/chart/OI
    context, but it must be strong enough to detect a developing 5-point move
    before candle close.
    """
    if df is None or df.empty:
        return 0.0

    row = df.iloc[-1]
    o = _safe(row.get("Open"))
    h = _safe(row.get("High"))
    l = _safe(row.get("Low"))
    c = _safe(row.get("Close"))

    rng = max(h - l, 0.01)
    body = c - o
    body_abs = abs(body)
    pos = _clip((c - l) / rng, 0.0, 1.0)
    atr = _atr(_completed(df) if len(df) >= 2 else df)

    score = 0.0

    # Direction/body strength.
    if body > 0:
        score += 1.2
    elif body < 0:
        score -= 1.2

    if body_abs >= max(1.5, atr * 0.20):
        score += 0.9 if body > 0 else -0.9

    if body_abs >= max(3.0, atr * 0.38):
        score += 0.8 if body > 0 else -0.8

    # Where price is currently trading inside the live candle.
    if pos >= 0.78:
        score += 0.7
    elif pos <= 0.22:
        score -= 0.7

    # Live break of last completed candle gives an early continuation clue.
    if len(df) >= 2:
        prev = df.iloc[-2]
        ph = _safe(prev.get("High"))
        pl = _safe(prev.get("Low"))
        pc = _safe(prev.get("Close"))

        if c > ph + max(0.5, atr * 0.04):
            score += 1.1
        elif c < pl - max(0.5, atr * 0.04):
            score -= 1.1

        delta = c - pc
        if delta >= max(2.0, atr * 0.22):
            score += 0.8
        elif delta <= -max(2.0, atr * 0.22):
            score -= 0.8

    return _clip(score, -5.5, 5.5)



# =========================================================
# DESTINATION -> REACTION -> DIRECTION
# =========================================================

def _wall_reaction_model(
    df,
    current,
    atr,
    support,
    resistance,
    oc_details,
    market_state,
):
    base = _completed(df)
    oc_details = oc_details or {}

    support_strength = _safe(oc_details.get("support_strength"), 0.0)
    resistance_strength = _safe(oc_details.get("resistance_strength"), 0.0)

    support = _safe(support, current - atr)
    resistance = _safe(resistance, current + atr)

    down_room = current - support if 0 < support < current else 999.0
    up_room = resistance - current if resistance > current else 999.0

    near_distance = max(5.0, atr * 0.75)
    near_support = 0 < down_room <= near_distance
    near_resistance = 0 < up_room <= near_distance

    support_hold = 0.0
    support_break = 0.0
    resistance_hold = 0.0
    resistance_break = 0.0
    reasons = []

    if near_support:
        # Proximity is context, not proof of a bounce.
        support_hold += 1.5 + min(2.0, support_strength * 4.0)
        reasons.append(f"Price is near support ({down_room:.1f} pts)")

    if near_resistance:
        # Proximity is context, not proof of a rejection.
        resistance_hold += 1.5 + min(2.0, resistance_strength * 4.0)
        reasons.append(f"Price is near resistance ({up_room:.1f} pts)")

    if not base.empty:
        recent = base.tail(min(5, len(base)))

        for _, row in recent.iterrows():
            o = _safe(row.get("Open"))
            h = _safe(row.get("High"))
            l = _safe(row.get("Low"))
            c = _safe(row.get("Close"))
            rng = max(h - l, 0.01)

            lower_wick = min(o, c) - l
            upper_wick = h - max(o, c)

            if abs(l - support) <= near_distance:
                if lower_wick / rng >= 0.30 and c > l + rng * 0.45:
                    support_hold += 2.0

            if abs(h - resistance) <= near_distance:
                if upper_wick / rng >= 0.30 and c < l + rng * 0.55:
                    resistance_hold += 2.0

        last2 = base.tail(2)

        support_break_count = int(
            (last2["Close"] < support - atr * 0.10).sum()
        )
        resistance_break_count = int(
            (last2["Close"] > resistance + atr * 0.10).sum()
        )

        if support_break_count == 1:
            support_break += 5.0
        elif support_break_count >= 2:
            support_break += 10.0
            reasons.append("Support breakdown accepted by completed closes")

        if resistance_break_count == 1:
            resistance_break += 5.0
        elif resistance_break_count >= 2:
            resistance_break += 10.0
            reasons.append("Resistance breakout accepted by completed closes")

        closes = recent["Close"].to_numpy(float)
        ranges = (
            recent["High"].to_numpy(float)
            - recent["Low"].to_numpy(float)
        )

        if len(closes) >= 3:
            recent_move = closes[-1] - closes[0]

            if near_support and recent_move < 0:
                support_hold += 2.0

            if near_resistance and recent_move > 0:
                resistance_hold += 2.0

        if len(ranges) >= 4:
            first = float(ranges[:2].mean())
            last = float(ranges[-2:].mean())
            shrinking = first > 0 and last <= first * 0.75

            if shrinking and near_support:
                support_hold += 2.0
                reasons.append("Selling range is shrinking into support")

            if shrinking and near_resistance:
                resistance_hold += 2.0
                reasons.append("Buying range is shrinking into resistance")

    if market_state == "REVERSAL_UP":
        support_hold += 5.0
        reasons.append("REVERSAL_UP favors support hold/bounce")

    elif market_state == "REVERSAL_DOWN":
        resistance_hold += 5.0
        reasons.append("REVERSAL_DOWN favors resistance rejection")

    elif market_state == "BREAKOUT_UP":
        resistance_break += 6.0

    elif market_state == "BREAKOUT_DOWN":
        support_break += 6.0

    elif market_state == "COMPRESSION":
        support_hold += 2.0
        resistance_hold += 2.0
        reasons.append("Compression: wait for accepted break")

    support_hold = max(0.0, support_hold)
    support_break = max(0.0, support_break)
    resistance_hold = max(0.0, resistance_hold)
    resistance_break = max(0.0, resistance_break)

    support_total = support_hold + support_break
    resistance_total = resistance_hold + resistance_break

    support_hold_prob = (
        support_hold / support_total if support_total > 0 else 0.50
    )
    resistance_hold_prob = (
        resistance_hold / resistance_total
        if resistance_total > 0 else 0.50
    )

    if down_room < up_room:
        next_wall = "SUPPORT"
        next_wall_price = support
        wall_distance = down_room
        wall_reaction = (
            "HOLD/BOUNCE"
            if support_hold_prob >= 0.58
            else "BREAK"
            if support_hold_prob <= 0.42
            else "UNCLEAR"
        )
        support_evidence = _clip(support_total / 10.0, 0.20, 0.88)
        wall_reaction_confidence = (
            abs(support_hold_prob - 0.50) * 200 * support_evidence
            if wall_reaction != "UNCLEAR"
            else _clip((support_total / 12.0) * 100.0, 0.0, 57.0)
        )
    else:
        next_wall = "RESISTANCE"
        next_wall_price = resistance
        wall_distance = up_room
        wall_reaction = (
            "HOLD/REJECT"
            if resistance_hold_prob >= 0.58
            else "BREAK"
            if resistance_hold_prob <= 0.42
            else "UNCLEAR"
        )
        resistance_evidence = _clip(resistance_total / 10.0, 0.20, 0.88)
        wall_reaction_confidence = (
            abs(resistance_hold_prob - 0.50) * 200 * resistance_evidence
            if wall_reaction != "UNCLEAR"
            else _clip((resistance_total / 12.0) * 100.0, 0.0, 57.0)
        )

    return {
        "near_support": near_support,
        "near_resistance": near_resistance,
        "support_hold_score": support_hold,
        "support_break_score": support_break,
        "resistance_hold_score": resistance_hold,
        "resistance_break_score": resistance_break,
        "support_hold_prob": support_hold_prob,
        "resistance_hold_prob": resistance_hold_prob,
        "next_wall": next_wall,
        "next_wall_price": float(next_wall_price),
        "wall_distance": float(wall_distance),
        "wall_reaction": wall_reaction,
        "wall_reaction_confidence": float(
            _clip(wall_reaction_confidence, 0.0, 88.0)
        ),
        "reasons": reasons,
    }


# =========================================================
# DESTINATION INTELLIGENCE
# =========================================================
def _destination_intelligence(
    current,
    atr,
    effective_support,
    effective_resistance,
    wall_model,
    market_state,
    chart_score,
    oc_score,
    run_score,
):
    current = _safe(current)
    atr = max(_safe(atr, 5.0), 1.0)
    support = _safe(effective_support, current - atr)
    resistance = _safe(effective_resistance, current + atr)

    down_room = current - support if 0 < support < current else 999.0
    up_room = resistance - current if resistance > current else 999.0
    near = max(5.0, atr * 0.75)

    near_support = 0 < down_room <= near
    near_resistance = 0 < up_room <= near

    support_hold = _safe(wall_model.get("support_hold_score"), 0.0)
    support_break = _safe(wall_model.get("support_break_score"), 0.0)
    resistance_hold = _safe(wall_model.get("resistance_hold_score"), 0.0)
    resistance_break = _safe(wall_model.get("resistance_break_score"), 0.0)

    reasons = []

    support_action = "UNCLEAR"
    if near_support:
        strong_support_hold = (
            support_hold >= 5.5
            and support_hold >= support_break + 2.5
            and run_score >= -1.2
            and oc_score > -12.0
        )
        strong_support_break = (
            support_break >= 5.0
            and support_break >= support_hold + 1.5
            and run_score <= 0.8
        )

        if strong_support_hold:
            support_action = "BOUNCE"
            reasons.append("Nearest support has confirmed HOLD/BOUNCE evidence")
        elif strong_support_break:
            support_action = "BREAK"
            reasons.append("Nearest support has confirmed BREAK evidence")
        else:
            reasons.append("Nearby support is present but not strong enough to force a bounce")

    resistance_action = "UNCLEAR"
    if near_resistance:
        strong_resistance_hold = (
            resistance_hold >= 5.5
            and resistance_hold >= resistance_break + 2.5
            and run_score <= 1.2
            and oc_score < 12.0
        )
        strong_resistance_break = (
            resistance_break >= 5.0
            and resistance_break >= resistance_hold + 1.5
            and run_score >= -0.8
        )

        if strong_resistance_hold:
            resistance_action = "REJECT"
            reasons.append("Nearest resistance has confirmed HOLD/REJECT evidence")
        elif strong_resistance_break:
            resistance_action = "BREAK"
            reasons.append("Nearest resistance has confirmed BREAKOUT evidence")
        else:
            reasons.append("Nearby resistance is present but not strong enough to force a rejection")

    if (
        market_state == "REVERSAL_UP"
        and near_support
        and support_action == "UNCLEAR"
        and support_hold >= 4.5
        and support_hold > support_break
    ):
        support_action = "BOUNCE"
        reasons.append("REVERSAL_UP + wall evidence supports support bounce")

    if (
        market_state == "REVERSAL_DOWN"
        and near_resistance
        and resistance_action == "UNCLEAR"
        and resistance_hold >= 4.5
        and resistance_hold > resistance_break
    ):
        resistance_action = "REJECT"
        reasons.append("REVERSAL_DOWN + wall evidence supports resistance rejection")

    ce_room_ok = up_room >= 4.5
    pe_room_ok = down_room >= 4.5

    ce_blocked = near_resistance and resistance_action == "REJECT" and resistance_break < resistance_hold
    pe_blocked = near_support and support_action == "BOUNCE" and support_break < support_hold

    ce_breakout_ok = (
        near_resistance
        and resistance_action == "BREAK"
        and resistance_break >= resistance_hold
        and run_score >= 0.5
        and chart_score > -8.0
    )

    pe_breakdown_ok = (
        near_support
        and support_action == "BREAK"
        and support_break >= support_hold
        and run_score <= -0.5
        and chart_score < 8.0
    )

    return {
        "near_support": near_support,
        "near_resistance": near_resistance,
        "support_action": support_action,
        "resistance_action": resistance_action,
        "ce_room_ok": ce_room_ok,
        "pe_room_ok": pe_room_ok,
        "ce_blocked": ce_blocked,
        "pe_blocked": pe_blocked,
        "ce_breakout_ok": ce_breakout_ok,
        "pe_breakdown_ok": pe_breakdown_ok,
        "up_room": up_room,
        "down_room": down_room,
        "reasons": reasons,
    }


# =========================================================
# FINAL FORECAST / TRADE
# =========================================================

def _probabilities(score):
    ce = int(round(_clip(50.0 + score * 0.38, 20.0, 80.0)))
    return ce, 100 - ce


def _trade_levels(signal, current, support, resistance, atr):
    target_move = _clip(max(5.0, atr * 0.90), 5.0, 12.0)
    stop_move = _clip(atr * 0.65, 3.0, 8.0)

    if signal == "BUY CE":
        target = current + target_move

        if resistance > current + 5.0:
            target = min(target, resistance - 0.5)

        return current, current - stop_move, target

    if signal == "BUY PE":
        target = current - target_move

        if 0 < support < current - 5.0:
            target = max(target, support + 0.5)

        return current, current + stop_move, target

    return current, current, current


def predict_market(
    df,
    support,
    resistance,
    bull_oc=0.0,
    bear_oc=0.0,
    oc_reasons=None,
    oc_details=None,
):
    df = _clean(df)

    if df.empty:
        return {
            "signal": "WAIT",
            "candidate_signal": "WAIT",
            "next_candle": "UNCLEAR",
            "forecast_confidence": 50,
            "trend": "No Market Data",
            "confidence": 0,
            "entry": 0.0,
            "invalidation": 0.0,
            "target": 0.0,
            "ce_probability": 50,
            "pe_probability": 50,
            "score": 0.0,
            "reasons": ["No usable market candles"],
            "candle_time": None,
        }

    current = _safe(df.iloc[-1]["Close"])
    atr = _atr(df)

    session, session_reasons, session_info = _session_context(df)
    recent, recent_reasons = _recent_path(df)

    chart_support, chart_resistance, chart_score, chart_reasons = (
        _chart_levels(df, support, resistance)
    )

    oc_score = _option_score(bull_oc, bear_oc)

    effective_support, effective_resistance, down_room, up_room, oi_path_reasons = (
        _effective_oi_levels(
            current,
            chart_support,
            chart_resistance,
            oc_details,
        )
    )

    ind_score = _indicator_score(df)
    run_score = _running_score(df)

    market_state, state_bull, state_bear, state_reasons, state_features = (
        classify_market_state(
            df,
            effective_support,
            effective_resistance,
        )
    )

    wall_model = _wall_reaction_model(
        df,
        current,
        atr,
        effective_support,
        effective_resistance,
        oc_details,
        market_state,
    )

    # Fuse live option-chain wall behavior into chart/price wall model.
    oc_support_behavior = str((oc_details or {}).get("support_behavior", "UNKNOWN"))
    oc_resistance_behavior = str((oc_details or {}).get("resistance_behavior", "UNKNOWN"))
    oc_support_hold = _safe((oc_details or {}).get("support_hold_score"), 0.0)
    oc_support_break = _safe((oc_details or {}).get("support_break_score"), 0.0)
    oc_resistance_hold = _safe((oc_details or {}).get("resistance_hold_score"), 0.0)
    oc_resistance_break = _safe((oc_details or {}).get("resistance_break_score"), 0.0)

    wall_model["support_hold_score"] += oc_support_hold
    wall_model["support_break_score"] += oc_support_break
    wall_model["resistance_hold_score"] += oc_resistance_hold
    wall_model["resistance_break_score"] += oc_resistance_break

    sh_total = wall_model["support_hold_score"] + wall_model["support_break_score"]
    rh_total = wall_model["resistance_hold_score"] + wall_model["resistance_break_score"]

    wall_model["support_hold_prob"] = (
        wall_model["support_hold_score"] / sh_total if sh_total > 0 else 0.50
    )
    wall_model["resistance_hold_prob"] = (
        wall_model["resistance_hold_score"] / rh_total if rh_total > 0 else 0.50
    )

    if oc_support_behavior != "UNKNOWN":
        wall_model["reasons"].append(
            f"Option wall support behavior: {oc_support_behavior}"
        )
    if oc_resistance_behavior != "UNKNOWN":
        wall_model["reasons"].append(
            f"Option wall resistance behavior: {oc_resistance_behavior}"
        )

    destination = _destination_intelligence(
        current,
        atr,
        effective_support,
        effective_resistance,
        wall_model,
        market_state,
        chart_score,
        oc_score,
        run_score,
    )

    state_net = _clip(state_bull - state_bear, -20.0, 20.0)

    # Rolling calibration from resolved historical predictions.
    # It never flips direction by itself; it only changes how strict
    # the engine should be in that market state.
    state_reliability = get_state_reliability(market_state)

    # Running candle remains low-weight, but a large move against the proposed
    # direction acts as a VETO. This prevents stale PE/CE calls when the live
    # candle has already invalidated the immediate path.
    completed_now = _completed(df)

    if not completed_now.empty:
        last_completed_close = _safe(completed_now.iloc[-1]["Close"], current)
    else:
        last_completed_close = current

    live_delta = current - last_completed_close
    live_veto_threshold = max(2.0, atr * 0.28)

    # Context stays important, but next-candle path gets a dedicated weight.
    # Session 32 + Chart S/R 22 + Option/OI 22 + Recent completed path 18
    # + Indicators 4 + Running candle 2 = 100.
    score = (
        (session / 45.0) * 28.0
        + (chart_score / 25.0) * 21.0
        + (oc_score / 25.0) * 22.0
        + (recent / 25.0) * 17.0
        + (state_net / 20.0) * 6.0
        + (ind_score / 4.0) * 2.0
        + (run_score / 5.5) * 6.0
    )

    score = _clip(score, -100.0, 100.0)

    ce, pe = _probabilities(score)

    # =====================================================
    # DEDICATED NEXT-CANDLE SCORE
    # =====================================================
    # User priority:
    # Full-day/session 25
    # Full-chart S/R 25
    # Option-chain/OI 25
    # Recent 3-6 completed candles 18
    # Market state 3
    # Running candle 3
    # Indicators 1
    #
    # Running candle and indicators can confirm/veto, but cannot dominate.
    forecast_score = (
        (session / 45.0) * 25.0
        + (chart_score / 25.0) * 25.0
        + (oc_score / 25.0) * 25.0
        + (recent / 25.0) * 18.0
        + (state_net / 20.0) * 3.0
        + (run_score / 5.5) * 3.0
        + (ind_score / 4.0) * 1.0
    )
    forecast_score = _clip(forecast_score, -100.0, 100.0)

    primary_bull_votes = (
        int(session >= 5.0)
        + int(chart_score >= 4.0)
        + int(oc_score >= 3.0)
    )
    primary_bear_votes = (
        int(session <= -5.0)
        + int(chart_score <= -4.0)
        + int(oc_score <= -3.0)
    )

    # ---------------------------
    # NEXT-CANDLE FORECAST — CLEAN DECISION FLOW
    # ---------------------------
    #
    # Priority:
    # 1) Full-session structure
    # 2) Full-chart support/resistance
    # 3) Option-chain/OI
    # 4) Recent 3-6 completed candles
    # 5) Market state
    # 6) Running candle (trigger only)
    # 7) Indicators (confirmation only)

    raw_forecast = "UNCLEAR"
    forecast_basis = "CONFLICT / NO CLEAN PATH"

    # High-priority pillar direction.
    session_bull = session >= 5.0
    session_bear = session <= -5.0
    chart_bull = chart_score >= 4.0
    chart_bear = chart_score <= -4.0
    oi_bull = oc_score >= 3.0
    oi_bear = oc_score <= -3.0

    primary_bull_votes = int(session_bull) + int(chart_bull) + int(oi_bull)
    primary_bear_votes = int(session_bear) + int(chart_bear) + int(oi_bear)

    # Strong versions prevent a weak opposite clue from overriding the full picture.
    strong_primary_bull = (
        int(session >= 10.0)
        + int(chart_score >= 7.0)
        + int(oc_score >= 6.0)
    )
    strong_primary_bear = (
        int(session <= -10.0)
        + int(chart_score <= -7.0)
        + int(oc_score <= -6.0)
    )

    # Real wall reactions.
    support_bounce = (
        destination["near_support"]
        and destination["support_action"] == "BOUNCE"
        and wall_model["support_hold_score"] >= 5.5
        and up_room > 0.5
    )

    resistance_reject = (
        destination["near_resistance"]
        and destination["resistance_action"] == "REJECT"
        and wall_model["resistance_hold_score"] >= 5.5
        and down_room > 0.5
    )

    support_break = (
        destination["near_support"]
        and destination["support_action"] == "BREAK"
        and wall_model["support_break_score"] >= 5.0
    )

    resistance_breakout = (
        destination["near_resistance"]
        and destination["resistance_action"] == "BREAK"
        and wall_model["resistance_break_score"] >= 5.0
    )

    # -----------------------------------------------------
    # 1) WALL REACTION HAS FIRST SAY FOR THE NEXT CANDLE
    # -----------------------------------------------------
    # A genuine support bounce can produce an UP candle even inside a bearish day.
    # A genuine resistance rejection can produce a DOWN candle even inside a bullish day.
    if (
        support_bounce
        and run_score >= 0.5
        and recent >= -6.0
        and strong_primary_bear < 2
    ):
        raw_forecast = "UP"
        forecast_basis = "SUPPORT BOUNCE"

    elif (
        resistance_reject
        and run_score <= -0.5
        and recent <= 6.0
        and strong_primary_bull < 2
    ):
        raw_forecast = "DOWN"
        forecast_basis = "RESISTANCE REJECTION"

    # -----------------------------------------------------
    # 2) ACCEPTED BREAKOUT / BREAKDOWN
    # -----------------------------------------------------
    elif (
        market_state == "BREAKOUT_UP"
        and not resistance_reject
        and (
            recent >= 0.0
            or run_score >= 0.8
            or resistance_breakout
        )
        and primary_bear_votes <= 1
    ):
        raw_forecast = "UP"
        forecast_basis = "BREAKOUT UP CONTINUATION"

    elif (
        market_state == "BREAKOUT_DOWN"
        and not support_bounce
        and (
            recent <= 0.0
            or run_score <= -0.8
            or support_break
        )
        and primary_bull_votes <= 1
    ):
        raw_forecast = "DOWN"
        forecast_basis = "BREAKOUT DOWN CONTINUATION"

    # -----------------------------------------------------
    # 3) RECENT REVERSAL + LOCATION
    # -----------------------------------------------------
    # Recent completed path can override an old full-day trend only when
    # at least one major pillar or the nearest wall supports the reversal.
    elif (
        recent >= 9.0
        and (
            support_bounce
            or chart_bull
            or oi_bull
            or market_state == "REVERSAL_UP"
        )
        and primary_bear_votes <= 1
        and run_score >= -0.6
    ):
        raw_forecast = "UP"
        forecast_basis = "RECENT BULLISH REVERSAL"

    elif (
        recent <= -9.0
        and (
            resistance_reject
            or chart_bear
            or oi_bear
            or market_state == "REVERSAL_DOWN"
        )
        and primary_bull_votes <= 1
        and run_score <= 0.6
    ):
        raw_forecast = "DOWN"
        forecast_basis = "RECENT BEARISH REVERSAL"

    # -----------------------------------------------------
    # 4) HIGH-PRIORITY CONSENSUS
    # -----------------------------------------------------
    elif (
        primary_bull_votes >= 2
        and primary_bear_votes == 0
        and forecast_score >= 8.0
        and recent >= -3.0
        and up_room >= 0.8
        and not resistance_reject
    ):
        raw_forecast = "UP"
        forecast_basis = "SESSION + CHART/OI BULLISH CONSENSUS"

    elif (
        primary_bear_votes >= 2
        and primary_bull_votes == 0
        and forecast_score <= -8.0
        and recent <= 3.0
        and down_room >= 0.8
        and not support_bounce
    ):
        raw_forecast = "DOWN"
        forecast_basis = "SESSION + CHART/OI BEARISH CONSENSUS"

    # -----------------------------------------------------
    # 5) OPEN-PATH CONTINUATION / LOCAL EDGE
    # -----------------------------------------------------
    # This fixes valid 5+ point moves being missed simply because the
    # three major pillars are not perfectly unanimous.
    #
    # It still requires:
    # - at least ONE high-priority pillar in the direction,
    # - recent completed candles not opposing,
    # - live candle only as a trigger,
    # - no confirmed opposite wall reaction.

    open_path_up = (
        up_room >= 4.5
        and not resistance_reject
        and primary_bull_votes >= 1
        and strong_primary_bear <= 1
        and recent >= 4.0
        and run_score >= 0.8
        and forecast_score >= 4.0
    )

    open_path_down = (
        down_room >= 4.5
        and not support_bounce
        and primary_bear_votes >= 1
        and strong_primary_bull <= 1
        and recent <= -4.0
        and run_score <= -0.8
        and forecast_score <= -4.0
    )

    # Strong session continuation can also stay tradable even when chart/OI
    # is neutral, provided recent structure is still aligned and there is room.
    session_continuation_up = (
        session >= 10.0
        and up_room >= 4.5
        and recent >= 1.0
        and run_score >= 0.4
        and primary_bear_votes <= 1
        and not resistance_reject
    )

    session_continuation_down = (
        session <= -10.0
        and down_room >= 4.5
        and recent <= -1.0
        and run_score <= -0.4
        and primary_bull_votes <= 1
        and not support_bounce
    )

    if raw_forecast == "UNCLEAR" and (open_path_up or session_continuation_up):
        raw_forecast = "UP"
        forecast_basis = (
            "OPEN UPSIDE PATH"
            if open_path_up
            else "SESSION CONTINUATION UP"
        )

    if raw_forecast == "UNCLEAR" and (open_path_down or session_continuation_down):
        raw_forecast = "DOWN"
        forecast_basis = (
            "OPEN DOWNSIDE PATH"
            if open_path_down
            else "SESSION CONTINUATION DOWN"
        )

    # -----------------------------------------------------
    # 6) EARLY LIVE TRIGGER — NEVER ALONE
    # -----------------------------------------------------
    # Current candle can trigger an early prediction only when at least
    # one major pillar already agrees and the opposite wall is not blocking.
    elif (
        run_score >= 2.6
        and primary_bull_votes >= 1
        and primary_bear_votes <= 1
        and recent >= -2.0
        and up_room >= 0.8
        and not resistance_reject
        and forecast_score >= 3.0
    ):
        raw_forecast = "UP"
        forecast_basis = "EARLY LIVE BULLISH TRIGGER"

    elif (
        run_score <= -2.6
        and primary_bear_votes >= 1
        and primary_bull_votes <= 1
        and recent <= 2.0
        and down_room >= 0.8
        and not support_bounce
        and forecast_score <= -3.0
    ):
        raw_forecast = "DOWN"
        forecast_basis = "EARLY LIVE BEARISH TRIGGER"

    # -----------------------------------------------------
    # 7) HARD CONTRADICTION / LOCATION VETO
    # -----------------------------------------------------
    # Never call DOWN into a confirmed support bounce.
    if raw_forecast == "DOWN" and support_bounce:
        raw_forecast = "UNCLEAR"
        forecast_basis = "DOWN BLOCKED BY SUPPORT BOUNCE"

    # Never call UP into a confirmed resistance rejection.
    if raw_forecast == "UP" and resistance_reject:
        raw_forecast = "UNCLEAR"
        forecast_basis = "UP BLOCKED BY RESISTANCE REJECTION"

    # Running candle cannot reverse the forecast alone, but a material move
    # directly against the call invalidates that immediate path.
    if raw_forecast == "DOWN" and live_delta >= live_veto_threshold:
        raw_forecast = "UNCLEAR"
        forecast_basis = "DOWN INVALIDATED BY LIVE PRICE"

    if raw_forecast == "UP" and live_delta <= -live_veto_threshold:
        raw_forecast = "UNCLEAR"
        forecast_basis = "UP INVALIDATED BY LIVE PRICE"

    # If the three highest-priority pillars are split and there is no
    # wall/reversal/breakout edge, stay UNCLEAR instead of guessing.
    major_split = primary_bull_votes >= 1 and primary_bear_votes >= 1

    if (
        major_split
        and forecast_basis in (
            "SESSION + CHART/OI BULLISH CONSENSUS",
            "SESSION + CHART/OI BEARISH CONSENSUS",
            "CONFLICT / NO CLEAN PATH",
        )
    ):
        raw_forecast = "UNCLEAR"
        forecast_basis = "MAJOR PILLARS CONFLICT"

    # Keep next-candle confidence tied to the dedicated forecast score
    # and reduce it when the direction comes from an early live trigger.
    forecast_confidence = int(
        round(
            _clip(
                50.0
                + abs(forecast_score) * 0.48
                + min(abs(recent), 12.0) * 0.40,
                50.0,
                84.0,
            )
        )
    )

    if forecast_basis.startswith("EARLY LIVE"):
        forecast_confidence = min(forecast_confidence, 68)

    if raw_forecast == "UNCLEAR":
        forecast_confidence = min(forecast_confidence, 59)

        # Do not silently sit on UNCLEAR: record the blocking reason.
        if resistance_reject:
            forecast_basis = "UNCLEAR: RESISTANCE REJECTION BLOCKS UP"
        elif support_bounce:
            forecast_basis = "UNCLEAR: SUPPORT BOUNCE BLOCKS DOWN"
        elif primary_bull_votes == 0 and primary_bear_votes == 0:
            forecast_basis = "UNCLEAR: NO HIGH-PRIORITY DIRECTION"
        elif primary_bull_votes >= 1 and primary_bear_votes >= 1:
            forecast_basis = "UNCLEAR: HIGH-PRIORITY PILLARS SPLIT"
        elif abs(recent) < 4.0 and abs(run_score) < 0.8:
            forecast_basis = "UNCLEAR: NO LOCAL MOMENTUM YET"

    # State reliability makes low-performing states stricter,
    # never looser than the base threshold.
    reliability_penalty = max(0.0, 0.50 - state_reliability) * 20.0

    # ---------------------------
    # TRADE SIGNAL - STRICTER
    # ---------------------------

    candidate = "WAIT"

    bull_primary_votes = (
        int(session >= 5.0)
        + int(chart_score >= 4.0)
        + int(oc_score >= 3.0)
        + int(recent >= 5.0)
    )

    bear_primary_votes = (
        int(session <= -5.0)
        + int(chart_score <= -4.0)
        + int(oc_score <= -3.0)
        + int(recent <= -5.0)
    )

    # Broad-consensus exceptions used by the trade layer.
    # These are intentionally stricter than the normal path.
    broad_bull_exception = (
        forecast_score >= 26.0
        and primary_bull_votes >= 2
        and primary_bear_votes == 0
        and recent >= -4.0
        and up_room >= 4.5
        and not resistance_reject
    )

    broad_bear_exception = (
        forecast_score <= -26.0
        and primary_bear_votes >= 2
        and primary_bull_votes == 0
        and recent <= 4.0
        and down_room >= 4.5
        and not support_bounce
    )

    # Practical intraday trade trigger:
    # Normal setup: score >= 14 + 2 major votes.
    # Strong local path setup: recent completed structure is strong and
    # either S/R or OI agrees, allowing an earlier 5-point trade.
    strong_up_path = (
        recent >= 8.0
        and (chart_score >= 3.0 or oc_score >= 3.0)
        and up_room >= 4.5
    )

    strong_down_path = (
        recent <= -8.0
        and (chart_score <= -3.0 or oc_score <= -3.0)
        and down_room >= 4.5
    )

    early_trade_up = (
        run_score >= 3.0
        and up_room >= 4.5
        and score >= 9.0
        and (chart_score >= 0.0 or oc_score >= 0.0 or recent >= 3.0)
    )

    early_trade_down = (
        run_score <= -3.0
        and down_room >= 4.5
        and score <= -9.0
        and (chart_score <= 0.0 or oc_score <= 0.0 or recent <= -3.0)
    )

    reversal_trade_up = (
        raw_forecast == "UP"
        and market_state == "REVERSAL_UP"
        and up_room >= 4.5
        and run_score >= 1.2
        and oc_score > -12.0
        and wall_model["support_break_score"] < 8.0
        and not (
            wall_model["near_resistance"]
            and wall_model["resistance_hold_prob"] >= 0.65
            and wall_model["resistance_break_score"] < wall_model["resistance_hold_score"]
        )
    )

    reversal_trade_down = (
        raw_forecast == "DOWN"
        and market_state == "REVERSAL_DOWN"
        and down_room >= 4.5
        and run_score <= -1.2
        and oc_score < 12.0
        and wall_model["resistance_break_score"] < 8.0
        and not (
            wall_model["near_support"]
            and wall_model["support_hold_prob"] >= 0.65
            and wall_model["support_break_score"] < wall_model["support_hold_score"]
        )
    )

    # Reversal trade gets first priority because the old trend can still be
    # bearish/bullish while the local state has already flipped.
    if reversal_trade_up:
        candidate = "BUY CE"

    elif reversal_trade_down:
        candidate = "BUY PE"

    elif (
        raw_forecast == "UP"
        and up_room >= 4.5
        and ce >= 55
        and (
            (
                recent >= 0.0
                and score >= 14.0
                and bull_primary_votes >= 2
            )
            or (
                recent >= 0.0
                and score >= (10.0 + reliability_penalty)
                and strong_up_path
                and bull_primary_votes >= 1
            )
            or early_trade_up
            or broad_bull_exception
        )
    ):
        candidate = "BUY CE"

    elif (
        raw_forecast == "DOWN"
        and down_room >= 4.5
        and pe >= 55
        and (
            (
                recent <= 0.0
                and score <= -14.0
                and bear_primary_votes >= 2
            )
            or (
                recent <= 0.0
                and score <= -(10.0 + reliability_penalty)
                and strong_down_path
                and bear_primary_votes >= 1
            )
            or early_trade_down
            or broad_bear_exception
        )
    ):
        candidate = "BUY PE"


    # =====================================================
    # DIRECT 5-POINT ACTIONABLE PATH
    # =====================================================
    # If next-candle direction is already clear and the market has >=4.5 pts
    # of realistic room, do not require every secondary trade vote to line up.
    # Still blocked by strong support/resistance reaction and opposite OI/chart.
    direct_5pt_up = (
        raw_forecast == "UP"
        and up_room >= 4.5
        and not resistance_reject
        and primary_bear_votes <= 1
        and (
            forecast_basis in (
                "OPEN UPSIDE PATH",
                "SESSION CONTINUATION UP",
                "SUPPORT BOUNCE",
                "BREAKOUT UP CONTINUATION",
                "RECENT BULLISH REVERSAL",
                "SESSION + CHART/OI BULLISH CONSENSUS",
                "EARLY LIVE BULLISH TRIGGER",
            )
            or forecast_score >= 7.0
        )
        and ce >= 55
    )

    direct_5pt_down = (
        raw_forecast == "DOWN"
        and down_room >= 4.5
        and not support_bounce
        and primary_bull_votes <= 1
        and (
            forecast_basis in (
                "OPEN DOWNSIDE PATH",
                "SESSION CONTINUATION DOWN",
                "RESISTANCE REJECTION",
                "BREAKOUT DOWN CONTINUATION",
                "RECENT BEARISH REVERSAL",
                "SESSION + CHART/OI BEARISH CONSENSUS",
                "EARLY LIVE BEARISH TRIGGER",
            )
            or forecast_score <= -7.0
        )
        and pe >= 55
    )

    if candidate == "WAIT" and direct_5pt_up:
        candidate = "BUY CE"
        reasons.append("Direct 5-point upside path qualified")

    if candidate == "WAIT" and direct_5pt_down:
        candidate = "BUY PE"
        reasons.append("Direct 5-point downside path qualified")

    # Direct 5+ point option-wall reaction trade.
    # This is still blocked below if chart wall evidence contradicts it.
    if (
        candidate == "WAIT"
        and raw_forecast == "UP"
        and oc_support_behavior == "HOLD/BOUNCE"
        and wall_model["near_support"]
        and up_room >= 4.5
        and run_score >= 0.8
        and ce >= 54
    ):
        candidate = "BUY CE"

    if (
        candidate == "WAIT"
        and raw_forecast == "DOWN"
        and oc_resistance_behavior == "HOLD/REJECT"
        and wall_model["near_resistance"]
        and down_room >= 4.5
        and run_score <= -0.8
        and pe >= 54
    ):
        candidate = "BUY PE"

    if (
        candidate == "BUY PE"
        and destination["near_support"]
        and destination["support_action"] == "BOUNCE"
        and wall_model["support_hold_score"] >= 5.5
    ):
        candidate = "WAIT"

    if (
        candidate == "BUY CE"
        and destination["near_resistance"]
        and destination["resistance_action"] == "REJECT"
        and wall_model["resistance_hold_score"] >= 5.5
    ):
        candidate = "WAIT"

    # HARD LOCATION GATES
    if candidate == "BUY CE" and destination["ce_blocked"]:
        candidate = "WAIT"

    if candidate == "BUY PE" and destination["pe_blocked"]:
        candidate = "WAIT"

    if candidate == "BUY CE" and not destination["ce_room_ok"]:
        candidate = "WAIT"

    if candidate == "BUY PE" and not destination["pe_room_ok"]:
        candidate = "WAIT"

    # =====================================================
    # FINAL COHERENCE GATE — NO CONTRADICTORY TRADE
    # =====================================================
    # BUY CE only when the next-candle forecast is UP.
    # BUY PE only when the next-candle forecast is DOWN.
    if candidate == "BUY CE" and raw_forecast != "UP":
        candidate = "WAIT"

    if candidate == "BUY PE" and raw_forecast != "DOWN":
        candidate = "WAIT"

    # Strong nearby rejection/bounce always wins over a conflicting trade.
    if (
        candidate == "BUY CE"
        and destination["near_resistance"]
        and destination["resistance_action"] == "REJECT"
    ):
        candidate = "WAIT"

    if (
        candidate == "BUY PE"
        and destination["near_support"]
        and destination["support_action"] == "BOUNCE"
    ):
        candidate = "WAIT"

    # Never take a trade with less than 5 NIFTY points of realistic room.
    if candidate == "BUY CE" and up_room < 4.5:
        candidate = "WAIT"

    if candidate == "BUY PE" and down_room < 4.5:
        candidate = "WAIT"

    # Weak-confidence trades are filtered out.
    # Forecast can still be shown, but trade remains WAIT.
    high_quality_fast_path = bool(
        reversal_trade_up
        or reversal_trade_down
        or strong_up_path
        or strong_down_path
        or early_trade_up
        or early_trade_down
    )

    if direct_5pt_up or direct_5pt_down:
        min_trade_probability = 55
    elif high_quality_fast_path:
        min_trade_probability = 58
    else:
        min_trade_probability = 60

    if candidate == "BUY CE" and ce < min_trade_probability:
        candidate = "WAIT"

    if candidate == "BUY PE" and pe < min_trade_probability:
        candidate = "WAIT"

    # Direction-aligned probabilities.
    # They remain confidence-style estimates, not guarantees.
    if raw_forecast == "UP":
        ce = max(ce, 56)
        if market_state == "REVERSAL_UP":
            ce = max(ce, 60)
        pe = 100 - ce
    elif raw_forecast == "DOWN":
        pe = max(pe, 56)
        if market_state == "REVERSAL_DOWN":
            pe = max(pe, 60)
        ce = 100 - pe

    entry, invalidation, target = _trade_levels(
        candidate,
        current,
        effective_support,
        effective_resistance,
        atr,
    )

    confidence = int(round(_clip(48.0 + abs(score) * 0.48, 44.0, 86.0)))

    if candidate == "WAIT":
        confidence = int(_clip(42.0 + abs(score) * 0.23, 42.0, 62.0))

    if market_state == "REVERSAL_UP" and raw_forecast == "UP":
        trend = "Bullish Reversal Setup"
    elif market_state == "REVERSAL_DOWN" and raw_forecast == "DOWN":
        trend = "Bearish Reversal Setup"
    elif score >= 14:
        trend = "Bullish Setup"
    elif score <= -14:
        trend = "Bearish Setup"
    elif score >= 8:
        trend = "Mild Bullish / WAIT"
    elif score <= -8:
        trend = "Mild Bearish / WAIT"
    else:
        trend = "Mixed / WAIT"

    reasons = [
        f"All available session candles used: {len(df)}",
        "Primary context: full session + chart S/R + option-chain/OI",
        "Next-candle guard: recent completed path + minimum 5-point room",
    ]

    reasons += [f"Session: {r}" for r in session_reasons[:7]]
    reasons += [f"Recent Path: {r}" for r in recent_reasons[:6]]
    reasons += [f"Chart S/R: {r}" for r in chart_reasons[:6]]
    reasons += [f"OI Path: {r}" for r in oi_path_reasons[:4]]

    for r in list(oc_reasons or [])[:8]:
        reasons.append(f"Option Chain: {r}")

    reasons += [f"Market State: {r}" for r in state_reasons[:6]]
    reasons += [f"Wall Model: {r}" for r in wall_model["reasons"][:8]]
    reasons += [f"Destination: {r}" for r in destination["reasons"][:6]]
    reasons.append(
        f"Next-candle score {forecast_score:.1f} | "
        f"Primary votes B{primary_bull_votes}/S{primary_bear_votes} | "
        f"Recent {recent:.1f} | Live {run_score:.1f}"
    )
    reasons.append(f"Next-candle basis: {forecast_basis}")
    reasons.append(
        f"Forecast path room U:{up_room:.1f}/D:{down_room:.1f} pts "
        f"(trade still requires 4.5 pts)"
    )

    reasons.append(
        f"Wall evidence: support hold {wall_model['support_hold_score']:.1f} / "
        f"break {wall_model['support_break_score']:.1f} | "
        f"resistance hold {wall_model['resistance_hold_score']:.1f} / "
        f"break {wall_model['resistance_break_score']:.1f}"
    )

    reasons.append(
        f"Option Wall: support={oc_support_behavior} | "
        f"resistance={oc_resistance_behavior}"
    )

    if strong_up_path:
        reasons.append("Early CE path: recent recovery + S/R/OI agreement + 5pt room")

    if strong_down_path:
        reasons.append("Early PE path: recent selloff + S/R/OI agreement + 5pt room")

    reasons.append(
        f"Room: UP {up_room:.1f} pts | DOWN {down_room:.1f} pts"
    )

    if raw_forecast == "UNCLEAR" and score <= -10.0 and recent > 0.0:
        reasons.append(
            "DOWN blocked: recent completed path is recovering"
        )

    if score <= -10.0 and live_delta >= live_veto_threshold:
        reasons.append(
            f"DOWN blocked: running candle recovered {live_delta:.1f} pts"
        )

    if score >= 10.0 and live_delta <= -live_veto_threshold:
        reasons.append(
            f"UP blocked: running candle sold off {abs(live_delta):.1f} pts"
        )

    if raw_forecast == "UNCLEAR" and score >= 10.0 and recent < 0.0:
        reasons.append(
            "UP blocked: recent completed path is selling off"
        )

    reasons.append(
        f"State {market_state} ({state_reliability:.0%}) | "
        f"Score {score:.1f} | Recent {recent:.1f} | "
        f"CE {ce}% / PE {pe}% | Session {session:.1f} | "
        f"Chart {chart_score:.1f} | OC {oc_score:.1f} | Live {run_score:.1f}"
    )

    return {
        "signal": candidate,
        "candidate_signal": candidate,
        "next_candle": raw_forecast,
        "forecast_confidence": int(forecast_confidence),
        "trend": trend,
        "confidence": int(confidence),
        "entry": float(entry),
        "invalidation": float(invalidation),
        "target": float(target),
        "ce_probability": int(ce),
        "pe_probability": int(pe),
        "score": float(score),
        "forecast_score": float(forecast_score),
        "forecast_basis": forecast_basis,
        "direct_5pt_up": bool(direct_5pt_up),
        "direct_5pt_down": bool(direct_5pt_down),
        "reasons": reasons,
        "candle_time": df.index[-1],
        "support": float(effective_support),
        "resistance": float(effective_resistance),
        "up_room": float(up_room),
        "down_room": float(down_room),
        "atr": float(atr),
        "market_state": market_state,
        "state_reliability": float(state_reliability),
        "next_wall": wall_model["next_wall"],
        "next_wall_price": float(wall_model["next_wall_price"]),
        "wall_distance": float(wall_model["wall_distance"]),
        "wall_reaction": wall_model["wall_reaction"],
        "wall_reaction_confidence": float(
            wall_model["wall_reaction_confidence"]
        ),
        "destination_support_action": destination["support_action"],
        "destination_resistance_action": destination["resistance_action"],
        "destination_ce_blocked": bool(destination["ce_blocked"]),
        "destination_pe_blocked": bool(destination["pe_blocked"]),
        "safe_to_trade": bool(candidate in ("BUY CE", "BUY PE")),
        "fast_setup": bool(
            reversal_trade_up
            or reversal_trade_down
            or early_trade_up
            or early_trade_down
            or (
                raw_forecast == "UP"
                and oc_support_behavior == "HOLD/BOUNCE"
                and wall_model["near_support"]
                and up_room >= 4.5
            )
            or (
                raw_forecast == "DOWN"
                and oc_resistance_behavior == "HOLD/REJECT"
                and wall_model["near_resistance"]
                and down_room >= 4.5
            )
        ),
    }


def predict_market_smart(
    df,
    support,
    resistance,
    bull_oc=0.0,
    bear_oc=0.0,
    oc_reasons=None,
    oc_details=None,
):
    return predict_market(
        df,
        support,
        resistance,
        bull_oc,
        bear_oc,
        oc_reasons,
        oc_details,
    )