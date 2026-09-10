import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


MIN_TRADE_ROOM = 5.0


def _f(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _col(df: pd.DataFrame, name: str) -> Optional[str]:
    for candidate in (name, name.lower(), name.upper(), name.capitalize()):
        if candidate in df.columns:
            return candidate
    return None


def _series(df: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    c = _col(df, name)
    if c is None:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[c], errors="coerce")


def _clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = _series(df, "High"), _series(df, "Low"), _series(df, "Close")
    prev = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    val = tr.rolling(period, min_periods=3).mean().iloc[-1]
    if not np.isfinite(val) or val <= 0:
        val = tr.tail(max(3, period)).mean()
    return max(_f(val, 5.0), 1.0)


def _candle(row: pd.Series, atr: float) -> Dict[str, float]:
    o, h, l, c = map(_f, [row["Open"], row["High"], row["Low"], row["Close"]])
    rng = max(h-l, 0.01)
    body = abs(c-o)
    upper = h-max(o,c)
    lower = min(o,c)-l
    return {
        "open": o, "high": h, "low": l, "close": c, "range": rng, "body": body,
        "upper": upper, "lower": lower, "bull": float(c > o), "bear": float(c < o),
        "body_ratio": body/rng, "upper_ratio": upper/rng, "lower_ratio": lower/rng,
        "strong_bull": float(c > o and body >= max(0.45*rng, 0.25*atr)),
        "strong_bear": float(c < o and body >= max(0.45*rng, 0.25*atr)),
    }


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for n in ("Open", "High", "Low", "Close", "Volume"):
        out[n] = _series(df, n, 0.0)
    for n in ("EMA20", "EMA50", "EMA200", "VWAP", "RSI", "MACD", "Signal_Line", "MACD_Signal"):
        c = _col(df, n)
        if c is not None:
            out[n] = pd.to_numeric(df[c], errors="coerce")
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def _cluster_levels(values: List[float], tolerance: float) -> List[Dict[str, float]]:
    vals = sorted(v for v in values if np.isfinite(v))
    clusters: List[List[float]] = []
    for v in vals:
        if not clusters or abs(v - np.mean(clusters[-1])) > tolerance:
            clusters.append([v])
        else:
            clusters[-1].append(v)
    return [
        {"price": float(np.mean(x)), "tests": float(len(x)), "spread": float(max(x)-min(x))}
        for x in clusters if len(x) >= 2
    ]


def _session_context(df: pd.DataFrame, atr: float, external_support: float, external_resistance: float) -> Dict[str, Any]:
    # Full completed-session structure. Repeated reaction clusters matter more than one isolated wick.
    h, l, c = df["High"], df["Low"], df["Close"]
    current = _f(c.iloc[-1])
    day_high, day_low = _f(h.max()), _f(l.min())
    day_range = max(day_high-day_low, 1.0)
    pos = _clip((current-day_low)/day_range, 0.0, 1.0)
    tol = max(2.0, min(0.55*atr, 7.0))

    highs, lows = [], []
    for i in range(2, len(df)-2):
        if h.iloc[i] >= h.iloc[i-2:i+3].max():
            highs.append(_f(h.iloc[i]))
        if l.iloc[i] <= l.iloc[i-2:i+3].min():
            lows.append(_f(l.iloc[i]))
    high_clusters = _cluster_levels(highs, tol)
    low_clusters = _cluster_levels(lows, tol)

    repeated_res = [x for x in high_clusters if x["price"] >= current-0.8*atr]
    repeated_sup = [x for x in low_clusters if x["price"] <= current+0.8*atr]
    best_res = min(repeated_res, key=lambda x: abs(x["price"]-current), default=None)
    best_sup = min(repeated_sup, key=lambda x: abs(x["price"]-current), default=None)

    res_candidates = [day_high]
    sup_candidates = [day_low]
    if external_resistance > current: res_candidates.append(external_resistance)
    if external_support < current: sup_candidates.append(external_support)
    if best_res: res_candidates.append(best_res["price"])
    if best_sup: sup_candidates.append(best_sup["price"])
    resistance = min((x for x in res_candidates if x >= current), default=day_high)
    support = max((x for x in sup_candidates if x <= current), default=day_low)

    # Trend efficiency distinguishes directional session from a large but choppy range.
    net = _f(c.iloc[-1] - c.iloc[0])
    path = _f(c.diff().abs().sum(), 1.0)
    efficiency = abs(net)/max(path, 1.0)
    recent = c.tail(min(18, len(c)))
    recent_net = _f(recent.iloc[-1]-recent.iloc[0]) if len(recent) > 1 else 0.0

    regime = "RANGE"
    if efficiency > 0.22 and net > 0.8*atr and pos > 0.55:
        regime = "TREND_UP"
    elif efficiency > 0.22 and net < -0.8*atr and pos < 0.45:
        regime = "TREND_DOWN"
    elif recent_net > 1.6*atr and net <= 0:
        regime = "REVERSAL_UP"
    elif recent_net < -1.6*atr and net >= 0:
        regime = "REVERSAL_DOWN"

    return {
        "day_high": day_high, "day_low": day_low, "day_range": day_range, "range_position": pos,
        "session_net": net, "efficiency": efficiency, "regime": regime,
        "support": support, "resistance": resistance,
        "repeated_resistance": best_res, "repeated_support": best_sup,
        "high_clusters": high_clusters, "low_clusters": low_clusters,
    }


def _structure(df: pd.DataFrame, atr: float) -> Dict[str, Any]:
    n = min(24, len(df))
    x = df.tail(n)
    highs, lows = x["High"].values, x["Low"].values
    swing_h, swing_l = [], []
    for i in range(1, len(x)-1):
        if highs[i] > highs[i-1] and highs[i] >= highs[i+1]: swing_h.append(float(highs[i]))
        if lows[i] < lows[i-1] and lows[i] <= lows[i+1]: swing_l.append(float(lows[i]))
    hh = len(swing_h) >= 2 and swing_h[-1] > swing_h[-2] + 0.12*atr
    lh = len(swing_h) >= 2 and swing_h[-1] < swing_h[-2] - 0.12*atr
    hl = len(swing_l) >= 2 and swing_l[-1] > swing_l[-2] + 0.12*atr
    ll = len(swing_l) >= 2 and swing_l[-1] < swing_l[-2] - 0.12*atr
    bull = int(hh)+int(hl)
    bear = int(lh)+int(ll)
    label = "BULLISH" if bull > bear else "BEARISH" if bear > bull else "MIXED"
    return {"label": label, "hh": hh, "hl": hl, "lh": lh, "ll": ll, "bull": bull, "bear": bear}


def _level_reaction(df: pd.DataFrame, support: float, resistance: float, atr: float) -> Dict[str, Any]:
    recent = df.tail(min(6, len(df)))
    last = _candle(recent.iloc[-1], atr)
    tol = max(1.5, 0.32*atr)
    r_tests = 0; s_tests = 0; r_failed_closes = 0; s_failed_closes = 0
    for _, row in recent.iterrows():
        cd = _candle(row, atr)
        if cd["high"] >= resistance-tol:
            r_tests += 1
            if cd["close"] < resistance-tol*0.20: r_failed_closes += 1
        if cd["low"] <= support+tol:
            s_tests += 1
            if cd["close"] > support+tol*0.20: s_failed_closes += 1

    near_r = resistance-last["close"] <= max(0.75*atr, 4.0)
    near_s = last["close"]-support <= max(0.75*atr, 4.0)
    upper_rejection = near_r and (last["upper_ratio"] >= 0.34 or (last["bear"] and last["high"] >= resistance-tol))
    lower_rejection = near_s and (last["lower_ratio"] >= 0.34 or (last["bull"] and last["low"] <= support+tol))
    failed_breakout = last["high"] > resistance and last["close"] < resistance-0.08*atr
    failed_breakdown = last["low"] < support and last["close"] > support+0.08*atr

    prev_close = _f(recent["Close"].iloc[-2]) if len(recent) >= 2 else last["close"]
    breakout_accept = last["close"] > resistance + max(0.12*atr, 1.2) and prev_close >= resistance-0.10*atr
    breakdown_accept = last["close"] < support - max(0.12*atr, 1.2) and prev_close <= support+0.10*atr

    return {
        "upper_rejection": bool(upper_rejection), "lower_rejection": bool(lower_rejection),
        "failed_breakout": bool(failed_breakout), "failed_breakdown": bool(failed_breakdown),
        "breakout_accept": bool(breakout_accept), "breakdown_accept": bool(breakdown_accept),
        "resistance_tests": r_tests, "support_tests": s_tests,
        "resistance_failed_closes": r_failed_closes, "support_failed_closes": s_failed_closes,
        "last": last,
    }


def _extension(df: pd.DataFrame, atr: float, session: Dict[str, Any]) -> Dict[str, Any]:
    close = _f(df["Close"].iloc[-1])
    up_dist = 0.0; down_dist = 0.0
    horizon_used = 0
    for n in (6, 10, 15, 24):
        x = df.tail(min(n, len(df)))
        if len(x) < 3: continue
        up = close-_f(x["Low"].min())
        down = _f(x["High"].max())-close
        if up > up_dist: up_dist, horizon_used = up, n
        if down > down_dist: down_dist = down
    near_top = session["day_high"]-close <= max(0.8*atr, 6.0) or session["range_position"] >= 0.78
    near_bottom = close-session["day_low"] <= max(0.8*atr, 6.0) or session["range_position"] <= 0.22
    late_up = up_dist >= max(2.2*atr, 24.0) and near_top
    late_down = down_dist >= max(2.2*atr, 24.0) and near_bottom
    return {"up_distance": up_dist, "down_distance": down_dist, "late_up": late_up, "late_down": late_down, "horizon": horizon_used}


def _oi_context(details: Optional[Dict[str, Any]], price: float) -> Dict[str, Any]:
    d = details or {}
    oi_sup = _f(d.get("oi_support"), 0.0)
    oi_res = _f(d.get("oi_resistance"), 0.0)
    s_hold = _f(d.get("support_hold_score"), 0.0); s_break = _f(d.get("support_break_score"), 0.0)
    r_hold = _f(d.get("resistance_hold_score"), 0.0); r_break = _f(d.get("resistance_break_score"), 0.0)
    sb = str(d.get("support_behavior", "UNKNOWN")).upper()
    rb = str(d.get("resistance_behavior", "UNKNOWN")).upper()
    support_holding = "HOLD" in sb or "BOUNCE" in sb or s_hold > s_break+0.5
    support_breaking = "BREAK" in sb and "BOUNCE" not in sb or s_break > s_hold+0.75
    resistance_holding = "HOLD" in rb or "REJECT" in rb or r_hold > r_break+0.5
    resistance_breaking = "BREAKOUT" in rb or r_break > r_hold+0.75
    return {
        "support": oi_sup, "resistance": oi_res,
        "support_holding": support_holding, "support_breaking": support_breaking,
        "resistance_holding": resistance_holding, "resistance_breaking": resistance_breaking,
        "support_behavior": sb, "resistance_behavior": rb,
        "support_hold": s_hold, "support_break": s_break, "resistance_hold": r_hold, "resistance_break": r_break,
    }


def _indicator_bias(df: pd.DataFrame) -> Tuple[float, List[str]]:
    # Indicators are confirmation only; they must never overpower price/level reaction.
    row = df.iloc[-1]; score = 0.0; reasons = []
    close = _f(row.get("Close"))
    ema20 = _f(row.get("EMA20"), close); ema50 = _f(row.get("EMA50"), close); vwap = _f(row.get("VWAP"), close)
    rsi = _f(row.get("RSI"), 50.0); macd = _f(row.get("MACD")); sig = _f(row.get("Signal_Line", row.get("MACD_Signal", 0.0)))
    if close > ema20 and ema20 >= ema50: score += 0.7
    elif close < ema20 and ema20 <= ema50: score -= 0.7
    if close > vwap: score += 0.35
    elif close < vwap: score -= 0.35
    if rsi >= 56: score += 0.35
    elif rsi <= 44: score -= 0.35
    if macd > sig: score += 0.25
    elif macd < sig: score -= 0.25
    if abs(score) >= 0.8: reasons.append("Indicators confirm bullish pressure" if score > 0 else "Indicators confirm bearish pressure")
    return score, reasons


def _next_destination(price: float, direction: str, session: Dict[str, Any], oi: Dict[str, Any], external_support: float, external_resistance: float, breakout: bool = False) -> Tuple[float, float, str]:
    levels: List[Tuple[float, str]] = []
    if direction == "UP":
        for p, name in [(external_resistance, "chart resistance"), (oi["resistance"], "OI call wall"), (session["day_high"], "day high")]:
            if p > price+0.2: levels.append((p, name))
        for x in session["high_clusters"]:
            if x["price"] > price+0.2: levels.append((x["price"], "repeated resistance"))
        if not levels:
            return price+max(5.0, session["day_range"]*0.12), max(5.0, session["day_range"]*0.12), "open upside"
        p, name = min(levels, key=lambda z: z[0])
        return p, p-price, name
    for p, name in [(external_support, "chart support"), (oi["support"], "OI put wall"), (session["day_low"], "day low")]:
        if p > 0 and p < price-0.2: levels.append((p, name))
    for x in session["low_clusters"]:
        if x["price"] < price-0.2: levels.append((x["price"], "repeated support"))
    if not levels:
        return price-max(5.0, session["day_range"]*0.12), max(5.0, session["day_range"]*0.12), "open downside"
    p, name = max(levels, key=lambda z: z[0])
    return p, price-p, name


def predict_market(
    df,
    support,
    resistance,
    bull_oc=0.0,
    bear_oc=0.0,
    oc_reasons=None,
    oc_details=None,
):
    oc_reasons = oc_reasons or []
    if df is None or len(df) < 20:
        return {
            "signal": "WAIT", "candidate_signal": "WAIT", "safe_to_trade": False,
            "next_candle": "WAIT", "forecast_ready": False, "forecast_confidence": 0,
            "trend": "Not Enough Data", "confidence": 0, "entry": None, "invalidation": None, "target": None,
            "ce_probability": 50, "pe_probability": 50, "score": 0, "reasons": ["Waiting for enough session candles"],
            "market_state": "NOT_ENOUGH_DATA", "state_reliability": 0,
            "support": _f(support), "resistance": _f(resistance), "up_room": 0, "down_room": 0,
            "breakout_room": 0, "breakdown_room": 0, "atr": 0,
            "next_wall": "NONE", "next_wall_price": None, "wall_distance": 0,
            "wall_reaction": "UNKNOWN", "wall_reaction_confidence": 0,
            "destination_support_action": "UNKNOWN", "destination_resistance_action": "UNKNOWN",
            "destination_pe_blocked": False, "destination_ce_blocked": False,
            "pressure_break_down": 0, "pressure_break_up": 0, "fast_setup": False,
        }

    x = _normalize(df)
    if len(x) < 20:
        return predict_market(None, support, resistance, bull_oc, bear_oc, oc_reasons, oc_details)

    atr = _atr(x)
    price = _f(x["Close"].iloc[-1])
    ext_support, ext_resistance = _f(support, price-atr), _f(resistance, price+atr)
    session = _session_context(x, atr, ext_support, ext_resistance)
    structure = _structure(x, atr)
    oi = _oi_context(oc_details, price)

    # Merge chart and OI walls, but only walls on the correct side of current price.
    effective_supports = [p for p in (ext_support, session["support"], oi["support"]) if p > 0 and p <= price]
    effective_res = [p for p in (ext_resistance, session["resistance"], oi["resistance"]) if p >= price]
    eff_support = max(effective_supports, default=session["day_low"])
    eff_resistance = min(effective_res, default=session["day_high"])

    reaction = _level_reaction(x, eff_support, eff_resistance, atr)
    extension = _extension(x, atr, session)
    ind_score, ind_reasons = _indicator_bias(x)

    bull = 0.0; bear = 0.0; reasons: List[str] = []

    # 1) Full-day regime, but not blindly: active structure/candles can override it.
    if session["regime"] == "TREND_UP": bull += 1.5; reasons.append("Full-session structure is upward")
    elif session["regime"] == "TREND_DOWN": bear += 1.5; reasons.append("Full-session structure is downward")
    elif session["regime"] == "REVERSAL_UP": bull += 1.7; reasons.append("Active regime has reversed upward")
    elif session["regime"] == "REVERSAL_DOWN": bear += 1.7; reasons.append("Active regime has reversed downward")
    else: reasons.append("Session is behaving like a range/chop market")

    if structure["label"] == "BULLISH": bull += 1.7; reasons.append("Recent swing structure is HH/HL biased")
    elif structure["label"] == "BEARISH": bear += 1.7; reasons.append("Recent swing structure is LH/LL biased")

    # 2) Candle-by-candle behavior at the actual destination wall.
    if reaction["failed_breakout"]:
        bear += 3.2; bull -= 2.4; reasons.append("Failed breakout: price pierced resistance but closed back below it")
    elif reaction["upper_rejection"]:
        bear += 2.2; bull -= 1.5; reasons.append("Resistance rejection visible in the latest completed candle")
    if reaction["failed_breakdown"]:
        bull += 3.2; bear -= 2.4; reasons.append("Failed breakdown: price pierced support and reclaimed it")
    elif reaction["lower_rejection"]:
        bull += 2.2; bear -= 1.5; reasons.append("Support bounce/rejection visible in the latest completed candle")

    if reaction["resistance_tests"] >= 2 and reaction["resistance_failed_closes"] >= 2 and not reaction["breakout_accept"]:
        bear += 1.6; bull -= 1.2; reasons.append("Repeated candles are failing to accept above the same resistance zone")
    if reaction["support_tests"] >= 2 and reaction["support_failed_closes"] >= 2 and not reaction["breakdown_accept"]:
        bull += 1.6; bear -= 1.2; reasons.append("Repeated candles are holding/reclaiming the same support zone")

    # 3) Accepted breakout/breakdown beats a static wall.
    if reaction["breakout_accept"]:
        bull += 3.0; bear -= 1.5; reasons.append("Completed-candle breakout acceptance above resistance")
    if reaction["breakdown_accept"]:
        bear += 3.0; bull -= 1.5; reasons.append("Completed-candle breakdown acceptance below support")

    # 4) Option chain = wall intelligence, never a standalone reversal trigger.
    oc_delta = _clip(_f(bull_oc)-_f(bear_oc), -5.0, 5.0)
    bull += max(0.0, oc_delta)*0.45; bear += max(0.0, -oc_delta)*0.45
    if oi["resistance_holding"] and eff_resistance-price <= max(atr, 8.0):
        bull -= 1.8; bear += 0.7; reasons.append("Option-chain call wall is holding/rejecting near price")
    if oi["resistance_breaking"]:
        bull += 1.7; reasons.append("Option-chain resistance is weakening/breaking")
    if oi["support_holding"] and price-eff_support <= max(atr, 8.0):
        bear -= 1.8; bull += 0.7; reasons.append("Option-chain put wall is holding/bouncing near price")
    if oi["support_breaking"]:
        bear += 1.7; reasons.append("Option-chain support is weakening/breaking")

    # 5) Range-location intelligence. This is the specific repeated CE/PE fix.
    range_top = session["range_position"] >= 0.78
    range_bottom = session["range_position"] <= 0.22
    if session["regime"] == "RANGE" and range_top and not reaction["breakout_accept"]:
        bull -= 2.2; reasons.append("Price is in the upper part of the established intraday range without breakout acceptance")
    if session["regime"] == "RANGE" and range_bottom and not reaction["breakdown_accept"]:
        bear -= 2.2; reasons.append("Price is in the lower part of the established intraday range without breakdown acceptance")

    # 6) Stronger multi-horizon late/chase detector.
    late_ce_block = False; late_pe_block = False
    if extension["late_up"]:
        bull -= 2.0; reasons.append(f"Upside move is already extended ({extension['up_distance']:.1f} pts from a recent swing low)")
        if reaction["upper_rejection"] or reaction["failed_breakout"] or (range_top and not reaction["breakout_accept"]):
            late_ce_block = True; bull -= 2.5; reasons.append("Fresh CE blocked: extended recovery has reached rejection/range-top territory")
    if extension["late_down"]:
        bear -= 2.0; reasons.append(f"Downside move is already extended ({extension['down_distance']:.1f} pts from a recent swing high)")
        if reaction["lower_rejection"] or reaction["failed_breakdown"] or (range_bottom and not reaction["breakdown_accept"]):
            late_pe_block = True; bear -= 2.5; reasons.append("Fresh PE blocked: extended fall has reached bounce/range-bottom territory")

    # A fresh continuation after extension needs actual new acceptance, no
