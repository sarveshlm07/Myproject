import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd

__all__ = ["predict_market", "predict_market_smart"]
MIN_TRADE_ROOM = 5.0


def predict_market(
    df,
    support,
    resistance,
    bull_oc=0.0,
    bear_oc=0.0,
    oc_reasons=None,
    oc_details=None,
):
    """Public engine interface used by app.py. Always returns a dict."""
    try:
        return _predict_market(
            df=df,
            support=support,
            resistance=resistance,
            bull_oc=bull_oc,
            bear_oc=bear_oc,
            oc_reasons=oc_reasons,
            oc_details=oc_details,
        )
    except Exception as exc:
        return _fallback(support, resistance, f"Engine safety fallback: {type(exc).__name__}")


def predict_market_smart(*args, **kwargs):
    return predict_market(*args, **kwargs)


def _safe(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _clean(df) -> pd.DataFrame:
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
        if k == "open": rename[c] = "Open"
        elif k == "high": rename[c] = "High"
        elif k == "low": rename[c] = "Low"
        elif k == "close": rename[c] = "Close"
        elif k == "volume": rename[c] = "Volume"
    x.rename(columns=rename, inplace=True)
    req = ["Open", "High", "Low", "Close"]
    if not all(c in x.columns for c in req):
        return pd.DataFrame()
    for c in req:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x.dropna(subset=req, inplace=True)
    if isinstance(x.index, pd.DatetimeIndex):
        x.sort_index(inplace=True)
    return x


def _completed(x: pd.DataFrame) -> pd.DataFrame:
    return x.iloc[:-1].copy() if len(x) >= 2 else x.iloc[0:0].copy()


def _atr(x: pd.DataFrame, n: int = 14) -> float:
    if len(x) < 2:
        return 5.0
    h, l, c = x["High"], x["Low"], x["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return max(_safe(tr.tail(n).mean(), 5.0), 1.0)


def _candle(row: pd.Series, atr: float) -> Dict[str, float]:
    o, h, l, c = [_safe(row[k]) for k in ("Open", "High", "Low", "Close")]
    rng = max(h-l, 0.01)
    body = abs(c-o)
    return {
        "o": o, "h": h, "l": l, "c": c,
        "body": body, "range": rng,
        "upper": h-max(o,c), "lower": min(o,c)-l,
        "bull": c > o, "bear": c < o,
        "upper_ratio": (h-max(o,c))/rng,
        "lower_ratio": (min(o,c)-l)/rng,
        "body_ratio": body/rng,
        "strong_bull": c > o and body >= max(0.45*rng, 0.22*atr),
        "strong_bear": c < o and body >= max(0.45*rng, 0.22*atr),
    }


def _recent_swings(base: pd.DataFrame, atr: float) -> Dict[str, Any]:
    if len(base) < 5:
        return {"swing_highs": [], "swing_lows": [], "repeated_high": None, "repeated_low": None}
    highs, lows = [], []
    h, l = base["High"].to_numpy(float), base["Low"].to_numpy(float)
    for i in range(2, len(base)-2):
        if h[i] >= np.max(h[i-2:i+3]): highs.append(float(h[i]))
        if l[i] <= np.min(l[i-2:i+3]): lows.append(float(l[i]))
    tol = max(2.0, min(0.6*atr, 7.0))
    def repeated(vals):
        if len(vals) < 2: return None
        vals = sorted(vals)
        groups = [[vals[0]]]
        for v in vals[1:]:
            if abs(v - np.mean(groups[-1])) <= tol: groups[-1].append(v)
            else: groups.append([v])
        groups = [g for g in groups if len(g) >= 2]
        if not groups: return None
        g = max(groups, key=len)
        return float(np.mean(g))
    return {"swing_highs": highs, "swing_lows": lows, "repeated_high": repeated(highs), "repeated_low": repeated(lows)}


def _oi_context(oc_details, spot: float) -> Dict[str, Any]:
    d = oc_details if isinstance(oc_details, dict) else {}
    sup = _safe(d.get("oi_support"), 0.0)
    res = _safe(d.get("oi_resistance"), 0.0)
    return {
        "support": sup if sup > 0 else None,
        "resistance": res if res > 0 else None,
        "support_behavior": str(d.get("support_behavior", "UNKNOWN")).upper(),
        "resistance_behavior": str(d.get("resistance_behavior", "UNKNOWN")).upper(),
        "support_hold": _safe(d.get("support_hold_score"), 0.0),
        "support_break": _safe(d.get("support_break_score"), 0.0),
        "resistance_hold": _safe(d.get("resistance_hold_score"), 0.0),
        "resistance_break": _safe(d.get("resistance_break_score"), 0.0),
    }


def _fallback(support, resistance, reason: str) -> Dict[str, Any]:
    return {
        "signal": "WAIT", "candidate_signal": "WAIT", "safe_to_trade": False,
        "next_candle": "UNCLEAR", "forecast_ready": False, "forecast_confidence": 50,
        "trend": "Scanning", "confidence": 0,
        "entry": 0.0, "invalidation": 0.0, "target": 0.0,
        "ce_probability": 50, "pe_probability": 50, "score": 0.0,
        "reasons": [reason], "market_state": "SCANNING", "state_reliability": 0,
        "support": _safe(support), "resistance": _safe(resistance),
        "up_room": 0.0, "down_room": 0.0, "breakout_room": 0.0, "breakdown_room": 0.0,
        "atr": 0.0, "next_wall": "NONE", "next_wall_price": None, "wall_distance": 0.0,
        "wall_reaction": "UNKNOWN", "wall_reaction_confidence": 0,
        "destination_support_action": "UNKNOWN", "destination_resistance_action": "UNKNOWN",
        "destination_pe_blocked": False, "destination_ce_blocked": False,
        "pressure_break_down": 0.0, "pressure_break_up": 0.0, "fast_setup": False,
    }


def _predict_market(df, support, resistance, bull_oc, bear_oc, oc_reasons, oc_details):
    x = _clean(df)
    if len(x) < 8:
        return _fallback(support, resistance, "Waiting for enough candles")

    base = _completed(x)
    if len(base) < 7:
        return _fallback(support, resistance, "Waiting for completed candles")

    atr = _atr(base)
    last = _candle(base.iloc[-1], atr)
    prev = _candle(base.iloc[-2], atr)
    spot = _safe(x.iloc[-1]["Close"], last["c"])
    close = last["c"]

    day_high = _safe(base["High"].max())
    day_low = _safe(base["Low"].min())
    day_open = _safe(base.iloc[0]["Open"])
    day_range = max(day_high-day_low, 1.0)
    range_pos = _clip((close-day_low)/day_range, 0.0, 1.0)

    swings = _recent_swings(base, atr)
    oi = _oi_context(oc_details, spot)

    chart_sup = _safe(support, day_low)
    chart_res = _safe(resistance, day_high)
    sup_candidates = [v for v in [chart_sup, swings["repeated_low"], oi["support"], day_low] if v is not None and v <= spot+0.25*atr]
    res_candidates = [v for v in [chart_res, swings["repeated_high"], oi["resistance"], day_high] if v is not None and v >= spot-0.25*atr]
    eff_support = max(sup_candidates) if sup_candidates else day_low
    eff_resistance = min(res_candidates) if res_candidates else day_high

    dist_sup = max(spot-eff_support, 0.0)
    dist_res = max(eff_resistance-spot, 0.0)

    recent = base.tail(min(15, len(base)))
    r6 = base.tail(min(6, len(base)))
    r10 = base.tail(min(10, len(base)))
    r15 = base.tail(min(15, len(base)))

    up_from_low = max(
        close-_safe(r6["Low"].min(), close),
        close-_safe(r10["Low"].min(), close),
        close-_safe(r15["Low"].min(), close),
    )
    down_from_high = max(
        _safe(r6["High"].max(), close)-close,
        _safe(r10["High"].max(), close)-close,
        _safe(r15["High"].max(), close)-close,
    )

    near_top = range_pos >= 0.78 or dist_res <= max(0.7*atr, 6.0)
    near_bottom = range_pos <= 0.22 or dist_sup <= max(0.7*atr, 6.0)
    late_up = up_from_low >= max(2.2*atr, 25.0) and near_top
    late_down = down_from_high >= max(2.2*atr, 25.0) and near_bottom

    upper_reject = last["upper_ratio"] >= 0.38 and near_top
    lower_reject = last["lower_ratio"] >= 0.38 and near_bottom
    failed_breakout = last["h"] > eff_resistance and last["c"] < eff_resistance
    failed_breakdown = last["l"] < eff_support and last["c"] > eff_support

    tol = max(2.0, min(0.55*atr, 6.0))
    recent3 = base.tail(3)
    upper_fail_count = int(((recent3["High"] >= eff_resistance-tol) & (recent3["Close"] < eff_resistance)).sum())
    lower_fail_count = int(((recent3["Low"] <= eff_support+tol) & (recent3["Close"] > eff_support)).sum())

    breakout_buffer = max(1.5, 0.22*atr)
    breakout_accept = last["c"] > eff_resistance+breakout_buffer and (last["strong_bull"] or prev["c"] > eff_resistance)
    breakdown_accept = last["c"] < eff_support-breakout_buffer and (last["strong_bear"] or prev["c"] < eff_support)

    oi_res_hold = oi["resistance_behavior"] in ("HOLD/REJECT", "REJECT", "HOLD") or oi["resistance_hold"] > oi["resistance_break"]+1.0
    oi_res_break = oi["resistance_behavior"] in ("BREAKOUT", "BREAK") or oi["resistance_break"] > oi["resistance_hold"]+1.0
    oi_sup_hold = oi["support_behavior"] in ("HOLD/BOUNCE", "BOUNCE", "HOLD") or oi["support_hold"] > oi["support_break"]+1.0
    oi_sup_break = oi["support_behavior"] in ("BREAK", "BREAKDOWN") or oi["support_break"] > oi["support_hold"]+1.0

    # Full session + active structure
    net = close-day_open
    efficiency = abs(net) / max(float(base["Close"].diff().abs().sum()), 1.0)
    recent_net = _safe(recent["Close"].iloc[-1] - recent["Close"].iloc[0]) if len(recent) > 1 else 0.0
    bull = 0.0
    bear = 0.0
    reasons: List[str] = []

    if net > 0: bull += min(14.0, abs(net)/max(atr,1)*2.2)
    elif net < 0: bear += min(14.0, abs(net)/max(atr,1)*2.2)
    if recent_net > 0: bull += min(18.0, abs(recent_net)/max(atr,1)*3.0)
    elif recent_net < 0: bear += min(18.0, abs(recent_net)/max(atr,1)*3.0)

    # Candle-by-candle structure has real authority near levels.
    if last["strong_bull"]: bull += 9
    if last["strong_bear"]: bear += 9
    if prev["strong_bull"] and last["c"] >= prev["c"]: bull += 4
    if prev["strong_bear"] and last["c"] <= prev["c"]: bear += 4
    if lower_reject: bull += 12; reasons.append("Bullish rejection from support area")
    if upper_reject: bear += 12; reasons.append("Bearish rejection from resistance area")
    if failed_breakout: bear += 18; reasons.append("Failed breakout / close back below resistance")
    if failed_breakdown: bull += 18; reasons.append("Failed breakdown / reclaim above support")
    if upper_fail_count >= 2: bear += 10; reasons.append("Repeated failure to close above upper zone")
    if lower_fail_count >= 2: bull += 10; reasons.append("Repeated failure to close below lower zone")
    if breakout_accept: bull += 20; reasons.append("Breakout accepted above resistance")
    if breakdown_accept: bear += 20; reasons.append("Breakdown accepted below support")

    # Option-chain is wall intelligence, not standalone trigger.
    bull += _clip(_safe(bull_oc)*0.55, 0, 10)
    bear += _clip(_safe(bear_oc)*0.55, 0, 10)
    if oi_res_hold and near_top: bear += 10; reasons.append("CE OI resistance is holding")
    if oi_res_break and near_top: bull += 8; reasons.append("CE OI resistance is weakening/breaking")
    if oi_sup_hold and near_bottom: bull += 10; reasons.append("PE OI support is holding")
    if oi_sup_break and near_bottom: bear += 8; reasons.append("PE OI support is weakening/breaking")

    # Indicator confirmation only.
    row = x.iloc[-1]
    ema20 = _safe(row.get("EMA20"), 0.0)
    ema50 = _safe(row.get("EMA50"), 0.0)
    vwap = _safe(row.get("VWAP"), 0.0)
    rsi = _safe(row.get("RSI"), 50.0)
    macd = _safe(row.get("MACD"), 0.0)
    sig = _safe(row.get("Signal_Line", row.get("MACD_Signal", 0.0)), 0.0)
    if ema20 and ema50:
        if ema20 > ema50: bull += 3
        elif ema20 < ema50: bear += 3
    if vwap:
        if spot > vwap: bull += 2
        elif spot < vwap: bear += 2
    if rsi >= 55: bull += 2
    elif rsi <= 45: bear += 2
    if macd > sig: bull += 2
    elif macd < sig: bear += 2

    # Late chase blockers: continuation needs a fresh accepted setup.
    ce_block = False
    pe_block = False
    if late_up and not (breakout_accept or (oi_res_break and last["strong_bull"])):
        ce_block = True; reasons.append("CE blocked: upside leg already extended near resistance")
    if late_down and not (breakdown_accept or (oi_sup_break and last["strong_bear"])):
        pe_block = True; reasons.append("PE blocked: downside leg already extended near support")
    if (upper_reject or failed_breakout or upper_fail_count >= 2) and not breakout_accept:
        ce_block = True
    if (lower_reject or failed_breakdown or lower_fail_count >= 2) and not breakdown_accept:
        pe_block = True
    if oi_res_hold and dist_res <= max(0.8*atr, 7.0) and not breakout_accept:
        ce_block = True
    if oi_sup_hold and dist_sup <= max(0.8*atr, 7.0) and not breakdown_accept:
        pe_block = True

    total = max(bull+bear, 1.0)
    ce_prob = int(round(_clip(100*bull/total, 5, 95)))
    pe_prob = 100-ce_prob
    delta = bull-bear
    if delta >= 7: next_candle = "UP"
    elif delta <= -7: next_candle = "DOWN"
    else: next_candle = "UNCLEAR"
    forecast_conf = int(round(_clip(50+abs(delta)*1.4, 50, 90)))

    # In a range, top/bottom reactions can override stale broad bias.
    if efficiency < 0.34 and day_range >= max(3.0*atr, 25.0):
        market_state = "RANGE"
    elif recent_net >= max(1.2*atr, 10):
        market_state = "TREND_UP" if net >= 0 else "REVERSAL_UP"
    elif recent_net <= -max(1.2*atr, 10):
        market_state = "TREND_DOWN" if net <= 0 else "REVERSAL_DOWN"
    else:
        market_state = "MIXED"
    if breakout_accept: market_state = "BREAKOUT_UP"
    if breakdown_accept: market_state = "BREAKDOWN"

    # realistic room; accepted break opens path beyond the broken wall.
    up_room = dist_res
    down_room = dist_sup
    if breakout_accept:
        higher = [v for v in [day_high, chart_res, oi["resistance"]] if v is not None and v > spot+MIN_TRADE_ROOM]
        up_room = min(higher)-spot if higher else max(atr*1.4, MIN_TRADE_ROOM+1)
    if breakdown_accept:
        lower = [v for v in [day_low, chart_sup, oi["support"]] if v is not None and v < spot-MIN_TRADE_ROOM]
        down_room = spot-max(lower) if lower else max(atr*1.4, MIN_TRADE_ROOM+1)

    candidate = "WAIT"
    if next_candle == "UP" and ce_prob >= 58: candidate = "BUY CE"
    if next_candle == "DOWN" and pe_prob >= 58: candidate = "BUY PE"

    if candidate == "BUY CE" and (ce_block or up_room < MIN_TRADE_ROOM): candidate = "WAIT"
    if candidate == "BUY PE" and (pe_block or down_room < MIN_TRADE_ROOM): candidate = "WAIT"

    confidence = ce_prob if candidate == "BUY CE" else pe_prob if candidate == "BUY PE" else max(ce_prob, pe_prob)
    safe_to_trade = candidate in ("BUY CE", "BUY PE") and confidence >= 58 and forecast_conf >= 56
    signal = candidate if safe_to_trade else "WAIT"

    # Trade levels from current price and ATR, preserving app-compatible numeric values.
    risk = max(8.0, min(15.0, atr*1.15))
    if signal == "BUY CE":
        entry = spot; invalidation = spot-risk; target = spot+max(MIN_TRADE_ROOM, min(max(up_room, MIN_TRADE_ROOM), 15.0))
        next_wall, next_wall_price, wall_distance = "RESISTANCE", eff_resistance, max(eff_resistance-spot, 0.0)
        wall_reaction = oi["resistance_behavior"]
    elif signal == "BUY PE":
        entry = spot; invalidation = spot+risk; target = spot-max(MIN_TRADE_ROOM, min(max(down_room, MIN_TRADE_ROOM), 15.0))
        next_wall, next_wall_price, wall_distance = "SUPPORT", eff_support, max(spot-eff_support, 0.0)
        wall_reaction = oi["support_behavior"]
    else:
        entry = spot; invalidation = spot; target = spot
        if dist_res <= dist_sup:
            next_wall, next_wall_price, wall_distance, wall_reaction = "RESISTANCE", eff_resistance, dist_res, oi["resistance_behavior"]
        else:
            next_wall, next_wall_price, wall_distance, wall_reaction = "SUPPORT", eff_support, dist_sup, oi["support_behavior"]

    if not reasons:
        reasons.append("Mixed evidence; waiting for cleaner alignment")
    if oc_reasons:
        for r in list(oc_reasons)[:3]:
            if str(r) not in reasons: reasons.append(str(r))

    trend = "Bullish" if delta >= 7 else "Bearish" if delta <= -7 else "Neutral"
    reliability = int(round(_clip(45 + min(abs(delta), 30), 35, 80)))
    fast_setup = safe_to_trade and confidence >= 70 and forecast_conf >= 68 and ((signal=="BUY CE" and breakout_accept) or (signal=="BUY PE" and breakdown_accept))

    return {
        "signal": signal,
        "candidate_signal": candidate,
        "safe_to_trade": bool(safe_to_trade),
        "next_candle": next_candle,
        "forecast_ready": True,
        "forecast_confidence": forecast_conf,
        "trend": trend,
        "confidence": int(confidence),
        "entry": round(entry, 2),
        "invalidation": round(invalidation, 2),
        "target": round(target, 2),
        "ce_probability": ce_prob,
        "pe_probability": pe_prob,
        "score": round(delta, 2),
        "reasons": reasons[:14],
        "market_state": market_state,
        "state_reliability": reliability,
        "support": round(eff_support, 2),
        "resistance": round(eff_resistance, 2),
        "up_room": round(max(up_room,0.0), 2),
        "down_room": round(max(down_room,0.0), 2),
        "breakout_room": round(max(up_room,0.0), 2),
        "breakdown_room": round(max(down_room,0.0), 2),
        "atr": round(atr, 2),
        "next_wall": next_wall,
        "next_wall_price": round(next_wall_price, 2) if next_wall_price is not None else None,
        "wall_distance": round(max(wall_distance,0.0), 2),
        "wall_reaction": wall_reaction,
        "wall_reaction_confidence": int(_clip(max(oi["resistance_hold"],oi["resistance_break"],oi["support_hold"],oi["support_break"])*10,0,100)),
        "destination_support_action": oi["support_behavior"],
        "destination_resistance_action": oi["resistance_behavior"],
        "destination_pe_blocked": bool(pe_block),
        "destination_ce_blocked": bool(ce_block),
        "pressure_break_down": round(oi["support_break"],2),
        "pressure_break_up": round(oi["resistance_break"],2),
        "fast_setup": bool(fast_setup),
        }
    
