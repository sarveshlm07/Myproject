# option_chain_ai.py
# NiftyAI - Smart Option Chain / OI Intelligence

import math
import re


def _num(value, default=0.0):
    try:
        value = float(str(value).replace(",", "").strip())
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return float(default)


def _clip(value, low, high):
    return max(low, min(high, value))


def _strike_from_symbol(symbol):
    try:
        match = re.search(
            r"(\d+(?:\.\d+)?)(?:CE|PE)$",
            str(symbol).upper()
        )
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return 0.0


def _collect(option_data):
    calls = []
    puts = []

    for symbol, row in (option_data or {}).items():
        if not isinstance(row, dict):
            continue

        symbol = str(row.get("symbol") or symbol).upper()

        strike = _num(row.get("strike"), 0.0)
        if strike <= 0:
            strike = _strike_from_symbol(symbol)

        if strike <= 0:
            continue

        item = {
            "symbol": symbol,
            "strike": strike,
            "oi": max(_num(row.get("oi")), 0.0),
            "change_oi": _num(row.get("change_oi")),
            "oi_change_short": _num(
                row.get("oi_change_short", row.get("change_oi"))
            ),
            "oi_slope": _num(row.get("oi_slope")),
            "volume": max(_num(row.get("volume")), 0.0),
            "volume_change_short": _num(row.get("volume_change_short")),
            "last_price": _num(row.get("last_price")),
            "price_slope": _num(row.get("price_slope")),
            "price_change_short": _num(row.get("price_change_short")),
            "buy_qty": max(
                _num(
                    row.get(
                        "buy_quantity",
                        row.get("buy_qty", 0.0)
                    )
                ),
                0.0
            ),
            "sell_qty": max(
                _num(
                    row.get(
                        "sell_quantity",
                        row.get("sell_qty", 0.0)
                    )
                ),
                0.0
            ),
        }

        if symbol.endswith("CE"):
            calls.append(item)
        elif symbol.endswith("PE"):
            puts.append(item)

    return calls, puts


def _rank_walls(rows, spot, side):
    if not rows:
        return []

    if side == "CE":
        candidates = [x for x in rows if x["strike"] >= spot]
    else:
        candidates = [x for x in rows if x["strike"] <= spot]

    if not candidates:
        candidates = list(rows)

    total_oi = max(
        sum(max(x["oi"], 0.0) for x in candidates),
        1.0
    )

    total_volume = max(
        sum(max(x["volume"], 0.0) for x in candidates),
        1.0
    )

    max_slope = max(
        [abs(x["oi_slope"]) for x in candidates] + [1.0]
    )

    ranked = []

    for row in candidates:
        distance = abs(row["strike"] - spot)
        proximity = 1.0 / (1.0 + distance / 100.0)

        oi_share = row["oi"] / total_oi
        volume_share = row["volume"] / total_volume

        build = max(row["oi_slope"], 0.0) / max_slope
        unwind = max(-row["oi_slope"], 0.0) / max_slope

        strength = (
            oi_share * 0.52
            + volume_share * 0.10
            + proximity * 0.22
            + min(build, 1.0) * 0.13
            - min(unwind, 1.0) * 0.08
            + min(abs(row["price_slope"]) / 2.0, 1.0) * 0.03
        )

        ranked.append(
            {
                **row,
                "distance": float(distance),
                "strength": float(_clip(strength, 0.0, 1.25)),
            }
        )

    ranked.sort(
        key=lambda x: x["strength"],
        reverse=True
    )

    return ranked


def _wall_evidence(wall, side):
    if not wall:
        return 0.0, 0.0, "UNKNOWN"

    hold = 0.0
    break_score = 0.0

    oi = max(wall["oi"], 1.0)

    oi_slope = wall["oi_slope"]
    oi_short = wall["oi_change_short"]

    price_slope = wall["price_slope"]
    price_short = wall["price_change_short"]

    strength = _clip(wall["strength"], 0.0, 1.0)
    distance = max(wall["distance"], 0.0)

    # Base wall importance
    hold += strength * 2.5

    if distance <= 25:
        hold += 0.7
    elif distance <= 50:
        hold += 0.35

    slope_ratio = oi_slope / oi
    short_ratio = oi_short / oi

    # OI building = wall strengthening
    if slope_ratio >= 0.00035:
        hold += 2.4
    elif slope_ratio > 0:
        hold += 1.2

    # OI unwinding = wall weakening
    if slope_ratio <= -0.00035:
        break_score += 2.6
    elif slope_ratio < 0:
        break_score += 1.2

    if short_ratio >= 0.002:
        hold += 1.4
    elif short_ratio <= -0.002:
        break_score += 1.6

    # PE WALL = support
    if side == "PE":
        if oi_slope > 0 and price_slope > 0:
            hold += 1.2

        if oi_slope < 0 and price_slope > 0:
            break_score += 1.6

        if oi_short > 0 and price_short > 0:
            hold += 0.6

        if oi_short < 0 and price_short > 0:
            break_score += 0.8

    # CE WALL = resistance
    else:
        if oi_slope > 0 and price_slope > 0:
            hold += 1.2

        if oi_slope < 0 and price_slope < 0:
            break_score += 1.6

        if oi_short > 0 and price_short > 0:
            hold += 0.6

        if oi_short < 0 and price_short < 0:
            break_score += 0.8

    if hold >= 3.0 and hold >= break_score + 1.5:
        if side == "PE":
            behaviour = "HOLD/BOUNCE"
        else:
            behaviour = "HOLD/REJECT"

    elif break_score >= 2.8 and break_score >= hold + 1.2:
        if side == "PE":
            behaviour = "BREAK"
        else:
            behaviour = "BREAKOUT"

    else:
        behaviour = "UNKNOWN"

    return float(hold), float(break_score), behaviour


def _option_bias(
    calls,
    puts,
    pcr,
    support_hold,
    support_break,
    resistance_hold,
    resistance_break,
):
    bullish = 0.0
    bearish = 0.0
    reasons = []

    # PCR only as context
    if 1.05 <= pcr <= 1.45:
        bullish += 1.5
        reasons.append(f"PCR {pcr:.2f}: mildly bullish")

    elif 0 < pcr < 0.90:
        bearish += 1.5
        reasons.append(f"PCR {pcr:.2f}: mildly bearish")

    elif pcr > 1.60:
        reasons.append(f"PCR {pcr:.2f}: elevated")

    call_slope = sum(x["oi_slope"] for x in calls)
    put_slope = sum(x["oi_slope"] for x in puts)

    # Put writing = support
    if put_slope > 0:
        bullish += min(2.5, abs(put_slope) / 8000.0)
    elif put_slope < 0:
        bearish += min(2.5, abs(put_slope) / 8000.0)

    # Call writing = resistance
    if call_slope > 0:
        bearish += min(2.5, abs(call_slope) / 8000.0)
    elif call_slope < 0:
        bullish += min(2.5, abs(call_slope) / 8000.0)

    # Wall behaviour gets major weight
    bullish += min(
        7.0,
        support_hold + resistance_break
    )

    bearish += min(
        7.0,
        resistance_hold + support_break
    )

    # Depth only minor confirmation
    ce_buy = sum(x["buy_qty"] for x in calls)
    ce_sell = sum(x["sell_qty"] for x in calls)

    pe_buy = sum(x["buy_qty"] for x in puts)
    pe_sell = sum(x["sell_qty"] for x in puts)

    if ce_buy > ce_sell * 1.15:
        bullish += 0.7

    if ce_sell > ce_buy * 1.15:
        bearish += 0.7

    if pe_buy > pe_sell * 1.15:
        bearish += 0.7

    if pe_sell > pe_buy * 1.15:
        bullish += 0.7

    bullish = _clip(bullish, 0.0, 15.0)
    bearish = _clip(bearish, 0.0, 15.0)

    return (
        float(bullish),
        float(bearish),
        reasons,
        float(call_slope),
        float(put_slope),
    )


def _wall_export(rows):
    result = []

    for x in rows[:5]:
        result.append(
            {
                "strike": float(x["strike"]),
                "oi": float(x["oi"]),
                "oi_slope": float(x["oi_slope"]),
                "price_slope": float(x["price_slope"]),
                "strength": float(x["strength"]),
                "distance": float(x["distance"]),
            }
        )

    return result


def analyze_option_chain(
    option_data,
    spot_price=None,
    return_details=False,
):
    calls, puts = _collect(option_data)

    if not calls and not puts:
        details = {
            "pcr": 0.0,
            "total_call_oi": 0.0,
            "total_put_oi": 0.0,
            "call_oi_change": 0.0,
            "put_oi_change": 0.0,
            "call_oi_slope": 0.0,
            "put_oi_slope": 0.0,
            "oi_support": None,
            "oi_resistance": None,
            "support_distance": None,
            "resistance_distance": None,
            "support_strength": 0.0,
            "resistance_strength": 0.0,
            "support_behavior": "UNKNOWN",
            "resistance_behavior": "UNKNOWN",
            "support_hold_score": 0.0,
            "support_break_score": 0.0,
            "resistance_hold_score": 0.0,
            "resistance_break_score": 0.0,
            "top_call_walls": [],
            "top_put_walls": [],
        }

        result = (
            0.0,
            0.0,
            ["No valid CE/PE option-chain data"],
        )

        if return_details:
            return result[0], result[1], result[2], details

        return result

    spot = _num(spot_price, 0.0)

    if spot <= 0:
        all_strikes = [
            x["strike"]
            for x in calls + puts
        ]

        spot = sum(all_strikes) / max(
            len(all_strikes),
            1
        )

    total_call_oi = sum(
        max(x["oi"], 0.0)
        for x in calls
    )

    total_put_oi = sum(
        max(x["oi"], 0.0)
        for x in puts
    )

    if total_call_oi > 0:
        pcr = total_put_oi / total_call_oi
    else:
        pcr = 0.0

    ranked_calls = _rank_walls(
        calls,
        spot,
        "CE"
    )

    ranked_puts = _rank_walls(
        puts,
        spot,
        "PE"
    )

    call_wall = (
        ranked_calls[0]
        if ranked_calls
        else None
    )

    put_wall = (
        ranked_puts[0]
        if ranked_puts
        else None
    )

    support_hold, support_break, support_behavior = _wall_evidence(
        put_wall,
        "PE"
    )

    resistance_hold, resistance_break, resistance_behavior = _wall_evidence(
        call_wall,
        "CE"
    )

    bull, bear, reasons, call_slope, put_slope = _option_bias(
        calls,
        puts,
        pcr,
        support_hold,
        support_break,
        resistance_hold,
        resistance_break,
    )

    if put_wall:
        reasons.append(
            f"PE OI support {put_wall['strike']:.0f} | "
            f"strength {put_wall['strength']:.2f} | "
            f"{support_behavior}"
        )

        if support_behavior == "HOLD/BOUNCE":
            reasons.append(
                "Support holding: avoid fresh PE unless breakdown confirms"
            )

        elif support_behavior == "BREAK":
            reasons.append(
                "Support weakening: downside continuation possible"
            )

    if call_wall:
        reasons.append(
            f"CE OI resistance {call_wall['strike']:.0f} | "
            f"strength {call_wall['strength']:.2f} | "
            f"{resistance_behavior}"
        )

        if resistance_behavior == "HOLD/REJECT":
            reasons.append(
                "Resistance holding: avoid fresh CE unless breakout confirms"
            )

        elif resistance_behavior == "BREAKOUT":
            reasons.append(
                "Resistance weakening: upside continuation possible"
            )

    if bull > bear + 2.0:
        final_bias = "BULLISH"
    elif bear > bull + 2.0:
        final_bias = "BEARISH"
    else:
        final_bias = "MIXED"

    reasons.append(
        f"FINAL OPTION BIAS: {final_bias}"
    )

    details = {
        "pcr": float(pcr),

        "total_call_oi": float(total_call_oi),
        "total_put_oi": float(total_put_oi),

        "call_oi_change": float(
            sum(x["change_oi"] for x in calls)
        ),

        "put_oi_change": float(
            sum(x["change_oi"] for x in puts)
        ),

        "call_oi_slope": float(call_slope),
        "put_oi_slope": float(put_slope),

        "oi_support": (
            float(put_wall["strike"])
            if put_wall
            else None
        ),

        "oi_resistance": (
            float(call_wall["strike"])
            if call_wall
            else None
        ),

        "support_distance": (
            float(spot - put_wall["strike"])
            if put_wall
            else None
        ),

        "resistance_distance": (
            float(call_wall["strike"] - spot)
            if call_wall
            else None
        ),

        "support_strength": (
            float(put_wall["strength"])
            if put_wall
            else 0.0
        ),

        "resistance_strength": (
            float(call_wall["strength"])
            if call_wall
            else 0.0
        ),

        "support_behavior": support_behavior,
        "resistance_behavior": resistance_behavior,

        "support_hold_score": float(support_hold),
        "support_break_score": float(support_break),

        "resistance_hold_score": float(resistance_hold),
        "resistance_break_score": float(resistance_break),

        "top_call_walls": _wall_export(
            ranked_calls
        ),

        "top_put_walls": _wall_export(
            ranked_puts
        ),
    }

    if return_details:
        return (
            bull,
            bear,
            reasons,
            details,
        )

    return (
        bull,
        bear,
        reasons,
    )
