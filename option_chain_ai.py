# option_chain_ai.py
# NiftyAI — OI wall / reversal / breakout intelligence

import math
import re


def _num(value, default=0.0):
    try:
        x = float(str(value).replace(",", "").strip())
        return x if math.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _clip(value, low, high):
    return max(low, min(high, value))


def _strike_from_symbol(symbol):
    try:
        match = re.search(
            r"(\d+(?:\.\d+)?)(?:CE|PE)$",
            str(symbol).upper(),
        )
        return float(match.group(1)) if match else 0.0
    except Exception:
        return 0.0


def _collect(option_data):
    calls = []
    puts = []

    for symbol, row in (option_data or {}).items():
        if not isinstance(row, dict):
            continue

        symbol = str(
            row.get("symbol") or symbol or ""
        ).upper()

        strike = _num(
            row.get("strike"),
            0.0,
        )

        if strike <= 0:
            strike = _strike_from_symbol(symbol)

        if strike <= 0:
            continue

        item = {
            "symbol": symbol,
            "strike": strike,

            "oi": max(
                _num(row.get("oi")),
                0.0,
            ),

            "change_oi": _num(
                row.get("change_oi")
            ),

            "oi_change_short": _num(
                row.get(
                    "oi_change_short",
                    row.get("change_oi"),
                )
            ),

            "oi_slope": _num(
                row.get("oi_slope")
            ),

            "volume": max(
                _num(row.get("volume")),
                0.0,
            ),

            "volume_change_short": _num(
                row.get("volume_change_short")
            ),

            "price_slope": _num(
                row.get("price_slope")
            ),

            "price_change_short": _num(
                row.get("price_change_short")
            ),

            "last_price": _num(
                row.get(
                    "last_price",
                    row.get("ltp"),
                )
            ),

            "buy_qty": max(
                _num(
                    row.get(
                        "buy_quantity",
                        row.get("buy_qty"),
                    )
                ),
                0.0,
            ),

            "sell_qty": max(
                _num(
                    row.get(
                        "sell_quantity",
                        row.get("sell_qty"),
                    )
                ),
                0.0,
            ),
        }

        if symbol.endswith("CE"):
            calls.append(item)

        elif symbol.endswith("PE"):
            puts.append(item)

    return calls, puts


def _rank_walls(items, spot, side):
    """
    Rank nearby OI walls.

    CE above spot = resistance candidates.
    PE below spot = support candidates.
    """

    if not items:
        return []

    if side == "CE":
        candidates = [
            x for x in items
            if x["strike"] >= spot
        ]
    else:
        candidates = [
            x for x in items
            if x["strike"] <= spot
        ]

    if not candidates:
        candidates = list(items)

    total_oi = max(
        sum(
            max(x["oi"], 0.0)
            for x in candidates
        ),
        1.0,
    )

    total_volume = max(
        sum(
            max(x["volume"], 0.0)
            for x in candidates
        ),
        1.0,
    )

    max_abs_slope = max(
        (
            abs(x["oi_slope"])
            for x in candidates
        ),
        default=1.0,
    )

    max_abs_slope = max(
        max_abs_slope,
        1.0,
    )

    ranked = []

    for x in candidates:
        distance = abs(
            x["strike"] - spot
        )

        proximity = (
            1.0 /
            (1.0 + distance / 100.0)
        )

        oi_share = (
            max(x["oi"], 0.0)
            / total_oi
        )

        volume_share = (
            max(x["volume"], 0.0)
            / total_volume
        )

        build = (
            max(x["oi_slope"], 0.0)
            / max_abs_slope
        )

        unwind = (
            max(-x["oi_slope"], 0.0)
            / max_abs_slope
        )

        strength = (
            0.52 * oi_share
            + 0.10 * volume_share
            + 0.22 * proximity
            + 0.13 * min(build, 1.0)
            - 0.08 * min(unwind, 1.0)
            + 0.03 * min(
                abs(x["price_slope"]) / 2.0,
                1.0,
            )
        )

        ranked.append({
            **x,
            "distance": float(distance),
            "strength": float(
                _clip(
                    strength,
                    0.0,
                    1.25,
                )
            ),
        })

    ranked.sort(
        key=lambda z: z["strength"],
        reverse=True,
    )

    return ranked


def _wall_evidence(wall, side):
    """
    PE:
        HOLD/BOUNCE = support holding
        BREAK       = support weakening

    CE:
        HOLD/REJECT = resistance holding
        BREAKOUT    = resistance weakening
    """

    if not wall:
        return 0.0, 0.0, "UNKNOWN"

    hold = 0.0
    break_score = 0.0

    oi = max(
        wall["oi"],
        1.0,
    )

    oi_slope = wall["oi_slope"]
    oi_short = wall["oi_change_short"]

    price_slope = wall["price_slope"]
    price_short = wall["price_change_short"]

    strength = _clip(
        wall["strength"],
        0.0,
        1.0,
    )

    distance = max(
        wall["distance"],
        0.0,
    )

    # -----------------------------------------------------
    # WALL STRENGTH
    # -----------------------------------------------------

    hold += strength * 2.4

    if distance <= 25:
        hold += 0.6

    elif distance <= 50:
        hold += 0.3

    # -----------------------------------------------------
    # OI BUILD / UNWIND
    # -----------------------------------------------------

    slope_ratio = (
        oi_slope / oi
    )

    short_ratio = (
        oi_short / oi
    )

    if slope_ratio > 0.00035:
        hold += 2.4

    elif slope_ratio > 0:
        hold += 1.2

    elif slope_ratio < -0.00035:
        break_score += 2.6

    elif slope_ratio < 0:
        break_score += 1.2

    if short_ratio > 0.002:
        hold += 1.6

    elif short_ratio < -0.002:
        break_score += 1.8

    # -----------------------------------------------------
    # PRICE + OI RELATION
    # -----------------------------------------------------

    if side == "PE":

        # PE wall getting stronger.
        if (
            oi_slope > 0
            and price_slope > 0
        ):
            hold += 1.4

        # PE wall unwinding while PE premium rises:
        # support failure risk.
        if (
            oi_slope < 0
            and price_slope > 0
        ):
            break_score += 1.8

        if (
            price_short > 0
            and oi_short > 0
        ):
            hold += 0.8

        if (
            price_short > 0
            and oi_short < 0
        ):
            break_score += 0.8

    else:

        # CE wall strengthening.
        if (
            oi_slope > 0
            and price_slope > 0
        ):
            hold += 1.4

        # CE OI unwinding + premium weakening:
        # resistance opening.
        if (
            oi_slope < 0
            and price_slope < 0
        ):
            break_score += 1.8

        if (
            price_short > 0
            and oi_short > 0
        ):
            hold += 0.8

        if (
            price_short < 0
            and oi_short < 0
        ):
            break_score += 0.8

    # -----------------------------------------------------
    # FINAL WALL STATE
    # -----------------------------------------------------

    if (
        hold >= break_score + 1.5
        and hold >= 3.0
    ):

        behaviour = (
            "HOLD/BOUNCE"
            if side == "PE"
            else "HOLD/REJECT"
        )

    elif (
        break_score >= hold + 1.2
        and break_score >= 2.8
    ):

        behaviour = (
            "BREAK"
            if side == "PE"
            else "BREAKOUT"
        )

    else:
        behaviour = "UNKNOWN"

    return (
        float(hold),
        float(break_score),
        behaviour,
    )


def _directional_bias(
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

    # -----------------------------------------------------
    # PCR — CONTEXT ONLY
    # -----------------------------------------------------

    if 1.05 <= pcr <= 1.45:
        bullish += 1.5

        reasons.append(
            f"PCR {pcr:.2f}: mildly supportive"
        )

    elif 0 < pcr < 0.90:
        bearish += 1.5

        reasons.append(
            f"PCR {pcr:.2f}: mildly bearish"
        )

    elif pcr > 1.60:
        reasons.append(
            f"PCR {pcr:.2f}: elevated/crowded"
        )

    # -----------------------------------------------------
    # AGGREGATE OI MOVEMENT
    # -----------------------------------------------------

    put_slope = sum(
        x["oi_slope"]
        for x in puts
    )

    call_slope = sum(
        x["oi_slope"]
        for x in calls
    )

    if put_slope > 0:
        bullish += min(
            2.5,
            abs(put_slope) / 8000.0,
        )

    elif put_slope < 0:
        bearish += min(
            2.5,
            abs(put_slope) / 8000.0,
        )

    if call_slope > 0:
        bearish += min(
            2.5,
            abs(call_slope) / 8000.0,
        )

    elif call_slope < 0:
        bullish += min(
            2.5,
            abs(call_slope) / 8000.0,
        )

    # -----------------------------------------------------
    # NEAREST WALL REACTION
    # More important than generic PCR.
    # -----------------------------------------------------

    bullish += min(
        7.0,
        support_hold
        + resistance_break,
    )

    bearish += min(
        7.0,
        resistance_hold
        + support_break,
    )

    # -----------------------------------------------------
    # DEPTH — SMALL CONFIRMATION ONLY
    # -----------------------------------------------------

    total_buy_ce = sum(
        x["buy_qty"]
        for x in calls
    )

    total_sell_ce = sum(
        x["sell_qty"]
        for x in calls
    )

    total_buy_pe = sum(
        x["buy_qty"]
        for x in puts
    )

    total_sell_pe = sum(
        x["sell_qty"]
        for x in puts
    )

    if (
        total_buy_ce >
        total_sell_ce * 1.15
    ):
        bullish += 0.8

    if (
        total_sell_ce >
        total_buy_ce * 1.15
    ):
        bearish += 0.8

    if (
        total_buy_pe >
        total_sell_pe * 1.15
    ):
        bearish += 0.8

    if (
        total_sell_pe >
        total_buy_pe * 1.15
    ):
        bullish += 0.8

    return (
        float(
            _clip(
                bullish,
                0.0,
                15.0,
            )
        ),
        float(
            _clip(
                bearish,
                0.0,
                15.0,
            )
        ),
        reasons,
        float(call_slope),
        float(put_slope),
    )


def analyze_option_chain(
    option_data,
    spot_price=None,
    return_details=False,
):
    """
    Main NiftyAI option-chain analyzer.

    Compatible with current app.py:

    bull_oc, bear_oc, reasons =
        analyze_option_chain(...)

    OR

    bull_oc, bear_oc, reasons, details =
        analyze_option_chain(
            ...,
            return_details=True
        )
    """

    calls, puts = _collect(
        option_data
    )

    # -----------------------------------------------------
    # NO DATA
    # -----------------------------------------------------

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
            [
                "No valid CE/PE option-chain data"
            ],
        )

        if return_details:
            return (
                *result,
                details,
            )

        return result

    # -----------------------------------------------------
    # SPOT
    # -----------------------------------------------------

    spot = _num(
        spot_price,
        0.0,
    )

    if spot <= 0:

        strikes = [
            x["strike"]
            for x in calls + puts
        ]

        spot = (
            sum(strikes)
            / max(len(strikes), 1)
        )

    # -----------------------------------------------------
    # PCR
    # -----------------------------------------------------

    total_call_oi = sum(
        max(x["oi"], 0.0)
        for x in calls
    )

    total_put_oi = sum(
        max(x["oi"], 0.0)
        for x in puts
    )

    pcr = (
        total_put_oi
        / total_call_oi
        if total_call_oi > 0
        else 0.0
    )

    # -----------------------------------------------------
    # FIND IMPORTANT WALLS
    # -----------------------------------------------------

    ranked_calls = _rank_walls(
        calls,
        spot,
        "CE",
    )

    ranked_puts = _rank_walls(
        puts,
        spot,
        "PE",
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

    # -----------------------------------------------------
    # SUPPORT / RESISTANCE BEHAVIOUR
    # -----------------------------------------------------

    (
        support_hold,
        support_break,
        support_behavior,
    ) = _wall_evidence(
        put_wall,
        "PE",
    )

    (
        resistance_hold,
        resistance_break,
        resistance_behavior,
    ) = _wall_evidence(
        call_wall,
        "CE",
    )

    # -----------------------------------------------------
    # DIRECTIONAL OPTION BIAS
    # -----------------------------------------------------

    (
        bull,
        bear,
        reasons,
        call_slope,
        put_slope,
    ) = _directional_bias(
        calls,
        puts,
        pcr,
        support_hold,
        support_break,
        resistance_hold,
        resistance_break,
    )

    # -----------------------------------------------------
    # HUMAN-READABLE REASONS
    # -----------------------------------------------------

    if put_wall:

        reasons.append(
            f"PE OI wall "
            f"{put_wall['strike']:.0f} | "
            f"strength "
            f"{put_wall['strength']:.2f} | "
            f"{support_behavior}"
        )

        if (
            support_behavior
            == "HOLD/BOUNCE"
        ):

            reasons.append(
                "PE wall holding: fresh PE should be blocked unless support breakdown confirms"
            )

        elif (
            support_behavior
            == "BREAK"
        ):

            reasons.append(
                "PE wall weakening: downside continuation becoming credible"
            )

    if call_wall:

        reasons.append(
            f"CE OI wall "
            f"{call_wall['strike']:.0f} | "
            f"strength "
            f"{call_wall['strength']:.2f} | "
            f"{resistance_behavior}"
        )

        if (
            resistance_behavior
            == "HOLD/REJECT"
        ):

            reasons.append(
                "CE wall holding: fresh CE should be blocked unless breakout confirms"
            )

        elif (
            resistance_behavior
            == "BREAKOUT"
        ):

            reasons.append(
                "CE wall weakening: upside continuation becoming credible"
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

    # -----------------------------------------------------
    # DETAILS FOR prediction_engine.py
    # -----------------------------------------------------

    details = {
        "pcr": float(pcr),

        "total_call_oi":
            float(total_call_oi),

        "total_put_oi":
            float(total_put_oi),

        "call_oi_change":
            float(
                sum(
                    x["change_oi"]
                    for x in calls
                )
            ),

        "put_oi_change":
            float(
                sum(
                    x["change_oi"]
                    for x in puts
                )
            ),

        "call_oi_slope":
            float(call_slope),

        "put_oi_slope":
            float(put_slope),

        "oi_support":
            float(put_wall["strike"])
            if put_wall
            else None,

        "oi_resistance":
            float(call_wall["strike"])
            if call_wall
            else None,

        "support_distance":
            float(
                spot
                - put_wall["strike"]
            )
            if put_wall
            else None,

        "resistance_distance":
            float(
                call_wall["strike"]
                - spot
            )
            if call_wall
            else None,

        "support_strength":
            float(
                put_wall["strength"]
            )
            if put_wall
            else 0.0,

        "resistance_strength":
            float(
                call_wall["strength"]
            )
            if call_wall
            else 0.0,

        "support_behavior":
            support_behavior,

        "resistance_behavior":
            resistance_behavior,

        "support_hold_score":
            float(support_hold),

        "support_break_score":
            float(support_break),

        "resistance_hold_score":
            float(resistance_hold),

        "resistance_break_score":
            float(resistance_break),

        "top_call_walls": [
            {
                "strike":
                    float(x["strike"]),

                "oi":
                    float(x["oi"]),

                "oi_slope":
                    float(x["oi_slope"]),

                "price_slope":
                    float(x["price_slope"]),

                "strength":
                    float(x["strength"]),

                "distance":
                    float(x["distance"]),
            }
            for x in ranked_calls[:5]
        ],

        "top_put_walls": [
            {
                "strike":
                    float(x["strike"]),

                "oi":
                    float(x["oi"]),

                "oi_slope":
                    float(x["oi_slope"]),

                "price_slope":
                    float(x["price_slope"]),

                "strength":
                    float(x["strength"]),

                "distance":
                    float(x["distance"]),
            }
            for x in ranked_p
