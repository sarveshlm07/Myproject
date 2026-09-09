import streamlit as st
from streamlit_autorefresh import st_autorefresh
import traceback
import pandas as pd

from prediction_logger import log_prediction, log_outcome
from support_resistance import get_support_resistance
from data import get_nifty_data
from indicators import calculate_indicators
from prediction_engine import predict_market
from chart import create_chart

from option_chain import (
    suggest_option,
    get_atm_strike,
    get_symbols,
    get_option_quotes,
)

from option_chain_ai import analyze_option_chain

import live_data


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="NiftyAI",
    page_icon="📈",
    layout="wide",
)


def _safe_display_df(frame):
    out = frame.copy()

    for col in out.columns:
        if out[col].dtype == "object":
            values = out[col].dropna()

            if (
                len(values)
                and values.map(
                    lambda x: isinstance(x, bool)
                ).all()
            ):
                out[col] = (
                    out[col]
                    .fillna(False)
                    .astype(bool)
                )
            else:
                out[col] = out[col].astype(str)

    return out


# Refresh every second
st_autorefresh(
    interval=1000,
    key="niftyai_refresh",
)


# =========================================================
# PROCESS-GLOBAL LOCK MEMORY
# Survives normal Streamlit reruns and browser refresh
# while the Streamlit server process remains alive.
# =========================================================

@st.cache_resource
def _global_signal_memory():
    return {
        "locks": {},
        "candidates": {},
        "logs": {},
    }


GLOBAL_MEMORY = _global_signal_memory()


# =========================================================
# WEBSOCKET
# =========================================================

if "ws_started" not in st.session_state:
    try:
        live_data.start_websocket()
    except Exception:
        pass

    st.session_state.ws_started = True


# =========================================================
# UI HEADER
# =========================================================

st.title("🤖 NiftyAI")

timeframe = st.selectbox(
    "Select Timeframe",
    ["1m", "3m", "5m"],
    index=2,
)


# =========================================================
# HELPERS
# =========================================================

def _timeframe_seconds(tf):
    return {
        "1m": 60,
        "3m": 180,
        "5m": 300,
    }.get(tf, 300)


def _now_for_index(index):
    if (
        isinstance(index, pd.DatetimeIndex)
        and index.tz is not None
    ):
        return pd.Timestamp.now(
            tz=index.tz
        )

    return pd.Timestamp.now()


def _actual_candle_start(now, tf):
    seconds = _timeframe_seconds(tf)

    anchor = (
        now.normalize()
        + pd.Timedelta(
            hours=9,
            minutes=15,
        )
    )

    if now < anchor:
        minutes = max(
            1,
            seconds // 60,
        )

        return now.floor(
            f"{minutes}min"
        )

    elapsed = max(
        (now - anchor).total_seconds(),
        0.0,
    )

    bucket = int(
        elapsed // seconds
    )

    return (
        anchor
        + pd.Timedelta(
            seconds=bucket * seconds
        )
    )


def _extract_live_price():
    try:
        getter = getattr(
            live_data,
            "get_live_price",
            None,
        )

        if callable(getter):
            value = getter()

            if value is not None:
                return float(value)

    except Exception:
        pass

    try:
        getter = getattr(
            live_data,
            "get_latest_tick",
            None,
        )

        if callable(getter):
            tick = getter()

            if isinstance(tick, dict):
                for key in (
                    "last_price",
                    "ltp",
                    "price",
                    "close",
                ):
                    value = tick.get(key)

                    if value is not None:
                        return float(value)

            elif tick is not None:
                return float(tick)

    except Exception:
        pass

    try:
        tick = getattr(
            live_data,
            "latest_tick",
            None,
        )

        if isinstance(tick, dict):
            for key in (
                "last_price",
                "ltp",
                "price",
                "close",
            ):
                value = tick.get(key)

                if value is not None:
                    return float(value)

    except Exception:
        pass

    return None


def _ensure_running_candle(
    raw_df,
    tf,
):
    if (
        raw_df is None
        or raw_df.empty
    ):
        return raw_df

    x = raw_df.copy()

    if not isinstance(
        x.index,
        pd.DatetimeIndex,
    ):
        x.index = pd.to_datetime(
            x.index
        )

    now = _now_for_index(
        x.index
    )

    candle_start = (
        _actual_candle_start(
            now,
            tf,
        )
    )

    if (
        x.index.tz is not None
        and candle_start.tz is None
    ):
        candle_start = (
            candle_start.tz_localize(
                x.index.tz
            )
        )

    elif (
        x.index.tz is None
        and getattr(
            candle_start,
            "tzinfo",
            None,
        )
        is not None
    ):
        candle_start = (
            candle_start.tz_localize(
                None
            )
        )

    live_price = (
        _extract_live_price()
    )

    hist_current = None

    if (
        len(x)
        and pd.Timestamp(
            x.index[-1]
        )
        == pd.Timestamp(
            candle_start
        )
    ):
        hist_current = (
            x.iloc[-1].copy()
        )

        x = x.iloc[:-1].copy()

    if hist_current is not None:
        base_open = float(
            hist_current.get(
                "Open",
                hist_current.get(
                    "Close",
                    0.0,
                ),
            )
        )

        base_high = float(
            hist_current.get(
                "High",
                base_open,
            )
        )

        base_low = float(
            hist_current.get(
                "Low",
                base_open,
            )
        )

        base_close = float(
            hist_current.get(
                "Close",
                base_open,
            )
        )

        base_volume = float(
            hist_current.get(
                "Volume",
                0.0,
            )
        )

    else:
        previous = x.iloc[-1]

        base_open = float(
            previous.get(
                "Close",
                0.0,
            )
        )

        base_high = base_open
        base_low = base_open
        base_close = base_open
        base_volume = 0.0

    if live_price is None:
        live_price = base_close

    cache_key = (
        f"live_candle_{tf}"
    )

    if cache_key not in st.session_state:
        st.session_state[
            cache_key
        ] = {}

    cache = st.session_state[
        cache_key
    ]

    candle_key = str(
        pd.Timestamp(
            candle_start
        )
    )

    if (
        cache.get("key")
        != candle_key
    ):
        cache = {
            "key": candle_key,
            "Open": base_open,
            "High": max(
                base_high,
                live_price,
            ),
            "Low": min(
                base_low,
                live_price,
            ),
            "Close": live_price,
            "Volume": base_volume,
        }

    else:
        cache["High"] = max(
            float(
                cache.get(
                    "High",
                    base_high,
                )
            ),
            base_high,
            live_price,
        )

        cache["Low"] = min(
            float(
                cache.get(
                    "Low",
                    base_low,
                )
            ),
            base_low,
            live_price,
        )

        cache["Close"] = (
            live_price
        )

        cache["Volume"] = max(
            float(
                cache.get(
                    "Volume",
                    0.0,
                )
            ),
            base_volume,
        )

    st.session_state[
        cache_key
    ] = cache

    columns = list(x.columns)

    row = {
        column: 0.0
        for column in columns
    }

    row.update(
        {
            "Open": float(
                cache["Open"]
            ),
            "High": float(
                cache["High"]
            ),
            "Low": float(
                cache["Low"]
            ),
            "Close": float(
                cache["Close"]
            ),
            "Volume": float(
                cache.get(
                    "Volume",
                    0.0,
                )
            ),
        }
    )

    running = pd.DataFrame(
        [row],
        index=pd.DatetimeIndex(
            [candle_start]
        ),
    )

    x = pd.concat(
        [x, running],
        axis=0,
    )

    x = (
        x[
            ~x.index.duplicated(
                keep="last"
            )
        ]
        .sort_index()
    )

    return x


def _lock_key(
    tf,
    candle,
):
    return (
        f"{tf}|"
        f"{pd.Timestamp(candle).isoformat()}"
    )


def _save_lock(
    key,
    target_candle,
    result,
    current_price,
    market_state,
):
    signal = result.get(
        "candidate_signal",
        result.get(
            "signal",
            "WAIT",
        ),
    )

    forecast = result.get(
        "next_candle",
        "UNCLEAR",
    )

    if signal == "BUY CE":
        forecast = "UP"

    elif signal == "BUY PE":
        forecast = "DOWN"

    lock = {
        "target_candle":
            pd.Timestamp(
                target_candle
            ),

        "signal": signal,

        "forecast": forecast,

        "confidence": int(
            result.get(
                "confidence",
                0,
            )
        ),

        "forecast_confidence":
            int(
                result.get(
                    "forecast_confidence",
                    50,
                )
            ),

        "trend": result.get(
            "trend",
            "Scanning",
        ),

        "entry": float(
            result.get(
                "entry",
                current_price,
            )
        ),

        "invalidation": float(
            result.get(
                "invalidation",
                current_price,
            )
        ),

        "target": float(
            result.get(
                "target",
                current_price,
            )
        ),

        "ce_probability": int(
            result.get(
                "ce_probability",
                50,
            )
        ),

        "pe_probability": int(
            result.get(
                "pe_probability",
                50,
            )
        ),

        "score": float(
            result.get(
                "score",
                0.0,
            )
        ),

        "reasons": list(
            result.get(
                "reasons",
                [],
            )
        ),

        "market_state":
            market_state,

        "locked_price":
            float(
                current_price
            ),

        "locked_at":
            pd.Timestamp.now(),
    }

    GLOBAL_MEMORY[
        "locks"
    ][key] = lock

    return lock


def _cleanup_memory(
    current_candle,
):
    current_candle = pd.Timestamp(
        current_candle
    )

    expired = []

    for key, value in list(
        GLOBAL_MEMORY[
            "locks"
        ].items()
    ):
        try:
            target = pd.Timestamp(
                value.get(
                    "target_candle"
                )
            )

            if (
                target
                < current_candle
            ):
                expired.append(
                    key
                )

        except Exception:
            expired.append(
                key
            )

    for key in expired:
        GLOBAL_MEMORY[
            "locks"
        ].pop(
            key,
            None,
        )

        GLOBAL_MEMORY[
            "candidates"
        ].pop(
            key,
            None,
        )


# =========================================================
# RESET
# =========================================================

if st.button("Reset Trade"):
    for key in list(
        GLOBAL_MEMORY[
            "locks"
        ].keys()
    ):
        if key.startswith(
            f"{timeframe}|"
        ):
            GLOBAL_MEMORY[
                "locks"
            ].pop(
                key,
                None,
            )

    for key in list(
        GLOBAL_MEMORY[
            "candidates"
        ].keys()
    ):
        if key.startswith(
            f"{timeframe}|"
        ):
            GLOBAL_MEMORY[
                "candidates"
            ].pop(
                key,
                None,
            )

    st.rerun()


# =========================================================
# DATA
# =========================================================

try:
    df = get_nifty_data(
        timeframe
    )

except Exception as e:
    st.error(
        f"Market data error: {e}"
    )
    st.stop()


if (
    df is None
    or df.empty
):
    st.warning(
        "Waiting for Zerodha market data..."
    )
    st.stop()


df = _ensure_running_candle(
    df,
    timeframe,
)


try:
    df = calculate_indicators(
        df
    )

except Exception as e:
    st.error(
        f"Indicator calculation error: {e}"
    )

    st.code(
        traceback.format_exc(),
        language="text",
    )

    st.stop()


if (
    df is None
    or df.empty
):
    st.warning(
        "Indicator data unavailable."
    )
    st.stop()


latest = df.iloc[-1]

current_candle = (
    pd.Timestamp(
        df.index[-1]
    )
)

current_price = float(
    latest.get(
        "Close",
        0.0,
    )
)

tf_seconds = (
    _timeframe_seconds(
        timeframe
    )
)

next_candle_start = (
    current_candle
    + pd.Timedelta(
        seconds=tf_seconds
    )
)

next_candle_end = (
    next_candle_start
    + pd.Timedelta(
        seconds=tf_seconds
    )
)

now = _now_for_index(
    df.index
)

seconds_to_next = (
    next_candle_start
    - now
).total_seconds()


_cleanup_memory(
    current_candle
)


# =========================================================
# RESOLVE PREVIOUS FORECAST
# =========================================================

previous_log_key = (
    f"{timeframe}|"
    f"{current_candle.isoformat()}"
)

previous_logged = (
    GLOBAL_MEMORY[
        "logs"
    ].get(
        previous_log_key
    )
)

if (
    previous_logged
    and not previous_logged.get(
        "resolved",
        False,
    )
    and len(df) >= 2
):
    try:
        actual_close = float(
            df.iloc[-2][
                "Close"
            ]
        )

        log_outcome(
            current_candle,
            previous_logged[
                "forecast"
            ],
            previous_logged[
                "market_state"
            ],
            previous_logged[
                "price"
            ],
            actual_close,
        )

        previous_logged[
            "resolved"
        ] = True

    except Exception:
        pass


# =========================================================
# SUPPORT / RESISTANCE
# =========================================================

try:
    support, resistance = (
        get_support_resistance(
            df
        )
    )

except Exception:
    support = float(
        df["Low"].min()
    )

    resistance = float(
        df["High"].max()
    )


# =========================================================
# OPTION CHAIN
# =========================================================

bull_oc = 0.0
bear_oc = 0.0
oc_reasons = []
oc_details = {}


try:
    atm = get_atm_strike(
        current_price
    )

    symbols = get_symbols(
        atm
    )

    quotes = get_option_quotes(
        symbols
    )

    (
        bull_oc,
        bear_oc,
        oc_reasons,
        oc_details,
    ) = analyze_option_chain(
        quotes,
        spot_price=current_price,
        return_details=True,
    )

except Exception as e:
    oc_reasons = [
        f"Option chain unavailable: {e}"
    ]

    oc_details = {}


# =========================================================
# PREDICTION ENGINE
# =========================================================

try:
    result = predict_market(
        df,
        support,
        resistance,
        bull_oc,
        bear_oc,
        oc_reasons,
        oc_details,
    )

except Exception as e:
    st.error(
        "❌ PREDICTION ENGINE ERROR"
    )

    st.error(
        str(e)
    )

    st.code(
        traceback.format_exc(),
        language="text",
    )

    st.stop()


if not isinstance(
    result,
    dict,
):
    st.error(
        "Prediction engine returned invalid data."
    )

    st.write(
        result
    )

    st.stop()


# =========================================================
# RAW ENGINE VALUES
# =========================================================

raw_forecast = result.get(
    "next_candle",
    "UNCLEAR",
)

raw_forecast_confidence = int(
    result.get(
        "forecast_confidence",
        50,
    )
)

raw_signal = result.get(
    "candidate_signal",
    result.get(
        "signal",
        "WAIT",
    ),
)

raw_confidence = int(
    result.get(
        "confidence",
        0,
    )
)

raw_trend = result.get(
    "trend",
    "Scanning",
)

raw_ce = int(
    result.get(
        "ce_probability",
        50,
    )
)

raw_pe = int(
    result.get(
        "pe_probability",
        50,
    )
)

raw_reasons = list(
    result.get(
        "reasons",
        [],
    )
)

raw_score = float(
    result.get(
        "score",
        0.0,
    )
)

raw_market_state = result.get(
    "market_state",
    "MIXED",
)

raw_state_reliability = float(
    result.get(
        "state_reliability",
        0.50,
    )
)

raw_next_wall = result.get(
    "next_wall",
    "—",
)

raw_next_wall_price = float(
    result.get(
        "next_wall_price",
        0.0,
    )
)

raw_wall_distance = float(
    result.get(
        "wall_distance",
        999.0,
    )
)

raw_wall_reaction = result.get(
    "wall_reaction",
    "UNCLEAR",
)

raw_wall_reaction_confidence = float(
    result.get(
        "wall_reaction_confidence",
        0.0,
    )
)

raw_fast_setup = bool(
    result.get(
        "fast_setup",
        False,
    )
)

raw_safe_to_trade = bool(
    result.get(
        "safe_to_trade",
        False,
    )
)


# =========================================================
# TARGET-CANDLE LOCK
# =========================================================

target_key = _lock_key(
    timeframe,
    next_candle_start,
)

existing_lock = (
    GLOBAL_MEMORY[
        "locks"
    ].get(
        target_key
    )
)


# ---------------------------------------------------------
# Candidate must be a genuine trade
# ---------------------------------------------------------

candidate_for_lock = (
    raw_signal
    if (
        raw_safe_to_trade
        and raw_signal
        in (
            "BUY CE",
            "BUY PE",
        )
    )
    else "WAIT"
)


# ---------------------------------------------------------
# Direction must agree with forecast
# ---------------------------------------------------------

direction_agrees = (
    (
        candidate_for_lock
        == "BUY CE"
        and raw_forecast
        == "UP"
    )
    or (
        candidate_for_lock
        == "BUY PE"
        and raw_forecast
        == "DOWN"
    )
)


# ---------------------------------------------------------
# Quality requirement
# ---------------------------------------------------------

quality_ok = (
    candidate_for_lock
    in (
        "BUY CE",
        "BUY PE",
    )
    and direction_agrees
    and raw_confidence >= 58
    and raw_forecast_confidence >= 56
)


# ---------------------------------------------------------
# Track consecutive stable scans in GLOBAL memory.
# Therefore browser refresh does not automatically reset
# confirmation count while server process remains alive.
# ---------------------------------------------------------

candidate_state = (
    GLOBAL_MEMORY[
        "candidates"
    ].get(
        target_key,
        {
            "signal": "WAIT",
            "count": 0,
        },
    )
)


if quality_ok:
    if (
        candidate_state.get(
            "signal"
        )
        == candidate_for_lock
    ):
        candidate_state[
            "count"
        ] = (
            int(
                candidate_state.get(
                    "count",
                    0,
                )
            )
            + 1
        )

    else:
        candidate_state = {
            "signal":
                candidate_for_lock,

            "count":
                1,
        }

else:
    # Unsafe/WAIT does not lock.
    # Also reset provisional confirmation.
    candidate_state = {
        "signal": "WAIT",
        "count": 0,
    }


GLOBAL_MEMORY[
    "candidates"
][target_key] = (
    candidate_state
)


# ---------------------------------------------------------
# Normal setup = 3 stable scans.
# Fast high-quality setup = 2 scans.
# No clock-window restriction.
# ---------------------------------------------------------

required_scans = (
    2
    if (
        raw_fast_setup
        and raw_confidence >= 62
        and raw_forecast_confidence >= 60
    )
    else 3
)


# ---------------------------------------------------------
# LOCK ONCE.
# After this point NO rerun can replace CE with WAIT/PE
# for the same target candle.
# ---------------------------------------------------------

if (
    existing_lock is None
    and quality_ok
    and candidate_state[
        "count"
    ] >= required_scans
):
    existing_lock = _save_lock(
        target_key,
        next_candle_start,
        result,
        current_price,
        raw_market_state,
    )

    try:
        locked_forecast = (
            existing_lock[
                "forecast"
            ]
        )

        if locked_forecast in (
            "UP",
            "DOWN",
        ):
            log_prediction(
                next_candle_start,
                locked_forecast,
                raw_market_state,
                current_price,
            )

            GLOBAL_MEMORY[
                "logs"
            ][target_key] = {
                "forecast":
                    locked_forecast,

                "market_state":
                    raw_market_state,

                "price":
                    current_price,

                "resolved":
                    False,
            }

    except Exception:
        pass


# =========================================================
# DISPLAY VALUES
# =========================================================

lock_is_active = (
    existing_lock is not None
)


if lock_is_active:
    display_signal = (
        existing_lock[
            "signal"
        ]
    )

    next_candle = (
        existing_lock[
            "forecast"
        ]
    )

    forecast_confidence = int(
        existing_lock[
            "forecast_confidence"
        ]
    )

    display_trend = (
        existing_lock[
            "trend"
        ]
    )

    confidence = int(
        existing_lock[
            "confidence"
        ]
    )

    entry_prediction = float(
        existing_lock[
            "entry"
        ]
    )

    invalidation = float(
        existing_lock[
            "invalidation"
        ]
    )

    prediction_target = float(
        existing_lock[
            "target"
        ]
    )

    ce_probability = int(
        existing_lock[
            "ce_probability"
        ]
    )

    pe_probability = int(
        existing_lock[
            "pe_probability"
        ]
    )

    reasons = list(
        existing_lock[
            "reasons"
        ]
    )

else:
    display_signal = "WAIT"

    next_candle = (
        raw_forecast
    )

    forecast_confidence = (
        raw_forecast_confidence
    )

    display_trend = (
        raw_trend
    )

    confidence = (
        raw_confidence
    )

    entry_prediction = (
        current_price
    )

    invalidation = (
        current_price
    )

    prediction_target = (
        current_price
    )

    ce_probability = (
        raw_ce
    )

    pe_probability = (
        raw_pe
    )

    reasons = (
        raw_reasons
    )


# =========================================================
# BUY / SELL DISPLAY
# =========================================================

if display_signal == "BUY CE":
    buy_above = (
        entry_prediction
    )

    sell_below = (
        invalidation
    )

elif display_signal == "BUY PE":
    buy_above = (
        invalidation
    )

    sell_below = (
        entry_prediction
    )

else:
    buy_above = (
        current_price + 1.0
    )

    sell_below = (
        current_price - 1.0
    )


try:
    option = suggest_option(
        display_signal,
        current_price,
    )

except Exception:
    option = "—"


risk_points = abs(
    entry_prediction
    - invalidation
)


# =========================================================
# DASHBOARD
# =========================================================

c1, c2, c3, c4 = (
    st.columns(4)
)

c5, c6, c7, c8 = (
    st.columns(4)
)

c9, c10, c11, c12 = (
    st.columns(4)
)

c13, c14 = (
    st.columns(2)
)


c1.metric(
    "NIFTY",
    f"{current_price:.2f}",
)

c2.metric(
    "Trend",
    display_trend,
)

c3.metric(
    "Signal",
    display_signal,
)

c4.metric(
    "Trade Confidence",
    f"{confidence}%",
)


c5.metric(
    "RSI",
    f"{float(latest.get('RSI', 0)):.2f}",
)

c6.metric(
    "MACD",
    f"{float(latest.get('MACD', 0)):.2f}",
)

c7.metric(
    "VWAP",
    f"{float(latest.get('VWAP', current_price)):.2f}",
)

c8.metric(
    "Option",
    option,
)


c9.metric(
    "Next Candle",
    next_candle,
)

c10.metric(
    "Forecast Confidence",
    f"{forecast_confidence}%",
)

c11.metric(
    "CE Probability",
    f"{ce_probability}%",
)

c12.metric(
    "PE Probability",
    f"{pe_probability}%",
)


c13.metric(
    "BUY ABOVE",
    f"{buy_above:.2f}",
)

c14.metric(
    "SELL BELOW",
    f"{sell_below:.2f}",
)


# =========================================================
# MARKET STATE
# =========================================================

m1, m2 = st.columns(2)

m1.metric(
    "Market State",
    raw_market_state,
)

m2.metric(
    "State Reliability",
    f"{raw_state_reliability:.0%}",
)


# =========================================================
# WALL INFO
# =========================================================

w1, w2, w3, w4 = (
    st.columns(4)
)

w1.metric(
    "Next Wall",
    raw_next_wall,
)

w2.metric(
    "Wall Price",
    (
        f"{raw_next_wall_price:.2f}"
        if raw_next_wall_price > 0
        else "—"
    ),
)

w3.metric(
    "Wall Distance",
    (
        f"{raw_wall_distance:.1f} pts"
        if raw_wall_distance < 900
        else "—"
    ),
)

w4.metric(
    "Expected Wall Reaction",
    (
        f"{raw_wall_reaction} "
        f"({raw_wall_reaction_confidence:.0f}%)"
    ),
)


# =========================================================
# LOCK STATUS
# =========================================================

st.divider()

if lock_is_active:
    if display_signal == "BUY CE":
        st.success(
            f"🔒 LOCKED BUY CE for "
            f"{next_candle_start.strftime('%H:%M')}–"
            f"{next_candle_end.strftime('%H:%M')} | "
            f"Entry {entry_prediction:.2f} | "
            f"Target {prediction_target:.2f} | "
            f"Invalidation {invalidation:.2f}"
        )

    elif display_signal == "BUY PE":
        st.error(
            f"🔒 LOCKED BUY PE for "
            f"{next_candle_start.strftime('%H:%M')}–"
            f"{next_candle_end.strftime('%H:%M')} | "
            f"Entry {entry_prediction:.2f} | "
            f"Target {prediction_target:.2f} | "
            f"Invalidation {invalidation:.2f}"
        )

else:
    scan_count = int(
        candidate_state.get(
            "count",
            0,
        )
    )

    if candidate_for_lock in (
        "BUY CE",
        "BUY PE",
    ):
        st.warning(
            f"🟡 CONFIRMING {candidate_for_lock}: "
            f"{scan_count}/{required_scans} stable scans | "
            f"Next candle {next_candle_start.strftime('%H:%M')}–"
            f"{next_candle_end.strftime('%H:%M')}"
        )

    else:
        st.warning(
            f"🟡 ANALYSING NEXT CANDLE "
            f"{next_candle_start.strftime('%H:%M')}–"
            f"{next_candle_end.strftime('%H:%M')} | "
            f"Signal not safe enough to lock yet."
        )


# =========================================================
# CHART
# =========================================================

st.divider()
st.subheader("NIFTY Chart")

try:
    fig = create_chart(
        df
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="nifty_live_chart",
    )

except Exception as e:
    st.error(
        f"Chart error: {e}"
    )

    st.code(
        traceback.format_exc(),
        language="text",
    )


# =========================================================
# TRADE INFO
# =========================================================

st.divider()
st.subheader(
    "Trade Information"
)

t1, t2, t3, t4 = (
    st.columns(4)
)

t1.metric(
    "Entry",
    f"{entry_prediction:.2f}",
)

t2.metric(
    "Invalidation",
    f"{invalidation:.2f}",
)

t3.metric(
    "Target",
    f"{prediction_target:.2f}",
)

t4.metric(
    "Risk",
    f"{risk_points:.2f} pts",
)


# =========================================================
# SUPPORT / RESISTANCE
# =========================================================

st.divider()

st.subheader(
    "Support / Resistance"
)

s1, s2, s3 = (
    st.columns(3)
)

s1.metric(
    "Chart Support",
    f"{support:.2f}",
)

s2.metric(
    "Current",
    f"{current_price:.2f}",
)

s3.metric(
    "Chart Resistance",
    f"{resistance:.2f}",
)


# =========================================================
# OPTION CHAIN
# =========================================================

st.subheader(
    "Option Chain / OI Levels"
)

oi_support = (
    oc_details.get(
        "oi_support"
    )
)

oi_resistance = (
    oc_details.get(
        "oi_resistance"
    )
)

pcr = float(
    oc_details.get(
        "pcr",
        0.0,
    )
)


o1, o2, o3, o4, o5 = (
    st.columns(5)
)

o1.metric(
    "CE Bias",
    f"{bull_oc:.2f}",
)

o2.metric(
    "PE Bias",
    f"{bear_oc:.2f}",
)

o3.metric(
    "OI Support",
    (
        f"{oi_support:.0f}"
        if oi_support
        else "—"
    ),
)

o4.metric(
    "OI Resistance",
    (
        f"{oi_resistance:.0f}"
        if oi_resistance
        else "—"
    ),
)

o5.metric(
    "PCR",
    f"{pcr:.2f}",
)


if oc_reasons:
    with st.expander(
        "Option Chain Details"
    ):
        for reason in oc_reasons:
            st.write(
                f"• {reason}"
            )


# =========================================================
# ANALYSIS
# =========================================================

st.subheader(
    "AI Prediction Analysis"
)

if reasons:
    for reason in reasons[:24]:
        st.write(
            f"• {reason}"
        )

else:
    st.write(
        "No analysis reasons available."
    )


# =========================================================
# MARKET DATA
# =========================================================

st.subheader(
    "Latest Market Data"
)

st.dataframe(
    _safe_display_df(
        df.tail(10)
    ),
    use_container_width=True,
)


# =========================================================
# LIVE TICK
# =========================================================

st.subheader(
    "Live Tick"
)

try:
    tick = getattr(
        live_data,
        "latest_tick",
        None,
    )

    st.write(
        tick
        if tick
        else "REST/live market fallback active"
    )

except Exception:
    st.write(
        "REST/live market fallback active"
    )


# =========================================================
# FOOTER
# =========================================================

st.caption(
    f"Current candle: "
    f"{current_candle.strftime('%H:%M')}–"
    f"{next_candle_start.strftime('%H:%M')} | "
    f"Prediction target: "
    f"{next_candle_start.strftime('%H:%M')}–"
    f"{next_candle_end.strftime('%H:%M')} | "
    f"Next candle in "
    f"{max(seconds_to_next, 0):.0f}s"
)
