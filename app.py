import streamlit as st
from streamlit_autorefresh import st_autorefresh
import traceback

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


st.set_page_config(page_title="NiftyAI", page_icon="📈", layout="wide")
st_autorefresh(interval=1000, key="niftyai_refresh")

if "ws_started" not in st.session_state:
    try:
        live_data.start_websocket()
    except Exception:
        pass
    st.session_state.ws_started = True

st.title("🤖 NiftyAI")
timeframe = st.selectbox("Select Timeframe", ["1m", "3m", "5m"], index=2)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "active_candle": None,

    # Raw candidate confirmation
    "candidate_signal": "WAIT",
    "candidate_count": 0,

    # Next-candle directional forecast has its own lock.
    "forecast_candidate": "UNCLEAR",
    "forecast_count": 0,
    "forecast_locked": False,
    "locked_forecast": "UNCLEAR",
    "locked_forecast_confidence": 0,

    # One locked trade prediction per running candle
    "prediction_locked": False,
    "locked_signal": "WAIT",
    "locked_confidence": 0,
    "locked_trend": "Scanning",
    "locked_entry": 0.0,
    "locked_invalidation": 0.0,
    "locked_target": 0.0,
    "locked_ce_probability": 50,
    "locked_pe_probability": 50,
    "locked_reasons": [],
    "locked_score": 0.0,
    "locked_next_candle": "UNCLEAR",

    # Latest raw values for display before lock
    "raw_confidence": 0,
    "raw_trend": "Scanning",
    "raw_ce_probability": 50,
    "raw_pe_probability": 50,
    "raw_reasons": [],
    "raw_score": 0.0,
    "raw_next_candle": "UNCLEAR",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_prediction_for_new_candle(candle):
    st.session_state.active_candle = candle
    st.session_state.candidate_signal = "WAIT"
    st.session_state.candidate_count = 0

    st.session_state.forecast_candidate = "UNCLEAR"
    st.session_state.forecast_count = 0
    st.session_state.forecast_locked = False
    st.session_state.locked_forecast = "UNCLEAR"
    st.session_state.locked_forecast_confidence = 0

    st.session_state.prediction_locked = False

    st.session_state.locked_signal = "WAIT"
    st.session_state.locked_confidence = 0
    st.session_state.locked_trend = "Scanning"
    st.session_state.locked_entry = 0.0
    st.session_state.locked_invalidation = 0.0
    st.session_state.locked_target = 0.0
    st.session_state.locked_ce_probability = 50
    st.session_state.locked_pe_probability = 50
    st.session_state.locked_reasons = []
    st.session_state.locked_score = 0.0
    st.session_state.locked_next_candle = "UNCLEAR"


if st.button("Reset Trade"):
    for key, value in defaults.items():
        st.session_state[key] = value
    st.rerun()


# =========================================================
# MARKET DATA
# =========================================================

try:
    df = get_nifty_data(timeframe)
except Exception as e:
    st.error(f"Market data error: {e}")
    st.stop()

if df is None or df.empty:
    st.warning("Waiting for Zerodha market data...")
    st.stop()


# =========================================================
# INDICATORS
# No 20/30 candle hard stop.
# =========================================================

try:
    df = calculate_indicators(df)
except Exception as e:
    st.error(f"Indicator calculation error: {e}")
    st.code(traceback.format_exc(), language="text")
    st.stop()

if df is None or df.empty:
    st.warning("Indicator data unavailable.")
    st.stop()


latest = df.iloc[-1]
current_candle = df.index[-1]
current_price = float(latest.get("Close", 0))


# New running candle = new prediction cycle.
if st.session_state.active_candle != current_candle:
    reset_prediction_for_new_candle(current_candle)


# =========================================================
# SUPPORT / RESISTANCE
# =========================================================

try:
    support, resistance = get_support_resistance(df)
except Exception:
    support = float(df["Low"].min())
    resistance = float(df["High"].max())


# =========================================================
# OPTION CHAIN
# =========================================================

bull_oc = 0.0
bear_oc = 0.0
oc_reasons = []
oc_details = {}

try:
    atm = get_atm_strike(current_price)
    symbols = get_symbols(atm)
    quotes = get_option_quotes(symbols)
    bull_oc, bear_oc, oc_reasons, oc_details = analyze_option_chain(
        quotes,
        spot_price=current_price,
        return_details=True,
    )
except Exception as e:
    oc_reasons = [f"Option chain unavailable: {e}"]


# =========================================================
# RAW ENGINE SCAN
# Runs every refresh, but does NOT directly flip locked signal.
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
    st.error("❌ PREDICTION ENGINE ERROR")
    st.error(str(e))
    st.code(traceback.format_exc(), language="text")
    st.stop()

if not isinstance(result, dict):
    st.error("Prediction engine returned invalid data.")
    st.write(result)
    st.stop()


raw_signal = result.get("candidate_signal", result.get("signal", "WAIT"))
raw_confidence = int(result.get("confidence", 0))
raw_trend = result.get("trend", "Scanning")
raw_ce = int(result.get("ce_probability", 50))
raw_pe = int(result.get("pe_probability", 50))
raw_reasons = result.get("reasons", [])
raw_score = float(result.get("score", 0))
raw_next = result.get("next_candle", "UNCLEAR")
raw_forecast_confidence = int(result.get("forecast_confidence", raw_confidence))


st.session_state.raw_confidence = raw_confidence
st.session_state.raw_trend = raw_trend
st.session_state.raw_ce_probability = raw_ce
st.session_state.raw_pe_probability = raw_pe
st.session_state.raw_reasons = raw_reasons
st.session_state.raw_score = raw_score
st.session_state.raw_next_candle = raw_next



# =========================================================
# NEXT-CANDLE FORECAST LOCK
#
# Directional forecast is easier to obtain than BUY CE/PE.
# Two consecutive UP/DOWN scans lock the next-candle forecast
# for the current running candle. This stops UP<->DOWN flipping.
# =========================================================

if not st.session_state.forecast_locked:
    if raw_next in ("UP", "DOWN"):
        if st.session_state.forecast_candidate == raw_next:
            st.session_state.forecast_count += 1
        else:
            st.session_state.forecast_candidate = raw_next
            st.session_state.forecast_count = 1

        if st.session_state.forecast_count >= 2:
            st.session_state.forecast_locked = True
            st.session_state.locked_forecast = raw_next
            st.session_state.locked_forecast_confidence = raw_forecast_confidence
    else:
        st.session_state.forecast_candidate = "UNCLEAR"
        st.session_state.forecast_count = 0

# =========================================================
# STABLE CONFIRMATION + LOCK
#
# Requirement:
# - Same running candle can be scanned continuously.
# - BUY CE/PE must appear in TWO consecutive scans.
# - Once locked, it cannot flip until the next candle starts.
# - WAIT is not permanently locked; engine can continue scanning.
# =========================================================

if not st.session_state.prediction_locked:
    if raw_signal in ("BUY CE", "BUY PE"):
        if st.session_state.candidate_signal == raw_signal:
            st.session_state.candidate_count += 1
        else:
            st.session_state.candidate_signal = raw_signal
            st.session_state.candidate_count = 1

        if st.session_state.candidate_count >= 2:
            st.session_state.prediction_locked = True
            st.session_state.locked_signal = raw_signal
            st.session_state.locked_confidence = raw_confidence
            st.session_state.locked_trend = raw_trend
            st.session_state.locked_entry = float(result.get("entry", current_price))
            st.session_state.locked_invalidation = float(
                result.get("invalidation", current_price)
            )
            st.session_state.locked_target = float(
                result.get("target", current_price)
            )
            st.session_state.locked_ce_probability = raw_ce
            st.session_state.locked_pe_probability = raw_pe
            st.session_state.locked_reasons = raw_reasons
            st.session_state.locked_score = raw_score
            st.session_state.locked_next_candle = raw_next
    else:
        st.session_state.candidate_signal = "WAIT"
        st.session_state.candidate_count = 0


# =========================================================
# DISPLAY VALUES
# =========================================================

if st.session_state.prediction_locked:
    signal = st.session_state.locked_signal
    trend = st.session_state.locked_trend
    confidence = st.session_state.locked_confidence
    entry_prediction = st.session_state.locked_entry
    invalidation = st.session_state.locked_invalidation
    prediction_target = st.session_state.locked_target
    ce_probability = st.session_state.locked_ce_probability
    pe_probability = st.session_state.locked_pe_probability
    reasons = st.session_state.locked_reasons
    score = st.session_state.locked_score
    next_candle = (
        st.session_state.locked_forecast
        if st.session_state.forecast_locked
        else st.session_state.locked_next_candle
    )
else:
    signal = "WAIT"
    trend = raw_trend
    confidence = raw_confidence
    entry_prediction = current_price
    invalidation = current_price
    prediction_target = current_price
    ce_probability = raw_ce
    pe_probability = raw_pe
    reasons = raw_reasons
    score = raw_score

    # Directional forecast is independent from the stricter trade signal.
    if st.session_state.forecast_locked:
        next_candle = st.session_state.locked_forecast
    elif raw_next in ("UP", "DOWN"):
        next_candle = f"{raw_next} (confirming {st.session_state.forecast_count}/2)"
    else:
        next_candle = "UNCLEAR"


display_signal = signal
display_trend = trend

# A locked trade can invalidate, but it never flips to the opposite side.
if signal == "BUY CE" and current_price <= invalidation:
    display_signal = "WAIT"
    display_trend = "CE Prediction Invalidated"

elif signal == "BUY PE" and current_price >= invalidation:
    display_signal = "WAIT"
    display_trend = "PE Prediction Invalidated"


if display_signal == "BUY CE":
    buy_above = entry_prediction
    sell_below = current_price

elif display_signal == "BUY PE":
    buy_above = current_price
    sell_below = entry_prediction

else:
    buy_above = current_price + 1.0
    sell_below = current_price - 1.0


try:
    option = suggest_option(display_signal, current_price)
except Exception:
    option = "—"


risk_points = abs(current_price - invalidation)


# =========================================================
# DASHBOARD
# =========================================================

c1, c2, c3, c4 = st.columns(4)
c5, c6, c7, c8 = st.columns(4)
c9, c10, c11, c12 = st.columns(4)
c13, c14 = st.columns(2)

c1.metric("NIFTY", f"{current_price:.2f}")
c2.metric("Trend", display_trend)
c3.metric("Signal", display_signal)
c4.metric("Confidence", f"{confidence}%")

c5.metric("RSI", f"{float(latest.get('RSI', 0)):.2f}")
c6.metric("MACD", f"{float(latest.get('MACD', 0)):.2f}")
c7.metric("VWAP", f"{float(latest.get('VWAP', current_price)):.2f}")
c8.metric("Option", option)

c9.metric("Next Candle", next_candle)
c10.metric("CE Probability", f"{ce_probability}%")
c11.metric("PE Probability", f"{pe_probability}%")
c12.metric("Expected Target", f"{prediction_target:.2f}")

c13.metric("BUY ABOVE", f"{buy_above:.2f}")
c14.metric("SELL BELOW", f"{sell_below:.2f}")


# =========================================================
# CHART
# =========================================================

st.divider()
st.subheader("NIFTY Chart")

try:
    fig = create_chart(df)
    st.plotly_chart(fig, use_container_width=True, key="nifty_live_chart")
except Exception as e:
    st.error(f"Chart error: {e}")
    st.code(traceback.format_exc(), language="text")


# =========================================================
# TRADE INFORMATION
# =========================================================

st.divider()
st.subheader("Trade Information")

t1, t2, t3, t4 = st.columns(4)
t1.metric("Entry", f"{entry_prediction:.2f}")
t2.metric("Invalidation", f"{invalidation:.2f}")
t3.metric("Target", f"{prediction_target:.2f}")
t4.metric("Risk", f"{risk_points:.2f} pts")


st.divider()

if display_signal == "BUY CE":
    st.success(
        f"🟢 LOCKED NEXT-CANDLE FORECAST: UP | BUY CE around "
        f"{entry_prediction:.2f} | Target ~{prediction_target:.2f} | "
        f"Invalid below {invalidation:.2f}"
    )

elif display_signal == "BUY PE":
    st.error(
        f"🔴 LOCKED NEXT-CANDLE FORECAST: DOWN | BUY PE around "
        f"{entry_prediction:.2f} | Target ~{prediction_target:.2f} | "
        f"Invalid above {invalidation:.2f}"
    )

elif st.session_state.prediction_locked:
    st.warning("🟡 Locked prediction was invalidated — no opposite flip this candle.")

else:
    st.warning(
        f"🟡 NEXT CANDLE: {next_candle} | "
        f"Trade candidate: {raw_signal} | "
        f"Trade confirmation {st.session_state.candidate_count}/2"
    )


# =========================================================
# S/R + OPTION CHAIN + AI REASONS
# =========================================================

st.subheader("Support / Resistance")
s1, s2, s3 = st.columns(3)
s1.metric("Support", f"{support:.2f}")
s2.metric("Current", f"{current_price:.2f}")
s3.metric("Resistance", f"{resistance:.2f}")


st.subheader("Option Chain Analysis")
o1, o2, o3 = st.columns(3)
o1.metric("CE Bias", f"{bull_oc:.2f}")
o2.metric("PE Bias", f"{bear_oc:.2f}")
o3.metric("OC Difference", f"{bull_oc - bear_oc:.2f}")

oi_support = oc_details.get("oi_support")
oi_resistance = oc_details.get("oi_resistance")
pcr = oc_details.get("pcr", 0.0)

oi1, oi2, oi3 = st.columns(3)
oi1.metric(
    "OI Support",
    f"{oi_support:.0f}" if oi_support else "—"
)
oi2.metric(
    "OI Resistance",
    f"{oi_resistance:.0f}" if oi_resistance else "—"
)
oi3.metric("PCR", f"{pcr:.2f}")

if oc_reasons:
    with st.expander("Option Chain Details"):
        for reason in oc_reasons:
            st.write(f"• {reason}")


st.subheader("AI Prediction Analysis")
if reasons:
    for reason in reasons[:20]:
        st.write(f"• {reason}")
else:
    st.write("No analysis reasons available.")


st.subheader("Latest Market Data")
st.dataframe(df.tail(10), use_container_width=True)


st.subheader("Live Tick")
try:
    tick = live_data.latest_tick
    st.write(tick if tick else "Waiting for live tick...")
except Exception:
    st.write("Waiting for live tick...")


trade_status = "LOCKED" if st.session_state.prediction_locked else "SCANNING"
forecast_status = "LOCKED" if st.session_state.forecast_locked else "CONFIRMING"

st.caption(
    f"Next-candle forecast: {forecast_status} | Trade signal: {trade_status} | "
    f"Current candle: {current_candle} | "
    "Forecast and trade are separate: direction can be shown early, "
    "while BUY CE/PE remains stricter."
)
