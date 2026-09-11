import traceback
import urllib.parse
import urllib.request
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from prediction_logger import log_prediction, log_outcome
from support_resistance import get_support_resistance
from data import get_nifty_data
from indicators import calculate_indicators
from prediction_engine import predict_market
from chart import create_chart
from option_chain import suggest_option, get_atm_strike, get_symbols, get_option_quotes
from option_chain_ai import analyze_option_chain
import live_data


st.set_page_config(page_title="NiftyAI", page_icon="📈", layout="wide")
st_autorefresh(interval=1000, key="niftyai_refresh")


def _safe_display_df(frame):
    out = frame.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype(str)
    return out


def _timeframe_seconds(tf):
    return {"1m": 60, "3m": 180, "5m": 300}.get(tf, 300)


def _now_for_index(index):
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        return pd.Timestamp.now(tz=index.tz)
    return pd.Timestamp.now()


def _same_time(a, b):
    try:
        return pd.Timestamp(a) == pd.Timestamp(b)
    except Exception:
        return False




def _send_telegram_signal(target, signal, result, current_price):
    """Send one Telegram push when a final BUY CE/PE trade is locked.

    Secrets required in Streamlit Cloud:
      TELEGRAM_BOT_TOKEN = "..."
      TELEGRAM_CHAT_ID = "..."
    """
    if signal not in ("BUY CE", "BUY PE"):
        return False

    try:
        bot_token = str(st.secrets.get("TELEGRAM_BOT_TOKEN", "")).strip()
        chat_id = str(st.secrets.get("TELEGRAM_CHAT_ID", "")).strip()
    except Exception:
        return False

    if not bot_token or not chat_id:
        return False

    try:
        target_ts = pd.Timestamp(target) if target is not None else None
        tf_seconds = _timeframe_seconds(timeframe) if "timeframe" in globals() else 300
        if target_ts is not None:
            end_ts = target_ts + pd.Timedelta(seconds=tf_seconds)
            candle_text = f"{target_ts.strftime('%H:%M')}–{end_ts.strftime('%H:%M')}"
            notify_key = f"{target_ts.isoformat()}|{signal}"
        else:
            candle_text = "Next candle"
            notify_key = f"unknown|{signal}"

        sent = st.session_state.get("sent_telegram_notifications", {})
        if notify_key in sent:
            return True

        confidence = int(result.get("confidence", 0))
        forecast = str(result.get("next_candle", "UNCLEAR"))
        forecast_conf = int(result.get("forecast_confidence", 0))
        entry = float(result.get("entry", current_price))
        target_price = float(result.get("target", current_price))
        invalidation = float(result.get("invalidation", current_price))

        emoji = "🟢" if signal == "BUY CE" else "🔴"
        message = (
            f"🚨 NiftyAI SIGNAL\n"
            f"{emoji} {signal}\n"
            f"Next candle: {candle_text}\n"
            f"Trade confidence: {confidence}%\n"
            f"Forecast: {forecast} ({forecast_conf}%)\n"
            f"NIFTY: {float(current_price):.2f}\n"
            f"Entry: {entry:.2f}\n"
            f"Target: {target_price:.2f}\n"
            f"Invalidation: {invalidation:.2f}"
        )

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode("utf-8")
        request = urllib.request.Request(url, data=payload, method="POST")
        with urllib.request.urlopen(request, timeout=8) as response:
            ok = 200 <= int(response.status) < 300

        if ok:
            sent[notify_key] = True
            st.session_state.sent_telegram_notifications = sent
            return True
    except Exception:
        return False

    return False


def _reset_target(target):
    st.session_state.target_candle = target
    st.session_state.forecast_candidate = "UNCLEAR"
    st.session_state.forecast_count = 0
    st.session_state.signal_candidate = "WAIT"
    st.session_state.signal_count = 0
    st.session_state.locked_for_candle = None
    st.session_state.locked_forecast = "UNCLEAR"
    st.session_state.locked_forecast_confidence = 0
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
    st.session_state.locked_market_state = "MIXED"
    st.session_state.trade_locked = False


def _lock_forecast(target, forecast, result, current_price):
    st.session_state.locked_for_candle = target
    st.session_state.locked_forecast = forecast
    st.session_state.locked_forecast_confidence = int(result.get("forecast_confidence", 50))
    st.session_state.locked_trend = result.get("trend", "Scanning")
    st.session_state.locked_ce_probability = int(result.get("ce_probability", 50))
    st.session_state.locked_pe_probability = int(result.get("pe_probability", 50))
    st.session_state.locked_reasons = list(result.get("reasons", []))
    st.session_state.locked_score = float(result.get("score", 0.0))
    st.session_state.locked_market_state = result.get("market_state", "MIXED")

    key = str(pd.Timestamp(target))
    if forecast in ("UP", "DOWN") and key not in st.session_state.pending_predictions:
        try:
            log_prediction(
                target,
                forecast,
                st.session_state.locked_market_state,
                current_price,
                st.session_state.locked_score,
                st.session_state.locked_forecast_confidence,
            )
        except Exception:
            pass
        st.session_state.pending_predictions[key] = {
            "target": target,
            "forecast": forecast,
            "state": st.session_state.locked_market_state,
            "price": float(current_price),
        }


def _lock_trade(signal, result, current_price):
    st.session_state.trade_locked = signal in ("BUY CE", "BUY PE")
    st.session_state.locked_signal = signal if st.session_state.trade_locked else "WAIT"
    st.session_state.locked_confidence = int(result.get("confidence", 0))
    st.session_state.locked_entry = float(result.get("entry", current_price))
    st.session_state.locked_invalidation = float(result.get("invalidation", current_price))
    st.session_state.locked_target = float(result.get("target", current_price))

    # Push exactly once when the final actionable trade is locked.
    if st.session_state.trade_locked:
        _send_telegram_signal(
            st.session_state.target_candle,
            st.session_state.locked_signal,
            result,
            current_price,
        )


DEFAULTS = {
    "ws_started": False,
    "target_candle": None,
    "forecast_candidate": "UNCLEAR",
    "forecast_count": 0,
    "signal_candidate": "WAIT",
    "signal_count": 0,
    "locked_for_candle": None,
    "locked_forecast": "UNCLEAR",
    "locked_forecast_confidence": 0,
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
    "locked_market_state": "MIXED",
    "trade_locked": False,
    "pending_predictions": {},
    "sent_telegram_notifications": {},
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


if not st.session_state.ws_started:
    try:
        live_data.start_websocket()
        st.session_state.ws_started = True
    except Exception as e:
        st.warning(f"Live websocket not ready: {e}")


st.title("🤖 NiftyAI")
timeframe = st.selectbox("Select Timeframe", ["1m", "3m", "5m"], index=2)


# =========================================================
# DATA — data.py is the ONLY owner of the running candle.
# =========================================================
try:
    df = get_nifty_data(timeframe)
except Exception as e:
    st.error(f"Market data error: {e}")
    st.stop()

if df is None or df.empty:
    st.warning("Waiting for Zerodha market data...")
    st.stop()

try:
    df = calculate_indicators(df)
except Exception as e:
    st.error(f"Indicator calculation error: {e}")
    st.code(traceback.format_exc(), language="text")
    st.stop()

if df is None or df.empty:
    st.stop()

latest = df.iloc[-1]
current_candle = pd.Timestamp(df.index[-1])
current_price = float(latest.get("Close", 0.0))
tf_seconds = _timeframe_seconds(timeframe)
next_candle_start = current_candle + pd.Timedelta(seconds=tf_seconds)
now = _now_for_index(df.index)
seconds_to_next = (next_candle_start - now).total_seconds()


# =========================================================
# RESOLVE ONLY TARGET CANDLES THAT HAVE ACTUALLY CLOSED.
# =========================================================
resolved = []
for key, pending in list(st.session_state.pending_predictions.items()):
    try:
        target = pd.Timestamp(pending["target"])
        if target >= current_candle:
            continue
        target_for_index = target
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is not None and target_for_index.tz is None:
                target_for_index = target_for_index.tz_localize(df.index.tz)
            elif df.index.tz is None and target_for_index.tz is not None:
                target_for_index = target_for_index.tz_localize(None)
        if target_for_index in df.index:
            actual_close = float(df.loc[target_for_index, "Close"])
            if isinstance(df.loc[target_for_index, "Close"], pd.Series):
                actual_close = float(df.loc[target_for_index, "Close"].iloc[-1])
            log_outcome(
                target,
                pending["forecast"],
                pending["state"],
                pending["price"],
                actual_close,
            )
            resolved.append(key)
    except Exception:
        pass
for key in resolved:
    st.session_state.pending_predictions.pop(key, None)


# New current candle means a new immediate-next target.
if st.session_state.target_candle is None or not _same_time(st.session_state.target_candle, next_candle_start):
    _reset_target(next_candle_start)


# =========================================================
# SUPPORT / RESISTANCE
# =========================================================
try:
    support, resistance = get_support_resistance(df)
except Exception:
    base = df.iloc[:-1] if len(df) > 1 else df
    support = float(base["Low"].min())
    resistance = float(base["High"].max())


# =========================================================
# OPTION CHAIN / OI
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
        quotes, spot_price=current_price, return_details=True
    )
except Exception as e:
    oc_reasons = [f"Option chain unavailable: {e}"]


# =========================================================
# ENGINE — continuous quality-based analysis.
# No fixed 'first N sec' or 'last 20 sec' locking rule.
# =========================================================
try:
    result = predict_market(
        df, support, resistance, bull_oc, bear_oc, oc_reasons, oc_details
    )
except Exception as e:
    st.error("❌ PREDICTION ENGINE ERROR")
    st.error(str(e))
    st.code(traceback.format_exc(), language="text")
    st.stop()

raw_forecast = result.get("next_candle", "UNCLEAR")
raw_forecast_confidence = int(result.get("forecast_confidence", 50))
raw_forecast_ready = bool(result.get("forecast_ready", False))
raw_signal = result.get("candidate_signal", result.get("signal", "WAIT"))
raw_safe_to_trade = bool(result.get("safe_to_trade", False))
raw_confidence = int(result.get("confidence", 0))
raw_trend = result.get("trend", "Scanning")
raw_ce = int(result.get("ce_probability", 50))
raw_pe = int(result.get("pe_probability", 50))
raw_reasons = list(result.get("reasons", []))
raw_score = float(result.get("score", 0.0))
raw_market_state = result.get("market_state", "MIXED")
raw_state_reliability = float(result.get("state_reliability", 0.50))
raw_next_wall = result.get("next_wall", "—")
raw_next_wall_price = float(result.get("next_wall_price", 0.0))
raw_wall_distance = float(result.get("wall_distance", 999.0))
raw_wall_reaction = result.get("wall_reaction", "UNCLEAR")
raw_wall_reaction_confidence = float(result.get("wall_reaction_confidence", 0.0))


# Forecast stability can mature at ANY point while the current candle is running.
forecast_candidate = raw_forecast if raw_forecast_ready else "UNCLEAR"
if forecast_candidate == st.session_state.forecast_candidate:
    st.session_state.forecast_count += 1
else:
    st.session_state.forecast_candidate = forecast_candidate
    st.session_state.forecast_count = 1

# Trade stability is tracked independently so a locked direction can later become
# an actionable CE/PE without ever flipping the frozen next-candle direction.
signal_candidate = raw_signal if raw_safe_to_trade else "WAIT"
if signal_candidate == st.session_state.signal_candidate:
    st.session_state.signal_count += 1
else:
    st.session_state.signal_candidate = signal_candidate
    st.session_state.signal_count = 1


forecast_locked = (
    st.session_state.locked_for_candle is not None
    and _same_time(st.session_state.locked_for_candle, next_candle_start)
    and st.session_state.locked_forecast in ("UP", "DOWN", "UNCLEAR")
)

if not forecast_locked and forecast_candidate in ("UP", "DOWN"):
    # Stronger setups can mature faster; ordinary setups need more repeated agreement.
    needed = 2 if (raw_safe_to_trade or raw_forecast_confidence >= 68) else 3
    if st.session_state.forecast_count >= needed:
        _lock_forecast(next_candle_start, forecast_candidate, result, current_price)
        forecast_locked = True

# Safety fallback only: if quality never matured, preserve the best directional
# view immediately before the boundary. This is NOT the primary locking method.
if not forecast_locked and 0 <= seconds_to_next <= 1.5:
    fallback = raw_forecast if raw_forecast in ("UP", "DOWN") else "UNCLEAR"
    _lock_forecast(next_candle_start, fallback, result, current_price)
    forecast_locked = True


# Once direction is frozen, only the matching trade can become final.
locked_forecast = st.session_state.locked_forecast if forecast_locked else "UNCLEAR"
matching_signal = (
    (locked_forecast == "UP" and signal_candidate == "BUY CE")
    or (locked_forecast == "DOWN" and signal_candidate == "BUY PE")
)
if forecast_locked and not st.session_state.trade_locked and matching_signal:
    needed_trade = 2 if raw_confidence < 72 else 1
    if st.session_state.signal_count >= needed_trade:
        _lock_trade(signal_candidate, result, current_price)


# =========================================================
# DISPLAY STATE
# =========================================================
if forecast_locked:
    next_candle = st.session_state.locked_forecast
    forecast_confidence = st.session_state.locked_forecast_confidence
    display_trend = st.session_state.locked_trend
    ce_probability = st.session_state.locked_ce_probability
    pe_probability = st.session_state.locked_pe_probability
    reasons = st.session_state.locked_reasons
else:
    next_candle = raw_forecast
    forecast_confidence = raw_forecast_confidence
    display_trend = raw_trend
    ce_probability = raw_ce
    pe_probability = raw_pe
    reasons = raw_reasons

if st.session_state.trade_locked:
    display_signal = st.session_state.locked_signal
    confidence = st.session_state.locked_confidence
    entry_prediction = st.session_state.locked_entry
    invalidation = st.session_state.locked_invalidation
    prediction_target = st.session_state.locked_target
else:
    display_signal = "WAIT"
    confidence = raw_confidence
    entry_prediction = current_price
    invalidation = current_price
    prediction_target = current_price

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
risk_points = abs(entry_prediction - invalidation)


# =========================================================
# DASHBOARD
# =========================================================
c1,c2,c3,c4 = st.columns(4)
c5,c6,c7,c8 = st.columns(4)
c9,c10,c11,c12 = st.columns(4)
c13,c14 = st.columns(2)

c1.metric("NIFTY", f"{current_price:.2f}")
c2.metric("Trend", display_trend)
c3.metric("Signal", display_signal)
c4.metric("Trade Confidence", f"{confidence}%")
c5.metric("RSI", f"{float(latest.get('RSI',0)):.2f}")
c6.metric("MACD", f"{float(latest.get('MACD',0)):.2f}")
c7.metric("VWAP", f"{float(latest.get('VWAP',current_price)):.2f}")
c8.metric("Option", option)
c9.metric("Next Candle", next_candle)
c10.metric("Forecast Confidence", f"{forecast_confidence}%")
c11.metric("CE Probability", f"{ce_probability}%")
c12.metric("PE Probability", f"{pe_probability}%")
c13.metric("BUY ABOVE", f"{buy_above:.2f}")
c14.metric("SELL BELOW", f"{sell_below:.2f}")

m1,m2 = st.columns(2)
m1.metric("Market State", raw_market_state)
m2.metric("State Reliability", f"{raw_state_reliability:.0%}")

w1,w2,w3,w4 = st.columns(4)
w1.metric("Next Wall", raw_next_wall)
w2.metric("Wall Price", f"{raw_next_wall_price:.2f}" if raw_next_wall_price>0 else "—")
w3.metric("Wall Distance", f"{raw_wall_distance:.1f} pts" if raw_wall_distance<900 else "—")
w4.metric("Expected Wall Reaction", f"{raw_wall_reaction} ({raw_wall_reaction_confidence:.0f}%)")

st.divider()
st.subheader("NIFTY Chart")
try:
    st.plotly_chart(create_chart(df), use_container_width=True, key="nifty_live_chart")
except Exception as e:
    st.error(f"Chart error: {e}")

st.divider()
st.subheader("Trade Information")
t1,t2,t3,t4 = st.columns(4)
t1.metric("Entry", f"{entry_prediction:.2f}")
t2.metric("Invalidation", f"{invalidation:.2f}")
t3.metric("Target", f"{prediction_target:.2f}")
t4.metric("Risk", f"{risk_points:.2f} pts")

if st.session_state.trade_locked and display_signal == "BUY CE":
    st.success(f"🟢 FINAL: BUY CE around {entry_prediction:.2f} | Target ~{prediction_target:.2f} | Invalid below {invalidation:.2f}")
elif st.session_state.trade_locked and display_signal == "BUY PE":
    st.error(f"🔴 FINAL: BUY PE around {entry_prediction:.2f} | Target ~{prediction_target:.2f} | Invalid above {invalidation:.2f}")
elif forecast_locked:
    st.warning(f"🟡 NEXT CANDLE FINAL: {next_candle}. Direction frozen; trade stays WAIT until matching ≥5-point setup is safe.")
else:
    st.info(f"🔎 Analysing {next_candle_start.strftime('%H:%M')} next candle continuously. No fixed early/late lock timer.")

st.subheader("Support / Resistance")
s1,s2,s3 = st.columns(3)
s1.metric("Chart Support", f"{support:.2f}")
s2.metric("Current", f"{current_price:.2f}")
s3.metric("Chart Resistance", f"{resistance:.2f}")

st.subheader("Option Chain / OI Levels")
oi_support = oc_details.get("oi_support")
oi_resistance = oc_details.get("oi_resistance")
pcr = float(oc_details.get("pcr",0.0) or 0.0)
o1,o2,o3,o4,o5 = st.columns(5)
o1.metric("CE Bias", f"{bull_oc:.2f}")
o2.metric("PE Bias", f"{bear_oc:.2f}")
o3.metric("OI Support", f"{oi_support:.0f}" if oi_support else "—")
o4.metric("OI Resistance", f"{oi_resistance:.0f}" if oi_resistance else "—")
o5.metric("PCR", f"{pcr:.2f}")

if oc_reasons:
    with st.expander("Option Chain Details"):
        for reason in oc_reasons:
            st.write(f"• {reason}")

st.subheader("AI Prediction Analysis")
for reason in reasons[:24]:
    st.write(f"• {reason}")

st.subheader("Latest Market Data")
st.dataframe(_safe_display_df(df.tail(10)), use_container_width=True)

st.subheader("Live Tick")
st.write(live_data.latest_tick if getattr(live_data,"latest_tick",None) else "Waiting for live tick...")

st.caption(
    f"Current candle: {current_candle.strftime('%H:%M')}–{next_candle_start.strftime('%H:%M')} | "
    f"Prediction target: {next_candle_start.strftime('%H:%M')}–"
    f"{(next_candle_start + pd.Timedelta(seconds=tf_seconds)).strftime('%H:%M')} | "
    f"Time to next: {max(seconds_to_next,0):.0f}s"
)
