from datetime import datetime
from zoneinfo import ZoneInfo
import os

import pandas as pd
import streamlit as st
from kiteconnect import KiteConnect

import live_data

NIFTY_TOKEN = 256265
IST = ZoneInfo("Asia/Kolkata")


def _get_secret(name):
    # Streamlit Community Cloud
    try:
        value = st.secrets.get(name)
        if value:
            return str(value).strip()
    except Exception:
        pass

    # Local / environment fallback
    value = os.getenv(name)
    if value:
        return str(value).strip()

    return None


API_KEY = _get_secret("NIFTY_API_KEY")
ACCESS_TOKEN = _get_secret("NIFTY_ACCESS_TOKEN")

kite = None

if API_KEY and ACCESS_TOKEN:
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)


def _require_kite():
    if not API_KEY:
        raise RuntimeError("NIFTY_API_KEY missing from Streamlit Secrets.")

    if not ACCESS_TOKEN:
        raise RuntimeError("NIFTY_ACCESS_TOKEN missing from Streamlit Secrets.")

    if kite is None:
        raise RuntimeError("Could not initialize Zerodha KiteConnect.")


def _interval_info(interval):
    mapping = {
        "1m": ("minute", 60),
        "3m": ("3minute", 180),
        "5m": ("5minute", 300),
    }
    return mapping.get(interval, mapping["5m"])


def _current_bucket(now, seconds):
    anchor = now.replace(
        hour=9,
        minute=15,
        second=0,
        microsecond=0
    )

    if now <= anchor:
        return pd.Timestamp(anchor)

    elapsed = int((now - anchor).total_seconds())

    return (
        pd.Timestamp(anchor)
        + pd.Timedelta(
            seconds=(elapsed // seconds) * seconds
        )
    )


def _clean(df):
    if df is None or df.empty:
        return pd.DataFrame()

    x = df.copy()

    x.rename(
        columns={
            "date": "Datetime",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )

    if "Datetime" not in x.columns:
        return pd.DataFrame()

    x["Datetime"] = pd.to_datetime(
        x["Datetime"],
        errors="coerce"
    )

    x = (
        x.dropna(subset=["Datetime"])
        .set_index("Datetime")
        .sort_index()
    )

    for col in (
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ):
        if col not in x.columns:
            x[col] = 0.0

        x[col] = pd.to_numeric(
            x[col],
            errors="coerce"
        )

    x = x.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    x["Volume"] = x["Volume"].fillna(0.0)

    return x[
        ["Open", "High", "Low", "Close", "Volume"]
    ]


def _align_timestamp(ts, index):
    ts = pd.Timestamp(ts)

    if (
        not isinstance(index, pd.DatetimeIndex)
        or len(index) == 0
    ):
        return ts

    if index.tz is not None and ts.tz is None:
        return ts.tz_localize(index.tz)

    if index.tz is None and ts.tz is not None:
        return ts.tz_localize(None)

    if index.tz is not None and ts.tz is not None:
        return ts.tz_convert(index.tz)

    return ts


def _ensure_single_running_candle(
    df,
    interval,
    now
):
    if df is None or df.empty:
        return df

    _, seconds = _interval_info(interval)

    bucket = _align_timestamp(
        _current_bucket(now, seconds),
        df.index
    )

    # live_data may be running in REST-only mode
    live_price = None

    try:
        if hasattr(live_data, "get_live_price"):
            live_price = live_data.get_live_price()

        elif hasattr(live_data, "get_latest_tick"):
            tick = live_data.get_latest_tick()

            if isinstance(tick, dict):
                live_price = tick.get("last_price")
            elif isinstance(tick, (int, float)):
                live_price = float(tick)

    except Exception:
        live_price = None

    x = df.copy()

    if bucket in x.index:
        row = x.loc[bucket].copy()

        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]

        if live_price is not None:
            live_price = float(live_price)

            x.loc[bucket, "High"] = max(
                float(row["High"]),
                live_price
            )

            x.loc[bucket, "Low"] = min(
                float(row["Low"]),
                live_price
            )

            x.loc[bucket, "Close"] = live_price

        return (
            x[~x.index.duplicated(keep="last")]
            .sort_index()
        )

    prev = x.iloc[-1]

    open_price = float(prev["Close"])

    if live_price is None:
        live_price = open_price

    live_price = float(live_price)

    running = pd.DataFrame(
        [{
            "Open": open_price,
            "High": max(open_price, live_price),
            "Low": min(open_price, live_price),
            "Close": live_price,
            "Volume": 0.0,
        }],
        index=pd.DatetimeIndex([bucket])
    )

    x = pd.concat([x, running])

    return (
        x[~x.index.duplicated(keep="last")]
        .sort_index()
    )


def get_nifty_data(interval="5m"):
    _require_kite()

    kite_interval, _ = _interval_info(interval)

    now = datetime.now(IST)

    market_open = now.replace(
        hour=9,
        minute=15,
        second=0,
        microsecond=0
    )

    market_close = now.replace(
        hour=15,
        minute=30,
        second=0,
        microsecond=0
    )

    if now < market_open:
        return pd.DataFrame()

    to_time = min(now, market_close)

    candles = kite.historical_data(
        instrument_token=NIFTY_TOKEN,
        from_date=market_open,
        to_date=to_time,
        interval=kite_interval,
    )

    df = _clean(pd.DataFrame(candles))

    if df.empty:
        return df

    try:
        today = now.date()
        df = df[df.index.date == today].copy()
    except Exception:
        pass

    if df.empty:
        return df

    if now <= market_close:
        df = _ensure_single_running_candle(
            df,
            interval,
            now
        )

    return df
