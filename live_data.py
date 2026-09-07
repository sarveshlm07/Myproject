import os
import threading
from kiteconnect import KiteTicker

API_KEY = os.getenv("NIFTY_API_KEY")
ACCESS_TOKEN = os.getenv("NIFTY_ACCESS_TOKEN")
NIFTY_TOKEN = 256265
latest_tick = {}
_kws = None
_started = False


def get_live_price():
    try:
        value = latest_tick.get("last_price")
        return float(value) if value is not None else None
    except Exception:
        return None


def on_ticks(ws, ticks):
    global latest_tick
    if ticks:
        latest_tick.clear()
        latest_tick.update(ticks[0])


def on_connect(ws, response):
    ws.subscribe([NIFTY_TOKEN])
    ws.set_mode(ws.MODE_FULL, [NIFTY_TOKEN])


def on_error(ws, code, reason):
    print("WebSocket error:", code, reason)


def start_websocket():
    global _kws, _started
    if _started:
        return
    if not API_KEY or not ACCESS_TOKEN:
        raise RuntimeError("Missing NIFTY_API_KEY / NIFTY_ACCESS_TOKEN")
    _kws = KiteTicker(API_KEY, ACCESS_TOKEN)
    _kws.on_ticks = on_ticks
    _kws.on_connect = on_connect
    _kws.on_error = on_error
    _started = True
    thread = threading.Thread(target=_kws.connect, kwargs={"threaded": True}, daemon=True)
    thread.start()
