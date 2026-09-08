import os

latest_tick = None
_websocket_started = False


def start_websocket():
    """
    Streamlit Community Cloud fallback.

    KiteTicker WebSocket is disabled because the cloud connection
    can fail during WebSocket upgrade.

    The app will continue using REST/historical/live candle data
    from data.py.
    """
    global _websocket_started

    if _websocket_started:
        return

    _websocket_started = True
    print("KiteTicker disabled: using REST data fallback.")


def get_latest_tick():
    return latest_tick
