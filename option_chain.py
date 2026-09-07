# option_chain.py
from datetime import datetime, date
import time
from data import kite

NIFTY_STRIKE_STEP = 50
STRIKE_WINDOW = 300
_INSTRUMENT_CACHE = {"loaded_at": 0.0, "expiry": None, "by_key": {}}
_HISTORY = {}
MAX_HISTORY = 180


def _to_date(value):
    if isinstance(value, date): return value
    for fmt in (None, "%Y-%m-%d"):
        try:
            return datetime.fromisoformat(str(value)).date() if fmt is None else datetime.strptime(str(value), fmt).date()
        except Exception:
            pass
    return None


def get_atm_strike(nifty_price):
    return int(round(float(nifty_price) / NIFTY_STRIKE_STEP) * NIFTY_STRIKE_STEP)


def _load_nifty_option_master(force=False):
    now=time.time()
    if (not force and _INSTRUMENT_CACHE["by_key"] and now-_INSTRUMENT_CACHE["loaded_at"]<600):
        return _INSTRUMENT_CACHE["expiry"], _INSTRUMENT_CACHE["by_key"]
    try: instruments=kite.instruments("NFO")
    except Exception: return _INSTRUMENT_CACHE["expiry"], _INSTRUMENT_CACHE["by_key"]
    today=datetime.now().date(); rows=[]
    for item in instruments:
        if not isinstance(item,dict): continue
        typ=str(item.get("instrument_type","")).upper()
        if str(item.get("name","")).upper()!="NIFTY" or typ not in ("CE","PE"): continue
        expiry=_to_date(item.get("expiry"))
        if expiry is None or expiry<today: continue
        try: strike=int(round(float(item.get("strike"))))
        except Exception: continue
        ts=str(item.get("tradingsymbol","")).strip()
        if ts: rows.append((expiry,strike,typ,f"NFO:{ts}"))
    if not rows: return None,{}
    expiry=min(x[0] for x in rows)
    master={(s,t):sym for e,s,t,sym in rows if e==expiry}
    _INSTRUMENT_CACHE.update({"loaded_at":now,"expiry":expiry,"by_key":master})
    return expiry,master


def get_current_expiry(): return _load_nifty_option_master()[0]

def get_nearby_strikes(atm):
    return list(range(int(atm)-STRIKE_WINDOW,int(atm)+STRIKE_WINDOW+NIFTY_STRIKE_STEP,NIFTY_STRIKE_STEP))

def get_symbols(atm):
    _,m=_load_nifty_option_master(); out=[]
    for strike in get_nearby_strikes(atm):
        for typ in ("CE","PE"):
            s=m.get((int(strike),typ))
            if s: out.append(s)
    return out


def _depth_qty(levels):
    total=0.0
    for x in levels or []:
        if isinstance(x,dict):
            try: total+=float(x.get("quantity",0) or 0)
            except Exception: pass
    return total


def _push(symbol,snapshot):
    h=_HISTORY.setdefault(symbol,[]); h.append(snapshot)
    if len(h)>MAX_HISTORY: del h[:-MAX_HISTORY]


def _hist(symbol):
    h=_HISTORY.get(symbol,[])
    if len(h)<2: return dict(oi_change_short=0.0,price_change_short=0.0,volume_change_short=0.0,oi_slope=0.0,price_slope=0.0)
    back=h[-min(15,len(h))]; last=h[-1]; w=h[-min(30,len(h)):]; n=max(len(w)-1,1)
    return {
      "oi_change_short": last["oi"]-back["oi"],
      "price_change_short": last["last_price"]-back["last_price"],
      "volume_change_short": last["volume"]-back["volume"],
      "oi_slope": (w[-1]["oi"]-w[0]["oi"])/n,
      "price_slope": (w[-1]["last_price"]-w[0]["last_price"])/n,
    }


def get_option_quotes(symbols):
    if not symbols: return {}
    try: raw=kite.quote(symbols)
    except Exception: return {}
    now=time.time(); out={}
    for symbol,data in raw.items():
        if not isinstance(data,dict): continue
        depth=data.get("depth") or {}; ohlc=data.get("ohlc") or {}
        oi=float(data.get("oi",0) or 0); vol=float(data.get("volume",0) or 0); lp=float(data.get("last_price",0) or 0)
        _push(symbol,{"ts":now,"oi":oi,"volume":vol,"last_price":lp}); hf=_hist(symbol)
        prev=max(abs(oi-hf["oi_change_short"]),1.0)
        out[symbol]={
          "symbol":symbol,"last_price":lp,"oi":oi,"change_oi":hf["oi_change_short"],
          "change_oi_pct":hf["oi_change_short"]/prev*100.0,"volume":vol,
          "buy_quantity":_depth_qty(depth.get("buy",[])),"sell_quantity":_depth_qty(depth.get("sell",[])),
          "open":float(ohlc.get("open",0) or 0),"high":float(ohlc.get("high",0) or 0),
          "low":float(ohlc.get("low",0) or 0),"close":float(ohlc.get("close",0) or 0),**hf
        }
    return out


def suggest_option(signal,nifty_price):
    atm=get_atm_strike(nifty_price)
    if signal=="BUY CE": return f"NIFTY {atm} CE"
    if signal=="BUY PE": return f"NIFTY {atm} PE"
    return "No Trade"
