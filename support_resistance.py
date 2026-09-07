import math
import numpy as np
import pandas as pd

def _safe(v,d=0.0):
    try:
        x=float(v); return x if math.isfinite(x) else float(d)
    except Exception:return float(d)

def _atr(df,n=10):
    if df is None or df.empty:return 5.0
    h=pd.to_numeric(df["High"],errors="coerce");l=pd.to_numeric(df["Low"],errors="coerce");c=pd.to_numeric(df["Close"],errors="coerce")
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return max(_safe(tr.tail(n).mean(),5),1)

def _cluster(vals,tol):
    groups=[]
    for v in sorted(float(x) for x in vals):
        if not groups or abs(v-np.mean(groups[-1]))>tol:groups.append([v])
        else:groups[-1].append(v)
    return [(float(np.mean(g)),len(g)) for g in groups]

def get_support_resistance_details(df):
    if df is None or df.empty:return {"support":0.0,"resistance":0.0,"zones":[]}
    current=_safe(df.iloc[-1]["Close"])
    base=df.iloc[:-1].copy() if len(df)>1 else df.copy()
    if base.empty:return {"support":current,"resistance":current,"zones":[]}
    atr=_atr(base);tol=max(1.5,atr*.32)
    groups=_cluster(list(base["High"])+list(base["Low"]),tol)
    day_high=_safe(base["High"].max());day_low=_safe(base["Low"].min())
    zones=[]
    for price,touches in groups:
        if touches>=2 or abs(price-day_high)<=tol or abs(price-day_low)<=tol:
            dist=abs(current-price);strength=min(1,touches/6)*.72+(1/(1+dist/max(atr,1)))*.28
            zones.append({"price":price,"touches":touches,"strength":strength,"distance":dist})
    sups=[z for z in zones if z["price"]<current-tol*.15]
    ress=[z for z in zones if z["price"]>current+tol*.15]
    support=max(sups,key=lambda z:z["price"])["price"] if sups else day_low
    resistance=min(ress,key=lambda z:z["price"])["price"] if ress else day_high
    return {"support":float(support),"resistance":float(resistance),"zones":sorted(zones,key=lambda z:z["distance"]),
            "day_high":day_high,"day_low":day_low,"atr":atr,"tolerance":tol}

def get_support_resistance(df,lookback=50):
    d=get_support_resistance_details(df);return d["support"],d["resistance"]
