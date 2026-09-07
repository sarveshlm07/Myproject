# option_chain_ai.py
import re

def _num(v):
    try:return float(str(v).replace(",","").strip())
    except Exception:return 0.0

def _strike(symbol):
    m=re.search(r"(\d+(?:\.\d+)?)(?:CE|PE)$",str(symbol).upper())
    return float(m.group(1)) if m else 0.0

def _collect(data):
    ce=[];pe=[]
    for symbol,d in (data or {}).items():
        if not isinstance(d,dict):continue
        s=_strike(symbol)
        if s<=0:continue
        x={"symbol":str(symbol).upper(),"strike":s,"oi":_num(d.get("oi")),"change_oi":_num(d.get("change_oi")),
           "volume":_num(d.get("volume")),"buy_qty":_num(d.get("buy_quantity")),"sell_qty":_num(d.get("sell_quantity")),
           "last_price":_num(d.get("last_price")),"oi_slope":_num(d.get("oi_slope")),"price_slope":_num(d.get("price_slope")),
           "price_change_short":_num(d.get("price_change_short")),"volume_change_short":_num(d.get("volume_change_short"))}
        (ce if x["symbol"].endswith("CE") else pe).append(x)
    return ce,pe

def _rank(items,spot,side):
    if not items:return []
    cand=[x for x in items if (x["strike"]>=spot if side=="CE" else x["strike"]<=spot)] or list(items)
    toi=max(sum(max(x["oi"],0) for x in cand),1.0);tv=max(sum(max(x["volume"],0) for x in cand),1.0)
    out=[]
    for x in cand:
        dist=abs(x["strike"]-spot); prox=1/(1+dist/100.0)
        build=max(x["oi_slope"],0); unwind=max(-x["oi_slope"],0)
        strength=(max(x["oi"],0)/toi)*.50+(max(x["volume"],0)/tv)*.10+prox*.20+min(build/5000,1)*.15-min(unwind/5000,1)*.10+min(abs(x["price_slope"])/2,1)*.05
        out.append({**x,"distance":dist,"strength":strength})
    return sorted(out,key=lambda z:z["strength"],reverse=True)

def analyze_option_chain(option_data,spot_price=None,return_details=False):
    calls,puts=_collect(option_data)
    if not calls and not puts:
        d={"pcr":0.0,"oi_support":None,"oi_resistance":None,"support_strength":0.0,"resistance_strength":0.0,
           "support_behavior":"UNKNOWN","resistance_behavior":"UNKNOWN","support_hold_score":0.0,"support_break_score":0.0,
           "resistance_hold_score":0.0,"resistance_break_score":0.0,"top_call_walls":[],"top_put_walls":[]}
        r=(0.0,0.0,["No valid CE/PE option-chain data"]); return (*r,d) if return_details else r
    spot=_num(spot_price) or sum(x["strike"] for x in calls+puts)/max(len(calls)+len(puts),1)
    tc=sum(max(x["oi"],0) for x in calls);tp=sum(max(x["oi"],0) for x in puts);pcr=tp/tc if tc>0 else 0.0
    cr=_rank(calls,spot,"CE");pr=_rank(puts,spot,"PE");cw=cr[0] if cr else None;pw=pr[0] if pr else None
    bull=bear=0.0;reasons=[f"PCR: {pcr:.2f}"]
    if 1.05<=pcr<=1.45:bull+=2
    elif 0<pcr<=.90:bear+=2
    sh=sb=rh=rb=0.0
    if pw:
        if pw["oi_slope"]>0: sh+=4;reasons.append(f"PE OI building at {pw['strike']:.0f}: support strengthening")
        elif pw["oi_slope"]<0: sb+=4;reasons.append(f"PE OI unwinding at {pw['strike']:.0f}: support weakening")
        if pw["oi_slope"]>0 and pw["price_slope"]>0: sh+=1
        elif pw["oi_slope"]<0 and pw["price_slope"]>0: sb+=1.5
    if cw:
        if cw["oi_slope"]>0: rh+=4;reasons.append(f"CE OI building at {cw['strike']:.0f}: resistance strengthening")
        elif cw["oi_slope"]<0: rb+=4;reasons.append(f"CE OI unwinding at {cw['strike']:.0f}: resistance weakening")
        if cw["oi_slope"]>0 and cw["price_slope"]>0: rh+=1
        elif cw["oi_slope"]<0 and cw["price_slope"]<0: rb+=1.5
    bull+=min(8,sh+rb);bear+=min(8,rh+sb)
    call_slope=sum(x["oi_slope"] for x in calls);put_slope=sum(x["oi_slope"] for x in puts)
    if put_slope>0:bull+=min(3,put_slope/10000)
    elif put_slope<0:bear+=min(3,abs(put_slope)/10000)
    if call_slope>0:bear+=min(3,call_slope/10000)
    elif call_slope<0:bull+=min(3,abs(call_slope)/10000)
    bull=max(0,min(bull,15));bear=max(0,min(bear,15))
    sbhv="HOLD/BOUNCE" if sh>=sb+1.5 else "BREAK" if sb>=sh+1.5 else "UNKNOWN"
    rbhv="HOLD/REJECT" if rh>=rb+1.5 else "BREAKOUT" if rb>=rh+1.5 else "UNKNOWN"
    d={"pcr":pcr,"total_call_oi":tc,"total_put_oi":tp,"call_oi_change":sum(x["change_oi"] for x in calls),
       "put_oi_change":sum(x["change_oi"] for x in puts),"oi_support":pw["strike"] if pw else None,"oi_resistance":cw["strike"] if cw else None,
       "support_strength":pw["strength"] if pw else 0.0,"resistance_strength":cw["strength"] if cw else 0.0,
       "support_distance":spot-pw["strike"] if pw else None,"resistance_distance":cw["strike"]-spot if cw else None,
       "support_behavior":sbhv,"resistance_behavior":rbhv,"support_hold_score":sh,"support_break_score":sb,
       "resistance_hold_score":rh,"resistance_break_score":rb,"call_oi_slope":call_slope,"put_oi_slope":put_slope,
       "top_call_walls":[{"strike":x["strike"],"oi":x["oi"],"oi_slope":x["oi_slope"],"strength":x["strength"]} for x in cr[:5]],
       "top_put_walls":[{"strike":x["strike"],"oi":x["oi"],"oi_slope":x["oi_slope"],"strength":x["strength"]} for x in pr[:5]]}
    reasons.append("FINAL OPTION BIAS: BULLISH" if bull>bear+2 else "FINAL OPTION BIAS: BEARISH" if bear>bull+2 else "FINAL OPTION BIAS: MIXED")
    r=(bull,bear,reasons);return (*r,d) if return_details else r
