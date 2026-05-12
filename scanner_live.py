"""
scanner_live.py
───────────────
• Scans ALL symbols IN PARALLEL (ThreadPoolExecutor)
• Loops forever — refreshes every 60 seconds by default
• Writes scanner_state.json after every full sweep
• Open scanner_dashboard.html in your browser for the live UI
  (the HTML page polls scanner_state.json every 5 s automatically)

Run:
    python scanner_live.py
    python scanner_live.py --interval 1m --refresh 30
"""

import sys, subprocess, importlib.util, argparse

def _install(pkg):
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for _p in ("pandas", "numpy", "requests"):
    _install(_p)

import os, time, json, threading
import requests
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone


# ══════════════════════════════════════════════════════════
#  SYMBOL LIST  (25 crypto + ~75 US stocks = ~98 total)
# ══════════════════════════════════════════════════════════

CRYPTO = [
    "BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD",
    "DOGE-USD","ADA-USD","AVAX-USD","LINK-USD","DOT-USD",
    "MATIC-USD","LTC-USD","UNI-USD","ATOM-USD","XLM-USD",
    "BCH-USD","ALGO-USD","FIL-USD","NEAR-USD","APT-USD",
    "ARB-USD","OP-USD","INJ-USD","SUI-USD","TIA-USD",
]

STOCKS = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","ORCL","AMD",
    "INTC","QCOM","TXN","MU","AMAT","LRCX","KLAC","MRVL","NXPI","ON",
    "JPM","BAC","GS","MS","WFC","BLK","C","AXP","V","MA",
    "UNH","JNJ","LLY","ABBV","PFE","MRK","TMO","ABT","DHR","BMY",
    "HD","MCD","NKE","SBUX","TGT","WMT","COST","LOW","TJX",
    "XOM","CVX","COP","EOG","SLB",
    "CAT","DE","GE","HON","RTX","BA","LMT","NOC","UPS","FDX",
    "PLTR","CRWD","NET","SNOW","DDOG","ZS","PANW","FTNT","SMCI","ARM",
    "SPY","QQQ","IWM","DIA","XLK",
]

ALL_SYMBOLS = list(dict.fromkeys(CRYPTO + STOCKS))   # deduplicated, order preserved


# ══════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════

@dataclass
class ScannerSettings:
    interval: str = "5m"
    yahoo_range: str = "10d"
    session_preset: str = "AUTO"
    timezone: str = "America/New_York"
    use_htf: bool = True
    htf_rule: str = "15min"
    loading_bars: int = 100
    atr_stop_mult: float = 1.2
    rr_tp1: float = 1.5
    min_stop_percent: float = 0.08
    use_last_closed_bar: bool = False
    max_workers: int = 12        # parallel threads
    refresh_seconds: int = 60    # seconds between full sweeps
    output_json: str = "scanner_state.json"


# ══════════════════════════════════════════════════════════
#  DATA FETCHING
# ══════════════════════════════════════════════════════════

def yahoo_fetch(symbol, settings):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}"
        params = {"interval": settings.interval, "range": settings.yahoo_range, "includePrePost": "true"}
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        r = requests.get(url, params=params, headers=headers, timeout=25)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        data = r.json()
        err = data.get("chart", {}).get("error")
        if err: return None, str(err)
        res = data.get("chart", {}).get("result")
        if not res: return None, "no result"
        res = res[0]
        ts = res.get("timestamp", [])
        qs = res.get("indicators", {}).get("quote", [])
        if not ts or not qs: return None, "no candles"
        q = qs[0]
        df = pd.DataFrame({"Open": q.get("open"), "High": q.get("high"),
                           "Low": q.get("low"), "Close": q.get("close"), "Volume": q.get("volume")})
        df["Time"] = pd.to_datetime(ts, unit="s", utc=True)
        df = df.set_index("Time").dropna(subset=["Open","High","Low","Close"])
        df["Volume"] = df["Volume"].fillna(0)
        return (df.sort_index(), "yahoo") if not df.empty else (None, "empty")
    except Exception as e:
        return None, str(e)


def fetch_data(symbol, settings):
    df, src = yahoo_fetch(symbol, settings)
    if df is not None: return df, src, ""
    return None, "none", src


# ══════════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════════

def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rma(s,n): return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def sma(s,n): return s.rolling(n,min_periods=n).mean()
def stdev(s,n): return s.rolling(n,min_periods=n).std(ddof=0)

def atr(hi,lo,cl,n):
    pc=cl.shift(1)
    tr=pd.concat([hi-lo,(hi-pc).abs(),(lo-pc).abs()],axis=1).max(axis=1)
    return rma(tr,n)

def rsi(cl,n):
    d=cl.diff()
    ag=rma(d.clip(lower=0),n); al=rma(-d.clip(upper=0),n)
    rs=ag/al; r=100-(100/(1+rs))
    return r.where(al!=0,100).where(ag!=0,0)

def macd(cl,fast=12,slow=26,sig=9):
    ml=ema(cl,fast)-ema(cl,slow); sl=ema(ml,sig)
    return ml,sl,ml-sl

def build_htf(df,settings):
    htf=df.resample(settings.htf_rule).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    if htf.empty: return pd.DataFrame(index=df.index)
    hef=ema(htf["Close"],9); hem=ema(htf["Close"],21); hatr=atr(htf["High"],htf["Low"],htf["Close"],14)
    p=pd.DataFrame({"htf_close":htf["Close"].shift(1),"htf_ema_fast":hef.shift(1),
                    "htf_ema_fast_prev":hef.shift(2),"htf_ema_mid":hem.shift(1),"htf_atr":hatr.shift(1)})
    return p.reindex(df.index,method="ffill")

def infer_preset(symbol,settings):
    if settings.session_preset!="AUTO": return settings.session_preset
    s=symbol.upper()
    if "-USD" in s and any(x in s for x in ["BTC","ETH","SOL","BNB","XRP","DOGE","ADA","AVAX","LINK","DOT","MATIC","LTC","UNI","ATOM","XLM","BCH","ALGO","FIL","NEAR","APT","ARB","OP","INJ","SUI","TIA"]):
        return "Crypto 24/7"
    if s.endswith("=X"): return "Forex London/NY"
    return "US Indices Extended"

def session_check(ts,preset,settings):
    local=ts.tz_convert(settings.timezone); cur=local.hour*60+local.minute
    if preset=="Crypto 24/7":
        return {"in_session":True,"session_ok":True,"is_midday_chop":False,"is_near_close":False,"breakout_phase_ok":True}
    if preset=="US Indices RTH": start,end=9*60+30,16*60
    elif preset=="US Indices Extended": start,end=8*60,17*60
    elif preset=="Forex London/NY": start,end=3*60,12*60
    else: start,end=0,24*60-1
    in_sess=start<=cur<=end
    mfo=cur-start; mtc=end-cur
    is_early=0<=mfo<=35; is_chop=11*60+30<=cur<=13*60+30
    is_close=0<=mtc<=30; is_power=30<=mtc<=60
    return {"in_session":bool(in_sess),"session_ok":bool(in_sess and not is_chop and not is_close),
            "is_midday_chop":bool(is_chop),"is_near_close":bool(is_close),"breakout_phase_ok":bool(is_early or is_power)}

def cf(x,d=6):
    try:
        if pd.isna(x) or np.isinf(x): return None
        return round(float(x),d)
    except: return None


# ══════════════════════════════════════════════════════════
#  CORE SCAN
# ══════════════════════════════════════════════════════════

def scan_symbol(symbol, settings):
    df,provider,error=fetch_data(symbol,settings)
    ts_now=datetime.now(timezone.utc).isoformat()

    def err(why):
        return {"symbol":symbol,"provider":provider,"decision":"ERROR","position":"n/a",
                "buy_percent":None,"sell_percent":None,"edge_percent":None,"confidence":None,
                "why":why,"market":"data error","last_price":None,"entry":None,"stop":None,"tp1":None,
                "rsi":None,"atr":None,"session":"","bar_time":ts_now,"scan_time":ts_now}

    if df is None or df.empty: return err(error)
    if settings.use_last_closed_bar and len(df)>2: df=df.iloc[:-1].copy()
    if len(df)<settings.loading_bars+60: return err(f"Not enough bars ({len(df)})")

    preset=infer_preset(symbol,settings)
    op=df["Open"];hi=df["High"];lo=df["Low"];cl=df["Close"];vol=df["Volume"]

    ef_s=ema(cl,9);em_s=ema(cl,21);es_s=ema(cl,50)
    atr_s=atr(hi,lo,cl,14);atr_slow_s=sma(atr_s,50)
    rsi_s=rsi(cl,14);vol_ma_s=sma(vol,20)
    ml,sl,hl=macd(cl)
    hib=hi.shift(1).rolling(12,min_periods=12).max()
    lob=lo.shift(1).rolling(12,min_periods=12).min()
    cr=hi-lo;bs=(cl-op).abs();bp_s=bs/cr.replace(0,np.nan)
    bb_b=sma(cl,20);bb_d=stdev(cl,20)*2
    bbw=(bb_d*2/bb_b.replace(0,np.nan))*100;bbwma=sma(bbw,50)
    dm=(cl-cl.shift(20)).abs();nm=sma((cl-cl.shift(1)).abs(),20)*20
    eff_s=dm/nm.replace(0,np.nan)
    htf=build_htf(df,settings)

    i=len(df)-1;ts=df.index[i]
    c=float(cl.iloc[i]);o=float(op.iloc[i]);h=float(hi.iloc[i]);l=float(lo.iloc[i]);v=float(vol.iloc[i])
    ef=float(ef_s.iloc[i]);em=float(em_s.iloc[i]);es=float(es_s.iloc[i])
    atrv=float(atr_s.iloc[i]);atrslv=float(atr_slow_s.iloc[i])
    rsiv=float(rsi_s.iloc[i]);vmav=float(vol_ma_s.iloc[i])
    macdv=float(ml.iloc[i]);sigv=float(sl.iloc[i]);histv=float(hl.iloc[i]);histpv=float(hl.iloc[i-1])
    hb=float(hib.iloc[i]);lb=float(lob.iloc[i])
    bodyp=float(bp_s.iloc[i]) if not pd.isna(bp_s.iloc[i]) else 0.0
    crange=float(cr.iloc[i])
    htfc=htf["htf_close"].iloc[i] if "htf_close" in htf else np.nan
    htfef=htf["htf_ema_fast"].iloc[i] if "htf_ema_fast" in htf else np.nan
    htfefp=htf["htf_ema_fast_prev"].iloc[i] if "htf_ema_fast_prev" in htf else np.nan
    htfem=htf["htf_ema_mid"].iloc[i] if "htf_ema_mid" in htf else np.nan
    htfatr=htf["htf_atr"].iloc[i] if "htf_atr" in htf else np.nan
    htf_tol=float(htfatr)*0.02 if not pd.isna(htfatr) else 0.0

    htf_bull=(not pd.isna(htfc) and htfc>htfef and htfef>htfem and htfef>=htfefp-htf_tol)
    htf_bear=(not pd.isna(htfc) and htfc<htfef and htfef<htfem and htfef<=htfefp+htf_tol)
    htf_long_ok=(not settings.use_htf) or htf_bull
    htf_short_ok=(not settings.use_htf) or htf_bear

    tu=ef>em and c>em;td=ef<em and c<em
    stu=ef>em and em>es and c>es;std=ef<em and em<es and c<es
    emp=float(em_s.iloc[i-10]);slope=(em-emp)/atrv if atrv>0 else 0
    sb=slope>0.08;sbr=slope<-0.08
    rsi_b=52<rsiv<78;rsi_s2=22<rsiv<48
    macd_b=macdv>sigv and histv>histpv;macd_s2=macdv<sigv and histv<histpv
    lm=int(rsi_b)+int(macd_b);sm2=int(rsi_s2)+int(macd_s2)
    lmok=lm>=1;smok=sm2>=1

    vstrong=v>vmav*1.05 if not pd.isna(vmav) else False
    spike=crange>atrv*2.2
    cnh=c>=h-crange*0.25 if crange>0 else False
    cnl=c<=l+crange*0.25 if crange>0 else False
    bkl=c>hb and c>o;bks=c<lb and c<o
    pbl=tu and l<=em+atrv*0.40 and c>ef
    pbs=td and h>=em-atrv*0.40 and c<ef
    sbkl=bkl and crange>=atrv*1.15 and vstrong and cnh and bodyp>0.65 and not spike
    sbks=bks and crange>=atrv*1.15 and vstrong and cnl and bodyp>0.65 and not spike
    sess=session_check(ts,preset,settings)
    nbkl=bkl and sess["breakout_phase_ok"] and bodyp>0.55 and not spike
    nbks=bks and sess["breakout_phase_ok"] and bodyp>0.55 and not spike

    atr_r=atrv/atrslv if atrslv>0 else 1
    bbwv=float(bbw.iloc[i]) if not pd.isna(bbw.iloc[i]) else 0
    bbwmav=float(bbwma.iloc[i]) if not pd.isna(bbwma.iloc[i]) else 0
    bbwr=bbwv/bbwmav if bbwmav>0 else 1
    effv=float(eff_s.iloc[i]) if not pd.isna(eff_s.iloc[i]) else 0
    vol_ok=atr_r>=0.85 and bbwr>=0.85;eff_ok=effv>=0.25
    lreg=not spike and (vol_ok and eff_ok and sb or sbkl)
    sreg=not spike and (vol_ok and eff_ok and sbr or sbks)
    quiet=not lreg and not sreg

    warm=(i<settings.loading_bars or pd.isna(es) or pd.isna(atrslv) or pd.isna(bbwmav)
          or (settings.use_htf and pd.isna(htfc)))

    if warm: mkt="loading"
    elif not sess["in_session"]: mkt="outside session"
    elif spike: mkt="spread shock"
    elif sess["is_midday_chop"]: mkt="midday chop"
    elif sess["is_near_close"]: mkt="near close"
    elif quiet: mkt="choppy"
    else: mkt="active"

    lts=25.0 if stu and htf_long_ok else 18.0 if tu and htf_long_ok else 12.0 if tu else 0
    sts=25.0 if std and htf_short_ok else 18.0 if td and htf_short_ok else 12.0 if td else 0
    lss=24.0 if pbl else 22.0 if sbkl else 16.0 if nbkl else 0
    sss=24.0 if pbs else 22.0 if sbks else 16.0 if nbks else 0
    lps=8.0 if sbkl else 20.0 if lm==2 else 10.0 if lm==1 else 0
    sps=8.0 if sbks else 20.0 if sm2==2 else 10.0 if sm2==1 else 0
    lcs=20.0 if sbkl else 16.0 if lreg else 0
    scs=20.0 if sbks else 16.0 if sreg else 0
    lvs=0.0 if sbkl else 5.0 if vstrong else 0
    svs=0.0 if sbks else 5.0 if vstrong else 0
    buy=min(lts+lss+lps+lcs+lvs,100.0)
    sell=min(sts+sss+sps+scs+svs,100.0)
    edge=abs(buy-sell);conf=max(buy,sell)

    ea=not warm and sess["session_ok"] and not spike
    ltrig=pbl or sbkl or nbkl;strig=pbs or sbks or nbks
    rlong=ea and htf_long_ok and lreg and ltrig and lmok and buy>sell and buy>=40 and buy>=sell+3
    rshort=ea and htf_short_ok and sreg and strig and smok and sell>buy and sell>=40 and sell>=buy+3

    if warm: dec,pos,why="LOADING","FLAT","warming up"
    elif rlong: dec,pos,why="BUY CONFIRMED","LONG","buy confirmed"
    elif rshort: dec,pos,why="SELL CONFIRMED","SHORT","sell confirmed"
    elif mkt!="active": dec,pos,why="WAIT","FLAT",mkt
    elif buy>sell: dec,pos,why="LEAN BUY","FLAT","buy stronger, unconfirmed"
    elif sell>buy: dec,pos,why="LEAN SELL","FLAT","sell stronger, unconfirmed"
    else: dec,pos,why="WAIT","FLAT","mixed"

    sd=max(atrv*settings.atr_stop_mult, c*settings.min_stop_percent/100)
    if dec in ["BUY CONFIRMED","LEAN BUY"]: stop=c-sd;tp1=c+sd*settings.rr_tp1
    elif dec in ["SELL CONFIRMED","LEAN SELL"]: stop=c+sd;tp1=c-sd*settings.rr_tp1
    else: stop=None;tp1=None

    return {
        "symbol":symbol,"provider":provider,"decision":dec,"position":pos,
        "buy_percent":cf(buy,2),"sell_percent":cf(sell,2),"edge_percent":cf(edge,2),"confidence":cf(conf,2),
        "why":why,"market":mkt,"last_price":cf(c,6),
        "entry":cf(c,6) if dec in ["BUY CONFIRMED","SELL CONFIRMED"] else None,
        "stop":cf(stop,6) if stop else None,"tp1":cf(tp1,6) if tp1 else None,
        "rsi":cf(rsiv,2),"atr":cf(atrv,6),"session":preset,
        "bar_time":str(ts),"scan_time":datetime.now(timezone.utc).isoformat(),
    }


# ══════════════════════════════════════════════════════════
#  PARALLEL SWEEP + LIVE STATE WRITER
# ══════════════════════════════════════════════════════════

_state_lock = threading.Lock()
_current_state: Dict = {}   # symbol -> last result


def sweep_all(symbols, settings):
    """Scan all symbols in parallel, return list of results."""
    results = {}
    with ThreadPoolExecutor(max_workers=settings.max_workers) as ex:
        fut_map = {ex.submit(scan_symbol, s, settings): s for s in symbols}
        for fut in as_completed(fut_map):
            sym = fut_map[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"symbol":sym,"provider":"none","decision":"ERROR","position":"n/a",
                     "why":str(e),"market":"error","buy_percent":None,"sell_percent":None,
                     "edge_percent":None,"confidence":None,"last_price":None,"entry":None,
                     "stop":None,"tp1":None,"rsi":None,"atr":None,"session":"",
                     "bar_time":"","scan_time":datetime.now(timezone.utc).isoformat()}
            results[sym] = r

            # Print live to terminal as each finishes
            dec = r["decision"]
            price = r["last_price"] or "?"
            buy = r["buy_percent"] or "-"
            sell = r["sell_percent"] or "-"
            icon = {"BUY CONFIRMED":"🟢","SELL CONFIRMED":"🔴","LEAN BUY":"🔼",
                    "LEAN SELL":"🔽","WAIT":"⏸ ","LOADING":"⏳","ERROR":"❌"}.get(dec,"  ")
            print(f"  {icon} {sym:<12} {dec:<16}  price={price:<12}  buy={buy:<6} sell={sell:<6}  {r['why']}", flush=True)

    return results


DECISION_ORDER = {
    "BUY CONFIRMED":0,"SELL CONFIRMED":1,
    "LEAN BUY":2,"LEAN SELL":3,
    "WAIT":4,"LOADING":5,"ERROR":6,
}

def sort_results(results_dict):
    rows = list(results_dict.values())
    rows.sort(key=lambda r: (DECISION_ORDER.get(r["decision"],9), -(r["confidence"] or 0)))
    return rows


def write_state(results_dict, settings, sweep_n, next_sweep_ts):
    rows = sort_results(results_dict)
    payload = {
        "sweep": sweep_n,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "next_sweep_at": next_sweep_ts,
        "interval": settings.interval,
        "refresh_seconds": settings.refresh_seconds,
        "total": len(rows),
        "confirmed": sum(1 for r in rows if r["decision"] in ("BUY CONFIRMED","SELL CONFIRMED")),
        "rows": rows,
    }
    tmp = settings.output_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, settings.output_json)   # atomic write
    print(f"\n  ✓ State written → {settings.output_json}  ({payload['confirmed']} confirmed)\n", flush=True)


def run_forever(symbols, settings):
    sweep_n = 0
    while True:
        sweep_n += 1
        now = datetime.now(timezone.utc)
        next_ts = (datetime.fromtimestamp(now.timestamp() + settings.refresh_seconds, tz=timezone.utc)).isoformat()

        print(f"\n{'═'*70}", flush=True)
        print(f"  SWEEP #{sweep_n}  |  {len(symbols)} symbols  |  {now.strftime('%Y-%m-%d %H:%M:%S')} UTC", flush=True)
        print(f"{'═'*70}", flush=True)

        results = sweep_all(symbols, settings)

        with _state_lock:
            _current_state.update(results)

        write_state(_current_state, settings, sweep_n, next_ts)

        print(f"  Next sweep in {settings.refresh_seconds}s …  (Ctrl+C to stop)", flush=True)
        time.sleep(settings.refresh_seconds)


# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel live scanner")
    parser.add_argument("--interval",  default="5m",  help="Candle interval (1m,5m,15m,…)")
    parser.add_argument("--range",     default="10d", help="Yahoo data range")
    parser.add_argument("--refresh",   default=60, type=int, help="Seconds between sweeps")
    parser.add_argument("--workers",   default=12, type=int, help="Parallel threads")
    parser.add_argument("--timezone",  default="America/New_York")
    parser.add_argument("--no-htf",    action="store_true", help="Disable HTF filter")
    args = parser.parse_args()

    settings = ScannerSettings(
        interval=args.interval,
        yahoo_range=args.range,
        session_preset="AUTO",
        timezone=args.timezone,
        use_htf=not args.no_htf,
        refresh_seconds=args.refresh,
        max_workers=args.workers,
    )

    print(f"\n  Scanner starting  |  {len(ALL_SYMBOLS)} symbols  |  {args.workers} threads  |  every {args.refresh}s")
    print(f"  Open scanner_dashboard.html in your browser for the live UI\n")

    try:
        run_forever(ALL_SYMBOLS, settings)
    except KeyboardInterrupt:
        print("\n  Stopped.")
