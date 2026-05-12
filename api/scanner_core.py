"""Server-side scanner aligned with the user's Pine Script v6 logic.

This module is designed for Vercel Python Functions. It performs one scan
when called and returns JSON; it does not run forever and does not write files.
"""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests


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

ALL_SYMBOLS = list(dict.fromkeys(CRYPTO + STOCKS))

DECISION_ORDER = {
    "BUY CONFIRMED": 0,
    "SELL CONFIRMED": 1,
    "SELL / EXIT LONG": 2,
    "BUY / EXIT SHORT": 3,
    "TRAIL LONG": 4,
    "TRAIL SHORT": 5,
    "HOLD LONG": 6,
    "HOLD SHORT": 7,
    "GET READY TO BUY": 8,
    "GET READY TO SELL": 9,
    "LEAN BUY": 10,
    "LEAN SELL": 11,
    "WAIT": 12,
    "LOADING": 13,
    "ERROR": 14,
}


@dataclass
class ScannerSettings:
    # Mirrors Pine inputs where possible.
    entry_mode: str = "Fast 5m"  # Fast 5m, Balanced, Conservative
    loading_bars: int = 100
    session_preset: str = "US Indices RTH"
    custom_session: str = "0800-1700"
    use_session_phase_filter: bool = True
    use_news_blackout: bool = False
    news_blackout_session: str = "0825-0840"
    min_stop_percent: float = 0.08
    atr_stop_mult: float = 1.2
    rr_tp1: float = 1.5
    use_trade_manager: bool = True
    exit_on_opposite_signal: bool = True
    exit_on_weakness_after_tp1: bool = True
    use_trailing_after_tp1: bool = True
    allow_same_bar_flip: bool = False
    max_signals_per_hour: int = 3
    cooldown_bars: int = 2
    use_htf: bool = True
    htf_rule: str = "15min"
    use_second_entry: bool = True
    cost_preset: str = "Normal"  # Tight, Normal, Wide, Volatile

    # Server/API settings.
    interval: str = "5m"
    yahoo_range: str = "10d"
    timezone: str = "America/New_York"
    max_workers: int = 12
    use_last_closed_bar: bool = True
    request_timeout: int = 20


def _safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default


def clean_number(x: Any, digits: int = 6) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return round(v, digits)
    except Exception:
        return None


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def stdev(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).std(ddof=0)


def atr_series(hi: pd.Series, lo: pd.Series, cl: pd.Series, n: int) -> pd.Series:
    prev_close = cl.shift(1)
    tr = pd.concat([(hi - lo), (hi - prev_close).abs(), (lo - prev_close).abs()], axis=1).max(axis=1)
    return rma(tr, n)


def rsi_series(cl: pd.Series, n: int) -> pd.Series:
    d = cl.diff()
    avg_gain = rma(d.clip(lower=0), n)
    avg_loss = rma(-d.clip(upper=0), n)
    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    out = out.where(avg_loss != 0, 100).where(avg_gain != 0, 0)
    return out


def macd_series(cl: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(cl, fast) - ema(cl, slow)
    signal = ema(line, sig)
    return line, signal, line - signal


def parse_session(session_str: str) -> Tuple[int, int]:
    # Pine input.session format like 0930-1600.
    try:
        a, b = session_str.split("-")
        start = int(a[:2]) * 60 + int(a[2:4])
        end = int(b[:2]) * 60 + int(b[2:4])
        return start, end
    except Exception:
        return 0, 24 * 60 - 1


def session_minutes_for_preset(settings: ScannerSettings) -> Tuple[int, int, str]:
    preset = settings.session_preset
    if preset == "Crypto 24/7":
        return 0, 24 * 60 - 1, "0000-2359"
    if preset == "US Indices RTH":
        return 9 * 60 + 30, 16 * 60, "0930-1600"
    if preset == "US Indices Extended":
        return 8 * 60, 17 * 60, "0800-1700"
    if preset == "Forex London/NY":
        return 3 * 60, 12 * 60, "0300-1200"
    s, e = parse_session(settings.custom_session)
    return s, e, settings.custom_session


def infer_session_preset(symbol: str, settings: ScannerSettings) -> str:
    if settings.session_preset != "AUTO":
        return settings.session_preset
    if symbol.upper().endswith("-USD"):
        return "Crypto 24/7"
    return "US Indices RTH"


def session_state(ts: pd.Timestamp, symbol: str, settings: ScannerSettings) -> Dict[str, Any]:
    local = ts.tz_convert(settings.timezone)
    minutes = local.hour * 60 + local.minute

    original_preset = settings.session_preset
    preset = infer_session_preset(symbol, settings)
    # Use a shallow temporary settings view for presets.
    tmp = ScannerSettings(**{**settings.__dict__, "session_preset": preset})
    start, end, _ = session_minutes_for_preset(tmp)

    crypto = preset == "Crypto 24/7"
    if crypto:
        in_session = True
    else:
        in_session = start <= minutes <= end if start <= end else (minutes >= start or minutes <= end)

    news_start, news_end = parse_session(settings.news_blackout_session)
    in_news_blackout = settings.use_news_blackout and (news_start <= minutes <= news_end)

    minutes_from_open = minutes - start
    minutes_to_close = end - minutes

    is_early = (not crypto) and 0 <= minutes_from_open <= 35
    is_midday_chop = (not crypto) and (11 * 60 + 30 <= minutes <= 13 * 60 + 30)
    is_near_close = (not crypto) and 0 <= minutes_to_close <= 30
    is_power_hour = (not crypto) and 30 <= minutes_to_close <= 60

    session_phase_ok = (not settings.use_session_phase_filter) or crypto or ((not is_midday_chop) and (not is_near_close))
    breakout_phase_ok = (not settings.use_session_phase_filter) or crypto or is_early or is_power_hour
    session_ok = bool(in_session and (not in_news_blackout) and session_phase_ok)

    return {
        "preset": preset,
        "in_session": bool(in_session),
        "in_news_blackout": bool(in_news_blackout),
        "is_midday_chop": bool(is_midday_chop),
        "is_near_close": bool(is_near_close),
        "session_phase_ok": bool(session_phase_ok),
        "breakout_phase_ok": bool(breakout_phase_ok),
        "session_ok": bool(session_ok),
    }


def yahoo_fetch(symbol: str, settings: ScannerSettings) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}"
        params = {"interval": settings.interval, "range": settings.yahoo_range, "includePrePost": "true"}
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        r = requests.get(url, params=params, headers=headers, timeout=settings.request_timeout)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        data = r.json()
        err = data.get("chart", {}).get("error")
        if err:
            return None, str(err)
        result = data.get("chart", {}).get("result") or []
        if not result:
            return None, "no result"
        result = result[0]
        ts = result.get("timestamp") or []
        quote_data = (result.get("indicators", {}).get("quote") or [{}])[0]
        if not ts or not quote_data:
            return None, "no candles"
        df = pd.DataFrame({
            "Open": quote_data.get("open"),
            "High": quote_data.get("high"),
            "Low": quote_data.get("low"),
            "Close": quote_data.get("close"),
            "Volume": quote_data.get("volume"),
        })
        df["Time"] = pd.to_datetime(ts, unit="s", utc=True)
        df = df.set_index("Time").dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
        df["Volume"] = df["Volume"].fillna(0)
        if settings.use_last_closed_bar and len(df) > 2:
            df = df.iloc[:-1].copy()
        return (df, "ok") if not df.empty else (None, "empty")
    except Exception as e:
        return None, str(e)


def build_htf(df: pd.DataFrame, settings: ScannerSettings) -> pd.DataFrame:
    htf = df.resample(settings.htf_rule).agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()
    if htf.empty:
        return pd.DataFrame(index=df.index)
    htf_ema_fast = ema(htf["Close"], 9)
    htf_ema_mid = ema(htf["Close"], 21)
    htf_atr = atr_series(htf["High"], htf["Low"], htf["Close"], 14)
    # Pine uses close[1], EMA[1], EMA[2], etc. inside the HTF request.
    p = pd.DataFrame({
        "htf_close": htf["Close"].shift(1),
        "htf_ema_fast": htf_ema_fast.shift(1),
        "htf_ema_fast_prev": htf_ema_fast.shift(2),
        "htf_ema_mid": htf_ema_mid.shift(1),
        "htf_atr": htf_atr.shift(1),
    })
    return p.reindex(df.index, method="ffill")


def _pivot_high_at(high: np.ndarray, center: int, left: int, right: int) -> Optional[float]:
    if center - left < 0 or center + right >= len(high):
        return None
    c = high[center]
    window = high[center - left:center + right + 1]
    if np.isnan(c) or np.isnan(window).any():
        return None
    return float(c) if c >= np.max(window) and np.sum(window == c) == 1 else None


def _pivot_low_at(low: np.ndarray, center: int, left: int, right: int) -> Optional[float]:
    if center - left < 0 or center + right >= len(low):
        return None
    c = low[center]
    window = low[center - left:center + right + 1]
    if np.isnan(c) or np.isnan(window).any():
        return None
    return float(c) if c <= np.min(window) and np.sum(window == c) == 1 else None


def _cost_tick(symbol: str, price: float) -> float:
    if symbol.endswith("-USD"):
        if price >= 1000:
            return 0.01
        if price >= 1:
            return 0.0001
        return 0.000001
    return 0.01


def _cost_settings(cost_preset: str) -> Tuple[int, float]:
    if cost_preset == "Tight":
        return 1, 4.0
    if cost_preset == "Wide":
        return 4, 6.0
    if cost_preset == "Volatile":
        return 8, 8.0
    return 2, 5.0


def scan_symbol(symbol: str, settings: ScannerSettings) -> Dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    df, status = yahoo_fetch(symbol, settings)

    def err(why: str) -> Dict[str, Any]:
        return {
            "symbol": symbol, "provider": "yahoo", "decision": "ERROR", "position": "n/a",
            "buy_percent": None, "sell_percent": None, "edge_percent": None, "confidence": None,
            "why": why, "market": "data error", "last_price": None, "entry": None,
            "stop": None, "tp1": None, "rsi": None, "atr": None, "session": "",
            "bar_time": "", "scan_time": started,
        }

    if df is None or df.empty:
        return err(status)
    if len(df) < max(settings.loading_bars + 60, 160):
        return err(f"Not enough bars ({len(df)})")

    # Pine constants.
    ema_fast_len = 9
    ema_mid_len = 21
    ema_slow_len = 50
    rsi_len = 14
    atr_len = 14
    vol_len = 20
    breakout_len = 12
    pullback_atr_band = 0.40
    bb_len = 20
    bb_mult = 2.0
    bb_width_avg_len = 50
    atr_regime_len = 50
    efficiency_len = 20
    slope_len = 10
    second_entry_max_bars = 10
    failed_breakout_bars = 4
    fakeout_cooldown_bars = 5
    pivot_left = 3
    pivot_right = 3

    is_fast = settings.entry_mode == "Fast 5m"
    is_balanced = settings.entry_mode == "Balanced"
    is_conservative = settings.entry_mode == "Conservative"
    base_min_score = 40.0 if is_fast else 55.0 if is_balanced else 70.0
    base_prepare_score = base_min_score - 10.0
    score_edge = 3.0 if is_fast else 6.0 if is_balanced else 10.0

    op = df["Open"].astype(float)
    hi = df["High"].astype(float)
    lo = df["Low"].astype(float)
    cl = df["Close"].astype(float)
    vol = df["Volume"].astype(float)

    ema_fast = ema(cl, ema_fast_len)
    ema_mid = ema(cl, ema_mid_len)
    ema_slow = ema(cl, ema_slow_len)
    atr = atr_series(hi, lo, cl, atr_len).fillna(hi - lo)
    atr_slow = sma(atr, atr_regime_len)
    rsi = rsi_series(cl, rsi_len)
    vol_ma = sma(vol, vol_len)
    macd_line, signal_line, hist_line = macd_series(cl)
    highest_before = hi.shift(1).rolling(breakout_len, min_periods=breakout_len).max()
    lowest_before = lo.shift(1).rolling(breakout_len, min_periods=breakout_len).min()
    candle_range = hi - lo
    body_size = (cl - op).abs()
    body_percent = body_size / candle_range.replace(0, np.nan)
    body_percent = body_percent.fillna(0)
    bb_basis = sma(cl, bb_len)
    bb_dev = stdev(cl, bb_len) * bb_mult
    bb_upper = bb_basis + bb_dev
    bb_lower = bb_basis - bb_dev
    bb_width = ((bb_upper - bb_lower) / bb_basis.replace(0, np.nan)) * 100.0
    bb_width = bb_width.fillna(0)
    bb_width_ma = sma(bb_width, bb_width_avg_len)
    direction_move = (cl - cl.shift(efficiency_len)).abs().fillna(0)
    noise_move = sma((cl - cl.shift(1)).abs(), efficiency_len) * efficiency_len
    efficiency = (direction_move / noise_move.replace(0, np.nan)).fillna(0)
    htf = build_htf(df, settings)

    high_arr = hi.to_numpy()
    low_arr = lo.to_numpy()

    # Stateful Pine vars.
    last_swing_high: Optional[float] = None
    last_swing_low: Optional[float] = None
    long_attempts = 0
    short_attempts = 0
    long_pullback_active = False
    short_pullback_active = False
    long_pullback_bar: Optional[int] = None
    short_pullback_bar: Optional[int] = None
    long_last_attempt_bar: Optional[int] = None
    short_last_attempt_bar: Optional[int] = None
    fakeout_until_bar: Optional[int] = None
    watch_long_breakout_level: Optional[float] = None
    watch_long_breakout_bar: Optional[int] = None
    watch_short_breakout_level: Optional[float] = None
    watch_short_breakout_bar: Optional[int] = None
    last_signal_bar: Optional[int] = None
    signal_bars: List[int] = []

    virtual_pos = 0
    virtual_entry: Optional[float] = None
    virtual_stop: Optional[float] = None
    virtual_tp1: Optional[float] = None
    virtual_high_since_entry: Optional[float] = None
    virtual_low_since_entry: Optional[float] = None
    virtual_tp1_hit = False
    virtual_entry_bar: Optional[int] = None
    last_trade_entry: Optional[float] = None
    last_trade_stop: Optional[float] = None
    last_trade_tp1: Optional[float] = None
    last_exit_price: Optional[float] = None
    last_exit_reason = ""

    last_row: Dict[str, Any] = {}

    for i in range(len(df)):
        ts = df.index[i]
        o = _safe_float(op.iloc[i])
        h = _safe_float(hi.iloc[i])
        l = _safe_float(lo.iloc[i])
        c = _safe_float(cl.iloc[i])
        v = _safe_float(vol.iloc[i], 0.0)
        ef = _safe_float(ema_fast.iloc[i])
        em = _safe_float(ema_mid.iloc[i])
        es = _safe_float(ema_slow.iloc[i])
        atrv = _safe_float(atr.iloc[i], max(h - l, 0.000001))
        atrslv = _safe_float(atr_slow.iloc[i])
        rsiv = _safe_float(rsi.iloc[i])
        vmav = _safe_float(vol_ma.iloc[i])
        macdv = _safe_float(macd_line.iloc[i])
        sigv = _safe_float(signal_line.iloc[i])
        histv = _safe_float(hist_line.iloc[i])
        hist_prev = _safe_float(hist_line.iloc[i - 1]) if i > 0 else np.nan
        hb = _safe_float(highest_before.iloc[i])
        lb = _safe_float(lowest_before.iloc[i])
        crange = _safe_float(candle_range.iloc[i], 0.0)
        bodyp = _safe_float(body_percent.iloc[i], 0.0)

        # Pivot values appear pivot_right bars after the actual pivot, matching Pine behavior.
        pivot_center = i - pivot_right
        ph = _pivot_high_at(high_arr, pivot_center, pivot_left, pivot_right) if pivot_center >= 0 else None
        pl = _pivot_low_at(low_arr, pivot_center, pivot_left, pivot_right) if pivot_center >= 0 else None
        if ph is not None:
            last_swing_high = ph
        if pl is not None:
            last_swing_low = pl

        htf_close = _safe_float(htf["htf_close"].iloc[i]) if "htf_close" in htf else np.nan
        htf_ema_fast = _safe_float(htf["htf_ema_fast"].iloc[i]) if "htf_ema_fast" in htf else np.nan
        htf_ema_fast_prev = _safe_float(htf["htf_ema_fast_prev"].iloc[i]) if "htf_ema_fast_prev" in htf else np.nan
        htf_ema_mid = _safe_float(htf["htf_ema_mid"].iloc[i]) if "htf_ema_mid" in htf else np.nan
        htf_atr = _safe_float(htf["htf_atr"].iloc[i]) if "htf_atr" in htf else np.nan
        htf_slope_tolerance = htf_atr * 0.02 if math.isfinite(htf_atr) else 0.0

        htf_bullish = all(math.isfinite(x) for x in [htf_close, htf_ema_fast, htf_ema_fast_prev, htf_ema_mid]) and htf_close > htf_ema_fast and htf_ema_fast > htf_ema_mid and htf_ema_fast >= htf_ema_fast_prev - htf_slope_tolerance
        htf_bearish = all(math.isfinite(x) for x in [htf_close, htf_ema_fast, htf_ema_fast_prev, htf_ema_mid]) and htf_close < htf_ema_fast and htf_ema_fast < htf_ema_mid and htf_ema_fast <= htf_ema_fast_prev + htf_slope_tolerance
        htf_long_ok = (not settings.use_htf) or htf_bullish
        htf_short_ok = (not settings.use_htf) or htf_bearish

        sess = session_state(ts, symbol, settings)

        trend_up = ef > em and c > em
        trend_down = ef < em and c < em
        strong_trend_up = ef > em and em > es and c > es
        strong_trend_down = ef < em and em < es and c < es
        ema_mid_prior = _safe_float(ema_mid.iloc[i - slope_len]) if i >= slope_len else np.nan
        ema_slope_atr = (em - ema_mid_prior) / atrv if atrv > 0 and math.isfinite(ema_mid_prior) else 0.0
        slope_bull = ema_slope_atr > 0.08
        slope_bear = ema_slope_atr < -0.08
        rsi_bull = rsiv > 52 and rsiv < 78
        rsi_bear = rsiv < 48 and rsiv > 22
        macd_bull = macdv > sigv and histv > hist_prev
        macd_bear = macdv < sigv and histv < hist_prev
        long_momentum_count = int(rsi_bull) + int(macd_bull)
        short_momentum_count = int(rsi_bear) + int(macd_bear)
        long_momentum_ok = long_momentum_count >= (1 if is_fast else 2)
        short_momentum_ok = short_momentum_count >= (1 if is_fast else 2)
        volume_strong = bool(math.isfinite(vmav) and v > vmav * 1.05)

        spread_shock = crange > atrv * 2.2 if atrv > 0 else False
        close_near_high = c >= h - crange * 0.25 if crange > 0 else False
        close_near_low = c <= l + crange * 0.25 if crange > 0 else False
        breakout_long = math.isfinite(hb) and c > hb and c > o
        breakout_short = math.isfinite(lb) and c < lb and c < o
        pullback_long = trend_up and l <= em + atrv * pullback_atr_band and c > ef
        pullback_short = trend_down and h >= em - atrv * pullback_atr_band and c < ef
        bull_candle_confirm = i > 0 and c > o and c > _safe_float(hi.iloc[i - 1])
        bear_candle_confirm = i > 0 and c < o and c < _safe_float(lo.iloc[i - 1])
        strong_breakout_long = breakout_long and crange >= atrv * 1.15 and volume_strong and close_near_high and bodyp > 0.65 and not spread_shock
        strong_breakout_short = breakout_short and crange >= atrv * 1.15 and volume_strong and close_near_low and bodyp > 0.65 and not spread_shock
        normal_breakout_long_ok = breakout_long and sess["breakout_phase_ok"] and bodyp > 0.55 and not spread_shock
        normal_breakout_short_ok = breakout_short and sess["breakout_phase_ok"] and bodyp > 0.55 and not spread_shock

        sweep_distance = atrv * 0.12
        reclaim_buffer = sweep_distance * 0.5
        sweep_low_reclaim = last_swing_low is not None and l < last_swing_low - sweep_distance and c > last_swing_low + reclaim_buffer and c > o and bodyp > 0.45
        sweep_high_reject = last_swing_high is not None and h > last_swing_high + sweep_distance and c < last_swing_high - reclaim_buffer and c < o and bodyp > 0.45
        range_low_reclaim = math.isfinite(lb) and l < lb - sweep_distance and c > lb + reclaim_buffer and c > o and bodyp > 0.45
        range_high_reject = math.isfinite(hb) and h > hb + sweep_distance and c < hb - reclaim_buffer and c < o and bodyp > 0.45
        liquidity_long = sweep_low_reclaim or range_low_reclaim
        liquidity_short = sweep_high_reject or range_high_reject

        second_entry_long = False
        second_entry_short = False
        long_structure_failed = (not trend_up) or c < em - atrv * 0.35 or breakout_short or spread_shock or (long_pullback_bar is not None and i - long_pullback_bar > second_entry_max_bars)
        short_structure_failed = (not trend_down) or c > em + atrv * 0.35 or breakout_long or spread_shock or (short_pullback_bar is not None and i - short_pullback_bar > second_entry_max_bars)

        if long_structure_failed:
            long_attempts = 0
            long_pullback_active = False
            long_pullback_bar = None
            long_last_attempt_bar = None
        else:
            if pullback_long and not long_pullback_active:
                long_pullback_active = True
                long_attempts = 0
                long_pullback_bar = i
                long_last_attempt_bar = None
            if long_pullback_active and bull_candle_confirm:
                if long_attempts == 0:
                    long_attempts = 1
                    long_last_attempt_bar = i
                elif long_last_attempt_bar is not None and i > long_last_attempt_bar + 1:
                    second_entry_long = True
                    long_attempts = 0
                    long_pullback_active = False
                    long_pullback_bar = None
                    long_last_attempt_bar = None

        if short_structure_failed:
            short_attempts = 0
            short_pullback_active = False
            short_pullback_bar = None
            short_last_attempt_bar = None
        else:
            if pullback_short and not short_pullback_active:
                short_pullback_active = True
                short_attempts = 0
                short_pullback_bar = i
                short_last_attempt_bar = None
            if short_pullback_active and bear_candle_confirm:
                if short_attempts == 0:
                    short_attempts = 1
                    short_last_attempt_bar = i
                elif short_last_attempt_bar is not None and i > short_last_attempt_bar + 1:
                    second_entry_short = True
                    short_attempts = 0
                    short_pullback_active = False
                    short_pullback_bar = None
                    short_last_attempt_bar = None

        atr_ratio = atrv / atrslv if math.isfinite(atrslv) and atrslv > 0 else 1.0
        bbw = _safe_float(bb_width.iloc[i], 0.0)
        bbw_ma = _safe_float(bb_width_ma.iloc[i])
        bbw_ratio = bbw / bbw_ma if math.isfinite(bbw_ma) and bbw_ma > 0 else 1.0
        eff = _safe_float(efficiency.iloc[i], 0.0)
        volatility_ok = atr_ratio >= 0.85 and bbw_ratio >= 0.85
        efficiency_ok = eff >= 0.25
        long_regime_ok = not spread_shock and ((volatility_ok and efficiency_ok and slope_bull) or liquidity_long or strong_breakout_long)
        short_regime_ok = not spread_shock and ((volatility_ok and efficiency_ok and slope_bear) or liquidity_short or strong_breakout_short)
        quiet_market = not long_regime_ok and not short_regime_ok

        if breakout_long:
            watch_long_breakout_level = hb
            watch_long_breakout_bar = i
        if breakout_short:
            watch_short_breakout_level = lb
            watch_short_breakout_bar = i

        long_breakout_failed = watch_long_breakout_bar is not None and i - watch_long_breakout_bar <= failed_breakout_bars and watch_long_breakout_level is not None and c < watch_long_breakout_level
        short_breakout_failed = watch_short_breakout_bar is not None and i - watch_short_breakout_bar <= failed_breakout_bars and watch_short_breakout_level is not None and c > watch_short_breakout_level
        if long_breakout_failed or short_breakout_failed:
            fakeout_until_bar = i + fakeout_cooldown_bars
            watch_long_breakout_level = None
            watch_long_breakout_bar = None
            watch_short_breakout_level = None
            watch_short_breakout_bar = None
        if watch_long_breakout_bar is not None and i - watch_long_breakout_bar > failed_breakout_bars:
            watch_long_breakout_level = None
            watch_long_breakout_bar = None
        if watch_short_breakout_bar is not None and i - watch_short_breakout_bar > failed_breakout_bars:
            watch_short_breakout_level = None
            watch_short_breakout_bar = None
        fakeout_active = fakeout_until_bar is not None and i <= fakeout_until_bar

        long_trend_score = 25.0 if strong_trend_up and htf_long_ok else 18.0 if trend_up and htf_long_ok else 12.0 if trend_up else 0.0
        short_trend_score = 25.0 if strong_trend_down and htf_short_ok else 18.0 if trend_down and htf_short_ok else 12.0 if trend_down else 0.0
        long_setup_score = 30.0 if liquidity_long else 28.0 if settings.use_second_entry and second_entry_long else 24.0 if pullback_long and bull_candle_confirm else 18.0 if pullback_long else 22.0 if strong_breakout_long else 16.0 if normal_breakout_long_ok else 0.0
        short_setup_score = 30.0 if liquidity_short else 28.0 if settings.use_second_entry and second_entry_short else 24.0 if pullback_short and bear_candle_confirm else 18.0 if pullback_short else 22.0 if strong_breakout_short else 16.0 if normal_breakout_short_ok else 0.0
        long_participation_score = 8.0 if strong_breakout_long else 10.0 if liquidity_long else 20.0 if long_momentum_count == 2 else 10.0 if long_momentum_count == 1 else 0.0
        short_participation_score = 8.0 if strong_breakout_short else 10.0 if liquidity_short else 20.0 if short_momentum_count == 2 else 10.0 if short_momentum_count == 1 else 0.0
        long_context_score = 20.0 if liquidity_long or strong_breakout_long else 16.0 if long_regime_ok else 0.0
        short_context_score = 20.0 if liquidity_short or strong_breakout_short else 16.0 if short_regime_ok else 0.0
        long_volume_score = 0.0 if strong_breakout_long or liquidity_long else 5.0 if volume_strong else 0.0
        short_volume_score = 0.0 if strong_breakout_short or liquidity_short else 5.0 if volume_strong else 0.0
        buy_percent = max(min(long_trend_score + long_setup_score + long_participation_score + long_context_score + long_volume_score, 100.0), 0.0)
        sell_percent = max(min(short_trend_score + short_setup_score + short_participation_score + short_context_score + short_volume_score, 100.0), 0.0)
        edge_percent = abs(buy_percent - sell_percent)
        confidence = max(buy_percent, sell_percent)

        data_ready = i >= settings.loading_bars and math.isfinite(es) and math.isfinite(atrslv) and math.isfinite(bbw_ma) and ((not settings.use_htf) or math.isfinite(htf_close))
        is_warming_up = not data_ready

        bars_after_signal = 999 if last_signal_bar is None else i - last_signal_bar
        in_cooldown = bars_after_signal < settings.cooldown_bars
        try:
            interval_minutes = float(settings.interval[:-1]) if settings.interval.endswith("m") else float(settings.interval[:-1]) * 60.0 if settings.interval.endswith("h") else 5.0
        except Exception:
            interval_minutes = 5.0
        interval_minutes = max(1.0, interval_minutes)
        bars_per_hour = max(1, int(round(60.0 / interval_minutes)))
        signal_bars = [b for b in signal_bars if i - b <= bars_per_hour]
        signals_this_hour = len(signal_bars)
        too_many_signals = signals_this_hour >= settings.max_signals_per_hour

        if is_warming_up:
            market_text = "loading / warming up"
        elif not sess["in_session"]:
            market_text = "outside session"
        elif sess["in_news_blackout"]:
            market_text = "news blackout"
        elif spread_shock:
            market_text = "spread shock / spike"
        elif sess["is_midday_chop"] and settings.use_session_phase_filter:
            market_text = "midday chop zone"
        elif sess["is_near_close"] and settings.use_session_phase_filter:
            market_text = "near session close"
        elif too_many_signals:
            market_text = "signal limit hit"
        elif fakeout_active:
            market_text = "recent fakeout"
        elif in_cooldown:
            market_text = "cooldown active"
        elif quiet_market:
            market_text = "market too choppy"
        else:
            market_text = "market active"

        dynamic_min_score = min(base_min_score + (8.0 if fakeout_active else 0.0), 95.0)
        long_bias = not is_warming_up and sess["session_ok"] and long_regime_ok and buy_percent >= base_prepare_score and buy_percent > sell_percent
        short_bias = not is_warming_up and sess["session_ok"] and short_regime_ok and sell_percent >= base_prepare_score and sell_percent > buy_percent
        entry_allowed = not is_warming_up and sess["session_ok"] and not in_cooldown and not too_many_signals and not fakeout_active and not spread_shock
        long_trigger = ((settings.use_second_entry and second_entry_long) or liquidity_long or (pullback_long and strong_trend_up and volume_strong)) if is_conservative else (liquidity_long or pullback_long or (settings.use_second_entry and second_entry_long) or strong_breakout_long or normal_breakout_long_ok)
        short_trigger = ((settings.use_second_entry and second_entry_short) or liquidity_short or (pullback_short and strong_trend_down and volume_strong)) if is_conservative else (liquidity_short or pullback_short or (settings.use_second_entry and second_entry_short) or strong_breakout_short or normal_breakout_short_ok)
        raw_long_signal = entry_allowed and htf_long_ok and long_regime_ok and long_trigger and long_momentum_ok and buy_percent > sell_percent and buy_percent >= dynamic_min_score and buy_percent >= sell_percent + score_edge
        raw_short_signal = entry_allowed and htf_short_ok and short_regime_ok and short_trigger and short_momentum_ok and sell_percent > buy_percent and sell_percent >= dynamic_min_score and sell_percent >= buy_percent + score_edge
        confirmed_long_signal = raw_long_signal
        confirmed_short_signal = raw_short_signal

        cost_ticks, min_stop_to_cost_ratio = _cost_settings(settings.cost_preset)
        estimated_cost = _cost_tick(symbol, c) * cost_ticks
        raw_stop_dist = max(atrv * settings.atr_stop_mult, c * settings.min_stop_percent / 100.0, estimated_cost * min_stop_to_cost_ratio)
        suggested_long_stop = c - raw_stop_dist
        suggested_long_tp1 = c + raw_stop_dist * settings.rr_tp1
        suggested_short_stop = c + raw_stop_dist
        suggested_short_tp1 = c - raw_stop_dist * settings.rr_tp1
        trail_atr_floor = atrslv * 0.75 if math.isfinite(atrslv) else atrv
        trail_atr = max(atrv, trail_atr_floor)
        trail_mult = 1.45 if eff > 0.50 else 1.20 if eff > 0.35 else 1.00

        buy_entry_now = False
        sell_entry_now = False
        exit_long_now = False
        exit_short_now = False
        tp1_long_now = False
        tp1_short_now = False
        exit_reason_now = ""

        if settings.use_trade_manager and virtual_pos == 1:
            virtual_high_since_entry = h if virtual_high_since_entry is None else max(virtual_high_since_entry, h)
            if not virtual_tp1_hit and virtual_tp1 is not None and h >= virtual_tp1:
                virtual_tp1_hit = True
                tp1_long_now = True
            if virtual_tp1_hit and settings.use_trailing_after_tp1 and virtual_entry is not None and virtual_stop is not None:
                breakeven_stop_long = virtual_entry + estimated_cost
                trail_stop_long = virtual_high_since_entry - trail_atr * trail_mult
                virtual_stop = max(virtual_stop, max(breakeven_stop_long, trail_stop_long))
            stop_exit_long = virtual_stop is not None and l <= virtual_stop
            weakness_exit_long = settings.exit_on_weakness_after_tp1 and virtual_tp1_hit and c < ef and rsiv < 50
            opposite_exit_long = settings.exit_on_opposite_signal and confirmed_short_signal
            if stop_exit_long or weakness_exit_long or opposite_exit_long:
                exit_long_now = True
                exit_reason_now = "stop / trail hit" if stop_exit_long else "momentum weakened after TP1" if weakness_exit_long else "opposite sell signal"
                last_trade_entry = virtual_entry
                last_trade_stop = virtual_stop
                last_trade_tp1 = virtual_tp1
                last_exit_price = c
                last_exit_reason = exit_reason_now
                virtual_pos = 0
                virtual_entry = None
                virtual_stop = None
                virtual_tp1 = None
                virtual_high_since_entry = None
                virtual_low_since_entry = None
                virtual_tp1_hit = False
                virtual_entry_bar = None

        if settings.use_trade_manager and virtual_pos == -1:
            virtual_low_since_entry = l if virtual_low_since_entry is None else min(virtual_low_since_entry, l)
            if not virtual_tp1_hit and virtual_tp1 is not None and l <= virtual_tp1:
                virtual_tp1_hit = True
                tp1_short_now = True
            if virtual_tp1_hit and settings.use_trailing_after_tp1 and virtual_entry is not None and virtual_stop is not None:
                breakeven_stop_short = virtual_entry - estimated_cost
                trail_stop_short = virtual_low_since_entry + trail_atr * trail_mult
                virtual_stop = min(virtual_stop, min(breakeven_stop_short, trail_stop_short))
            stop_exit_short = virtual_stop is not None and h >= virtual_stop
            weakness_exit_short = settings.exit_on_weakness_after_tp1 and virtual_tp1_hit and c > ef and rsiv > 50
            opposite_exit_short = settings.exit_on_opposite_signal and confirmed_long_signal
            if stop_exit_short or weakness_exit_short or opposite_exit_short:
                exit_short_now = True
                exit_reason_now = "stop / trail hit" if stop_exit_short else "momentum weakened after TP1" if weakness_exit_short else "opposite buy signal"
                last_trade_entry = virtual_entry
                last_trade_stop = virtual_stop
                last_trade_tp1 = virtual_tp1
                last_exit_price = c
                last_exit_reason = exit_reason_now
                virtual_pos = 0
                virtual_entry = None
                virtual_stop = None
                virtual_tp1 = None
                virtual_high_since_entry = None
                virtual_low_since_entry = None
                virtual_tp1_hit = False
                virtual_entry_bar = None

        can_open_new_virtual_trade = settings.use_trade_manager and virtual_pos == 0 and (settings.allow_same_bar_flip or (not exit_long_now and not exit_short_now))
        if can_open_new_virtual_trade and confirmed_long_signal:
            virtual_pos = 1
            virtual_entry = c
            virtual_stop = suggested_long_stop
            virtual_tp1 = suggested_long_tp1
            virtual_high_since_entry = h
            virtual_low_since_entry = None
            virtual_tp1_hit = False
            virtual_entry_bar = i
            last_signal_bar = i
            signal_bars.append(i)
            buy_entry_now = True
        elif can_open_new_virtual_trade and confirmed_short_signal:
            virtual_pos = -1
            virtual_entry = c
            virtual_stop = suggested_short_stop
            virtual_tp1 = suggested_short_tp1
            virtual_high_since_entry = None
            virtual_low_since_entry = l
            virtual_tp1_hit = False
            virtual_entry_bar = i
            last_signal_bar = i
            signal_bars.append(i)
            sell_entry_now = True

        if not settings.use_trade_manager and (raw_long_signal or raw_short_signal):
            last_signal_bar = i
            signal_bars.append(i)

        long_setup_text = "liquidity sweep buy setup" if liquidity_long else "second-entry buy setup" if settings.use_second_entry and second_entry_long else "pullback buy setup" if pullback_long else "strong body breakout setup" if strong_breakout_long else "session breakout setup" if normal_breakout_long_ok else "buy setup forming"
        short_setup_text = "liquidity sweep sell setup" if liquidity_short else "second-entry sell setup" if settings.use_second_entry and second_entry_short else "pullback sell setup" if pullback_short else "strong body breakdown setup" if strong_breakout_short else "session breakdown setup" if normal_breakout_short_ok else "sell setup forming"
        if is_warming_up:
            base_reason = "indicator is warming up"
        elif market_text != "market active":
            base_reason = market_text
        elif buy_percent > sell_percent:
            base_reason = long_setup_text
        elif sell_percent > buy_percent:
            base_reason = short_setup_text
        else:
            base_reason = "mixed signals"

        if is_warming_up:
            action = "LOADING"
        elif settings.use_trade_manager and buy_entry_now:
            action = "BUY CONFIRMED"
        elif settings.use_trade_manager and sell_entry_now:
            action = "SELL CONFIRMED"
        elif settings.use_trade_manager and exit_long_now:
            action = "SELL / EXIT LONG"
        elif settings.use_trade_manager and exit_short_now:
            action = "BUY / EXIT SHORT"
        elif settings.use_trade_manager and virtual_pos == 1 and virtual_tp1_hit:
            action = "TRAIL LONG"
        elif settings.use_trade_manager and virtual_pos == 1:
            action = "HOLD LONG"
        elif settings.use_trade_manager and virtual_pos == -1 and virtual_tp1_hit:
            action = "TRAIL SHORT"
        elif settings.use_trade_manager and virtual_pos == -1:
            action = "HOLD SHORT"
        elif not settings.use_trade_manager and raw_long_signal:
            action = "BUY"
        elif not settings.use_trade_manager and raw_short_signal:
            action = "SELL"
        elif market_text != "market active":
            action = "WAIT"
        elif long_bias:
            action = "GET READY TO BUY"
        elif short_bias:
            action = "GET READY TO SELL"
        elif buy_percent > sell_percent:
            action = "LEAN BUY"
        elif sell_percent > buy_percent:
            action = "LEAN SELL"
        else:
            action = "WAIT"

        if exit_long_now or exit_short_now:
            main_reason = exit_reason_now
        elif tp1_long_now:
            main_reason = "TP1 reached; long trail/management active"
        elif tp1_short_now:
            main_reason = "TP1 reached; short trail/management active"
        elif action == "HOLD LONG":
            main_reason = "long trade active from prior buy"
        elif action == "HOLD SHORT":
            main_reason = "short trade active from prior sell"
        elif action == "TRAIL LONG":
            main_reason = "TP1 reached; trailing long stop"
        elif action == "TRAIL SHORT":
            main_reason = "TP1 reached; trailing short stop"
        else:
            main_reason = base_reason

        if settings.use_trade_manager:
            if virtual_pos == 1:
                position_text = "LONG"
            elif virtual_pos == -1:
                position_text = "SHORT"
            elif exit_long_now:
                position_text = "EXITED LONG"
            elif exit_short_now:
                position_text = "EXITED SHORT"
            else:
                position_text = "FLAT"
        else:
            position_text = "OFF"

        entry_display = virtual_entry if virtual_pos != 0 else last_trade_entry if (exit_long_now or exit_short_now) else None
        stop_display = virtual_stop if virtual_pos != 0 else last_trade_stop if (exit_long_now or exit_short_now) else suggested_short_stop if action in ["SELL", "SELL CONFIRMED", "GET READY TO SELL", "LEAN SELL"] else suggested_long_stop
        tp1_display = virtual_tp1 if virtual_pos != 0 else last_trade_tp1 if (exit_long_now or exit_short_now) else suggested_short_tp1 if action in ["SELL", "SELL CONFIRMED", "GET READY TO SELL", "LEAN SELL"] else suggested_long_tp1

        last_row = {
            "symbol": symbol,
            "provider": "yahoo",
            "decision": action,
            "position": position_text,
            "buy_percent": clean_number(buy_percent, 2),
            "sell_percent": clean_number(sell_percent, 2),
            "edge_percent": clean_number(edge_percent, 2),
            "confidence": clean_number(confidence, 2),
            "why": main_reason,
            "market": market_text,
            "last_price": clean_number(c, 6),
            "entry": clean_number(entry_display, 6),
            "stop": clean_number(stop_display, 6),
            "tp1": clean_number(tp1_display, 6),
            "rsi": clean_number(rsiv, 2),
            "atr": clean_number(atrv, 6),
            "session": sess["preset"],
            "signals_this_hour": signals_this_hour,
            "max_signals_per_hour": settings.max_signals_per_hour,
            "bar_time": str(ts),
            "scan_time": datetime.now(timezone.utc).isoformat(),
        }

    return last_row or err("scan produced no result")


def sort_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(r: Dict[str, Any]) -> Tuple[int, float]:
        return (DECISION_ORDER.get(r.get("decision", "ERROR"), 99), -(r.get("confidence") or 0.0))
    return sorted(rows, key=key)


def run_scan(settings: ScannerSettings, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    symbols = symbols or ALL_SYMBOLS
    started = time.time()
    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=settings.max_workers) as ex:
        futs = {ex.submit(scan_symbol, sym, settings): sym for sym in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                rows.append(fut.result())
            except Exception as e:
                now = datetime.now(timezone.utc).isoformat()
                rows.append({
                    "symbol": sym, "provider": "yahoo", "decision": "ERROR", "position": "n/a",
                    "buy_percent": None, "sell_percent": None, "edge_percent": None, "confidence": None,
                    "why": str(e), "market": "error", "last_price": None, "entry": None,
                    "stop": None, "tp1": None, "rsi": None, "atr": None, "session": "",
                    "bar_time": "", "scan_time": now,
                })
    rows = sort_results(rows)
    action_counts: Dict[str, int] = {}
    for row in rows:
        action_counts[row["decision"]] = action_counts.get(row["decision"], 0) + 1
    return {
        "sweep": 1,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.time() - started, 2),
        "interval": settings.interval,
        "range": settings.yahoo_range,
        "entry_mode": settings.entry_mode,
        "session_preset": settings.session_preset,
        "use_htf": settings.use_htf,
        "total": len(rows),
        "confirmed": sum(1 for r in rows if r["decision"] in ("BUY CONFIRMED", "SELL CONFIRMED")),
        "active_positions": sum(1 for r in rows if r["position"] in ("LONG", "SHORT")),
        "action_counts": action_counts,
        "rows": rows,
    }


def settings_from_query(query: Dict[str, str]) -> ScannerSettings:
    def q(name: str, default: Any) -> Any:
        v = query.get(name)
        return default if v in (None, "") else v

    def q_bool(name: str, default: bool) -> bool:
        v = str(q(name, str(default))).lower()
        return v in ("1", "true", "yes", "on")

    def q_int(name: str, default: int, lo: int, hi: int) -> int:
        try:
            return max(lo, min(hi, int(q(name, default))))
        except Exception:
            return default

    entry_mode = q("entryMode", "Fast 5m")
    if entry_mode not in ("Fast 5m", "Balanced", "Conservative"):
        entry_mode = "Fast 5m"
    session_preset = q("sessionPreset", "AUTO")
    if session_preset not in ("AUTO", "Crypto 24/7", "US Indices RTH", "US Indices Extended", "Forex London/NY", "Custom"):
        session_preset = "AUTO"
    cost_preset = q("costPreset", "Normal")
    if cost_preset not in ("Tight", "Normal", "Wide", "Volatile"):
        cost_preset = "Normal"
    interval = q("interval", "5m")
    if interval not in ("1m", "2m", "5m", "15m", "30m", "60m"):
        interval = "5m"
    yahoo_range = q("range", "10d")
    if yahoo_range not in ("1d", "5d", "10d", "30d", "60d"):
        yahoo_range = "10d"

    return ScannerSettings(
        entry_mode=entry_mode,
        loading_bars=q_int("loadingBars", 100, 20, 500),
        session_preset=session_preset,
        use_session_phase_filter=q_bool("useSessionPhaseFilter", True),
        min_stop_percent=float(q("minStopPercent", 0.08)),
        atr_stop_mult=float(q("atrStopMult", 1.2)),
        rr_tp1=float(q("rrTP1", 1.5)),
        use_trade_manager=q_bool("useTradeManager", True),
        exit_on_opposite_signal=q_bool("exitOnOppositeSignal", True),
        exit_on_weakness_after_tp1=q_bool("exitOnWeaknessAfterTP1", True),
        use_trailing_after_tp1=q_bool("useTrailingAfterTP1", True),
        max_signals_per_hour=q_int("maxSignalsPerHour", 3, 1, 10),
        cooldown_bars=q_int("cooldownBars", 2, 0, 30),
        use_htf=q_bool("useHTF", True),
        use_second_entry=q_bool("useSecondEntry", True),
        cost_preset=cost_preset,
        interval=interval,
        yahoo_range=yahoo_range,
        max_workers=q_int("workers", 12, 1, 20),
    )
