
from typing import List, Literal, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import uvicorn
from datetime import datetime, timezone, timedelta

app = FastAPI(title="OHLCV 1m Signal API", version="1.0")

# ------------------ Utilities ------------------
def _to_np(x):
    return np.asarray(x, dtype=float)

def ema(arr: np.ndarray, period: int) -> np.ndarray:
    arr = _to_np(arr)
    alpha = 2 / (period + 1.0)
    out = np.empty_like(arr, dtype=float)
    out[:] = np.nan
    if len(arr) == 0:
        return out
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out

def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    close = _to_np(close)
    deltas = np.diff(close, prepend=close[0])
    gains = np.clip(deltas, 0, None)
    losses = np.clip(-deltas, 0, None)
    avg_gain = ema(gains, period)
    avg_loss = ema(losses, period)
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(close: np.ndarray, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(close: np.ndarray, period=20, stdev=2.0):
    close = _to_np(close)
    ma = np.convolve(close, np.ones(period)/period, mode='same')
    # For std, compute rolling with simple loop to avoid pandas
    std = np.empty_like(close, dtype=float)
    std[:] = np.nan
    half = period // 2
    for i in range(len(close)):
        start = max(0, i - half)
        end = min(len(close), start + period)
        window = close[start:end]
        if len(window) >= max(5, period//2):  # need enough points
            std[i] = np.std(window, ddof=0)
    upper = ma + stdev * std
    lower = ma - stdev * std
    return ma, upper, lower

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period=14):
    high = _to_np(high); low=_to_np(low); close=_to_np(close)
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return ema(tr, period)

def sma(arr: np.ndarray, period: int):
    arr = _to_np(arr)
    if len(arr) < period:
        out = np.full_like(arr, np.nan, dtype=float)
        if len(arr)>0:
            out[:] = np.mean(arr)
        return out
    kernel = np.ones(period)/period
    out = np.convolve(arr, kernel, mode='same')
    return out

# ------------------ Signal Logic ------------------
def compute_signal(t, o, h, l, c, v):
    t = np.array(t, dtype=float)
    o = _to_np(o); h = _to_np(h); l = _to_np(l); c = _to_np(c); v = _to_np(v)

    assert len(t)==len(o)==len(h)==len(l)==len(c)==len(v), "All arrays must be same length"
    n = len(c)
    if n < 60:
        raise ValueError("Provide at least 60 candles for stable indicators.")

    ema9 = ema(c, 9)
    ema21 = ema(c, 21)
    rsi14 = rsi(c, 14)
    macd_line, signal_line, hist = macd(c, 12, 26, 9)
    bb_ma, bb_up, bb_lo = bollinger(c, 20, 2.0)
    atr14 = atr(h, l, c, 14)
    vma20 = sma(v, 20)

    # latest values
    i = n - 1

    # Votes with weights
    score = 0.0
    weights = {
        "trend": 2.0,
        "momentum": 1.5,
        "macd": 1.5,
        "boll": 1.0,
        "volume": 0.5,
        "volatility": 0.5,
        "price_action": 1.0,
    }

    # Trend: EMA9 vs EMA21
    if ema9[i] > ema21[i]:
        score += weights["trend"]
    else:
        score -= weights["trend"]

    # Momentum: RSI
    if rsi14[i] >= 55:
        score += weights["momentum"]
    elif rsi14[i] <= 45:
        score -= weights["momentum"]
    # between 45-55 -> neutral

    # MACD
    if macd_line[i] > signal_line[i] and hist[i] > 0:
        score += weights["macd"]
    elif macd_line[i] < signal_line[i] and hist[i] < 0:
        score -= weights["macd"]

    # Bollinger
    if c[i] < bb_lo[i]:
        score += weights["boll"] * 0.5  # oversold bounce potential
    elif c[i] > bb_up[i]:
        score -= weights["boll"] * 0.5  # overbought mean reversion

    # Volume confirmation
    if v[i] > vma20[i] and np.sign(c[i]-o[i]) == np.sign(score):
        score += weights["volume"] * 0.5

    # Volatility filter: avoid calls when ATR extremely low (range-bound)
    atr_pct = atr14[i]/c[i] if c[i] else 0
    if atr_pct < 0.001:  # <0.1%
        # penalize confidence
        score *= 0.7

    # Price action: last candle body direction & size
    body = c[i] - o[i]
    rng = max(h[i]-l[i], 1e-12)
    body_pct = abs(body) / rng
    if body > 0 and body_pct > 0.5:
        score += weights["price_action"] * 0.5
    elif body < 0 and body_pct > 0.5:
        score -= weights["price_action"] * 0.5

    # Thresholds
    buy_th = 1.5
    sell_th = -1.5
    if score >= buy_th:
        action = "COMPRA"
    elif score <= sell_th:
        action = "VENDA"
    else:
        # tie-breaker by candle direction
        action = "COMPRA" if body >= 0 else "VENDA"

    # Next timestamp = last + 60s
    next_ts = int(t[i] + 60)

    # ------------------ Accuracy Estimate via walk-forward ------------------
    # Predict next-candle direction on past window and score accuracy.
    window = min(500, n-30)  # use up to 500 samples
    correct = 0; total = 0
    for j in range(n - window - 1, n - 1):
        # recompute a lightweight score using precomputed arrays
        sc = 0.0
        if ema9[j] > ema21[j]: sc += weights["trend"]
        else: sc -= weights["trend"]
        if rsi14[j] >= 55: sc += weights["momentum"]
        elif rsi14[j] <= 45: sc -= weights["momentum"]
        if macd_line[j] > signal_line[j] and hist[j] > 0: sc += weights["macd"]
        elif macd_line[j] < signal_line[j] and hist[j] < 0: sc -= weights["macd"]
        if c[j] < bb_lo[j]: sc += weights["boll"] * 0.5
        elif c[j] > bb_up[j]: sc -= weights["boll"] * 0.5
        if v[j] > vma20[j] and np.sign(c[j]-o[j]) == np.sign(sc): sc += weights["volume"] * 0.5
        atr_pct_j = atr14[j]/c[j] if c[j] else 0
        if atr_pct_j < 0.001: sc *= 0.7
        body_j = c[j]-o[j]; rng_j = max(h[j]-l[j], 1e-12)
        body_pct_j = abs(body_j)/rng_j
        if body_j > 0 and body_pct_j > 0.5: sc += weights["price_action"] * 0.5
        elif body_j < 0 and body_pct_j > 0.5: sc -= weights["price_action"] * 0.5

        pred_up = sc >= buy_th or (buy_th > sc > sell_th and body_j >= 0)
        # realized next 1m direction:
        up_next = (c[j+1] - c[j]) > 0
        if pred_up == up_next:
            correct += 1
        total += 1
    acc = float(correct)/total*100 if total>0 else 50.0

    # Confidence combines accuracy and absolute score magnitude (squashed 0-1)
    conf = min(1.0, max(0.0, 0.5*(acc/100.0) + 0.5*(min(3.0, abs(score))/3.0)))
    conf_pct = round(conf*100, 2)

    return action, next_ts, round(acc, 2), conf_pct, score

class OHLCVPayload(BaseModel):
    timestamp: List[float] = Field(..., description="Unix seconds for each 1m candle")
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[float]
    symbol: Optional[str] = Field(default=None, description="(optional) symbol/ticker")
    timeframe: Literal["1m"] = "1m"

class SignalResponse(BaseModel):
    symbol: Optional[str]
    timeframe: str
    last_timestamp: int
    decision_timestamp: int
    decision_time_iso: str
    decision: Literal["COMPRA", "VENDA"]
    backtest_accuracy_pct: float
    confidence_pct: float
    score: float

@app.post("/signal", response_model=SignalResponse)
def signal(payload: OHLCVPayload):
    action, next_ts, acc, conf, score = compute_signal(
        payload.timestamp, payload.open, payload.high, payload.low, payload.close, payload.volume
    )
    last_ts = int(payload.timestamp[-1])
    decision_iso = datetime.fromtimestamp(next_ts, tz=timezone.utc).isoformat()
    return SignalResponse(
        symbol=payload.symbol,
        timeframe=payload.timeframe,
        last_timestamp=last_ts,
        decision_timestamp=next_ts,
        decision_time_iso=decision_iso,
        decision=action,
        backtest_accuracy_pct=acc,
        confidence_pct=conf,
        score=round(float(score), 3),
    )

if __name__ == "__main__":
    uvicorn.run("ohlcv_signal_api:app", host="0.0.0.0", port=8000, reload=False)
