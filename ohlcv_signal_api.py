from typing import List, Literal, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import numpy as np
from datetime import datetime, timezone

app = FastAPI(title="OHLCV 1m Signal API", version="1.2")

# ------------------ Utils ------------------
def _to_np(x):
    return np.asarray(x, dtype=float)

def _csv_to_list(v):
    """Aceita '1,2,3' ou '[1,2,3]' e converte para List[float]."""
    if isinstance(v, str):
        s = v.strip().replace("[", "").replace("]", "")
        if not s:
            return []
        parts = [p.strip() for p in s.split(",")]
        return [float(p) for p in parts if p]
    return v

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
    return 100 - (100 / (1 + rs))

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
    std = np.empty_like(close, dtype=float); std[:] = np.nan
    half = period // 2
    for i in range(len(close)):
        start = max(0, i - half)
        end = min(len(close), start + period)
        window = close[start:end]
        if len(window) >= max(5, period//2):
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
    return np.convolve(arr, np.ones(period)/period, mode='same')

# ------------------ Core ------------------
def compute_signal(t, o, h, l, c, v):
    t = np.array(t, dtype=float)
    # Normaliza timestamps ms -> s, se necessário
    if np.nanmedian(t) > 1e11:
        t = t / 1000.0

    o = _to_np(o); h = _to_np(h); l = _to_np(l); c = _to_np(c); v = _to_np(v)
    assert len(t)==len(o)==len(h)==len(l)==len(c)==len(v), "Arrays com tamanhos diferentes."
    n = len(c)

    # Mínimo exigido
    if n < 50:
        raise HTTPException(status_code=422, detail="Forneça pelo menos 50 candles (timestamp/open/high/low/close/volume).")

    # Períodos: adaptativos em 50–59, padrão em >=60
    if n < 60:
        ema_fast_p = max(6, min(9,  n//3))     # alvo ~9
        ema_slow_p = max(12, min(21, n//2))    # alvo ~21
        rsi_p      = max(8, min(14, n//4))     # alvo ~14
        macd_fast  = max(8, min(12, n//4))     # alvo ~12
        macd_slow  = max(16, min(26, n//2))    # alvo ~26
        macd_sig   = max(5, min(9,  n//6))     # alvo ~9
        bb_p       = max(14, min(20, n//2))    # alvo ~20
        atr_p      = max(8, min(14, n//4))     # alvo ~14
        vma_p      = max(12, min(20, n//3))    # alvo ~20
    else:
        ema_fast_p, ema_slow_p = 9, 21
        rsi_p = 14
        macd_fast, macd_slow, macd_sig = 12, 26, 9
        bb_p = 20
        atr_p = 14
        vma_p = 20

    # Indicadores
    ema9  = ema(c, ema_fast_p)
    ema21 = ema(c, ema_slow_p)
    rsi14 = rsi(c, rsi_p)

    ema_fast = ema(c, macd_fast)
    ema_slow = ema(c, macd_slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, macd_sig)
    hist = macd_line - signal_line

    bb_ma, bb_up, bb_lo = bollinger(c, bb_p, 2.0)
    atr14 = atr(h, l, c, atr_p)
    vma20 = sma(v, vma_p)

    i = n - 1
    score = 0.0
    weights = {"trend":2.0,"momentum":1.5,"macd":1.5,"boll":1.0,"volume":0.5,"volatility":0.5,"price_action":1.0}

    # Trend
    score += weights["trend"] if ema9[i] > ema21[i] else -weights["trend"]
    # Momentum (RSI)
    if rsi14[i] >= 55: score += weights["momentum"]
    elif rsi14[i] <= 45: score -= weights["momentum"]
    # MACD
    if macd_line[i] > signal_line[i] and hist[i] > 0: score += weights["macd"]
    elif macd_line[i] < signal_line[i] and hist[i] < 0: score -= weights["macd"]
    # Bollinger
    if c[i] < bb_lo[i]: score += weights["boll"]*0.5
    elif c[i] > bb_up[i]: score -= weights["boll"]*0.5
    # Volume
    if v[i] > vma20[i] and np.sign(c[i]-o[i]) == np.sign(score): score += weights["volume"]*0.5
    # Volatilidade
    atr_pct = atr14[i]/c[i] if c[i] else 0
    if atr_pct < 0.001: score *= 0.7
    # Price action
    body = c[i]-o[i]; rng = max(h[i]-l[i], 1e-12); body_pct = abs(body)/rng
    if body > 0 and body_pct > 0.5: score += weights["price_action"]*0.5
    elif body < 0 and body_pct > 0.5: score -= weights["price_action"]*0.5

    # Decisão
    buy_th, sell_th = 1.5, -1.5
    if score >= buy_th: action = "COMPRA"
    elif score <= sell_th: action = "VENDA"
    else: action = "COMPRA" if body >= 0 else "VENDA"

    next_ts = int(t[i] + 60)

    # Backtest adaptativo (usa até 500, mas garante >=8 comparações quando possível)
    window = max(8, min(500, n-2))
    start = max(1, n - window - 1)
    correct = 0; total = 0
    for j in range(start, n - 1):
        sc = 0.0
        sc += weights["trend"] if ema9[j] > ema21[j] else -weights["trend"]
        if rsi14[j] >= 55: sc += weights["momentum"]
        elif rsi14[j] <= 45: sc -= weights["momentum"]
        if macd_line[j] > signal_line[j] and hist[j] > 0: sc += weights["macd"]
        elif macd_line[j] < signal_line[j] and hist[j] < 0: sc -= weights["macd"]
        if c[j] < bb_lo[j]: sc += weights["boll"]*0.5
        elif c[j] > bb_up[j]: sc -= weights["boll"]*0.5
        if v[j] > vma20[j] and np.sign(c[j]-o[j]) == np.sign(sc): sc += weights["volume"]*0.5
        atr_pct_j = atr14[j]/c[j] if c[j] else 0
        if atr_pct_j < 0.001: sc *= 0.7
        body_j = c[j]-o[j]; rng_j = max(h[j]-l[j], 1e-12); body_pct_j = abs(body_j)/rng_j
        if body_j > 0 and body_pct_j > 0.5: sc += weights["price_action"]*0.5
        elif body_j < 0 and body_pct_j > 0.5: sc -= weights["price_action"]*0.5

        pred_up = sc >= buy_th or (buy_th > sc > sell_th and body_j >= 0)
        up_next = (c[j+1] - c[j]) > 0
        correct += int(pred_up == up_next); total += 1

    acc = (correct/total*100) if total>0 else 50.0
    conf = min(1.0, max(0.0, 0.5*(acc/100.0) + 0.5*(min(3.0, abs(score))/3.0)))
    conf_pct = round(conf*100, 2)

    return action, next_ts, round(acc, 2), conf_pct, score

# ------------------ Models ------------------
class OHLCVPayload(BaseModel):
    # Aceita lista OU string CSV
    timestamp: Union[List[float], str] = Field(..., description="Unix seconds (1m). Lista JSON ou CSV string.")
    open: Union[List[float], str]
    high: Union[List[float], str]
    low: Union[List[float], str]
    close: Union[List[float], str]
    volume: Union[List[float], str]
    symbol: Optional[str] = None
    timeframe: Literal["1m"] = "1m"

    # converter CSV -> lista de float
    @field_validator("timestamp", "open", "high", "low", "close", "volume", mode="before")
    @classmethod
    def _parse_csv(cls, v):
        return _csv_to_list(v)

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

# ------------------ Endpoint ------------------
@app.post("/signal", response_model=SignalResponse)
def signal(payload: OHLCVPayload):
    action, next_ts, acc, conf, score = compute_signal(
        payload.timestamp, payload.open, payload.high, payload.low, payload.close, payload.volume
    )
    last_ts = int(_to_np(payload.timestamp)[-1])
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
