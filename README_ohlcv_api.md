# OHLCV 1m Signal API (FastAPI + Docker + Render)

API que analisa candles de **1 minuto** (`T, O, H, L, C, V`) com um **ensemble de indicadores técnicos** e retorna a decisão **COMPRA** ou **VENDA** para **1 minuto após** o último timestamp.  
Além do sinal, a API retorna uma **estimativa de precisão histórica** via **walk-forward backtest** e uma **confiança** combinando força do sinal atual com essa precisão.

## Sumário
- [Arquitetura](#arquitetura)
- [Instalação e uso local (sem Docker)](#instalação-e-uso-local-sem-docker)
- [Uso com Docker](#uso-com-docker)
- [Deploy no Render (Docker)](#deploy-no-render-docker)
- [Endpoint e Contratos](#endpoint-e-contratos)
  - [POST /signal](#post-signal)
  - [Schema de entrada](#schema-de-entrada)
  - [Schema de saída](#schema-de-saída)
  - [Exemplos de requisição e resposta](#exemplos-de-requisição-e-resposta)
- [Indicadores e Lógica de Decisão](#indicadores-e-lógica-de-decisão)
- [Boas práticas e observações](#boas-práticas-e-observações)
- [Licença](#licença)

---

## Arquitetura
- **Framework**: FastAPI
- **Servidor**: Uvicorn
- **Linguagem**: Python 3.11+
- **Dependências**: `fastapi`, `uvicorn[standard]`, `pydantic`, `numpy`
- **Timeframe suportado**: 1 minuto (arrays com candles 1m)

---

## Instalação e uso local (sem Docker)

1) Crie e ative um ambiente virtual (opcional):
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

2) Instale as dependências:
```bash
pip install fastapi uvicorn[standard] pydantic numpy
```

3) Rode a API:
```bash
uvicorn ohlcv_signal_api:app --host 0.0.0.0 --port 8000
```

4) Acesse:
- Docs interativas (Swagger): `http://localhost:8000/docs`
- Redoc: `http://localhost:8000/redoc`

---

## Uso com Docker

### Dockerfile
```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \\
    tzdata build-essential && \\
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ohlcv_signal_api.py .

RUN useradd -m appuser
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn ohlcv_signal_api:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

### Build/Run local
```bash
docker build -t ohlcv-signal-api .
docker run --rm -p 8000:8000 --name ohlcv ohlcv-signal-api
# http://localhost:8000/docs
```

### Docker Compose (opcional)
```yaml
services:
  api:
    build: .
    image: ohlcv-signal-api:latest
    ports:
      - "8000:8000"
    restart: unless-stopped
```
```bash
docker compose up --build -d
```

---

## Deploy no Render (Docker)

1. Suba o repo no **GitHub** (branch `main`).  
2. No Render: **New +** → **Web Service** → **Build & deploy from a Git repository**.  
3. Selecione o repo. Em **Runtime**, escolha **Docker**.  
4. Deploy.  
5. Teste em: `https://<seu-servico>.onrender.com/docs`

---

## Endpoint e Contratos

### POST `/signal`
Recebe arrays OHLCV de 1 minuto e retorna a decisão **COMPRA** ou **VENDA**.

#### Schema de entrada
```json
{
  "timestamp": [float],
  "open": [float],
  "high": [float],
  "low": [float],
  "close": [float],
  "volume": [float],
  "symbol": "BTCUSDT",
  "timeframe": "1m"
}
```

#### Schema de saída
```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1m",
  "last_timestamp": 1757178120,
  "decision_timestamp": 1757178180,
  "decision_time_iso": "2025-09-06T15:11:00+00:00",
  "decision": "COMPRA",
  "backtest_accuracy_pct": 58.67,
  "confidence_pct": 71.2,
  "score": 2.05
}
```

---

## Indicadores e Lógica de Decisão
- **EMA(9/21)** → tendência
- **RSI(14)** → momentum
- **MACD(12,26,9)** → convergência/divergência
- **Bandas de Bollinger(20, 2σ)** → sobrecompra/sobrevenda
- **ATR(14)** → volatilidade
- **SMA20 do volume** → confirmação de volume
- **Price Action** → corpo da vela corrente

Decisão = soma ponderada → COMPRA, VENDA, ou neutro (desempate pela vela).  
Backtest = walk-forward no histórico enviado.  

---

## Boas práticas e observações
- Envie ≥ 60 candles (ideal 300–500).  
- Converta timestamps de **ms para s** antes de enviar.  
- Use como sinal técnico; não substitui gestão de risco.  

---

## Licença
Uso livre, sem garantias.  
