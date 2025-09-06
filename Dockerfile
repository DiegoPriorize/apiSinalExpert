# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Variáveis de ambiente básicas
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

# Dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata build-essential && \
    rm -rf /var/lib/apt/lists/*

# Diretório de trabalho
WORKDIR /app

# Instala as dependências
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copia a API
COPY ohlcv_signal_api.py .

# Cria usuário não-root
RUN useradd -m appuser
USER appuser

# Porta exposta (Render usa $PORT dinamicamente)
EXPOSE 8000

# Start command exigido pelo Render
CMD ["sh", "-c", "uvicorn ohlcv_signal_api:app --host 0.0.0.0 --port ${PORT:-8000}"]
