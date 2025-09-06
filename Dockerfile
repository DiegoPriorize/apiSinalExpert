
# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1     TZ=UTC

# System deps (optional: tzdata for TZ)
RUN apt-get update && apt-get install -y --no-install-recommends tzdata build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (cache layer)
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy app
COPY ohlcv_signal_api.py ./

# Non-root user (good practice)
RUN useradd -m appuser
USER appuser

EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "ohlcv_signal_api:app", "--host", "0.0.0.0", "--port", "8000"]
