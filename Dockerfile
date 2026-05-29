FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    postgresql-client \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Create artifact and log directories
RUN mkdir -p artifacts/models artifacts/features artifacts/labels \
    artifacts/backtest_results artifacts/trade_outcomes artifacts/knowledge/session_plans \
    artifacts/knowledge logs

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "zeus.live.trading_loop"]
