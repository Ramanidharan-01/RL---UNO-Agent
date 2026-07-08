# ── Stage 1: dependency layer ─────────────────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /app

# Install system libs required by JAX / numpy.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Upgrade pip first to avoid legacy resolver issues.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed site-packages from the deps stage.
COPY --from=deps /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application source.
COPY . .

# The checkpoint is mounted at runtime via docker-compose volume.
RUN mkdir -p /app/checkpoints

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JAX_PLATFORM_NAME=cpu

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
