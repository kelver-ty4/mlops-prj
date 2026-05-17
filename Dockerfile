# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY madewithml/ ./madewithml/
COPY data/       ./data/

# Model artifacts are mounted at runtime (not baked into the image)
# docker run -v /tmp/mlflow:/tmp/mlflow ...
ENV MODEL_DIR=/tmp/eval_artifacts

EXPOSE 8000

# Health check — matches the Jenkins Health Check stage
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "madewithml.serve:app", "--host", "0.0.0.0", "--port", "8000"]
