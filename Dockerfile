# Multi-Intent Question Generator Dockerfile

# Use slim Python for smaller image size
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if any (none critical for now, but keeping apt-get update for safety)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for cache optimization
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Build-time arguments (Secrets) ───────────────────────────
# WARNING: Values passed here will be visible in docker history.
# Use --build-arg VARIABLE=value or pass .env file at runtime.
ARG GROQ_API_KEYS
ARG GEMINI_API_KEYS
ARG HF_API_KEYS
ARG OPENROUTER_API_KEYS
ARG MONGO_URI
ARG MONGO_DB_NAME

# Set environment variables from build args
ENV GROQ_API_KEYS=${GROQ_API_KEYS}
ENV GEMINI_API_KEYS=${GEMINI_API_KEYS}
ENV HF_API_KEYS=${HF_API_KEYS}
ENV OPENROUTER_API_KEYS=${OPENROUTER_API_KEYS}
ENV MONGO_URI=${MONGO_URI}
ENV MONGO_DB_NAME=${MONGO_DB_NAME}
# ─────────────────────────────────────────────────────────────

# Copy source code
COPY . .

# Default command
# This can be overridden: docker run my-image python main.py --total 100
ENTRYPOINT ["python", "main.py"]
CMD ["--total", "50"]
