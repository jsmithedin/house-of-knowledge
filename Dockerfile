FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Pre-download BGE-M3 at build time
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY app/ app/
COPY scripts/ scripts/

EXPOSE 7860
CMD ["uv", "run", "streamlit", "run", "app/main.py",
     "--server.port=7860",
     "--server.address=0.0.0.0",
     "--server.headless=true"]
