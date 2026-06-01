FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download BGE-M3 at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY app/ app/
COPY scripts/ scripts/

EXPOSE 7860
CMD [
    "streamlit", "run", "app/main.py",
    "--server.port=7860",
    "--server.address=0.0.0.0",
    "--server.headless=true",
]
