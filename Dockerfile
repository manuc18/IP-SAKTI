# ─── IP-SAKTI Python AI Backend — Production Docker Image ──────────────────
# Builds a lean image (~200 MB) containing only what's needed to serve
# the grounded pipeline with Groq as the answer provider.
# torch and tesseract are excluded (--no-dense flag skips BGE-M3 embeddings).

FROM python:3.12-slim

# Metadata
LABEL org.opencontainers.image.title="IP-SAKTI AI Pipeline"
LABEL org.opencontainers.image.description="Grounded IP legal Q&A — Render deployment"

# Don't buffer Python output so logs show in real time on Render
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /srv

# ── 1. Install Python deps (minimal — no ML frameworks) ─────────────────────
COPY requirements_prod.txt .
RUN pip install --no-cache-dir -r requirements_prod.txt

# ── 2. Copy source ────────────────────────────────────────────────────────────
# ai/ — retrieval, reranking, grounding, answering logic
COPY ai/ ./ai/
# app/ — HTTP server and pipeline wiring
COPY app/ ./app/
# corpus — 38 MB of processed legal texts (chunks + structured docs)
COPY data/processed/ ./data/processed/
COPY data/metadata/ ./data/metadata/
# seed cache — validated answers from real pipeline runs, shipped so the demo
# questions survive Render's ephemeral filesystem across restarts and redeploys
COPY data/demo_cache/ ./data/demo_cache/

# ── 3. Expose & Run ──────────────────────────────────────────────────────────
# Render injects $PORT automatically. Shell form is required to expand it.
EXPOSE 8000
CMD python3 -m app.server \
      --provider groq \
      --no-dense \
      --host 0.0.0.0 \
      --port ${PORT:-8000}
