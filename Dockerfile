# Multi-stage build for the Delhi PM2.5 agent API.
# Targets linux/arm64 (Oracle Ampere A1). One image serves both the API and the
# Streamlit UI; docker-compose runs them as two processes with different CMDs.
#
#   docker build -t pm25-agent .
#   (CI builds/pushes the arm64 image to GHCR; the VM pulls it.)

# ---------- builder: compile/install deps into a venv ----------
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install all deps for arm64 (Oracle Ampere A1). Notes specific to aarch64:
#  - There is NO `tensorflow-cpu` wheel for arm64; the full `tensorflow` wheel is
#    already CPU-only on aarch64, so requirements.txt's plain `tensorflow` is correct.
#  - torch's default PyPI wheel is CPU-only on arm64, so no special --index-url is
#    needed (that trick only avoids the CUDA build on x86_64).
# Hence we just install requirements.txt as-is. If the TF aarch64 wheel is ever
# dropped upstream, the fallback is TFLite (see DEPLOY.md).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so retrieval never fetches from HF at runtime.
ENV HF_HOME=/opt/hf
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# ---------- final: slim, non-root ----------
FROM python:3.10-slim AS final

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf /opt/hf
COPY src/ ./src/
COPY models/ ./models/
COPY data/chroma_db/ ./data/chroma_db/
COPY data/processed/pm25_daily_final.csv ./data/processed/pm25_daily_final.csv

RUN chown -R appuser:appuser /app /opt/hf
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
