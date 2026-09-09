# ── Stage: runtime image ─────────────────────────────────────────────────────
# Use a slim Python base.  The CPU-only torch wheel pulled transitively by
# optimum during the preload step is large; a multi-stage build could strip it
# after export, but a single stage keeps the file simple for now.
FROM python:3.11-slim

# Prevent .pyc files and enable unbuffered stdout/stderr for clean log output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Cap CPU parallelism at the image level so every process that reads these env
# vars (ORT, MKL, OpenMP, tokenizers) inherits the limit without extra code.
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

# CORS origins — override at `docker run` time, e.g.:
#   -e ALLOWED_ORIGINS="https://myapp.example.com"
ENV ALLOWED_ORIGINS="http://localhost:5173"

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer independently of code
# changes.
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY . .

# ── Build-time model export and quantization ─────────────────────────────────
# preload.py exports deepset/minilm-uncased-squad2 and the cross-encoder to
# ONNX + INT8 and saves them to onnx_model/ and onnx_reranker/ respectively.
# Running this during the build bakes the model weights into the image so the
# container starts cold with no download latency.
RUN python preload.py

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8000

# Single worker: the pipeline is CPU-bound and memory-constrained (512 MB
# ceiling), so multiple workers would each load a separate model copy.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
