# PaperQA — PDF Question Answering

PaperQA answers natural-language questions about an uploaded PDF (research papers, in
particular) by extracting the answer as a verbatim span from the document's own text and
reporting the PDF page it came from. It does not generate or paraphrase — every answer is a
substring of the source document, so it can be checked directly against the page it cites.
The backend is designed to run inside a memory-constrained container (INT8-quantized ONNX
models, single worker) rather than assuming a GPU or a large memory budget.

## Key Features

- **Extractive, page-grounded answers** — the QA model selects a character span from the
  retrieved context; the backend maps that span back through the document's sentence/page
  metadata to report the 0-indexed PDF page the answer came from (`page_number` in the API
  response).
- **Layout-aware PDF extraction with two-column support** — text is extracted with PyMuPDF
  using word-level bounding boxes. A gutter-based detector (`_detect_column_gutter` in
  `extraction.py`) looks for a genuine, consistently-positioned gap between columns before
  treating a page as two-column, rather than assuming any page with words on both sides of
  its horizontal midpoint is multi-column. Falls back to a `pypdf`-based extractor if the
  PyMuPDF path fails.
- **TF-IDF retrieval with a cross-encoder reranking stage** — candidate passages ("windows"
  of up to three overlapping sentences) are shortlisted by TF-IDF cosine similarity against the
  question, then re-scored by an ONNX cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
  and pruned by a score-gap margin before being assembled into the QA model's context, subject
  to a token budget. (`RETRIEVAL_MODE` also accepts `dense` and `hybrid`; only `tfidf` is
  implemented today — the other two values currently fall back to it, reserved for a future
  dense-retrieval addition.)
- **ONNX-quantized inference** — both the QA model (`deepset/minilm-uncased-squad2`) and the
  reranker are exported to ONNX and dynamically quantized to INT8 at build time
  (`preload.py`), then served with `onnxruntime` on CPU. Both are lazily loaded on first use
  and unloaded again after 10 minutes of inactivity.
- **References/bibliography filtering** — sentences after a detected references heading are
  excluded from the retrieval candidate pool so citation-list text can't outrank real body
  text on keyword overlap.
- **TF-IDF fallback for QA** — if the ONNX QA model is unavailable or its top span confidence
  falls below threshold, a TF-IDF keyword match against the retrieved context is returned
  instead of failing the request.
- **Bounded resource usage** — a 25-page cap on extraction, a 20MB upload cap, a SHA-256-keyed
  in-memory document cache (LRU, 3 entries), and single-threaded ONNX/BLAS execution, all
  aimed at running the full pipeline inside a 512MB container.

## Architecture

**Pipeline** (`backend/main.py` wires these together; each stage lives in its own module):

```
PDF upload
  -> extraction.py    parse PDF (layout-aware PyMuPDF, or pypdf fallback), split into
                       sentences (nltk), tag each sentence with its page number and
                       whether it falls after a references heading
  -> retrieval.py      TF-IDF-shortlist candidate sentence windows, apply an adaptive
                        percentile score threshold, optionally rerank the shortlist with
                        the ONNX cross-encoder, assemble the surviving windows into a
                        single context string within a token budget
  -> qa.py              run the ONNX QA model over (question, context) to extract an answer
                        span, or fall back to a TF-IDF keyword match on low confidence
  -> retrieval.py       resolve_page_number() maps the answer's character offset back
                        through the window/sentence offset tables to a PDF page number
  -> response           answer, its context, char offsets, confidence, and page_number
```

**Deployment topology:**

- **Backend** — a Docker container (`backend/Dockerfile`, `python:3.11-slim`) running
  `uvicorn main:app` on port 8000, deployed on **Render**. There is no `render.yaml` in this
  repo; the service's build settings (runtime, root directory, Dockerfile path) are
  configured through the Render dashboard.
- **Frontend** — a static Vite/React build (`frontend/`), deployed on **Vercel**, calling the
  backend over HTTPS.

## Setup & Installation

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
python preload.py               # exports + INT8-quantizes the QA model and reranker to
                                 # onnx_model/ and onnx_reranker/, and downloads the nltk
                                 # punkt tokenizer data to nltk_data/
uvicorn main:app --reload --port 8000
```

`preload.py` downloads and converts both models (`deepset/minilm-uncased-squad2` and
`cross-encoder/ms-marco-MiniLM-L-6-v2`) the first time it runs — this needs network access
and can take a few minutes. It's also run automatically inside `backend/Dockerfile` during
the image build, so a locally-built Docker image doesn't need this step repeated.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Runs on `http://localhost:5173` (Vite's default) and calls the backend at the URL in
`VITE_API_URL`, defaulting to `http://localhost:8000` if unset.

## Environment Variables

| Variable | Component | Default | Purpose |
|---|---|---|---|
| `ALLOWED_ORIGINS` | Backend, runtime | `http://localhost:5173` | Comma-separated list of origins allowed by CORS. Set to the deployed frontend's URL in production. |
| `RETRIEVAL_MODE` | Backend, runtime | `hybrid` | `tfidf` \| `dense` \| `hybrid`. Only `tfidf` is implemented; `dense`/`hybrid` currently fall back to it. |
| `USE_RERANKER` | Backend, runtime | `true` | Enables the cross-encoder reranking stage. When `false`, the reranker model is never loaded. |
| `VITE_API_URL` | Frontend, build-time | `http://localhost:8000` | Base URL of the backend the frontend calls. Vite inlines this at build time — changing it requires a rebuild/redeploy, not just a settings change. |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS` | Backend, runtime | `1` | Caps CPU thread-pool parallelism for `onnxruntime`/BLAS so multiple concurrent requests don't oversubscribe a memory-constrained container. Set via `os.environ.setdefault`, so still overridable. |
| `TOKENIZERS_PARALLELISM` | Backend, runtime | `false` | Silences the HuggingFace `tokenizers` fork-safety warning. |

## Deployment

### Backend — Render

There is no `render.yaml` in this repo; configure the service through the Render dashboard:

- **Environment**: Docker
- **Root Directory**: `backend`
- **Dockerfile Path**: `backend/Dockerfile`
- **Environment variables**: `ALLOWED_ORIGINS` (required — set to the frontend's URL);
  `RETRIEVAL_MODE` / `USE_RERANKER` if overriding the defaults.

The Dockerfile installs pinned dependencies from `requirements.txt`, runs `preload.py` at
build time to bake the quantized ONNX models and nltk data into the image, and starts the
server with a single `uvicorn` worker (`CMD ["uvicorn", "main:app", "--host", "0.0.0.0",
"--port", "8000", "--workers", "1"]`) — one worker because the pipeline is CPU-bound and
memory-constrained (512MB ceiling); additional workers would each load a separate copy of
both models.

### Frontend — Vercel

- Root directory: `frontend`
- Framework preset: Vite (build command `vite build`, output directory `dist`)
- Environment variable: `VITE_API_URL` set to the deployed Render backend's URL, for the
  Production (and Preview, if used) environment. Because Vite inlines env vars at build time,
  setting or changing this requires a new deploy to take effect.

## Usage

```bash
curl -X POST https://<your-backend>.onrender.com/ask \
  -F "file=@paper.pdf" \
  -F "question=What method does this paper use?"
```

Response (`AskResponse`, see `backend/schemas.py`):

```json
{
  "answer": "six",
  "context": "The encoder consists of six identical layers each with two sub-layers. ...",
  "char_start": 22,
  "char_end": 25,
  "confidence": 0.5876,
  "found": true,
  "retrieval_confidence": 1.0,
  "page_number": 0,
  "error": null
}
```

`GET /health` returns `{"status": "ok", "cached_docs": <n>}`.

## Testing

These are standalone scripts, not a `pytest` suite — each is run directly and takes a PDF
path as an argument. `smoke_test.py` defaults to `backend/sample.pdf` if the argument is
omitted; the others require it.

```bash
cd backend

# Compare layout-aware (PyMuPDF) vs pypdf extraction output for a given PDF.
python test_extraction_comparison.py sample.pdf

# Run the full pipeline for a specific question and print the resolved PDF page,
# to check against the page you've confirmed manually.
python test_page_grounding.py sample.pdf "How many layers does the encoder have?"

# Run one question through every combination of layout-aware/pypdf extraction,
# retrieval mode, and reranker on/off (12 combinations), checking none of them crash.
python smoke_test.py sample.pdf
```

Other scripts in `backend/`, each with usage documented in its own docstring:
`test_qa.py` (assert-based checks on caching, exact-offset extraction, and null-answer
rejection), `test_adaptive_cutoff.py`, `test_token_budget.py`, `test_concurrency.py`
(concurrent requests against a running server), `eval.py` (retrieval/reranker configuration
sweep with recall metrics), and `docker_memory_test.py <pdf> [--retrieval-mode ...]
[--use-reranker ...]` (builds the Docker image and confirms it stays under a 512MB container
memory limit).

## License

MIT — see [LICENSE](LICENSE).
