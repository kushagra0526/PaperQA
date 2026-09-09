"""
FastAPI application entry point.

This module only wires routes and middleware.  All pipeline logic lives in:
  extraction.py — PDF parsing, sentence chunking, window construction
  retrieval.py  — TF-IDF retrieval, reranking, page-number resolution
  qa.py         — model loading, extractive QA inference
  schemas.py    — Pydantic request/response models

Environment variables:
  ALLOWED_ORIGINS  Comma-separated list of origins for CORS.
                   Defaults to "http://localhost:5173" when unset.
"""

from __future__ import annotations

import os

# Cap CPU parallelism before any heavy imports so thread pools pick up the
# limits from the start.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from extraction import DOC_CACHE, extract_and_chunk_pdf
from qa import get_answer
from retrieval import resolve_page_number, select_context_with_meta
from schemas import AskResponse, HealthResponse

# ---------------------------------------------------------------------------
# CORS — driven by environment variable so it can be tightened per-deployment
# without a code change.
# ---------------------------------------------------------------------------
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173")
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app = FastAPI(title="Scientific Paper QA")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def home() -> dict:
    return {
        "status":    "online",
        "service":   "PaperQA API",
        "docs":      "/docs",
        "endpoints": {
            "ask":    "POST /ask (multipart/form-data: file, question)",
            "health": "GET /health",
        },
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", cached_docs=len(DOC_CACHE))


_EMPTY_RESPONSE = AskResponse(
    answer="", context="", char_start=-1, char_end=-1,
    confidence=0.0, found=False, retrieval_confidence=0.0,
)


@app.post("/",    response_model=AskResponse)
@app.post("/ask", response_model=AskResponse)
def ask(
    file:     UploadFile = File(...),
    question: str        = Form(...),
) -> AskResponse:
    try:
        if not question or not question.strip():
            return _EMPTY_RESPONSE

        file.file.seek(0)
        file_bytes = file.file.read()
        if not file_bytes:
            return _EMPTY_RESPONSE

        if len(file_bytes) > 20_000_000:
            return AskResponse(
                answer="", context="", char_start=-1, char_end=-1,
                confidence=0.0, found=False, retrieval_confidence=0.0,
                error="File too large (max 20 MB).",
            )

        doc_data  = extract_and_chunk_pdf(file_bytes)
        retrieval = select_context_with_meta(doc_data, question)

        if not retrieval["found"]:
            return AskResponse(
                answer="", context="", char_start=-1, char_end=-1,
                confidence=0.0, found=False,
                retrieval_confidence=retrieval["retrieval_confidence"],
            )

        context = retrieval["context"]
        qa_res  = get_answer(question, context)

        page_number, _sentence_index = resolve_page_number(
            char_start          = qa_res.get("char_start", -1),
            window_offsets      = retrieval["window_offsets"],
            window_sent_offsets = retrieval["window_sent_offsets"],
            sentence_meta       = doc_data.get("sentence_meta", []),
        )

        return AskResponse(
            answer               = qa_res.get("answer", ""),
            context              = context,
            char_start           = qa_res.get("char_start", -1),
            char_end             = qa_res.get("char_end", -1),
            confidence           = qa_res.get("confidence", 0.0),
            found                = qa_res.get("found", False),
            retrieval_confidence = retrieval["retrieval_confidence"],
            page_number          = page_number,
        )

    except Exception as exc:
        return AskResponse(
            answer="",
            context=f"Error processing document: {exc}",
            char_start=-1, char_end=-1,
            confidence=0.0, found=False,
            retrieval_confidence=0.0,
            error=str(exc),
        )
