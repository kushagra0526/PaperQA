#!/usr/bin/env python3
"""
test_token_budget.py — measure actual token count of the assembled context.

Usage:
    python test_token_budget.py <path/to/paper.pdf>

Sets RETRIEVAL_MODE="hybrid" and USE_RERANKER=True, runs extract_and_chunk_pdf
and select_context_with_meta for a summarisation-style question, then encodes
the resulting context with the QA tokenizer and prints the token count alongside
a clear UNDER 512 BUDGET: True/False verdict.
"""

import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

if len(sys.argv) < 2:
    print("Usage: python test_token_budget.py <path/to/paper.pdf>")
    sys.exit(1)

pdf_path = sys.argv[1]
if not os.path.isfile(pdf_path):
    print(f"File not found: {pdf_path}")
    sys.exit(1)

import extraction
import retrieval
from qa import get_model

QUESTION   = "Summarize the main contribution"
TOKEN_HARD_LIMIT = 512

# Patch flags for the worst-case scenario (hybrid + reranker).
retrieval.RETRIEVAL_MODE = "hybrid"
retrieval.USE_RERANKER   = True

with open(pdf_path, "rb") as fh:
    pdf_bytes = fh.read()

doc_data = extraction.extract_and_chunk_pdf(pdf_bytes)
ret      = retrieval.select_context_with_meta(doc_data, QUESTION)

context  = ret["context"]
tok, _   = get_model()
enc      = tok.encode(QUESTION, context)
n_tokens = len(enc.ids)
under    = n_tokens < TOKEN_HARD_LIMIT

print(f"PDF:              {pdf_path}")
print(f"Question:         {QUESTION!r}")
print(f"RETRIEVAL_MODE:   {retrieval.RETRIEVAL_MODE}")
print(f"USE_RERANKER:     {retrieval.USE_RERANKER}")
print(f"retrieval found:  {ret['found']}")
print(f"Context length:   {len(context):,} chars")
print(f"Token count:      {n_tokens}")
print(f"UNDER 512 BUDGET: {under}")

if not under:
    print(f"\nWARNING: token count {n_tokens} exceeds 512 — tail will be truncated.")

# Restore defaults.
retrieval.RETRIEVAL_MODE = "tfidf"
retrieval.USE_RERANKER   = False
