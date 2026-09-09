#!/usr/bin/env python3
"""
test_page_grounding.py — verify page_number resolution against a real PDF.

Usage:
    python test_page_grounding.py <path/to/paper.pdf> "<question>"

Runs the full pipeline and prints the extracted answer together with the
resolved page_number so you can compare against the page you've manually
confirmed in the PDF.  Page numbers are 0-indexed (page 0 = first page).
"""

import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

if len(sys.argv) < 3:
    print("Usage: python test_page_grounding.py <path/to/paper.pdf> \"<question>\"")
    sys.exit(1)

pdf_path = sys.argv[1]
question = sys.argv[2]

if not os.path.isfile(pdf_path):
    print(f"File not found: {pdf_path}")
    sys.exit(1)

import extraction
import retrieval
from qa import get_answer

with open(pdf_path, "rb") as fh:
    pdf_bytes = fh.read()

doc_data  = extraction.extract_and_chunk_pdf(pdf_bytes)
ret       = retrieval.select_context_with_meta(doc_data, question)

print(f"PDF:      {pdf_path}")
print(f"Question: {question}")
print("-" * 60)

if not ret["found"]:
    print("retrieval found: False")
    print("retrieval_confidence:", ret["retrieval_confidence"])
    print("No passage cleared the relevance threshold — answer not attempted.")
    sys.exit(0)

context = ret["context"]
qa_res  = get_answer(question, context)

page_number, _sent_idx = retrieval.resolve_page_number(
    char_start          = qa_res.get("char_start", -1),
    window_offsets      = ret["window_offsets"],
    window_sent_offsets = ret["window_sent_offsets"],
    sentence_meta       = doc_data.get("sentence_meta", []),
)

print(f"Answer:              {qa_res.get('answer', '')!r}")
print(f"Confidence:          {qa_res.get('confidence', 0.0):.4f}")
print(f"found:               {qa_res.get('found', False)}")
print(f"page_number:         {page_number}  (0-indexed; None = could not resolve)")
print(f"retrieval_confidence:{ret['retrieval_confidence']:.4f}")
print()
print("Context snippet (first 200 chars):")
print(context[:200])
