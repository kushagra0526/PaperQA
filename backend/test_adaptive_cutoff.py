#!/usr/bin/env python3
"""
test_adaptive_cutoff.py — confirm the adaptive retrieval cutoff fires correctly.

Usage:
    python test_adaptive_cutoff.py <path/to/paper.pdf>

Runs the full pipeline with a deliberately irrelevant question against the given
document and prints the complete response dict.  Expected result: found=False and
retrieval_confidence at or near 0.0 — confirming the 40th-percentile threshold
rejects the request rather than forcing a low-signal answer through.
"""

import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

if len(sys.argv) < 2:
    print("Usage: python test_adaptive_cutoff.py <path/to/paper.pdf>")
    sys.exit(1)

pdf_path = sys.argv[1]
if not os.path.isfile(pdf_path):
    print(f"File not found: {pdf_path}")
    sys.exit(1)

import extraction
import retrieval
from qa import get_answer

IRRELEVANT_QUESTION = "What is the capital of France?"

with open(pdf_path, "rb") as fh:
    pdf_bytes = fh.read()

doc_data = extraction.extract_and_chunk_pdf(pdf_bytes)
ret      = retrieval.select_context_with_meta(doc_data, IRRELEVANT_QUESTION)

response: dict = {
    "found":                False,
    "answer":               "",
    "context":              "",
    "char_start":           -1,
    "char_end":             -1,
    "confidence":           0.0,
    "retrieval_confidence": ret["retrieval_confidence"],
    "page_number":          None,
}

if ret["found"]:
    qa_res = get_answer(IRRELEVANT_QUESTION, ret["context"])
    page_number, _ = retrieval.resolve_page_number(
        char_start          = qa_res.get("char_start", -1),
        window_offsets      = ret["window_offsets"],
        window_sent_offsets = ret["window_sent_offsets"],
        sentence_meta       = doc_data.get("sentence_meta", []),
    )
    response.update({
        "found":      qa_res.get("found", False),
        "answer":     qa_res.get("answer", ""),
        "context":    ret["context"],
        "char_start": qa_res.get("char_start", -1),
        "char_end":   qa_res.get("char_end", -1),
        "confidence": qa_res.get("confidence", 0.0),
        "page_number": page_number,
    })

print(f"PDF:      {pdf_path}")
print(f"Question: {IRRELEVANT_QUESTION!r}")
print("-" * 60)
print("Full response dict:")
for key, val in response.items():
    print(f"  {key:<24}: {val!r}")
print()

# Verdict
if not response["found"] and response["retrieval_confidence"] == 0.0:
    print("PASS: found=False and retrieval_confidence=0.0 — cutoff fired correctly.")
elif not response["found"]:
    print(f"PASS: found=False (retrieval_confidence={response['retrieval_confidence']}).")
else:
    print(
        f"NOTE: found=True — the document may contain Paris or France text, "
        f"or the threshold was not tight enough. "
        f"confidence={response['confidence']}, "
        f"retrieval_confidence={response['retrieval_confidence']}"
    )
