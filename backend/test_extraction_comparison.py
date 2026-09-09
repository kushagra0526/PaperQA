#!/usr/bin/env python3
"""
test_extraction_comparison.py — compare layout-aware vs pypdf extraction.

Usage:
    python test_extraction_comparison.py <path/to/paper.pdf>

Runs extract_and_chunk_pdf twice — once with USE_LAYOUT_AWARE_EXTRACTION=True
(PyMuPDF word-level, two-column-aware) and once with False (pypdf fallback) —
and prints the first 800 characters of each result so you can visually check
whether the layout-aware version reads as coherent sentence order on a
two-column PDF, versus the old pypdf version which may show column-interleaving.
"""

import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

if len(sys.argv) < 2:
    print("Usage: python test_extraction_comparison.py <path/to/paper.pdf>")
    sys.exit(1)

pdf_path = sys.argv[1]
if not os.path.isfile(pdf_path):
    print(f"File not found: {pdf_path}")
    sys.exit(1)

import extraction

PREVIEW_CHARS = 800

with open(pdf_path, "rb") as fh:
    pdf_bytes = fh.read()

for layout, label in [(True, "LAYOUT-AWARE (PyMuPDF)"), (False, "ORIGINAL (pypdf)")]:
    extraction.USE_LAYOUT_AWARE_EXTRACTION = layout
    extraction.DOC_CACHE.clear()

    doc_data = extraction.extract_and_chunk_pdf(pdf_bytes)
    text     = doc_data.get("text", "")
    n_sents  = len(doc_data.get("sentences", []))

    print("=" * 70)
    print(f"  {label}")
    print(f"  Total characters: {len(text):,}   Sentences extracted: {n_sents}")
    print("-" * 70)
    print(text[:PREVIEW_CHARS])
    if len(text) > PREVIEW_CHARS:
        print(f"  … (truncated; {len(text) - PREVIEW_CHARS:,} more characters)")
    print()

# Restore default.
extraction.USE_LAYOUT_AWARE_EXTRACTION = True
extraction.DOC_CACHE.clear()
