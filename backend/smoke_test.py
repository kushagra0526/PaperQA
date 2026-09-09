#!/usr/bin/env python3
"""
smoke_test.py — end-to-end smoke test across all flag combinations.

Usage:
    python smoke_test.py [path/to/paper.pdf]

Runs 12 combinations of:
    USE_LAYOUT_AWARE_EXTRACTION  True / False          (2)
    RETRIEVAL_MODE               tfidf / dense / hybrid (3)
    USE_RERANKER                 True / False          (2)
                                                  total 12

For each combination, monkeypatches the relevant module flags, runs the full
pipeline, and prints either:
    OK   layout=<x> mode=<y> rerank=<z> -> <first 40 chars of answer>
or:
    FAIL layout=<x> mode=<y> rerank=<z> -> ExcType: message
"""

import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Accept PDF path from argv; fall back to sample.pdf in the same directory.
pdf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sample.pdf"
)

if not os.path.isfile(pdf_path):
    print(f"Usage: python smoke_test.py <path/to/paper.pdf>")
    print(f"File not found: {pdf_path}")
    sys.exit(1)

import extraction
import retrieval
from qa import get_answer

QUESTION = "What method does this paper use?"

LAYOUT_FLAGS   = [True, False]
RETRIEVAL_MODES = ["tfidf", "dense", "hybrid"]
RERANKER_FLAGS  = [True, False]

print(f"Smoke-testing with PDF: {pdf_path}")
print(f"Question: {QUESTION!r}")
print("-" * 70)

passed = failed = 0

with open(pdf_path, "rb") as fh:
    pdf_bytes = fh.read()

for layout in LAYOUT_FLAGS:
    for mode in RETRIEVAL_MODES:
        for rerank in RERANKER_FLAGS:
            label = f"layout={str(layout)[0]} mode={mode:<6} rerank={str(rerank)[0]}"
            try:
                # Patch flags on their owning modules.
                extraction.USE_LAYOUT_AWARE_EXTRACTION = layout
                retrieval.RETRIEVAL_MODE = mode
                retrieval.USE_RERANKER   = rerank

                # Clear the doc cache so each run actually re-extracts.
                extraction.DOC_CACHE.clear()

                doc_data = extraction.extract_and_chunk_pdf(pdf_bytes)
                ctx_result = retrieval.select_context_with_meta(doc_data, QUESTION)
                context = ctx_result["context"]

                if not ctx_result["found"]:
                    snippet = "(no passage found)"
                else:
                    qa_res  = get_answer(QUESTION, context)
                    answer  = qa_res.get("answer", "")
                    snippet = (answer[:40] + "…") if len(answer) > 40 else (answer or "(empty)")

                print(f"OK   {label} -> {snippet}")
                passed += 1

            except Exception as exc:
                print(f"FAIL {label} -> {type(exc).__name__}: {exc}")
                failed += 1
            finally:
                # Restore defaults after each run.
                extraction.USE_LAYOUT_AWARE_EXTRACTION = True
                retrieval.RETRIEVAL_MODE = "tfidf"
                retrieval.USE_RERANKER   = False
                extraction.DOC_CACHE.clear()

print("-" * 70)
print(f"Results: {passed} passed, {failed} failed out of 12 combinations.")
sys.exit(0 if failed == 0 else 1)
