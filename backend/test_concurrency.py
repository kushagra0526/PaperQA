#!/usr/bin/env python3
"""
test_concurrency.py — fire 5 simultaneous /ask requests to a running server.

Usage:
    python test_concurrency.py <path/to/paper.pdf> [http://localhost:8000]

Requires the server to already be running:
    uvicorn main:app --host 127.0.0.1 --port 8000

Sends 5 concurrent POST /ask requests (via ThreadPoolExecutor) each with a
different question but the same PDF, and prints each response's status code
and a snippet of the answer (or the exception if the request failed).  Used
to verify that the shared model-loading locks and idle-unload threads don't
cause data races or crashes under concurrent load.
"""

import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    print("The 'requests' library is required: pip install requests")
    sys.exit(1)

if len(sys.argv) < 2:
    print("Usage: python test_concurrency.py <path/to/paper.pdf> [server_url]")
    print("       server_url defaults to http://localhost:8000")
    sys.exit(1)

pdf_path   = sys.argv[1]
server_url = sys.argv[2].rstrip("/") if len(sys.argv) > 2 else "http://localhost:8000"

if not os.path.isfile(pdf_path):
    print(f"File not found: {pdf_path}")
    sys.exit(1)

QUESTIONS = [
    "What method does this paper use?",
    "What is the main contribution of this work?",
    "What datasets were used in the experiments?",
    "What are the limitations mentioned by the authors?",
    "What future work do the authors suggest?",
]

ASK_URL = f"{server_url}/ask"

with open(pdf_path, "rb") as fh:
    pdf_bytes = fh.read()

print(f"Server:  {ASK_URL}")
print(f"PDF:     {pdf_path}  ({len(pdf_bytes):,} bytes)")
print(f"Workers: {len(QUESTIONS)} concurrent requests")
print("-" * 70)


def send_request(question: str) -> dict:
    """Send one POST /ask and return a result summary dict."""
    try:
        resp = requests.post(
            ASK_URL,
            files={"file": ("paper.pdf", pdf_bytes, "application/pdf")},
            data={"question": question},
            timeout=120,
        )
        body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        return {
            "question": question,
            "status":   resp.status_code,
            "found":    body.get("found"),
            "answer":   (body.get("answer") or "")[:60],
            "confidence": body.get("confidence"),
            "page_number": body.get("page_number"),
            "error":    body.get("error"),
            "exc":      None,
        }
    except Exception as exc:
        return {
            "question": question,
            "status":   None,
            "found":    None,
            "answer":   "",
            "confidence": None,
            "page_number": None,
            "error":    None,
            "exc":      f"{type(exc).__name__}: {exc}",
        }


with ThreadPoolExecutor(max_workers=len(QUESTIONS)) as pool:
    futures = {pool.submit(send_request, q): q for q in QUESTIONS}
    results = []
    for future in as_completed(futures):
        results.append(future.result())

# Print in original question order for consistent output.
results.sort(key=lambda r: QUESTIONS.index(r["question"]))

all_ok = True
for i, r in enumerate(results, 1):
    if r["exc"]:
        print(f"[{i}] EXCEPTION  Q: {r['question'][:50]!r}")
        print(f"     -> {r['exc']}")
        all_ok = False
    elif r["status"] != 200:
        print(f"[{i}] HTTP {r['status']}  Q: {r['question'][:50]!r}")
        if r["error"]:
            print(f"     error: {r['error']}")
        all_ok = False
    else:
        ans_snippet = (r["answer"] + "…") if len(r["answer"]) == 60 else r["answer"]
        print(
            f"[{i}] HTTP {r['status']}  found={r['found']}  "
            f"conf={r['confidence']}  page={r['page_number']}"
        )
        print(f"     Q: {r['question'][:55]!r}")
        print(f"     A: {ans_snippet!r}")

print("-" * 70)
print(f"{'All requests succeeded.' if all_ok else 'One or more requests failed.'}")
sys.exit(0 if all_ok else 1)
