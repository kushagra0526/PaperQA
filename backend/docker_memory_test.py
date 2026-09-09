#!/usr/bin/env python3
"""
docker_memory_test.py — automated memory-limit test of the Dockerized backend.

Usage:
    python docker_memory_test.py <path/to/paper.pdf>
                                  [--retrieval-mode tfidf|dense|hybrid]
                                  [--use-reranker true|false]

Requires Docker Desktop to be running.  Run from inside backend/.

--retrieval-mode / --use-reranker are forwarded to the container as
RETRIEVAL_MODE / USE_RERANKER environment variables (-e flags on
`docker run`), matching the overrides main.py reads at startup — so testing
a different config no longer requires hand-writing a `docker run` command.
Omit either flag to use the image's built-in defaults.

Steps:
  1. docker build -t paperqa-backend .
  2. docker run -d --rm -p 8000:8000 --memory=512m --memory-swap=512m
     [-e RETRIEVAL_MODE=... ] [-e USE_RERANKER=... ]
  3. Poll /health until 200 or 60s timeout
  4. Background thread polls docker stats every 1s
  5. POST /ask with the given PDF
  6. Stop stats thread; print memory timeline + peak
  7. docker inspect --format "{{.State.OOMKilled}}"
  8. docker stop paperqa-memtest  (always, in finally)
  9. Print PASS/FAIL verdict block
"""

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime

try:
    import requests as _requests
except ImportError:
    print("The 'requests' library is required: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(
    description="Automated Docker memory-limit test of the PaperQA backend."
)
_parser.add_argument("pdf_path", help="Path to a PDF to send in the /ask request.")
_parser.add_argument(
    "--retrieval-mode",
    choices=["tfidf", "dense", "hybrid"],
    default=None,
    help="Overrides RETRIEVAL_MODE inside the container (default: image default).",
)
_parser.add_argument(
    "--use-reranker",
    choices=["true", "false"],
    default=None,
    help="Overrides USE_RERANKER inside the container (default: image default).",
)
_args = _parser.parse_args()

pdf_path = _args.pdf_path
if not os.path.isfile(pdf_path):
    print(f"File not found: {pdf_path}")
    sys.exit(1)

# -e flags to append to `docker run`, built from whichever overrides were
# actually passed — leaving both unset preserves the image's own defaults.
_env_overrides: list[str] = []
if _args.retrieval_mode is not None:
    _env_overrides += ["-e", f"RETRIEVAL_MODE={_args.retrieval_mode}"]
if _args.use_reranker is not None:
    _env_overrides += ["-e", f"USE_RERANKER={_args.use_reranker}"]

BACKEND_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGE_NAME    = "paperqa-backend"
CONTAINER_NAME = "paperqa-memtest"
ASK_URL       = "http://localhost:8000/ask"
HEALTH_URL    = "http://localhost:8000/health"
MEMORY_LIMIT  = "512m"
QUESTION      = "What method does this paper use?"

# ---------------------------------------------------------------------------
# Helper: run a subprocess and stream its output live
# ---------------------------------------------------------------------------

def stream_run(cmd: list[str], cwd: str | None = None) -> int:
    """Run *cmd*, stream stdout+stderr live to console, return exit code."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=cwd, text=True, bufsize=1,
        encoding="utf-8", errors="replace",
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    return proc.returncode


def capture_run(cmd: list[str]) -> tuple[int, str]:
    """Run *cmd* silently, return (returncode, combined output)."""
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    return result.returncode, (result.stdout + result.stderr).strip()


# ---------------------------------------------------------------------------
# Memory stats polling (background thread)
# ---------------------------------------------------------------------------

class StatsPoller(threading.Thread):
    """
    Polls `docker stats <name> --no-stream` every second and records
    (timestamp, raw_usage_string, bytes_used) into self.readings.
    """
    def __init__(self, container_name: str):
        super().__init__(daemon=True)
        self.container_name = container_name
        self.readings: list[dict] = []   # {ts, raw, bytes}
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    @staticmethod
    def _parse_bytes(s: str) -> float:
        """
        Convert a docker mem string like "234.5MiB" or "512MiB" to bytes.
        Returns 0.0 if parsing fails.
        """
        s = s.strip()
        try:
            m = re.match(r"([\d.]+)\s*([KMGTPEi]*B)", s, re.IGNORECASE)
            if not m:
                return 0.0
            value, unit = float(m.group(1)), m.group(2).upper()
            multipliers = {
                "B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9,
                "KIB": 1024, "MIB": 1024**2, "GIB": 1024**3,
            }
            return value * multipliers.get(unit, 1)
        except Exception:
            return 0.0

    def run(self) -> None:
        while not self._stop_event.is_set():
            rc, out = capture_run([
                "docker", "stats", self.container_name,
                "--no-stream", "--format", "{{.MemUsage}}",
            ])
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            if rc == 0 and out:
                # MemUsage is "used / limit" — take the used half
                used_str = out.split("/")[0].strip()
                used_bytes = self._parse_bytes(used_str)
                self.readings.append({
                    "ts":    ts,
                    "raw":   out,
                    "bytes": used_bytes,
                })
            time.sleep(1)

    def peak(self) -> tuple[float, str]:
        """Return (peak_bytes, raw_string_at_peak)."""
        if not self.readings:
            return 0.0, "n/a"
        best = max(self.readings, key=lambda r: r["bytes"])
        return best["bytes"], best["raw"].split("/")[0].strip()


# ---------------------------------------------------------------------------
# Main — wrapped in try/finally for guaranteed cleanup
# ---------------------------------------------------------------------------

container_running = False
ask_response      = None
ask_duration_s    = None
ask_error         = None
oom_killed        = None
poller            = None

print("=" * 65)
print("  PaperQA Docker Memory Test")
print(f"  PDF:    {pdf_path}")
print(f"  Limit:  {MEMORY_LIMIT}  |  Image: {IMAGE_NAME}")
print(f"  RETRIEVAL_MODE override: {_args.retrieval_mode or '(image default)'}")
print(f"  USE_RERANKER override:   {_args.use_reranker or '(image default)'}")
print("=" * 65)

try:
    # ── Step 1: Build ────────────────────────────────────────────────────────
    print("\n[1/8] Building Docker image …")
    rc = stream_run(
        ["docker", "build", "-t", IMAGE_NAME, "."],
        cwd=BACKEND_DIR,
    )
    if rc != 0:
        print(f"\nBUILD FAILED (exit code {rc})")
        sys.exit(1)
    print("Build succeeded.")

    # ── Step 2: Start container ──────────────────────────────────────────────
    print(f"\n[2/8] Starting container with --memory={MEMORY_LIMIT} …")
    rc, container_id = capture_run([
        "docker", "run", "-d", "--rm",
        "-p", "8000:8000",
        f"--memory={MEMORY_LIMIT}",
        f"--memory-swap={MEMORY_LIMIT}",
        *_env_overrides,
        "--name", CONTAINER_NAME,
        IMAGE_NAME,
    ])
    if rc != 0:
        print(f"Failed to start container:\n{container_id}")
        sys.exit(1)
    container_running = True
    print(f"Container ID: {container_id[:12]}")

    # ── Step 3: Wait for /health ─────────────────────────────────────────────
    print(f"\n[3/8] Waiting for {HEALTH_URL} …")
    deadline = time.time() + 60
    healthy  = False
    while time.time() < deadline:
        try:
            r = _requests.get(HEALTH_URL, timeout=3)
            if r.status_code == 200:
                healthy = True
                print(f"  /health returned 200 after {60 - (deadline - time.time()):.0f}s")
                break
        except Exception:
            pass
        time.sleep(2)
        print("  … waiting", flush=True)

    if not healthy:
        print("CONTAINER FAILED TO START — skipping request test.")
        # Jump to cleanup via finally; verdict printed there.
    else:
        # ── Step 4: Start memory stats poller ────────────────────────────────
        print("\n[4/8] Starting background memory stats poller …")
        poller = StatsPoller(CONTAINER_NAME)
        poller.start()

        # ── Step 5: POST /ask ─────────────────────────────────────────────────
        print(f"\n[5/8] Sending POST {ASK_URL} …")
        print(f"  question: {QUESTION!r}")
        t0 = time.time()
        try:
            with open(pdf_path, "rb") as fh:
                pdf_bytes = fh.read()
            resp = _requests.post(
                ASK_URL,
                files={"file": ("paper.pdf", pdf_bytes, "application/pdf")},
                data={"question": QUESTION},
                timeout=120,
            )
            ask_duration_s = time.time() - t0
            ask_response   = resp.json() if resp.headers.get(
                "content-type", ""
            ).startswith("application/json") else {"raw": resp.text}
            print(f"  Status:   {resp.status_code}")
            print(f"  Duration: {ask_duration_s:.2f}s")
        except Exception as exc:
            ask_duration_s = time.time() - t0
            ask_error      = f"{type(exc).__name__}: {exc}"
            print(f"  Request failed: {ask_error}")

        # ── Step 6: Stop poller and print timeline ────────────────────────────
        print("\n[6/8] Stopping stats poller …")
        poller.stop()
        poller.join(timeout=3)

        if poller.readings:
            print(f"\n  Memory timeline ({len(poller.readings)} readings):")
            peak_bytes, peak_raw = poller.peak()
            for r in poller.readings:
                marker = " ◀ PEAK" if r["bytes"] == peak_bytes and r["raw"].split("/")[0].strip() == peak_raw else ""
                print(f"    {r['ts']}  {r['raw']}{marker}")
        else:
            print("  No stats readings recorded (container may have exited early).")
            peak_bytes, peak_raw = 0.0, "n/a"

    # ── Step 7: OOM-kill check ────────────────────────────────────────────────
    print("\n[7/8] Checking OOMKilled status …")
    rc, out = capture_run([
        "docker", "inspect", CONTAINER_NAME,
        "--format", "{{.State.OOMKilled}}",
    ])
    if rc == 0:
        oom_killed = out.strip().lower() == "true"
        print(f"  OOMKilled: {oom_killed}")
    else:
        # inspect failed — container may already be gone due to OOM + --rm
        oom_killed = None
        print(
            "  WARNING: docker inspect failed — container may have been "
            "removed by OOM killer (--rm flag removes it immediately).\n"
            "  Treat this as a possible OOM signal."
        )

finally:
    # ── Step 8: Cleanup ───────────────────────────────────────────────────────
    print("\n[8/8] Cleaning up …")
    if container_running:
        _rc, _out = capture_run(["docker", "stop", CONTAINER_NAME])
        if _rc == 0:
            print("  Container stopped.")
        else:
            # --rm may have already removed it; that's fine
            print(f"  docker stop returned {_rc} (container may already be gone).")

# ---------------------------------------------------------------------------
# Step 9: Final verdict
# ---------------------------------------------------------------------------
print()
print("=" * 65)
print("  FINAL VERDICT")
print("=" * 65)

peak_bytes = 0.0
peak_raw   = "n/a"
if poller and poller.readings:
    peak_bytes, peak_raw = poller.peak()

peak_mb      = peak_bytes / (1024 ** 2)
limit_mb     = 512.0
within_limit = peak_bytes > 0 and peak_mb < limit_mb
request_ok   = ask_response is not None and ask_error is None
oom_str      = (
    "YES — OOM kill detected!" if oom_killed
    else "possibly (inspect unavailable)" if oom_killed is None
    else "No"
)

overall_pass = healthy and request_ok and not oom_killed and (peak_bytes == 0 or within_limit)

print(f"  Result:           {'PASS ✓' if overall_pass else 'FAIL ✗'}")
print(f"  Container healthy: {healthy if 'healthy' in dir() else False}")
print(f"  Peak memory:       {peak_raw}  ({peak_mb:.1f} MB)")
print(f"  Under 512 MB:      {within_limit if peak_bytes > 0 else 'no data'}")
print(f"  OOMKilled:         {oom_str}")
print(f"  Request duration:  {f'{ask_duration_s:.2f}s' if ask_duration_s else 'n/a'}")

if ask_error:
    print(f"  /ask error:        {ask_error}")
elif ask_response:
    print(f"  /ask found:        {ask_response.get('found')}")
    ans = ask_response.get('answer', '')
    print(f"  /ask answer:       {(ans[:60] + '…') if len(ans) > 60 else ans!r}")
    print(f"  /ask confidence:   {ask_response.get('confidence')}")
    print(f"  /ask page_number:  {ask_response.get('page_number')}")
else:
    print("  /ask response:     not attempted")

print("=" * 65)
sys.exit(0 if overall_pass else 1)
