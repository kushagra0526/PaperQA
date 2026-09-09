"""
PDF extraction, sentence chunking, and window construction.

Public API used by the rest of the pipeline:
  extract_and_chunk_pdf(file_bytes)  -> doc_data dict
  build_windows(sentences)           -> list[str]
  build_windows_with_offsets(sents)  -> list[(str, [(int,int,int)])]
  DOC_CACHE                          -> the live cache dict (for test teardown)

NLTK punkt data is downloaded once during the Docker build step by preload.py
into backend/nltk_data/, which is baked into the image.  The runtime container
never needs network access to NLTK's servers.
"""

from __future__ import annotations

import bisect
import hashlib
import os
import re
import threading
from io import BytesIO

import fitz          # PyMuPDF
import nltk
import numpy as np   # used by caller modules; re-exported for convenience
from pypdf import PdfReader

# Point nltk at the data directory baked into the image by preload.py.
# This must happen before any nltk.data.find() or sent_tokenize() call.
_NLTK_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
if _NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, _NLTK_DATA_DIR)

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

# When True, extract_and_chunk_pdf uses the layout-aware PyMuPDF path
# (extract_text_layout_aware) which handles two-column papers correctly.
# Set to False to fall back to the original pypdf-based extraction.
USE_LAYOUT_AWARE_EXTRACTION: bool = True

# ---------------------------------------------------------------------------
# Document cache
# ---------------------------------------------------------------------------

DOC_CACHE: dict = {}

# Lock protecting all mutations to DOC_CACHE (eviction + insertion) because
# dict.pop() and key assignment are not atomic across threads in all Python
# builds.
_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Two-column gutter detection
# ---------------------------------------------------------------------------

# A gutter must be wider than ordinary inter-word spacing by at least this
# multiple of the page's own median word gap, and at least this many points
# absolute — both guards matter: the multiple alone would false-positive on
# pages with almost no spacing information, and the absolute floor alone
# would false-positive on pages set in an unusually loose font.
_GUTTER_MEDIAN_MULTIPLE = 4.0
_GUTTER_MIN_ABSOLUTE_PT = 12.0

# A candidate gutter location must recur at a consistent x-position across
# at least this fraction of lines that have enough words to reveal a gap at
# all — a single line's coincidental wide space (e.g. before an en-dash, or
# a short line at a paragraph end) must not be enough on its own.
_GUTTER_MIN_LINE_FRACTION = 0.5
_GUTTER_TOLERANCE_PT = 20.0


def _detect_column_gutter(words: list[tuple]) -> float | None:
    """
    Decide whether *words* (one page's ``page.get_text("words")`` output)
    come from a genuine multi-column layout, and if so return the gutter's
    x-coordinate — the vertical band consistently free of word content that
    separates the columns.  Returns None when no such stable gutter exists,
    meaning the page should be treated as an ordinary single column.

    This replaces a purely aggregate check ("some words are left of the
    page's horizontal midpoint, some are right") which false-positives on
    almost any single-column page: a normal paragraph column with standard
    margins straddles the page's horizontal center on nearly every line, so
    roughly half its words fall on each side by construction — that is not
    evidence of two columns.

    A genuine gutter instead shows up as an unusually wide gap *within
    individual rows*, at the *same x-position across most rows* — a
    single-column page's line-wrapping point moves around from line to
    line, so it essentially never produces a gap at a consistent location.

    Rows are built from raw y-coordinates spanning the *whole page width*,
    not from PyMuPDF's own (block_no, line_no) tags: a genuine two-column
    layout is normally reported as two separate text blocks (one per
    column), so grouping by block first would put every column's words in
    their own line group and the two columns could never be compared
    against each other — there would be no gap to find, because no row
    would ever contain words from both columns to begin with.
    """
    if not words:
        return None

    # Cluster words into rows by y0 alone, tolerant of the small baseline
    # jitter within one visual line but tight enough not to merge adjacent
    # lines — calibrated from this page's own median word height so it
    # adapts to font size instead of assuming one.
    by_y = sorted(words, key=lambda w: w[1])
    heights = [w[3] - w[1] for w in by_y if w[3] > w[1]]
    heights.sort()
    row_tolerance = max((heights[len(heights) // 2] * 0.5) if heights else 3.0, 1.5)

    lines: list[list[tuple]] = []
    current: list[tuple] = [by_y[0]]
    row_anchor = by_y[0][1]
    for w in by_y[1:]:
        if abs(w[1] - row_anchor) <= row_tolerance:
            current.append(w)
        else:
            lines.append(current)
            current = [w]
            row_anchor = w[1]
    lines.append(current)

    # Calibrate "unusually wide" against this page's own typical inter-word
    # spacing rather than a fixed constant, so it adapts to font size.
    normal_gaps: list[float] = []
    for line_words in lines:
        ordered = sorted(line_words, key=lambda w: w[0])
        for a, b in zip(ordered, ordered[1:]):
            gap = b[0] - a[2]
            if gap > 0:
                normal_gaps.append(gap)
    if not normal_gaps:
        return None
    normal_gaps.sort()
    median_gap = normal_gaps[len(normal_gaps) // 2]
    gutter_threshold = max(median_gap * _GUTTER_MEDIAN_MULTIPLE, _GUTTER_MIN_ABSOLUTE_PT)

    # For each line with enough words to show an internal gap, record the
    # center of its single widest gap (if any) that clears the threshold.
    candidate_centers: list[float] = []
    eligible_lines = 0
    for line_words in lines:
        ordered = sorted(line_words, key=lambda w: w[0])
        if len(ordered) < 2:
            continue
        eligible_lines += 1
        widest: tuple[float, float, float] | None = None   # (start, end, size)
        for a, b in zip(ordered, ordered[1:]):
            gap_start, gap_end = a[2], b[0]
            gap = gap_end - gap_start
            if gap >= gutter_threshold and (widest is None or gap > widest[2]):
                widest = (gap_start, gap_end, gap)
        if widest is not None:
            candidate_centers.append((widest[0] + widest[1]) / 2.0)

    # Need enough multi-word lines to say anything meaningful, and at least
    # one candidate gap at all.
    if eligible_lines < 3 or not candidate_centers:
        return None

    # The gutter location must recur consistently, not just appear once.
    candidate_centers.sort()
    median_center = candidate_centers[len(candidate_centers) // 2]
    consistent = [
        c for c in candidate_centers if abs(c - median_center) <= _GUTTER_TOLERANCE_PT
    ]
    if len(consistent) / eligible_lines < _GUTTER_MIN_LINE_FRACTION:
        return None

    return sum(consistent) / len(consistent)


# ---------------------------------------------------------------------------
# Layout-aware extraction (PyMuPDF)
# ---------------------------------------------------------------------------

def extract_text_layout_aware(file_bytes: bytes) -> tuple[str, list[int], list[str]]:
    """
    Extract text from a PDF using PyMuPDF with layout-aware word ordering.

    For each page, word bounding boxes are used to detect two-column layout
    via _detect_column_gutter: a genuine, consistently-positioned gap between
    columns, not merely "some words left of center, some right" (see that
    function's docstring for why the naive aggregate check false-positives on
    ordinary single-column pages).  When a stable gutter is found, words are
    split at it and sorted left-column-first, then right.  Otherwise the page
    is single-column and words are sorted by (y0, x0) as normal reading order.

    Returns:
        raw_text        — all pages joined by spaces (matches pypdf format).
        word_page_index — flat list of page numbers, one entry per word in
                          the same order as the reconstructed text, used by
                          _build_sentence_meta to map sentences to pages.
        page_words      — flat list of the word strings in the same order as
                          word_page_index, used by _build_sentence_meta to
                          locate character offsets.  Computed in the same
                          single pass — the PDF is never opened twice.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception:
        return "", [], []

    page_texts: list[str] = []
    word_page_index: list[int] = []
    page_words: list[str] = []

    try:
        for page_no, page in enumerate(doc):
            if page_no >= 25:           # mirror the 25-page cap
                break

            words = page.get_text("words")   # (x0,y0,x1,y1,word,blk,ln,wn)
            if not words:
                continue

            gutter_x = _detect_column_gutter(words)

            if gutter_x is not None:
                left_words  = [w for w in words if w[0] < gutter_x]
                right_words = [w for w in words if w[0] >= gutter_x]
                ordered = (
                    sorted(left_words,  key=lambda w: (w[1], w[0]))
                    + sorted(right_words, key=lambda w: (w[1], w[0]))
                )
            else:
                ordered = sorted(words, key=lambda w: (w[1], w[0]))

            page_texts.append(" ".join(w[4] for w in ordered))
            word_page_index.extend([page_no] * len(ordered))
            page_words.extend(w[4] for w in ordered)
    finally:
        doc.close()

    return " ".join(page_texts), word_page_index, page_words


# ---------------------------------------------------------------------------
# Sentence-to-page metadata
# ---------------------------------------------------------------------------

def _build_sentence_meta(
    sentences: list[str],
    raw_text: str,
    word_page_index: list[int],
    page_words: list[str],
) -> list[dict]:
    """
    Align each sentence to a page number by finding the sentence's character
    offset in raw_text, then mapping it to the nearest word's page.

    page_words is the flat list of word strings in the same order as
    word_page_index (used to locate character positions).
    """
    if not word_page_index or not page_words:
        return [{"page": 0}] * len(sentences)

    # Build (char_start, page_no) for every word so we can scan forward.
    word_char_starts: list[int] = []
    pos = 0
    for word in page_words:
        idx = raw_text.find(word, pos)
        if idx == -1:
            idx = pos
        word_char_starts.append(idx)
        pos = idx + len(word)

    # word_char_starts is non-decreasing by construction (each word is found
    # at or after the previous word's position), so a per-sentence page
    # lookup can use bisect instead of a linear scan.
    #
    # sent_start is likewise located with a forward-moving cursor rather than
    # raw_text.find(sent) from position 0 every time: sentences occur in
    # document order, and searching from 0 each time both re-scans the same
    # prefix of raw_text repeatedly (O(sentences × len(raw_text)) worst case)
    # and can mis-attribute a sentence to an earlier duplicate occurrence
    # (repeated boilerplate/headers) instead of its real position.
    meta: list[dict] = []
    pos = 0
    for sent in sentences:
        sent_start = raw_text.find(sent, pos)
        if sent_start == -1:
            # Out-of-order relative to the cursor (shouldn't normally happen)
            # — fall back to a full-text search before giving up.
            sent_start = raw_text.find(sent)
        if sent_start == -1:
            meta.append({"page": 0})
            continue
        pos = sent_start + len(sent)

        word_idx = bisect.bisect_right(word_char_starts, sent_start) - 1
        page_no = word_page_index[word_idx] if word_idx >= 0 else 0
        meta.append({"page": page_no})
    return meta


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def _mark_references_section(
    sentences: list[str],
    sentence_meta: list[dict],
    raw_text: str = "",
) -> None:
    """
    Detect the start of a references/bibliography section and set
    ``"past_references": True`` on every sentence_meta entry from that
    point onward.  Mutates *sentence_meta* in-place; returns nothing.

    Two-pass detection — either is sufficient to trigger the filter:

    Pass 1 — sentence-level: a sentence that matches a canonical heading
    string case-insensitively AND is ≤ 40 characters is treated as the
    boundary.  This catches PDFs where the heading is its own paragraph.

    Pass 2 — raw-text-level: if pass 1 finds nothing, scan raw_text for
    the heading appearing as a word-boundary-delimited token (e.g. embedded
    as "...REFERENCES [1]...").  Map the character offset back to the first
    sentence that starts at or after that offset.

    Headings matched: "references", "bibliography", "works cited",
    "literature cited", "reference list".

    Fallback: if neither pass detects a heading, nothing is marked — avoids
    false positives on papers with unusual formatting.
    """
    import re as _re

    _HEADING_PATTERN = (
        r"(?:references|bibliography|works\s+cited|"
        r"literature\s+cited|reference\s+list)"
    )
    # Pass 1 — full-sentence heading (standalone line / short sentence)
    _SENT_RE = _re.compile(
        r"^\s*" + _HEADING_PATTERN + r"\s*[.:]*\s*$",
        _re.IGNORECASE,
    )
    boundary = None
    for i, sent in enumerate(sentences):
        if len(sent) <= 40 and _SENT_RE.match(sent):
            boundary = i
            break

    # Pass 2 — heading embedded inline in raw_text (e.g. "...REFERENCES [1]")
    # Only search the latter 40% of the document to avoid matching the word
    # "references" in journal title headers, in-body citations, or section
    # headings that appear early in the paper (e.g. "3. Related References").
    if boundary is None and raw_text:
        search_start = int(len(raw_text) * 0.60)
        tail_text    = raw_text[search_start:]
        # Require the heading to be preceded by whitespace/punctuation and
        # followed by either whitespace+digit (citation "[1]") or end-of-word,
        # to distinguish "REFERENCES" the section heading from "references" used
        # as a common noun mid-sentence.
        _TEXT_RE = _re.compile(
            r"(?:^|[\s.])(" + _HEADING_PATTERN + r")(?=\s*[\[\d\s]|$)",
            _re.IGNORECASE | _re.MULTILINE,
        )
        m = _TEXT_RE.search(tail_text)
        if m:
            heading_char = search_start + m.start(1)

            # Locate each sentence's position in raw_text once, with a
            # forward-moving cursor (sentences occur in document order), and
            # binary-search that table instead of re-running raw_text.find()
            # for every sentence against the full document — the previous
            # approach was O(sentences × len(raw_text)) in the common case
            # where most sentences don't match near heading_char at all.
            sentence_starts: list[int] = []
            cur = 0
            for sent in sentences:
                sp = raw_text.find(sent, cur)
                if sp == -1:
                    sp = raw_text.find(sent)
                if sp == -1:
                    sp = cur  # unknown position — keep the table monotonic
                sentence_starts.append(sp)
                cur = sp + len(sent)

            # Sentence whose occurrence is closest to (and at or after)
            # heading_char — mirrors the original "nearest match" intent,
            # now via bisect over the precomputed, in-order positions.
            idx = bisect.bisect_left(sentence_starts, heading_char)
            if idx < len(sentences):
                boundary = idx
            elif sentence_starts and (
                sentence_starts[-1] <= heading_char
                < sentence_starts[-1] + len(sentences[-1])
            ):
                # heading_char falls inside the last sentence (run-on
                # heading) — start filtering from the next one, i.e. none.
                boundary = len(sentences)

    if boundary is not None:
        for i in range(boundary, len(sentence_meta)):
            sentence_meta[i]["past_references"] = True


def extract_and_chunk_pdf(file_bytes: bytes) -> dict:
    """
    Parse *file_bytes* as a PDF, extract text (layout-aware or pypdf depending
    on USE_LAYOUT_AWARE_EXTRACTION), split into sentences via nltk, and return:

        {
            "text":          str,         # full raw text
            "sentences":     list[str],   # filtered sentences (len > 15)
            "sentence_meta": list[dict],  # [{"page": int}, ...]
        }

    Results are cached by SHA-256 hash of the raw bytes (LRU eviction at 3
    entries).  Repeated calls with the same bytes are free.
    """
    pdf_hash = hashlib.sha256(file_bytes).hexdigest()
    if pdf_hash in DOC_CACHE:
        return DOC_CACHE[pdf_hash]

    word_page_index: list[int] = []
    page_words: list[str] = []

    if USE_LAYOUT_AWARE_EXTRACTION:
        raw_text, word_page_index, page_words = extract_text_layout_aware(file_bytes)
    else:
        # Original pypdf path — unchanged.
        try:
            reader = PdfReader(BytesIO(file_bytes))
            pages = reader.pages[:25]
            raw_text = " ".join(page.extract_text() or "" for page in pages)
        except Exception:
            raw_text = ""

    # Sentence splitting: nltk sent_tokenize handles abbreviations (et al.,
    # Fig., e.g.,) and decimal numbers that the old regex mis-split.
    try:
        raw_sentences = nltk.sent_tokenize(raw_text) if raw_text.strip() else []
    except Exception:
        raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", raw_text)

    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 15]
    if not sentences:
        sentences = (
            [raw_text.strip()] if raw_text.strip() else ["No text found in document."]
        )

    if USE_LAYOUT_AWARE_EXTRACTION and word_page_index:
        sentence_meta = _build_sentence_meta(
            sentences, raw_text, word_page_index, page_words
        )
    else:
        sentence_meta = [{"page": 0}] * len(sentences)

    # Mark sentences that fall after the references/bibliography heading so
    # select_context can exclude them from the retrieval candidate pool.
    # This is purely additive — the sentences remain in doc_data["sentences"]
    # and sentence_meta for debugging/display; only retrieval skips them.
    _mark_references_section(sentences, sentence_meta, raw_text)

    doc_data = {"text": raw_text, "sentences": sentences, "sentence_meta": sentence_meta}

    with _cache_lock:
        if len(DOC_CACHE) >= 3:
            DOC_CACHE.pop(next(iter(DOC_CACHE)))
        DOC_CACHE[pdf_hash] = doc_data
    return doc_data


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------

def build_windows_with_offsets(
    sentences: list[str],
) -> list[tuple[str, list[tuple[int, int, int]]]]:
    """
    Group sentences into overlapping windows of 3 (step 2, overlap 1) and
    record the character offsets of each constituent sentence within its window.

    Returns a list of (window_text, sent_offsets) pairs where:
      window_text  — sentences joined with a single space
      sent_offsets — [(sentence_index, start_in_window, end_in_window), ...]

    Example for sentences[2..4]:
      window_text  = "Sent2. Sent3. Sent4."
      sent_offsets = [(2, 0, 6), (3, 7, 13), (4, 14, 20)]

    When fewer than 3 sentences are present each sentence becomes its own
    single-sentence window (mirrors the old build_windows short-circuit).
    """
    if len(sentences) < 3:
        return [(s, [(i, 0, len(s))]) for i, s in enumerate(sentences)]

    window_size = 3
    step = 2   # overlap = window_size - step = 1

    def _make_window(
        indices: list[int],
    ) -> tuple[str, list[tuple[int, int, int]]]:
        offsets: list[tuple[int, int, int]] = []
        pos = 0
        for si in indices:
            s = sentences[si]
            offsets.append((si, pos, pos + len(s)))
            pos += len(s) + 1   # +1 for the joining space
        return " ".join(sentences[si] for si in indices), offsets

    result: list[tuple[str, list[tuple[int, int, int]]]] = []
    for start in range(0, len(sentences) - window_size + 1, step):
        result.append(_make_window(list(range(start, start + window_size))))

    # Trailing window: ensure the last sentence(s) are always covered.
    last_start = ((len(sentences) - window_size) // step) * step
    if last_start + window_size < len(sentences):
        result.append(
            _make_window(list(range(len(sentences) - window_size, len(sentences))))
        )

    return result


def build_windows(sentences: list[str]) -> list[str]:
    """
    Convenience wrapper: return just the window strings without offset tables.
    Used by TF-IDF scoring and any caller that doesn't need page resolution.
    """
    return [w for w, _ in build_windows_with_offsets(sentences)]
