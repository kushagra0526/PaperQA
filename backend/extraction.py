"""
PDF extraction, sentence chunking, and window construction.

Public API used by the rest of the pipeline:
  extract_and_chunk_pdf(file_bytes)  -> doc_data dict
  build_windows(sentences)           -> list[str]
  build_windows_with_offsets(sents)  -> list[(str, [(int,int,int)])]
  DOC_CACHE                          -> the live cache dict (for test teardown)
"""

from __future__ import annotations

import hashlib
import re
import threading
from io import BytesIO

import fitz          # PyMuPDF
import nltk
import numpy as np   # used by caller modules; re-exported for convenience
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# NLTK data guard — download punkt once, silently, if not already present.
# ---------------------------------------------------------------------------
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

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
# Layout-aware extraction (PyMuPDF)
# ---------------------------------------------------------------------------

def extract_text_layout_aware(file_bytes: bytes) -> tuple[str, list[int]]:
    """
    Extract text from a PDF using PyMuPDF with layout-aware word ordering.

    For each page, word bounding boxes are used to detect two-column layout.
    Two-column detection: if a meaningful fraction of words cluster to the left
    of the page midpoint AND a meaningful fraction cluster to the right, the page
    is treated as two-column and words are sorted left-column-first, then right.
    Single-column pages are sorted by (y0, x0) as normal reading order.

    Returns:
        raw_text        — all pages joined by spaces (matches pypdf format).
        word_page_index — flat list of page numbers, one entry per word in
                          the same order as the reconstructed text, used by
                          _build_sentence_meta to map sentences to pages.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception:
        return "", []

    page_texts: list[str] = []
    word_page_index: list[int] = []

    try:
        for page_no, page in enumerate(doc):
            if page_no >= 25:           # mirror the 25-page cap
                break

            words = page.get_text("words")   # (x0,y0,x1,y1,word,blk,ln,wn)
            if not words:
                continue

            mid_x = page.rect.width / 2.0
            left_words  = [w for w in words if w[0] < mid_x]
            right_words = [w for w in words if w[0] >= mid_x]

            # Two-column heuristic: both halves must each hold ≥15% of words.
            total = len(words)
            is_two_column = (
                total > 0
                and len(left_words) / total >= 0.15
                and len(right_words) / total >= 0.15
            )

            if is_two_column:
                ordered = (
                    sorted(left_words,  key=lambda w: (w[1], w[0]))
                    + sorted(right_words, key=lambda w: (w[1], w[0]))
                )
            else:
                ordered = sorted(words, key=lambda w: (w[1], w[0]))

            page_texts.append(" ".join(w[4] for w in ordered))
            word_page_index.extend([page_no] * len(ordered))
    finally:
        doc.close()

    return " ".join(page_texts), word_page_index


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

    meta: list[dict] = []
    for sent in sentences:
        sent_start = raw_text.find(sent)
        if sent_start == -1:
            meta.append({"page": 0})
            continue
        page_no = 0
        for i, cs in enumerate(word_char_starts):
            if cs <= sent_start:
                page_no = word_page_index[i]
            else:
                break
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
            # Find the sentence whose occurrence in raw_text is closest to
            # (and at or after) heading_char.  We can't just take the first
            # sentence in list order whose text appears after heading_char,
            # because repeated headers/watermarks mean a short sentence may
            # match a second occurrence later in the text — we want the
            # sentence positioned nearest to the actual heading.
            best_i   = None
            best_pos = len(raw_text) + 1
            for i, sent in enumerate(sentences):
                pos = raw_text.find(sent, max(0, heading_char - 5))
                if pos != -1 and pos >= heading_char and pos < best_pos:
                    best_pos = pos
                    best_i   = i
            if best_i is not None:
                boundary = best_i
            # If every sentence-level search missed (heading is run-on),
            # fall back: find the sentence that *contains* heading_char and
            # start filtering from the next one.
            if boundary is None:
                for i, sent in enumerate(sentences):
                    pos = raw_text.find(sent)
                    if pos != -1 and pos <= heading_char < pos + len(sent):
                        boundary = i + 1
                        break

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
        raw_text, word_page_index = extract_text_layout_aware(file_bytes)
        # Reconstruct the flat word list in the same order as word_page_index
        # so _build_sentence_meta can locate character offsets.
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            try:
                for page_no, page in enumerate(doc):
                    if page_no >= 25:
                        break
                    words = page.get_text("words")
                    if not words:
                        continue
                    mid_x = page.rect.width / 2.0
                    left_words  = [w for w in words if w[0] < mid_x]
                    right_words = [w for w in words if w[0] >= mid_x]
                    total = len(words)
                    is_two_column = (
                        total > 0
                        and len(left_words) / total >= 0.15
                        and len(right_words) / total >= 0.15
                    )
                    if is_two_column:
                        ordered = (
                            sorted(left_words,  key=lambda w: (w[1], w[0]))
                            + sorted(right_words, key=lambda w: (w[1], w[0]))
                        )
                    else:
                        ordered = sorted(words, key=lambda w: (w[1], w[0]))
                    page_words.extend(w[4] for w in ordered)
            finally:
                doc.close()
        except Exception:
            pass
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
