"""
Retrieval, reranking, and page-number resolution.

Public API:
  select_context(doc_data, question)           -> str
  select_context_with_meta(doc_data, question) -> dict
  rerank_windows(question, windows)            -> list[tuple[int, str]]
  resolve_page_number(char_start, ...)         -> (int|None, int|None)

Module-level flags (monkeypatch-safe — eval.py patches these between runs):
  RETRIEVAL_MODE      "tfidf" | "dense" | "hybrid"
  USE_RERANKER        bool
  RERANKER_CANDIDATES int
  RERANKER_MARGIN     float
  MIN_RERANKED        int
  MAX_RERANKED        int
  RERANKER_SCORE_FLOOR float
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from extraction import build_windows_with_offsets
from qa import CUSTOM_STOPWORDS, get_model, get_reranker

# ---------------------------------------------------------------------------
# Retrieval flags
# ---------------------------------------------------------------------------

# Controls which retrieval strategy select_context uses for the candidate
# window shortlist before (optionally) reranking.
#   "tfidf"  — TF-IDF cosine similarity only (default, zero extra memory)
#   "dense"  — dense bi-encoder retrieval (requires embedding model in memory)
#   "hybrid" — RRF fusion of TF-IDF + dense scores
# Only "tfidf" is fully implemented; dense/hybrid fall through to TF-IDF.
RETRIEVAL_MODE: str = "tfidf"

# When True, the top-N candidates from RETRIEVAL_MODE are re-scored and
# re-sorted by the cross-encoder before being passed to the QA model.
# Set to False (default) to skip reranking entirely — the reranker model is
# never loaded when this flag is False.
USE_RERANKER: bool = False

# Candidate pool size when USE_RERANKER is True.
RERANKER_CANDIDATES: int = 20

# Score-gap margin: keep windows within this many logit units of the top score.
RERANKER_MARGIN: float = 2.0
MIN_RERANKED: int = 1
MAX_RERANKED: int = 4

# Absolute relevance floor: if the top reranker score is below this value, the
# reranker considers no window relevant and returns just the single best window.
RERANKER_SCORE_FLOOR: float = -5.0


# ---------------------------------------------------------------------------
# Context assembly helper
# ---------------------------------------------------------------------------

def _assemble_context(
    windows: list[str],
    sent_offsets_per_window: list[list[tuple[int, int, int]]],
) -> tuple[str, list[tuple[int, int, int]], list[list[tuple[int, int, int]]]]:
    """
    Join *windows* with a single-space separator while building the two-level
    offset tables required by resolve_page_number().

    Returns:
      context          — assembled string
      window_offsets   — [(win_idx, ctx_start, ctx_end), ...] one per window
      window_sent_offs — parallel list; each entry is the
                         [(sent_idx, win_start, win_end), ...] table for that
                         window (passed through unchanged from input)

    Walking positions one window at a time (not str.join + search) makes
    offsets exact by construction.
    """
    context_parts:   list[str]                          = []
    window_offsets:  list[tuple[int, int, int]]         = []
    window_sent_offs: list[list[tuple[int, int, int]]]  = []

    running_pos = 0
    sep_len = 1   # single space separator

    for i, (win_text, s_offs) in enumerate(
        zip(windows, sent_offsets_per_window)
    ):
        ctx_start = running_pos
        ctx_end   = running_pos + len(win_text)
        window_offsets.append((i, ctx_start, ctx_end))
        window_sent_offs.append(s_offs)
        context_parts.append(win_text)
        running_pos = ctx_end + sep_len

    return " ".join(context_parts), window_offsets, window_sent_offs


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

def rerank_windows(
    question: str,
    windows: list[str],
) -> list[tuple[int, str]]:
    """
    Score each (question, window) pair with the ONNX cross-encoder in a single
    batched session.run() call and return windows sorted by descending score,
    pruned by a score-gap margin and an absolute floor.

    Returns a list of (original_index, window_text) pairs so callers can look
    up per-window metadata by the original index rather than by window text
    (which is not unique — duplicate windows from repeated journal headers or
    boilerplate would silently collapse a text-keyed dict).

    Cutoff:
      - Keep windows within RERANKER_MARGIN logit units of the top score.
      - Bounds: MIN_RERANKED ≤ returned count ≤ MAX_RERANKED.
      - If top score < RERANKER_SCORE_FLOOR return only the single best window.

    Reranker failure never breaks a request — input is returned as
    [(i, windows[i]) for i in range(len(windows))].
    """
    if not windows:
        return []

    try:
        rtok, rsess = get_reranker()

        encodings = [rtok.encode(question, w) for w in windows]
        max_len = max(len(e.ids) for e in encodings)

        ids_batch   = np.zeros((len(encodings), max_len), dtype=np.int64)
        mask_batch  = np.zeros((len(encodings), max_len), dtype=np.int64)
        ttids_batch = np.zeros((len(encodings), max_len), dtype=np.int64)

        for i, enc in enumerate(encodings):
            n = len(enc.ids)
            ids_batch[i, :n]   = enc.ids
            mask_batch[i, :n]  = enc.attention_mask
            ttids_batch[i, :n] = enc.type_ids

        input_names = {inp.name for inp in rsess.get_inputs()}
        input_feed: dict = {
            "input_ids":      ids_batch,
            "attention_mask": mask_batch,
        }
        if "token_type_ids" in input_names:
            input_feed["token_type_ids"] = ttids_batch

        raw_out = rsess.run(output_names=None, input_feed=input_feed)
        # Output may be [batch, 1] or [batch] — flatten either way.
        scores    = np.array(raw_out[0], dtype=np.float32).flatten()
        top_score = float(scores.max())

        if top_score < RERANKER_SCORE_FLOOR:
            best = int(scores.argmax())
            return [(best, windows[best])]

        ranked: list[int] = sorted(
            range(len(windows)), key=lambda i: scores[i], reverse=True
        )
        selected: list[tuple[int, str]] = []
        for rank_pos, win_idx in enumerate(ranked):
            if rank_pos >= MAX_RERANKED:
                break
            if rank_pos >= MIN_RERANKED and (
                top_score - float(scores[win_idx])
            ) > RERANKER_MARGIN:
                break
            selected.append((win_idx, windows[win_idx]))

        return selected if selected else [(ranked[0], windows[ranked[0]])]

    except Exception:
        return list(enumerate(windows))


# ---------------------------------------------------------------------------
# Page-number resolution
# ---------------------------------------------------------------------------

def resolve_page_number(
    char_start: int,
    window_offsets: list[tuple[int, int, int]],
    window_sent_offsets: list[list[tuple[int, int, int]]],
    sentence_meta: list[dict],
) -> tuple[int | None, int | None]:
    """
    Map *char_start* (an offset into the assembled context string returned by
    get_answer) back to a PDF page number via two-level offset tables.

    Step 1 — find which selected window contains char_start.
    Step 2 — convert to window-relative position.
    Step 3 — find which sentence within that window contains the position.
    Step 4 — look up sentence_meta[sent_idx]["page"].

    Uses char_start (not char_end or midpoint): the page where the answer
    *begins* is the natural source page for the user to verify.

    The sentence shared between adjacent overlapping windows is resolved
    within the specific selected window instance, not searched globally.

    Returns (page_number, sentence_index); both None on any failure.
    """
    if char_start < 0 or not window_offsets:
        return None, None

    try:
        # Step 1 — locate the containing window.
        containing = None
        for pos, (_, ctx_start, ctx_end) in enumerate(window_offsets):
            if ctx_start <= char_start < ctx_end:
                containing = pos
                break
        if containing is None:
            containing = len(window_offsets) - 1   # trailing-whitespace fallback

        _, ctx_start, _ = window_offsets[containing]
        sent_offsets    = window_sent_offsets[containing]

        # Step 2 — window-relative position.
        win_pos = char_start - ctx_start

        # Step 3 — find the containing sentence.
        best_sent_idx = sent_offsets[0][0]
        for sent_idx, win_start, win_end in sent_offsets:
            if win_start <= win_pos < win_end:
                best_sent_idx = sent_idx
                break
            # Update as we scan so inter-sentence whitespace lands on the
            # nearest preceding sentence rather than defaulting to sent 0.
            if win_start <= win_pos:
                best_sent_idx = sent_idx

        # Step 4 — page lookup.
        if best_sent_idx < len(sentence_meta):
            return sentence_meta[best_sent_idx].get("page"), best_sent_idx

        return None, best_sent_idx

    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------

def select_context(doc_data: dict, question: str, top_n: int = 5) -> str:
    """
    Public API for tests and legacy callers.  Returns the assembled context
    string, or "" when no passage clears the relevance threshold.
    """
    return select_context_with_meta(doc_data, question, top_n=top_n)["context"]


def select_context_with_meta(
    doc_data: dict, question: str, top_n: int = 5
) -> dict:
    """
    Retrieve the most relevant windows for *question*, apply score-threshold
    pruning, and (optionally) rerank via the cross-encoder.

    Returns:
      {
        "context":              str,
        "retrieval_confidence": float,   # top score / max_score ∈ [0, 1]
        "found":                bool,    # False → no passage cleared threshold
        "window_offsets":       list,    # [(win_idx, ctx_start, ctx_end), ...]
        "window_sent_offsets":  list,    # [[(sent_idx, ws, we), ...], ...]
      }

    Cutoff: keep windows at or above the 40th percentile of this query's own
    score distribution; clamp to [1, 8].  If max_score == 0 (no vocabulary
    overlap) return found=False immediately without calling the QA model.
    """
    _EMPTY: dict = {
        "context": "", "retrieval_confidence": 0.0, "found": False,
        "window_offsets": [], "window_sent_offsets": [],
    }

    sentences = doc_data.get("sentences", [])
    if not sentences:
        return _EMPTY

    # ── Filter out sentences that fall after the references/bibliography ──
    # heading.  sentence_meta entries with past_references=True are kept in
    # doc_data for display/debugging but excluded from the retrieval pool so
    # bibliography entries don't outrank actual body text on keyword overlap.
    # If no heading was detected (past_references never set), nothing changes.
    sentence_meta_full = doc_data.get("sentence_meta", [{"page": 0}] * len(sentences))
    body_indices = [
        i for i, m in enumerate(sentence_meta_full)
        if not m.get("past_references", False)
    ]
    # Fall back to all sentences if the filter would leave us with nothing
    # (shouldn't happen in practice, but guards against edge cases).
    if not body_indices:
        body_indices = list(range(len(sentences)))
    body_sentences = [sentences[i] for i in body_indices]

    windows_with_offsets    = build_windows_with_offsets(body_sentences)
    windows                 = [w for w, _ in windows_with_offsets]
    sent_offsets_per_window = [so for _, so in windows_with_offsets]

    # Remap the sentence indices in sent_offsets_per_window back to the
    # original global sentence indices so resolve_page_number can look up
    # the correct sentence_meta entry.
    def _remap(so: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        return [(body_indices[local_i], ws, we) for local_i, ws, we in so]

    sent_offsets_per_window = [_remap(so) for so in sent_offsets_per_window]

    # ── TF-IDF scoring ──────────────────────────────────────────────────────
    try:
        vectorizer   = TfidfVectorizer(stop_words=CUSTOM_STOPWORDS)
        tfidf_matrix = vectorizer.fit_transform([question] + windows)
        raw_scores   = cosine_similarity(
            tfidf_matrix[0:1], tfidf_matrix[1:]
        ).flatten()
    except Exception:
        # All-stopword question or empty vocabulary — return first window.
        if not windows:
            return _EMPTY
        ctx, w_off, ws_off = _assemble_context(
            [windows[0]], [sent_offsets_per_window[0]]
        )
        return {
            "context": ctx, "retrieval_confidence": 0.0, "found": True,
            "window_offsets": w_off, "window_sent_offsets": ws_off,
        }

    max_score = float(raw_scores.max()) if len(raw_scores) else 0.0
    if max_score == 0.0:
        return _EMPTY

    # ── 40th-percentile threshold, bounds [1, 8] ────────────────────────────
    threshold     = float(np.percentile(raw_scores, 40))
    MAX_CANDS, MIN_CANDS = 8, 1

    ranked_idx = sorted(
        range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True
    )
    above = [i for i in ranked_idx if raw_scores[i] >= threshold]

    if len(above) < MIN_CANDS:
        above = ranked_idx[:MIN_CANDS]
    elif len(above) > MAX_CANDS:
        above = above[:MAX_CANDS]

    retrieval_confidence = round(float(raw_scores[above[0]]) / max_score, 4)

    # Widen the pool for reranking.
    if USE_RERANKER and len(above) < RERANKER_CANDIDATES:
        extra = [i for i in ranked_idx if i not in set(above)]
        above = above + extra[: RERANKER_CANDIDATES - len(above)]

    # Document order for the fast path (preserves reading coherence).
    doc_order       = sorted(above)
    cand_windows    = [windows[i]                for i in doc_order]
    cand_sent_offs  = [sent_offsets_per_window[i] for i in doc_order]

    # ── Fast path: no reranking ──────────────────────────────────────────────
    if not USE_RERANKER:
        ctx, w_off, ws_off = _assemble_context(cand_windows, cand_sent_offs)
        return {
            "context": ctx, "retrieval_confidence": retrieval_confidence,
            "found": True,
            "window_offsets": w_off, "window_sent_offsets": ws_off,
        }

    # ── Reranking path ───────────────────────────────────────────────────────
    # rerank_windows returns (original_index, text) pairs so we can look up
    # the correct sent_offsets by index — avoids the text-keyed dict that
    # silently collapses duplicate window texts.
    reranked_pairs = rerank_windows(question, cand_windows)

    TOKEN_BUDGET = 480   # 512 − ~32 special tokens
    sel_windows:   list[str]         = []
    sel_sent_offs: list[list[tuple]] = []
    cumulative = 0

    for orig_idx, window in reranked_pairs:
        try:
            tok, _ = get_model()
            n_toks = len(tok.encode(window).ids)
        except Exception:
            n_toks = len(window.split()) * 2
        if cumulative + n_toks > TOKEN_BUDGET:
            break
        sel_windows.append(window)
        sel_sent_offs.append(cand_sent_offs[orig_idx])
        cumulative += n_toks

    if not sel_windows and reranked_pairs:
        first_idx, first_win = reranked_pairs[0]
        sel_windows   = [first_win]
        sel_sent_offs = [cand_sent_offs[first_idx]]

    ctx, w_off, ws_off = _assemble_context(sel_windows, sel_sent_offs)
    return {
        "context": ctx, "retrieval_confidence": retrieval_confidence,
        "found": True,
        "window_offsets": w_off, "window_sent_offsets": ws_off,
    }
