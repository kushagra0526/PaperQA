import os
# Disable parallelism in tokenizers and cap CPU threads to 1 to avoid memory pool
# fragmentation in multi-threaded environments, which matters especially on low-RAM hosts.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gc
import hashlib
import re
import threading
import time
from io import BytesIO
import fitz  # PyMuPDF
import nltk
import numpy as np
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from tokenizers import Tokenizer
import onnxruntime as ort

# Ensure the punkt sentence tokenizer data is present.  nltk.download is a no-op
# if the resource is already on disk, so this is safe to call at import time.
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Preserve negation terms that sklearn's built-in "english" stopword list removes.
# Stripping "not", "no", "none", "cannot", "never" breaks retrieval for questions
# like "What does NOT apply?" or context sentences that contain negations.
NEGATIONS = {"not", "no", "none", "cannot", "never"}
CUSTOM_STOPWORDS = list(ENGLISH_STOP_WORDS - NEGATIONS)

app = FastAPI(title="Scientific Paper QA")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

QA_MODEL_NAME = "deepset/minilm-uncased-squad2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Local directory where preload.py saves the ONNX-exported, INT8-quantized model
# during the build step.  At runtime, get_model() loads from here if available.
ONNX_MODEL_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_model")
ONNX_RERANKER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_reranker")

# When True, extract_and_chunk_pdf uses the layout-aware PyMuPDF path
# (extract_text_layout_aware) which handles two-column papers correctly.
# Set to False to fall back to the original pypdf-based extraction.
USE_LAYOUT_AWARE_EXTRACTION = True

# Controls which retrieval strategy select_context uses to build the candidate
# window shortlist before (optionally) reranking.
#   "tfidf"  — TF-IDF cosine similarity only (default, zero extra memory)
#   "dense"  — dense bi-encoder retrieval (requires embedding model in memory)
#   "hybrid" — RRF fusion of TF-IDF + dense scores
# Only "tfidf" is fully implemented in this pass; the flag is wired so that
# adding dense/hybrid later only requires filling in the retrieval path.
RETRIEVAL_MODE = "tfidf"

# When True, the top-N candidates from RETRIEVAL_MODE are re-scored and
# re-sorted by a cross-encoder before being passed to the QA model.
# Set to False (default) to skip reranking entirely — the reranker model is
# never loaded when this flag is False, keeping memory usage unchanged.
USE_RERANKER = False

# Number of candidate windows to retrieve before reranking.  Only used when
# USE_RERANKER is True; otherwise select_context uses top_n directly.
RERANKER_CANDIDATES = 20

# Reranker score margin: windows whose score is within this many logit units of
# the top score are kept.  Enforces MIN_RERANKED / MAX_RERANKED bounds.
RERANKER_MARGIN   = 2.0
MIN_RERANKED      = 1
MAX_RERANKED      = 4

# Absolute score floor: if the top reranker score is below this threshold the
# reranker considers no window sufficiently relevant and the fallback path in
# select_context returns the raw top-1 TF-IDF candidate instead.
RERANKER_SCORE_FLOOR = -5.0

tokenizer = None
model = None
DOC_CACHE = {}

# Timestamp of the most recent call to get_model(), used by the idle-unload
# thread to decide when to evict the model from memory.
_last_model_use = 0.0
_MODEL_IDLE_TIMEOUT = 600  # 10 minutes in seconds

# Lock protecting lazy model initialisation so that concurrent FastAPI threadpool
# workers cannot trigger duplicate downloads or simultaneous model loads.
_model_lock = threading.Lock()

# Lock protecting all mutations to DOC_CACHE (eviction + insertion) because
# dict.pop() and key assignment are not atomic across threads in all Python builds.
_cache_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Reranker model state — independent of the QA model.
# Only ever loaded when USE_RERANKER is True; a tfidf-mode request with
# USE_RERANKER=False never touches these globals.
# ---------------------------------------------------------------------------
_reranker_tokenizer = None   # tokenizers.Tokenizer for the cross-encoder
_reranker_session   = None   # ort.InferenceSession for the cross-encoder
_last_reranker_use  = 0.0
_reranker_lock      = threading.Lock()


def get_model():
    global tokenizer, model, _last_model_use
    # First check without the lock for the common hot-path where model is already loaded.
    if tokenizer is None or model is None:
        with _model_lock:
            # Re-check inside the lock: another thread may have loaded the model
            # between our outer check and acquiring the lock.
            if tokenizer is None or model is None:
                # Pin ONNX Runtime to single-threaded inference.  The OMP/MKL
                # env vars at the top of this file do not control ORT's own
                # thread pool, so without this it may spawn extra threads.
                session_options = ort.SessionOptions()
                session_options.intra_op_num_threads = 1
                session_options.inter_op_num_threads = 1

                if os.path.isdir(ONNX_MODEL_DIR):
                    # Load the standalone tokenizer directly from tokenizer.json.
                    # This avoids importing transformers (and its conditional torch
                    # import via is_torch_available()) entirely from the runtime path.
                    _tok_path = os.path.join(ONNX_MODEL_DIR, "tokenizer.json")
                    tokenizer = Tokenizer.from_file(_tok_path)
                    tokenizer.enable_truncation(max_length=512)
                    model = ort.InferenceSession(
                        os.path.join(ONNX_MODEL_DIR, "model_quantized.onnx"),
                        sess_options=session_options,
                        providers=["CPUExecutionProvider"],
                    )
                else:
                    # Dev fallback: export + quantize on-the-fly if preload.py
                    # hasn't been run yet.  Uses optimum/transformers only at build
                    # time; the resulting session is a plain ort.InferenceSession.
                    from optimum.onnxruntime import ORTModelForQuestionAnswering, ORTQuantizer
                    from optimum.onnxruntime.configuration import AutoQuantizationConfig
                    from transformers import AutoTokenizer as _AutoTokenizer
                    from tempfile import TemporaryDirectory
                    with TemporaryDirectory() as tmp_dir:
                        _ort_model = ORTModelForQuestionAnswering.from_pretrained(
                            QA_MODEL_NAME, export=True
                        )
                        _ort_model.save_pretrained(tmp_dir)
                        _quantizer = ORTQuantizer.from_pretrained(tmp_dir)
                        _qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
                        _quantizer.quantize(save_dir=ONNX_MODEL_DIR, quantization_config=_qconfig)
                    _AutoTokenizer.from_pretrained(QA_MODEL_NAME).save_pretrained(ONNX_MODEL_DIR)
                    _tok_path = os.path.join(ONNX_MODEL_DIR, "tokenizer.json")
                    tokenizer = Tokenizer.from_file(_tok_path)
                    tokenizer.enable_truncation(max_length=512)
                    model = ort.InferenceSession(
                        os.path.join(ONNX_MODEL_DIR, "model_quantized.onnx"),
                        sess_options=session_options,
                        providers=["CPUExecutionProvider"],
                    )
                gc.collect()
    _last_model_use = time.time()
    return tokenizer, model


def get_reranker():
    """
    Lazy-load the ONNX cross-encoder reranker under a double-checked lock,
    mirroring get_model()'s pattern exactly.  Returns (tokenizer, session).

    Only called when USE_RERANKER is True — a tfidf-mode request that doesn't
    need the reranker never enters this function, so the reranker model is
    never loaded into memory in that case.
    """
    global _reranker_tokenizer, _reranker_session, _last_reranker_use
    if _reranker_tokenizer is None or _reranker_session is None:
        with _reranker_lock:
            if _reranker_tokenizer is None or _reranker_session is None:
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 1
                sess_options.inter_op_num_threads = 1

                if os.path.isdir(ONNX_RERANKER_DIR):
                    _tok_path = os.path.join(ONNX_RERANKER_DIR, "tokenizer.json")
                    _reranker_tokenizer = Tokenizer.from_file(_tok_path)
                    _reranker_tokenizer.enable_truncation(max_length=512)
                    _reranker_session = ort.InferenceSession(
                        os.path.join(ONNX_RERANKER_DIR, "model_quantized.onnx"),
                        sess_opts=sess_options,
                        providers=["CPUExecutionProvider"],
                    )
                else:
                    # Dev fallback: export + quantize on-the-fly.
                    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
                    from optimum.onnxruntime.configuration import AutoQuantizationConfig
                    from transformers import AutoTokenizer as _AutoTokenizer
                    from tempfile import TemporaryDirectory
                    with TemporaryDirectory() as tmp_dir:
                        _m = ORTModelForSequenceClassification.from_pretrained(
                            RERANKER_MODEL_NAME, export=True
                        )
                        _m.save_pretrained(tmp_dir)
                        _q = ORTQuantizer.from_pretrained(tmp_dir)
                        _qcfg = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
                        _q.quantize(save_dir=ONNX_RERANKER_DIR, quantization_config=_qcfg)
                    _AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME).save_pretrained(ONNX_RERANKER_DIR)
                    _tok_path = os.path.join(ONNX_RERANKER_DIR, "tokenizer.json")
                    _reranker_tokenizer = Tokenizer.from_file(_tok_path)
                    _reranker_tokenizer.enable_truncation(max_length=512)
                    _reranker_session = ort.InferenceSession(
                        os.path.join(ONNX_RERANKER_DIR, "model_quantized.onnx"),
                        sess_opts=sess_options,
                        providers=["CPUExecutionProvider"],
                    )
                gc.collect()
    _last_reranker_use = time.time()
    return _reranker_tokenizer, _reranker_session


def _numpy_softmax(x):
    """Numerically stable softmax over a 1-D numpy array."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# Start a daemon thread immediately after the server process starts so the model
# is loaded into RAM before the first real request arrives, avoiding a cold-start
# latency spike for the initial user.
threading.Thread(target=get_model, daemon=True).start()


def _idle_unload_loop():
    """Periodically checks whether the model has been idle and unloads it to free memory."""
    global tokenizer, model
    while True:
        time.sleep(120)  # Check every 2 minutes
        if model is not None and (time.time() - _last_model_use) > _MODEL_IDLE_TIMEOUT:
            with _model_lock:
                # Double-check inside the lock in case a request just arrived.
                if model is not None and (time.time() - _last_model_use) > _MODEL_IDLE_TIMEOUT:
                    tokenizer = None
                    model = None
                    gc.collect()


# Background daemon that reclaims model memory after 10 minutes of inactivity.
# The existing lazy-loading in get_model() will reload it on the next request.
threading.Thread(target=_idle_unload_loop, daemon=True).start()


def _reranker_idle_unload_loop():
    """
    Periodically evicts the reranker from memory when idle, mirroring the QA
    model's _idle_unload_loop.  Only worth checking when USE_RERANKER is True,
    but the loop is harmless when it's False (the globals stay None permanently).
    """
    global _reranker_tokenizer, _reranker_session
    while True:
        time.sleep(120)
        if _reranker_session is not None and \
                (time.time() - _last_reranker_use) > _MODEL_IDLE_TIMEOUT:
            with _reranker_lock:
                if _reranker_session is not None and \
                        (time.time() - _last_reranker_use) > _MODEL_IDLE_TIMEOUT:
                    _reranker_tokenizer = None
                    _reranker_session   = None
                    gc.collect()


threading.Thread(target=_reranker_idle_unload_loop, daemon=True).start()


def extract_text_layout_aware(file_bytes: bytes) -> tuple[str, list[dict]]:
    """
    Extract text from a PDF using PyMuPDF with layout-aware word ordering.

    For each page, word bounding boxes are used to detect two-column layout.
    Two-column detection: if a meaningful fraction of words cluster to the left
    of the page midpoint AND a meaningful fraction cluster to the right, the page
    is treated as two-column and words are sorted left-column-first, then right.
    Single-column pages are sorted by (y0, x0) as normal reading order.

    Returns:
        raw_text:      all pages joined by spaces (matches pypdf format).
        sentence_meta_words: list of {"page": int} dicts, one per word, used by
                       the caller to build per-sentence page metadata.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception:
        return "", []

    page_texts: list[str] = []
    # Track which page each word came from so sentence_meta can be populated.
    word_page_index: list[int] = []   # parallel to the flat word list across all pages

    try:
        for page_no, page in enumerate(doc):
            if page_no >= 25:   # mirror the existing 25-page cap
                break

            words = page.get_text("words")  # list of (x0,y0,x1,y1,word,blk,ln,wn)
            if not words:
                continue

            mid_x = page.rect.width / 2.0

            left_words  = [w for w in words if w[0] < mid_x]
            right_words = [w for w in words if w[0] >= mid_x]

            # Two-column heuristic: both halves must each hold at least 15% of
            # the total words on the page.  This avoids misclassifying pages with
            # a small sidebar, header, or margin note as two-column.
            total = len(words)
            is_two_column = (
                total > 0
                and len(left_words) / total >= 0.15
                and len(right_words) / total >= 0.15
            )

            if is_two_column:
                ordered = sorted(left_words,  key=lambda w: (w[1], w[0])) + \
                          sorted(right_words, key=lambda w: (w[1], w[0]))
            else:
                ordered = sorted(words, key=lambda w: (w[1], w[0]))

            page_text = " ".join(w[4] for w in ordered)
            page_texts.append(page_text)

            # Record the source page for every word so the caller can map
            # sentences back to page numbers.
            word_page_index.extend([page_no] * len(ordered))
    finally:
        doc.close()

    raw_text = " ".join(page_texts)
    return raw_text, word_page_index


def _build_sentence_meta(
    sentences: list[str],
    raw_text: str,
    word_page_index: list[int],
    page_words: list[str],
) -> list[dict]:
    """
    Align each sentence to a page number by finding the sentence's character
    offset in raw_text, then mapping that to the nearest word's page.

    page_words is a flat list of the word strings in the same order as
    word_page_index (used to locate character positions).
    """
    if not word_page_index or not page_words:
        return [{"page": 0}] * len(sentences)

    # Build a list of (char_start, page_no) for each word in raw_text so we can
    # binary-search the sentence's start position to find its page.
    word_char_starts: list[int] = []
    pos = 0
    for word in page_words:
        # raw_text was built with single spaces between words; find each word's
        # actual position in the string to stay robust to fitz word content.
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
        # Find the last word whose char_start is <= sent_start.
        page_no = 0
        for i, cs in enumerate(word_char_starts):
            if cs <= sent_start:
                page_no = word_page_index[i]
            else:
                break
        meta.append({"page": page_no})
    return meta


def extract_and_chunk_pdf(file_bytes: bytes) -> dict:
    pdf_hash = hashlib.sha256(file_bytes).hexdigest()
    if pdf_hash in DOC_CACHE:
        return DOC_CACHE[pdf_hash]

    word_page_index: list[int] = []
    page_words:      list[str] = []

    if USE_LAYOUT_AWARE_EXTRACTION:
        # Layout-aware path: PyMuPDF word-level extraction with two-column detection.
        raw_text, word_page_index = extract_text_layout_aware(file_bytes)
        # Reconstruct flat word list in the same order as word_page_index so
        # _build_sentence_meta can locate character offsets.
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
                        ordered = sorted(left_words,  key=lambda w: (w[1], w[0])) + \
                                  sorted(right_words, key=lambda w: (w[1], w[0]))
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
            raw_text = ' '.join(page.extract_text() or '' for page in pages)
        except Exception:
            raw_text = ""

    # Replace the regex sentence splitter with nltk sent_tokenize, which correctly
    # handles abbreviations (et al., Fig., e.g.,) and decimal numbers.
    try:
        raw_sentences = nltk.sent_tokenize(raw_text) if raw_text.strip() else []
    except Exception:
        # Fallback to the old regex if nltk fails for any reason.
        raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', raw_text)

    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 15]
    if not sentences:
        sentences = [raw_text.strip()] if raw_text.strip() else ["No text found in document."]

    # Build per-sentence page metadata (additive — does not affect existing keys).
    if USE_LAYOUT_AWARE_EXTRACTION and word_page_index:
        sentence_meta = _build_sentence_meta(sentences, raw_text, word_page_index, page_words)
    else:
        sentence_meta = [{"page": 0}] * len(sentences)

    doc_data = {"text": raw_text, "sentences": sentences, "sentence_meta": sentence_meta}

    # Guard cache eviction and insertion together so no other thread can observe
    # a half-updated cache state or evict an entry another thread just inserted.
    with _cache_lock:
        if len(DOC_CACHE) >= 3:
            DOC_CACHE.pop(next(iter(DOC_CACHE)))
        DOC_CACHE[pdf_hash] = doc_data
    return doc_data


def build_windows_with_offsets(
    sentences: list[str],
) -> list[tuple[str, list[tuple[int, int, int]]]]:
    """
    Same windowing logic as build_windows(), but also returns per-window
    sentence-offset tables needed for page-number resolution.

    Returns a list of (window_text, sent_offsets) pairs where:
      window_text  — the window string (sentences joined with a single space)
      sent_offsets — list of (sentence_index, start_in_window, end_in_window)
                     one entry per sentence in this window, offsets are
                     character positions within window_text.

    Example for a window made from sentences[2..4]:
      window_text  = "Sent2. Sent3. Sent4."
      sent_offsets = [(2, 0, 6), (3, 7, 13), (4, 14, 20)]
    """
    if len(sentences) < 3:
        # Mirror build_windows(): return each sentence as its own "window".
        result = []
        for i, s in enumerate(sentences):
            result.append((s, [(i, 0, len(s))]))
        return result

    window_size = 3
    step = 2
    result: list[tuple[str, list[tuple[int, int, int]]]] = []

    def _make_window(sent_slice_indices: list[int]):
        """Build one window entry from a list of sentence indices."""
        offsets: list[tuple[int, int, int]] = []
        pos = 0
        parts = []
        for si in sent_slice_indices:
            s = sentences[si]
            offsets.append((si, pos, pos + len(s)))
            parts.append(s)
            pos += len(s) + 1  # +1 for the joining space
        return (" ".join(sentences[si] for si in sent_slice_indices), offsets)

    for start in range(0, len(sentences) - window_size + 1, step):
        result.append(_make_window(list(range(start, start + window_size))))

    last_start = ((len(sentences) - window_size) // step) * step
    if last_start + window_size < len(sentences):
        result.append(_make_window(list(range(len(sentences) - window_size, len(sentences)))))

    return result


def build_windows(sentences: list[str]) -> list[str]:
    """
    Group sentences into overlapping windows of 3 with a 1-sentence step,
    so each window shares its last sentence with the start of the next.

    Window indices: [0,1,2], [2,3,4], [4,5,6], ...
    (step = window_size - overlap = 3 - 1 = 2)

    Each window is returned as a single string with sentences joined by a space,
    giving the TF-IDF scorer a richer unit of context than a bare sentence while
    keeping the vocabulary overlap with the question higher than a full paragraph.

    If there are fewer than 3 sentences the list is returned unchanged — the
    caller's existing short-circuit handles that case.
    """
    return [w for w, _ in build_windows_with_offsets(sentences)]


def rerank_windows(question: str, windows: list[str]) -> list[str]:
    """
    Score each (question, window) pair with the ONNX cross-encoder and return
    windows sorted by descending relevance score, pruned by a score-gap
    margin and an absolute score floor.

    Scoring:
      - Tokenize each pair with the reranker's own tokenizer (truncated to 512).
      - Run a single batched ort.InferenceSession.run() call — all pairs in one
        call rather than a Python loop, so ORT can exploit any intra-batch
        parallelism and we pay the Python overhead only once.
      - The session output is a single [batch, 1] or [batch] logit tensor;
        higher logit = more relevant.  No sigmoid/softmax needed for ranking.

    Cutoff (consistent with the spec's "better version"):
      - Keep windows whose score is within RERANKER_MARGIN logit units of the
        top score (i.e. score >= top_score - RERANKER_MARGIN).
      - Hard bounds: always keep at least MIN_RERANKED, at most MAX_RERANKED.
      - If the top score is below RERANKER_SCORE_FLOOR (absolute floor), the
        reranker considers no window relevant enough; fall back to returning the
        single best window from the input order rather than forcing low-signal
        context through.

    Returns the final ordered list of window strings (best-first).
    """
    if not windows:
        return windows

    try:
        rtok, rsess = get_reranker()

        # Build a batched numpy input: shape [n_pairs, seq_len] padded to the
        # longest encoded sequence in this batch.
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

        # Output shape is [batch, 1] for binary classifiers or [batch] for
        # regression heads — flatten to a 1-D array either way.
        scores = np.array(raw_out[0], dtype=np.float32).flatten()

        top_score = float(scores.max())

        # Absolute floor: if even the best window is irrelevant, return it alone
        # rather than concatenating multiple low-signal windows.
        if top_score < RERANKER_SCORE_FLOOR:
            best_idx = int(scores.argmax())
            return [windows[best_idx]]

        # Score-gap margin cutoff with MIN/MAX bounds.
        ranked = sorted(range(len(windows)), key=lambda i: scores[i], reverse=True)
        selected: list[str] = []
        for rank_pos, win_idx in enumerate(ranked):
            if rank_pos >= MAX_RERANKED:
                break
            if rank_pos >= MIN_RERANKED and \
                    (top_score - float(scores[win_idx])) > RERANKER_MARGIN:
                break
            selected.append(windows[win_idx])

        return selected if selected else [windows[ranked[0]]]

    except Exception:
        # Reranker failure must never break the request — return input unchanged.
        return windows


def resolve_page_number(
    char_start: int,
    window_offsets: list[tuple[int, int, int]],
    window_sent_offsets: list[list[tuple[int, int, int]]],
    sentence_meta: list[dict],
) -> tuple[int | None, int | None]:
    """
    Map a char_start offset in the assembled context string back to a page number
    and sentence index using two-level offset tables built during context assembly.

    Arguments:
      char_start          — offset into the final joined context string
                            (the value returned by get_answer as "char_start")
      window_offsets      — list of (window_idx, ctx_start, ctx_end) triples,
                            one per selected window, in the order they appear in
                            the assembled context string
      window_sent_offsets — list aligned with window_offsets; each entry is a list
                            of (sentence_idx, win_start, win_end) triples for the
                            sentences that make up that window
      sentence_meta       — doc_data["sentence_meta"]: list of {"page": int} dicts,
                            indexed by global sentence index

    Returns:
      (page_number, sentence_index) — both None if resolution fails for any reason.

    Resolution uses char_start (not char_end or midpoint) per spec: the page where
    the answer begins is the natural "source page" for the user to verify.

    The overlapping sentence shared between adjacent windows is not ambiguous:
    resolution happens within the specific selected window that was chosen, using
    that window's own offset table.
    """
    if char_start < 0 or not window_offsets:
        return None, None

    try:
        # Step 1: find which selected window contains char_start.
        containing_win_pos = None   # position in window_offsets list
        for pos, (win_idx, ctx_start, ctx_end) in enumerate(window_offsets):
            if ctx_start <= char_start < ctx_end:
                containing_win_pos = pos
                break
        # Fallback: if char_start is exactly at or past the last window's end
        # (can happen with trailing whitespace), use the last window.
        if containing_win_pos is None:
            containing_win_pos = len(window_offsets) - 1

        win_idx, ctx_start, _ = window_offsets[containing_win_pos]
        sent_offsets = window_sent_offsets[containing_win_pos]

        # Step 2: convert to a position relative to the window's own text.
        win_relative_pos = char_start - ctx_start

        # Step 3: find which sentence within this window contains win_relative_pos.
        best_sent_idx = sent_offsets[0][0]   # default to first sentence in window
        for sent_idx, win_start, win_end in sent_offsets:
            if win_start <= win_relative_pos < win_end:
                best_sent_idx = sent_idx
                break
            # Keep updating best_sent_idx as we scan forward so that if
            # win_relative_pos falls in inter-sentence whitespace we land on the
            # nearest preceding sentence rather than defaulting to sent 0.
            if win_start <= win_relative_pos:
                best_sent_idx = sent_idx

        # Step 4: look up the page number from sentence_meta.
        if best_sent_idx < len(sentence_meta):
            page_no = sentence_meta[best_sent_idx].get("page")
            return page_no, best_sent_idx

        return None, best_sent_idx

    except Exception:
        return None, None


def select_context(doc_data: dict, question: str, top_n: int = 5) -> str:
    """
    Public API used by tests and legacy callers.  Returns the assembled context
    string, or "" if no relevant passage was found.  See select_context_with_meta
    for the richer return value used by the /ask endpoint.
    """
    result = select_context_with_meta(doc_data, question, top_n=top_n)
    return result["context"]


def select_context_with_meta(doc_data: dict, question: str, top_n: int = 5) -> dict:
    """
    Retrieve the most relevant windows for *question* from *doc_data*, apply
    score-threshold pruning, and (optionally) rerank.

    Returns a dict:
      {
        "context":              str,    # assembled context string ('' if not found)
        "retrieval_confidence": float,  # top candidate's normalised score [0, 1]
        "found":                bool,   # False → no passage cleared the threshold
        "window_offsets":       list,   # [(win_idx, ctx_start, ctx_end), ...]
        "window_sent_offsets":  list,   # [[(sent_idx, win_start, win_end)], ...]
      }

    window_offsets and window_sent_offsets are the two-level offset tables used by
    resolve_page_number() to map get_answer's char_start back to a page number.
    They are parallel lists: window_offsets[i] describes where the i-th selected
    window sits in the assembled context string; window_sent_offsets[i] describes
    where each of that window's constituent sentences sit within the window's text.

    Retrieval cutoff (replaces the old fixed top_n):
      - Score every window with TF-IDF cosine similarity.
      - Keep only windows whose score is >= the 40th percentile of *that query's*
        score distribution across all windows.
      - Hard bounds: at least 1 window (unless all windows score 0), at most 8.
      - If zero windows survive (all scores are 0 or the distribution is flat),
        treat this as "no relevant passage" and return found=False immediately,
        without calling the QA model.

    retrieval_confidence:
      The top candidate's TF-IDF score normalised to [0, 1] by the max score
      seen across all windows for this query.  After reranking, this remains
      the TF-IDF top score (the cross-encoder score is on a different scale and
      not exposed here; its effect is already reflected in which windows are
      selected).
    """
    _EMPTY = {
        "context": "", "retrieval_confidence": 0.0, "found": False,
        "window_offsets": [], "window_sent_offsets": [],
    }

    sentences = doc_data.get("sentences", [])
    if not sentences:
        return _EMPTY

    # Build windows with per-window sentence-offset tables (Step 2 of spec).
    windows_with_offsets = build_windows_with_offsets(sentences)
    windows = [w for w, _ in windows_with_offsets]
    # sent_offsets_per_window[i] = [(sent_idx, start_in_win, end_in_win), ...]
    sent_offsets_per_window = [so for _, so in windows_with_offsets]

    # ------------------------------------------------------------------ #
    # Score all windows with TF-IDF cosine similarity.                    #
    # ------------------------------------------------------------------ #
    try:
        vectorizer = TfidfVectorizer(stop_words=CUSTOM_STOPWORDS)
        tfidf_matrix = vectorizer.fit_transform([question] + windows)
        raw_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    except Exception:
        # TF-IDF failure (e.g. all-stopword question): fall back to first window.
        if not windows:
            return _EMPTY
        ctx, w_off, ws_off = _assemble_context([windows[0]], [sent_offsets_per_window[0]])
        return {
            "context": ctx, "retrieval_confidence": 0.0, "found": True,
            "window_offsets": w_off, "window_sent_offsets": ws_off,
        }

    max_score = float(raw_scores.max()) if len(raw_scores) else 0.0

    # ------------------------------------------------------------------ #
    # Score-threshold pruning: 40th-percentile floor, bounds [1, 8].     #
    # ------------------------------------------------------------------ #
    if max_score == 0.0:
        # Every window scored 0 — no vocabulary overlap with the question.
        return _EMPTY

    threshold = float(np.percentile(raw_scores, 40))
    MAX_CANDIDATES = 8
    MIN_CANDIDATES = 1

    # Sort descending by score, apply threshold, then enforce bounds.
    ranked_idx = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)
    above = [i for i in ranked_idx if raw_scores[i] >= threshold]

    # Clamp to [MIN_CANDIDATES, MAX_CANDIDATES].
    if len(above) < MIN_CANDIDATES:
        above = ranked_idx[:MIN_CANDIDATES]
    elif len(above) > MAX_CANDIDATES:
        above = above[:MAX_CANDIDATES]

    # Normalised top-candidate score for retrieval_confidence.
    retrieval_confidence = round(float(raw_scores[above[0]]) / max_score, 4)

    # Widen the pool further when reranking so the cross-encoder sees more candidates.
    if USE_RERANKER and len(above) < RERANKER_CANDIDATES:
        extra = [i for i in ranked_idx if i not in set(above)]
        above = above + extra[: RERANKER_CANDIDATES - len(above)]

    # Restore document order for the non-reranker path (preserves reading coherence).
    doc_order_idx = sorted(above)
    candidate_windows   = [windows[i]                for i in doc_order_idx]
    candidate_sent_offs = [sent_offsets_per_window[i] for i in doc_order_idx]

    # ------------------------------------------------------------------ #
    # Fast path: no reranking.                                            #
    # ------------------------------------------------------------------ #
    if not USE_RERANKER:
        ctx, w_off, ws_off = _assemble_context(candidate_windows, candidate_sent_offs)
        return {
            "context": ctx, "retrieval_confidence": retrieval_confidence, "found": True,
            "window_offsets": w_off, "window_sent_offsets": ws_off,
        }

    # ------------------------------------------------------------------ #
    # Reranking path (USE_RERANKER=True).                                 #
    # ------------------------------------------------------------------ #
    reranked_texts = rerank_windows(question, candidate_windows)
    # Rebuild the sent_offsets list in the same reranked order.
    text_to_sent_offs = dict(zip(candidate_windows, candidate_sent_offs))

    TOKEN_BUDGET = 480  # 512 minus ~32 for special tokens
    selected_windows:   list[str]               = []
    selected_sent_offs: list[list[tuple]]       = []
    cumulative_tokens = 0
    for window in reranked_texts:
        try:
            tok, _ = get_model()
            window_tokens = len(tok.encode(window).ids)
        except Exception:
            window_tokens = len(window.split()) * 2
        if cumulative_tokens + window_tokens > TOKEN_BUDGET:
            break
        selected_windows.append(window)
        selected_sent_offs.append(text_to_sent_offs.get(window, []))
        cumulative_tokens += window_tokens

    if not selected_windows:
        selected_windows   = reranked_texts[:1]
        selected_sent_offs = [text_to_sent_offs.get(reranked_texts[0], [])]

    ctx, w_off, ws_off = _assemble_context(selected_windows, selected_sent_offs)
    return {
        "context": ctx, "retrieval_confidence": retrieval_confidence, "found": True,
        "window_offsets": w_off, "window_sent_offsets": ws_off,
    }


def _assemble_context(
    windows: list[str],
    sent_offsets_per_window: list[list[tuple[int, int, int]]],
) -> tuple[str, list[tuple[int, int, int]], list[list[tuple[int, int, int]]]]:
    """
    Join *windows* with a single space separator and simultaneously build the
    two-level offset tables required by resolve_page_number().

    Returns:
      context          — the assembled string
      window_offsets   — [(win_idx, ctx_start, ctx_end), ...] one per window,
                         where win_idx is the window's position in *windows*
      window_sent_offs — parallel to window_offsets; each entry is the
                         [(sent_idx, win_start, win_end), ...] table for that
                         window (unchanged from sent_offsets_per_window)

    Walking character by character (rather than using str.join then scanning)
    means the offsets are exact and require no post-hoc search.
    """
    context_parts: list[str] = []
    window_offsets: list[tuple[int, int, int]] = []
    window_sent_offs: list[list[tuple[int, int, int]]] = []

    running_pos = 0
    sep = " "
    sep_len = len(sep)

    for i, (win_text, s_offs) in enumerate(zip(windows, sent_offsets_per_window)):
        ctx_start = running_pos
        ctx_end   = running_pos + len(win_text)
        window_offsets.append((i, ctx_start, ctx_end))
        window_sent_offs.append(s_offs)
        context_parts.append(win_text)
        running_pos = ctx_end + sep_len  # advance past the separator

    return sep.join(context_parts), window_offsets, window_sent_offs


def extract_fallback_span(question: str, context: str) -> dict:
    # Keyword-based TF-IDF fallback that returns a best-matching sentence span when
    # the main QA model is unavailable or confidence is too low. Requires no model
    # weights in memory and completes in microseconds.
    try:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context) if len(s.strip()) > 10]
        if not sentences:
            sentences = [context]
        vectorizer = TfidfVectorizer(stop_words=CUSTOM_STOPWORDS)
        tfidf = vectorizer.fit_transform([question] + sentences)
        scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        best_idx = int(scores.argmax())
        best_sent = sentences[best_idx]
        start = context.find(best_sent)
        end = start + len(best_sent) if start != -1 else len(context)
        return {
            "answer": best_sent,
            "char_start": max(0, start),
            "char_end": end,
            "confidence": round(float(scores[best_idx]), 2),
            "found": True
        }
    except Exception:
        return {"answer": context[:120], "char_start": 0, "char_end": min(120, len(context)), "confidence": 0.5, "found": True}


def get_answer(question: str, context: str) -> dict:
    if not context.strip() or not question.strip():
        return {"answer": "", "char_start": -1, "char_end": -1, "confidence": 0.0, "found": False}

    try:
        tok, model = get_model()

        # tokenizers.Tokenizer.encode() returns an Encoding object directly.
        # Unlike the transformers tokenizer called as a function, it does not add
        # a batch dimension — enc.ids, enc.offsets, enc.sequence_ids are flat lists.
        enc = tok.encode(question, context)

        # Build numpy input arrays (batch size = 1) for the ONNX session.
        inputs = {
            "input_ids":      np.array([enc.ids],           dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([enc.type_ids],       dtype=np.int64),
        }

        # enc.offsets: flat list of (char_start, char_end) tuples, one per token.
        # enc.sequence_ids: flat list of None (special), 0 (question), 1 (context).
        offset_mapping  = enc.offsets        # no [0] needed — already unbatched
        seq_ids         = enc.sequence_ids
        context_indices = [i for i, s in enumerate(seq_ids) if s == 1]
        if not context_indices:
            return extract_fallback_span(question, context)

        # Session inputs:  input_ids, attention_mask, token_type_ids  (all int64)
        # Session outputs: start_logits, end_logits                   (float32)
        # Verified via session.get_inputs()/get_outputs() against model_quantized.onnx.
        input_names = {inp.name for inp in model.get_inputs()}
        input_feed = {
            "input_ids":      inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if "token_type_ids" in input_names:
            input_feed["token_type_ids"] = inputs["token_type_ids"]

        raw_outputs = model.run(output_names=None, input_feed=input_feed)

        # Map outputs by name so the order from get_outputs() doesn't matter.
        output_names = [out.name for out in model.get_outputs()]
        output_map = dict(zip(output_names, raw_outputs))
        start_logits = output_map["start_logits"][0]
        end_logits   = output_map["end_logits"][0]

        null_score = float(start_logits[0] + end_logits[0])
        best_score = float('-inf')
        best_start, best_end = -1, -1
        start_bound = context_indices[0]
        end_bound = context_indices[-1]

        for i in range(start_bound, end_bound + 1):
            for j in range(i, min(i + 30, end_bound + 1)):
                score = float(start_logits[i] + end_logits[j])
                if score > best_score:
                    best_score = score
                    best_start, best_end = i, j

        if best_start == -1 or best_score < null_score:
            return extract_fallback_span(question, context)

        start_prob = float(_numpy_softmax(start_logits)[best_start])
        end_prob = float(_numpy_softmax(end_logits)[best_end])
        confidence = round(start_prob * end_prob, 4)

        if confidence < 0.05:
            return extract_fallback_span(question, context)

        char_start = int(offset_mapping[best_start][0])
        char_end = int(offset_mapping[best_end][1])
        raw_slice = context[char_start:char_end]

        l_strip = len(raw_slice) - len(raw_slice.lstrip())
        r_strip = len(raw_slice) - len(raw_slice.rstrip())
        char_start += l_strip
        char_end -= r_strip
        answer = context[char_start:char_end]

        return {
            "answer": answer,
            "char_start": char_start,
            "char_end": char_end,
            "confidence": confidence,
            "found": bool(answer)
        }
    except Exception:
        return extract_fallback_span(question, context)


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "PaperQA API",
        "docs": "/docs",
        "endpoints": {
            "ask": "POST /ask (multipart/form-data with file and question)",
            "health": "GET /health"
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "cached_docs": len(DOC_CACHE)}


# Register the handler under both POST "/" and POST "/ask" so the frontend can
# point to either URL without needing a path-specific configuration change.
@app.post("/")
@app.post("/ask")
def ask(file: UploadFile = File(...), question: str = Form(...)):
    try:
        if not question or not question.strip():
            return {
                "answer": "", "context": "", "char_start": -1, "char_end": -1,
                "confidence": 0.0, "found": False, "retrieval_confidence": 0.0,
            }

        file.file.seek(0)
        file_bytes = file.file.read()
        if not file_bytes:
            return {
                "answer": "", "context": "", "char_start": -1, "char_end": -1,
                "confidence": 0.0, "found": False, "retrieval_confidence": 0.0,
            }

        if len(file_bytes) > 20_000_000:
            return {
                "answer": "", "context": "", "char_start": -1, "char_end": -1,
                "confidence": 0.0, "found": False, "retrieval_confidence": 0.0,
                "error": "File too large (max 20MB).",
            }

        doc_data = extract_and_chunk_pdf(file_bytes)
        retrieval = select_context_with_meta(doc_data, question)

        # If retrieval found no passage above the relevance threshold, return
        # immediately without invoking the QA model — there is nothing to extract.
        if not retrieval["found"]:
            return {
                "answer": "", "context": "", "char_start": -1, "char_end": -1,
                "confidence": 0.0, "found": False,
                "retrieval_confidence": retrieval["retrieval_confidence"],
                "page_number": None,
            }

        context = retrieval["context"]
        qa_res = get_answer(question, context)

        # Map the QA model's char_start back to a page number via the two-level
        # offset tables built during context assembly.  Uses char_start (not
        # char_end) per spec: the page where the answer begins is the natural
        # "source page" for the user to verify.
        page_number, _sentence_index = resolve_page_number(
            char_start          = qa_res.get("char_start", -1),
            window_offsets      = retrieval["window_offsets"],
            window_sent_offsets = retrieval["window_sent_offsets"],
            sentence_meta       = doc_data.get("sentence_meta", []),
        )

        return {
            "answer":               qa_res.get("answer", ""),
            "context":              context,
            "char_start":           qa_res.get("char_start", -1),
            "char_end":             qa_res.get("char_end", -1),
            "confidence":           qa_res.get("confidence", 0.0),
            "found":                qa_res.get("found", False),
            "retrieval_confidence": retrieval["retrieval_confidence"],
            "page_number":          page_number,
        }
    except Exception as exc:
        return {
            "answer": "", "context": f"Error processing document: {str(exc)}",
            "char_start": -1, "char_end": -1, "confidence": 0.0, "found": False,
            "retrieval_confidence": 0.0, "page_number": None, "error": str(exc),
        }
