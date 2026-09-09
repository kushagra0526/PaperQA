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
# Local directory where preload.py saves the ONNX-exported, INT8-quantized model
# during the build step.  At runtime, get_model() loads from here if available.
ONNX_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_model")

# When True, extract_and_chunk_pdf uses the layout-aware PyMuPDF path
# (extract_text_layout_aware) which handles two-column papers correctly.
# Set to False to fall back to the original pypdf-based extraction.
USE_LAYOUT_AWARE_EXTRACTION = True

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


def select_context(doc_data: dict, question: str, top_n: int = 5) -> str:
    sentences = doc_data.get("sentences", [])
    if not sentences:
        return ""
    if len(sentences) <= top_n:
        return ' '.join(sentences)

    try:
        # TF-IDF is used here instead of a neural embedding model because it needs
        # only ~1MB of RAM and runs in under 1ms, avoiding the multi-hundred-MB
        # memory spike that sentence-transformers would cause on memory-constrained hosts.
        vectorizer = TfidfVectorizer(stop_words=CUSTOM_STOPWORDS)
        tfidf_matrix = vectorizer.fit_transform([question] + sentences)
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        top_n_count = min(top_n, len(scores))
        top_idx = sorted(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n_count])
        return ' '.join(sentences[i] for i in top_idx)
    except Exception:
        return ' '.join(sentences[:top_n])


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
            return {"answer": "", "context": "", "char_start": -1, "char_end": -1, "confidence": 0.0, "found": False}

        file.file.seek(0)
        file_bytes = file.file.read()
        if not file_bytes:
            return {"answer": "", "context": "", "char_start": -1, "char_end": -1, "confidence": 0.0, "found": False}

        if len(file_bytes) > 20_000_000:
            return {
                "answer": "",
                "context": "",
                "char_start": -1,
                "char_end": -1,
                "confidence": 0.0,
                "found": False,
                "error": "File too large (max 20MB)."
            }

        doc_data = extract_and_chunk_pdf(file_bytes)
        context = select_context(doc_data, question)
        qa_res = get_answer(question, context)
        return {
            "answer": qa_res.get("answer", ""),
            "context": context,
            "char_start": qa_res.get("char_start", -1),
            "char_end": qa_res.get("char_end", -1),
            "confidence": qa_res.get("confidence", 0.0),
            "found": qa_res.get("found", False)
        }
    except Exception as exc:
        return {
            "answer": "",
            "context": f"Error processing document: {str(exc)}",
            "char_start": -1,
            "char_end": -1,
            "confidence": 0.0,
            "found": False,
            "error": str(exc)
        }
