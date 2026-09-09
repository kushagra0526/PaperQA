"""
QA model loading and inference.

Public API:
  get_model()            -> (Tokenizer, ort.InferenceSession)
  get_reranker()         -> (Tokenizer, ort.InferenceSession)
  get_answer(q, ctx)     -> dict
  extract_fallback_span(q, ctx) -> dict

Model state is module-level so it survives across requests.  Both the QA
model and the reranker use the same double-checked-locking lazy-load pattern
and the same idle-unload daemon (evict after 10 min of inactivity).

The reranker is never loaded when USE_RERANKER is False — every code path
that calls get_reranker() is gated on that flag in retrieval.py.
"""

from __future__ import annotations

import gc
import os
import re
import threading
import time

import numpy as np
import onnxruntime as ort
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tokenizers import Tokenizer

# ---------------------------------------------------------------------------
# Paths and model names
# ---------------------------------------------------------------------------

QA_MODEL_NAME       = "deepset/minilm-uncased-squad2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_BACKEND_DIR    = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL_DIR    = os.path.join(_BACKEND_DIR, "onnx_model")
ONNX_RERANKER_DIR = os.path.join(_BACKEND_DIR, "onnx_reranker")

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_MODEL_IDLE_TIMEOUT = 600   # seconds — evict after 10 min of inactivity

# Preserve negation terms stripped by sklearn's built-in stopword list so that
# questions like "What does NOT apply?" still retrieve correctly.
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
_NEGATIONS = {"not", "no", "none", "cannot", "never"}
CUSTOM_STOPWORDS = list(ENGLISH_STOP_WORDS - _NEGATIONS)

# ---------------------------------------------------------------------------
# QA model state
# ---------------------------------------------------------------------------

tokenizer = None       # tokenizers.Tokenizer
model     = None       # ort.InferenceSession
_last_model_use = 0.0
_model_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Reranker model state
# ---------------------------------------------------------------------------

_reranker_tokenizer = None   # tokenizers.Tokenizer
_reranker_session   = None   # ort.InferenceSession
_last_reranker_use  = 0.0
_reranker_lock      = threading.Lock()


# ---------------------------------------------------------------------------
# QA model — lazy loader
# ---------------------------------------------------------------------------

def get_model() -> tuple:
    """
    Lazy-load the QA tokenizer and ONNX session under a double-checked lock.
    Returns (tokenizer, session).  Safe to call from multiple threads.
    """
    global tokenizer, model, _last_model_use
    if tokenizer is None or model is None:
        with _model_lock:
            if tokenizer is None or model is None:
                sess_opts = ort.SessionOptions()
                sess_opts.intra_op_num_threads = 1
                sess_opts.inter_op_num_threads = 1

                if os.path.isdir(ONNX_MODEL_DIR):
                    # Hot path: pre-exported quantized model from preload.py.
                    _tok_path = os.path.join(ONNX_MODEL_DIR, "tokenizer.json")
                    tokenizer = Tokenizer.from_file(_tok_path)
                    tokenizer.enable_truncation(max_length=512)
                    model = ort.InferenceSession(
                        os.path.join(ONNX_MODEL_DIR, "model_quantized.onnx"),
                        sess_options=sess_opts,
                        providers=["CPUExecutionProvider"],
                    )
                else:
                    # Dev fallback: export + quantize on-the-fly.
                    # optimum/transformers are imported here (not at module level)
                    # so they never appear in the runtime process on a normal deploy.
                    from optimum.onnxruntime import ORTModelForQuestionAnswering, ORTQuantizer
                    from optimum.onnxruntime.configuration import AutoQuantizationConfig
                    from transformers import AutoTokenizer as _AutoTok
                    from tempfile import TemporaryDirectory

                    with TemporaryDirectory() as tmp:
                        _m = ORTModelForQuestionAnswering.from_pretrained(
                            QA_MODEL_NAME, export=True
                        )
                        _m.save_pretrained(tmp)
                        _q = ORTQuantizer.from_pretrained(tmp)
                        _q.quantize(
                            save_dir=ONNX_MODEL_DIR,
                            quantization_config=AutoQuantizationConfig.avx2(
                                is_static=False, per_channel=False
                            ),
                        )
                    _AutoTok.from_pretrained(QA_MODEL_NAME).save_pretrained(ONNX_MODEL_DIR)
                    _tok_path = os.path.join(ONNX_MODEL_DIR, "tokenizer.json")
                    tokenizer = Tokenizer.from_file(_tok_path)
                    tokenizer.enable_truncation(max_length=512)
                    model = ort.InferenceSession(
                        os.path.join(ONNX_MODEL_DIR, "model_quantized.onnx"),
                        sess_options=sess_opts,
                        providers=["CPUExecutionProvider"],
                    )
                gc.collect()
    _last_model_use = time.time()
    return tokenizer, model


def _idle_unload_loop() -> None:
    """Evict the QA model after _MODEL_IDLE_TIMEOUT seconds of inactivity."""
    global tokenizer, model
    while True:
        time.sleep(120)
        if model is not None and (time.time() - _last_model_use) > _MODEL_IDLE_TIMEOUT:
            with _model_lock:
                if model is not None and (time.time() - _last_model_use) > _MODEL_IDLE_TIMEOUT:
                    tokenizer = None
                    model = None
                    gc.collect()


# Warm up the QA model immediately so the first real request has no cold-start.
threading.Thread(target=get_model, daemon=True).start()
threading.Thread(target=_idle_unload_loop, daemon=True).start()


# ---------------------------------------------------------------------------
# Reranker — lazy loader
# ---------------------------------------------------------------------------

def get_reranker() -> tuple:
    """
    Lazy-load the cross-encoder reranker under a double-checked lock.
    Returns (tokenizer, session).

    Only called when USE_RERANKER is True (enforced in retrieval.py) —
    the reranker is never loaded when that flag is False.
    """
    global _reranker_tokenizer, _reranker_session, _last_reranker_use
    if _reranker_tokenizer is None or _reranker_session is None:
        with _reranker_lock:
            if _reranker_tokenizer is None or _reranker_session is None:
                sess_opts = ort.SessionOptions()
                sess_opts.intra_op_num_threads = 1
                sess_opts.inter_op_num_threads = 1

                if os.path.isdir(ONNX_RERANKER_DIR):
                    _tok_path = os.path.join(ONNX_RERANKER_DIR, "tokenizer.json")
                    _reranker_tokenizer = Tokenizer.from_file(_tok_path)
                    _reranker_tokenizer.enable_truncation(max_length=512)
                    _reranker_session = ort.InferenceSession(
                        os.path.join(ONNX_RERANKER_DIR, "model_quantized.onnx"),
                        sess_opts=sess_opts,
                        providers=["CPUExecutionProvider"],
                    )
                else:
                    from optimum.onnxruntime import (
                        ORTModelForSequenceClassification,
                        ORTQuantizer,
                    )
                    from optimum.onnxruntime.configuration import AutoQuantizationConfig
                    from transformers import AutoTokenizer as _AutoTok
                    from tempfile import TemporaryDirectory

                    with TemporaryDirectory() as tmp:
                        _m = ORTModelForSequenceClassification.from_pretrained(
                            RERANKER_MODEL_NAME, export=True
                        )
                        _m.save_pretrained(tmp)
                        _q = ORTQuantizer.from_pretrained(tmp)
                        _q.quantize(
                            save_dir=ONNX_RERANKER_DIR,
                            quantization_config=AutoQuantizationConfig.avx2(
                                is_static=False, per_channel=False
                            ),
                        )
                    _AutoTok.from_pretrained(RERANKER_MODEL_NAME).save_pretrained(
                        ONNX_RERANKER_DIR
                    )
                    _tok_path = os.path.join(ONNX_RERANKER_DIR, "tokenizer.json")
                    _reranker_tokenizer = Tokenizer.from_file(_tok_path)
                    _reranker_tokenizer.enable_truncation(max_length=512)
                    _reranker_session = ort.InferenceSession(
                        os.path.join(ONNX_RERANKER_DIR, "model_quantized.onnx"),
                        sess_opts=sess_opts,
                        providers=["CPUExecutionProvider"],
                    )
                gc.collect()
    _last_reranker_use = time.time()
    return _reranker_tokenizer, _reranker_session


def _reranker_idle_unload_loop() -> None:
    """Evict the reranker after _MODEL_IDLE_TIMEOUT seconds of inactivity."""
    global _reranker_tokenizer, _reranker_session
    while True:
        time.sleep(120)
        if _reranker_session is not None and (
            time.time() - _last_reranker_use
        ) > _MODEL_IDLE_TIMEOUT:
            with _reranker_lock:
                if _reranker_session is not None and (
                    time.time() - _last_reranker_use
                ) > _MODEL_IDLE_TIMEOUT:
                    _reranker_tokenizer = None
                    _reranker_session   = None
                    gc.collect()


threading.Thread(target=_reranker_idle_unload_loop, daemon=True).start()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _numpy_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D numpy array."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def extract_fallback_span(question: str, context: str) -> dict:
    """
    TF-IDF keyword fallback used when the QA model is unavailable or returns
    a span below the confidence threshold.  Requires no model weights and
    completes in microseconds.
    """
    try:
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", context)
            if len(s.strip()) > 10
        ]
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
            "answer":     best_sent,
            "char_start": max(0, start),
            "char_end":   end,
            "confidence": round(float(scores[best_idx]), 2),
            "found":      True,
        }
    except Exception:
        return {
            "answer":     context[:120],
            "char_start": 0,
            "char_end":   min(120, len(context)),
            "confidence": 0.5,
            "found":      True,
        }


def get_answer(question: str, context: str) -> dict:
    """
    Run extractive QA over *context* for *question*.

    Returns a dict with keys: answer, char_start, char_end, confidence, found.
    Falls back to extract_fallback_span on model rejection or low confidence.
    """
    if not context.strip() or not question.strip():
        return {
            "answer": "", "char_start": -1, "char_end": -1,
            "confidence": 0.0, "found": False,
        }

    try:
        tok, sess = get_model()

        # tokenizers.Tokenizer.encode() returns an Encoding with no batch dim.
        enc = tok.encode(question, context)

        inputs = {
            "input_ids":      np.array([enc.ids],           dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([enc.type_ids],       dtype=np.int64),
        }

        # enc.offsets: flat list of (char_start, char_end) tuples per token.
        # enc.sequence_ids: flat list of None (special) / 0 (question) / 1 (context).
        offset_mapping  = enc.offsets
        seq_ids         = enc.sequence_ids
        context_indices = [i for i, s in enumerate(seq_ids) if s == 1]
        if not context_indices:
            return extract_fallback_span(question, context)

        # Session inputs:  input_ids, attention_mask, token_type_ids (int64)
        # Session outputs: start_logits, end_logits                  (float32)
        input_names = {inp.name for inp in sess.get_inputs()}
        input_feed: dict = {
            "input_ids":      inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if "token_type_ids" in input_names:
            input_feed["token_type_ids"] = inputs["token_type_ids"]

        raw_outputs  = sess.run(output_names=None, input_feed=input_feed)
        output_names = [out.name for out in sess.get_outputs()]
        output_map   = dict(zip(output_names, raw_outputs))
        start_logits = output_map["start_logits"][0]
        end_logits   = output_map["end_logits"][0]

        null_score  = float(start_logits[0] + end_logits[0])
        best_score  = float("-inf")
        best_start  = best_end = -1
        start_bound = context_indices[0]
        end_bound   = context_indices[-1]

        for i in range(start_bound, end_bound + 1):
            for j in range(i, min(i + 30, end_bound + 1)):
                score = float(start_logits[i] + end_logits[j])
                if score > best_score:
                    best_score = score
                    best_start, best_end = i, j

        if best_start == -1 or best_score < null_score:
            return extract_fallback_span(question, context)

        start_prob = float(_numpy_softmax(start_logits)[best_start])
        end_prob   = float(_numpy_softmax(end_logits)[best_end])
        confidence = round(start_prob * end_prob, 4)

        if confidence < 0.05:
            return extract_fallback_span(question, context)

        char_start = int(offset_mapping[best_start][0])
        char_end   = int(offset_mapping[best_end][1])
        raw_slice  = context[char_start:char_end]

        l_strip = len(raw_slice) - len(raw_slice.lstrip())
        r_strip = len(raw_slice) - len(raw_slice.rstrip())
        char_start += l_strip
        char_end   -= r_strip
        answer = context[char_start:char_end]

        return {
            "answer":     answer,
            "char_start": char_start,
            "char_end":   char_end,
            "confidence": confidence,
            "found":      bool(answer),
        }
    except Exception:
        return extract_fallback_span(question, context)
