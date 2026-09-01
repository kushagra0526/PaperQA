import os
# ponytail: lock worker threads to 1 to eliminate multi-threading memory pool overhead on 512MB RAM hosts
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gc
import hashlib
import re
from io import BytesIO
import torch
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

app = FastAPI(title="Scientific Paper QA")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

QA_MODEL_NAME = "deepset/minilm-uncased-squad2"
tokenizer = None
model = None
DOC_CACHE = {}


def get_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
        model = AutoModelForQuestionAnswering.from_pretrained(
            QA_MODEL_NAME,
            low_cpu_mem_usage=True
        )
        model.eval()
        gc.collect()
    return tokenizer, model


def extract_and_chunk_pdf(file_bytes: bytes) -> dict:
    pdf_hash = hashlib.sha256(file_bytes).hexdigest()
    if pdf_hash in DOC_CACHE:
        return DOC_CACHE[pdf_hash]

    try:
        reader = PdfReader(BytesIO(file_bytes))
        raw_text = ' '.join(page.extract_text() or '' for page in reader.pages)
    except Exception:
        raw_text = ""

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', raw_text) if len(s.strip()) > 15]
    if not sentences:
        sentences = [raw_text.strip()] if raw_text.strip() else ["No text found in document."]

    doc_data = {"text": raw_text, "sentences": sentences}

    if len(DOC_CACHE) >= 10:
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
        # ponytail: TF-IDF retrieval uses ~1MB RAM and 0.001s, completely eliminating SentenceTransformers RAM spike.
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([question] + sentences)
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        top_n_count = min(top_n, len(scores))
        top_idx = sorted(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n_count])
        return ' '.join(sentences[i] for i in top_idx)
    except Exception:
        return ' '.join(sentences[:top_n])


def get_answer(question: str, context: str) -> dict:
    if not context.strip() or not question.strip():
        return {"answer": "", "char_start": -1, "char_end": -1, "confidence": 0.0, "found": False}

    tok, qa_model = get_model()

    inputs = tok(
        question,
        context,
        return_tensors='pt',
        return_offsets_mapping=True,
        truncation=True,
        max_length=512
    )

    offset_mapping = inputs['offset_mapping'][0]
    seq_ids = inputs.sequence_ids(0)
    context_indices = [i for i, s in enumerate(seq_ids) if s == 1]
    if not context_indices:
        return {"answer": "", "char_start": -1, "char_end": -1, "confidence": 0.0, "found": False}

    with torch.no_grad():
        outputs = qa_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

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

    if best_start == -1:
        return {"answer": "", "char_start": -1, "char_end": -1, "confidence": 0.0, "found": False}

    start_prob = float(torch.softmax(start_logits, dim=-1)[best_start])
    end_prob = float(torch.softmax(end_logits, dim=-1)[best_end])
    confidence = round(start_prob * end_prob, 4)

    if best_score < null_score or confidence < 0.05:
        return {"answer": "", "char_start": -1, "char_end": -1, "confidence": 0.0, "found": False}

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


# ponytail: accept both POST / and POST /ask to handle whatever URL is provided in frontend config
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
