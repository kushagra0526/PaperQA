import hashlib
import re
from io import BytesIO
import torch
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, BertForQuestionAnswering

app = FastAPI(title="Scientific Paper QA")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model.to(device)
model.eval()

# ponytail: in-memory dict cache keyed by document SHA-256. Bounded process memory, no persistence.
# Upgrade path: disk-backed SQLite or vector store (Chroma/Qdrant) for multi-worker setups.
DOC_CACHE = {}


def extract_and_chunk_pdf(file_bytes: bytes) -> dict:
    pdf_hash = hashlib.sha256(file_bytes).hexdigest()
    if pdf_hash in DOC_CACHE:
        return DOC_CACHE[pdf_hash]

    try:
        reader = PdfReader(BytesIO(file_bytes))
        raw_text = ' '.join(page.extract_text() or '' for page in reader.pages)
    except Exception:
        raw_text = ""

    # ponytail: regex sentence boundary checking capitalized words.
    # Upgrade path: layout-aware parser (docling/marker) for multi-column academic PDFs.
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', raw_text) if len(s.strip()) > 15]
    if not sentences:
        sentences = [raw_text.strip()] if raw_text.strip() else ["No text found in document."]

    embeddings = embedder.encode(sentences, show_progress_bar=False, normalize_embeddings=True)
    doc_data = {"text": raw_text, "sentences": sentences, "embeddings": embeddings}

    if len(DOC_CACHE) >= 20:
        DOC_CACHE.pop(next(iter(DOC_CACHE)))
    DOC_CACHE[pdf_hash] = doc_data
    return doc_data


def select_context(doc_data: dict, question: str, top_n: int = 5) -> str:
    sentences = doc_data.get("sentences", [])
    embeddings = doc_data.get("embeddings")
    if not sentences or embeddings is None or len(embeddings) == 0:
        return ""
    q_emb = embedder.encode([question], show_progress_bar=False, normalize_embeddings=True)
    scores = cosine_similarity(q_emb, embeddings).flatten()
    top_n_count = min(top_n, len(scores))
    top_idx = sorted(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n_count])
    return ' '.join(sentences[i] for i in top_idx)


def get_answer(question: str, context: str) -> dict:
    if not context.strip() or not question.strip():
        return {"answer": "", "char_start": -1, "char_end": -1, "confidence": 0.0, "found": False}

    inputs = tokenizer(
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
        outputs = model(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device)
        )

    start_logits = outputs.start_logits[0].cpu()
    end_logits = outputs.end_logits[0].cpu()

    # Null-answer score from [CLS] token (SQuAD null representation)
    null_score = float(start_logits[0] + end_logits[0])

    best_score = float('-inf')
    best_start, best_end = -1, -1
    start_bound = context_indices[0]
    end_bound = context_indices[-1]

    # Fast contiguous span evaluation (max span length = 30 tokens)
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

    # Rejection threshold for unanswerable / out-of-domain queries
    if best_score < null_score or confidence < 0.05:
        return {"answer": "", "char_start": -1, "char_end": -1, "confidence": 0.0, "found": False}

    char_start = int(offset_mapping[best_start][0])
    char_end = int(offset_mapping[best_end][1])
    raw_slice = context[char_start:char_end]

    # Clean whitespace boundaries while keeping offsets exact
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


@app.get("/health")
def health():
    return {"status": "ok", "device": device, "cached_docs": len(DOC_CACHE)}


# ponytail: synchronous def so FastAPI runs CPU/PyTorch workloads in worker threads.
@app.post("/ask")
def ask(file: UploadFile, question: str = Form(...)):
    if not question.strip():
        return {"answer": "", "context": "", "char_start": -1, "char_end": -1, "confidence": 0.0, "found": False}

    file.file.seek(0)
    file_bytes = file.file.read()
    if not file_bytes:
        return {"answer": "", "context": "", "char_start": -1, "char_end": -1, "confidence": 0.0, "found": False}

    doc_data = extract_and_chunk_pdf(file_bytes)
    context = select_context(doc_data, question)
    qa_res = get_answer(question, context)
    return {
        "answer": qa_res["answer"],
        "context": context,
        "char_start": qa_res["char_start"],
        "char_end": qa_res["char_end"],
        "confidence": qa_res["confidence"],
        "found": qa_res["found"]
    }
