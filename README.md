# PaperQA — PDF Question Answering System

Ask a question about a PDF paper; the answer is extracted from the paper's own text and shown
highlighted in its original context (not generated/hallucinated).

**Pipeline:** PDF text extraction (pypdf) → sentence-embedding retrieval (`sentence-transformers`,
cosine similarity) to find the most relevant sentences → BERT (`bert-large-uncased-whole-word-masking-finetuned-squad`)
extractive question answering to pull the exact answer span.

## Run the backend

```bash
cd backend
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

First run downloads the BERT model (~1.3GB) and the sentence-embedding model — needs internet access
and a few minutes.

## Run backend tests

```bash
cd backend
python test_qa.py
```

## Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Opens on `http://localhost:5173` and calls the backend at `http://localhost:8000/ask`.

## Notes

- **Extractive QA:** BERT finds a span within the retrieved text—answers are always verifiable substrings of the paper (no hallucination).
- **Exact Highlighting:** Uses fast tokenizer character offset mapping (`char_start`, `char_end`) to highlight directly in the original context without casing or subword distortion.
- **Confidence Calibration:** Computes span probabilities and evaluates against `[CLS]` null-answer logits to reject unanswerable or out-of-domain queries.
- **In-Memory Embedding Cache:** Hashes uploaded PDFs (`SHA-256`) to reuse sentence embeddings across multiple queries on the same paper.
- **Retrieval:** Uses sentence embeddings (`all-MiniLM-L6-v2`) with cosine similarity for semantic matching.
