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

## Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Opens on `http://localhost:5173` and calls the backend at `http://localhost:8000/ask`.

## Notes

- This is extractive QA (BERT finds a span within the retrieved text), not a generative LLM —
  answers are always verifiable substrings of the paper, which is the point: no hallucination.
- Retrieval uses sentence embeddings (`all-MiniLM-L6-v2`) rather than TF-IDF for better semantic matching
  on paraphrased questions.
