import re
import torch
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import BertForQuestionAnswering, BertTokenizer

app = FastAPI(title="Scientific Paper QA")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

embedder = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


def select_context(text, question, top_n=5):
    # Step 1: Split text into individual sentences on . ! ?
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return text  # nothing to filter, return original

    # Step 2: Embed question + all sentences
    corpus_embeddings = embedder.encode([question] + sentences)

    # Step 3: Cosine similarity - question (row 0) vs each sentence
    scores = cosine_similarity([corpus_embeddings[0]], corpus_embeddings[1:]).flatten()

    # Step 4: Pick top_n indices by highest score, restore reading order
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    top_idx = sorted(top_idx)
    return ' '.join(sentences[i] for i in top_idx)


def get_answer(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1
    return tokenizer.decode(inputs['input_ids'][0][start:end], skip_special_tokens=True)


def extract_pdf_text(file_bytes):
    reader = PdfReader(BytesIO(file_bytes))
    return ' '.join(page.extract_text() or '' for page in reader.pages)


@app.post("/ask")
async def ask(file: UploadFile, question: str = Form(...)):
    text = extract_pdf_text(await file.read())
    context = select_context(text, question)
    answer = get_answer(question, context)
    return {"answer": answer, "context": context}
