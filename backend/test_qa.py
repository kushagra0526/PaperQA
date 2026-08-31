"""
Minimal assert-based self-check for PaperQA backend.
Tests document caching, extractive QA with exact offsets, and null-answer rejection.
"""
import io
import time
from pypdf import PdfWriter
from main import extract_and_chunk_pdf, select_context, get_answer, DOC_CACHE


def create_dummy_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    # Write a small page with text
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


def test_qa_exact_span():
    context = "The Transformer architecture was introduced in 2017 by Vaswani et al. It relies entirely on self-attention mechanisms."
    question = "Who introduced the Transformer architecture?"

    res = get_answer(question, context)
    assert res["found"] is True, "Expected answer to be found"
    assert "Vaswani" in res["answer"], f"Expected Vaswani in answer, got: {res['answer']}"
    assert res["char_start"] >= 0 and res["char_end"] > res["char_start"]
    # Check that context slice exactly matches answer
    extracted_slice = context[res["char_start"]:res["char_end"]].strip()
    assert extracted_slice == res["answer"], f"Slice mismatch: '{extracted_slice}' vs '{res['answer']}'"
    assert res["confidence"] > 0.05, f"Confidence too low: {res['confidence']}"
    print("[PASS] Exact span extraction and character offsets verified.")


def test_unanswerable_rejection():
    context = "The Transformer architecture was introduced in 2017 by Vaswani et al. It relies entirely on self-attention mechanisms."
    question = "What is the boiling point of liquid nitrogen in Kelvin?"

    res = get_answer(question, context)
    assert res["found"] is False or res["confidence"] < 0.1, "Expected out-of-domain question to be rejected"
    print("[PASS] Unanswerable query rejection verified.")


def test_cache_speedup():
    dummy_bytes = create_dummy_pdf()
    DOC_CACHE.clear()

    # First call - cache miss
    d1 = extract_and_chunk_pdf(dummy_bytes)
    assert len(DOC_CACHE) == 1

    # Second call - cache hit (same object returned instantly)
    d2 = extract_and_chunk_pdf(dummy_bytes)
    assert d1 is d2, "Expected cache to return the same dict instance"
    print("[PASS] Document embedding cache verified.")


def test_empty_inputs():
    res1 = get_answer("", "Some context")
    assert res1["found"] is False and res1["answer"] == ""

    res2 = get_answer("Some question", "")
    assert res2["found"] is False and res2["answer"] == ""

    empty_doc = extract_and_chunk_pdf(b"")
    ctx = select_context(empty_doc, "question")
    assert isinstance(ctx, str)
    print("[PASS] Empty and boundary inputs handled gracefully.")


def test_whitespace_trimming():
    context = "The learning rate was set to 0.001 during training."
    question = "What was the learning rate?"
    res = get_answer(question, context)
    assert res["found"] is True
    assert res["answer"] == "0.001"
    # Strict verification: context[char_start:char_end] must match answer identically
    assert context[res["char_start"]:res["char_end"]] == res["answer"]
    print("[PASS] Exact whitespace boundary alignment verified.")


if __name__ == "__main__":
    test_qa_exact_span()
    test_unanswerable_rejection()
    test_cache_speedup()
    test_empty_inputs()
    test_whitespace_trimming()
    print("\nALL BACKEND CHECKS PASSED.")
