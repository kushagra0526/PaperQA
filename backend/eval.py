#!/usr/bin/env python3
"""
Evaluation script for PaperQA backend.

Evaluates extractive QA pipeline on a benchmark JSON dataset:
- Retrieval Hit Rate: whether gold_answer substring appears in retrieved context
- Exact Match (EM): whether qa_res["answer"].strip().lower() == gold_answer.strip().lower()
- Token F1 Score: token-overlap F1 between qa_res["answer"] and gold_answer

Usage:
    python eval.py eval_set.json
"""

import argparse
import collections
import json
import os
import re
import string
import sys
from datetime import datetime
from pathlib import Path

# Import pipeline functions from main.py
from main import extract_and_chunk_pdf, select_context, get_answer


def compute_token_f1(prediction: str, gold: str) -> float:
    """
    Computes token-overlap F1 score between predicted answer and gold answer.
    """
    pred_clean = prediction.strip().lower()
    gold_clean = gold.strip().lower()

    if pred_clean == gold_clean:
        return 1.0 if pred_clean else 0.0
    if not pred_clean or not gold_clean:
        return 0.0

    # Tokenize by stripping surrounding punctuation from whitespace-separated words
    pred_tokens = [t.strip(string.punctuation) for t in pred_clean.split()]
    pred_tokens = [t for t in pred_tokens if t] or pred_clean.split()

    gold_tokens = [t.strip(string.punctuation) for t in gold_clean.split()]
    gold_tokens = [t for t in gold_tokens if t] or gold_clean.split()

    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2.0 * precision * recall) / (precision + recall)
    return f1


def resolve_paper_path(paper_path_str: str, eval_set_path: Path) -> Path:
    """
    Resolves the PDF path relative to cwd, eval_set directory, backend, or project root.
    """
    p = Path(paper_path_str)
    if p.is_file():
        return p

    # Relative to eval_set file
    candidate = eval_set_path.parent / paper_path_str
    if candidate.is_file():
        return candidate

    # Relative to backend directory
    backend_dir = Path(__file__).resolve().parent
    candidate = backend_dir / paper_path_str
    if candidate.is_file():
        return candidate

    # Relative to project root
    project_root = backend_dir.parent
    candidate = project_root / paper_path_str
    if candidate.is_file():
        return candidate

    return p


def evaluate_dataset(eval_set_path: Path, results_dir: Path) -> dict:
    if not eval_set_path.is_file():
        print(f"Error: Evaluation set file not found: {eval_set_path}")
        print("\nExpected a JSON file formatted as:")
        print('[\n  {\n    "paper_path": "papers/sample1.pdf",\n    "question": "What is ...?",\n    "gold_answer": "...",\n    "gold_page": 4\n  }\n]')
        sys.exit(1)

    with open(eval_set_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Error reading JSON from {eval_set_path}: {e}")
            sys.exit(1)

    if not isinstance(data, list):
        print(f"Error: Expected a JSON list of objects in {eval_set_path}, got {type(data).__name__}")
        sys.exit(1)

    total_samples = len(data)
    print(f"\nLoaded {total_samples} test case(s) from {eval_set_path}")
    print("=" * 80)

    results = []
    total_hit = 0
    total_em = 0
    total_f1 = 0.0

    for idx, item in enumerate(data, start=1):
        paper_path_raw = item.get("paper_path", "")
        question = item.get("question", "")
        gold_answer = item.get("gold_answer", "")
        gold_page = item.get("gold_page")

        resolved_paper = resolve_paper_path(paper_path_raw, eval_set_path)

        record = {
            "id": idx,
            "paper_path": paper_path_raw,
            "question": question,
            "gold_answer": gold_answer,
            "gold_page": gold_page,
            "predicted_answer": "",
            "confidence": 0.0,
            "retrieval_hit": False,
            "exact_match": False,
            "f1": 0.0,
            "retrieved_context": "",
            "error": None
        }

        if not resolved_paper.is_file():
            record["error"] = f"File not found: {paper_path_raw}"
            print(f"[{idx}/{total_samples}] FAIL - PDF file not found: {paper_path_raw}")
            results.append(record)
            continue

        try:
            with open(resolved_paper, "rb") as f:
                pdf_bytes = f.read()

            doc_data = extract_and_chunk_pdf(pdf_bytes)
            context = select_context(doc_data, question)
            qa_res = get_answer(question, context)

            pred_answer = qa_res.get("answer", "")
            confidence = qa_res.get("confidence", 0.0)

            # 1. Retrieval hit: gold_answer substring appears anywhere in retrieved context
            gold_clean = str(gold_answer).strip()
            retrieval_hit = bool(gold_clean and gold_clean.lower() in context.lower())

            # 2. Exact match: does qa_res["answer"].strip().lower() == gold_answer.strip().lower()
            exact_match = bool(pred_answer.strip().lower() == gold_clean.lower())

            # 3. Simple token-overlap F1
            f1_score = compute_token_f1(pred_answer, gold_answer)

            record.update({
                "predicted_answer": pred_answer,
                "confidence": confidence,
                "retrieval_hit": retrieval_hit,
                "exact_match": exact_match,
                "f1": round(f1_score, 4),
                "retrieved_context": context
            })

            if retrieval_hit:
                total_hit += 1
            if exact_match:
                total_em += 1
            total_f1 += f1_score

            hit_str = "YES" if retrieval_hit else "NO"
            em_str = "YES" if exact_match else "NO"
            print(f"[{idx}/{total_samples}] Q: {question[:60]}{'...' if len(question) > 60 else ''}")
            print(f"      Gold: {gold_answer}")
            print(f"      Pred: {pred_answer} (conf: {confidence})")
            print(f"      Hit: {hit_str} | EM: {em_str} | F1: {f1_score:.4f}\n")

        except Exception as exc:
            record["error"] = str(exc)
            print(f"[{idx}/{total_samples}] ERROR processing entry: {exc}")

        results.append(record)

    hit_rate = (total_hit / total_samples) if total_samples > 0 else 0.0
    avg_em = (total_em / total_samples) if total_samples > 0 else 0.0
    avg_f1 = (total_f1 / total_samples) if total_samples > 0 else 0.0

    summary = {
        "total_samples": total_samples,
        "retrieval_hits": total_hit,
        "retrieval_hit_rate": round(hit_rate, 4),
        "exact_matches": total_em,
        "average_em": round(avg_em, 4),
        "average_f1": round(avg_f1, 4)
    }

    # Print summary table
    print("=" * 80)
    print(f"{'EVALUATION SUMMARY':^80}")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Score':<10} | {'Percentage / Count':<30}")
    print("-" * 80)
    print(f"{'Total Samples':<30} | {total_samples:<10} | {total_samples} samples")
    print(f"{'Retrieval Hit Rate':<30} | {hit_rate:.4f}     | {hit_rate * 100:.1f}% ({total_hit}/{total_samples})")
    print(f"{'Exact Match (EM)':<30} | {avg_em:.4f}     | {avg_em * 100:.1f}% ({total_em}/{total_samples})")
    print(f"{'Average F1':<30} | {avg_f1:.4f}     | {avg_f1 * 100:.1f}%")
    print("=" * 80)

    # Save to results/run_<timestamp>.json
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = results_dir / f"run_{timestamp}.json"

    run_payload = {
        "run_id": f"run_{timestamp}",
        "timestamp": datetime.now().isoformat(),
        "eval_set_path": str(eval_set_path),
        "summary": summary,
        "results": results
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(run_payload, f, indent=2)

    print(f"Results saved to: {out_file}\n")
    return run_payload


def main():
    parser = argparse.ArgumentParser(description="Evaluate PaperQA pipeline against gold-standard evaluation set.")
    parser.add_argument(
        "eval_set",
        nargs="?",
        default="eval_set.json",
        help="Path to evaluation JSON file (default: eval_set.json)"
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory where run results JSON will be written (default: results)"
    )
    args = parser.parse_args()

    eval_set_path = Path(args.eval_set)
    results_dir = Path(args.results_dir)

    evaluate_dataset(eval_set_path, results_dir)


if __name__ == "__main__":
    main()
