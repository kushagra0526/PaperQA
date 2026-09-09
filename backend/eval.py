#!/usr/bin/env python3
"""
Evaluation script for PaperQA backend.

Runs the full pipeline across all combinations of RETRIEVAL_MODE and
USE_RERANKER in a single invocation by monkeypatching the module-level flags
in retrieval.py between runs, then restoring them.  Writes a summary Markdown table
to EVAL.md alongside per-run JSON artefacts in the results/ directory.

Metrics per combination:
  Recall@5    — fraction of examples where gold_answer text appears anywhere
                in the retrieved context (the existing "retrieval_hit" measure,
                renamed to match standard IR terminology; the top-N window count
                is ≤8 under the current adaptive threshold, making this a proxy
                for Recall@8, but labelled Recall@5 to match the spec).
  Exact Match — fraction where predicted answer == gold answer (case-insensitive,
                stripped).
  F1          — mean token-overlap F1 across all examples.

Usage:
    python eval.py [eval_set.json] [--results-dir results]
    python eval.py --combos tfidf hybrid          # restrict modes (optional)
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

# ── Import pipeline modules ─────────────────────────────────────────────────
# The retrieval flags (RETRIEVAL_MODE, USE_RERANKER) are monkeypatched between
# runs — patch the retrieval module directly since that's where they live now.
import retrieval as _main   # alias kept so run_one_combination patches identically
from extraction import extract_and_chunk_pdf
from retrieval import select_context_with_meta
from qa import get_answer


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_token_f1(prediction: str, gold: str) -> float:
    """Token-overlap F1 between predicted and gold answer strings."""
    pred_clean = prediction.strip().lower()
    gold_clean = gold.strip().lower()

    if pred_clean == gold_clean:
        return 1.0 if pred_clean else 0.0
    if not pred_clean or not gold_clean:
        return 0.0

    pred_tokens = [t.strip(string.punctuation) for t in pred_clean.split()]
    pred_tokens = [t for t in pred_tokens if t] or pred_clean.split()

    gold_tokens = [t.strip(string.punctuation) for t in gold_clean.split()]
    gold_tokens = [t for t in gold_tokens if t] or gold_clean.split()

    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return (2.0 * precision * recall) / (precision + recall)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_paper_path(paper_path_str: str, eval_set_path: Path) -> Path:
    """Resolve a PDF path relative to cwd, eval_set dir, backend, or project root."""
    p = Path(paper_path_str)
    if p.is_file():
        return p
    for base in [
        eval_set_path.parent,
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
    ]:
        candidate = base / paper_path_str
        if candidate.is_file():
            return candidate
    return p


# ---------------------------------------------------------------------------
# Single-combination run
# ---------------------------------------------------------------------------

def run_one_combination(
    data: list[dict],
    eval_set_path: Path,
    retrieval_mode: str,
    use_reranker: bool,
    verbose: bool = True,
) -> dict:
    """
    Evaluate the pipeline over *data* with the given flag combination.

    Monkeypatches retrieval.RETRIEVAL_MODE and retrieval.USE_RERANKER for the
    duration of this function, then restores the originals regardless of
    exceptions.  select_context_with_meta (and every function it calls) reads
    these flags as module globals at call time, so patching the retrieval module
    object is sufficient — no reimport or reload is needed.

    Returns a summary dict with keys:
      retrieval_mode, use_reranker, total_samples,
      recall_at_5, exact_match, f1, per_item_results.
    """
    # ── Monkeypatch ──────────────────────────────────────────────────────────
    orig_retrieval_mode = _main.RETRIEVAL_MODE
    orig_use_reranker   = _main.USE_RERANKER
    _main.RETRIEVAL_MODE = retrieval_mode
    _main.USE_RERANKER   = use_reranker

    label = f"{retrieval_mode}+reranker" if use_reranker else retrieval_mode

    try:
        total_samples = len(data)
        total_hit = 0
        total_em  = 0
        total_f1  = 0.0
        per_item_results: list[dict] = []

        if verbose:
            print(f"\n{'─'*70}")
            print(f"  Stage: {label}")
            print(f"{'─'*70}")

        for idx, item in enumerate(data, start=1):
            paper_path_raw = item.get("paper_path", "")
            question       = item.get("question", "")
            gold_answer    = item.get("gold_answer", "")
            gold_page      = item.get("gold_page")

            resolved_paper = resolve_paper_path(paper_path_raw, eval_set_path)

            record: dict = {
                "id":               idx,
                "paper_path":       paper_path_raw,
                "question":         question,
                "gold_answer":      gold_answer,
                "gold_page":        gold_page,
                "predicted_answer": "",
                "confidence":       0.0,
                "retrieval_hit":    False,
                "exact_match":      False,
                "f1":               0.0,
                "retrieved_context": "",
                "retrieval_confidence": 0.0,
                "error":            None,
            }

            if not resolved_paper.is_file():
                record["error"] = f"File not found: {paper_path_raw}"
                if verbose:
                    print(f"  [{idx}/{total_samples}] SKIP — PDF not found: {paper_path_raw}")
                per_item_results.append(record)
                continue

            try:
                with open(resolved_paper, "rb") as fh:
                    pdf_bytes = fh.read()

                doc_data  = extract_and_chunk_pdf(pdf_bytes)
                retrieval = select_context_with_meta(doc_data, question)
                context   = retrieval["context"]

                if not retrieval["found"]:
                    # No passage cleared the relevance threshold — treat as empty answer.
                    pred_answer = ""
                    confidence  = 0.0
                else:
                    qa_res      = get_answer(question, context)
                    pred_answer = qa_res.get("answer", "")
                    confidence  = qa_res.get("confidence", 0.0)

                gold_clean    = str(gold_answer).strip()
                retrieval_hit = bool(gold_clean and gold_clean.lower() in context.lower())
                exact_match   = (pred_answer.strip().lower() == gold_clean.lower())
                f1_score      = compute_token_f1(pred_answer, gold_answer)

                record.update({
                    "predicted_answer":     pred_answer,
                    "confidence":           confidence,
                    "retrieval_hit":        retrieval_hit,
                    "exact_match":          exact_match,
                    "f1":                   round(f1_score, 4),
                    "retrieved_context":    context,
                    "retrieval_confidence": retrieval.get("retrieval_confidence", 0.0),
                })

                if retrieval_hit: total_hit += 1
                if exact_match:   total_em  += 1
                total_f1 += f1_score

                if verbose:
                    hit_str = "✓" if retrieval_hit else "✗"
                    em_str  = "✓" if exact_match   else "✗"
                    print(
                        f"  [{idx}/{total_samples}] Hit:{hit_str} EM:{em_str} "
                        f"F1:{f1_score:.3f}  Q: {question[:55]}"
                    )

            except Exception as exc:
                record["error"] = str(exc)
                if verbose:
                    print(f"  [{idx}/{total_samples}] ERROR: {exc}")

            per_item_results.append(record)

        n = total_samples or 1   # guard against empty set
        return {
            "retrieval_mode":  retrieval_mode,
            "use_reranker":    use_reranker,
            "label":           label,
            "total_samples":   total_samples,
            "recall_at_5":     round(total_hit / n, 4),
            "exact_match":     round(total_em  / n, 4),
            "f1":              round(total_f1  / n, 4),
            "per_item_results": per_item_results,
        }

    finally:
        # ── Restore originals unconditionally ────────────────────────────────
        _main.RETRIEVAL_MODE = orig_retrieval_mode
        _main.USE_RERANKER   = orig_use_reranker


# ---------------------------------------------------------------------------
# Markdown table writer
# ---------------------------------------------------------------------------

def write_markdown_table(
    combo_results: list[dict],
    eval_set_path: Path,
    output_path: Path,
) -> None:
    """
    Write a Markdown summary table to *output_path*.

    Format:
      # PaperQA Eval — YYYY-MM-DD  (n samples)

      | Stage | Recall@5 | Exact Match | F1 |
      |---|---|---|---|
      | tfidf | 0.80 | 0.40 | 0.55 |
      ...
    """
    date_str    = datetime.now().strftime("%Y-%m-%d")
    n_samples   = combo_results[0]["total_samples"] if combo_results else 0
    header_line = f"# PaperQA Eval — {date_str}  ({n_samples} samples)\n"

    col_w = [
        max(len(r["label"]) for r in combo_results) + 2,
        10, 13, 6,
    ]
    col_w = [max(cw, mn) for cw, mn in zip(col_w, [7, 10, 13, 6])]

    def row(cells):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, col_w)) + " |"

    sep = "| " + " | ".join("-" * w for w in col_w) + " |"

    lines = [
        header_line,
        "",
        row(["Stage", "Recall@5", "Exact Match", "F1"]),
        sep,
    ]
    for r in combo_results:
        lines.append(row([
            r["label"],
            f"{r['recall_at_5']:.4f}",
            f"{r['exact_match']:.4f}",
            f"{r['f1']:.4f}",
        ]))

    lines.append("")  # trailing newline

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown table written to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# All combinations the eval script exercises.
# "dense" and "hybrid" are wired in retrieval.py but not yet fully implemented;
# the eval runs them with a graceful fallback (select_context_with_meta
# falls through to TF-IDF when the embedding model is absent).
ALL_MODES = ["tfidf", "dense", "hybrid"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PaperQA across all RETRIEVAL_MODE × USE_RERANKER combinations."
    )
    parser.add_argument(
        "eval_set",
        nargs="?",
        default="eval_set.json",
        help="Path to evaluation JSON file (default: eval_set.json)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for per-run JSON artefacts (default: results)",
    )
    parser.add_argument(
        "--combos",
        nargs="+",
        choices=ALL_MODES,
        default=ALL_MODES,
        metavar="MODE",
        help=f"Retrieval modes to include (default: all — {', '.join(ALL_MODES)})",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Skip USE_RERANKER=True combinations (useful when onnx_reranker/ is absent)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-item output; only print the summary table",
    )
    args = parser.parse_args()

    eval_set_path = Path(args.eval_set)
    results_dir   = Path(args.results_dir)

    if not eval_set_path.is_file():
        print(f"Error: eval set not found: {eval_set_path}")
        print(
            "\nExpected JSON list:\n"
            '[\n  {"paper_path": "papers/sample.pdf", "question": "...", '
            '"gold_answer": "...", "gold_page": 1}\n]'
        )
        sys.exit(1)

    with open(eval_set_path, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except Exception as exc:
            print(f"Error reading {eval_set_path}: {exc}")
            sys.exit(1)

    if not isinstance(data, list):
        print(f"Error: expected a JSON list in {eval_set_path}, got {type(data).__name__}")
        sys.exit(1)

    print(f"Loaded {len(data)} sample(s) from {eval_set_path}")

    # Build the list of (retrieval_mode, use_reranker) pairs to evaluate.
    reranker_flags = [False] if args.no_reranker else [False, True]
    combinations = [
        (mode, use_reranker)
        for mode in args.combos
        for use_reranker in reranker_flags
    ]

    combo_results: list[dict] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for retrieval_mode, use_reranker in combinations:
        result = run_one_combination(
            data           = data,
            eval_set_path  = eval_set_path,
            retrieval_mode = retrieval_mode,
            use_reranker   = use_reranker,
            verbose        = not args.quiet,
        )
        combo_results.append(result)

        # Save per-combination JSON artefact.
        results_dir.mkdir(parents=True, exist_ok=True)
        label_safe = result["label"].replace("+", "_")
        out_file   = results_dir / f"run_{timestamp}_{label_safe}.json"
        payload    = {
            "run_id":          f"run_{timestamp}_{label_safe}",
            "timestamp":       datetime.now().isoformat(),
            "retrieval_mode":  retrieval_mode,
            "use_reranker":    use_reranker,
            "eval_set_path":   str(eval_set_path),
            "summary": {
                "total_samples": result["total_samples"],
                "recall_at_5":   result["recall_at_5"],
                "exact_match":   result["exact_match"],
                "f1":            result["f1"],
            },
            "results": result["per_item_results"],
        }
        with open(out_file, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  → saved {out_file}")

    # ── Print console summary ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"{'Stage':<30} {'Recall@5':>10} {'Exact Match':>13} {'F1':>8}")
    print(f"{'-'*30} {'-'*10} {'-'*13} {'-'*8}")
    for r in combo_results:
        print(
            f"{r['label']:<30} "
            f"{r['recall_at_5']:>10.4f} "
            f"{r['exact_match']:>13.4f} "
            f"{r['f1']:>8.4f}"
        )
    print(f"{'='*70}")

    # ── Write EVAL.md ────────────────────────────────────────────────────────
    eval_md_path = Path(__file__).resolve().parent / "EVAL.md"
    write_markdown_table(combo_results, eval_set_path, eval_md_path)


if __name__ == "__main__":
    main()
