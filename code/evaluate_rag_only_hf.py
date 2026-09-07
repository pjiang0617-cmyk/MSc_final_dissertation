"""
Batch evaluation of the RAG-only configuration (base model, no fine-tuning,
with retrieval), on the same held-out test set and scoring logic as
evaluate_finetune_hf.py and evaluate_hybrid_hf.py, so all four configurations
(Baseline, Fine-tuned, RAG-only, Hybrid) are directly comparable.

Identical to evaluate_hybrid_hf.py otherwise -- same retrieval query
construction, stratified retrieval, and prompt assembly -- except load_model()
is called with adapter_path=None, so no LoRA adapter is applied.

Usage:
    python evaluate_rag_only_hf.py
    python evaluate_rag_only_hf.py --limit 5
"""

import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from build_finetune_dataset import build_user_message
from evaluate_finetune_hf import (
    load_valid_rule_numbers,
    load_ground_truth_by_url,
    score_answer,
    summarize,
    load_model,
    run_model,
)
from retrieve import load_index, retrieve_stratified, MODEL_NAME as EMBED_MODEL_NAME
from evaluate_hybrid_hf import load_raw_records_by_url, build_retrieval_query, format_context

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
FINETUNE_DIR = Path(__file__).resolve().parent.parent / "data" / "finetune"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-rules", type=int, default=3)
    ap.add_argument("--k-cases", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=500)
    args = ap.parse_args()

    valid_rules = load_valid_rule_numbers()
    ground_truth = load_ground_truth_by_url()
    raw_records = load_raw_records_by_url()
    print(f"Loaded {len(valid_rules)} real CAP/BCAP rule numbers, {len(ground_truth)} ground-truth rulings")

    test_examples = [json.loads(l) for l in open(FINETUNE_DIR / "test.jsonl", encoding="utf-8")]
    if args.limit:
        test_examples = test_examples[:args.limit]
    print(f"Evaluating on {len(test_examples)} held-out test examples\n")

    print("Loading retrieval index...")
    chunks, embeddings = load_index()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Loading base model (no adapter)...")
    model, tokenizer = load_model(adapter_path=None)

    results = []
    for i, ex in enumerate(test_examples):
        gt = ground_truth.get(ex["source_url"])
        record = raw_records.get(ex["source_url"])
        if gt is None or record is None:
            continue

        query = build_retrieval_query(record)
        retrieved = retrieve_stratified(query, chunks, embeddings, embed_model, k_rules=args.k_rules, k_cases=args.k_cases)
        context = format_context(retrieved)
        user_content = f"Relevant rules and legislation:\n{context}\n\n{build_user_message(record)}"

        answer = run_model(model, tokenizer, user_content, args.max_tokens)
        results.append(score_answer(answer, gt, valid_rules))
        print(f"  [{i + 1}/{len(test_examples)}] {ex['source_url']}")

    summarize("RAG-only (base model + retrieval, no fine-tuning)", results)


if __name__ == "__main__":
    main()
