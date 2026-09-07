"""
Batch evaluation of the Hybrid (RAG + fine-tuned model) configuration, on the
same held-out test set and scoring logic as evaluate_finetune_hf.py, so the
two are directly comparable. Retrieval uses a short, undecorated query
(ad_description + issue), while the LLM-facing prompt uses the full
"Advertisement: ... Does this breach ...?" template from build_finetune_dataset.py,
with the retrieved context prepended in front of it.

Usage:
    python evaluate_hybrid_hf.py --adapter-path ../adapters/legal_lora_v1_gpu
    python evaluate_hybrid_hf.py --adapter-path ../adapters/legal_lora_v1_gpu --limit 5
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

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
FINETUNE_DIR = Path(__file__).resolve().parent.parent / "data" / "finetune"


def load_raw_records_by_url():
    records = {}
    for fname in ("asa_rulings_group_b.jsonl", "asa_rulings_group_d.jsonl"):
        with open(RAW_DIR / fname, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                records[rec["source_url"]] = rec
    return records


def build_retrieval_query(record):
    # Not build_user_message()'s output: that adds "Advertisement:"/"Complaint /
    # issue raised:" labels and the trailing "Does this breach...?" question,
    # which dilute retrieval toward generic wording. Retrieval gets just the facts.
    parts = [record.get("ad_description") or "", record.get("issue") or ""]
    return "\n\n".join(p for p in parts if p)


def format_context(results):
    return "\n\n".join(f"[{i}] {chunk['citation']}\n{chunk['text']}" for i, (score, chunk) in enumerate(results, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-path", required=True)
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

    print("Loading fine-tuned model (adapter applied)...")
    model, tokenizer = load_model(adapter_path=args.adapter_path)

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

    summarize("Hybrid (RAG + fine-tuned)", results)


if __name__ == "__main__":
    main()
