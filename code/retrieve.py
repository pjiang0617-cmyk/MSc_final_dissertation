"""
Brute-force cosine-similarity retrieval over the chunks embedded by
build_embeddings.py. Embeddings are pre-normalized, so a plain dot product
gives cosine similarity directly.

Usage:
    python retrieve.py "does a weight-loss ad breach the code if it claims a specific kg loss?"
    python retrieve.py "gambling ad aimed at under-18s" --k 5 --group group_d
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

RAG_DIR = Path(__file__).resolve().parent.parent / "data" / "rag"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# bge models are trained with this instruction prefix on queries (not passages)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_index():
    chunks = []
    with open(RAG_DIR / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    embeddings = np.load(RAG_DIR / "chunk_embeddings.npy")
    assert len(chunks) == embeddings.shape[0], "chunks.jsonl and chunk_embeddings.npy are out of sync -- rerun build_embeddings.py"
    return chunks, embeddings


def retrieve(query, chunks, embeddings, model, k=5, group=None, source_type=None):
    query_vec = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
    scores = embeddings @ query_vec  # cosine similarity (both sides pre-normalized)

    order = np.argsort(-scores)
    results = []
    for idx in order:
        chunk = chunks[idx]
        if group is not None and chunk["metadata"].get("group") != group:
            continue
        if source_type is not None and chunk["source_type"] != source_type:
            continue
        results.append((float(scores[idx]), chunk))
        if len(results) >= k:
            break
    return results


# retrieve() ranks every source type on one combined similarity score, which lets
# long ASA case narratives (closer in wording to a typical query) systematically
# outscore short, abstractly worded rule/legislation chunks and crowd them out of
# the top-k. retrieve_stratified() avoids this by retrieving rule/legislation
# chunks and case chunks from two separate pools, so neither can crowd out the
# other -- mirroring how an ASA ruling itself pairs a case discussion with an
# explicit Code citation.
RULE_LIKE_TYPES = {"cap_rule", "bcap_rule", "legislation_section"}
CASE_LIKE_TYPES = {"asa_case_summary", "asa_case_assessment"}


def retrieve_stratified(query, chunks, embeddings, model, k_rules=3, k_cases=2, group=None):
    query_vec = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
    scores = embeddings @ query_vec

    order = np.argsort(-scores)
    rule_results, case_results = [], []
    for idx in order:
        if len(rule_results) >= k_rules and len(case_results) >= k_cases:
            break
        chunk = chunks[idx]
        if group is not None and chunk["metadata"].get("group") != group:
            continue
        source_type = chunk["source_type"]
        if source_type in RULE_LIKE_TYPES and len(rule_results) < k_rules:
            rule_results.append((float(scores[idx]), chunk))
        elif source_type in CASE_LIKE_TYPES and len(case_results) < k_cases:
            case_results.append((float(scores[idx]), chunk))
    return rule_results + case_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--group", default=None, help="filter to group_b or group_d")
    ap.add_argument("--source-type", default=None,
                     help="filter to cap_rule | bcap_rule | legislation_section | asa_case_summary | asa_case_assessment")
    args = ap.parse_args()

    chunks, embeddings = load_index()
    model = SentenceTransformer(MODEL_NAME)

    results = retrieve(args.query, chunks, embeddings, model, k=args.k, group=args.group, source_type=args.source_type)

    print(f"\nQuery: {args.query}\n")
    for rank, (score, chunk) in enumerate(results, 1):
        print(f"[{rank}] score={score:.3f}  {chunk['citation']}  ({chunk['source_type']})")
        print(f"    {chunk['text'][:220]}{'...' if len(chunk['text']) > 220 else ''}")
        print(f"    {chunk['source_url']}")
        print()


if __name__ == "__main__":
    main()
