"""
RAG pipeline (Approach 2 -- hand-rolled): stage 2 of 3.

Embeds every chunk from chunks.jsonl with a small local sentence-transformers
model and saves the vectors alongside the chunk order, so retrieve.py can do
a brute-force cosine-similarity search (fine at this scale: ~2.5k chunks,
no need for a dedicated vector database or ANN index).

Model choice: BAAI/bge-small-en-v1.5 -- small (~130MB), CPU/Metal-friendly,
good retrieval quality for its size, widely used for exactly this kind of
small-corpus semantic search.

Usage:
    python build_embeddings.py
Writes:
    ../data/rag/chunk_embeddings.npy
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

RAG_DIR = Path(__file__).resolve().parent.parent / "data" / "rag"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# bge models expect this prefix on passages (not on queries) for best retrieval quality
PASSAGE_PREFIX = ""


def load_chunks():
    chunks = []
    with open(RAG_DIR / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    model = SentenceTransformer(MODEL_NAME)
    texts = [PASSAGE_PREFIX + c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # so cosine similarity == dot product at query time
    )

    out_path = RAG_DIR / "chunk_embeddings.npy"
    np.save(out_path, embeddings.astype(np.float32))
    print(f"Saved embeddings {embeddings.shape} -> {out_path}")


if __name__ == "__main__":
    main()