# Embeds chunks.jsonl with BAAI/bge-small-en-v1.5, saves chunk_embeddings.npy
# in the same row order as chunks.jsonl.

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