#!/usr/bin/env python
"""
Embed the corpus chunks and write a vector index aligned to corpus.jsonl.

Reads `corpus.jsonl` (from build_corpus.py), embeds each chunk's text with the
backend selected in config.EMBED_BACKEND, L2-normalizes the vectors (so cosine
similarity is a dot product), and saves them to `index_vectors.npy` in the same
order as the corpus. ask.py loads corpus.jsonl + index_vectors.npy together.

Usage:
    python docs_bot/build_index.py
"""
import json

import numpy as np

import config
from embeddings import embed_texts, backend_name


def main():
    if not config.CORPUS_PATH.exists():
        raise SystemExit(
            f"{config.CORPUS_PATH} not found. Run build_corpus.py first."
        )

    records = [json.loads(line) for line in open(config.CORPUS_PATH, encoding="utf-8")]
    texts = [r["text"] for r in records]
    print(f"Embedding {len(texts)} chunks with backend '{backend_name()}' ...")

    vectors = embed_texts(texts)               # (N, D) float32, L2-normalized
    np.save(config.VECTORS_PATH, vectors.astype(np.float32))

    meta = {
        "backend": backend_name(),
        "n_chunks": len(texts),
        "dim": int(vectors.shape[1]),
    }
    config.INDEX_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Saved {vectors.shape} vectors to {config.VECTORS_PATH}")
    print(f"Index meta: {meta}")


if __name__ == "__main__":
    main()
