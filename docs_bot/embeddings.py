"""
Pluggable embedding backend for the ORCHARD docs assistant.

Selected by config.EMBED_BACKEND. Both build_index.py (corpus) and ask.py
(query) embed through here so the same model is always used on both sides.
All returned vectors are L2-normalized, so cosine similarity == dot product.

Backends:
  "sentence-transformers" : local, free, no API key (pulls torch). DEFAULT.
  "openai"                : needs OPENAI_API_KEY.
  "voyage"                : needs VOYAGE_API_KEY (Anthropic's recommended partner).
"""
import numpy as np

import config

_st_model = None  # cached SentenceTransformer instance


def backend_name():
    return config.EMBED_BACKEND


def _normalize(vecs):
    vecs = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def _embed_sentence_transformers(texts):
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer

        _st_model = SentenceTransformer(config.EMBED_MODEL_ST)
    vecs = _st_model.encode(texts, batch_size=64, show_progress_bar=len(texts) > 64)
    return _normalize(vecs)


def _embed_openai(texts):
    from openai import OpenAI

    client = OpenAI()
    out = []
    for i in range(0, len(texts), 256):  # batch
        resp = client.embeddings.create(model=config.EMBED_MODEL_OPENAI, input=texts[i : i + 256])
        out.extend(d.embedding for d in resp.data)
    return _normalize(out)


def _embed_voyage(texts):
    import voyageai

    client = voyageai.Client()
    out = []
    for i in range(0, len(texts), 128):
        resp = client.embed(texts[i : i + 128], model=config.EMBED_MODEL_VOYAGE, input_type="document")
        out.extend(resp.embeddings)
    return _normalize(out)


def embed_texts(texts):
    """Embed a list of strings -> (N, D) normalized float32 array."""
    backend = config.EMBED_BACKEND
    if backend == "sentence-transformers":
        return _embed_sentence_transformers(texts)
    if backend == "openai":
        return _embed_openai(texts)
    if backend == "voyage":
        return _embed_voyage(texts)
    raise ValueError(f"Unknown EMBED_BACKEND: {backend!r}")


def embed_query(text):
    """Embed a single query string -> (D,) normalized float32 vector."""
    return embed_texts([text])[0]
