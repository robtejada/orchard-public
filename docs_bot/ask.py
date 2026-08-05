#!/usr/bin/env python
"""
Answer a question about using ORCHARD, grounded in the indexed public docs.

Pipeline: embed the query -> cosine-rank corpus chunks -> if the best match is
below config.MIN_SCORE, escalate (do NOT call the model and do NOT guess);
otherwise pass the top-k excerpts to Claude with a strict grounding system
prompt and return the answer plus citations.

Usage:
    python docs_bot/ask.py "How do I run a 2 Jupiter-mass planet with helium rain?"

Programmatic:
    from ask import answer
    result = answer("...")   # -> {"answer", "sources", "grounded"}
"""
import json
import sys

import numpy as np

import config
from embeddings import embed_query

DISCLAIMER = (
    "AI-generated from the ORCHARD documentation; verify against the docs and "
    "open a GitHub issue/discussion if anything is unclear."
)

_corpus = None
_vectors = None
_resolved_model = None


def _pick_model(ids):
    """Choose a model id by config.LLM_MODEL_PREFERENCE keyword order."""
    for kw in config.LLM_MODEL_PREFERENCE:
        for mid in ids:
            if kw in mid.lower():
                return mid
    return ids[0] if ids else None


def _autopick_model(client, force=False):
    """Discover a current model available to this account (cached)."""
    global _resolved_model
    if _resolved_model and not force:
        return _resolved_model
    ids = [m.id for m in client.models.list().data]
    pick = _pick_model(ids)
    if not pick:
        raise RuntimeError("No Claude models are available to this API key.")
    _resolved_model = pick
    return pick


def _load():
    global _corpus, _vectors
    if _corpus is not None:
        return
    if not config.CORPUS_PATH.exists() or not config.VECTORS_PATH.exists():
        raise SystemExit("Index not found. Run build_corpus.py then build_index.py first.")
    _corpus = [json.loads(line) for line in open(config.CORPUS_PATH, encoding="utf-8")]
    _vectors = np.load(config.VECTORS_PATH)
    if len(_corpus) != _vectors.shape[0]:
        raise SystemExit("Corpus/vectors out of sync. Rebuild the index.")


def retrieve(question, k=None):
    """Return the top-k corpus records with cosine scores (highest first)."""
    _load()
    k = k or config.TOP_K
    q = embed_query(question)                 # (D,) normalized
    scores = _vectors @ q                      # cosine sim (vectors normalized)
    order = np.argsort(-scores)[:k]
    return [(_corpus[i], float(scores[i])) for i in order]


def answer(question, history=None):
    """Answer a user message. Returns {"answer", "sources", "grounded"}.

    The model is ALWAYS called (it has an identity and can handle greetings /
    "what do you do?" without docs). Retrieval decides whether to attach
    documentation context and show sources: above config.MIN_SCORE we pass the
    excerpts and the model grounds + cites; below it we tell the model no
    relevant docs were found, so it answers meta/greetings from its role and
    escalates genuine ORCHARD questions instead of guessing.

    `history` is an optional list of (user, assistant) tuples for follow-ups.
    """
    hits = retrieve(question)
    best = hits[0][1] if hits else 0.0
    relevant = best >= config.MIN_SCORE

    if relevant:
        context = "\n\n".join(
            f"[{i + 1}] (source: {rec['source']} — {rec['title']})\n{rec['text']}"
            for i, (rec, _) in enumerate(hits)
        )
        user_msg = (
            f"Documentation excerpts that may be relevant:\n\n{context}\n\n"
            f"---\nUser message: {question}"
        )
    else:
        user_msg = (
            "(No strongly-relevant ORCHARD documentation was retrieved for this message. "
            "If it is a greeting or a question about you, answer it from your role. If it is "
            "an ORCHARD usage question, say you don't have it in the docs and point to the "
            "docs / GitHub rather than guessing.)\n\n"
            f"---\nUser message: {question}"
        )

    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def _create(model_id):
        msgs = []
        for prev_q, prev_a in (history or [])[-3:]:
            msgs.append({"role": "user", "content": prev_q})
            msgs.append({"role": "assistant", "content": prev_a})
        msgs.append({"role": "user", "content": user_msg})
        return client.messages.create(
            model=model_id,
            max_tokens=config.LLM_MAX_TOKENS,
            system=config.SYSTEM_PROMPT,
            messages=msgs,
        )

    model = config.LLM_MODEL or _autopick_model(client)
    try:
        resp = _create(model)
    except anthropic.NotFoundError:
        # Pinned model retired/unavailable -> discover a current one and retry.
        model = _autopick_model(client, force=True)
        resp = _create(model)
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    # Show sources only when retrieval actually found relevant docs.
    sources = []
    if relevant:
        seen = set()
        for rec, score in hits:
            if rec["source"] not in seen:
                seen.add(rec["source"])
                sources.append({"source": rec["source"], "title": rec["title"], "score": round(score, 3)})
    return {"answer": text, "sources": sources, "grounded": relevant}


def main():
    if len(sys.argv) < 2:
        print('Usage: python docs_bot/ask.py "your question"')
        raise SystemExit(1)
    result = answer(" ".join(sys.argv[1:]))
    print("\n" + result["answer"] + "\n")
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f"  - {s['source']}  ({s['title']})  [score {s['score']}]")
    print("\n" + DISCLAIMER)


if __name__ == "__main__":
    main()
