# ORCHARD documentation assistant (RAG starter)

A small retrieval-augmented chatbot that answers "how do I use ORCHARD?"
questions from the project's **public** documentation. It is designed to be a
**first responder** that deflects common questions and points users to the right
config parameters, examples, and tutorials — and that **escalates to the
maintainers** (GitHub Issues/Discussions) when it doesn't know.

```
docs_bot/
  config.py          # corpus allowlist, models, thresholds, grounding prompt
  build_corpus.py    # public docs -> corpus.jsonl   (stdlib only; no install needed)
  build_index.py     # corpus.jsonl -> index_vectors.npy (embeddings)
  embeddings.py      # pluggable embedding backend (local / OpenAI / Voyage)
  ask.py             # retrieve -> (escalate | answer with Claude + citations); CLI
  app.py             # Streamlit web widget
  requirements.txt   # bot-only dependencies
```

## What it indexes (and what it deliberately does not)

The corpus is an **allowlist of source globs** (`config.PUBLIC_SOURCES`) —
`README.md`, `FAQ.md`, `parameter_descriptions.md`, the method
writeups (`writeups/**/*.tex`), `tutorials/*.ipynb`, `parameter_examples/*.ini`,
and the docstrings of the user-facing modules — **intersected with the files git
actually tracks**.

That git-tracked intersection is the safety guarantee: the public bot only ever
indexes **committed** files, so it cannot leak uncommitted/WIP material or cite
files a fresh clone wouldn't have, and any
gitignored example configs are skipped automatically. `build_corpus.py` reports
how many matched files it skipped for this reason.

`build_corpus.py --internal` lifts the tracked-only restriction, for a local
maintainer bot that should also see uncommitted WIP docs.

## Quick start

```bash
# 1) Isolated environment (keep separate from orchard_env)
python -m venv .botenv && source .botenv/bin/activate
pip install -r docs_bot/requirements.txt

# 2) Build the corpus (no API key needed; stdlib only)
python docs_bot/build_corpus.py

# 3) Embed it into a vector index (local model by default; downloads ~80 MB once)
python docs_bot/build_index.py

# 4) Ask a question (needs an Anthropic key for the answer step)
export ANTHROPIC_API_KEY=sk-ant-...
python docs_bot/ask.py "How do I model a sub-Neptune with an H/He envelope on a rocky core?"

# 5) Or launch the web widget
streamlit run docs_bot/app.py
```

## Design choices (and how to change them)

- **Grounded answers only.** `config.SYSTEM_PROMPT` instructs the model to answer
  *only* from retrieved excerpts, cite them as `[n]`, and never invent parameter
  or file names. For a scientific code, a confident wrong answer is worse than
  "I don't know."
- **Confidence gate.** If the best retrieval score is below `config.MIN_SCORE`,
  `ask.py` returns an escalation message and **does not call the model** — it
  points the user to the docs and to GitHub. Tune `MIN_SCORE` / `TOP_K` to taste.
- **Pluggable embeddings.** Default is local `sentence-transformers` (free, no
  key). Set `config.EMBED_BACKEND` to `"openai"` or `"voyage"` for hosted
  embeddings (lighter install, needs a key). Rebuild the index after changing it.
- **Model.** With `config.LLM_MODEL = None` the bot queries the models available
  to your key and picks the first match in `config.LLM_MODEL_PREFERENCE`
  (Haiku first, the cheapest; then Sonnet, then Opus). Reorder that tuple, or
  pin `LLM_MODEL` to an exact model id, to change it.
- **Links.** Update `REPO_URL` in `config.py` once the public URLs are live.

## Keeping it current

Re-run steps 2–3 whenever the docs change. A GitHub Action can do this on every
push to `main` (commit the rebuilt index, or rebuild at deploy time):

```yaml
# .github/workflows/reindex-docs-bot.yml   (example — add if you want auto-reindex)
name: Reindex docs bot
on:
  push:
    branches: [main]
    paths: ["FAQ.md", "README.md", "parameter_descriptions.md",
            "tutorials/**", "parameter_examples/**", "docs_bot/**"]
jobs:
  reindex:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r docs_bot/requirements.txt
      - run: python docs_bot/build_corpus.py && python docs_bot/build_index.py
      # then redeploy the widget, or commit the artifacts
```

## Generated files

`build_corpus.py` and `build_index.py` write `corpus.jsonl`,
`index_vectors.npy`, and `index_meta.json` into `docs_bot/`. These are
regenerable build artifacts and are already gitignored (along with `.botenv/`).

## Deploying the website widget

`app.py` runs anywhere Streamlit does. Free options: **Streamlit Community Cloud**
or **Hugging Face Spaces** — point it at this repo, set the `ANTHROPIC_API_KEY`
secret, and (if you don't commit the index) run the two build steps in the start
command. Embed it on the ORCHARD website via an `<iframe>`, or link to it from the
README.

## Roadmap ideas

- A GitHub Action that auto-answers new issues/discussions labeled `question`
  (with an "AI answer — a maintainer will follow up" disclaimer).
- Prompt caching on the system prompt to cut token cost.
- Logging of asked questions to find documentation gaps.
- A BM25 keyword-retrieval fallback for fully offline / zero-key operation.
