"""
Configuration for the ORCHARD documentation assistant (RAG starter).

Design choices (see docs_bot/README.md for the rationale):
  * Corpus = an allowlist of source globs, further restricted to files that are
    *tracked by git* (build_corpus.py). So the public bot only ever indexes
    committed, public files -- it cannot leak uncommitted/WIP material or cite
    files a fresh clone wouldn't have. (models/, LaTeX build products, and
    anything gitignored are excluded automatically.) The `--internal` flag lifts
    the tracked-only restriction for a local maintainer bot.
  * Answers are grounded strictly in retrieved excerpts, with citations and an
    explicit "I don't know -> open an issue/discussion" fallback. A scientific
    code must prefer "I'm not sure" over a confident wrong answer.
"""
from pathlib import Path

# Repo root is the parent of this docs_bot/ directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
BOT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Corpus sources (PUBLIC by default) -- each entry: (glob, kind)
# `kind` is metadata used for display/weighting; globs are relative to REPO_ROOT.
# ---------------------------------------------------------------------------
PUBLIC_SOURCES = [
    ("README.md", "readme"),
    ("FAQ.md", "faq"),
    ("parameter_descriptions.md", "parameters"),
    ("writeups/**/*.tex", "writeup"),       # method writeups; tracked-filter drops the WIP draft
    ("tutorial_*.ipynb", "tutorial"),
    ("parameter_examples/*.ini", "example"),
    # API docstrings from the user-facing modules (module + def/class docstrings):
    ("evolution.py", "api"),
    ("static.py", "api"),
    ("initial.py", "api"),
    ("transport.py", "api"),
    ("hydrostatic.py", "api"),
    ("atm_bc.py", "api"),
    ("load_models.py", "api"),
    ("plots.py", "api"),
    ("plot_class.py", "api"),
]

# Opt-in only (build_corpus.py --internal): also index files that are NOT yet
# committed/tracked (work-in-progress docs on a maintainer's machine). With the
# default public build the git-tracked filter already restricts the corpus to
# committed files.
INTERNAL_SOURCES = []

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150

# ---------------------------------------------------------------------------
# Embeddings (retrieval). Backend is pluggable; default is fully local + free.
#   "sentence-transformers" : local model, no API key (pulls torch)
#   "openai"                : OPENAI_API_KEY, model EMBED_MODEL_OPENAI
#   "voyage"                : VOYAGE_API_KEY (Anthropic's recommended partner)
# ---------------------------------------------------------------------------
EMBED_BACKEND = "sentence-transformers"
EMBED_MODEL_ST = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, decent
EMBED_MODEL_OPENAI = "text-embedding-3-small"
EMBED_MODEL_VOYAGE = "voyage-3"

# ---------------------------------------------------------------------------
# Generation (answering) -- Anthropic Claude.
# Model names change over time and old ones get retired (a stale id returns a
# 404 "model: ... not_found_error"). Leave LLM_MODEL = None to AUTO-SELECT a
# current model available to YOUR account (ask.py queries the models list and
# prefers a Sonnet, then Haiku, then Opus). Or pin a specific id -- list them with:
#   python -c "import anthropic;[print(m.id) for m in anthropic.Anthropic().models.list().data]"
# ---------------------------------------------------------------------------
LLM_MODEL = None          # None = auto-pick a valid model; or pin a specific id
LLM_MODEL_PREFERENCE = ("sonnet", "haiku", "opus")  # auto-pick order; haiku first = cheapest
LLM_MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# Retrieval behavior
# ---------------------------------------------------------------------------
TOP_K = 6                 # excerpts passed to the model
MIN_SCORE = 0.25          # cosine sim below this for the best hit -> escalate, don't answer

# ---------------------------------------------------------------------------
# Artifact paths (generated; recommend gitignoring these -- see README)
# ---------------------------------------------------------------------------
CORPUS_PATH = BOT_DIR / "corpus.jsonl"
VECTORS_PATH = BOT_DIR / "index_vectors.npy"
INDEX_META_PATH = BOT_DIR / "index_meta.json"

# ---------------------------------------------------------------------------
# Help links shown in fallbacks / disclaimers. Update once URLs are live.
# ---------------------------------------------------------------------------
REPO_URL = "https://github.com/robtejada/orchard"
ISSUES_URL = REPO_URL + "/issues"
DISCUSSIONS_URL = REPO_URL + "/discussions"

# ---------------------------------------------------------------------------
# Grounding system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are the ORCHARD documentation assistant -- a friendly, concise guide that \
helps people use ORCHARD, a public planetary interior-structure and evolution code.

You can:
- Introduce yourself and explain what you help with (running and configuring ORCHARD, finding the \
right parameters, example configs, and tutorials), and respond naturally to greetings and simple \
conversational turns. You do NOT need documentation excerpts for these.
- Answer questions about how to use ORCHARD from the numbered documentation excerpts provided in the \
user message (drawn from ORCHARD's README, FAQ, parameter descriptions, tutorials, example configs, \
method writeups, and code docstrings).

Rules:
- For questions about USING or CONFIGURING ORCHARD: answer only from the provided excerpts and cite \
them with bracketed numbers like [1], [2]. Prefer concrete parameters, example files \
(parameter_examples/...), and tutorials, with short config snippets when helpful. Never invent \
parameter names, file names, defaults, or behavior, and never fabricate citations.
- If no excerpt covers an ORCHARD usage question, say so plainly and point the user to \
parameter_descriptions.md and the tutorials, and to open a GitHub Discussion ({DISCUSSIONS_URL}) for \
usage questions or an Issue ({ISSUES_URL}) for bugs.
- Use the earlier conversation turns for context when answering follow-ups.
- Politely decline requests unrelated to ORCHARD or planetary interior/evolution modeling, and steer \
back to how you can help.

Keep a helpful, conversational tone, and answers focused and practical."""
