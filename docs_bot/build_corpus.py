#!/usr/bin/env python
"""
Build the ORCHARD docs-assistant corpus from PUBLIC sources only.

Reads the allowlisted sources in config.PUBLIC_SOURCES (README, FAQ,
parameter_descriptions, tutorials, example .ini files, and code docstrings),
converts each to plain text, splits into overlapping chunks, and writes
`corpus.jsonl` (one JSON record per chunk).

Stdlib only -- no third-party dependencies -- so it runs in the base
environment without installing anything. Embedding/answering happen later
(build_index.py / ask.py).

Usage:
    python docs_bot/build_corpus.py            # public corpus (default)
    python docs_bot/build_corpus.py --internal # ALSO index untracked maintainer docs
                                               # (private/maintainer bot ONLY)
"""
import argparse
import ast
import glob
import json
import re
import subprocess
from pathlib import Path

import config  # docs_bot/config.py


# --------------------------------------------------------------------------- #
# Source -> text converters
# --------------------------------------------------------------------------- #
def text_from_markdown_or_ini(path):
    return path.read_text(encoding="utf-8", errors="replace")


def text_from_notebook(path):
    """Concatenate markdown + code cells of a Jupyter notebook into plain text."""
    nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    parts = []
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        if cell.get("cell_type") == "code":
            parts.append("```python\n" + src + "\n```")
        else:
            parts.append(src)
    return "\n\n".join(parts)


def text_from_python_docstrings(path):
    """Extract the module docstring + each def/class docstring as readable text."""
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    out = []
    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        out.append(f"Module {path.name}:\n{mod_doc}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node)
            if doc:
                kind = "class" if isinstance(node, ast.ClassDef) else "def"
                out.append(f"{kind} {node.name} ({path.name}):\n{doc}")
    return "\n\n".join(out)


def to_text(path):
    suffix = path.suffix.lower()
    if suffix == ".ipynb":
        return text_from_notebook(path)
    if suffix == ".py":
        return text_from_python_docstrings(path)
    # .md, .ini, .tex, .txt, ...
    return text_from_markdown_or_ini(path)


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
_HEADER_RE = re.compile(r"^(#{1,4})\s+(.*)$", re.MULTILINE)


def split_markdown_sections(text):
    """Yield (heading, section_text) for a markdown doc; falls back to one section."""
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        yield (None, text)
        return
    if matches[0].start() > 0:
        yield (None, text[: matches[0].start()])
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        yield (m.group(2).strip(), text[start:end])


def pack_paragraphs(text, max_chars, overlap):
    """Greedily pack paragraphs into <=max_chars chunks with a char overlap tail."""
    paras = re.split(r"\n\s*\n", text)
    chunks, cur = [], ""
    for para in paras:
        para = para.strip("\n")
        if not para.strip():
            continue
        if len(para) > max_chars:  # hard-split an oversized paragraph
            if cur:
                chunks.append(cur)
                cur = ""
            for i in range(0, len(para), max_chars - overlap):
                chunks.append(para[i : i + max_chars])
            continue
        if len(cur) + len(para) + 2 <= max_chars:
            cur = (cur + "\n\n" + para) if cur else para
        else:
            if cur:
                chunks.append(cur)
            tail = cur[-overlap:] if cur else ""
            cur = (tail + "\n\n" + para) if tail else para
    if cur.strip():
        chunks.append(cur)
    return chunks


def iter_chunks(text, source, kind, base_title):
    """Yield chunk records for one source document."""
    text = text.strip()
    if not text:
        return
    if source.endswith(".md"):
        sections = list(split_markdown_sections(text))
    else:
        sections = [(None, text)]
    for heading, sec in sections:
        title = f"{base_title}" + (f" — {heading}" if heading else "")
        for piece in pack_paragraphs(sec, config.CHUNK_MAX_CHARS, config.CHUNK_OVERLAP_CHARS):
            piece = piece.strip()
            if len(piece) < 30:  # skip trivially short fragments
                continue
            yield {"source": source, "title": title, "kind": kind, "text": piece}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def tracked_files():
    """Set of git-tracked paths (relative, posix), or None if git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "ls-files"], cwd=config.REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return None
    return {line for line in out.splitlines() if line}


def build(sources, include_untracked=False):
    # Public safety: restrict the corpus to git-tracked (committed) files, so the
    # bot can never index uncommitted/WIP/gitignored material. --internal lifts this.
    tracked = None if include_untracked else tracked_files()
    if tracked is None and not include_untracked:
        print("  [warn] git unavailable; cannot apply the public tracked-only filter "
              "-- verify no private files are indexed.")
    records = []
    skipped = 0
    for pattern, kind in sources:
        matched = sorted(glob.glob(str(config.REPO_ROOT / pattern), recursive=True))
        if not matched:
            print(f"  [warn] no files matched: {pattern}")
            continue
        for fp in matched:
            path = Path(fp)
            rel = str(path.relative_to(config.REPO_ROOT))
            if tracked is not None and rel not in tracked:
                skipped += 1
                continue
            try:
                text = to_text(path)
            except Exception as exc:  # never let one bad file kill the build
                print(f"  [skip] {rel}: {exc}")
                continue
            n_before = len(records)
            for rec in iter_chunks(text, rel, kind, path.name):
                rec["id"] = len(records)
                records.append(rec)
            print(f"  {rel:48s} -> {len(records) - n_before} chunks")
    if skipped:
        print(f"  [public filter] skipped {skipped} matched file(s) not tracked by git")
    return records


def main():
    ap = argparse.ArgumentParser(description="Build the ORCHARD docs-assistant corpus.")
    ap.add_argument(
        "--internal",
        action="store_true",
        help="ALSO index non-public, untracked maintainer docs. Private bot only.",
    )
    args = ap.parse_args()

    sources = list(config.PUBLIC_SOURCES)
    if args.internal:
        print("[!] --internal: indexing non-public sources. Do NOT deploy this corpus publicly.")
        sources += list(config.INTERNAL_SOURCES)

    print(f"Building corpus from {len(sources)} source patterns under {config.REPO_ROOT} ...")
    records = build(sources, include_untracked=args.internal)

    with open(config.CORPUS_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_kind = {}
    for r in records:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
    print(f"\nWrote {len(records)} chunks to {config.CORPUS_PATH}")
    print("  by kind: " + ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))


if __name__ == "__main__":
    main()
