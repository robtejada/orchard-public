#!/usr/bin/env python
"""
Generate an automated first-pass answer for a newly opened GitHub issue.

Run by the docs-bot-issue-responder workflow. Reads the issue from the GitHub
event payload (GITHUB_EVENT_PATH), runs the grounded RAG answerer, and writes the
comment markdown to `bot_comment.md`. The workflow posts it with `gh issue
comment` ONLY if that file exists, so this script simply skips (writes nothing)
for cases that should not be answered (pull requests, bot authors, empty issues).
"""
import json
import os
import sys

from ask import answer, DISCLAIMER

COMMENT_PATH = "bot_comment.md"


def main():
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path or not os.path.exists(event_path):
        print("No GITHUB_EVENT_PATH; nothing to do.")
        return 0
    event = json.load(open(event_path, encoding="utf-8"))
    issue = event.get("issue") or {}

    if not issue or "pull_request" in issue:
        print("Not an issue (or is a PR); skipping.")
        return 0
    author = ((issue.get("user") or {}).get("login") or "").lower()
    if author.endswith("[bot]"):
        print("Issue opened by a bot; skipping to avoid loops.")
        return 0

    title = (issue.get("title") or "").strip()
    body = (issue.get("body") or "").strip()
    question = (title + "\n\n" + body).strip()
    if len(question) < 8:
        print("Issue text too short to answer; skipping.")
        return 0

    result = answer(question)

    lines = [
        "🤖 **Automated first-pass answer** from the ORCHARD docs assistant.",
        "",
        "> AI-generated from the ORCHARD documentation. It can be wrong — a maintainer "
        "will follow up. Please don't close the issue based on this alone.",
        "",
        result["answer"],
    ]
    if result.get("sources"):
        lines += ["", "**Sources:**"]
        lines += [f"- `{s['source']}` — {s['title']}" for s in result["sources"]]
    lines += ["", "---", f"_{DISCLAIMER}_"]

    with open(COMMENT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {COMMENT_PATH} (grounded={result.get('grounded')}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
