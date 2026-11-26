"""Context packs: the smallest useful bundle of "what this user knows and is doing" for a task.

An agent asks `build_pack(task)` and gets back, within a token budget:

    profile      facts from the Promptly profile that fit the task (privacy gate applies)
    relevant     pages the user read that match the task, each with its best passage and why
    session      what the user is doing right now (current session trail + searches)
    highlights   things the user explicitly highlighted that match the task
    interests    long-running topics and sources, for tone and defaults

Sections are filled in priority order and trimmed to fit, so a small budget still gets the
most relevant items rather than a truncated blob.
"""

from __future__ import annotations

from ..context import Retriever
from ..models import Fact
from .memory import BrowserMemory

CHARS_PER_TOKEN = 4


def _cost(obj) -> int:
    return len(str(obj)) // CHARS_PER_TOKEN + 1


def build_pack(memory: BrowserMemory, task: str, *, facts: list[Fact] | None = None, retriever: Retriever | None = None,
               max_tokens: int = 1500, max_sensitivity: str = "personal", include_session: bool = True) -> dict:
    budget = max_tokens
    pack: dict = {"task": task, "profile": [], "relevant": [], "session": [], "highlights": [], "interests": {},
                  "withheld_fact_ids": [], "token_estimate": 0}

    def take(section: str, items: list) -> None:
        nonlocal budget
        for it in items:
            c = _cost(it)
            if c > budget:
                break
            pack[section].append(it)
            budget -= c

    if facts and retriever:
        ranked, held = retriever.select(task, facts, max_sensitivity, limit=8)
        pack["withheld_fact_ids"] = [f.id for f in held]
        take("profile", [{"id": r.fact.id, "text": r.fact.text, "category": r.fact.category}
                         for r in ranked if r.score > 0.2][:6])

    hits = memory.search(task, k=8)
    take("relevant", [h.as_dict() for h in hits if h.score > 0.2])

    if include_session:
        take("session", memory.current_session()[-10:])

    hl_hits = [e for e in memory.events("highlight", limit=200)
               if e["actor"] == "user" and _overlaps(task, e["text"] + " " + (e["title"] or ""))]
    take("highlights", [{"text": e["text"][:300], "url": e["url"], "at": e["at"]} for e in hl_hits[:5]])

    ints = memory.interests(limit=6)
    small = {"topics": [t["topic"] for t in ints["topics"]], "domains": [d["domain"] for d in ints["domains"]]}
    if _cost(small) <= budget:
        pack["interests"] = small
        budget -= _cost(small)

    pack["token_estimate"] = max_tokens - budget
    return pack


def _overlaps(a: str, b: str) -> bool:
    from ..context import tokens

    ta, tb = tokens(a), tokens(b)
    return bool(ta) and len(ta & tb) / len(ta) >= 0.25


def pack_to_markdown(pack: dict) -> str:
    out = [f"# Context for: {pack['task']}"]
    if pack["profile"]:
        out.append("\n## About the user")
        out += [f"- {f['text']}" for f in pack["profile"]]
    if pack["session"]:
        out.append("\n## What they are doing right now")
        for s in pack["session"]:
            out.append(f"- searched: {s['search']}" if "search" in s else f"- {s['title'] or s['url']} ({s['url']}, {s['dwell_s']}s)")
    if pack["relevant"]:
        out.append("\n## Pages they have read that relate")
        for h in pack["relevant"]:
            out.append(f"- [{h['title'] or h['url']}]({h['url']}) — {', '.join(h['why']) or 'related'}\n  > {h['snippet']}")
    if pack["highlights"]:
        out.append("\n## Things they highlighted")
        out += [f"- \"{h['text']}\" ({h['url']})" for h in pack["highlights"]]
    if pack["interests"].get("topics"):
        out.append("\n## Ongoing interests")
        out.append(f"- topics: {', '.join(pack['interests']['topics'])}")
        if pack["interests"].get("domains"):
            out.append(f"- usual sources: {', '.join(pack['interests']['domains'])}")
    return "\n".join(out) + "\n"
