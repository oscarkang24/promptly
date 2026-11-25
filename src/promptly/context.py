"""Choosing which personal facts are allowed, and relevant, for a prompt.

Two gates run before anything is sent to a model:

1. Privacy gate: facts above the request's max sensitivity never leave the machine.
2. Relevance shortlist: a cheap lexical scorer keeps the top-N candidates, so large
   profiles stay cheap and the model only sees facts with some plausible bearing.

The model then makes the final call on which shortlisted facts actually matter.
"""

from __future__ import annotations

import re

from .models import SENSITIVITY_RANK, Fact

_WORD = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")
_STOP = set(
    """a an the and or but if of to in on for with at by from as is are was were be been am
    i me my mine you your it its this that these those we our they their them he she his her
    do does did can could should would will shall may might must have has had not no yes so
    what which who whom how why when where please help want need make get give tell write""".split()
)

# Facts in these categories shape almost every answer, so they get a base score.
_ALWAYS_RELEVANT = {"preference": 0.6, "skill": 0.3, "identity": 0.2, "work": 0.2}

# Task words that hint at which categories matter.
_TASK_HINTS: dict[str, set[str]] = {
    "work": {"email", "manager", "team", "meeting", "slack", "review", "resume", "career", "job", "interview"},
    "project": {"code", "bug", "build", "repo", "deploy", "feature", "api", "app", "project", "design"},
    "constraint": {"plan", "trip", "budget", "recipe", "meal", "schedule", "buy", "workout", "diet"},
    "goal": {"plan", "learn", "improve", "strategy", "roadmap", "career", "goal"},
    "skill": {"explain", "learn", "understand", "teach", "how", "tutorial"},
}


def tokens(text: str) -> set[str]:
    return {w.strip(".-") for w in _WORD.findall(text.lower())} - _STOP - {""}


def allowed(facts: list[Fact], max_sensitivity: str) -> tuple[list[Fact], list[Fact]]:
    limit = SENSITIVITY_RANK[max_sensitivity]
    ok = [f for f in facts if SENSITIVITY_RANK[f.sensitivity] <= limit]
    held = [f for f in facts if SENSITIVITY_RANK[f.sensitivity] > limit]
    return ok, held


def score(prompt: str, fact: Fact) -> float:
    p = tokens(prompt)
    f = tokens(fact.text) | {t.lower() for t in fact.tags}
    overlap = len(p & f)
    s = overlap * 1.0 + _ALWAYS_RELEVANT.get(fact.category, 0.0)
    if p & _TASK_HINTS.get(fact.category, set()):
        s += 0.5
    # Prefix matches catch "deploying" vs "deploy", "python" vs "python3".
    s += 0.3 * sum(1 for a in p for b in f if a != b and len(a) > 3 and len(b) > 3 and (a.startswith(b) or b.startswith(a)))
    return s


def shortlist(prompt: str, facts: list[Fact], limit: int = 25) -> list[Fact]:
    if len(facts) <= limit:
        return list(facts)
    ranked = sorted(facts, key=lambda f: score(prompt, f), reverse=True)
    return ranked[:limit]


def select(prompt: str, facts: list[Fact], max_sensitivity: str, limit: int = 25) -> tuple[list[Fact], list[Fact]]:
    """Return (candidates to share with the engine, facts withheld for privacy)."""
    ok, held = allowed(facts, max_sensitivity)
    return shortlist(prompt, ok, limit), held
