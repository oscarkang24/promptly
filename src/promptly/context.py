"""Choosing which personal facts are allowed, and relevant, for a prompt.

Pipeline before anything reaches a model:

1. Privacy gate: facts above the request's max sensitivity never leave the machine.
2. Latent-vector retrieval: the prompt and every fact live in the same embedding space.
   Each fact gets a hybrid score

       score = W_SEM * cos(prompt, fact)
             + W_LEX * keyword overlap
             + category prior
             + W_FB  * feedback(prompt, fact)

   where feedback() looks up past prompts near this one in latent space and averages the
   👍/👎 the user gave this fact there. So the ranking learns "for prompts like this, that
   fact helped / was noise" without any training step.
3. The top-N go to the engine, which makes the final call.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import Embedder, HashingEmbedder, VectorCache
from .models import SENSITIVITY_RANK, Fact

W_SEM = 1.0
W_LEX = 0.15
W_FB = 0.5
FB_SIM_FLOOR = 0.35   # past prompts less similar than this don't vote
FB_MIN_WEIGHT = 0.15  # below this total neighbor weight, feedback is ignored

# Facts in these categories shape almost every answer, so they get a small base score.
CATEGORY_PRIOR = {"preference": 0.12, "skill": 0.06, "identity": 0.03, "work": 0.03}

_WORD = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")
_STOP = set(
    """a an the and or but if of to in on for with at by from as is are was were be been am
    i me my mine you your it its this that these those we our they their them he she his her
    do does did can could should would will shall may might must have has had not no yes so
    what which who whom how why when where please help want need make get give tell write""".split()
)


def tokens(text: str) -> set[str]:
    return {w.strip(".-") for w in _WORD.findall(text.lower())} - _STOP - {""}


def lexical(prompt: str, fact: Fact) -> float:
    p = tokens(prompt)
    f = tokens(fact.text) | {t.lower() for t in fact.tags}
    if not p or not f:
        return 0.0
    return len(p & f) / min(len(p), len(f))


def allowed(facts: list[Fact], max_sensitivity: str) -> tuple[list[Fact], list[Fact]]:
    limit = SENSITIVITY_RANK[max_sensitivity]
    ok = [f for f in facts if SENSITIVITY_RANK[f.sensitivity] <= limit]
    held = [f for f in facts if SENSITIVITY_RANK[f.sensitivity] > limit]
    return ok, held


def fact_text(f: Fact) -> str:
    """What gets embedded for a fact: its text plus tags as extra signal."""
    return f.text + (" | " + ", ".join(f.tags) if f.tags else "")


@dataclass
class Feedback:
    prompt: str
    fact_id: str
    label: float  # +1 helpful, -1 noise; implicit signals use smaller magnitudes
    ts: float


class FeedbackLog:
    """Append-only JSONL of (prompt, fact, label). Prompt vectors come from the VectorCache."""

    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.Lock()
        self._items: list[Feedback] = []
        if path and path.exists():
            for line in path.read_text().splitlines():
                try:
                    self._items.append(Feedback(**json.loads(line)))
                except (ValueError, TypeError):
                    continue

    def add(self, prompt: str, fact_id: str, label: float) -> Feedback:
        fb = Feedback(prompt=prompt, fact_id=fact_id, label=max(-1.0, min(1.0, label)), ts=time.time())
        with self._lock:
            self._items.append(fb)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as fh:
                    fh.write(json.dumps(fb.__dict__, ensure_ascii=False) + "\n")
        return fb

    def all(self) -> list[Feedback]:
        with self._lock:
            return list(self._items)


@dataclass
class Scored:
    fact: Fact
    score: float
    semantic: float
    lexical: float
    feedback: float


class Retriever:
    def __init__(self, embedder: Embedder | None = None, home: Path | None = None):
        self.embedder = embedder or HashingEmbedder()
        self.vectors = VectorCache(self.embedder, home / "vectors.json" if home else None)
        self.feedback = FeedbackLog(home / "feedback.jsonl" if home else None)

    def _feedback_scores(self, qv: np.ndarray, fact_ids: list[str]) -> dict[str, float]:
        items = [fb for fb in self.feedback.all() if fb.fact_id in set(fact_ids)]
        if not items:
            return {}
        pv = self.vectors.get_many([fb.prompt for fb in items])
        sims = pv @ qv
        num: dict[str, float] = {}
        den: dict[str, float] = {}
        for fb, s in zip(items, sims):
            w = float(s) - FB_SIM_FLOOR
            if w <= 0:
                continue
            num[fb.fact_id] = num.get(fb.fact_id, 0.0) + w * fb.label
            den[fb.fact_id] = den.get(fb.fact_id, 0.0) + w
        # Weighted mean label, shrunk toward 0 when there is little nearby evidence.
        return {
            fid: (num[fid] / den[fid]) * min(1.0, den[fid] / (FB_MIN_WEIGHT * 4))
            for fid in den
            if den[fid] >= FB_MIN_WEIGHT
        }

    def rank(self, prompt: str, facts: list[Fact]) -> list[Scored]:
        if not facts:
            return []
        qv = self.vectors.get(prompt)
        fv = self.vectors.get_many([fact_text(f) for f in facts])
        sem = fv @ qv
        fb = self._feedback_scores(qv, [f.id for f in facts])
        out = []
        for f, s in zip(facts, sem):
            lex = lexical(prompt, f)
            b = fb.get(f.id, 0.0)
            total = W_SEM * float(s) + W_LEX * lex + CATEGORY_PRIOR.get(f.category, 0.0) + W_FB * b
            out.append(Scored(f, total, float(s), lex, b))
        out.sort(key=lambda x: x.score, reverse=True)
        return out

    def select(self, prompt: str, facts: list[Fact], max_sensitivity: str, limit: int = 25) -> tuple[list[Scored], list[Fact]]:
        """Return (ranked candidates to share with the engine, facts withheld for privacy)."""
        ok, held = allowed(facts, max_sensitivity)
        return self.rank(prompt, ok)[:limit], held
