"""The Promptly pipeline: privacy gate -> shortlist -> engine -> validation."""

from __future__ import annotations

from .context import select
from .engines import Engine, default_engine
from .models import Fact, FactDraft, InjectedFact, RewriteRequest, RewriteResult
from .store import FactStore


class Promptly:
    def __init__(self, store: FactStore | None = None, engine: Engine | None = None, shortlist_size: int = 25):
        self.store = store or FactStore()
        self.engine = engine or default_engine()
        self.shortlist_size = shortlist_size

    def rewrite(self, req: RewriteRequest) -> RewriteResult:
        if not req.prompt.strip():
            raise ValueError("Prompt is empty.")

        facts = self.store.all()
        if req.fact_ids is not None:
            pinned = set(req.fact_ids)
            facts = [f for f in facts if f.id in pinned]
        candidates, withheld = select(req.prompt, facts, req.max_sensitivity, self.shortlist_size)

        draft = self.engine.rewrite(req.prompt, candidates, req.target, req.answers)

        # Only accept fact ids we actually sent; drop anything the model made up.
        by_id = {f.id: f for f in candidates}
        injected, seen = [], set()
        for used in draft.used_facts:
            fact = by_id.get(used.fact_id)
            if fact and fact.id not in seen:
                seen.add(fact.id)
                injected.append(InjectedFact(id=fact.id, text=fact.text, category=fact.category, reason=used.reason))

        return RewriteResult(
            original_prompt=req.prompt,
            rewritten_prompt=draft.rewritten_prompt.strip(),
            intent=draft.intent,
            injected=injected,
            assumptions=draft.assumptions,
            clarifying_questions=draft.clarifying_questions[:3],
            confidence=draft.confidence,
            engine=self.engine.name,
            shared_fact_ids=[f.id for f in candidates],
            withheld_fact_ids=[f.id for f in withheld],
        )

    def learn(self, text: str) -> list[FactDraft]:
        """Propose facts from free text. Nothing is saved until the caller adds them."""
        if not text.strip():
            return []
        existing = {f.text.strip().lower() for f in self.store.all()}
        return [d for d in self.engine.learn(text) if d.text.strip().lower() not in existing]

    def accept(self, drafts: list[FactDraft], source: str = "learned") -> list[Fact]:
        return [self.store.add(Fact(**d.model_dump(), source=source)) for d in drafts]
