"""The Promptly pipeline: privacy gate -> latent-vector retrieval -> engine -> validation."""

from __future__ import annotations

from .context import Retriever
from .embeddings import default_embedder
from .engines import Engine, default_engine
from .models import Fact, FactDraft, InjectedFact, RewriteRequest, RewriteResult
from .store import FactStore

# Implicit feedback is weaker than an explicit 👍/👎.
COPY_SIGNAL = 0.3


def browsing_facts(memory, prompt: str, k: int = 4) -> list[Fact]:
    out = []
    for h in memory.search(prompt, k=k):
        if h.score < 0.3:
            continue
        out.append(Fact(id=f"web{h.page_id}", text=f"Recently read \"{h.title or h.url}\" ({h.url}): {h.snippet[:220]}",
                        category="project", tags=["browsing", h.domain], sensitivity="personal", source="browsing"))
    return out


class Promptly:
    def __init__(
        self,
        store: FactStore | None = None,
        engine: Engine | None = None,
        retriever: Retriever | None = None,
        shortlist_size: int = 25,
        browsing=None,
    ):
        self.store = store or FactStore()
        self.engine = engine or default_engine()
        self.retriever = retriever or Retriever(default_embedder(), self.store.path.parent)
        self.shortlist_size = shortlist_size
        # Optional BrowserMemory. Matching pages become temporary, never-saved facts for one rewrite.
        self.browsing = browsing

    def rewrite(self, req: RewriteRequest) -> RewriteResult:
        if not req.prompt.strip():
            raise ValueError("Prompt is empty.")

        facts = self.store.all()
        if req.fact_ids is not None:
            pinned = set(req.fact_ids)
            facts = [f for f in facts if f.id in pinned]
        if req.use_browsing and self.browsing is not None:
            facts = facts + browsing_facts(self.browsing, req.prompt)
        ranked, withheld = self.retriever.select(req.prompt, facts, req.max_sensitivity, self.shortlist_size)
        candidates = [r.fact for r in ranked]
        relevance = {r.fact.id: round(r.score, 3) for r in ranked}

        draft = self.engine.rewrite(req.prompt, candidates, req.target, req.answers, relevance)

        # Only accept fact ids we actually sent; drop anything the model made up.
        by_id = {f.id: f for f in candidates}
        injected, seen = [], set()
        for used in draft.used_facts:
            fact = by_id.get(used.fact_id)
            if fact and fact.id not in seen:
                seen.add(fact.id)
                injected.append(InjectedFact(id=fact.id, text=fact.text, category=fact.category,
                                             reason=used.reason, relevance=relevance[fact.id]))

        return RewriteResult(
            original_prompt=req.prompt,
            rewritten_prompt=draft.rewritten_prompt.strip(),
            already_clear=draft.already_clear,
            gaps=list(dict.fromkeys(draft.gaps)),
            intent=draft.intent,
            injected=injected,
            assumptions=draft.assumptions,
            clarifying_questions=draft.clarifying_questions[:3],
            confidence=draft.confidence,
            engine=self.engine.name,
            embedder=self.retriever.embedder.name,
            shared_fact_ids=[f.id for f in candidates],
            withheld_fact_ids=[f.id for f in withheld],
        )

    def feedback(self, prompt: str, fact_id: str, label: float) -> None:
        """Record that a fact helped (+) or was noise (-) for a prompt. Shapes future ranking."""
        if not self.store.get(fact_id):
            raise ValueError("No such fact.")
        self.retriever.feedback.add(prompt, fact_id, label)

    def learn(self, text: str) -> list[FactDraft]:
        """Propose facts from free text. Nothing is saved until the caller adds them."""
        if not text.strip():
            return []
        existing = {f.text.strip().lower() for f in self.store.all()}
        return [d for d in self.engine.learn(text) if d.text.strip().lower() not in existing]

    def accept(self, drafts: list[FactDraft], source: str = "learned") -> list[Fact]:
        return [self.store.add(Fact(**d.model_dump(), source=source)) for d in drafts]
