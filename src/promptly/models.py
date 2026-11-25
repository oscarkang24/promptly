"""Data models shared by the store, the engines and the API."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

Sensitivity = Literal["public", "personal", "sensitive"]
SENSITIVITY_RANK: dict[str, int] = {"public": 0, "personal": 1, "sensitive": 2}

Category = Literal[
    "identity",     # name, role, location, languages
    "work",         # job, company, team, stack
    "project",      # things currently being built
    "preference",   # tone, format, length, tools
    "skill",        # expertise level in topics
    "goal",         # what the user is trying to achieve
    "constraint",   # budget, deadlines, dietary, accessibility
    "other",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Fact(BaseModel):
    """One piece of personal context Promptly may inject into a prompt."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    text: str
    category: Category = "other"
    tags: list[str] = Field(default_factory=list)
    sensitivity: Sensitivity = "personal"
    source: str = "manual"
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


class FactDraft(BaseModel):
    """A fact proposed by the learner, before the user accepts it."""

    text: str
    category: Category
    tags: list[str]
    sensitivity: Sensitivity


class Intent(BaseModel):
    surface_request: str = Field(description="What the user literally typed, restated in one line.")
    true_goal: str = Field(description="What the user is actually trying to accomplish.")
    task_type: str = Field(description="e.g. writing, coding, research, planning, analysis, advice, learning.")
    audience: str = Field(description="Who the output is for; 'the user' if for themselves.")
    desired_output: str = Field(description="Format, length and shape of the ideal answer.")
    constraints: list[str] = Field(description="Hard requirements implied by the request or the facts.")
    success_criteria: list[str] = Field(description="How the user will judge the answer good.")


class UsedFact(BaseModel):
    fact_id: str
    reason: str = Field(description="Why this fact changes the answer for this prompt.")


class RewriteDraft(BaseModel):
    """What an engine returns. Validated and post-processed into RewriteResult."""

    intent: Intent
    rewritten_prompt: str
    used_facts: list[UsedFact]
    assumptions: list[str]
    clarifying_questions: list[str]
    confidence: Literal["low", "medium", "high"]


class InjectedFact(BaseModel):
    id: str
    text: str
    category: str
    reason: str


class RewriteResult(BaseModel):
    original_prompt: str
    rewritten_prompt: str
    intent: Intent
    injected: list[InjectedFact]
    assumptions: list[str]
    clarifying_questions: list[str]
    confidence: Literal["low", "medium", "high"]
    engine: str
    # Privacy transparency: exactly which facts left the machine, and which were held back.
    shared_fact_ids: list[str]
    withheld_fact_ids: list[str]


class RewriteRequest(BaseModel):
    prompt: str
    target: str = "generic"
    max_sensitivity: Sensitivity = "personal"
    answers: dict[str, str] = Field(default_factory=dict)
    fact_ids: list[str] | None = None  # pin an explicit set of facts; None = auto-select
