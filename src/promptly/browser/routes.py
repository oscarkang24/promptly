"""/api/browser: the shell writes behavior here; agents read context from here."""

from __future__ import annotations

import hmac
import time
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from ..models import Sensitivity
from .pack import pack_to_markdown
from .service import BrowserService, ShellError


def _service() -> BrowserService:
    from ..server import get_promptly  # shares the profile store and vector cache with the rewriter

    return get_promptly().browser_service


def service(x_promptly_token: str | None = Header(default=None)) -> BrowserService:
    svc = _service()
    if not x_promptly_token or not hmac.compare_digest(x_promptly_token, svc.token):
        raise HTTPException(401, "missing or wrong X-Promptly-Token (see ~/.promptly/browser.token)")
    return svc


router = APIRouter(prefix="/api/browser", tags=["browser"])

Actor = Literal["user", "agent"]


class VisitStart(BaseModel):
    url: str
    title: str = ""
    tab_id: str | None = None
    actor: Actor = "user"
    from_url: str | None = None
    private: bool = False


class VisitEnd(BaseModel):
    visit_id: int
    dwell_ms: int = Field(ge=0)
    scroll: float = Field(default=0, ge=0, le=1)


class Content(BaseModel):
    url: str
    title: str = ""
    text: str = Field(default="", max_length=200_000)
    private: bool = False


class EventIn(BaseModel):
    kind: Literal["highlight", "search", "copy", "bookmark", "note"]
    text: str = Field(max_length=10_000)
    url: str | None = None
    actor: Actor = "user"
    private: bool = False


class ContextIn(BaseModel):
    task: str
    max_tokens: int = Field(default=1500, ge=100, le=20000)
    max_sensitivity: Sensitivity = "personal"
    format: Literal["json", "markdown"] = "json"


class PolicyIn(BaseModel):
    paused: bool | None = None
    blocked: list[str] | None = None
    retention_days: int | None = Field(default=None, ge=1, le=3650)


class ForgetIn(BaseModel):
    url: str | None = None
    domain: str | None = None
    since_minutes: float | None = Field(default=None, gt=0)
    everything: bool = False


@router.post("/visit/start")
def visit_start(b: VisitStart, svc: BrowserService = Depends(service)):
    return {"visit_id": svc.memory.start_visit(b.url, b.title, tab_id=b.tab_id, actor=b.actor,
                                               from_url=b.from_url, private=b.private)}


@router.post("/visit/end")
def visit_end(b: VisitEnd, svc: BrowserService = Depends(service)):
    svc.memory.end_visit(b.visit_id, dwell_ms=b.dwell_ms, scroll=b.scroll)
    return {"ok": True}


@router.post("/content")
def content(b: Content, svc: BrowserService = Depends(service)):
    return {"stored": svc.memory.set_content(b.url, b.title, b.text, private=b.private)}


@router.post("/event")
def event(b: EventIn, svc: BrowserService = Depends(service)):
    return {"stored": svc.memory.add_event(b.kind, b.text, b.url, actor=b.actor, private=b.private)}


@router.get("/search")
def search(q: str, k: int = 8, include_agent: bool = False, svc: BrowserService = Depends(service)):
    return [h.as_dict() for h in svc.memory.search(q, k=min(k, 50), include_agent=include_agent)]


@router.post("/context")
def context(b: ContextIn, svc: BrowserService = Depends(service)):
    pack = svc.context(b.task, b.max_tokens, b.max_sensitivity)
    return PlainTextResponse(pack_to_markdown(pack)) if b.format == "markdown" else pack


@router.get("/recent")
def recent(limit: int = 20, include_agent: bool = False, svc: BrowserService = Depends(service)):
    return svc.memory.recent(min(limit, 200), actor=None if include_agent else "user")


@router.get("/session")
def session(svc: BrowserService = Depends(service)):
    return svc.memory.current_session()


@router.get("/interests")
def interests(svc: BrowserService = Depends(service)):
    return svc.memory.interests()


@router.get("/suggest-facts")
def suggest_facts(svc: BrowserService = Depends(service)):
    return svc.memory.suggest_facts(f.text for f in svc.store.all())


@router.get("/stats")
def stats(svc: BrowserService = Depends(service)):
    return svc.memory.stats()


@router.get("/policy")
def get_policy(svc: BrowserService = Depends(service)):
    p = svc.memory.policy
    return {"paused": p.paused, "blocked": p.blocked, "retention_days": p.retention_days}


@router.post("/policy")
def set_policy(b: PolicyIn, svc: BrowserService = Depends(service)):
    p = svc.memory.policy
    if b.paused is not None:
        p.paused = b.paused
    if b.blocked is not None:
        p.blocked = [d.strip().lower() for d in b.blocked if d.strip()]
    if b.retention_days is not None:
        p.retention_days = b.retention_days
    p.save()
    return get_policy(svc)


@router.post("/forget")
def forget(b: ForgetIn, svc: BrowserService = Depends(service)):
    since = time.time() - b.since_minutes * 60 if b.since_minutes else None
    if not (b.url or b.domain or since or b.everything):
        raise HTTPException(400, "say what to forget: url, domain, since_minutes or everything")
    return {"pages_removed": svc.memory.forget(url=b.url, domain=b.domain, since=since, everything=b.everything)}


@router.get("/shell/{path:path}")
def shell_get(path: str, svc: BrowserService = Depends(service)):
    """Read-only passthrough to the shell (e.g. tabs), for agents that only speak to the daemon."""
    if path not in ("tabs",):
        raise HTTPException(404)
    try:
        return svc.shell.call("GET", "/" + path)
    except ShellError as e:
        raise HTTPException(503, str(e))
