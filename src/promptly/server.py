"""HTTP API and web UI. Run with `promptly serve`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .core import Promptly
from .engines import TARGET_STYLES, EngineError
from .models import Category, Fact, FactDraft, RewriteRequest, RewriteResult, Sensitivity

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Promptly", version="0.1.0")


@lru_cache
def get_promptly() -> Promptly:
    return Promptly()


class FactIn(BaseModel):
    text: str
    category: Category = "other"
    tags: list[str] = []
    sensitivity: Sensitivity = "personal"


class FactPatch(BaseModel):
    text: str | None = None
    category: Category | None = None
    tags: list[str] | None = None
    sensitivity: Sensitivity | None = None


class LearnIn(BaseModel):
    text: str


class AcceptIn(BaseModel):
    facts: list[FactDraft]


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    p = get_promptly()
    return {"engine": p.engine.name, "targets": sorted(TARGET_STYLES), "facts": len(p.store.all())}


@app.post("/api/rewrite", response_model=RewriteResult)
def rewrite(req: RewriteRequest):
    try:
        return get_promptly().rewrite(req)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except EngineError as e:
        raise HTTPException(502, str(e))


@app.get("/api/facts", response_model=list[Fact])
def list_facts():
    return get_promptly().store.all()


@app.post("/api/facts", response_model=Fact)
def add_fact(body: FactIn):
    return get_promptly().store.add(Fact(**body.model_dump()))


@app.patch("/api/facts/{fact_id}", response_model=Fact)
def patch_fact(fact_id: str, body: FactPatch):
    f = get_promptly().store.update(fact_id, **body.model_dump())
    if not f:
        raise HTTPException(404, "No such fact")
    return f


@app.delete("/api/facts/{fact_id}")
def delete_fact(fact_id: str):
    if not get_promptly().store.delete(fact_id):
        raise HTTPException(404, "No such fact")
    return {"ok": True}


@app.post("/api/learn", response_model=list[FactDraft])
def learn(body: LearnIn):
    try:
        return get_promptly().learn(body.text)
    except EngineError as e:
        raise HTTPException(502, str(e))


@app.post("/api/learn/accept", response_model=list[Fact])
def accept(body: AcceptIn):
    return get_promptly().accept(body.facts)
