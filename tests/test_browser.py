import time

import pytest
from fastapi.testclient import TestClient

from promptly.browser.memory import BrowserMemory, chunk_text
from promptly.browser.pack import build_pack, pack_to_markdown
from promptly.browser.policy import Policy, clean_url
from promptly.context import Retriever
from promptly.core import Promptly
from promptly.embeddings import HashingEmbedder
from promptly.engines import OfflineEngine
from promptly.models import Fact, RewriteRequest
from promptly.store import FactStore

RUST = ("Async Rust with Tokio. The tokio runtime schedules futures across worker threads. "
        "Use tokio::spawn for concurrent tasks and select! to race them. Backpressure with bounded channels.")
COOK = "A weeknight pasta recipe with garlic, olive oil and chili flakes. Boil the spaghetti until al dente."


@pytest.fixture
def mem(tmp_path):
    return BrowserMemory(tmp_path / "b.db", HashingEmbedder(), Policy())


def visit(m, url, title, text, dwell_ms=120_000, actor="user", now=None):
    vid = m.start_visit(url, title, actor=actor, now=now)
    m.set_content(url, title, text)
    m.end_visit(vid, dwell_ms=dwell_ms, scroll=0.8, now=now)
    return vid


def test_policy_strips_secrets_and_blocks():
    assert clean_url("https://Ex.com/a?q=1&access_token=xyz&code=9#frag") == "https://ex.com/a?q=1"
    p = Policy()
    assert not p.may_record("https://www.chase.com/login")
    assert not p.may_record("https://accounts.google.com/")
    assert not p.may_record("file:///etc/passwd")
    assert not p.may_record("https://example.com", private=True)
    assert p.may_record("https://docs.rs/tokio")
    p.paused = True
    assert not p.may_record("https://docs.rs/tokio")


def test_blocked_and_private_visits_are_not_stored(mem):
    assert mem.start_visit("https://mybank.com/acct") is None
    assert mem.start_visit("https://example.com", private=True) is None
    assert not mem.set_content("https://example.com", "x", "secret", private=True)
    assert mem.stats()["pages"] == 0


def test_chunking_bounds():
    chunks = chunk_text("Sentence one. " * 400)
    assert chunks and all(len(c) <= 720 for c in chunks)
    assert chunk_text("") == []


def test_search_finds_by_meaning_and_explains(mem):
    visit(mem, "https://tokio.rs/tutorial", "Tokio tutorial", RUST)
    visit(mem, "https://food.example/pasta", "Pasta tonight", COOK, dwell_ms=5_000)
    hits = mem.search("how do I run concurrent tasks with tokio")
    assert hits[0].url == "https://tokio.rs/tutorial"
    assert hits[0].why
    assert "tokio" in hits[0].snippet.lower()


def test_agent_pages_hidden_from_user_search_and_interests(mem):
    visit(mem, "https://agent.example/tokio", "Tokio notes", RUST, actor="agent")
    assert mem.search("tokio runtime") == []
    assert mem.search("tokio runtime", include_agent=True)
    assert mem.interests()["domains"] == []


def test_dwell_only_counts_for_user_and_once(mem):
    vid = mem.start_visit("https://a.example/x", "x")
    mem.end_visit(vid, dwell_ms=60_000)
    mem.end_visit(vid, dwell_ms=60_000)  # duplicate end is ignored
    assert mem.db.execute("SELECT dwell_ms FROM pages").fetchone()[0] == 60_000


def test_sessions_split_on_gap(mem):
    t0 = time.time() - 7200
    visit(mem, "https://a.example/1", "one", "alpha", now=t0)
    visit(mem, "https://a.example/2", "two", "beta", now=t0 + 60)
    visit(mem, "https://a.example/3", "three", "gamma", now=t0 + 3600)
    sessions = [r["session"] for r in mem.recent()]
    assert sessions[0] != sessions[1] and sessions[1] == sessions[2]


def test_current_session_and_searches(mem):
    visit(mem, "https://tokio.rs/tutorial", "Tokio tutorial", RUST)
    mem.add_event("search", "tokio select macro", "https://duckduckgo.com/?q=tokio")
    visit(mem, "https://tokio.rs/tutorial", "Tokio tutorial", RUST, dwell_ms=30_000)
    trail = mem.current_session()
    assert trail[0]["title"] == "Tokio tutorial"
    assert trail[0]["visits"] == 2 and trail[0]["dwell_s"] == 150
    assert {"search": "tokio select macro"} in trail


def test_interests_and_suggestions(mem):
    for i in range(3):
        visit(mem, f"https://tokio.rs/p{i}", f"Tokio async runtime part {i}", RUST, dwell_ms=600_000)
    ints = mem.interests()
    assert ints["domains"][0]["domain"] == "tokio.rs"
    assert any(t["topic"] == "tokio" for t in ints["topics"])
    drafts = mem.suggest_facts()
    assert drafts and "tokio" in drafts[0].text.lower()
    assert mem.suggest_facts([d.text for d in drafts]) == [] or len(mem.suggest_facts([d.text for d in drafts])) < len(drafts)


def test_forget(mem):
    visit(mem, "https://tokio.rs/a", "a", RUST)
    visit(mem, "https://docs.tokio.rs/b", "b", RUST)
    visit(mem, "https://food.example/c", "c", COOK)
    assert mem.forget(domain="tokio.rs") == 2
    assert mem.stats()["passages"] == 1
    assert mem.forget(everything=True) == 1
    assert mem.stats()["pages"] == 0


def test_retention(mem):
    old = time.time() - 200 * 86400
    visit(mem, "https://old.example", "old", "ancient", now=old)
    mem.db.execute("UPDATE pages SET last_seen = ?", (old,))
    assert mem.apply_retention() == 1


def test_context_pack_respects_budget_and_privacy(mem, tmp_path):
    visit(mem, "https://tokio.rs/tutorial", "Tokio tutorial", RUST)
    mem.add_event("highlight", "Backpressure with bounded channels", "https://tokio.rs/tutorial")
    facts = [Fact(text="Writes Rust services with tokio", category="skill", sensitivity="public"),
             Fact(text="Has a tokio-related medical condition", category="other", sensitivity="sensitive")]
    retr = Retriever(HashingEmbedder(), None)
    pack = build_pack(mem, "add backpressure to my tokio service", facts=facts, retriever=retr)
    assert pack["relevant"][0]["url"] == "https://tokio.rs/tutorial"
    assert pack["highlights"]
    assert len(pack["withheld_fact_ids"]) == 1
    assert all("medical" not in f["text"] for f in pack["profile"])
    md = pack_to_markdown(pack)
    assert "Tokio tutorial" in md and "medical" not in md
    tiny = build_pack(mem, "tokio", max_tokens=100)
    assert tiny["token_estimate"] <= 100


def test_rewrite_can_use_browsing(mem, tmp_path):
    visit(mem, "https://tokio.rs/tutorial", "Tokio tutorial", RUST)
    store = FactStore(tmp_path / "profile.json")
    p = Promptly(store=store, engine=OfflineEngine(), retriever=Retriever(HashingEmbedder(), tmp_path), browsing=mem)
    res = p.rewrite(RewriteRequest(prompt="help me use tokio spawn for concurrent tasks", use_browsing=True))
    assert any(fid.startswith("web") for fid in res.shared_fact_ids)
    res2 = p.rewrite(RewriteRequest(prompt="help me use tokio spawn for concurrent tasks"))
    assert not any(fid.startswith("web") for fid in res2.shared_fact_ids)


def test_api_requires_token_and_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTLY_HOME", str(tmp_path))
    from promptly import server

    server.get_promptly.cache_clear()
    c = TestClient(server.app)
    assert c.get("/api/browser/stats").status_code == 401
    assert c.get("/api/browser/stats", headers={"X-Promptly-Token": "nope"}).status_code == 401
    tok = (tmp_path / "browser.token").read_text().strip()
    assert (tmp_path / "browser.token").stat().st_mode & 0o077 == 0
    h = {"X-Promptly-Token": tok}
    vid = c.post("/api/browser/visit/start", json={"url": "https://tokio.rs/t", "title": "Tokio"}, headers=h).json()["visit_id"]
    c.post("/api/browser/content", json={"url": "https://tokio.rs/t", "title": "Tokio", "text": RUST}, headers=h)
    c.post("/api/browser/visit/end", json={"visit_id": vid, "dwell_ms": 90000, "scroll": 0.5}, headers=h)
    hits = c.get("/api/browser/search", params={"q": "tokio spawn"}, headers=h).json()
    assert hits[0]["url"] == "https://tokio.rs/t"
    md = c.post("/api/browser/context", json={"task": "tokio concurrency", "format": "markdown"}, headers=h).text
    assert "Tokio" in md
    c.post("/api/browser/policy", json={"paused": True}, headers=h)
    assert c.post("/api/browser/visit/start", json={"url": "https://x.example"}, headers=h).json()["visit_id"] is None
    assert c.post("/api/browser/forget", json={"everything": True}, headers=h).json()["pages_removed"] == 1
    server.get_promptly.cache_clear()


def test_mcp_tools_registered():
    import asyncio

    from promptly.browser.mcp_server import mcp

    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert {"get_context", "search_history", "open_tab", "read_tab", "click", "type_text"} <= names
