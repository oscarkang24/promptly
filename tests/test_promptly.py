from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from promptly import server
from promptly.cli import main as cli_main
from promptly.context import select
from promptly.core import Promptly
from promptly.engines import ClaudeEngine, EngineError, OfflineEngine
from promptly.models import Fact, FactDraft, Intent, RewriteDraft, RewriteRequest, UsedFact
from promptly.store import FactStore


@pytest.fixture
def store(tmp_path):
    s = FactStore(tmp_path / "profile.json")
    s.add(Fact(id="pref", text="Prefers concise answers with code first", category="preference", sensitivity="public"))
    s.add(Fact(id="stack", text="Backend engineer working mostly in Go and Postgres", category="work", tags=["go", "postgres"]))
    s.add(Fact(id="health", text="Has type 1 diabetes", category="constraint", sensitivity="sensitive"))
    return s


def _intent():
    return Intent(surface_request="x", true_goal="y", task_type="coding", audience="the user",
                  desired_output="code", constraints=[], success_criteria=[])


class RecordingEngine:
    name = "fake"

    def __init__(self, used):
        self.used = used
        self.seen_facts = None

    def rewrite(self, prompt, facts, target, answers):
        self.seen_facts = facts
        return RewriteDraft(intent=_intent(), rewritten_prompt="  better  ",
                            used_facts=[UsedFact(fact_id=i, reason="r") for i in self.used],
                            assumptions=[], clarifying_questions=["a", "b", "c", "d"], confidence="high")

    def learn(self, text):
        return [FactDraft(text="Prefers concise answers with code first", category="preference", tags=[], sensitivity="public"),
                FactDraft(text="Lives in Lisbon", category="identity", tags=[], sensitivity="personal")]


# -- store -------------------------------------------------------------------

def test_store_crud_and_dedupe(tmp_path):
    s = FactStore(tmp_path / "p.json")
    a = s.add(Fact(text="Likes tea"))
    assert s.add(Fact(text="  likes TEA ")).id == a.id
    assert s.update(a.id, text="Likes green tea").text == "Likes green tea"
    assert FactStore(tmp_path / "p.json").get(a.id).text == "Likes green tea"
    assert s.delete(a.id) and not s.delete(a.id)
    assert s.all() == []


# -- privacy + selection -----------------------------------------------------

def test_sensitive_facts_never_reach_engine(store):
    engine = RecordingEngine(used=["health"])
    res = Promptly(store, engine).rewrite(RewriteRequest(prompt="plan meals for my week"))
    assert "health" not in {f.id for f in engine.seen_facts}
    assert res.withheld_fact_ids == ["health"]
    assert res.injected == []  # the engine claimed it, but it was never shared


def test_sensitive_allowed_when_opted_in(store):
    engine = RecordingEngine(used=["health"])
    res = Promptly(store, engine).rewrite(RewriteRequest(prompt="plan meals", max_sensitivity="sensitive"))
    assert [f.id for f in res.injected] == ["health"]


def test_public_only(store):
    cands, held = select("anything", store.all(), "public")
    assert [f.id for f in cands] == ["pref"] and {f.id for f in held} == {"stack", "health"}


def test_hallucinated_and_duplicate_ids_dropped_and_output_cleaned(store):
    res = Promptly(store, RecordingEngine(used=["stack", "nope", "stack"])).rewrite(RewriteRequest(prompt="fix my go bug"))
    assert [f.id for f in res.injected] == ["stack"]
    assert res.rewritten_prompt == "better"
    assert len(res.clarifying_questions) == 3


def test_pinned_fact_ids(store):
    engine = RecordingEngine(used=[])
    Promptly(store, engine).rewrite(RewriteRequest(prompt="x", fact_ids=["pref"]))
    assert [f.id for f in engine.seen_facts] == ["pref"]


def test_shortlist_prefers_relevant(tmp_path):
    s = FactStore(tmp_path / "p.json")
    for i in range(40):
        s.add(Fact(text=f"Unrelated hobby number {i}", category="other"))
    s.add(Fact(id="k8s", text="Runs services on Kubernetes", category="project", tags=["kubernetes"]))
    cands, _ = select("why does my kubernetes deploy keep crashing", s.all(), "personal", limit=5)
    assert cands[0].id == "k8s"


def test_empty_prompt_rejected(store):
    with pytest.raises(ValueError):
        Promptly(store, RecordingEngine([])).rewrite(RewriteRequest(prompt="   "))


def test_learn_skips_known_facts(store):
    p = Promptly(store, RecordingEngine([]))
    drafts = p.learn("blah")
    assert [d.text for d in drafts] == ["Lives in Lisbon"]
    saved = p.accept(drafts)
    assert saved[0].source == "learned" and store.get(saved[0].id)


# -- engines -----------------------------------------------------------------

def test_offline_engine_injects_relevant_facts(store):
    res = Promptly(store, OfflineEngine()).rewrite(RewriteRequest(prompt="write a go function to retry postgres queries", target="claude"))
    assert "stack" in {f.id for f in res.injected}
    assert "<task>" in res.rewritten_prompt and "Postgres" in res.rewritten_prompt
    assert res.intent.task_type == "coding"


def test_offline_learn():
    drafts = OfflineEngine().learn("I'm a product designer. I prefer bullet points. I work at Acme.")
    texts = {d.text for d in drafts}
    assert {"Is a product designer", "Prefers bullet points", "Works at Acme"} <= texts


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _fake_client(response):
    msgs = FakeMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=msgs)), msgs


def test_claude_engine_request_shape(store):
    draft = RecordingEngine([]).rewrite("", [], "", {})
    client, msgs = _fake_client(SimpleNamespace(stop_reason="end_turn", parsed_output=draft))
    eng = ClaudeEngine(client=client)
    facts = [Fact(id="f1", text="Uses <Go> & Rust", category="work")]
    out = eng.rewrite("fix my code", facts, "claude", {"Which file?": "main.go"})
    assert out is draft
    k = msgs.kwargs
    assert k["model"] == "claude-opus-5-5"
    assert k["output_format"] is RewriteDraft
    assert k["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in k["betas"]
    content = k["messages"][0]["content"]
    assert '<fact id="f1" category="work">Uses &lt;Go&gt; &amp; Rust</fact>' in content
    assert "<a>main.go</a>" in content and "XML tags" in k["system"]


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_claude_engine_bad_stops(stop):
    client, _ = _fake_client(SimpleNamespace(stop_reason=stop, parsed_output=None))
    with pytest.raises(EngineError):
        ClaudeEngine(client=client).rewrite("x", [], "generic", {})


# -- API + CLI ---------------------------------------------------------------

def test_http_api(store, monkeypatch):
    p = Promptly(store, OfflineEngine())
    monkeypatch.setattr(server, "get_promptly", lambda: p)
    c = TestClient(server.app)
    assert c.get("/").status_code == 200
    assert c.get("/api/status").json()["engine"] == "offline"
    new = c.post("/api/facts", json={"text": "Writes in British English", "category": "preference", "sensitivity": "public"}).json()
    assert c.patch(f"/api/facts/{new['id']}", json={"tags": ["spelling"]}).json()["tags"] == ["spelling"]
    r = c.post("/api/rewrite", json={"prompt": "draft an email to my team about the go migration"}).json()
    assert r["engine"] == "offline" and "health" in r["withheld_fact_ids"]
    assert c.post("/api/rewrite", json={"prompt": ""}).status_code == 400
    drafts = c.post("/api/learn", json={"text": "I prefer dark mode."}).json()
    assert c.post("/api/learn/accept", json={"facts": drafts}).status_code == 200
    assert c.delete(f"/api/facts/{new['id']}").json() == {"ok": True}
    assert c.delete("/api/facts/missing").status_code == 404


def test_cli_roundtrip(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PROMPTLY_HOME", str(tmp_path))
    monkeypatch.setenv("PROMPTLY_ENGINE", "offline")
    assert cli_main(["fact", "add", "Prefers short answers", "--category", "preference", "--sensitivity", "public"]) == 0
    assert cli_main(["rewrite", "explain how dns works", "--json"]) == 0
    out = capsys.readouterr().out
    assert '"Prefers short answers"' in out or "Prefers short answers" in out
    assert cli_main(["learn", "I work at Initech.", "-y"]) == 0
    assert "Works at Initech" in capsys.readouterr().out
