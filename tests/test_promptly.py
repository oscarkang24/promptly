from types import SimpleNamespace

import numpy as np

import pytest
from fastapi.testclient import TestClient

from promptly import server
from promptly.cli import main as cli_main
from promptly.context import Retriever
from promptly.embeddings import HashingEmbedder, VectorCache
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

    def rewrite(self, prompt, facts, target, answers, relevance=None):
        self.seen_facts = facts
        self.relevance = relevance
        return RewriteDraft(already_clear=False, gaps=["done", "done"], intent=_intent(), rewritten_prompt="  better  ",
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
    cands, held = Retriever().select("anything", store.all(), "public")
    assert [r.fact.id for r in cands] == ["pref"] and {f.id for f in held} == {"stack", "health"}


def test_hallucinated_and_duplicate_ids_dropped_and_output_cleaned(store):
    res = Promptly(store, RecordingEngine(used=["stack", "nope", "stack"])).rewrite(RewriteRequest(prompt="fix my go bug"))
    assert [f.id for f in res.injected] == ["stack"]
    assert res.rewritten_prompt == "better"
    assert len(res.clarifying_questions) == 3
    assert res.gaps == ["done"]
    assert res.injected[0].relevance != 0


def test_pinned_fact_ids(store):
    engine = RecordingEngine(used=[])
    Promptly(store, engine).rewrite(RewriteRequest(prompt="x", fact_ids=["pref"]))
    assert [f.id for f in engine.seen_facts] == ["pref"]


def test_shortlist_prefers_relevant():
    facts = [Fact(text=f"Unrelated hobby number {i}", category="other") for i in range(40)]
    facts.append(Fact(id="k8s", text="Runs services on Kubernetes", category="project", tags=["kubernetes"]))
    cands, _ = Retriever().select("why does my kubernetes deploy keep crashing", facts, "personal", limit=5)
    assert cands[0].fact.id == "k8s" and len(cands) == 5


# -- latent vectors ------------------------------------------------------------

def test_hashing_embedder_is_normalized_and_deterministic():
    e = HashingEmbedder()
    a, b = e.embed(["hello world", "hello world"]), e.embed(["hello world"])
    assert abs(float(np.linalg.norm(a[0])) - 1) < 1e-5 and np.allclose(a[0], b[0])
    assert e.embed([""]).shape == (1, e.dim)


def test_chinese_retrieval():
    facts = [
        Fact(id="zh", text="我是素食者，不吃肉", category="constraint"),
        Fact(id="go", text="Backend engineer writing Go", category="work"),
        Fact(id="cat", text="家里养了一只猫", category="other"),
    ]
    ranked = Retriever().rank("帮我做一周的素食菜谱", facts)
    assert ranked[0].fact.id == "zh"


def test_feedback_moves_ranking_for_similar_prompts_only(tmp_path):
    r = Retriever(home=tmp_path)
    facts = [
        Fact(id="a", text="Enjoys hiking and trail running", category="other"),
        Fact(id="b", text="Enjoys hiking with a dog named Rex", category="other"),
    ]
    prompt = "suggest a weekend hiking trip near the mountains"
    first = [x.fact.id for x in r.rank(prompt, facts)]
    loser = first[1]
    for _ in range(3):
        r.feedback.add(prompt, loser, 1.0)
        r.feedback.add(prompt, first[0], -1.0)
    near = r.rank("plan a hiking trip near the mountains this weekend", facts)
    assert near[0].fact.id == loser and near[0].feedback > 0
    far = r.rank("refactor this sql query", facts)
    assert all(x.feedback == 0 for x in far)
    # Feedback survives a restart.
    assert len(Retriever(home=tmp_path).feedback.all()) == 6


def test_vector_cache_persists_and_is_model_scoped(tmp_path):
    path = tmp_path / "v.json"
    calls = []

    class Counting(HashingEmbedder):
        def embed(self, texts):
            calls.append(list(texts))
            return super().embed(texts)

    c = VectorCache(Counting(), path)
    c.get_many(["x", "y", "x"])
    assert calls == [["x", "y"]]
    c2 = VectorCache(Counting(), path)
    c2.get("x")
    assert calls == [["x", "y"]]  # served from disk
    VectorCache(HashingEmbedder(dim=64), path).get("x")  # different model -> recompute
    assert VectorCache(HashingEmbedder(dim=64), path).get("x").shape == (64,)


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

def test_offline_engine_triage(store):
    p = Promptly(store, OfflineEngine())
    vague = p.rewrite(RewriteRequest(prompt="fix it"))
    assert not vague.already_clear and "what" in vague.gaps and vague.clarifying_questions
    clear = p.rewrite(RewriteRequest(prompt="Write a short bullet list of steps for rotating my Postgres credentials in our Go service"))
    assert clear.already_clear and clear.gaps == [] and clear.clarifying_questions == []


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
    out = eng.rewrite("fix my code", facts, "claude", {"Which file?": "main.go"}, {"f1": 0.734})
    assert out is draft
    k = msgs.kwargs
    assert k["model"] == "claude-opus-5-5"
    assert k["output_format"] is RewriteDraft
    assert k["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in k["betas"]
    content = k["messages"][0]["content"]
    assert '<fact id="f1" category="work" relevance="0.73">Uses &lt;Go&gt; &amp; Rust</fact>' in content
    assert "Triage" in k["system"]
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
    assert r["embedder"].startswith("hashing")
    assert c.post("/api/feedback", json={"prompt": "x", "fact_id": "stack", "label": 1}).json() == {"ok": True}
    assert c.post("/api/feedback", json={"prompt": "x", "fact_id": "nope", "label": 1}).status_code == 404
    assert c.post("/api/feedback", json={"prompt": "x", "fact_id": "stack", "label": 5}).status_code == 422
    assert c.post("/api/rewrite", json={"prompt": ""}).status_code == 400
    drafts = c.post("/api/learn", json={"text": "I prefer dark mode."}).json()
    assert c.post("/api/learn/accept", json={"facts": drafts}).status_code == 200
    assert c.delete(f"/api/facts/{new['id']}").json() == {"ok": True}
    assert c.delete("/api/facts/missing").status_code == 404


def test_cli_roundtrip(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PROMPTLY_HOME", str(tmp_path))
    assert cli_main(["fact", "add", "Prefers short answers", "--category", "preference", "--sensitivity", "public"]) == 0
    assert cli_main(["rewrite", "explain how dns works", "--json"]) == 0
    out = capsys.readouterr().out
    assert '"Prefers short answers"' in out or "Prefers short answers" in out
    assert cli_main(["learn", "I work at Initech.", "-y"]) == 0
    assert "Works at Initech" in capsys.readouterr().out
    fid = Promptly().store.all()[0].id
    assert cli_main(["feedback", fid, "1", "explain how dns works"]) == 0
