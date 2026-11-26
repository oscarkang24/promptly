"""Local behavior memory for the browser: SQLite + latent vectors.

What gets stored (only after Policy allows it):

    pages     one row per cleaned URL: title, readable text excerpt, totals (visits, dwell, scroll)
    chunks    the page text split into ~700-char passages, each with an embedding
    visits    one row per time a tab showed a page: dwell, scroll depth, who drove it (user/agent),
              the page it came from, and a session id (a gap of 30+ minutes starts a new session)
    events    explicit signals: highlights, searches, copies, bookmarks, notes

Agent-driven visits are stored with actor='agent' and never count toward what the browser
learns about *you* (engagement, interests, suggested facts), but agents can still search them.
"""

from __future__ import annotations

import math
import re
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

from ..context import tokens
from ..embeddings import Embedder, HashingEmbedder
from ..models import FactDraft
from .policy import Policy, clean_url, domain_of

Actor = Literal["user", "agent"]
EventKind = Literal["highlight", "search", "copy", "bookmark", "note"]

SESSION_GAP_S = 30 * 60
CHUNK_CHARS = 700
MAX_CHUNKS_PER_PAGE = 24
MAX_TEXT_CHARS = CHUNK_CHARS * MAX_CHUNKS_PER_PAGE
RECENCY_HALF_LIFE_DAYS = 14.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
  id INTEGER PRIMARY KEY,
  url TEXT UNIQUE NOT NULL,
  domain TEXT NOT NULL,
  title TEXT NOT NULL DEFAULT '',
  excerpt TEXT NOT NULL DEFAULT '',
  first_seen REAL NOT NULL,
  last_seen REAL NOT NULL,
  user_visits INTEGER NOT NULL DEFAULT 0,
  agent_visits INTEGER NOT NULL DEFAULT 0,
  dwell_ms INTEGER NOT NULL DEFAULT 0,
  max_scroll REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS pages_domain ON pages(domain);
CREATE TABLE IF NOT EXISTS chunks (
  page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
  idx INTEGER NOT NULL,
  text TEXT NOT NULL,
  model TEXT NOT NULL,
  vec BLOB NOT NULL,
  PRIMARY KEY (page_id, idx)
);
CREATE TABLE IF NOT EXISTS visits (
  id INTEGER PRIMARY KEY,
  page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
  tab_id TEXT,
  actor TEXT NOT NULL,
  started REAL NOT NULL,
  ended REAL,
  dwell_ms INTEGER NOT NULL DEFAULT 0,
  scroll REAL NOT NULL DEFAULT 0,
  from_page_id INTEGER,
  session_id INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS visits_started ON visits(started);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY,
  ts REAL NOT NULL,
  kind TEXT NOT NULL,
  page_id INTEGER REFERENCES pages(id) ON DELETE CASCADE,
  actor TEXT NOT NULL,
  text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
"""


def chunk_text(text: str) -> list[str]:
    """Split readable text into passages on paragraph/sentence boundaries."""
    text = re.sub(r"[ \t]+", " ", text or "").strip()[:MAX_TEXT_CHARS]
    if not text:
        return []
    parts = re.split(r"\n{1,}|(?<=[.!?。！？])\s+", text)
    chunks, cur = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(cur) + len(p) + 1 > CHUNK_CHARS and cur:
            chunks.append(cur)
            cur = ""
        cur = (cur + " " + p).strip() if cur else p[:CHUNK_CHARS]
    if cur:
        chunks.append(cur)
    return chunks[:MAX_CHUNKS_PER_PAGE]


def _recency(ts: float, now: float) -> float:
    return 0.5 ** (max(0.0, now - ts) / 86400 / RECENCY_HALF_LIFE_DAYS)


def _engagement(dwell_ms: int, visits: int, scroll: float, highlights: int) -> float:
    """0..~1. Minutes of attention matter most; repeat visits and highlights add."""
    return min(1.0, 0.45 * math.log1p(dwell_ms / 60000) + 0.12 * math.log1p(visits)
               + 0.15 * scroll + 0.2 * min(highlights, 3) / 3)


@dataclass
class Hit:
    page_id: int
    url: str
    title: str
    domain: str
    snippet: str
    score: float
    semantic: float
    engagement: float
    last_seen: float
    why: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"url": self.url, "title": self.title, "domain": self.domain, "snippet": self.snippet,
                "score": round(self.score, 3), "last_seen": _iso(self.last_seen), "why": self.why}


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


class BrowserMemory:
    def __init__(self, path: Path | None, embedder: Embedder | None = None, policy: Policy | None = None):
        self.embedder = embedder or HashingEmbedder()
        self.policy = policy or Policy()
        self._lock = threading.RLock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(SCHEMA)
        if path:
            try:
                path.chmod(0o600)
            except OSError:
                pass

    # ------------------------------------------------------------------ writes
    def _page(self, url: str, title: str = "", now: float | None = None) -> int:
        now = now or time.time()
        cur = self.db.execute("SELECT id FROM pages WHERE url = ?", (url,)).fetchone()
        if cur:
            if title:
                self.db.execute("UPDATE pages SET title = ?, last_seen = ? WHERE id = ?", (title[:300], now, cur["id"]))
            return cur["id"]
        return self.db.execute(
            "INSERT INTO pages (url, domain, title, first_seen, last_seen) VALUES (?, ?, ?, ?, ?)",
            (url, domain_of(url), title[:300], now, now),
        ).lastrowid

    def _session_for(self, actor: str, now: float) -> int:
        row = self.db.execute(
            "SELECT session_id, COALESCE(ended, started) AS t FROM visits WHERE actor = ? ORDER BY started DESC LIMIT 1",
            (actor,),
        ).fetchone()
        if row and now - row["t"] < SESSION_GAP_S:
            return row["session_id"]
        top = self.db.execute("SELECT COALESCE(MAX(session_id), 0) AS m FROM visits").fetchone()["m"]
        return top + 1

    def start_visit(self, url: str, title: str = "", *, tab_id: str | None = None, actor: Actor = "user",
                    from_url: str | None = None, private: bool = False, now: float | None = None) -> int | None:
        """A tab started showing `url`. Returns a visit id, or None if policy says don't record."""
        if not self.policy.may_record(url, private):
            return None
        now = now or time.time()
        url = clean_url(url)
        with self._lock, self.db:
            pid = self._page(url, title, now)
            col = "user_visits" if actor == "user" else "agent_visits"
            self.db.execute(f"UPDATE pages SET {col} = {col} + 1, last_seen = ? WHERE id = ?", (now, pid))
            from_id = None
            if from_url and self.policy.may_record(from_url):
                r = self.db.execute("SELECT id FROM pages WHERE url = ?", (clean_url(from_url),)).fetchone()
                from_id = r["id"] if r else None
            return self.db.execute(
                "INSERT INTO visits (page_id, tab_id, actor, started, from_page_id, session_id) VALUES (?, ?, ?, ?, ?, ?)",
                (pid, tab_id, actor, now, from_id, self._session_for(actor, now)),
            ).lastrowid

    def end_visit(self, visit_id: int, *, dwell_ms: int, scroll: float = 0.0, now: float | None = None) -> None:
        """The tab left the page. `dwell_ms` is foreground, visible time only."""
        now = now or time.time()
        dwell_ms = max(0, min(int(dwell_ms), 4 * 3600 * 1000))
        scroll = max(0.0, min(1.0, float(scroll)))
        with self._lock, self.db:
            v = self.db.execute("SELECT page_id, actor, ended FROM visits WHERE id = ?", (visit_id,)).fetchone()
            if not v or v["ended"] is not None:
                return
            self.db.execute("UPDATE visits SET ended = ?, dwell_ms = ?, scroll = ? WHERE id = ?",
                            (now, dwell_ms, scroll, visit_id))
            if v["actor"] == "user":
                self.db.execute(
                    "UPDATE pages SET dwell_ms = dwell_ms + ?, max_scroll = MAX(max_scroll, ?) WHERE id = ?",
                    (dwell_ms, scroll, v["page_id"]),
                )

    def set_content(self, url: str, title: str, text: str, *, private: bool = False) -> bool:
        """Store a page's readable text and embed its passages. Re-embeds only if the text changed."""
        if not self.policy.may_record(url, private):
            return False
        url = clean_url(url)
        chunks = chunk_text(text)
        with self._lock:
            with self.db:
                pid = self._page(url, title)
                excerpt = (chunks[0] if chunks else "")[:400]
                old = [r["text"] for r in self.db.execute("SELECT text FROM chunks WHERE page_id = ? AND model = ? ORDER BY idx", (pid, self.embedder.name))]
                self.db.execute("UPDATE pages SET excerpt = ? WHERE id = ?", (excerpt, pid))
                if old == chunks and old:
                    return True
            # Embed outside the transaction; title is prepended so passages keep their context.
            vecs = self.embedder.embed([f"{title}\n{c}" for c in chunks]) if chunks else np.zeros((0, self.embedder.dim))
            if not chunks:
                vecs = self.embedder.embed([title or url])
                chunks = [title or url]
            with self.db:
                self.db.execute("DELETE FROM chunks WHERE page_id = ?", (pid,))
                self.db.executemany(
                    "INSERT INTO chunks (page_id, idx, text, model, vec) VALUES (?, ?, ?, ?, ?)",
                    [(pid, i, c, self.embedder.name, v.astype(np.float32).tobytes()) for i, (c, v) in enumerate(zip(chunks, vecs))],
                )
        return True

    def add_event(self, kind: EventKind, text: str, url: str | None = None, *, actor: Actor = "user",
                  private: bool = False, now: float | None = None) -> bool:
        if url is not None and not self.policy.may_record(url, private):
            return False
        if url is None and (self.policy.paused or private):
            return False
        text = (text or "").strip()[:2000]
        if not text:
            return False
        now = now or time.time()
        with self._lock, self.db:
            pid = self._page(clean_url(url), now=now) if url else None
            self.db.execute("INSERT INTO events (ts, kind, page_id, actor, text) VALUES (?, ?, ?, ?, ?)",
                            (now, kind, pid, actor, text))
        return True

    # ------------------------------------------------------------------ forgetting
    def forget(self, *, url: str | None = None, domain: str | None = None, since: float | None = None,
               everything: bool = False) -> int:
        """Delete pages (and their visits, chunks, events). Returns the number of pages removed."""
        with self._lock, self.db:
            if everything:
                n = self.db.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
                self.db.execute("DELETE FROM events")
                self.db.execute("DELETE FROM pages")
                return n
            if url:
                ids = [r[0] for r in self.db.execute("SELECT id FROM pages WHERE url = ?", (clean_url(url),))]
            elif domain:
                d = domain.lower().removeprefix("www.")
                ids = [r[0] for r in self.db.execute("SELECT id FROM pages WHERE domain = ? OR domain LIKE ?", (d, "%." + d))]
            elif since is not None:
                ids = [r[0] for r in self.db.execute("SELECT id FROM pages WHERE first_seen >= ?", (since,))]
                self.db.execute("DELETE FROM visits WHERE started >= ?", (since,))
                self.db.execute("DELETE FROM events WHERE ts >= ?", (since,))
            else:
                return 0
            self.db.executemany("DELETE FROM pages WHERE id = ?", [(i,) for i in ids])
            return len(ids)

    def apply_retention(self, now: float | None = None) -> int:
        cutoff = (now or time.time()) - self.policy.retention_days * 86400
        with self._lock, self.db:
            self.db.execute("DELETE FROM visits WHERE started < ?", (cutoff,))
            self.db.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
            return self.db.execute("DELETE FROM pages WHERE last_seen < ?", (cutoff,)).rowcount

    # ------------------------------------------------------------------ reads
    def _highlights_by_page(self) -> Counter:
        return Counter(r[0] for r in self.db.execute(
            "SELECT page_id FROM events WHERE kind = 'highlight' AND actor = 'user' AND page_id IS NOT NULL"))

    def search(self, query: str, k: int = 8, *, include_agent: bool = False, since: float | None = None,
               now: float | None = None) -> list[Hit]:
        """Hybrid search over everything browsed: passage similarity + keywords + engagement + recency."""
        now = now or time.time()
        query = (query or "").strip()
        if not query:
            return []
        qv = self.embedder.embed([query])[0]
        qtok = tokens(query)
        with self._lock:
            where = ["c.model = ?"]
            args: list = [self.embedder.name]
            if not include_agent:
                where.append("p.user_visits > 0 OR EXISTS (SELECT 1 FROM events e WHERE e.page_id = p.id AND e.actor = 'user')")
            if since is not None:
                where.append("p.last_seen >= ?")
                args.append(since)
            rows = self.db.execute(
                f"SELECT c.page_id, c.text, c.vec, p.url, p.title, p.domain, p.last_seen, p.dwell_ms, p.user_visits, p.max_scroll "
                f"FROM chunks c JOIN pages p ON p.id = c.page_id WHERE {' AND '.join('(' + w + ')' for w in where)}",
                args,
            ).fetchall()
            hl = self._highlights_by_page()
            notes = self._notes_by_page()
        if not rows:
            return []
        mat = np.frombuffer(b"".join(r["vec"] for r in rows), dtype=np.float32).reshape(len(rows), -1)
        sims = mat @ qv
        best: dict[int, tuple[float, sqlite3.Row]] = {}
        for r, s in zip(rows, sims):
            if r["page_id"] not in best or s > best[r["page_id"]][0]:
                best[r["page_id"]] = (float(s), r)
        hits = []
        for pid, (sem, r) in best.items():
            ptok = tokens(r["title"] + " " + r["text"] + " " + notes.get(pid, ""))
            lex = len(qtok & ptok) / max(1, len(qtok)) if qtok else 0.0
            eng = _engagement(r["dwell_ms"], r["user_visits"], r["max_scroll"], hl[pid])
            rec = _recency(r["last_seen"], now)
            score = sem + 0.2 * lex + 0.15 * eng + 0.1 * rec
            why = []
            if sem > 0.3:
                why.append(f"similar content ({sem:.2f})")
            if lex > 0:
                why.append(f"keyword match {lex:.0%}")
            if eng > 0.4:
                why.append(f"you spent {r['dwell_ms'] // 60000} min here" if r["dwell_ms"] >= 60000 else "you engaged with it")
            if hl[pid]:
                why.append(f"{hl[pid]} highlight(s)")
            if rec > 0.8:
                why.append("recent")
            hits.append(Hit(pid, r["url"], r["title"], r["domain"], r["text"][:320], score, sem, eng, r["last_seen"], why))
        hits.sort(key=lambda h: h.score, reverse=True)
        return [h for h in hits if h.semantic > 0.05 or h.score > 0.15][:k]

    def _notes_by_page(self) -> dict[int, str]:
        out: dict[int, str] = {}
        for r in self.db.execute("SELECT page_id, text FROM events WHERE page_id IS NOT NULL AND kind IN ('highlight','note','bookmark')"):
            out[r[0]] = out.get(r[0], "") + " " + r[1]
        return out

    def recent(self, limit: int = 20, *, actor: Actor | None = "user") -> list[dict]:
        q = ("SELECT v.id, v.actor, v.started, v.dwell_ms, v.scroll, v.session_id, p.url, p.title, p.domain "
             "FROM visits v JOIN pages p ON p.id = v.page_id")
        args: list = []
        if actor:
            q += " WHERE v.actor = ?"
            args.append(actor)
        q += " ORDER BY v.started DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            return [{"url": r["url"], "title": r["title"], "domain": r["domain"], "actor": r["actor"],
                     "at": _iso(r["started"]), "dwell_s": r["dwell_ms"] // 1000, "scroll": round(r["scroll"], 2),
                     "session": r["session_id"]} for r in self.db.execute(q, args)]

    def current_session(self, *, now: float | None = None) -> list[dict]:
        """The user's trail in their latest session, oldest first, if it is still live."""
        now = now or time.time()
        with self._lock:
            last = self.db.execute("SELECT session_id, COALESCE(ended, started) AS t FROM visits WHERE actor='user' "
                                   "ORDER BY started DESC LIMIT 1").fetchone()
            if not last or now - last["t"] > SESSION_GAP_S:
                return []
            rows = self.db.execute(
                "SELECT p.url, p.title, v.started, v.dwell_ms FROM visits v JOIN pages p ON p.id = v.page_id "
                "WHERE v.session_id = ? ORDER BY v.started", (last["session_id"],)).fetchall()
            searches = self.db.execute("SELECT text FROM events WHERE kind='search' AND actor='user' AND ts >= ? ORDER BY ts",
                                       (rows[0]["started"] if rows else now,)).fetchall()
        # One entry per page (first time seen this session), with dwell summed across revisits.
        trail: dict[str, dict] = {}
        for r in rows:
            if r["url"] in trail:
                trail[r["url"]]["dwell_s"] += r["dwell_ms"] // 1000
                trail[r["url"]]["visits"] += 1
            else:
                trail[r["url"]] = {"url": r["url"], "title": r["title"], "at": _iso(r["started"]),
                                   "dwell_s": r["dwell_ms"] // 1000, "visits": 1}
        return list(trail.values()) + [{"search": q} for q in dict.fromkeys(x["text"] for x in searches)]

    def events(self, kind: EventKind | None = None, limit: int = 50) -> list[dict]:
        q = "SELECT e.ts, e.kind, e.text, e.actor, p.url, p.title FROM events e LEFT JOIN pages p ON p.id = e.page_id"
        args: list = []
        if kind:
            q += " WHERE e.kind = ?"
            args.append(kind)
        q += " ORDER BY e.ts DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            return [{"at": _iso(r["ts"]), "kind": r["kind"], "text": r["text"], "actor": r["actor"],
                     "url": r["url"], "title": r["title"]} for r in self.db.execute(q, args)]

    def interests(self, limit: int = 12, *, now: float | None = None) -> dict:
        """What the user keeps coming back to: domains and topics, weighted by attention and recency."""
        now = now or time.time()
        with self._lock:
            rows = self.db.execute("SELECT id, domain, title, excerpt, dwell_ms, user_visits, max_scroll, last_seen "
                                   "FROM pages WHERE user_visits > 0").fetchall()
            hl = self._highlights_by_page()
            searches = [r[0] for r in self.db.execute("SELECT text FROM events WHERE kind='search' AND actor='user'")]
        domains: Counter = Counter()
        topics: Counter = Counter()
        for r in rows:
            w = (0.2 + _engagement(r["dwell_ms"], r["user_visits"], r["max_scroll"], hl[r["id"]])) * _recency(r["last_seen"], now)
            domains[r["domain"]] += w
            for t in tokens(r["title"]):
                if len(t) > 2 and not t.isdigit():
                    topics[t] += w
        for s in searches:
            for t in tokens(s):
                if len(t) > 2:
                    topics[t] += 0.5
        # A term that appears in nearly every title (site names) says little.
        n = max(1, len(rows))
        df = Counter(t for r in rows for t in set(tokens(r["title"])))
        for t in list(topics):
            if n >= 5 and df[t] / n > 0.6:
                del topics[t]
        return {
            "domains": [{"domain": d, "weight": round(w, 3)} for d, w in domains.most_common(limit)],
            "topics": [{"topic": t, "weight": round(w, 3)} for t, w in topics.most_common(limit)],
        }

    def suggest_facts(self, existing: Iterable[str] = ()) -> list[FactDraft]:
        """Propose profile facts from browsing. Nothing is saved until the user accepts one."""
        ints = self.interests(limit=8)
        seen = {e.strip().lower() for e in existing}
        drafts = []
        topics = [t["topic"] for t in ints["topics"] if t["weight"] >= 0.5][:5]
        doms = [d["domain"] for d in ints["domains"] if d["weight"] >= 0.5][:3]
        if topics:
            drafts.append(FactDraft(text=f"Has been reading a lot about {', '.join(topics)}",
                                    category="project", tags=["browsing"] + topics[:3], sensitivity="personal"))
        if doms:
            drafts.append(FactDraft(text=f"Often uses {', '.join(doms)}", category="preference",
                                    tags=["browsing", "sources"], sensitivity="personal"))
        return [d for d in drafts if d.text.strip().lower() not in seen]

    def stats(self) -> dict:
        with self._lock:
            one = lambda q: self.db.execute(q).fetchone()[0]  # noqa: E731
            return {"pages": one("SELECT COUNT(*) FROM pages"), "visits": one("SELECT COUNT(*) FROM visits"),
                    "events": one("SELECT COUNT(*) FROM events"), "passages": one("SELECT COUNT(*) FROM chunks"),
                    "embedder": self.embedder.name, "paused": self.policy.paused,
                    "retention_days": self.policy.retention_days}

    def reindex(self) -> int:
        """Re-embed passages produced by a different embedder (after switching models)."""
        with self._lock:
            rows = self.db.execute("SELECT c.page_id, c.idx, c.text, p.title FROM chunks c JOIN pages p ON p.id = c.page_id "
                                   "WHERE c.model != ?", (self.embedder.name,)).fetchall()
        if not rows:
            return 0
        vecs = self.embedder.embed([f"{r['title']}\n{r['text']}" for r in rows])
        with self._lock, self.db:
            self.db.executemany("UPDATE chunks SET model = ?, vec = ? WHERE page_id = ? AND idx = ?",
                                [(self.embedder.name, v.astype(np.float32).tobytes(), r["page_id"], r["idx"]) for r, v in zip(rows, vecs)])
        return len(rows)
