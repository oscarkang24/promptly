# Promptly

Type a prompt the way you'd say it. Promptly works out what you actually mean, folds in the
parts of your personal context that matter, and hands back the prompt you would have written
if you had the time.

```
you type:   write a python script to retry flaky api calls
you get:    Context about me:
            - Staff engineer building LLM agent infrastructure in Python
            - Prefers concise answers with code first
            Task: ...
            What I need back: working code first, then how to verify it.
```

## How it works

```
prompt ──► privacy gate ──► latent-vector retrieval ──► Claude (one structured call) ──► validation ──► result
              │                     │                            │                            │
   facts above your limit   prompt + facts embedded      triage (what/where/done),     drops any fact id it
   never leave the device   locally; hybrid score:       infers intent, picks facts,   wasn't sent; caps
                            cosine + keywords +          rewrites, asks ≤3 questions   questions at 3
                            feedback from similar        in one round
                            past prompts; top 25
                                    ▲
                     👍/👎 and Copy ─┘ (feedback memory in the same vector space)
```

### Latent vectors

Every fact and every prompt is embedded **on your machine** (nothing is sent to an embedding API):

- `fastembed` + `paraphrase-multilingual-MiniLM-L12-v2` (384-d, ~50 languages including Chinese)
  when installed with `pip install -e ".[embed]"`. The model (~220 MB) downloads once on first use.
- Otherwise a dependency-free hashing embedder (word + character n-gram feature hashing, 1024-d).
  It handles Chinese through character bigrams, so it runs offline and in tests.

A fact's score is `cos(prompt, fact) + 0.15·keyword overlap + category prior + 0.5·feedback`.
The feedback term comes from a nearest-neighbour vote: each 👍/👎 you give stores the prompt and
the fact. Later prompts that land near that prompt in vector space (cosine > 0.35) inherit the vote,
weighted by similarity. Copying a rewrite counts as a weak 👍 (+0.3) for the facts it used. Ranking
adapts to you without any training step, and feedback on unrelated prompts has no effect.
Vectors are cached in `~/.promptly/vectors.json`, scoped to the embedder that produced them.
Feedback is stored in `~/.promptly/feedback.jsonl`.

### Triage first (borrowed from noodle's `refine` skill)

Promptly first checks whether the request already says **what** is wanted, **where** it applies,
and what **done** looks like. Missing parts are returned as `gaps` and are the only things it asks
about, in a single round. Requests that are already clear get a light touch: context is added and
the wording barely changes. Every rewrite must stand on its own for an assistant with no other
context.

- **Intent inference.** Each result separates the surface request from the real goal, and names
  the audience, the ideal output, the constraints and the success criteria.
- **Personal context injection.** Facts live in `~/.promptly/profile.json` (file mode 0600). Each fact
  has a category and a sensitivity: `public`, `personal` or `sensitive`. The model only uses a fact
  when it would change the answer, and gives the reason for each one it uses.
- **Privacy by construction.** Each request has a sharing limit, `personal` by default. Facts above
  that limit are never sent to the model. Every result lists which facts were sent and which were
  withheld.
- **Clarify loop.** When a guess would matter, Promptly asks up to 3 questions. Answer them and it
  rewrites again.
- **Learn.** Paste a bio or a rambling "about me". Promptly suggests facts, and nothing is saved
  until you accept it.
- **Targets.** `generic`, `claude` (XML tags), `chatgpt` (Markdown), `image`, `code-agent`.

## Promptly Browser

A browser that learns how you work, keeps what it learns on your machine, and hands it to your
agents as compact, task-shaped context.

```
 Promptly Browser (Electron, browser/)             promptly serve (127.0.0.1:8787)          agents
 ┌───────────────────────────────┐  visits, dwell,  ┌───────────────────────────────┐  MCP   ┌──────────────┐
 │ tabs · address bar · ✦ panel  │  scroll, text,   │ BrowserMemory  ~/.promptly/   │◄──────►│ Claude Code, │
 │ preload: scroll, copy, idle   │──searches, ─────►│  browser.db (SQLite, 0600)    │ stdio  │ Desktop, any │
 │ Ctrl+Shift+H: remember text   │  highlights      │  passages + local embeddings  │        │ MCP client   │
 │ control server 127.0.0.1:rand │◄─────────────────┤  sessions · interests · packs │        └──────┬───────┘
 └───────────────▲───────────────┘   tab control    └───────────────────────────────┘               │
                 └──────────────────── open / read / click / type (agent tabs only) ────────────────┘
```

### What it learns

| Signal | How |
|---|---|
| Visits | every page a tab shows, the page it came from, and a session id (a 30-minute gap starts a new session) |
| Attention | dwell time counted only while the tab is in front, the window is focused and you're not idle; max scroll depth |
| Content | the page's readable text (article/main, minus nav, forms and scripts), split into passages and embedded locally |
| Searches | queries on Google, Bing, DuckDuckGo, Brave, Kagi, GitHub, Stack Overflow, YouTube and Wikipedia |
| Explicit | text you copy, and highlights you save with Ctrl/Cmd+Shift+H or the right-click menu |

From that it derives **interests** (topics and sites, weighted by attention with a 14-day half-life),
your **current session** (what you're doing right now), and **suggested profile facts** that are
saved only when you accept them.

### What it never does

- Records nothing in private windows (Ctrl/Cmd+Shift+N), while paused, or on blocked sites. Banks,
  password managers, health portals and login pages are blocked by default.
- Never reads form fields or keystrokes. Strips secret-looking URL parameters (`token`, `code`,
  `session`, `key`, `sig`, ...).
- Sends nothing off the machine. Embeddings are computed locally, and pages and visits are deleted after 90 days (you can change this).
- Lets you forget a page, a site, the last hour, or everything, from the panel, the CLI or the API.

### For agents

Agent visits are stored with `actor='agent'`. They never count toward what the browser learns
about you, but agents can still search them.

- **MCP** (`promptly mcp`): `get_context`, `search_history`, `recent_activity`, `current_session`,
  `interests`, `remember`, plus tab tools `list_tabs`, `open_tab`, `read_tab`, `navigate`, `click`,
  `type_text`, `close_tab`. The memory tools work even when the browser is closed.
- **Guardrails**: agents can read any tab. They can click, type and navigate only in tabs they opened,
  or in a tab where you pressed **🤖 Let agent drive**. Agent tabs carry an orange `agent` badge.
  `type_text` refuses password, payment and one-time-code fields.
- **CDP**: start with `PROMPTLY_CDP_PORT=9333` and Playwright can `connectOverCDP("http://127.0.0.1:9333")`.
- **Local auth**: the daemon and the shell's control port both require `X-Promptly-Token` from
  `~/.promptly/browser.token` (0600). The shell refuses any request that carries an `Origin` header,
  so websites can't reach it through localhost.

Context packs fill sections in priority order within a token budget (profile facts → related pages
with their best passage and why they matched → current session → matching highlights → interests),
so a small budget keeps the most relevant items instead of cutting text off midway. The ✦ panel's
**Copy for any AI chat** button puts the same pack on your clipboard.

`POST /api/rewrite` with `"use_browsing": true` also offers matching pages to the rewriter as
temporary facts for that one rewrite. They pass through the same privacy gate and are never saved.

### Run it

```bash
pip install -e ".[dev]"            # daemon + MCP
```

```bash
cd browser && npm install && npm start   # starts `promptly serve` itself if it isn't running
```

Add it to Claude Code as an MCP server:

```bash
claude mcp add promptly-browser -- promptly mcp
```

CLI:

```bash
promptly browser context "write the migration plan for our API"
promptly browser search "that post about vector indexes"
promptly browser interests
promptly browser suggest           # propose profile facts from browsing, asks before saving
promptly browser forget --domain example.com     # or --url, --last-minutes 60, --everything
promptly browser pause             # / resume
```

Browser settings: `PROMPTLY_DAEMON` (default `http://127.0.0.1:8787`), `PROMPTLY_CMD` (how to start
the daemon, default `promptly`), `PROMPTLY_CDP_PORT`, `PROMPTLY_CONTROL_PORT` (default random),
`PROMPTLY_START_URL`.

| Method | Path (all need `X-Promptly-Token`) | |
|---|---|---|
| POST | `/api/browser/visit/start` · `/visit/end` · `/content` · `/event` | written by the shell |
| GET | `/api/browser/search?q=` · `/recent` · `/session` · `/interests` · `/stats` · `/suggest-facts` | |
| POST | `/api/browser/context` | `{task, max_tokens?, max_sensitivity?, format: json\|markdown}` |
| GET/POST | `/api/browser/policy` | `{paused, blocked, retention_days}` |
| POST | `/api/browser/forget` | `{url}` \| `{domain}` \| `{since_minutes}` \| `{everything: true}` |

## Run it

```bash
pip install -e ".[dev]"
```

```bash
export ANTHROPIC_API_KEY=...        # without it, Promptly falls back to an offline heuristic engine
```

```bash
promptly serve                       # web app on http://127.0.0.1:8787
```

CLI:

```bash
promptly fact add "Prefers concise answers with code first" --category preference --sensitivity public
promptly learn "I'm a backend engineer, mostly Go, and I hate fluff."
promptly rewrite "fix my email to the landlord" -i          # -i answers clarifying questions
promptly rewrite "plan meals this week" --allow-sensitive --target claude
promptly feedback <fact-id> -1 "plan meals this week"       # this fact was noise here
```

Settings: `PROMPTLY_MODEL` (default `claude-opus-5-5`), `PROMPTLY_HOME` (default `~/.promptly`),
`PROMPTLY_ENGINE=offline|claude`, `PROMPTLY_EMBEDDER=auto|fastembed|hashing`,
`PROMPTLY_EMBED_MODEL` (any fastembed text model).

The Claude engine uses structured outputs and server-side refusal fallbacks
(`fallbacks: "default"`), so if Claude declines a request the API retries it on a fallback model.

## API

| Method | Path | |
|---|---|---|
| POST | `/api/rewrite` | `{prompt, target?, max_sensitivity?, answers?, fact_ids?}` |
| POST | `/api/feedback` | `{prompt, fact_id, label}`: label in [-1, 1] |
| GET/POST | `/api/facts` | list / add |
| PATCH/DELETE | `/api/facts/{id}` | edit / forget |
| POST | `/api/learn` | propose facts from text |
| POST | `/api/learn/accept` | save proposed facts |

## Layout

```
src/promptly/
  models.py    data models (Fact, Intent, RewriteResult, ...)
  store.py     local JSON fact store
  embeddings.py  local embedders (fastembed / hashing) + on-disk vector cache
  context.py   privacy gate + hybrid latent-vector retrieval + feedback memory
  engines.py   ClaudeEngine (structured output) and OfflineEngine
  core.py      the pipeline + validation
  cli.py       `promptly` command
  server.py    FastAPI app; static/index.html is the UI
  browser/
    policy.py      what may be recorded (private, paused, blocklist, secret URL params)
    memory.py      SQLite behavior memory: pages, passages+vectors, visits, events, interests
    pack.py        token-budgeted context packs for a task
    routes.py      /api/browser (token-protected)
    service.py     wiring, local auth token, client for the shell's control port
    mcp_server.py  MCP tools for agents
browser/         Electron shell: src/main.js (tabs, capture, agent control), ui/ (toolbar, panel)
tests/         pytest suite (no network); browser/test/ for the shell's helpers
```

## Next ideas

- Rewrite in place inside ChatGPT / Claude / Gemini text boxes, now that the browser is ours.
- Group sessions into named tasks by meaning, not only by time gaps.
- Learn facts passively from accepted rewrites and from edits people make to them.
- A user-level latent vector (for example a running average of accepted rewrites) to steer tone and
  format when no individual fact matches.
- Mine past assistant conversations for facts, like noodle's `ruminate` / `reflect` skills.
- An eval set of loose prompts → ideal rewrites to tune the system prompt against.
