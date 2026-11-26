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

**Demo:** [`demo/promptly-demo.mp4`](demo/promptly-demo.mp4) (96 s). It was recorded live with the local
multilingual embeddings and the offline rewrite engine, without Claude. Re-record it with
`node demo/record_demo.js` (instructions at the top of that file).

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
tests/         pytest suite (no network)
```

## Next ideas

- Browser extension that rewrites in place inside ChatGPT / Claude / Gemini text boxes.
- Learn facts passively from accepted rewrites and from edits people make to them.
- A user-level latent vector (for example a running average of accepted rewrites) to steer tone and
  format when no individual fact matches.
- Mine past assistant conversations for facts, like noodle's `ruminate` / `reflect` skills.
- An eval set of loose prompts → ideal rewrites to tune the system prompt against.
