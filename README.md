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
prompt ──► privacy gate ──► relevance shortlist ──► Claude (one structured call) ──► validation ──► result
              │                    │                         │                          │
   facts above your limit   cheap lexical scorer      infers intent, picks the     drops any fact id it
   never leave the device   keeps the top 25          facts that change the        wasn't sent; caps
                                                      answer, rewrites, asks       questions at 3
                                                      up to 3 questions
```

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
```

Settings: `PROMPTLY_MODEL` (default `claude-opus-5-5`), `PROMPTLY_HOME` (default `~/.promptly`),
`PROMPTLY_ENGINE=offline|claude`.

The Claude engine uses structured outputs and server-side refusal fallbacks
(`fallbacks: "default"`), so if Claude declines a request the API retries it on a fallback model.

## API

| Method | Path | |
|---|---|---|
| POST | `/api/rewrite` | `{prompt, target?, max_sensitivity?, answers?, fact_ids?}` |
| GET/POST | `/api/facts` | list / add |
| PATCH/DELETE | `/api/facts/{id}` | edit / forget |
| POST | `/api/learn` | propose facts from text |
| POST | `/api/learn/accept` | save proposed facts |

## Layout

```
src/promptly/
  models.py    data models (Fact, Intent, RewriteResult, ...)
  store.py     local JSON fact store
  context.py   privacy gate + relevance shortlist
  engines.py   ClaudeEngine (structured output) and OfflineEngine
  core.py      the pipeline + validation
  cli.py       `promptly` command
  server.py    FastAPI app; static/index.html is the UI
tests/         pytest suite (no network)
```

## Next ideas

- Browser extension that rewrites in place inside ChatGPT / Claude / Gemini text boxes.
- Learn facts passively from accepted rewrites and from edits people make to them.
- Embedding-based retrieval once profiles grow past a few hundred facts.
- An eval set of loose prompts → ideal rewrites to tune the system prompt against.
