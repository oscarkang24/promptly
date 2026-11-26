"""Command-line interface.

    promptly rewrite "help me write an email to my landlord"
    promptly fact add "Prefers concise answers" --category preference --sensitivity public
    promptly fact list
    promptly learn "I'm a backend engineer, I mostly write Go and I hate fluff."
    promptly serve
    promptly mcp                                  # MCP server for agents (stdio)
    promptly browser context "plan the API migration"
    promptly browser search "rate limiting"
    promptly browser forget --domain example.com
"""

from __future__ import annotations

import argparse
import json
import sys

from .core import Promptly
from .engines import TARGET_STYLES, EngineError
from .models import Fact, RewriteRequest


def _print_result(res, as_json: bool) -> None:
    if as_json:
        print(res.model_dump_json(indent=2))
        return
    print(res.rewritten_prompt)
    print("\n---")
    print(f"goal:       {res.intent.true_goal}")
    print(f"engine:     {res.engine}   embedder: {res.embedder}   confidence: {res.confidence}")
    print(f"clarity:    {'already clear' if res.already_clear else 'missing ' + ', '.join(res.gaps)}")
    if res.injected:
        print("injected:")
        for f in res.injected:
            print(f"  [{f.id}] {f.relevance:+.2f}  {f.text}  ({f.reason})")
    if res.withheld_fact_ids:
        print(f"withheld:   {len(res.withheld_fact_ids)} fact(s) above the sensitivity limit")
    for a in res.assumptions:
        print(f"assumed:    {a}")
    for q in res.clarifying_questions:
        print(f"question:   {q}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="promptly", description="Rewrite prompts into what you actually mean.")
    sub = p.add_subparsers(dest="cmd", required=True)

    rw = sub.add_parser("rewrite", help="Rewrite a prompt (reads stdin if no prompt is given)")
    rw.add_argument("prompt", nargs="?")
    rw.add_argument("--target", default="generic", choices=sorted(TARGET_STYLES))
    rw.add_argument("--allow-sensitive", action="store_true", help="Let sensitive facts be used")
    rw.add_argument("--no-personal", action="store_true", help="Only use public facts")
    rw.add_argument("--json", action="store_true")
    rw.add_argument("-i", "--interactive", action="store_true", help="Answer clarifying questions, then rewrite again")

    fact = sub.add_parser("fact", help="Manage personal facts")
    fsub = fact.add_subparsers(dest="fact_cmd", required=True)
    fa = fsub.add_parser("add")
    fa.add_argument("text")
    fa.add_argument("--category", default="other")
    fa.add_argument("--tags", default="", help="comma-separated")
    fa.add_argument("--sensitivity", default="personal", choices=["public", "personal", "sensitive"])
    fsub.add_parser("list").add_argument("--json", action="store_true")
    fsub.add_parser("rm").add_argument("id")

    ln = sub.add_parser("learn", help="Propose facts from text about yourself")
    ln.add_argument("text", nargs="?")
    ln.add_argument("-y", "--yes", action="store_true", help="Save all proposals without asking")

    fb = sub.add_parser("feedback", help="Tell Promptly a fact helped (+1) or was noise (-1) for a prompt")
    fb.add_argument("fact_id")
    fb.add_argument("label", type=float)
    fb.add_argument("prompt")

    sv = sub.add_parser("serve", help="Run the web app")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8787)

    sub.add_parser("mcp", help="Run the MCP server for agents (stdio)")

    br = sub.add_parser("browser", help="Inspect and manage the local browsing memory")
    bsub = br.add_subparsers(dest="browser_cmd", required=True)
    bc = bsub.add_parser("context", help="Context pack for a task")
    bc.add_argument("task")
    bc.add_argument("--max-tokens", type=int, default=1500)
    bc.add_argument("--json", action="store_true")
    bs = bsub.add_parser("search")
    bs.add_argument("query")
    bs.add_argument("-k", type=int, default=8)
    bsub.add_parser("recent").add_argument("--limit", type=int, default=20)
    bsub.add_parser("interests")
    bsub.add_parser("stats")
    bsub.add_parser("suggest", help="Propose profile facts from browsing; asks before saving")
    bsub.add_parser("pause")
    bsub.add_parser("resume")
    bsub.add_parser("token", help="Print the local auth token the browser shell and agents use")
    bf = bsub.add_parser("forget")
    g = bf.add_mutually_exclusive_group(required=True)
    g.add_argument("--url")
    g.add_argument("--domain")
    g.add_argument("--last-minutes", type=float)
    g.add_argument("--everything", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "mcp":
        from .browser.mcp_server import main as mcp_main

        mcp_main()
        return 0

    if args.cmd == "browser":
        return _browser(args)

    if args.cmd == "serve":
        import uvicorn

        print(f"Promptly on http://{args.host}:{args.port}")
        uvicorn.run("promptly.server:app", host=args.host, port=args.port)
        return 0

    app = Promptly()

    try:
        if args.cmd == "rewrite":
            prompt = args.prompt if args.prompt is not None else sys.stdin.read()
            level = "sensitive" if args.allow_sensitive else "public" if args.no_personal else "personal"
            req = RewriteRequest(prompt=prompt, target=args.target, max_sensitivity=level)
            res = app.rewrite(req)
            if args.interactive and res.clarifying_questions:
                for q in res.clarifying_questions:
                    ans = input(f"{q}\n> ").strip()
                    if ans:
                        req.answers[q] = ans
                if req.answers:
                    res = app.rewrite(req)
            _print_result(res, args.json)

        elif args.cmd == "fact":
            if args.fact_cmd == "add":
                tags = [t.strip() for t in args.tags.split(",") if t.strip()]
                f = app.store.add(Fact(text=args.text, category=args.category, tags=tags, sensitivity=args.sensitivity))
                print(f"saved [{f.id}] {f.text}")
            elif args.fact_cmd == "list":
                facts = app.store.all()
                if args.json:
                    print(json.dumps([f.model_dump() for f in facts], indent=2, ensure_ascii=False))
                for f in [] if args.json else facts:
                    print(f"[{f.id}] {f.category:<10} {f.sensitivity:<9} {f.text}")
                if not facts and not args.json:
                    print("No facts yet. Add one with `promptly fact add` or `promptly learn`.")
            elif args.fact_cmd == "rm":
                print("deleted" if app.store.delete(args.id) else "no such fact")
                return 0

        elif args.cmd == "feedback":
            app.feedback(args.prompt, args.fact_id, max(-1.0, min(1.0, args.label)))
            print("noted")

        elif args.cmd == "learn":
            text = args.text if args.text is not None else sys.stdin.read()
            drafts = app.learn(text)
            if not drafts:
                print("Nothing new to remember.")
                return 0
            keep = []
            for d in drafts:
                if args.yes:
                    keep.append(d)
                    continue
                ans = input(f"Remember \"{d.text}\" ({d.category}, {d.sensitivity})? [Y/n] ").strip().lower()
                if ans in ("", "y", "yes"):
                    keep.append(d)
            for f in app.accept(keep):
                print(f"saved [{f.id}] {f.text}")
    except (EngineError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def _browser(args) -> int:
    import time

    from .browser.pack import pack_to_markdown
    from .browser.service import get_service

    svc = get_service()
    m = svc.memory
    dump = lambda o: print(json.dumps(o, indent=2, ensure_ascii=False))  # noqa: E731
    c = args.browser_cmd
    if c == "context":
        pack = svc.context(args.task, max_tokens=args.max_tokens)
        dump(pack) if args.json else print(pack_to_markdown(pack))
    elif c == "search":
        for h in m.search(args.query, k=args.k):
            print(f"{h.score:5.2f}  {h.title or h.url}\n       {h.url}\n       {h.snippet[:160]}")
    elif c == "recent":
        for r in m.recent(args.limit):
            print(f"{r['at']}  {r['dwell_s']:>5}s  {r['title'] or r['url']}")
    elif c == "interests":
        dump(m.interests())
    elif c == "stats":
        dump(m.stats())
    elif c == "suggest":
        drafts = m.suggest_facts(f.text for f in svc.store.all())
        if not drafts:
            print("Nothing to suggest yet. Browse a bit more.")
        for d in drafts:
            if input(f"Remember \"{d.text}\"? [y/N] ").strip().lower() in ("y", "yes"):
                f = svc.store.add(Fact(**d.model_dump(), source="browsing"))
                print(f"saved [{f.id}]")
    elif c in ("pause", "resume"):
        m.policy.paused = c == "pause"
        m.policy.save()
        print("recording paused" if m.policy.paused else "recording on")
    elif c == "token":
        print(svc.token)
    elif c == "forget":
        since = time.time() - args.last_minutes * 60 if args.last_minutes else None
        n = m.forget(url=args.url, domain=args.domain, since=since, everything=args.everything)
        print(f"forgot {n} page(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
