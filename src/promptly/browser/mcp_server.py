"""MCP server: lets any MCP-capable agent (Claude Code, Claude Desktop, ...) use the browser's memory
and drive the browser.

    promptly mcp            # stdio transport

Memory tools read ~/.promptly/browser.db directly and work even when the browser is closed.
Tab tools talk to the running shell. Agents can read any tab, but they can only click, type or
navigate in tabs they opened themselves, or tabs the user has handed over ("Let agent drive").
Everything an agent does is recorded with actor='agent' so it never skews what the browser
learns about the user.
"""

from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer

from .pack import pack_to_markdown
from .service import ShellError, get_service

INSTRUCTIONS = """Promptly Browser gives you the user's local browsing memory and their browser.
Start a task with get_context(task) to learn what the user already read, is doing now, and
prefers. Use search_history for specific lookups. Use the tab tools to act on the web; prefer
opening your own tab over touching the user's. Never paste secrets you see on a page into
your reply unless the user asked for them."""

mcp = MCPServer(name="promptly-browser", instructions=INSTRUCTIONS)


def _shell(method: str, path: str, body: dict | None = None) -> str:
    try:
        return json.dumps(get_service().shell.call(method, path, body), ensure_ascii=False)
    except ShellError as e:
        return f"error: {e}"


@mcp.tool()
def get_context(task: str, max_tokens: int = 1500) -> str:
    """A compact context pack for a task: relevant profile facts, pages the user read about it (with the
    best passage), their current browsing session, matching highlights, and ongoing interests."""
    return pack_to_markdown(get_service().context(task, max_tokens=max(200, min(max_tokens, 20000))))


@mcp.tool()
def search_history(query: str, k: int = 8, include_agent_pages: bool = False) -> str:
    """Semantic + keyword search over every page the user has read. Returns title, url, best passage and why it matched."""
    hits = get_service().memory.search(query, k=max(1, min(k, 50)), include_agent=include_agent_pages)
    return json.dumps([h.as_dict() for h in hits], ensure_ascii=False)


@mcp.tool()
def recent_activity(limit: int = 20) -> str:
    """The user's most recent page visits, newest first, with dwell time and session ids."""
    return json.dumps(get_service().memory.recent(max(1, min(limit, 200))), ensure_ascii=False)


@mcp.tool()
def current_session() -> str:
    """What the user is doing right now: pages in their live browsing session plus searches, oldest first."""
    return json.dumps(get_service().memory.current_session(), ensure_ascii=False)


@mcp.tool()
def interests() -> str:
    """Topics and sites the user keeps returning to, weighted by attention and recency."""
    return json.dumps(get_service().memory.interests(), ensure_ascii=False)


@mcp.tool()
def remember(note: str, url: str | None = None) -> str:
    """Save a note into the browser memory (marked as written by an agent) so later tasks can find it."""
    ok = get_service().memory.add_event("note", note, url, actor="agent")
    return "saved" if ok else "not saved (recording is paused or the site is blocked)"


@mcp.tool()
def list_tabs() -> str:
    """Open tabs: id, url, title, whether it is active, and whether you may drive it."""
    return _shell("GET", "/tabs")


@mcp.tool()
def open_tab(url: str, background: bool = True) -> str:
    """Open a URL in a new agent tab. Returns the tab id. You may drive tabs you open."""
    return _shell("POST", "/tabs", {"url": url, "background": background})


@mcp.tool()
def read_tab(tab_id: str, max_chars: int = 20000) -> str:
    """Readable text of a tab (title, url, main text, links). Works on any tab."""
    return _shell("GET", f"/tabs/{tab_id}/read?max_chars={max(500, min(max_chars, 200000))}")


@mcp.tool()
def navigate(tab_id: str, url: str) -> str:
    """Load a URL in a tab you are allowed to drive."""
    return _shell("POST", f"/tabs/{tab_id}/navigate", {"url": url})


@mcp.tool()
def click(tab_id: str, selector: str) -> str:
    """Click the first element matching a CSS selector in a tab you are allowed to drive."""
    return _shell("POST", f"/tabs/{tab_id}/click", {"selector": selector})


@mcp.tool()
def type_text(tab_id: str, selector: str, text: str, submit: bool = False) -> str:
    """Type into an input/textarea matching a CSS selector in a tab you are allowed to drive. Refuses password fields."""
    return _shell("POST", f"/tabs/{tab_id}/type", {"selector": selector, "text": text, "submit": submit})


@mcp.tool()
def close_tab(tab_id: str) -> str:
    """Close a tab you opened."""
    return _shell("DELETE", f"/tabs/{tab_id}")


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
