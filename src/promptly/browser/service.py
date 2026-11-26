"""Wiring: one BrowserMemory + the Promptly profile, a local auth token, and a client for the shell.

Local auth: every /api/browser call and every call to the shell's control port must carry
`X-Promptly-Token`, read from ~/.promptly/browser.token (mode 0600). Web pages can't read that
file, and the custom header forces a CORS preflight that these servers never approve, so a
site you visit cannot read or write your browsing memory through localhost.
"""

from __future__ import annotations

import json
import os
import secrets
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

from ..context import Retriever
from ..embeddings import default_embedder
from ..store import FactStore, default_home
from .memory import BrowserMemory
from .pack import build_pack
from .policy import Policy


def load_token(home: Path) -> str:
    path = home / "browser.token"
    if path.exists():
        tok = path.read_text().strip()
        if tok:
            return tok
    home.mkdir(parents=True, exist_ok=True)
    tok = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(tok)
    return tok


class ShellError(RuntimeError):
    pass


class ShellClient:
    """Talks to the running browser shell's control port (written to ~/.promptly/shell.json)."""

    def __init__(self, home: Path, token: str):
        self.home = home
        self.token = token

    def _base(self) -> str:
        path = self.home / "shell.json"
        if not path.exists():
            raise ShellError("The Promptly browser isn't running. Start it with `npm start` in browser/.")
        info = json.loads(path.read_text())
        return f"http://127.0.0.1:{int(info['port'])}"

    def call(self, method: str, path: str, body: dict | None = None, timeout: float = 30) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self._base() + path, data=data, method=method,
                                     headers={"X-Promptly-Token": self.token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            raise ShellError(f"browser said {e.code}: {e.read().decode(errors='replace')[:300]}") from e
        except (urllib.error.URLError, OSError) as e:
            raise ShellError(f"can't reach the Promptly browser ({e}). Is it running?") from e


class BrowserService:
    def __init__(self, home: Path | None = None, memory: BrowserMemory | None = None, store: FactStore | None = None,
                 retriever: Retriever | None = None):
        self.home = home or default_home()
        self.store = store or FactStore(self.home / "profile.json")
        embedder = retriever.embedder if retriever else default_embedder()
        self.retriever = retriever or Retriever(embedder, self.home)
        self.memory = memory or BrowserMemory(self.home / "browser.db", embedder, Policy.load(self.home / "browser-policy.json"))
        self.token = load_token(self.home)
        self.shell = ShellClient(self.home, self.token)

    def context(self, task: str, max_tokens: int = 1500, max_sensitivity: str = "personal") -> dict:
        return build_pack(self.memory, task, facts=self.store.all(), retriever=self.retriever,
                          max_tokens=max_tokens, max_sensitivity=max_sensitivity)


@lru_cache
def get_service() -> BrowserService:
    """Standalone service (MCP server, CLI). The web server shares its Promptly instance instead."""
    return BrowserService()
