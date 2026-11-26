"""Promptly Browser: a local behavior memory that a browser writes to and agents read from.

    browser shell (Electron)  ──events──►  BrowserMemory (SQLite, ~/.promptly/browser.db)
                                                 │
            agents (MCP / HTTP)  ◄──context packs─┘   ──tab control──►  browser shell

Nothing here sends browsing data off the machine.
"""

from .memory import BrowserMemory
from .policy import Policy

__all__ = ["BrowserMemory", "Policy"]
