"""What the browser is allowed to remember.

Rules applied before anything is written to disk:

- Private windows, paused recording and blocked domains are never recorded.
- Only http(s) pages are recorded (no file://, chrome://, data:, about:).
- Query parameters that look like secrets (token, code, session, key, sig, ...) are stripped.
- Default-blocked: banking, password managers, health portals, auth pages. Users can add more.
- Form input is never captured: the shell only sends page text, titles, dwell, scroll and
  text the user explicitly highlights.
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SECRET_PARAM = re.compile(
    r"(token|code|session|sess|sid|auth|key|secret|sig|signature|password|pass|pwd|otp|"
    r"state|nonce|ticket|jwt|credential|apikey|access|refresh)",
    re.I,
)

DEFAULT_BLOCKED = [
    "accounts.google.com", "login.*", "auth.*", "*.okta.com", "*.auth0.com",
    "*bank*", "*.chase.com", "*.paypal.com", "*.venmo.com", "*.wellsfargo.com",
    "*.1password.com", "*.lastpass.com", "*.bitwarden.com",
    "*mychart*", "*.patient*", "*health*portal*",
]


def domain_of(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def clean_url(url: str) -> str:
    """Drop fragments and secret-looking query params."""
    parts = urlsplit(url)
    q = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not SECRET_PARAM.search(k)]
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path or "/", urlencode(q), ""))


@dataclass
class Policy:
    blocked: list[str] = field(default_factory=lambda: list(DEFAULT_BLOCKED))
    paused: bool = False
    retention_days: int = 90
    path: Path | None = None

    @classmethod
    def load(cls, path: Path) -> "Policy":
        if path.exists():
            try:
                d = json.loads(path.read_text())
                return cls(blocked=d.get("blocked", list(DEFAULT_BLOCKED)), paused=bool(d.get("paused", False)),
                           retention_days=int(d.get("retention_days", 90)), path=path)
            except (ValueError, TypeError):
                pass
        return cls(path=path)

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"blocked": self.blocked, "paused": self.paused,
                                         "retention_days": self.retention_days}, indent=2))

    def is_blocked(self, url: str) -> bool:
        d = domain_of(url)
        for pat in self.blocked:
            base = pat[2:] if pat.startswith("*.") else pat
            if fnmatch.fnmatch(d, pat) or d == base or d.endswith("." + base):
                return True
        return False

    def may_record(self, url: str, private: bool = False) -> bool:
        if self.paused or private:
            return False
        if urlsplit(url).scheme not in ("http", "https"):
            return False
        return not self.is_blocked(url)
