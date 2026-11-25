"""Local, file-backed store for the user's personal context.

Everything lives in one JSON file under PROMPTLY_HOME (default ~/.promptly).
Nothing here talks to the network.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

from .models import Fact, _now


def default_home() -> Path:
    return Path(os.environ.get("PROMPTLY_HOME", Path.home() / ".promptly"))


class FactStore:
    def __init__(self, path: Path | None = None):
        self.path = path or default_home() / "profile.json"
        self._lock = threading.Lock()

    # -- persistence -------------------------------------------------------
    def _load(self) -> list[Fact]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text() or "{}")
        return [Fact.model_validate(f) for f in data.get("facts", [])]

    def _save(self, facts: list[Fact]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "facts": [f.model_dump() for f in facts]}
        # Atomic write so a crash never leaves a half-written profile.
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    # -- CRUD --------------------------------------------------------------
    def all(self) -> list[Fact]:
        with self._lock:
            return self._load()

    def get(self, fact_id: str) -> Fact | None:
        return next((f for f in self.all() if f.id == fact_id), None)

    def add(self, fact: Fact) -> Fact:
        with self._lock:
            facts = self._load()
            # Skip exact duplicates instead of piling them up.
            for existing in facts:
                if existing.text.strip().lower() == fact.text.strip().lower():
                    return existing
            facts.append(fact)
            self._save(facts)
            return fact

    def update(self, fact_id: str, **changes) -> Fact | None:
        with self._lock:
            facts = self._load()
            for i, f in enumerate(facts):
                if f.id == fact_id:
                    data = f.model_dump() | {k: v for k, v in changes.items() if v is not None}
                    data["updated_at"] = _now()
                    facts[i] = Fact.model_validate(data)
                    self._save(facts)
                    return facts[i]
            return None

    def delete(self, fact_id: str) -> bool:
        with self._lock:
            facts = self._load()
            kept = [f for f in facts if f.id != fact_id]
            if len(kept) == len(facts):
                return False
            self._save(kept)
            return True
