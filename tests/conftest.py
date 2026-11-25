import pytest


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    # Deterministic and network-free: no neural model download, no API calls.
    monkeypatch.setenv("PROMPTLY_EMBEDDER", "hashing")
    monkeypatch.setenv("PROMPTLY_ENGINE", "offline")
