"""Latent vectors for facts and prompts, computed on this machine.

Two embedders:

- FastEmbedEmbedder: a real multilingual sentence model (ONNX, CPU) via `fastembed`.
  Install with `pip install -e ".[embed]"`. The model (~220 MB) downloads once on first use.
- HashingEmbedder: dependency-free signed feature hashing over words and character n-grams.
  Works for Chinese and English, needs no download, and is what the tests use.

Vectors are cached on disk per (model, text) so a fact is embedded once, not on every request.
Nothing here sends text over the network except fastembed's one-time model download.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_FASTEMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized vectors."""
        ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
_LATIN_WORD = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")


class HashingEmbedder:
    """Signed feature hashing. Deterministic, ~instant, no model files."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> list[tuple[str, float]]:
        t = text.lower()
        feats: list[tuple[str, float]] = []
        for w in _LATIN_WORD.findall(t):
            w = w.strip(".-")
            if not w:
                continue
            feats.append(("w:" + w, 1.0))
            padded = f"<{w}>"
            feats += [("c:" + padded[i : i + 4], 0.4) for i in range(max(1, len(padded) - 3))]
        # CJK has no spaces: use character unigrams and bigrams.
        cjk = "".join(ch if _CJK.match(ch) else " " for ch in t).split()
        for run in cjk:
            feats += [("u:" + ch, 0.6) for ch in run]
            feats += [("b:" + run[i : i + 2], 1.0) for i in range(len(run) - 1)]
        return feats

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, weight in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[row, h % self.dim] += sign * weight
        return _normalize(out)


class FastEmbedEmbedder:
    def __init__(self, model: str = DEFAULT_FASTEMBED_MODEL):
        from fastembed import TextEmbedding  # optional dependency

        self._model = TextEmbedding(model_name=model)
        self.name = f"fastembed:{model}"
        self.dim = int(self._model.embedding_size)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack(list(self._model.embed(list(texts)))).astype(np.float32))


def default_embedder() -> Embedder:
    """PROMPTLY_EMBEDDER = auto (default) | fastembed | hashing."""
    choice = os.environ.get("PROMPTLY_EMBEDDER", "auto")
    if choice == "hashing":
        return HashingEmbedder()
    model = os.environ.get("PROMPTLY_EMBED_MODEL", DEFAULT_FASTEMBED_MODEL)
    try:
        return FastEmbedEmbedder(model)
    except Exception as e:  # not installed, model download blocked, onnx missing, ...
        if choice == "fastembed":
            raise
        log.warning("fastembed unavailable (%s); using the hashing embedder", e)
        return HashingEmbedder()


class VectorCache:
    """Disk cache of text -> vector, namespaced by embedder name."""

    def __init__(self, embedder: Embedder, path: Path | None):
        self.embedder = embedder
        self.path = path
        self._lock = threading.Lock()
        self._mem: dict[str, np.ndarray] = {}
        self._dirty = False
        if path and path.exists():
            try:
                data = json.loads(path.read_text())
                if data.get("model") == embedder.name:
                    self._mem = {k: np.asarray(v, dtype=np.float32) for k, v in data["vectors"].items()}
            except (ValueError, KeyError):
                log.warning("ignoring unreadable vector cache at %s", path)

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.strip().encode()).hexdigest()[:24]

    def get_many(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedder.dim), dtype=np.float32)
        keys = [self.key(t) for t in texts]
        with self._lock:
            missing = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self._mem]
        if missing:
            vecs = self.embedder.embed([t for _, t in missing])
            with self._lock:
                for (k, _), v in zip(missing, vecs):
                    self._mem[k] = v
                self._dirty = True
            self.flush()
        with self._lock:
            return np.stack([self._mem[k] for k in keys])

    def get(self, text: str) -> np.ndarray:
        return self.get_many([text])[0]

    def prune(self, keep_texts: Sequence[str]) -> None:
        keep = {self.key(t) for t in keep_texts}
        with self._lock:
            before = len(self._mem)
            self._mem = {k: v for k, v in self._mem.items() if k in keep}
            self._dirty |= len(self._mem) != before
        self.flush()

    def flush(self) -> None:
        if not self.path or not self._dirty:
            return
        with self._lock:
            payload = {
                "model": self.embedder.name,
                "dim": self.embedder.dim,
                "vectors": {k: [round(float(x), 5) for x in v] for k, v in self._mem.items()},
            }
            self._dirty = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, self.path)
