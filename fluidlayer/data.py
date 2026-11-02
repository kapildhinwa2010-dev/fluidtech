"""
fluidlayer.data — tiny corpora + char-level tokenizer.

Prefers a real public-domain text (Project Gutenberg).  If the sandbox has no
outbound network access it falls back to a deterministic synthetic corpus so
that the A/B training run is always reproducible.
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np

__all__ = ["get_corpus", "CharTokenizer", "batchify", "GUTENBERG_URLS"]

GUTENBERG_URLS = [
    "https://www.gutenberg.org/files/11/11-0.txt",     # Alice in Wonderland
    "https://www.gutenberg.org/cache/epub/11/pg11.txt",
]


def get_corpus(max_chars: int = 120_000, cache: str | None = None) -> str:
    """Return a text corpus (>= max_chars if possible)."""
    cache = cache or os.path.join(os.path.dirname(__file__), "..", "corpus.txt")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8", errors="ignore") as f:
            t = f.read()
        if len(t) >= 1000:
            return t[:max_chars]
    for url in GUTENBERG_URLS:
        try:
            with urllib.request.urlopen(url, timeout=12) as r:
                raw = r.read().decode("utf-8", errors="ignore")
            start = raw.find("*** START")
            if start >= 0:
                raw = raw[start:]
            end = raw.find("*** END")
            if end > 0:
                raw = raw[:end]
            raw = raw.replace("\r\n", "\n")
            if len(raw) > 2000:
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
                    with open(cache, "w", encoding="utf-8") as f:
                        f.write(raw)
                except OSError:
                    pass
                return raw[:max_chars]
        except Exception:
            continue
    return _synthetic_corpus(max_chars)


def _synthetic_corpus(n: int) -> str:
    """Deterministic fallback: a Zipfian character process with word structure."""
    rng = np.random.default_rng(1234)
    words = ["the", "cat", "sat", "on", "a", "mat", "and", "then", "said", "little",
             "queen", "of", "hearts", "alice", "rabbit", "hole", "tea", "party",
             "night", "tree", "house", "river", "cloud", "stone", "light", "voice"]
    p = np.arange(1, len(words) + 1) ** -1.1
    p /= p.sum()
    out = []
    while sum(len(w) + 1 for w in out) < n:
        k = rng.choice(len(words), size=rng.integers(4, 14), p=p)
        out.extend(words[i] for i in k)
    s = " ".join(out)
    return s[:n]


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = self.chars

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.stoi[c] for c in text], dtype=np.int64)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    @property
    def vocab(self) -> int:
        return len(self.chars)


def batchify(tokens: np.ndarray, batch_size: int) -> np.ndarray:
    """(N,) -> (batch_size, N//batch_size) contiguous split."""
    nb = len(tokens) // batch_size
    return tokens[: nb * batch_size].reshape(batch_size, nb)
