"""
fluidlayer/bpe.py — a byte-level BPE tokenizer, in pure Python (no downloads).

WHY THIS EXISTS
---------------
Every experiment so far ran at the CHARACTER level (vocab 76).  That is
convenient but it is not what a real deployment looks like: real LLM
tokenizers are subword, with a vocabulary of 32k-256k and a very different
redundancy structure.  The character-level corpus is *maximally* redundant
(a 12-gram of characters repeats constantly), which flatters a
content-keyed store.  A subword corpus repeats less.

The point of this file is therefore not "a tokenizer" -- it is the ability
to re-run every measurement at a realistic token granularity, on CPU,
without downloading anything.

THE ALGORITHM
-------------
1. split the text on a word-ish regex,
2. count word frequencies,
3. represent every word as its UTF-8 BYTES (so the tokenizer is lossless
   for any input, including text the training corpus never saw),
4. repeatedly merge the most frequent adjacent symbol pair,
5. emit a vocabulary: the 256 byte tokens + every learned merge.

`encode` is then exact and reversible (`decode(encode(s)) == s` for any
string, because the byte fallback is always available).
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter

__all__ = ["BPETokenizer", "train_bpe"]

_WORD = re.compile(r"\S+|\s+")


class BPETokenizer:
    """A byte-level BPE tokenizer."""

    def __init__(self, merges: dict, id_bytes: dict):
        # merges: {(a, b): new_id}  -- the merge ORDER matters
        # id_bytes: {id: bytes}   -- every id, bytes and merged alike
        self.merges = merges
        self.ranks = {p: i for i, p in enumerate(merges.keys())}
        self.id_to_bytes = dict(id_bytes)

    # ------------------------------------------------------------------ size
    @property
    def vocab_size(self) -> int:
        return len(self.id_to_bytes)

    # --------------------------------------------------------------- encode
    def _word_symbols(self, word: str):
        b = word.encode("utf-8")
        return tuple(b)

    def _merge_word(self, symbols):
        while len(symbols) > 1:
            best = None
            best_i = -1
            for i in range(len(symbols) - 1):
                r = self.ranks.get((symbols[i], symbols[i + 1]))
                if r is not None and (best is None or r < best):
                    best = r
                    best_i = i
            if best is None:
                break
            a, b = symbols[best_i], symbols[best_i + 1]
            new = self.merges.get((a, b))
            if new is None:
                break
            symbols = symbols[:best_i] + (new,) + symbols[best_i + 2:]
        return symbols

    def encode(self, text: str):
        ids = []
        for w in _WORD.findall(text):
            ids.extend(self._merge_word(self._word_symbols(w)))
        return ids

    def decode(self, ids) -> str:
        out = bytearray()
        for i in ids:
            out += self.id_to_bytes[int(i)]
        return out.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ io
    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "merges": [[list(k), v] for k, v in self.merges.items()],
                "id_bytes": {str(k): list(v) for k, v in self.id_to_bytes.items()},
            }, f)

    @staticmethod
    def load(path: str) -> BPETokenizer:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        merges = {tuple(m[0]): m[1] for m in d["merges"]}
        id_bytes = {int(k): bytes(v) for k, v in d["id_bytes"].items()}
        return BPETokenizer(merges, id_bytes)


def train_bpe(text: str, vocab_size: int = 1024, verbose: bool = True):
    """Train a byte-level BPE on `text`.  Deterministic, no randomness."""
    words = Counter(_WORD.findall(text))
    # each word -> its symbol tuple; count pairs over the whole corpus
    word_syms = {w: tuple(w.encode("utf-8")) for w in words}
    merges: dict = {}
    id_bytes: dict = {i: bytes([i]) for i in range(256)}

    def recount():
        pairs = Counter()
        for w, s in word_syms.items():
            c = words[w]
            for i in range(len(s) - 1):
                pairs[(s[i], s[i + 1])] += c
        return pairs

    n_merges = vocab_size - 256
    for step in range(n_merges):
        pairs = recount()
        if not pairs:
            break
        # deterministic tie-break: the highest count, then the lowest ids
        best = max(pairs.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
        pair, cnt = best
        if cnt < 2:
            break
        new_id = 256 + len(merges)
        merges[pair] = new_id
        id_bytes[new_id] = id_bytes[pair[0]] + id_bytes[pair[1]]
        # rewrite every word
        for w in list(word_syms):
            s = word_syms[w]
            out = []
            i = 0
            while i < len(s):
                if i + 1 < len(s) and (s[i], s[i + 1]) == pair:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            word_syms[w] = tuple(out)
        if verbose and step % 100 == 0:
            print(f"    merge {step}: {pair} -> {new_id}  (count {cnt})")
    return BPETokenizer(merges, id_bytes)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus.txt")
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--out", default="prior/bpe_1024.json")
    args = ap.parse_args()
    with open(args.corpus, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    tok = train_bpe(text, args.vocab)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok.save(args.out)
    ids = tok.encode("Alice was beginning to get very tired of sitting by "
                      "her sister on the bank, and of having nothing to do.")
    print(f"    vocab {tok.vocab_size}  tokens {len(ids)}  "
          f"-> {ids[:12]} ...")
    print(f"    round-trip ok: {tok.decode(ids)[:60]!r}")
    print(f"    wrote {args.out}")
