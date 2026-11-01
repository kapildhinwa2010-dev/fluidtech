"""
superfluid.gpt2_tokenizer — the GPT-2 byte-level BPE, in pure Python.

WHY THIS FILE EXISTS
--------------------
The project's claim is that a memory layer can be attached to **any**
pretrained LLM.  To test that on a real checkpoint without dragging in
`transformers` (and its `regex` dependency), this file implements the
GPT-2 tokenizer directly: the byte-to-unicode map, the BPE merges, and
the pre-tokenizer regex.  It reads `vocab.json` + `merges.txt` from a
checkpoint directory.

It is exact: `decode(encode(text)) == text` for any text, including text
the checkpoint never saw (the byte fallback is always available).
"""

from __future__ import annotations

import json
import os
import re

__all__ = ["GPT2Tokenizer", "bytes_to_unicode"]


def bytes_to_unicode() -> dict:
    """GPT-2's reversible byte -> unicode map."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _get_pairs(word):
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


# GPT-2's pre-tokenizer pattern (the stdlib `re` handles it fine)
_PRE = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\w+|\S+|\s+""")


class GPT2Tokenizer:
    """The GPT-2 BPE tokenizer, loadable from a checkpoint directory."""

    def __init__(self, vocab: dict, merges: list):
        self.encoder = vocab
        self.decoder = {v: k for k, v in vocab.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = {tuple(m): i for i, m in enumerate(merges)}
        self.cache: dict = {}

    @classmethod
    def from_pretrained(cls, path: str) -> "GPT2Tokenizer":
        with open(os.path.join(path, "vocab.json"), encoding="utf-8") as f:
            vocab = json.load(f)
        merges = []
        p = os.path.join(path, "merges.txt")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                lines = f.read().split("\n")
            merges = [tuple(l.split()) for l in lines[1:] if l.strip()]
        return cls(vocab, merges)

    # ------------------------------------------------------------------ BPE
    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = _get_pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, 1e10))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        out = " ".join(word)
        self.cache[token] = out
        return out

    # --------------------------------------------------------------- public
    def encode(self, text: str):
        ids = []
        for tok in _PRE.findall(text):
            tok = "".join(self.byte_encoder[b] for b in tok.encode("utf-8"))
            ids.extend(self.encoder[t] for t in self.bpe(tok).split(" "))
        return ids

    def decode(self, ids) -> str:
        text = "".join(self.decoder[int(i)] for i in ids)
        raw = bytearray()
        for c in text:
            raw.append(self.byte_decoder[c])
        return raw.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)
