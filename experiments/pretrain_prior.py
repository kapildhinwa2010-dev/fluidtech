#!/usr/bin/env python3
"""
tools/pretrain_prior.py — build the downloadable DOMAIN PRIOR artefact.

The prior is the thing a user actually downloads:

    prior/
      domain_prior.npz     the state + codebook + landmark table
      manifest.json        human-readable description and the budget split

It contains FOUR parts, and each one is paid for separately:

  1. CODEBOOK     (M x d fp16)  the static prototypes: the geometry.  Shared
                               across tenants, so it is not per-session cost.
  2. FLUID STATE  (M x r fp16)  the running "gist" of the domain.  This is
                               the CONSTANT part -- it never grows.
  3. LANDMARKS    (K x span)    the exact text of the structurally important
                               spans, indexed by the persistence barcode.  A
                               landmark is retrievable EXACTLY.
  4. REGISTER     (K x 20 B)    the append-only exact-fact slots: the frozen
                               register.  20 bytes per fact.

The budget law (see DESIGN-pinned-memory.md): naming one anchor out of M
needs log2(M) bits while an anchor holds r*16 bits, so r is grossly
over-provisioned for ADDRESSING.  A bigger budget should therefore buy MORE
PINS, not more anchors -- the fluid's per-token compute is linear in M, while
the register costs nothing at inference.

Usage
-----
    python3 tools/pretrain_prior.py --text corpus.txt
    python3 tools/pretrain_prior.py --synthetic --tokens 200000
    python3 tools/pretrain_prior.py --synthetic --tokens 200000 --budget 5MB
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from superfluid.memory import (SheafMemoryAdapter, FrozenRegister, SLOT_BYTES,
                               embedding_signature, ctx_key, budget_split)
from superfluid.topology import (hodge_relevance, hodge_embedding,
                                 hodge_edge_participation, persistence_landmarks,
                                 local_density)

DEFAULT_OUT = os.path.join(ROOT, "prior")


# ------------------------------------------------------------------ corpus
def load_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def hash_tokens(text, vocab_size=4096):
    """Deterministic byte-level vocabulary (no external tokenizer needed)."""
    arr = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    return (arr.astype(np.int64) % vocab_size), vocab_size


def synthetic_corpus(n_tokens=200_000, vocab_size=4096, n_topics=20, seed=0):
    """Markov-ish synthetic text with topic structure (same family as E5)."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_topics, 8))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    topic = rng.integers(0, n_topics, size=n_tokens)
    within = rng.integers(0, vocab_size // n_topics, size=n_tokens)
    toks = topic * (vocab_size // n_topics) + within
    return toks.astype(np.int64), vocab_size, centers


# ---------------------------------------------------------------- codebook
def kmeans(X, k, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    C = np.empty((k, X.shape[1]), dtype=X.dtype)
    C[0] = X[rng.integers(n)]
    d2 = ((X - C[0]) ** 2).sum(1)
    for i in range(1, k):
        p = d2 / max(d2.sum(), 1e-12)
        C[i] = X[rng.choice(n, p=p)]
        d2 = np.minimum(d2, ((X - C[i]) ** 2).sum(1))
    for _ in range(iters):
        lab = np.empty(n, dtype=np.int64)
        for s in range(0, n, 4096):
            blk = X[s:s + 4096]
            sc = blk @ C.T
            sc -= (blk ** 2).sum(1)[:, None]
            sc -= (C ** 2).sum(1)[None, :]
            lab[s:s + 4096] = np.argmin(sc, axis=1)
        counts = np.bincount(lab, minlength=k)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
        empty = np.nonzero(counts == 0)[0]
        if empty.size:
            d = np.linalg.norm(X - C[lab], axis=1)
            far = np.argsort(-d)
            for t, j in enumerate(empty):
                C[j] = X[far[t % far.size]]
    return C


# -------------------------------------------------------------------- main
def build(tokens, vocab_size, budget_bytes=16 * 1024, M=1024, r=8, d=64,
          span_len=64, seed=0, out_dir=DEFAULT_OUT, cheb_m=12, chunk=8192,
          pins=256, verbose=True):
    t0 = time.time()
    log = print if verbose else (lambda *a: None)
    os.makedirs(out_dir, exist_ok=True)

    # --- embeddings: one random unit vector per symbol --------------------
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((vocab_size, d))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    # --- codebook ---------------------------------------------------------
    log(f"[1/5] codebook: k-means over {vocab_size} symbols ...")
    protos = kmeans(emb, M, iters=25, seed=seed)

    # --- budget law -------------------------------------------------------
    # The runtime budget is split between the FLUID (addressing + gist) and
    # the REGISTER (exactness).  Anchoring needs only log2(M) bits per cell
    # while a cell holds r*16 bits, so the fluid is deliberately kept at half
    # the budget and the rest buys exact slots.
    codebook_bytes = int((protos.size + d * r) * 2)
    reserve = int(pins) * SLOT_BYTES
    M_cap = max((budget_bytes - reserve) // (r * 2), 1)
    M_req = int(min(M, M_cap))
    split = budget_split(budget_bytes, M=M_req, r=r, slot_bytes=SLOT_BYTES,
                         codebook_bytes=codebook_bytes)
    M_use = split["M_used"]
    protos = protos[:M_use]
    n_slots = split["pins"]
    log(f"      budget {budget_bytes} B -> M={M_use} anchors (asked {M}), "
        f"fluid {split['fluid_bytes']} B, register {split['register_bytes']} B "
        f"({n_slots} slots)")

    # --- stream the corpus through the fluid ------------------------------
    log(f"[2/5] streaming {len(tokens):,} tokens through the fluid ...")
    ad = SheafMemoryAdapter(M=M_use, r=r, d=d, capacity=max(n_slots, 16), seed=seed)
    ad.set_prototypes(protos)
    n_chunks = (len(tokens) + chunk - 1) // chunk
    t_chunk = 1.0 / max(n_chunks, 1)
    for s in range(0, len(tokens), chunk):
        blk = emb[tokens[s:s + chunk]]
        ad.write(blk, t=t_chunk, cheb_m=cheb_m, diffuse=True)

    # --- anchor geometry + persistence landmarks --------------------------
    log("[3/5] anchor geometry and persistence landmarks ...")
    edges, alpha, _ = ad.anchor_graph(k=8)
    hp = hodge_relevance(M_use, edges, alpha)
    dens = local_density(protos, k=8)
    lm = persistence_landmarks(M_use, edges, alpha, k=min(M_use, 256))

    # --- landmark table: the EXACT spans at the structural landmarks ------
    # a landmark is an anchor; the span is the most frequent token window
    # that routed to it.  We keep the span's exact token ids.
    from superfluid.topology import route_topk
    q = emb[tokens]
    Pn = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12)
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    anchor_of = np.argmax(qn @ Pn.T, axis=1)
    spans = {}
    for a in lm[:min(len(lm), 256)]:
        idx = np.nonzero(anchor_of == a)[0]
        if idx.size == 0:
            continue
        mid = int(idx[idx.size // 2])
        lo = max(0, mid - span_len // 2)
        hi = min(len(tokens), lo + span_len)
        spans[int(a)] = tokens[lo:hi].astype(np.int32)
    # --- frozen register: one slot per landmark ---------------------------
    reg = FrozenRegister(capacity=max(n_slots, len(spans) + 1))
    for a, span in spans.items():
        if reg.n >= reg.capacity:
            break
        key = ctx_key([int(x) for x in span[:4]])
        reg.promote(int(span[-1]), key,
                    sig=embedding_signature(emb[int(span[0])]),
                    anchor=int(a), doc_id=0, trust=1)
    log(f"      {len(spans)} landmarks, {reg.n} frozen slots")

    # --- save -------------------------------------------------------------
    log("[4/5] writing the artefact ...")
    npz = os.path.join(out_dir, "domain_prior.npz")
    np.savez_compressed(
        npz,
        protos=protos.astype(np.float16),
        anchors=ad.anchors.astype(np.float16),
        W_in=ad.W_in.astype(np.float16),
        hodge_relevance=hp.astype(np.float32),
        local_density=dens.astype(np.float32),
        reg_key=reg.key, reg_val=reg.val, reg_sig=reg.sig,
        reg_anchor=reg.anchor, reg_meta=reg.meta, reg_used=reg.used,
        landmark_ids=np.array(sorted(spans.keys()), dtype=np.int32),
        landmark_spans=np.stack([spans[k] for k in sorted(spans)]) if spans
        else np.zeros((0, 0), dtype=np.int32),
    )
    size = os.path.getsize(npz)

    manifest = {
        "name": "superfluid domain prior",
        "version": "1.0",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_state_bytes": int(ad.state_bytes()),
        "codebook_bytes": int(codebook_bytes),
        "artefact_bytes_on_disk": int(size),
        "budget": {
            "requested_bytes": int(budget_bytes),
            "M": int(M_use),
            "r": int(r),
            "slots": int(n_slots),
            "landmark_slots_written": int(reg.n),
            "slot_bytes": SLOT_BYTES,
            "fluid_bytes": int(split["fluid_bytes"]),
            "register_bytes": int(split["register_bytes"]),
        },
        "contents": {
            "codebook": "prototypes (M x d) fp16 -- the static geometry",
            "fluid_state": "anchors (M x r) fp16 -- the constant-size gist",
            "landmarks": "exact token spans at the persistent structural classes",
            "register": "append-only exact-fact slots (20 B each)",
        },
        "guarantees": [
            "a frozen slot's value is bit-identical to the source token id",
            "the register is append-only (audit log, first-write-wins)",
            "the runtime state is constant size -- it never grows with input",
        ],
        "not_guaranteed": [
            "positional recall",
            "verbatim recall of text that is not in the landmark table",
        ],
        "tokens_streamed": int(len(tokens)),
        "runtime_sec": round(time.time() - t0, 2),
    }
    mpath = os.path.join(out_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"[5/5] done in {time.time() - t0:.1f}s")
    log(f"      {npz}  ({size:,} bytes on disk)")
    log(f"      {mpath}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None, help="a UTF-8 text file")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--tokens", type=int, default=200_000)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--budget", type=str, default="16KB",
                    help="runtime state budget: 16KB / 1MB / 5MB ...")
    ap.add_argument("--M", type=int, default=1024)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--pins", type=int, default=256,
                    help="how many exact slots to reserve in the runtime budget")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    units = {"KB": 1024, "MB": 1024 ** 2, "B": 1}
    b = args.budget.upper()
    mult = units.get(b.replace(" ", "")[-2:], None)
    if mult is None:
        mult = 1
    num = float("".join(ch for ch in b if ch.isdigit() or ch == "."))
    budget = int(num * mult)

    if args.text:
        text = load_text(args.text)
        tokens, vocab = hash_tokens(text, args.vocab)
    elif args.synthetic:
        tokens, vocab, _ = synthetic_corpus(args.tokens, args.vocab)
    else:
        print("error: pass --text FILE or --synthetic", file=sys.stderr)
        return 2

    m = build(tokens, vocab, budget_bytes=budget, M=args.M, r=args.r, d=args.d,
              pins=args.pins, out_dir=args.out)
    print(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
