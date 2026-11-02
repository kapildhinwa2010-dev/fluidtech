#!/usr/bin/env python3
"""
experiments/e5_graft.py — THE money experiment.

    "Can a 16 KB memory return the EXACT next token for a selected context out
     of a 1,000,000-token document?"

What this experiment measures
-----------------------------
A constant-size state CANNOT do instance-level recall: 16 KB over 60,000
tokens is 0.27 bytes per token, so the fluid necessarily answers "the gist of
this neighbourhood", not "this specific token".  That is not a defect of the
design -- it is the reason the FROZEN REGISTER exists.

So the experiment is split into two questions:

  Q1 (the fluid)    Can the 16 KB state carry a useful AGGREGATE?
                    -> cosine between the diffused anchor content and the true
                       mean of the tokens that routed there (local gist), and
                       against the global mean (the harmonic limit).

  Q2 (the register) Which contexts are worth the exactness budget, and does
                    the selection criterion find them?
                    -> PIN PRECISION (of the K promoted slots, how many are
                       real fact contexts) and FACT RECALL (of the test facts,
                       how many can be returned bit-exactly).

Semantics of a promoted slot
----------------------------
The memory records (context -> next token), exactly like an LM does:

    key   = hash of the W=4 tokens ending at position q
    val   = tokens[q + 1]                 <- the exact next token id
    sig   = 4-byte signature of tokens[q]  <- verifiable at recall time
    anchor= the anchor the context routed to

A fact is recoverable iff position q was promoted AND the recalled value is the
fact's value.  The value is an INTEGER token id, so a successful recall is
bit-exact, not approximate.

The policies that compete for the budget (all get the SAME budget, and NONE
of them is allowed to look at the fact list):

  random     uniformly random positions                        (the CONTROL)
  surprise   positions whose (context, next) PAIR is new        (Titans-style)
  rarity     positions whose NEXT token is a first occurrence   (novelty gate)
  topology   positions in the anchors with the highest HODGE    (our criterion)
             RELEVANCE
  crowding   positions in the anchors with the highest TOKEN
             OCCUPANCY (a streaming pre-pass; no fact knowledge)
  oracle     the fact contexts                                (UPPER BOUND)

Everything runs on CPU in pure NumPy.

Usage
-----
    python3 experiments/e5_graft.py                 # 1M tokens (default)
    python3 experiments/e5_graft.py --tokens 250000
    python3 experiments/e5_graft.py --quick         # small smoke run
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

from fluidlayer.memory import (
    SLOT_BYTES,
    FrozenRegister,
    SheafMemoryAdapter,
    ctx_key,
    embedding_signature,
    logit_override,
)
from fluidlayer.topology import (
    hodge_edge_participation,
    hodge_embedding,
    hodge_relevance,
    local_density,
    persistence_landmarks,
)

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CTX = 4          # context window length (tokens ending at the promoted position)


# ----------------------------------------------------------------- corpus
class Corpus:
    """Synthetic corpus: topic-blocked Markov text + injected facts.

    The text is a Markov chain inside topic blocks, so ordinary bigrams
    REPEAT.  A fact is a (key symbol, value symbol) pair that occurs ONCE,
    which is exactly what makes it findable by a novelty criterion and
    invisible to a frequency criterion.
    """

    def __init__(self, n_tokens=1_000_000, d=64, n_topics=20, symbols_per_topic=200,
                 n_crowded_topics=5, facts_per_crowded=30, facts_per_isolated=3,
                 key_reps=5, val_reps=5, seed=0):
        rng = np.random.default_rng(seed)
        self.n_tokens = int(n_tokens)
        self.d = d
        self.n_topics = n_topics
        self.symbols_per_topic = symbols_per_topic
        self.key_reps = key_reps
        self.val_reps = val_reps
        self.rng = rng

        n_base = n_topics * symbols_per_topic
        self.n_base = n_base
        self.centers = rng.standard_normal((n_topics, d))
        self.centers /= np.linalg.norm(self.centers, axis=1, keepdims=True)
        self.topic_of_symbol = np.repeat(np.arange(n_topics), symbols_per_topic)
        # the noise is NORM-RELATIVE: a unit centre plus a perturbation of
        # length 0.30.  (An unnormalised perturbation of length ~2.4 would
        # destroy the topic structure entirely -- measured base-base cosine
        # collapsed from 0.80 to 0.14.)
        base_emb = (self.centers[self.topic_of_symbol] +
                    0.30 * rng.standard_normal((n_base, d)) / np.sqrt(d))
        base_emb /= np.linalg.norm(base_emb, axis=1, keepdims=True)

        # --- facts ---------------------------------------------------------
        facts = []
        for t in range(n_crowded_topics):
            facts += [(t, "crowded")] * facts_per_crowded
        for t in range(n_crowded_topics, n_topics):
            facts += [(t, "isolated")] * facts_per_isolated
        self.facts = facts
        self.n_facts = len(facts)

        f_keys = np.zeros((self.n_facts, d))
        f_vals = np.zeros((self.n_facts, d))
        self.fact_key_sym = np.zeros(self.n_facts, dtype=np.int64)
        self.fact_val_sym = np.zeros(self.n_facts, dtype=np.int64)
        self.fact_kind = []
        for i, (t, kind) in enumerate(facts):
            spread = 0.03 if kind == "crowded" else 0.25
            c = self.centers[t]
            k = c + spread * rng.standard_normal(d) / np.sqrt(d)
            k /= np.linalg.norm(k)
            v = k + 0.05 * rng.standard_normal(d) / np.sqrt(d)   # value next to key
            v /= np.linalg.norm(v)
            f_keys[i] = k
            f_vals[i] = v
            self.fact_key_sym[i] = n_base + 2 * i
            self.fact_val_sym[i] = n_base + 2 * i + 1
            self.fact_kind.append(kind)

        self.emb = np.concatenate([base_emb, f_keys, f_vals], axis=0)
        self.n_symbols = self.emb.shape[0]
        self.key_emb = f_keys
        self.val_emb = f_vals

        # --- token stream --------------------------------------------------
        self.fact_pos = np.full(self.n_facts, -1, dtype=np.int64)
        self.tokens = self._build_stream(rng)

    # -- a Markov run inside one topic ------------------------------------
    def _markov_run(self, rng, table, length, start):
        out = np.empty(length, dtype=np.int64)
        r = rng.integers(0, table.shape[1], size=length)
        s = int(start)
        for i in range(length):
            out[i] = s
            s = int(table[s, r[i]])
        return out

    def _build_stream(self, rng):
        n = self.n_tokens
        # per-topic transition tables over that topic's symbol pool
        tables = []
        for _ in range(self.n_topics):
            tab = np.zeros((self.symbols_per_topic, 6), dtype=np.int64)
            for s in range(self.symbols_per_topic):
                # no self-loops: a token repeated twice is not a new
                # association, and self-loops would swamp the candidate set
                nxt = rng.integers(0, self.symbols_per_topic - 1, size=6)
                tab[s] = np.where(nxt >= s, nxt + 1, nxt)
            tables.append(tab)
        # topic blocks
        tok = np.zeros(n, dtype=np.int64)
        pos = 0
        while pos < n:
            t = int(rng.integers(0, self.n_topics))
            L = int(rng.integers(500, 3000))
            L = min(L, n - pos)
            run = self._markov_run(rng, tables[t], L,
                                   int(rng.integers(0, self.symbols_per_topic)))
            tok[pos:pos + L] = t * self.symbols_per_topic + run
            pos += L
        # inject the facts: the pair (K_i, V_i) occurs once, but K_i and V_i
        # each also appear key_reps/val_reps times elsewhere so that raw
        # frequency carries no signal.
        for i in range(self.n_facts):
            K = int(self.fact_key_sym[i])
            V = int(self.fact_val_sym[i])
            # the fact occurrence first, so that no filler can overwrite it
            p = int(rng.integers(CTX + 2, n - 2))
            tok[p] = K
            tok[p + 1] = V
            self.fact_pos[i] = p
            # filler occurrences of K and V elsewhere, never adjacent to the
            # fact occurrence (so raw frequency carries no signal)
            forbidden = {p, p + 1, p - 1}
            pool = np.array([e for e in range(10, n - 10) if e not in forbidden])
            n_fill = self.key_reps + self.val_reps - 2
            if pool.size > n_fill:
                extra = np.sort(rng.choice(pool, size=n_fill, replace=False))
                for j, e in enumerate(extra):
                    tok[e] = K if j < self.key_reps - 1 else V
        return tok


# ------------------------------------------------------------- codebook
def kmeans(X: np.ndarray, k: int, iters: int = 25, seed: int = 0):
    """Plain k-means (k-means++ init) -- the 'domain prior' is LEARNED, not placed."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = int(min(k, n))
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
        # --- re-seed empty centroids -------------------------------------
        # With tight clusters, k-means++ can place several centroids in the
        # SAME ball; they then collapse onto one another and half the codebook
        # ends up empty (measured: median occupancy 0).  The standard fix is
        # to re-seed an empty centroid at the point that is currently worst
        # represented.  This is what makes the codebook actually USABLE.
        empty = np.nonzero(counts == 0)[0]
        if empty.size:
            d = np.linalg.norm(X - C[lab], axis=1)
            far = np.argsort(-d)
            for t, j in enumerate(empty):
                C[j] = X[far[t % far.size]]
    return C


# --------------------------------------------------------------- helpers
def route_all(emb: np.ndarray, protos: np.ndarray, chunk: int = 8192):
    """Nearest prototype for every row of `emb` -> (n,) anchor ids."""
    P = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12)
    out = np.empty(emb.shape[0], dtype=np.int64)
    for s in range(0, emb.shape[0], chunk):
        blk = emb[s:s + chunk]
        blk = blk / (np.linalg.norm(blk, axis=1, keepdims=True) + 1e-12)
        out[s:s + chunk] = np.argmax(blk @ P.T, axis=1)
    return out


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float = 1e-2):
    """Ridge regression  W = argmin ||XW - Y||^2 + lam||W||^2.  Returns W (p, q)."""
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ Y)


def cos_mean(a: np.ndarray, b: np.ndarray) -> float:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return float((an * bn).sum(1).mean())


def window_keys(tokens: np.ndarray, q: np.ndarray) -> np.ndarray:
    """64-bit context keys for the W tokens ending at each position in q."""
    out = np.empty(q.size, dtype=np.uint64)
    for i, p in enumerate(q):
        out[i] = ctx_key([int(x) for x in tokens[p - CTX + 1: p + 1]])
    return out


# ------------------------------------------------------------- the experiment
def run(n_tokens=1_000_000, d=64, M=1024, r=8, budget=100, seed=0, cheb_m=12,
        diffuse_total=1.0, chunk=8192, n_facts_test=200, verbose=True):
    t0 = time.time()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    log(f"[1/8] building corpus ({n_tokens:,} tokens, Markov blocks) ...")
    corp = Corpus(n_tokens=n_tokens, d=d, seed=seed)
    log(f"      symbols={corp.n_symbols}  facts={corp.n_facts}  "
        f"({corp.fact_kind.count('crowded')} crowded, "
        f"{corp.fact_kind.count('isolated')} isolated)")

    log(f"[2/8] learning the codebook (k-means, M={M}) ...")
    sample_idx = np.random.default_rng(seed).choice(corp.n_symbols, size=4096,
                                                     replace=False)
    protos = kmeans(corp.emb[sample_idx].astype(np.float64), M, iters=25, seed=seed)
    ad = SheafMemoryAdapter(M=M, r=r, d=d, capacity=max(budget * 4, 64), seed=seed)
    ad.set_prototypes(protos)
    log(f"      done in {time.time() - t0:.1f}s")

    # ---- pre-pass: token -> anchor occupancy (no fact knowledge used) ---
    log("[3/8] pre-pass: token->anchor occupancy (no fact knowledge used) ...")
    hist = np.zeros(M, dtype=np.int64)
    sum_proj = np.zeros((M, r), dtype=np.float64)
    tok_anchor = np.empty(corp.n_tokens, dtype=np.int64)
    W_in = ad.W_in
    for s in range(0, corp.n_tokens, chunk):
        toks = corp.tokens[s:s + chunk]
        blk = corp.emb[toks]
        a = route_all(blk, protos)
        tok_anchor[s:s + chunk] = a
        np.add.at(hist, a, 1)
        np.add.at(sum_proj, a, blk @ W_in)
    fact_positions = set(int(p) for p in corp.fact_pos)

    log("[4/8] anchor geometry: Hodge relevance + local density ...")
    edges, alpha, _ = ad.anchor_graph(k=8)
    hp = hodge_relevance(M, edges, alpha)
    dens = local_density(protos, k=8)
    log(f"      Hodge relevance: min={hp.min():.4f} med={np.median(hp):.4f} max={hp.max():.4f}")
    log(f"      occupancy: max={hist.max()} med={int(np.median(hist))} anchors")

    # ---- stream the document through the fluid -------------------------
    log(f"[5/8] streaming the document through the {ad.state_bytes()}-byte fluid ...")
    n_chunks = (corp.n_tokens + chunk - 1) // chunk
    t_chunk = diffuse_total / max(n_chunks, 1)
    for s in range(0, corp.n_tokens, chunk):
        blk = corp.emb[corp.tokens[s:s + chunk]]
        ad.write(blk, t=t_chunk, cheb_m=cheb_m, diffuse=True)
    log(f"      done in {time.time() - t0:.1f}s  (tau={diffuse_total}, {n_chunks} chunks)")

    # ---- Q1: does the fluid carry a useful AGGREGATE? -------------------
    counts = np.maximum(hist, 1)[:, None]
    true_local = sum_proj / counts
    global_mean = sum_proj.sum(0) / max(hist.sum(), 1)
    live = hist > 0
    gist_local = cos_mean(ad.anchors[live], true_local[live])
    gist_global = cos_mean(ad.anchors[live],
                            np.repeat(global_mean[None, :], int(live.sum()), axis=0))
    log(f"      Q1 gist cosine: local={gist_local:.4f}  global={gist_global:.4f}")

    # ---- Q2: which contexts deserve the exact budget? -------------------
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(corp.n_facts)
    n_train = max(corp.n_facts // 3, 1)
    te = perm[n_train:n_train + n_facts_test]
    log(f"      Q2 test facts: {len(te)}   budget: {budget} slots "
        f"({budget * SLOT_BYTES / 1024:.1f} KB register)")

    # candidate positions: q in [CTX, N-1) (a context needs CTX tokens)
    q_all = np.arange(CTX, corp.n_tokens - 1, dtype=np.int64)
    # a self-pair is not a new association -> drop it from the candidates
    valid = corp.tokens[q_all] != corp.tokens[np.minimum(q_all + 1, corp.n_tokens - 1)]
    q_all = q_all[valid]
    q_anchor = tok_anchor[q_all]
    toks = corp.tokens

    # novelty signals (all computable in a streaming pre-pass) ------------
    # (a) pair novelty: the bigram (tokens[q], tokens[q+1]) is new
    pair_code = toks[:-1] * corp.n_symbols + toks[1:]
    srt = np.argsort(pair_code, kind="stable")
    pc = pair_code[srt]
    newpair = np.ones(pc.size, dtype=bool)
    newpair[1:] = pc[1:] != pc[:-1]
    pair_first = np.zeros(pair_code.size, dtype=bool)
    pair_first[srt[newpair]] = True
    pair_novel = pair_first[q_all]          # pair (tokens[q], tokens[q+1])
    # (b) rarity: the NEXT token is a first occurrence of its symbol
    srt2 = np.argsort(toks, kind="stable")
    st = toks[srt2]
    newgrp = np.ones(st.size, dtype=bool)
    newgrp[1:] = st[1:] != st[:-1]
    first_pos = np.zeros(corp.n_tokens, dtype=bool)
    first_pos[srt2[newgrp]] = True
    rarity = first_pos[np.minimum(q_all + 1, corp.n_tokens - 1)]

    rngp = np.random.default_rng(seed + 2)
    rand_score = rngp.random(q_all.size)
    hp_score = hp[q_anchor]
    cr_score = hist[q_anchor].astype(np.float64)

    # --- the sheaf-based criterion --------------------------------------
    # A FACT is a NEW association that the geometry says belongs together.
    # The affinity of the pair (tokens[q], tokens[q+1]) is measured by the
    # HODGE EDGE PARTICIPATION of their two cells in the anchor complex.
    # A distractor (a new but unrelated co-occurrence) has low affinity.
    Psi = hodge_embedding(M, edges, alpha)
    a_q = tok_anchor[q_all]
    b_q = tok_anchor[np.minimum(q_all + 1, corp.n_tokens - 1)]
    compat_topo = hodge_edge_participation(Psi, a_q, b_q)
    e_q = corp.emb[toks[q_all]]
    e_q1 = corp.emb[toks[np.minimum(q_all + 1, corp.n_tokens - 1)]]
    compat_sem = np.sum(e_q * e_q1, axis=1) / (
        np.linalg.norm(e_q, axis=1) * np.linalg.norm(e_q1, axis=1) + 1e-12)
    topo_score = pair_novel * compat_topo
    sem_score = pair_novel * compat_sem

    # ---- RESOLUTION ABLATION -------------------------------------------
    # The criterion above lives on the RUNTIME anchor complex (M anchors),
    # so it is quantised: two tokens in the same cell score 1 even if their
    # fine geometry differs a lot.  The selection task may need finer
    # resolution than the runtime state does.  So we also build a FINER
    # codebook (M' = 4M) -- it is part of the static prior artefact, not the
    # 16 KB runtime state -- and recompute the Hodge participation there.
    Mf = int(min(M * 4, 4096))
    fine_idx = np.random.default_rng(seed + 7).choice(
        corp.n_symbols, size=min(8192, corp.n_symbols), replace=False)
    protos_f = kmeans(corp.emb[fine_idx].astype(np.float64), Mf, iters=20, seed=seed + 7)
    anchor_f = route_all(corp.emb, protos_f)
    ad_f = SheafMemoryAdapter(M=Mf, r=r, d=d, capacity=64, seed=seed + 7)
    ad_f.set_prototypes(protos_f)
    e_f, a_f, _ = ad_f.anchor_graph(k=8)
    Psi_f = hodge_embedding(Mf, e_f, a_f)
    compat_topo_f = hodge_edge_participation(
        Psi_f, anchor_f[toks[q_all]], anchor_f[toks[np.minimum(q_all + 1, corp.n_tokens - 1)]])
    topo_fine_score = pair_novel * compat_topo_f

    # persistence landmarks: is the context's cell one of the long-lived
    # structural classes of the corpus?  (a binary gate)
    lm = set(int(x) for x in persistence_landmarks(M, edges, alpha, k=64))
    persist_score = np.array([1.0 if a in lm else 0.0 for a in q_anchor])
    persist_score = pair_novel * persist_score
    # local density of the context's cell
    density_score = pair_novel * dens[q_anchor]

    def topk_positions(score, k):
        k = int(min(k, score.shape[0]))
        idx = np.argpartition(-score, kth=k - 1)[:k]
        return set(int(x) for x in q_all[idx])

    fact_pos_sorted = np.array(sorted(fact_positions), dtype=np.int64)
    policies = {
        "random": topk_positions(rand_score, budget),
        "surprise": topk_positions(pair_novel.astype(np.float64), budget),
        "rarity": topk_positions(rarity.astype(np.float64), budget),
        "topology": topk_positions(topo_score, budget),
        "topology_fine": topk_positions(topo_fine_score, budget),
        "semantic": topk_positions(sem_score, budget),
        "crowding": topk_positions(cr_score, budget),
        "persistence": topk_positions(persist_score, budget),
        "density": topk_positions(density_score, budget),
        "hodge": topk_positions(hp_score, budget),
        "oracle": set(int(x) for x in fact_pos_sorted[:budget]),
    }

    log("[6/8] evaluating policies ...")
    results = {}
    for name, positions in policies.items():
        reg = FrozenRegister(capacity=len(positions) + 8)
        for p in sorted(positions):
            w = [int(x) for x in toks[p - CTX + 1: p + 1]]
            reg.promote(int(toks[p + 1]), ctx_key(w, doc_id=0),
                        sig=embedding_signature(corp.emb[toks[p]]),
                        anchor=int(tok_anchor[p]), doc_id=0)
        n_fact_slots = sum(1 for p in positions if p in fact_positions)
        precision = n_fact_slots / max(len(positions), 1)
        exact = 0
        wrong = 0
        for i in te:
            p = int(corp.fact_pos[i])
            if p not in positions:
                continue
            w = [int(x) for x in toks[p - CTX + 1: p + 1]]
            got = reg.recall(ctx_key(w, doc_id=0),
                             sig=embedding_signature(corp.emb[toks[p]]))
            if got is None:
                wrong += 1
            elif int(got[1]) == int(corp.fact_val_sym[i]):
                exact += 1
            else:
                wrong += 1
        recall = exact / max(len(te), 1)
        n_promoted_test = sum(1 for i in te if int(corp.fact_pos[i]) in positions)
        exact_given = exact / max(n_promoted_test, 1)
        results[name] = {
            "pin_precision": precision,
            "fact_recall": recall,
            "promoted_test_facts": int(n_promoted_test),
            "exact_given_promoted": float(exact_given),
            "exact_hits": int(exact),
            "wrong_recalls": int(wrong),
            "promoted_fact_slots": int(n_fact_slots),
            "promoted_total": int(len(positions)),
            "n_test_facts": int(len(te)),
            "register_bytes": reg.nbytes(),
        }
        log(f"      {name:12s} precision={precision:.3f}  fact_recall={recall:.3f}  "
            f"exact|promoted={exact_given:.3f}  (exact={exact} wrong={wrong} of {len(te)})")

    log("[7/8] logit-override demo ...")
    reg_demo = FrozenRegister(16)
    reg_demo.promote(1234, ctx_key([7]), sig=0, anchor=0, doc_id=0)
    lg = np.random.default_rng(0).standard_normal(corp.n_symbols)
    out = logit_override(lg, 1234)
    demo = {"argmax_before": int(lg.argmax()), "argmax_after": int(out.argmax()),
            "override_is_exact": int(out.argmax()) == 1234}

    log("[8/8] cost accounting ...")
    kv_bytes_per_token = 2 * 1024 * 2 * 32          # GQA-8 8B, fp16, 32 layers
    kv_total = kv_bytes_per_token * corp.n_tokens
    out = {
        "config": {
            "n_tokens": int(corp.n_tokens), "d": int(d), "M": int(M), "r": int(r),
            "budget": int(budget), "seed": int(seed), "cheb_m": int(cheb_m),
            "diffuse_total_tau": float(diffuse_total), "chunk": int(chunk),
            "n_facts": int(corp.n_facts), "n_test_facts": int(len(te)),
            "slot_bytes": SLOT_BYTES, "ctx_window": CTX,
        },
        "state_bytes": int(ad.state_bytes()),
        "codebook_bytes": int(ad.codebook_bytes()),
        "gist_cosine_local": gist_local,
        "gist_cosine_global": gist_global,
        "policies": results,
        "kv_upper_bound": {
            "accuracy": 1.0,
            "bytes": int(kv_total),
            "bytes_per_token": kv_bytes_per_token,
        },
        "exact_bytes_per_fact": SLOT_BYTES,
        "kv_bytes_per_fact": kv_bytes_per_token,
        "kv_over_ours": kv_bytes_per_token / SLOT_BYTES,
        "logit_override": demo,
        "runtime_sec": round(time.time() - t0, 2),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=1_000_000)
    ap.add_argument("--M", type=int, default=1024)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=1.0, help="total diffusion time")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "e5_graft.json"))
    args = ap.parse_args()
    if args.quick:
        args.tokens = min(args.tokens, 60_000)

    res = run(n_tokens=args.tokens, M=args.M, r=args.r, budget=args.budget,
              seed=args.seed, diffuse_total=args.tau)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {args.out}")
    print(f"runtime {res['runtime_sec']}s   gist_local={res['gist_cosine_local']:.4f}  "
          f"gist_global={res['gist_cosine_global']:.4f}   "
          f"state={res['state_bytes']}B  codebook={res['codebook_bytes']}B")
    for k, v in res["policies"].items():
        print(f"  {k:9s} precision={v['pin_precision']:.3f}  "
              f"fact_recall={v['fact_recall']:.3f}")


if __name__ == "__main__":
    main()
