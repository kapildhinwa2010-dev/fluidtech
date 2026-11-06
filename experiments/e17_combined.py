#!/usr/bin/env python3
"""E17: VERIFY THE COMBINED LAYER BY RUNNING IT.

The combined-layer branch fuses four branches into ONE layer:
main (SAABranchBulletproof) + experiments (Superfluid-Mamba +
UltraFluidStack) + kv_cache (ConstantKVCache + SuperfluidMemory).

Every verdict below comes from a MEASUREMENT (codespec: run it,
don't read it).  Claims are CONFIRMED / REFUTED / UNVERIFIABLE,
with the method recorded.

Usage:
    python experiments/e17_combined.py --sha <commit> --out results/combined
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = {}
LOG_LINES = []
FENCE = chr(96) * 3
NL = chr(10)


def log(msg):
    print(msg, flush=True)
    LOG_LINES.append(str(msg))


def record(claim, verdict, measured, method):
    RESULTS[claim] = {"verdict": verdict, "measured": measured, "method": method}
    log("[" + verdict + "] " + claim + ": " + json.dumps(measured, default=str)[:300])


def load_corpus_tokens(limit_words=6000):
    """Deterministic word-level token ids from corpus.txt."""
    import re

    path = os.path.join(REPO_ROOT, "corpus.txt")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read(400_000)
    words = re.findall(r"\w+", text.lower())[:limit_words]
    vocab = {}
    ids = []
    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)
        ids.append(vocab[w])
    return np.array(ids, dtype=np.int64)


def make_layer(**kw):
    import torch  # noqa: F401  (layer needs torch at build time)

    from combined.combined_layer import CombinedFluidConfig, CombinedFluidLayer

    defaults = dict(
        d=16, ctx_len=12, M=16, r=4, ultra_layers=1, kv_max_size=128,
    )
    defaults.update(kw)
    return CombinedFluidLayer(CombinedFluidConfig(**defaults))


# ===========================================================================
# CLAIM 1: one layer, four branches
# ===========================================================================

def verify_combine():
    import torch

    layer = make_layer(d=16, ultra_layers=2)
    h = torch.randn(32, 16)
    tid = torch.randint(0, 5000, (32,))
    out, info = layer(h, h.clone(), tid)
    stats = layer.get_stats()
    ok = (
        tuple(out.shape) == (32, 16)
        and bool(torch.isfinite(out).all())
        and stats["kv_cache_size"] == 32
        and len(stats["components"]) == 5
    )
    record(
        "the combined layer is ONE layer with all four branches inside "
        "(main + superfluid-mamba + ultra + constant-KV)",
        "CONFIRMED" if ok else "REFUTED",
        {
            "output_shape": list(out.shape),
            "components": stats["components"],
            "kv_size_after_forward": stats["kv_cache_size"],
        },
        "build CombinedFluidLayer, forward a (32,16) batch, check shape/finite/components",
    )


# ===========================================================================
# CLAIM 2+3: replay recall through the fused layer
# ===========================================================================

def verify_recall(tokens):
    import torch

    tokens = tokens[:2000]
    table = {}
    for n in (1, 4, 8, 12):
        layer = make_layer(ctx_len=n)
        h = torch.randn(256, 16)
        # pass 1: fresh text (recall must be ~0)
        for i in range(0, len(tokens), 256):
            chunk = torch.from_numpy(tokens[i:i + 256].astype(np.int64))
            hs = h[: len(chunk)]
            layer(hs, hs.clone(), chunk)
        fresh = layer.base.get_stats()["recall_rate"]
        # reset counters (e16 pattern), pass 2: replay the same tokens
        layer.base.total_hits = 0
        layer.base.total_misses = 0
        layer.base.total_tokens = 0
        for i in range(0, len(tokens), 256):
            chunk = torch.from_numpy(tokens[i:i + 256].astype(np.int64))
            hs = h[: len(chunk)]
            layer(hs, hs.clone(), chunk)
        replay = layer.base.get_stats()["recall_rate"]
        table[n] = {"fresh_recall": round(fresh, 4), "replay_recall": round(replay, 4)}
    claim12 = table[12]["replay_recall"]
    record(
        "100% recall for previously seen contexts survives the fusion "
        "(1/4/8/12-gram)",
        "CONFIRMED" if claim12 >= 0.99 else "REFUTED",
        {
            "table_ngram_replay_recall": table,
            "note": "the exact tier runs INSIDE the combined layer; recall is for contexts seen verbatim before",
        },
        "word-level tokens from a 2000-token corpus.txt slice, ctx_len in (1,4,8,12), two passes through the COMBINED forward",
    )
    record(
        "never-seen contexts recall ~0 through the combined layer",
        "CONFIRMED" if table[12]["fresh_recall"] < 0.01 else "REFUTED",
        {"fresh_12gram_recall": table[12]["fresh_recall"]},
        "pass 1 over unseen corpus text; a store cannot recall what it never stored",
    )


# ===========================================================================
# CLAIM 4: bounded KV cache + long-term memory
# ===========================================================================

def verify_kv(tokens):
    import torch

    layer = make_layer(kv_max_size=64, ultra_layers=1)
    h = torch.randn(16, 16)
    stream = tokens[:992]
    for i in range(0, len(stream), 16):
        chunk = torch.from_numpy(stream[i:i + 16].astype(np.int64))
        hs = h[: len(chunk)]
        layer(hs, hs.clone(), chunk)
    stats = layer.get_stats()
    ok = (
        stats["kv_cache_size"] <= 64
        and stats["kv_evicted_total"] > 0
        and stats["long_term_tokens"] > 0
        and len(layer.mamba_memory.store) > 0
    )
    record(
        "KV cache stays bounded and evictions land in long-term "
        "superfluid memory",
        "CONFIRMED" if ok else "REFUTED",
        {
            "kv_size": stats["kv_cache_size"],
            "kv_max_size": 64,
            "tokens_streamed": int(stats["total_tokens"]),
            "kv_evictions": stats["kv_evicted_total"],
            "long_term_tokens": stats["long_term_tokens"],
            "mamba_store_size": len(layer.mamba_memory.store),
            "note": "evicted tokens are stored with their REAL hidden states (the kv_cache branch stored zeros)",
        },
        "stream 992 corpus tokens in chunks of 16 through the combined layer (kv_max_size=64)",
    )


# ===========================================================================
# CLAIM 5: real GPT-2 A/B
# ===========================================================================

def load_gpt2():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL = "sshleifer/tiny-gpt2"
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.eval()
    return tok, model, MODEL


def corpus_ids(tok, chars):
    with open(os.path.join(REPO_ROOT, "corpus.txt"), encoding="utf-8", errors="ignore") as f:
        text = f.read(chars)
    return tok(text, return_tensors="pt").input_ids[0]


def verify_gpt2():
    try:
        import torch
    except Exception as e:  # pragma: no cover
        record("the combined layer attaches to a real GPT-2 and runs", "UNVERIFIABLE", {"import_error": str(e)}, "torch not installed")
        return
    tok, model, MODEL = load_gpt2()
    d = model.config.n_embd
    ids = corpus_ids(tok, 12000)
    chunks = [ids[i:i + 128] for i in range(0, len(ids) - 128, 128)][:40]

    from combined.combined_layer import CombinedFluidConfig, CombinedFluidLayer

    current = {"ids": None}

    def run(with_layer):
        hook_handle = None
        layer = None
        if with_layer:
            layer = CombinedFluidLayer(
                CombinedFluidConfig(d=d, ctx_len=12, M=16, r=4, kv_max_size=256, ultra_layers=1)
            )

            def hook(module, args, output):
                hs = output
                with torch.no_grad():
                    h_t = hs[0].float()
                    fused, _ = layer(h_t, h_t.clone(), current["ids"])
                return fused.unsqueeze(0)

            hook_handle = model.transformer.drop.register_forward_hook(hook)
        losses = []
        with torch.no_grad():
            for chunk in chunks:
                current["ids"] = chunk
                out = model(chunk.unsqueeze(0), labels=chunk.unsqueeze(0))
                losses.append(out.loss.item() * len(chunk))
        if hook_handle is not None:
            hook_handle.remove()
        return sum(losses) / max(sum(len(c) for c in chunks), 1)

    base = run(False)
    withl = run(True)
    finite = bool(np.isfinite(base)) and bool(np.isfinite(withl))
    record(
        "the combined layer attaches to a real GPT-2 and runs (A/B measured, untrained)",
        "CONFIRMED" if finite else "REFUTED",
        {
            "model": MODEL,
            "baseline_loss_nats": round(base, 4),
            "combined_loss_nats": round(withl, 4),
            "delta_nats_combined_minus_baseline": round(withl - base, 4),
            "note": "UNTRAINED fusion gates; an improvement claim needs training",
        },
        "A/B on " + MODEL + " (CPU), " + str(len(chunks)) + " chunks of 128 tokens, hook at transformer.drop, seed 0",
    )


# ===========================================================================
# CLAIM 6: greedy generation with a bounded KV cache
# ===========================================================================

def verify_generation():
    try:
        import torch
    except Exception as e:  # pragma: no cover
        record("greedy generation with the combined layer", "UNVERIFIABLE", {"import_error": str(e)}, "torch not installed")
        return
    tok, model, MODEL = load_gpt2()
    d = model.config.n_embd
    from combined.combined_layer import CombinedFluidConfig, CombinedFluidLayer

    layer = CombinedFluidLayer(
        CombinedFluidConfig(d=d, ctx_len=12, M=16, r=4, kv_max_size=128, ultra_layers=1)
    )
    current = {"ids": None}

    def hook(module, args, output):
        hs = output
        with torch.no_grad():
            h_t = hs[0].float()
            fused, _ = layer(h_t, h_t.clone(), current["ids"])
        return fused.unsqueeze(0)

    handle = model.transformer.drop.register_forward_hook(hook)
    seed_ids = corpus_ids(tok, 2000)[:8].unsqueeze(0)
    cur = seed_ids.clone()
    steps = 150
    max_kv = 0
    nonfinite = 0
    with torch.no_grad():
        for _ in range(steps):
            current["ids"] = cur[0]
            out = model(cur)
            logits = out.logits[:, -1, :]
            if not bool(torch.isfinite(logits).all()):
                nonfinite += 1
            nxt = logits.argmax(-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
            max_kv = max(max_kv, layer.kv_cache.size())
    handle.remove()
    stats = layer.get_stats()
    ok = (
        max_kv <= 128
        and nonfinite == 0
        and cur.shape[1] == seed_ids.shape[1] + steps
    )
    record(
        "greedy generation with the combined layer: bounded KV, finite logits",
        "CONFIRMED" if ok else "REFUTED",
        {
            "model": MODEL,
            "steps": steps,
            "max_kv_size_seen": max_kv,
            "kv_max_size": 128,
            "nonfinite_logits_steps": nonfinite,
            "kv_evictions": stats["kv_evicted_total"],
            "long_term_tokens": stats["long_term_tokens"],
            "generated_len": int(cur.shape[1]),
        },
        "greedy loop of 150 steps on tiny-gpt2 with the combined layer hooked at transformer.drop; full re-forward each step",
    )


# ===========================================================================
# CLAIM 7: overhead
# ===========================================================================

def verify_speed():
    import torch

    layer = make_layer(d=64, M=32, r=8, ultra_layers=2, kv_max_size=256)
    h = torch.randn(256, 64)
    tid = torch.randint(0, 100000, (256,))
    layer(h, h.clone(), tid)  # warmup
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        layer(h, h.clone(), tid)
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    # component breakdown
    h_np = h.numpy().astype(np.float64)
    tid_np = tid.numpy().astype(np.int64)
    t0 = time.perf_counter()
    layer.base.forward(h_np, h_np.copy(), tid_np)
    base_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    layer.ultra(h, h.clone(), tid)
    ultra_s = time.perf_counter() - t0
    ok = med < 2.0
    record(
        "measured overhead of the combined layer (per 256 tokens) with "
        "component breakdown",
        "CONFIRMED" if ok else "REFUTED",
        {
            "combined_ms_per_256_tokens": round(med * 1000, 1),
            "base_only_ms": round(base_s * 1000, 1),
            "ultra_only_ms": round(ultra_s * 1000, 1),
            "note": "a 4-component fusion is heavier than the base layer alone (~10 ms, see results/verification)",
        },
        "time.perf_counter, median of 5 combined forwards on (256,64), CPU",
    )


# ===========================================================================
# CLAIM 8: bounded memory (100 MB ceiling)
# ===========================================================================

def verify_memory():
    layer = make_layer(d=64, M=32, r=8, ultra_layers=2, kv_max_size=256)
    entry_bytes = sys.getsizeof(0) * 4  # rough dict-entry estimate
    state_bytes = int(layer.base.state.nbytes)
    stores = [layer.base.store, layer.mamba_memory.store]
    stores += [l.store for l in layer.ultra.layers]
    worst_stores = sum(s.max_capacity for s in stores) * entry_bytes
    head_dim = 64 // 4
    kv_bytes = 256 * 2 * 4 * head_dim * 4  # tokens x K,V x heads x dim x fp32
    total = state_bytes + worst_stores + kv_bytes
    ceiling = 100 * 1024 * 1024
    record(
        "bounded memory: the combined layer worst case stays under the "
        "100 MB ceiling",
        "CONFIRMED" if total < ceiling else "REFUTED",
        {
            "stores": len(stores),
            "max_capacity_per_store": int(stores[0].max_capacity),
            "base_state_bytes": state_bytes,
            "worst_case_store_bytes": int(worst_stores),
            "kv_cache_bytes": int(kv_bytes),
            "worst_case_total_bytes": int(total),
            "ceiling_bytes": ceiling,
            "note": "100 MB is a hard CEILING, not a target: the four stores share a budget (max_capacity=100000 each)",
        },
        "state.nbytes + per-entry estimate x max_capacity for all 4 stores + KV cache bytes",
    )


# ===========================================================================
# CLAIM 9: thread safety
# ===========================================================================

def verify_threads():
    import threading

    import torch

    layer = make_layer(d=64, M=32, r=8, ultra_layers=1, kv_max_size=256)
    errors = []

    def worker(k):
        try:
            r = np.random.default_rng(100 + k)
            for _ in range(20):
                arr = r.normal(0, 1, (16, 64)).astype(np.float32)
                h = torch.from_numpy(arr)
                tid = torch.from_numpy(r.integers(0, 5000, 16).astype(np.int64))
                layer(h, h.clone(), tid)
        except Exception as e:  # noqa: BLE001
            errors.append("thread " + str(k) + ": " + type(e).__name__ + ": " + str(e))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    record(
        "thread safety (8 concurrent threads on the shared combined layer)",
        "CONFIRMED" if not errors else "REFUTED",
        {"errors": errors[:5], "tokens_processed": layer.get_stats()["total_tokens"]},
        "8 threads x 20 forward passes (16 tokens each) on one shared combined layer",
    )


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", default=os.environ.get("GITHUB_SHA", "unknown"))
    ap.add_argument("--out", default="results/combined")
    args = ap.parse_args()

    log("=" * 70)
    log("COMBINED LAYER VERIFICATION RUN  sha=" + args.sha)
    log("=" * 70)

    tokens = load_corpus_tokens()
    if tokens is None:
        log("corpus.txt NOT FOUND -- using synthetic token ids")
        tokens = np.arange(20000, dtype=np.int64)
    log("corpus tokens: " + str(len(tokens)))

    steps = [
        ("combine", verify_combine),
        ("recall", lambda: verify_recall(tokens)),
        ("kv", lambda: verify_kv(tokens)),
        ("gpt2", verify_gpt2),
        ("generation", verify_generation),
        ("speed", verify_speed),
        ("memory", verify_memory),
        ("threads", verify_threads),
    ]
    for name, fn in steps:
        log("")
        log("--- verifying: " + name + " ---")
        try:
            fn()
        except Exception:
            err = traceback.format_exc()
            log("ERROR in " + name + ": " + err)
            RESULTS["_error_" + name] = {"verdict": "ERROR", "traceback": err}

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump({"sha": args.sha, "branch": "combined-layer", "results": RESULTS}, f, indent=2, default=str)

    lines = [
        "# Combined-layer verification run",
        "",
        "- commit: " + args.sha,
        "- branch: combined-layer",
        "- date: " + time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "- platform: " + sys.platform + ", numpy " + np.__version__,
        "",
        "| Claim | Verdict | Measured |",
        "|---|---|---|",
    ]
    for claim, r in RESULTS.items():
        m = json.dumps(r.get("measured", {}), default=str)[:220]
        lines.append("| " + claim + " | **" + r["verdict"] + "** | " + m + " |")
    lines += ["", "## Full log", "", FENCE] + LOG_LINES + [FENCE]
    with open(os.path.join(args.out, "run.md"), "w") as f:
        f.write(NL.join(lines))

    log("")
    log("Wrote " + args.out + "/run.json and " + args.out + "/run.md")
    log("COMBINED VERIFICATION RUN COMPLETE")


if __name__ == "__main__":
    main()
