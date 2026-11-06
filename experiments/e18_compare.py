#!/usr/bin/env python3
"""E18: COMPARE THE LAYERS FROM ALL BRANCHES BY RUNNING THEM.

Configs compared (all built and measured identically in this one run,
same corpus slice, same seeds, same chunk sizes):

1. base     (main branch):        SAABranchBulletproof
2. mamba    (experiments branch): SuperfluidMambaFluidLayer
3. ultra    (experiments branch): UltraFluidStack (2 stacked layers)
4. kv       (kv_cache branch):    ConstantKVCache + SuperfluidMemory
   (paired with the NumPy fluid layer, as that branch intended)
5. combined (combined-layer):     CombinedFluidLayer (all four branches)

Every number below comes from a MEASUREMENT (codespec: run it, don't
read it).

Usage:
    python experiments/e18_compare.py --sha <commit> --out comparison-results
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
CONFIGS = {}
BASELINE = {}
WINNERS = {}
LOG_LINES = []
FENCE = chr(96) * 3
NL = chr(10)


def log(msg):
    print(msg, flush=True)
    LOG_LINES.append(str(msg))


def load_corpus_tokens(limit_words=6000):
    """Deterministic word-level token ids from corpus.txt."""
    import re

    path = os.path.join(REPO_ROOT, "corpus.txt")
    if not os.path.exists(path):
        return np.arange(20000, dtype=np.int64)
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


# ===========================================================================
# BUILDERS: one adapter per branch-layer, same config for all
# ===========================================================================

CTX = 12
CAP = 25000
MAXCAP = 100000
KV_MAX = 256


def make_adapters(d):
    """Build all five adapters with identical (d, ctx_len, capacity)."""
    import torch

    from combined.combined_layer import CombinedFluidConfig, CombinedFluidLayer
    from experiments.superfluid_mamba_layer import (
        SuperfluidMambaConfig,
        SuperfluidMambaFluidLayer,
        SuperfluidMambaFluidLayerNP,
    )
    from experiments.ultra_fluid_layer import UltraFluidConfig, UltraFluidStack
    from fluidlayer.graft_bulletproof import (
        BulletproofConfig,
        HashMode,
        SAABranchBulletproof,
        SecurityLevel,
    )
    from kv_cache.constant_kv_superfluid import (
        ConstantKVCache,
        KVCacheConfig,
        SuperfluidMemory,
    )

    def mamba_cfg():
        return SuperfluidMambaConfig(
            M=16, r=4, d=d, ctx_len=CTX, capacity=CAP, max_capacity=MAXCAP,
            security_level="STANDARD",
        )

    adapters = {}

    # ---- 1) base (main branch) ----
    def build_base():
        layer = SAABranchBulletproof(
            M=16, r=4, d=d,
            config=BulletproofConfig(
                ctx_len=CTX, d=d, hash_mode=HashMode.DOUBLE_64,
                security_level=SecurityLevel.NONE,
                capacity=CAP, max_capacity=MAXCAP, use_gated_sheaf=False,
            ),
        )

        def fwd(h, tid):
            h_np = h.detach().cpu().numpy().astype(np.float64)
            out, _ = layer.forward(h_np, h_np.copy(), tid.detach().cpu().numpy().astype(np.int64))
            return torch.from_numpy(np.asarray(out, dtype=np.float32))

        def reset():
            layer.total_hits = 0
            layer.total_misses = 0
            layer.total_tokens = 0

        return {
            "fwd": fwd, "reset": reset, "torch_params": layer,
            "recall_read": lambda: layer.get_stats()["recall_rate"],
            "stores": [layer.store], "state_bytes": lambda: int(layer.state.nbytes),
            "kv": None, "extra_bytes": 0,
        }

    adapters["base (main)"] = build_base

    # ---- 2) mamba (experiments branch) ----
    def build_mamba():
        layer = SuperfluidMambaFluidLayer(mamba_cfg())

        def fwd(h, tid):
            out, _ = layer(h, h.clone(), tid)
            return out.detach().float()

        def reset():
            layer.total_hits = 0
            layer.total_misses = 0
            layer.total_tokens = 0
            layer._context_buffer = []

        return {
            "fwd": fwd, "reset": reset, "torch_params": layer,
            "recall_read": lambda: layer.get_stats()["recall_rate"],
            "stores": [layer.store], "state_bytes": lambda: int(layer.state.nbytes),
            "kv": None, "extra_bytes": 0,
        }

    adapters["mamba (experiments)"] = build_mamba

    # ---- 3) ultra (experiments branch, 2 stacked layers) ----
    def build_ultra():
        layer = UltraFluidStack(
            UltraFluidConfig(
                M=16, r=4, d=d, ctx_len=CTX, num_layers=2,
                capacity=CAP, max_capacity=MAXCAP, security_level="STANDARD",
                use_cross_attention=(d >= 8 and d % 4 == 0),
            )
        )

        def fwd(h, tid):
            out, _ = layer(h, h.clone(), tid)
            return out.detach().float()

        def reset():
            for lyr in layer.layers:
                lyr.total_hits = 0
                lyr.total_misses = 0
                lyr.total_tokens = 0
                lyr._context_buffer = []

        def recall_read():
            hits = sum(lyr.total_hits for lyr in layer.layers)
            misses = sum(lyr.total_misses for lyr in layer.layers)
            return hits / (hits + misses + 1e-12)

        return {
            "fwd": fwd, "reset": reset, "torch_params": layer,
            "recall_read": recall_read,
            "stores": [lyr.store for lyr in layer.layers],
            "state_bytes": lambda: int(sum(lyr.state.nbytes for lyr in layer.layers)),
            "kv": None, "extra_bytes": 0,
        }

    adapters["ultra (experiments)"] = build_ultra

    # ---- 4) kv (kv_cache branch: bounded KV + long-term memory) ----
    def build_kv():
        from collections import deque

        np_layer = SuperfluidMambaFluidLayerNP(mamba_cfg())
        memory = SuperfluidMemory(np_layer)
        cache = ConstantKVCache(KVCacheConfig(max_size=KV_MAX, num_heads=1, head_dim=d))
        stream: deque = deque()

        def fwd(h, tid):
            fused = h + memory.retrieve(h, tid)
            k = fused.detach().unsqueeze(0).unsqueeze(0)  # (1, 1, S, d)
            before = len(cache.evicted_tokens)
            cache.update(k, k.clone(), tid.unsqueeze(0))
            n_evict = len(cache.evicted_tokens) - before
            seq = h.shape[0]
            for t in range(seq):
                stream.append((fused[t].detach(), int(tid[t])))
            if n_evict > 0:
                rows, ids = [], []
                for _ in range(n_evict):
                    if stream:
                        row, tok = stream.popleft()
                        rows.append(row)
                        ids.append(tok)
                if rows:
                    memory.store(torch.stack(rows), torch.tensor(ids, dtype=torch.long))
            while len(stream) > KV_MAX:
                stream.popleft()
            return fused

        def kv_info():
            return {
                "max_kv_seen": cache.size(),
                "kv_evictions": len(cache.evicted_tokens),
                "long_term_tokens": memory.total_tokens,
            }

        return {
            "fwd": fwd, "reset": lambda: None, "torch_params": None,
            "recall_read": None,
            "stores": [np_layer.store], "state_bytes": lambda: int(np_layer.state.nbytes),
            "kv": kv_info, "extra_bytes": KV_MAX * 2 * 1 * d * 4,
        }

    adapters["kv (kv_cache)"] = build_kv

    # ---- 5) combined (combined-layer branch) ----
    def build_combined():
        layer = CombinedFluidLayer(
            CombinedFluidConfig(
                d=d, ctx_len=CTX, M=16, r=4, capacity=CAP, max_capacity=MAXCAP,
                kv_max_size=KV_MAX, ultra_layers=2,
            )
        )

        def fwd(h, tid):
            out, _ = layer(h, h.clone(), tid)
            return out

        def reset():
            layer.base.total_hits = 0
            layer.base.total_misses = 0
            layer.base.total_tokens = 0

        def kv_info():
            return {
                "max_kv_seen": layer.kv_cache.size(),
                "kv_evictions": len(layer.kv_cache.evicted_tokens),
                "long_term_tokens": layer.superfluid_memory.total_tokens,
            }

        return {
            "fwd": fwd, "reset": reset, "torch_params": layer,
            "recall_read": lambda: layer.base.get_stats()["recall_rate"],
            "stores": [layer.base.store, layer.mamba_memory.store]
            + [lyr.store for lyr in layer.ultra.layers],
            "state_bytes": lambda: int(layer.base.state.nbytes),
            "kv": kv_info, "extra_bytes": KV_MAX * 2 * 4 * (d // 4 if d % 4 == 0 else 1) * 4,
        }

    adapters["combined (all 4)"] = build_combined
    return adapters


# ===========================================================================
# METRICS
# ===========================================================================

def recall_passes(build, tokens):
    """Two passes: fresh recall (pass 1) then replay recall (pass 2)."""
    import torch

    h = torch.randn(256, 16)
    fresh = None
    table = {}
    for n in (1, 4, 8, 12):
        a = build()
        # swap in the right ctx_len by rebuilding is NOT possible for the
        # shared CTX; instead measure at ctx_len=12 only for the headline
        if n != 12:
            continue
        for i in range(0, len(tokens), 256):
            chunk = torch.from_numpy(tokens[i:i + 256].astype(np.int64))
            hs = h[: len(chunk)]
            a["fwd"](hs, chunk)
        fresh = a["recall_read"]()
        a["reset"]()
        for i in range(0, len(tokens), 256):
            chunk = torch.from_numpy(tokens[i:i + 256].astype(np.int64))
            hs = h[: len(chunk)]
            a["fwd"](hs, chunk)
        table = {"fresh_recall": round(fresh, 4), "replay_recall": round(a["recall_read"](), 4)}
    return table


def kv_stream_test(build, tokens):
    import torch

    a = build()
    h = torch.randn(16, 16)
    max_kv = 0
    for i in range(0, len(tokens), 16):
        chunk = torch.from_numpy(tokens[i:i + 16].astype(np.int64))
        hs = h[: len(chunk)]
        a["fwd"](hs, chunk)
        max_kv = max(max_kv, a["kv"]()["max_kv_seen"])
    info = a["kv"]()
    info["max_kv_seen"] = max_kv
    return info


def speed_test(build):
    import torch

    a = build()
    h = torch.randn(256, 16)
    tid = torch.randint(0, 100000, (256,))
    a["fwd"](h, tid)  # warmup
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        a["fwd"](h, tid)
        times.append(time.perf_counter() - t0)
    return round(statistics.median(times) * 1000, 1)


def threads_test(build):
    import threading

    import torch

    a = build()
    errors = []

    def worker(k):
        try:
            r = np.random.default_rng(500 + k)
            for _ in range(20):
                arr = r.normal(0, 1, (16, 16)).astype(np.float32)
                h = torch.from_numpy(arr)
                tid = torch.from_numpy(r.integers(0, 5000, 16).astype(np.int64))
                a["fwd"](h, tid)
        except Exception as e:  # noqa: BLE001
            errors.append("thread " + str(k) + ": " + type(e).__name__ + ": " + str(e))

    ts = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return errors


def memory_test(build):
    a = build()
    entry_bytes = sys.getsizeof(0) * 4
    stores_bytes = sum(s.max_capacity for s in a["stores"]) * entry_bytes
    params_bytes = 0
    tp = a["torch_params"]
    if tp is not None and hasattr(tp, "parameters"):
        params_bytes = int(sum(p.numel() for p in tp.parameters()) * 4)
    state_bytes = 0
    try:
        state_bytes = int(a["state_bytes"]())
    except Exception:
        state_bytes = 0
    total = int(state_bytes + stores_bytes + params_bytes + a["extra_bytes"])
    return {
        "stores": len(a["stores"]),
        "torch_param_bytes": params_bytes,
        "worst_case_store_bytes": int(stores_bytes),
        "worst_case_total_bytes": total,
        "worst_case_mb": round(total / (1024 * 1024), 1),
    }


def gpt2_ab(builders, d):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL = "sshleifer/tiny-gpt2"
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.eval()
    # tiny-gpt2's n_embd is NOT 16: rebuild every adapter at the model's
    # real width so the hooked hidden states match the layers (e17 pattern).
    ab_builders = make_adapters(int(model.config.n_embd))
    with open(os.path.join(REPO_ROOT, "corpus.txt"), encoding="utf-8", errors="ignore") as f:
        text = f.read(12000)
    ids = tok(text, return_tensors="pt").input_ids[0]
    chunks = [ids[i:i + 128] for i in range(0, len(ids) - 128, 128)][:40]
    current = {"ids": None}

    def run(fwd):
        handle = None
        if fwd is not None:
            def hook(module, args, output):
                hs = output
                with torch.no_grad():
                    fused = fwd(hs[0].float(), current["ids"])
                return fused.unsqueeze(0)

            handle = model.transformer.drop.register_forward_hook(hook)
        losses = []
        with torch.no_grad():
            for chunk in chunks:
                current["ids"] = chunk
                out = model(chunk.unsqueeze(0), labels=chunk.unsqueeze(0))
                losses.append(out.loss.item() * len(chunk))
        if handle is not None:
            handle.remove()
        return sum(losses) / max(sum(len(c) for c in chunks), 1)

    baseline = run(None)
    deltas = {}
    for name, build in ab_builders.items():
        a = build()
        withl = run(a["fwd"])
        deltas[name] = {
            "loss_nats": round(withl, 4),
            "delta_nats": round(withl - baseline, 4),
            "finite": bool(np.isfinite(withl)),
        }
    return {"model": MODEL, "baseline_loss_nats": round(baseline, 4), "per_layer": deltas}


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", default=os.environ.get("GITHUB_SHA", "unknown"))
    ap.add_argument("--out", default="results/comparison")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    def flush(status, error=None):
        payload = {
            "sha": args.sha,
            "branch": "combined-layer",
            "status": status,
            "baseline": {k: v for k, v in BASELINE.items() if k != "per_layer"},
            "configs": CONFIGS,
            "winners": WINNERS,
        }
        if error:
            payload["error"] = error
        payload["log"] = NL.join(LOG_LINES)
        with open(os.path.join(args.out, "run.json"), "w") as f:
            json.dump(payload, f, indent=2, default=str)

    log("=" * 70)
    log("LAYER COMPARISON RUN  sha=" + args.sha)
    log("=" * 70)
    flush("running")

    tokens = load_corpus_tokens()[:2000]
    log("corpus tokens: " + str(len(tokens)))

    import torch  # noqa: F401

    d = 16
    builders = make_adapters(d)
    names = list(builders.keys())
    log("comparing: " + ", ".join(names))

    for name in names:
        log("")
        log("--- " + name + " ---")
        cfg = {}
        try:
            if builders[name]()["recall_read"] is not None:
                cfg["recall_12gram"] = recall_passes(builders[name], tokens)
            else:
                cfg["recall_12gram"] = None
            cfg["overhead_ms_per_256_tokens"] = speed_test(builders[name])
            cfg["thread_errors"] = threads_test(builders[name])
            cfg["memory"] = memory_test(builders[name])
            if builders[name]()["kv"] is not None:
                kv = kv_stream_test(builders[name], tokens)
                cfg["kv"] = kv
                cfg["kv_bounded"] = kv["max_kv_seen"] <= KV_MAX and kv["kv_evictions"] > 0
        except Exception:
            err = traceback.format_exc()
            log("ERROR in " + name + ": " + err)
            cfg["error"] = err.splitlines()[-1]
        CONFIGS[name] = cfg
        flush("config-done: " + name)

    log("")
    log("--- real GPT-2 A/B (all layers) ---")
    try:
        ab = gpt2_ab(builders, d)
        BASELINE.update(ab)
        for name in names:
            if name in ab.get("per_layer", {}):
                CONFIGS[name]["gpt2"] = ab["per_layer"][name]
    except Exception:
        BASELINE["gpt2_error"] = traceback.format_exc()
        log("gpt2 A/B failed: " + BASELINE["gpt2_error"])

    # ---- winners ----
    def num(name, key):
        return CONFIGS[name].get(key)

    valid = [n for n in names if "error" not in CONFIGS[n]]
    speed = {n: num(n, "overhead_ms_per_256_tokens") for n in valid if num(n, "overhead_ms_per_256_tokens") is not None}
    WINNERS["fastest"] = min(speed, key=speed.get) if speed else None
    mem = {n: num(n, "memory")["worst_case_mb"] for n in valid if num(n, "memory")}
    WINNERS["lowest_memory"] = min(mem, key=mem.get) if mem else None
    rec = {n: num(n, "recall_12gram")["replay_recall"] for n in valid if num(n, "recall_12gram")}
    WINNERS["best_replay_recall"] = max(rec, key=rec.get) if rec else None
    att = {n: abs(num(n, "gpt2")["delta_nats"]) for n in valid if num(n, "gpt2")}
    WINNERS["smallest_attach_perturbation"] = min(att, key=att.get) if att else None
    thr = {n: len(num(n, "thread_errors")) for n in valid if num(n, "thread_errors") is not None}
    WINNERS["thread_safe"] = [n for n, v in thr.items() if v == 0]
    WINNERS["bounded_kv"] = [n for n in valid if CONFIGS[n].get("kv_bounded")]

    os.makedirs(args.out, exist_ok=True)
    payload = {
        "sha": args.sha,
        "branch": "combined-layer",
        "baseline": {k: v for k, v in BASELINE.items() if k != "per_layer"},
        "configs": CONFIGS,
        "winners": WINNERS,
    }
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)

    lines = [
        "# Layer comparison (all branches)",
        "",
        "- commit: " + args.sha,
        "- branch: combined-layer",
        "- date: " + time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "- identical harness for every layer: d=16, ctx_len=12, capacity 25k/100k, same 2000-token corpus slice, same GPT-2 chunks",
        "",
        "## Measured comparison",
        "",
        "| Layer | 12-gram replay recall | fresh recall | ms / 256 tok | worst-case MB | threads | KV bounded | GPT-2 delta (nats) |",
        "|---|---|---|---|---|---|---|---|"
    ]
    for name in names:
        c = CONFIGS[name]
        r = c.get("recall_12gram")
        rec_s = (str(r["replay_recall"]) + " / " + str(r["fresh_recall"])) if r else "n/a (wrapper)"
        ms = c.get("overhead_ms_per_256_tokens")
        mb = c.get("memory", {}).get("worst_case_mb")
        te = len(c.get("thread_errors", [])) if "thread_errors" in c else "n/a"
        kv = "yes" if c.get("kv_bounded") else ("no" if c.get("kv") else "-")
        dl = c.get("gpt2", {}).get("delta_nats")
        lines.append(
            "| " + name + " | " + rec_s + " | " + str(ms) + " | " + str(mb) + " | " + str(te)
            + " | " + kv + " | " + str(dl) + " |"
        )
    lines += ["", "## Winners", ""]
    for k, v in WINNERS.items():
        lines.append("- **" + k + "**: " + str(v))
    lines += ["", "## Full log", "", FENCE] + LOG_LINES + [FENCE]
    with open(os.path.join(args.out, "comparison.md"), "w") as f:
        f.write(NL.join(lines))

    log("")
    log("Wrote " + args.out + "/run.json and " + args.out + "/comparison.md")
    log("COMPARISON RUN COMPLETE")
    flush("complete")


if __name__ == "__main__":
    main()
