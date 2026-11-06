#!/usr/bin/env python3
"""E19: BASE + KV WRAPPER -- the lean middle ground.

Attaches the kv_cache branch's wrapper (ConstantKVCache bounded KV +
SuperfluidMemory long-term memory, evictions stored with their REAL
hidden states) DIRECTLY to the main branch's base layer
(SAABranchBulletproof) -- WITHOUT the experiments branch's mamba/ultra
stack that makes the combined layer ~9x slower (see e18).

Question this run answers by MEASURING it: does base+kv deliver the
combined layer's bounded-KV + long-term features at close-to-base
speed and memory?

Same harness and constants as e18 (d=16, ctx_len=12, capacity
25k/100k, KV_MAX=256, same 2000-token corpus slice, same GPT-2
chunks); it reuses e18's test functions verbatim, so the numbers are
directly comparable with the e18 comparison table.

Usage:
    python experiments/e19_base_kv.py --sha <commit> --out basekv-results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from experiments.e18_compare import (
    CAP,
    CTX,
    KV_MAX,
    MAXCAP,
    kv_stream_test,
    load_corpus_tokens,
    memory_test,
    recall_passes,
    speed_test,
    threads_test,
)
from experiments.superfluid_mamba_layer import (
    SuperfluidMambaConfig,
    SuperfluidMambaFluidLayerNP,
)
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = {}
BASELINE = {}
WINNERS = {}
CLAIMS = {}
LOG_LINES = []
FENCE = chr(96) * 3
NL = chr(10)


def log(msg):
    print(msg, flush=True)
    LOG_LINES.append(str(msg))


# ===========================================================================
# THE E19 LAYER: base (main branch) + kv wrapper (kv_cache branch)
# ===========================================================================


class BaseKVLayer:
    """Base layer (main branch) with the kv_cache branch's wrapper attached.

    forward = base forward -> bounded KV update; every eviction is stored
    to the long-term SuperfluidMemory with its REAL hidden state (the
    same stream pattern the combined layer uses).  SuperfluidMemory is
    paired with the NumPy fluid layer -- the pairing the kv_cache branch
    intended and e18 measured.
    """

    def __init__(self, d: int):
        self.d = d
        self.base = SAABranchBulletproof(
            M=16,
            r=4,
            d=d,
            config=BulletproofConfig(
                ctx_len=CTX,
                d=d,
                hash_mode=HashMode.DOUBLE_64,
                security_level=SecurityLevel.NONE,
                capacity=CAP,
                max_capacity=MAXCAP,
                use_gated_sheaf=False,
            ),
        )
        self.np_layer = SuperfluidMambaFluidLayerNP(
            SuperfluidMambaConfig(
                M=16, r=4, d=d, ctx_len=CTX, capacity=CAP,
                max_capacity=MAXCAP, security_level="STANDARD",
            )
        )
        self.memory = SuperfluidMemory(self.np_layer)
        self.kv_cache = ConstantKVCache(
            KVCacheConfig(max_size=KV_MAX, num_heads=1, head_dim=d)
        )
        self._stream: deque = deque()

    def forward(self, h, tid):
        h_np = h.detach().cpu().numpy().astype(np.float64)
        tid_np = tid.detach().cpu().numpy().astype(np.int64)
        out, _ = self.base.forward(h_np, h_np.copy(), tid_np)
        fused = torch.from_numpy(np.asarray(out, dtype=np.float32))

        # bounded KV update; evictions -> long-term memory, REAL states
        k = fused.detach().unsqueeze(0).unsqueeze(0)  # (1, 1, S, d)
        before = len(self.kv_cache.evicted_tokens)
        self.kv_cache.update(k, k.clone(), tid.unsqueeze(0))
        n_evict = len(self.kv_cache.evicted_tokens) - before
        for t in range(h.shape[0]):
            self._stream.append((fused[t].detach(), int(tid_np[t])))
        if n_evict > 0:
            rows, ids = [], []
            for _ in range(n_evict):
                if self._stream:
                    row, tok = self._stream.popleft()
                    rows.append(row)
                    ids.append(tok)
            if rows:
                self.memory.store(
                    torch.stack(rows), torch.tensor(ids, dtype=torch.long)
                )
        while len(self._stream) > KV_MAX:
            self._stream.popleft()
        return fused

    def reset(self):
        self.base.total_hits = 0
        self.base.total_misses = 0
        self.base.total_tokens = 0


# ===========================================================================
# ADAPTERS (same interface as e18's, so e18's test functions work as-is)
# ===========================================================================


def make_adapters_e19(d):
    """Two adapters: the base layer alone, and base+kv (this experiment)."""

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
            out, _ = layer.forward(
                h_np, h_np.copy(), tid.detach().cpu().numpy().astype(np.int64)
            )
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

    def build_base_kv():
        layer = BaseKVLayer(d)

        def kv_info():
            return {
                "max_kv_seen": layer.kv_cache.size(),
                "kv_evictions": len(layer.kv_cache.evicted_tokens),
                "long_term_tokens": layer.memory.total_tokens,
            }

        return {
            "fwd": layer.forward, "reset": layer.reset, "torch_params": None,
            "recall_read": lambda: layer.base.get_stats()["recall_rate"],
            "stores": [layer.base.store, layer.np_layer.store],
            "state_bytes": lambda: int(layer.base.state.nbytes)
            + int(layer.np_layer.state.nbytes),
            "kv": kv_info, "extra_bytes": KV_MAX * 2 * 1 * d * 4,
        }

    return {"base (main)": build_base, "base+kv (e19)": build_base_kv}


# ===========================================================================
# GPT-2 A/B (e17/e18 pattern: adapters rebuilt at the model's REAL n_embd)
# ===========================================================================


def gpt2_ab_e19():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL = "sshleifer/tiny-gpt2"
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.eval()
    # tiny-gpt2's n_embd is NOT 16: build at the model's real width
    ab_builders = make_adapters_e19(int(model.config.n_embd))
    with open(os.path.join(REPO_ROOT, "corpus.txt"), encoding="utf-8",
              errors="ignore") as f:
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
    return {
        "model": MODEL,
        "baseline_loss_nats": round(baseline, 4),
        "per_layer": deltas,
    }


# ===========================================================================
# MAIN
# ===========================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", default=os.environ.get("GITHUB_SHA", "unknown"))
    ap.add_argument("--out", default="basekv-results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    def flush(status, error=None):
        payload = {
            "sha": args.sha,
            "branch": os.environ.get("GITHUB_REF_NAME", "local"),
            "status": status,
            "baseline": {k: v for k, v in BASELINE.items() if k != "per_layer"},
            "configs": CONFIGS,
            "winners": WINNERS,
            "claims": CLAIMS,
        }
        if error:
            payload["error"] = error
        payload["log"] = NL.join(LOG_LINES)
        with open(os.path.join(args.out, "run.json"), "w") as f:
            json.dump(payload, f, indent=2, default=str)

    log("=" * 70)
    log("BASE + KV WRAPPER RUN (e19)  sha=" + args.sha)
    log("=" * 70)
    flush("running")

    tokens = load_corpus_tokens()[:2000]
    log("corpus tokens: " + str(len(tokens)))

    d = 16
    builders = make_adapters_e19(d)
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
    log("--- real GPT-2 A/B (base vs base+kv) ---")
    try:
        ab = gpt2_ab_e19()
        BASELINE.update(ab)
        for name in names:
            if name in ab.get("per_layer", {}):
                CONFIGS[name]["gpt2"] = ab["per_layer"][name]
    except Exception:
        BASELINE["gpt2_error"] = traceback.format_exc()
        log("gpt2 A/B failed: " + BASELINE["gpt2_error"])

    # ---- winners (same keys as e18 for comparability) ----
    def num(name, key):
        return CONFIGS[name].get(key)

    valid = [n for n in names if "error" not in CONFIGS[n]]
    speed = {
        n: num(n, "overhead_ms_per_256_tokens")
        for n in valid if num(n, "overhead_ms_per_256_tokens") is not None
    }
    WINNERS["fastest"] = min(speed, key=speed.get) if speed else None
    mem = {n: num(n, "memory")["worst_case_mb"] for n in valid if num(n, "memory")}
    WINNERS["lowest_memory"] = min(mem, key=mem.get) if mem else None
    rec = {
        n: num(n, "recall_12gram")["replay_recall"]
        for n in valid if num(n, "recall_12gram")
    }
    WINNERS["best_replay_recall"] = max(rec, key=rec.get) if rec else None
    att = {
        n: abs(num(n, "gpt2")["delta_nats"]) for n in valid if num(n, "gpt2")
    }
    WINNERS["smallest_attach_perturbation"] = min(att, key=att.get) if att else None
    thr = {
        n: len(num(n, "thread_errors"))
        for n in valid if num(n, "thread_errors") is not None
    }
    WINNERS["thread_safe"] = [n for n, v in thr.items() if v == 0]
    WINNERS["bounded_kv"] = [n for n in valid if CONFIGS[n].get("kv_bounded")]

    # ---- e19 claims (all measured) ----
    try:
        b, k = "base (main)", "base+kv (e19)"
        if "error" not in CONFIGS[b] and "error" not in CONFIGS[k]:
            CLAIMS["recall_parity_with_base"] = bool(
                CONFIGS[k]["recall_12gram"]["replay_recall"]
                >= CONFIGS[b]["recall_12gram"]["replay_recall"] - 0.001
            )
            CLAIMS["kv_stays_bounded"] = CONFIGS[k].get("kv_bounded") is True
            CLAIMS["evictions_go_to_long_term"] = bool(
                (CONFIGS[k].get("kv") or {}).get("long_term_tokens", 0) > 0
            )
            CLAIMS["memory_under_100mb_ceiling"] = bool(
                (CONFIGS[k].get("memory") or {}).get("worst_case_mb", 1e9) < 100
            )
            CLAIMS["thread_safe"] = len(CONFIGS[k].get("thread_errors", [])) == 0
            gk = CONFIGS[k].get("gpt2") or {}
            CLAIMS["gpt2_attach_finite"] = bool(gk.get("finite"))
            CLAIMS["gpt2_delta_small"] = bool(abs(gk.get("delta_nats", 99)) <= 0.05)
            if CONFIGS[b].get("overhead_ms_per_256_tokens"):
                CLAIMS["speed_overhead_vs_base"] = round(
                    CONFIGS[k]["overhead_ms_per_256_tokens"]
                    / CONFIGS[b]["overhead_ms_per_256_tokens"], 2
                )
    except Exception:
        CLAIMS["error"] = traceback.format_exc().splitlines()[-1]

    payload = {
        "sha": args.sha,
        "branch": os.environ.get("GITHUB_REF_NAME", "local"),
        "baseline": {k: v for k, v in BASELINE.items() if k != "per_layer"},
        "configs": CONFIGS,
        "winners": WINNERS,
        "claims": CLAIMS,
    }
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)

    lines = [
        "# Base + KV wrapper (e19)",
        "",
        "- commit: " + args.sha,
        "- branch: combined-layer",
        "- date: " + time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "- same harness as e18: d=16, ctx_len=12, capacity 25k/100k, KV_MAX=256",
        "- question: does base+kv give the combined layer's bounded-KV",
        "  + long-term features at close-to-base speed and memory?",
        "",
        "## Measured (base vs base+kv)",
        "",
        "| Layer | 12-gram replay recall | fresh recall | ms / 256 tok | worst-case MB | threads | KV bounded | GPT-2 delta (nats) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        c = CONFIGS[name]
        r = c.get("recall_12gram")
        rec_s = (
            str(r["replay_recall"]) + " / " + str(r["fresh_recall"]) if r else "n/a"
        )
        ms = c.get("overhead_ms_per_256_tokens")
        mb = (c.get("memory") or {}).get("worst_case_mb")
        te = len(c.get("thread_errors", [])) if "thread_errors" in c else "n/a"
        kv = "yes" if c.get("kv_bounded") else ("no" if c.get("kv") else "-")
        dl = (c.get("gpt2") or {}).get("delta_nats")
        lines.append(
            "| " + name + " | " + rec_s + " | " + str(ms) + " | " + str(mb)
            + " | " + str(te) + " | " + kv + " | " + str(dl) + " |"
        )
    lines += ["", "## Winners", ""]
    for k2, v in WINNERS.items():
        lines.append("- **" + k2 + "**: " + str(v))
    lines += ["", "## Claims (measured in THIS run)", ""]
    for k2, v in CLAIMS.items():
        lines.append("- **" + k2 + "**: " + str(v))
    lines += [
        "",
        "Reference (e18, same harness): combined (all 4 branches) measured",
        "26.4 ms/256tok and 42.9 MB worst-case with the same bounded-KV",
        "behaviour; base was 2.9 ms and 10.7 MB.",
        "",
        "## Full log",
        "",
        FENCE,
    ] + LOG_LINES + [FENCE]
    with open(os.path.join(args.out, "basekv.md"), "w") as f:
        f.write(NL.join(lines))

    log("")
    log("Wrote " + args.out + "/run.json and " + args.out + "/basekv.md")
    log("BASE+KV RUN COMPLETE")
    flush("complete")


if __name__ == "__main__":
    main()
