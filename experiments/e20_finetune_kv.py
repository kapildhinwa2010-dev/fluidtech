#!/usr/bin/env python3
"""E20: SHORT-BUDGET FINE-TUNE + KV WRAPPER (the new idea).

Idea (Kapil): a model par base+KV wrapper layer lagao, phir us
model ko SIRF 5-10 minute ke liye fine-tune karo -- kya hota hai?
Ye run usko MEASURE karta hai, assume nahi.

Design (three phases, all measured):
  Phase A (before): plain model eval loss + wrapper-attached eval
    loss (attach perturbation, e19 ka GPT-2 A/B pattern).
  Phase B (train): LoRA fine-tune via peft (fallback: full-param
    fine-tune if the model is tiny and peft is absent).  The
    wrapper is NOT attached during training: bounded-KV
    evictions would corrupt the training context, so KV is OFF
    during training BY DESIGN.
  Phase C (after): eval loss again, wrapper attach A/B again,
    bounded-KV check (max size + evictions) and the wrapper
    layer's own recall rate on the eval chunks.

Self-contained for the experiments branch: the bounded KV cache
is inline here (the kv_cache/ package lives on main); the base
layer import is the same SAABranchBulletproof this branch uses.

Data: wikitext-2 via datasets, with a corpus.txt fallback.
Model: tiny-gpt2 default (CI CPU friendly); --model Qwen/Qwen2.5-
1.5B --minutes 10 --lora-r 16 for the Colab T4 version.

Usage:
    python experiments/e20_finetune_kv.py --sha <c> --out e20-out
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from fluidlayer.graft_bulletproof import (
    BulletproofConfig,
    HashMode,
    SAABranchBulletproof,
    SecurityLevel,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTX = 12
CAP = 25000
MAXCAP = 100000
KV_MAX = 256
CHUNK = 128
CONFIGS = {}
CLAIMS = {}
LOG_LINES = []
FENCE = chr(96) * 3
NL = chr(10)


def log(msg):
    print(msg, flush=True)
    LOG_LINES.append(str(msg))


# ===========================================================================
# INLINE BOUNDED KV (self-contained: kv_cache/ lives on main)
# ===========================================================================


class MiniKVCache:
    """Bounded FIFO KV cache: max_size ke baad eviction count
    hota hai (same measured semantics as e19: side buffer, does
    not change the forward output)."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buf: deque = deque()
        self.evictions = 0

    def update(self, h: torch.Tensor):
        for row in h.detach():
            if len(self.buf) >= self.max_size:
                self.buf.popleft()
                self.evictions += 1
            self.buf.append(row)

    def size(self):
        return len(self.buf)


class FinetuneKVWrapper:
    """Base layer (SAABranchBulletproof) + bounded KV side cache.

    kv_enabled=False during training (KV OFF by design);
    kv_enabled=True during the post-fine-tune attach A/B.
    """

    def __init__(self, d: int):
        self.base = SAABranchBulletproof(
            M=16, r=4, d=d,
            config=BulletproofConfig(
                ctx_len=CTX, d=d,
                hash_mode=HashMode.DOUBLE_64,
                security_level=SecurityLevel.NONE,
                capacity=CAP, max_capacity=MAXCAP,
                use_gated_sheaf=False,
            ),
        )
        self.kv = MiniKVCache(KV_MAX)
        self.kv_enabled = False

    def __call__(self, h, tid):
        return self.forward(h, tid)
    def forward(self, h, tid):
        h_np = h.detach().cpu().numpy().astype(np.float64)
        tid_np = tid.detach().cpu().numpy().astype(np.int64)
        out, _ = self.base.forward(h_np, h_np.copy(), tid_np)
        fused = torch.from_numpy(np.asarray(out, dtype=np.float32))
        if self.kv_enabled:
            self.kv.update(fused)
        return fused

    def reset(self):
        self.base.total_hits = 0
        self.base.total_misses = 0
        self.base.total_tokens = 0


# ===========================================================================
# DATA (wikitext-2 preferred, corpus.txt fallback)
# ===========================================================================


def load_ids(tokenizer):
    try:
        from datasets import load_dataset
        ds = None
        for repo, cfg_name in (
            ("wikitext", "wikitext-2-raw-v1"),
            ("Salesforce/wikitext_wikistext2", "wikitext-2-raw-v1"),
            ("wikitext", None),
        ):
            try:
                if cfg_name is None:
                    ds = load_dataset(repo, split="train")
                else:
                    ds = load_dataset(repo, cfg_name, split="train")
                source = "wikitext-2-raw-v1"
                break
            except Exception:
                continue
        if ds is None:
            raise RuntimeError("no wikitext variant loaded")
        text = " ".join(x["text"] for x in list(ds)[:3000])
    except Exception as exc:
        log("datasets/wikitext unavailable: " + str(exc)[:120])
        with open(os.path.join(REPO_ROOT, "corpus.txt"),
                  encoding="utf-8", errors="ignore") as f:
            text = f.read(60000)
        source = "corpus.txt fallback (60k chars)"
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    return ids, source


def make_chunks(ids, n_eval=12):
    chunks = [ids[i:i + CHUNK]
              for i in range(0, len(ids) - CHUNK, CHUNK)]
    n_train = max(len(chunks) - n_eval, 1)
    return chunks[:n_train], chunks[n_train:n_train + n_eval]


# ===========================================================================
# EVAL + ATTACH (e19 GPT-2 A/B pattern, generic hook target)
# ===========================================================================


def find_hook_target(model):
    for name, mod in model.named_modules():
        if name.endswith("drop") and isinstance(mod, torch.nn.Dropout):
            return mod
    return model.get_input_embeddings()


def eval_loss(model, chunks, wrapper=None):
    holder = {"ids": None}
    handle = None
    was_training = model.training
    if wrapper is not None:
        target = find_hook_target(model)

        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            fused = wrapper(hs[0].float(), holder["ids"])
            out = fused.unsqueeze(0)
            if isinstance(output, tuple):
                return (out,) + output[1:]
            return out

        handle = target.register_forward_hook(hook)
    losses = []
    model.eval()
    with torch.no_grad():
        for c in chunks:
            holder["ids"] = c
            o = model(c.unsqueeze(0), labels=c.unsqueeze(0))
            losses.append(o.loss.item() * len(c))
    if handle is not None:
        handle.remove()
    if was_training:
        model.train()
    return sum(losses) / max(sum(len(x) for x in chunks), 1)


# ===========================================================================
# TRAIN (LoRA preferred; time-budgeted)
# ===========================================================================


def build_lora(model, r):
    try:
        from peft import LoraConfig, get_peft_model
        if hasattr(model, "transformer"):
            targets = ["c_attn"]
        else:
            targets = ["q_proj", "k_proj", "v_proj"]
        cfg = LoraConfig(
            r=r, lora_alpha=2 * r, lora_dropout=0.0,
            target_modules=targets, task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(model, cfg)
        total = sum(p.numel() for p in peft_model.parameters())
        trainable = sum(
            p.numel() for p in peft_model.parameters() if p.requires_grad
        )
        return peft_model, {
            "used": True,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(100.0 * trainable / max(total, 1), 4),
        }
    except Exception as exc:
        log("peft unavailable, full-param fallback: " + str(exc)[:120])
        for p in model.parameters():
            p.requires_grad_(True)
        total = sum(p.numel() for p in model.parameters())
        return model, {
            "used": False,
            "trainable_params": total,
            "total_params": total,
            "trainable_pct": 100.0,
        }


def train_budget(model, chunks, minutes, lr, batch_size, rng):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    budget_s = minutes * 60
    t0 = time.time()
    step = 0
    first = None
    last = None
    curve = []
    model.train()
    while time.time() - t0 < budget_s and step < 100000:
        idx = rng.sample(range(len(chunks)),
                        min(batch_size, len(chunks)))
        batch = torch.stack([chunks[i] for i in idx])
        out = model(batch, labels=batch)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad()
        step += 1
        last = out.loss.item()
        if first is None:
            first = last
        if step == 1 or step % 10 == 0:
            curve.append({
                "step": step,
                "loss": round(last, 4),
                "elapsed_s": round(time.time() - t0, 1),
            })
    model.eval()
    return {
        "steps": step,
        "train_seconds": round(time.time() - t0, 1),
        "budget_minutes": minutes,
        "first_step_loss": round(first, 4) if first is not None else None,
        "last_step_loss": round(last, 4) if last is not None else None,
        "loss_curve": curve,
    }


# ===========================================================================
# MAIN
# ===========================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", default=os.environ.get("GITHUB_SHA", "unknown"))
    ap.add_argument("--out", default="e20-out")
    ap.add_argument("--model", default="sshleifer/tiny-gpt2")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-chunks", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    def flush(status, error=None):
        payload = {
            "sha": args.sha,
            "branch": os.environ.get("GITHUB_REF_NAME", "local"),
            "experiment": "e20-finetune-kv",
            "status": status,
            "model": args.model,
            "configs": CONFIGS,
            "claims": CLAIMS,
        }
        if error:
            payload["error"] = error
        payload["log"] = NL.join(LOG_LINES)
        with open(os.path.join(args.out, "run.json"), "w") as f:
            json.dump(payload, f, indent=2, default=str)

    log("=" * 70)
    log("E20: FINE-TUNE + KV WRAPPER  sha=" + args.sha)
    log("model=" + args.model + "  budget=" + str(args.minutes) + " min")
    log("=" * 70)
    flush("running")

    CLAIMS["kv_off_during_training"] = True  # by design
    CLAIMS["design_note"] = (
        "wrapper has no trainable params; training updates the base "
        + "model (LoRA); evictions would corrupt training context"
    )

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        err = traceback.format_exc()
        log("transformers missing: " + err.splitlines()[-1])
        flush("skipped: no transformers", err.splitlines()[-1])
        return

    try:
        torch.manual_seed(0)
        random.seed(0)
        rng = random.Random(0)
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
        model.eval()
        cfg = model.config
        d = int(getattr(cfg, "hidden_size", None)
                  or getattr(cfg, "n_embd", 64))

        ids, source = load_ids(tok)
        train_chunks, eval_chunks = make_chunks(ids)
        CONFIGS["data"] = {
            "source": source,
            "train_chunks": len(train_chunks),
            "eval_chunks": len(eval_chunks),
            "chunk_len": CHUNK,
        }
        log("data: " + source + " -> " + str(len(train_chunks))
            + " train / " + str(len(eval_chunks)) + " eval chunks")
        flush("data-loaded")

        # ---- Phase A: before fine-tune ----
        log("")
        log("--- Phase A: before fine-tune ---")
        loss_before = eval_loss(model, eval_chunks)
        wrapper = FinetuneKVWrapper(d)
        wrapper.reset()
        loss_wrapped_before = eval_loss(model, eval_chunks, wrapper)
        CONFIGS["phase_a"] = {
            "plain_eval_loss_nats": round(loss_before, 4),
            "wrapped_eval_loss_nats": round(loss_wrapped_before, 4),
            "attach_delta_nats": round(
                loss_wrapped_before - loss_before, 4),
        }
        log("plain loss: " + str(round(loss_before, 4))
            + "  wrapped loss: " + str(round(loss_wrapped_before, 4)))
        flush("phase-a-done")

        # ---- Phase B: time-budgeted fine-tune (wrapper OFF) ----
        log("")
        log("--- Phase B: fine-tune (" + str(args.minutes)
            + " min budget, wrapper OFF) ---")
        model, lora_info = build_lora(model, args.lora_r)
        CONFIGS["lora"] = lora_info
        log("lora used: " + str(lora_info["used"])
            + "  trainable: " + str(lora_info["trainable_pct"]) + "%")
        train_info = train_budget(
            model, train_chunks, args.minutes, args.lr,
            args.batch_chunks, rng,
        )
        CONFIGS["train"] = train_info
        log("steps: " + str(train_info["steps"])
            + "  seconds: " + str(train_info["train_seconds"])
            + "  loss: " + str(train_info["first_step_loss"])
            + " -> " + str(train_info["last_step_loss"]))
        flush("phase-b-done")

        # ---- Phase C: after fine-tune ----
        log("")
        log("--- Phase C: after fine-tune ---")
        loss_after = eval_loss(model, eval_chunks)
        wrapper2 = FinetuneKVWrapper(d)
        wrapper2.kv_enabled = True
        wrapper2.reset()
        loss_wrapped_after = eval_loss(model, eval_chunks, wrapper2)
        kv_after = {
            "kv_max": wrapper2.kv.size(),
            "kv_evictions": wrapper2.kv.evictions,
        }
        recall_after = wrapper2.base.get_stats()["recall_rate"]
        CONFIGS["phase_c"] = {
            "plain_eval_loss_nats": round(loss_after, 4),
            "wrapped_eval_loss_nats": round(loss_wrapped_after, 4),
            "attach_delta_nats": round(
                loss_wrapped_after - loss_after, 4),
            "kv": kv_after,
            "wrapper_recall_rate": round(recall_after, 4),
        }
        log("plain loss: " + str(round(loss_after, 4))
            + "  wrapped loss: " + str(round(loss_wrapped_after, 4)))
        log("kv max: " + str(kv_after["kv_max"])
            + "  evictions: " + str(kv_after["kv_evictions"]))
        flush("phase-c-done")

        # ---- claims (all measured in THIS run) ----
        CLAIMS["trained"] = train_info["steps"] > 0
        CLAIMS["within_time_budget"] = bool(
            train_info["train_seconds"] <= args.minutes * 60 + 5)
        CLAIMS["train_loss_decreased"] = bool(
            train_info["last_step_loss"] is not None
            and train_info["first_step_loss"] is not None
            and train_info["last_step_loss"]
            < train_info["first_step_loss"])
        CLAIMS["eval_loss_improved"] = bool(loss_after < loss_before)
        CLAIMS["eval_improvement_nats"] = round(
            loss_before - loss_after, 4)
        CLAIMS["attach_delta_before_nats"] = round(
            loss_wrapped_before - loss_before, 4)
        CLAIMS["attach_delta_after_nats"] = round(
            loss_wrapped_after - loss_after, 4)
        CLAIMS["kv_stays_bounded_after_finetune"] = bool(
            kv_after["kv_max"] <= KV_MAX
            and kv_after["kv_evictions"] > 0)
        CLAIMS["wrapper_recall_after_finetune"] = round(
            recall_after, 4)
        CLAIMS["lora_used"] = lora_info["used"]
    except Exception:
        err = traceback.format_exc()
        log("ERROR: " + err)
        flush("error", err.splitlines()[-1])
        return

    # ---- markdown report ----
    a = CONFIGS.get("phase_a", {})
    c = CONFIGS.get("phase_c", {})
    t = CONFIGS.get("train", {})
    kv = c.get("kv") or {}
    lines = [
        "# E20: short-budget fine-tune + KV wrapper",
        "",
        "- commit: " + args.sha,
        "- model: " + args.model,
        "- budget: " + str(args.minutes) + " min",
        "- KV wrapper OFF during training (by design),",
        "  attached + bounded-KV checked AFTER fine-tuning",
        "",
        "## Measured",
        "",
        "| Metric | Before fine-tune | After fine-tune |",
        "|---|---|---|",
        "| plain eval loss (nats) | "
        + str(a.get("plain_eval_loss_nats")) + " | "
        + str(c.get("plain_eval_loss_nats")) + " |",
        "| wrapped eval loss (nats) | "
        + str(a.get("wrapped_eval_loss_nats")) + " | "
        + str(c.get("wrapped_eval_loss_nats")) + " |",
        "| attach delta (nats) | "
        + str(a.get("attach_delta_nats")) + " | "
        + str(c.get("attach_delta_nats")) + " |",
        "",
        "- train steps: " + str(t.get("steps"))
        + " in " + str(t.get("train_seconds")) + " s",
        "- train loss: " + str(t.get("first_step_loss"))
        + " -> " + str(t.get("last_step_loss")),
        "- KV after fine-tune: max " + str(kv.get("kv_max"))
        + ", evictions " + str(kv.get("kv_evictions"))
        + " (bounded by " + str(KV_MAX) + ")",
        "- wrapper recall after fine-tune: "
        + str(c.get("wrapper_recall_rate")),
        "",
        "## Claims (measured in THIS run)",
        "",
    ]
    for k, v in CLAIMS.items():
        lines.append("- **" + str(k) + "**: " + str(v))
    lines += ["", "## Full log", "", FENCE] + LOG_LINES + [FENCE]
    with open(os.path.join(args.out, "finetune.md"), "w") as f:
        f.write(NL.join(lines))

    log("")
    log("Wrote " + args.out + "/run.json and "
        + args.out + "/finetune.md")
    log("E20 COMPLETE")
    flush("complete")


if __name__ == "__main__":
    main()
