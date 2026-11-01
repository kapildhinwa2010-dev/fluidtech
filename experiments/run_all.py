#!/usr/bin/env python3
"""
experiments/run_all.py — run every check and every experiment, in order.

    python3 experiments/run_all.py            # the standard sweep
    python3 experiments/run_all.py --quick    # a fast smoke run
    python3 experiments/run_all.py --full     # the 1,000,000-token E5 run

Writes results/*.json and prints a summary table.  Every number quoted in
README.md and in the design documents comes from these files, so nothing in
the repository is a claim that was not measured here.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

PY = sys.executable


def run(cmd, label):
    print("\n" + "=" * 78)
    print(f"### {label}")
    print("=" * 78)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = p.stdout or ""
    tail = out.strip().splitlines()[-14:]
    print("\n".join(tail))
    if p.returncode != 0:
        print("STDERR:", (p.stderr or "")[-2000:])
        print(f"!!! {label} FAILED (rc={p.returncode})")
    else:
        print(f"--- {label}: ok in {time.time() - t0:.1f}s")
    return p.returncode


def show(path, keys):
    f = os.path.join(RESULTS, path)
    if not os.path.exists(f):
        print(f"    (missing {path})")
        return
    d = json.load(open(f))
    for k in keys:
        if k in d:
            v = d[k]
            if isinstance(v, float):
                print(f"    {k:34s} {v:.4f}")
            elif isinstance(v, dict):
                print(f"    {k:34s} {v}")
            else:
                print(f"    {k:34s} {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--skip", default="", help="comma separated step names")
    args = ap.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    rc_all = 0

    steps = [
        ("selftest", [PY, "selftest.py"]),
        ("bundle", [PY, "superfluid_single.py", "--selftest"]),
        ("e1", [PY, "experiments/e1_prior_perplexity.py",
                "--steps", "2" if args.quick else "250",
                "--d", "32" if args.quick else "64",
                "--layers", "1" if args.quick else "2",
                "--ctx", "32" if args.quick else "64",
                "--batch", "4" if args.quick else "16",
                "--windows", "12" if args.quick else "1000"]),
        ("e6", [PY, "experiments/e6_graft.py",
                "--steps", "2" if args.quick else "400",
                "--branch-steps", "2" if args.quick else "40",
                "--docs", "4" if args.quick else "40",
                "--train-docs", "8" if args.quick else "200",
                "--d", "16" if args.quick else "64",
                "--layers", "1" if args.quick else "2",
                "--ctx", "32" if args.quick else "128"]),
        ("e7", [PY, "experiments/e7_lossy_fix.py",
                "--steps", "2" if args.quick else "400",
                "--layer-steps", "2" if args.quick else "60",
                "--d", "16" if args.quick else "64",
                "--ctx", "32" if args.quick else "128",
                "--max-chars", "4096" if args.quick else "120000"]),
        ("e6r", [PY, "experiments/e6_real_corpus.py",
                 "--steps", "2" if args.quick else "400",
                 "--branch-steps", "1" if args.quick else "40",
                 "--d", "16" if args.quick else "64",
                 "--ctx", "32" if args.quick else "128",
                 "--max-chars", "4096" if args.quick else "120000"]),
        ("e5", [PY, "experiments/e5_graft.py",
                "--tokens", "10000" if args.quick else ("1000000" if args.full else "200000"),
                "--budget", "40" if args.quick else "100"]),
        ("e2", [PY, "experiments/e2_scaling.py",
                "--tokens", "10000" if args.quick else "200000"]),
        ("e4", [PY, "experiments/e4_gbalance.py", "--trials", "3" if args.quick else "30"]),
        ("prior", [PY, "experiments/pretrain_prior.py", "--synthetic",
                   "--tokens", "10000" if args.quick else "200000"]),
    ]
    for name, cmd in steps:
        if name in skip:
            print(f"\n(skipping {name})")
            continue
        rc = run(cmd, name)
        rc_all |= rc

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    show("selftest.json", ["n_pass", "n_total"])
    show("e5_graft.json", ["gist_cosine_local", "gist_cosine_global", "state_bytes",
                            "codebook_bytes", "kv_over_ours"])
    f = os.path.join(RESULTS, "e5_graft.json")
    if os.path.exists(f):
        d = json.load(open(f))
        print("    E5 policies:")
        for k, v in d["policies"].items():
            print(f"      {k:14s} precision={v['pin_precision']:.3f}  "
                  f"fact_recall={v['fact_recall']:.3f}  "
                  f"exact|promoted={v['exact_given_promoted']:.3f}")
    f = os.path.join(RESULTS, "e1_prior.json")
    if os.path.exists(f):
        d = json.load(open(f))
        b = d["base"]["val_nll"]
        print(f"    E1 (frozen LM + training-free prior), base val nats {b:.4f}:")
        for k in ("fluid_prior_static", "fluid_prior_live", "unigram_bias",
                  "unigram_lowrank", "bigram_bias"):
            if k in d:
                print(f"      {k:20s} {d[k]['val_nll']:.4f}  "
                      f"delta {d[k]['nats_gained']:+.4f}  "
                      f"{d[k]['nats_per_kb']:.2f} nats/KB  "
                      f"({d[k]['state_bytes']:,} B)")
    show("e2_scaling.json", ["runtime_sec"])
    f = os.path.join(RESULTS, "e2_scaling.json")
    if os.path.exists(f):
        d = json.load(open(f))
        print("    E2 axis A (anchors M):")
        for r in d.get("axis_anchors_M", []):
            print(f"      M={r['M']:>6}  gist={r['gist_cosine']:.4f}  "
                  f"flops/tok={r['flops_per_token']:,}")
        print("    E2 axis B (exact budget K):")
        for r in d.get("axis_exact_budget_K", []):
            print(f"      K={r['budget_slots']:>6}  recall={r['fact_recall']:.3f}  "
                  f"precision={r['pin_precision']:.3f}  "
                  f"register={r['register_bytes']:,} B")
    show("e4_gbalance.json", ["runtime_sec"])
    f = os.path.join(RESULTS, "e4_gbalance.json")
    if os.path.exists(f):
        d = json.load(open(f))
        print(f"    frustration agreement: {d['frustration_agreement']['n_agree']}/"
              f"{d['frustration_agreement']['n']}")
        print(f"    pearson(lam2, quality) = {d['lam2_vs_quality']['pearson']:.4f}  "
              f"spearman = {d['lam2_vs_quality']['spearman']:.4f}")
        c = d["contraction_certificate"]
        print(f"    contraction certificate: {c['n_holding']}/{c['n_checked']} hold")
    f = os.path.join(ROOT, "prior", "manifest.json")
    if os.path.exists(f):
        d = json.load(open(f))
        print(f"    prior: runtime {d['runtime_state_bytes']} B, "
              f"artefact {d['artefact_bytes_on_disk']} B, "
              f"M={d['budget']['M']}, slots={d['budget']['slots']}")

    print("\n" + "=" * 78)
    print("DONE" if rc_all == 0 else "SOME STEPS FAILED")
    print("=" * 78)
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
