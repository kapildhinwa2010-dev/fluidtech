#!/usr/bin/env python3
"""
tools/download_gpt2.py -- fetch a real GPT-2 checkpoint (no HF libraries needed).

    python tools/download_gpt2.py --out models/gpt2

Downloads `config.json`, `vocab.json`, `merges.txt` and `model.safetensors`
from the HuggingFace CDN with plain HTTPS (and a resumable transfer, because
the sandbox's proxy truncates long downloads).

The checkpoint is ~550 MB.  `experiments/e14_real_gpt2.py` then runs the
memory layer on it.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

BASE = "https://huggingface.co/gpt2/resolve/main"
FILES = ["config.json", "vocab.json", "merges.txt", "model.safetensors"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/gpt2")
    ap.add_argument("--only", nargs="*", default=None,
                    help="download only these files")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for f in (args.only or FILES):
        dest = os.path.join(args.out, f)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"  {f}: already there")
            continue
        print(f"  downloading {f} ...")
        r = subprocess.run(["curl", "-sL", "--retry", "10", "--retry-all-errors",
                            "--retry-delay", "3", "-C", "-", "-o", dest,
                            f"{BASE}/{f}"])
        if r.returncode != 0:
            sys.exit(f"failed to download {f}")
        print(f"    {os.path.getsize(dest):,} bytes")
    print(f"the checkpoint is in {args.out}")


if __name__ == "__main__":
    main()
