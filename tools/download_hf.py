#!/usr/bin/env python3
"""
tools/download_hf.py -- fetch a HuggingFace checkpoint without the `huggingface_hub`
package (and without `transformers`).

    python tools/download_hf.py Qwen/Qwen2.5-0.5B --out models/qwen05b
    python tools/download_hf.py gpt2 --out models/gpt2

Resumable: an interrupted download continues with an HTTP Range request.
Only the files the framework actually needs are fetched (the config, the
weights and the tokenizer), never the whole repository.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

BASE = "https://huggingface.co/{repo}/resolve/main/{name}"

# the files a GPT-2-style or Qwen-style checkpoint needs
NEEDED = ["config.json", "generation_config.json", "model.safetensors",
          "tokenizer.json", "tokenizer_config.json", "vocab.json",
          "merges.txt", "vocab.bpe", "tokenizer.model"]
# repos that shard the weights
SHARDS = ["model.safetensors.index.json"]


def _content_length(url: str) -> int:
    """The real size of the file.  A truncated read looks exactly like a
    clean EOF, so the size is the only trustworthy check."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status in (301, 302, 307, 308) and r.headers.get("Location"):
                return _content_length(r.headers["Location"])
            return int(r.headers.get("Content-Length", 0))
    except Exception:
        return 0


def _download(url: str, dest: str) -> None:
    """Download `url` to `dest`, resuming and verifying the size.

    Two failure modes are handled explicitly, because both look like a
    successful EOF otherwise:
      * a truncated read (the connection drops)  -> resume with a Range
      * a server that IGNORES the Range header  -> restart from zero
    """
    tmp = dest + ".part"
    want = _content_length(url)
    pos = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if want and pos > want:
        os.remove(tmp)
        pos = 0
    for attempt in range(8):
        if want and pos == want:
            os.replace(tmp, dest)
            print(f"  ok  {os.path.basename(dest)}  {os.path.getsize(dest)/1e6:.1f} MB",
                  flush=True)
            return
        req = urllib.request.Request(url)
        if pos:
            req.add_header("Range", f"bytes={pos}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                append = pos > 0 and r.status == 206
                if pos and not append:
                    # the server ignored the Range: start over
                    print("    the server ignored the Range -- restarting", flush=True)
                    pos = 0
                mode = "ab" if append else "wb"
                with open(tmp, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        pos += len(chunk)
                        if pos % (1 << 25) < (1 << 20):
                            print(f"    {pos/1e6:.0f} / {(want or 0)/1e6:.0f} MB", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"    retry {attempt+1}: {exc}", flush=True)
            time.sleep(2 + 2 * attempt)
            continue
        if not want:
            os.replace(tmp, dest)
            print(f"  ok  {os.path.basename(dest)}  {os.path.getsize(dest)/1e6:.1f} MB",
                  flush=True)
            return
        print(f"    truncated: {pos}/{want} -- resuming", flush=True)
        time.sleep(1)
    raise SystemExit(f"could not download {url}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    repo = sys.argv[1]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else f"models/{repo.split('/')[-1]}"
    os.makedirs(out, exist_ok=True)
    names = list(NEEDED)
    # the weight file may be sharded
    idx = os.path.join(out, "model.safetensors.index.json")
    try:
        with urllib.request.urlopen(BASE.format(repo=repo, name="model.safetensors.index.json"), timeout=60) as r:
            data = json.load(r)
        shards = sorted(set(data["weight_map"].values()))
        names += shards
    except Exception:
        pass
    for name in names:
        dest = os.path.join(out, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"  skip {name} (present)")
            continue
        try:
            _download(BASE.format(repo=repo, name=name), dest)
        except SystemExit:
            continue
    print(f"\nthe checkpoint is in {out}/")


if __name__ == "__main__":
    main()
