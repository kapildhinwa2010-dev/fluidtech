#!/usr/bin/env python3
"""
tools/fetch_pypi.py -- install packages from PyPI without pip's index client.

The sandbox's pip cannot reach the PyPI *simple* index reliably (the
connection resets), but plain HTTPS to pypi.org and files.pythonhosted.org
works.  So this script does the resolution itself: query the JSON API,
pick the right wheel for this interpreter, download it, and let pip
install the local file.

    python3 tools/fetch_pypi.py transformers safetensors
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request

CACHE = "/tmp/wheels"
PY_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"
MACHINE = platform.machine()


def _score(fn: str) -> int:
    """Higher is better.  Prefer a wheel matching this interpreter."""
    if fn.endswith("-none-any.whl"):
        return 10
    if PY_TAG in fn and "manylinux" in fn and MACHINE in fn:
        return 100
    if PY_TAG in fn and "manylinux" in fn:
        return 90
    if PY_TAG in fn and "linux" in fn:
        return 80
    if PY_TAG in fn:
        return 70
    if "manylinux" in fn and MACHINE in fn:
        return 60
    return 1


def fetch(pkg: str) -> str | None:
    os.makedirs(CACHE, exist_ok=True)
    url = f"https://pypi.org/pypi/{pkg}/json"
    with urllib.request.urlopen(url, timeout=60) as r:
        meta = json.load(r)
    best = None
    for f in meta["urls"]:
        if f["packagetype"] != "bdist_wheel":
            continue
        fn = f["filename"]
        sc = _score(fn)
        if best is None or sc > best[0]:
            best = (sc, fn, f["url"], f.get("digests", {}).get("sha256"))
    if best is None:
        return None
    _, fn, furl, sha = best
    dest = os.path.join(CACHE, fn)
    def _ok(path):
        if not os.path.exists(path):
            return False
        if not sha:
            return True
        return hashlib.sha256(open(path, "rb").read()).hexdigest() == sha

    for attempt in range(8):
        if _ok(dest):
            break
        # curl with --continue-at: resumes a truncated transfer, which is what
        # this sandbox's proxy needs for the larger wheels.
        cmd = ["curl", "-sL", "--retry", "8", "--retry-all-errors",
               "--retry-delay", "2", "-C", "-", "-o", dest, furl]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if _ok(dest):
            break
        print(f"    attempt {attempt + 1}: incomplete "
              f"({os.path.getsize(dest) if os.path.exists(dest) else 0} B)")
        time.sleep(2)
    if not os.path.exists(dest):
        return None
    return dest


def main(pkgs) -> int:
    paths = []
    for p in pkgs:
        try:
            pth = fetch(p)
        except Exception as e:                       # noqa: BLE001
            print(f"  !! {p}: {e}")
            pth = None
        if pth:
            paths.append(pth)
    if paths:
        cmd = [sys.executable, "-m", "pip", "install", "--break-system-packages",
               "--no-cache-dir", "--no-deps"] + paths
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-1500:])
        if r.returncode:
            print(r.stderr[-1500:])
            return r.returncode
    for p in pkgs:
        mod = p.replace("-", "_")
        try:
            __import__(mod)
            print(f"  {p}: OK")
        except Exception as e:                       # noqa: BLE001
            print(f"  {p}: FAILED ({e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
