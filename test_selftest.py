"""The self-verification suite, run under pytest.

`selftest.py` is the project's own harness (it must work without pytest,
because it is what a stranger runs first).  This file just runs it and
asserts that everything passed, so CI cannot go green on a broken harness.
"""
import os
import subprocess
import sys


def test_the_whole_suite_passes(root):
    st = os.path.join(root, "selftest.py")
    r = subprocess.run([sys.executable, st], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    line = [ln for ln in r.stdout.splitlines() if "checks passed" in ln]
    assert line, "the harness printed no result line:\n" + r.stdout[-3000:]
    passed, total = line[-1].split()[1].split("/")
    assert passed == total, f"{passed}/{total} checks passed"
    assert int(total) >= 38, f"expected at least 38 checks, got {total}"


def test_the_bundle_passes(root):
    r = subprocess.run(
        [sys.executable, os.path.join(root, "superfluid_single.py"), "--selftest"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "checks passed" in r.stdout
