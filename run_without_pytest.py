#!/usr/bin/env python3
"""
run_without_pytest.py -- run the test suite with NO pytest.

pytest is the normal way to run these tests, and it is what CI uses.  But
this project is meant to work with nothing but numpy, on a laptop, with no
network -- so this runner provides the tiny subset of pytest the tests
actually use (the `fixture` decorator, `pytest.raises`, `pytest.mark`) and
runs everything.

    python run_without_pytest.py

Exit code 0 = all tests passed.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import traceback
import types

ROOT = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(ROOT, "tests")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# --------------------------------------------------------------------------
# a very small pytest shim
# --------------------------------------------------------------------------
def _install_shim():
    try:
        import pytest  # noqa: F401
        return "pytest"
    except ImportError:
        pass

    mod = types.ModuleType("pytest")

    class _Skipped(Exception):
        pass

    def fixture(*args, **kwargs):
        def deco(fn):
            fn.__is_fixture__ = True
            return fn
        if args and callable(args[0]):
            return deco(args[0])
        return deco

    class _Raises:
        def __init__(self, expected):
            self.expected = expected
            self.value = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                raise AssertionError(f"DID NOT RAISE {self.expected}")
            if not issubclass(exc_type, self.expected):
                return False
            self.value = exc
            return True

    mod.fixture = fixture
    mod.raises = _Raises
    mod.mark = types.SimpleNamespace(parametrize=lambda *a, **k: (lambda f: f))
    mod.importorskip = lambda name: _import_or_skip(name, _Skipped)
    sys.modules["pytest"] = mod
    return "the built-in shim", _Skipped


def _import_or_skip(name, skipped):
    try:
        return __import__(name)
    except ImportError as error:
        raise skipped(str(error)) from error


# --------------------------------------------------------------------------
# the fixtures the tests ask for
# --------------------------------------------------------------------------
def _fixtures():
    from fluidlayer import TinyTransformer

    def root():
        return ROOT

    def model():
        return TinyTransformer(32, d=32, n_layers=2, n_ctx=32, seed=0)

    return {"root": root, "model": model}


def _load(path):
    name = "t_" + os.path.basename(path)[:-3]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    installed = _install_shim()
    if isinstance(installed, str):
        print("pytest is installed; delegating to pytest")
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-q", TESTS], check=False
        ).returncode

    which, skipped_exception = installed
    print(f"running the tests with {which}\n")
    fx = _fixtures()
    files = sorted(f for f in os.listdir(TESTS)
                   if f.startswith("test_") and f.endswith(".py"))
    passed = failed = skipped = 0
    failures = []
    for f in files:
        try:
            mod = _load(os.path.join(TESTS, f))
        except skipped_exception as error:
            skipped += 1
            print(f"  [SKIP] {f}: {error}")
            continue
        names = [n for n in dir(mod) if n.startswith("test_")]
        for n in sorted(names):
            fn = getattr(mod, n)
            if not callable(fn):
                continue
            kwargs = {}
            code = fn.__code__
            for arg in code.co_varnames[: code.co_argcount]:
                if arg in fx:
                    kwargs[arg] = fx[arg]()
            try:
                fn(**kwargs)
                passed += 1
                print(f"  [PASS] {f}::{n}")
            except skipped_exception as error:
                skipped += 1
                print(f"  [SKIP] {f}::{n}: {error}")
            except Exception:
                failed += 1
                failures.append((f, n, traceback.format_exc()))
                print(f"  [FAIL] {f}::{n}")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    for f, n, tb in failures:
        print(f"\n--- {f}::{n} ---\n{tb}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
