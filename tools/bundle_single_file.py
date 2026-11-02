#!/usr/bin/env python3
"""
tools/bundle_single_file.py — generate a single self-contained, self-verifying
Python file from the `fluidlayer/` package + `selftest.py`.

    python3 tools/bundle_single_file.py
    python3 fluidlayer_single.py --selftest      # runs the self-checks

The bundle registers a real `fluidlayer` package inside sys.modules, so the
relative imports (`from .diffusion import ...`) keep working exactly as in the
normal package layout — no source rewriting, no drift.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MODULES = [
    "__init__",
    "autodiff",
    "sheaf",
    "diffusion",
    "topology",
    "blocks",
    "memory",
    "model",
    "losses",
    "optim",
    "data",
    "graft",
    "gpt2_tokenizer",
    "bpe",
]
TEMPLATE = os.path.join(HERE, "_bundle_template.py")
PLACEHOLDER = "%%SOURCES%%"


def main():
    parts = []
    for name in MODULES:
        path = os.path.join(ROOT, "fluidlayer", name + ".py")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"bundle source module does not exist: {path}")
        with open(path, encoding="utf-8") as source_file:
            parts.append(f"#--- FILE: fluidlayer/{name}.py ---")
            parts.append(source_file.read())
    with open(os.path.join(ROOT, "selftest.py"), encoding="utf-8") as f:
        parts.append("#--- FILE: selftest.py ---")
        parts.append(f.read())
    body = "\n".join(parts)

    # The sources are embedded in a raw triple-single-quoted string, so they
    # must not contain a triple single quote.  Check instead of corrupting.
    if "'''" in body:
        raise SystemExit("a source file contains a triple single quote; "
                         "the single-file bundle cannot be generated safely")

    with open(TEMPLATE, encoding="utf-8") as template_file:
        tpl = template_file.read()
    if PLACEHOLDER not in tpl:
        raise SystemExit(f"template is missing the {PLACEHOLDER} placeholder")
    out = tpl.replace(PLACEHOLDER, body)

    dest = os.path.join(ROOT, "fluidlayer_single.py")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"wrote {dest} ({len(out)} bytes, {out.count(chr(10)) + 1} lines)")


if __name__ == "__main__":
    main()
