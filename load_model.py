"""
load_model.py -- ONLY the model loading.  Copy this file into your app.

This file does one thing: it loads a REAL pretrained checkpoint (with its
tokenizer) so that the memory layer can be attached to it.  It has no
opinion about the layer -- that is `attach_layer.py`.

    from load_model import load
    model, tok = load()                 # Qwen2.5-0.5B by default
    model, tok = load("/path/to/gpt2")  # GPT-2 works too

Requirements: torch + safetensors.  The forward pass is written from
scratch in this repository (`qwen2_model.py` / `gpt2_model.py`), so
`transformers` is NOT needed.

The layer package is imported as `fluidlayer` (the published name) or
`superfluid` (the original name); if neither is installed, the sibling
files in this folder are used.
"""
from __future__ import annotations

import os
import sys

# where the checkpoint lives (override with the MODEL_DIR environment
# variable)
DEFAULT_MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/models/qwen05b")


def _import_loader(model_dir: str):
    """Return (ModelClass, TokenizerClass) from whichever copy of the
    package is available."""
    try:
        from fluidlayer.qwen2_model import Qwen2Model
        from fluidlayer.gpt2_tokenizer import GPT2Tokenizer
        return Qwen2Model, GPT2Tokenizer
    except ImportError:
        pass
    try:
        from superfluid.qwen2_model import Qwen2Model
        from superfluid.gpt2_tokenizer import GPT2Tokenizer
        return Qwen2Model, GPT2Tokenizer
    except ImportError:
        pass
    # the sibling files, or a checkout of this repository
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(here, "qwen2_model.py")):
        if here not in sys.path:
            sys.path.insert(0, here)
        from qwen2_model import Qwen2Model
        from gpt2_tokenizer import GPT2Tokenizer
        return Qwen2Model, GPT2Tokenizer
    # walk up looking for a checkout that contains the package
    cur = here
    for _ in range(5):
        for pkg in ("superfluid", "fluidlayer"):
            if os.path.exists(os.path.join(cur, pkg, "qwen2_model.py")):
                if cur not in sys.path:
                    sys.path.insert(0, cur)
                # import the submodule directly: the package __init__
                # deliberately does not import the torch-dependent modules
                m = __import__(f"{pkg}.qwen2_model", fromlist=["Qwen2Model"])
                t = __import__(f"{pkg}.gpt2_tokenizer",
                               fromlist=["GPT2Tokenizer"])
                return (m.Qwen2Model, t.GPT2Tokenizer)
        cur = os.path.dirname(cur)
    raise ImportError(
        "the fluidlayer package (or qwen2_model.py next to this file) "
        "is required: pip install fluidlayer")


def load(model_dir: str = DEFAULT_MODEL_DIR, dtype: str = "bfloat16",
         device: str = "cpu"):
    """Load the checkpoint and its tokenizer.

    Returns (model, tokenizer).  The model exposes:
        model.forward(ids, h_override=None) -> logits (T, V)
    `h_override` is the hook the memory layer needs.
    """
    import torch
    Qwen2Model, GPT2Tokenizer = _import_loader(model_dir)

    arch = "gpt2"
    cfg_path = os.path.join(model_dir, "config.json")
    if os.path.exists(cfg_path):
        import json
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        arch = cfg.get("model_type", cfg.get("architectures", ["gpt2"])[0]
                       if isinstance(cfg.get("architectures"), list) else "gpt2")
    if "qwen" in str(arch).lower():
        model = Qwen2Model(model_dir, dtype=getattr(torch, dtype), device=device)
    else:
        from gpt2_model import GPT2Model      # the sibling GPT-2 loader
        model = GPT2Model(model_dir, dtype=getattr(torch, dtype), device=device)
    tok = GPT2Tokenizer.from_pretrained(model_dir)
    return model, tok


if __name__ == "__main__":
    m, t = load()
    print("the model:", type(m).__name__,
          "| the vocab:", len(t.encoder), "tokens")
    ids = t.encode("The capital of France is")
    out = m.generate(ids, n_new=10)
    print("the generation:", repr(t.decode(out)))
