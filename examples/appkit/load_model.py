"""Load a supported local GPT-2 or Qwen2 checkpoint for the app-kit demo.

Requires the published ``fluidlayer[gpu]`` extra and local model weights.
GPT-2 uses the package's tokenizer; Qwen2 uses its Hugging Face tokenizer.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DEFAULT_MODEL_DIR = os.environ.get("MODEL_DIR", "models/qwen05b")


def _ensure_checkout_importable():
    """Make repository examples runnable without an editable install."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "fluidlayer" / "__init__.py").is_file():
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return


def load(model_dir: str = DEFAULT_MODEL_DIR, dtype: str = "bfloat16",
         device: str = "cpu"):
    """Load the checkpoint and its tokenizer.

    Returns (model, tokenizer).  The model exposes:
        model.forward(ids, h_override=None) -> logits (T, V)
    `h_override` is the hook the memory layer needs.
    """
    import torch
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    arch = str(cfg.get("model_type", "")).lower()
    _ensure_checkout_importable()
    if arch == "qwen2":
        from transformers import AutoTokenizer

        from fluidlayer.qwen2_model import Qwen2Model
        model = Qwen2Model(model_dir, dtype=getattr(torch, dtype), device=device)
        tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    elif arch == "gpt2":
        from fluidlayer.gpt2_model import GPT2Model
        from fluidlayer.gpt2_tokenizer import GPT2Tokenizer

        model = GPT2Model(model_dir, dtype=getattr(torch, dtype), device=device)
        tok = GPT2Tokenizer.from_pretrained(model_dir)
    else:
        raise ValueError(f"unsupported model_type in {cfg_path}: {arch!r}")
    return model, tok


if __name__ == "__main__":
    m, t = load()
    print("the model:", type(m).__name__,
          "| the vocab:", t.vocab_size, "tokens")
    ids = t.encode("The capital of France is")
    out = m.generate(ids, n_new=10)
    print("the generation:", repr(t.decode(out)))
