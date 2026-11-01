"""
superfluid.gpt2_model — a minimal GPT-2 forward pass, in PyTorch.

WHY
---
To attach the memory layer to a REAL pretrained checkpoint without
depending on `transformers`, this file reads a GPT-2 checkpoint
(`config.json` + `model.safetensors`) and implements the forward pass
directly.  It is ~120 lines and it is verified by generation: the model
must produce coherent English, which it does.

The project itself stays dependency-free (the numpy package does not
import this module); torch is needed only for this file and for
`superfluid.torch_graft`.
"""

from __future__ import annotations

import json
import math
import os

__all__ = ["GPT2Model"]


def _layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps) * g + b


def _gelu(x):
    # GPT-2 uses the tanh approximation; the erf version is within 1e-3
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


import torch  # noqa: E402  (deliberately after the helpers, so the numpy

                         #  package never imports torch)


class GPT2Model:
    """A GPT-2 (and GPT-2-family) causal LM loaded from a local folder."""

    def __init__(self, path: str, dtype=torch.float32, device="cpu"):
        from safetensors.torch import load_file
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        self.cfg = cfg
        self.n_layer = cfg["n_layer"]
        self.n_head = cfg["n_head"]
        self.n_embd = cfg["n_embd"]
        self.device = device
        sd = load_file(os.path.join(path, "model.safetensors"))
        self.wte = sd["wte.weight"].to(device=device, dtype=dtype)
        self.wpe = sd["wpe.weight"].to(device=device, dtype=dtype)
        self.blocks = []
        for i in range(self.n_layer):
            p = f"h.{i}."
            self.blocks.append({
                "ln1_g": sd[p + "ln_1.weight"].to(device=device, dtype=dtype),
                "ln1_b": sd[p + "ln_1.bias"].to(device=device, dtype=dtype),
                "attn_w": sd[p + "attn.c_attn.weight"].to(device=device, dtype=dtype),
                "attn_b": sd[p + "attn.c_attn.bias"].to(device=device, dtype=dtype),
                "proj_w": sd[p + "attn.c_proj.weight"].to(device=device, dtype=dtype),
                "proj_b": sd[p + "attn.c_proj.bias"].to(device=device, dtype=dtype),
                "ln2_g": sd[p + "ln_2.weight"].to(device=device, dtype=dtype),
                "ln2_b": sd[p + "ln_2.bias"].to(device=device, dtype=dtype),
                "fc_w": sd[p + "mlp.c_fc.weight"].to(device=device, dtype=dtype),
                "fc_b": sd[p + "mlp.c_fc.bias"].to(device=device, dtype=dtype),
                "mlp_w": sd[p + "mlp.c_proj.weight"].to(device=device, dtype=dtype),
                "mlp_b": sd[p + "mlp.c_proj.bias"].to(device=device, dtype=dtype),
            })
        self.ln_f_g = sd["ln_f.weight"].to(device=device, dtype=dtype)
        self.ln_f_b = sd["ln_f.bias"].to(device=device, dtype=dtype)
        self.dtype = dtype

    # ------------------------------------------------------------------ core
    def forward(self, ids, past=None, h_override=None):
        """ids: (T,) int64 -> logits (T, V).

        `h_override` replaces the embedding output: that is the hook a
        grafted memory layer needs (the layer rewrites the residual stream,
        the frozen model continues from it).
        """
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        T = ids.shape[0]
        H, D = self.n_head, self.n_embd
        dh = D // H
        h = self.wte[ids] + self.wpe[:T]
        if h_override is not None:
            h = h_override
        mask = torch.tril(torch.ones(T, T, device=self.device, dtype=self.dtype))
        for b in self.blocks:
            x = _layernorm(h, b["ln1_g"], b["ln1_b"])
            qkv = x @ b["attn_w"] + b["attn_b"]
            q, k, v = qkv.split(D, dim=-1)
            q = q.view(T, H, dh).transpose(0, 1)
            k = k.view(T, H, dh).transpose(0, 1)
            v = v.view(T, H, dh).transpose(0, 1)
            att = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
            att = att.masked_fill(mask == 0, float("-inf"))
            att = torch.softmax(att, dim=-1)
            o = (att @ v).transpose(0, 1).reshape(T, D)
            h = h + (o @ b["proj_w"] + b["proj_b"])
            y = _layernorm(h, b["ln2_g"], b["ln2_b"])
            y = _gelu(y @ b["fc_w"] + b["fc_b"])
            h = h + (y @ b["mlp_w"] + b["mlp_b"])
        h = _layernorm(h, self.ln_f_g, self.ln_f_b)
        return h @ self.wte.T

    # ------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, ids, n_new=20, temperature=0.0, seed=0):
        ids = list(ids)
        for _ in range(n_new):
            window = ids[-self.cfg.get("n_positions", 1024):]
            logits = self.forward(torch.tensor(window, dtype=torch.long,
                                               device=self.device))
            nxt = logits[-1]
            if temperature > 0:
                torch.manual_seed(seed)
                probs = torch.softmax(nxt / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1)
            else:
                nxt = nxt.argmax(dim=-1)
            ids.append(int(nxt))
        return ids
