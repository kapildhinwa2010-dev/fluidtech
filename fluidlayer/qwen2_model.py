"""
fluidlayer.qwen2_model — a minimal Qwen2 forward pass, in PyTorch.

This optional module loads a local Qwen2 checkpoint from its config and
safetensors weights. The NumPy core does not import it; PyTorch and
safetensors are required only when this module or the model is used.

The implementation supports RMSNorm, grouped-query attention with RoPE,
SwiGLU MLPs, and tied or separate output embeddings.
"""

from __future__ import annotations

import json
import os

__all__ = ["Qwen2Model"]

import torch  # noqa: E402


def _rms_norm(x, weight, eps: float):
    dt = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * weight.float()).to(dt)


class Qwen2Model:
    """A Qwen2 causal language model loaded from a local checkpoint folder."""

    def __init__(self, path: str, dtype=torch.float32, device="cpu"):
        from safetensors.torch import load_file

        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        self.cfg = cfg
        self.n_layer = cfg["num_hidden_layers"]
        self.n_head = cfg["num_attention_heads"]
        self.n_kv = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
        self.n_embd = cfg["hidden_size"]
        self.head_dim = self.n_embd // self.n_head
        self.eps = cfg.get("rms_norm_eps", 1e-6)
        self.rope_theta = cfg.get("rope_theta", 10000.0)
        self.device = device
        self.dtype = dtype

        state = load_file(os.path.join(path, "model.safetensors"))

        def tensor(name):
            return state[name].to(device=device, dtype=dtype)

        self.wte = tensor("model.embed_tokens.weight")
        self.norm_w = tensor("model.norm.weight")
        self.lm_head_w = (
            tensor("lm_head.weight") if "lm_head.weight" in state else self.wte
        )
        self.layers = []
        for i in range(self.n_layer):
            prefix = f"model.layers.{i}."
            q_bias = prefix + "self_attn.q_proj.bias"
            k_bias = prefix + "self_attn.k_proj.bias"
            v_bias = prefix + "self_attn.v_proj.bias"
            self.layers.append({
                "ln1": tensor(prefix + "input_layernorm.weight"),
                "ln2": tensor(prefix + "post_attention_layernorm.weight"),
                "q": tensor(prefix + "self_attn.q_proj.weight"),
                "k": tensor(prefix + "self_attn.k_proj.weight"),
                "v": tensor(prefix + "self_attn.v_proj.weight"),
                "q_bias": tensor(q_bias) if q_bias in state else None,
                "k_bias": tensor(k_bias) if k_bias in state else None,
                "v_bias": tensor(v_bias) if v_bias in state else None,
                "o": tensor(prefix + "self_attn.o_proj.weight"),
                "gate": tensor(prefix + "mlp.gate_proj.weight"),
                "up": tensor(prefix + "mlp.up_proj.weight"),
                "down": tensor(prefix + "mlp.down_proj.weight"),
            })

    def _rope(self, length: int):
        dim = self.head_dim
        inv = 1.0 / (
            self.rope_theta ** (
                torch.arange(0, dim, 2, device=self.device, dtype=torch.float32)
                / dim
            )
        )
        pos = torch.arange(length, device=self.device, dtype=torch.float32)
        freqs = torch.outer(pos, inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None].to(self.dtype)
        sin = emb.sin()[None, None].to(self.dtype)
        return cos, sin

    @staticmethod
    def _apply_rope(x, cos, sin):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin

    def forward(self, ids, h_override=None, keep_last=False):
        """Return logits for a one-dimensional sequence of token IDs."""
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        length = ids.shape[0]
        hidden = self.wte[ids]
        if h_override is not None:
            hidden = h_override
        cos, sin = self._rope(length)
        for layer in self.layers:
            x = _rms_norm(hidden, layer["ln1"], self.eps)
            q = x @ layer["q"].T
            k = x @ layer["k"].T
            v = x @ layer["v"].T
            if layer["q_bias"] is not None:
                q = q + layer["q_bias"]
                k = k + layer["k_bias"]
                v = v + layer["v_bias"]
            q = q.view(1, length, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(1, length, self.n_kv, self.head_dim).transpose(1, 2)
            v = v.view(1, length, self.n_kv, self.head_dim).transpose(1, 2)
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)
            if self.n_kv != self.n_head:
                repeats = self.n_head // self.n_kv
                k = k.repeat_interleave(repeats, dim=1)
                v = v.repeat_interleave(repeats, dim=1)
            attn = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )
            attn = attn.transpose(1, 2).reshape(1, length, self.n_embd)
            hidden = hidden + (attn @ layer["o"].T)
            y = _rms_norm(hidden, layer["ln2"], self.eps)
            y = torch.nn.functional.silu(y @ layer["gate"].T) * (y @ layer["up"].T)
            hidden = hidden + (y @ layer["down"].T)
        hidden = _rms_norm(hidden, self.norm_w, self.eps)
        if keep_last:
            return (hidden[0, -1:] @ self.lm_head_w.T).float()
        return (hidden[0] @ self.lm_head_w.T).float()

    @torch.no_grad()
    def generate(self, ids, n_new=20, temperature=0.0, seed=0, max_ctx=2048):
        ids = list(ids)
        for _ in range(n_new):
            window = ids[-max_ctx:]
            logits = self.forward(
                torch.tensor(window, dtype=torch.long, device=self.device)
            )
            next_token = logits[-1]
            if temperature > 0:
                torch.manual_seed(seed)
                probs = torch.softmax(next_token / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = next_token.argmax(dim=-1)
            ids.append(int(next_token))
        return ids
