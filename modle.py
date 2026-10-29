"""
superfluid.model — SuperfluidLM and a param-matched transformer baseline.
Both models share: embedding, learned absolute positions, RMSNorm pre-norm
residual blocks, tied readout.  The only difference is the block body, so the
A/B in `train.py` isolates exactly one thing: dense attention vs. the
topological fluid.
"""
from __future__ import annotations

import numpy as np

from .autodiff import Tensor, softmax_rows, concat
from .blocks import SuperfluidBlock, RMSNorm

__all__ = ["SuperfluidLM", "TinyTransformer"]


def _collect(module, prefix, out):
    for name, val in vars(module).items():
        if isinstance(val, Tensor):
            out[f"{prefix}.{name}"] = val
    return out


class SuperfluidLM:
    def __init__(self, vocab: int, d: int = 64, n_layers: int = 2, n_ctx: int = 128,
                 seed: int = 0, **block_kw):
        rng = np.random.default_rng(seed)
        self.vocab, self.d, self.n_ctx = vocab, d, n_ctx
        self.emb = Tensor(rng.normal(0, 0.02, (vocab, d)))
        self.pos = Tensor(rng.normal(0, 0.01, (n_ctx, d)))
        self.blocks = [SuperfluidBlock(d, seed=seed + 17 * (i + 1), **block_kw)
                       for i in range(n_layers)]
        self.norm = RMSNorm(d)

    def forward(self, idx: np.ndarray) -> Tensor:
        idx = np.asarray(idx, dtype=np.int64)
        h = self.emb[idx] + self.pos[: idx.shape[0]]
        for b in self.blocks:
            h = b(h)
        h = self.norm(h)
        return h @ self.emb.T

    def forward_batch(self, x):
        """x: (B, T) token ids -> logits (B*T, V).  The block is sequence-local
        (the diffusion mixes positions inside one sequence), so we loop."""
        outs = [self.forward(x[b]) for b in range(x.shape[0])]
        return concat(outs, axis=0)

    def params(self) -> dict:
        p = {}
        _collect(self, "emb", p)
        _collect(self, "pos", p)
        for i, b in enumerate(self.blocks):
            _collect(b, f"b{i}", p)
            _collect(b.norm, f"b{i}.norm", p)
        _collect(self.norm, "norm", p)
        return p


class TinyTransformer:
    """Post-GPT transformer block (causal MHA + MLP), numpy/autodiff."""

    def __init__(self, vocab: int, d: int = 64, n_layers: int = 2, n_ctx: int = 128,
                 n_heads: int = 4, d_ff: int | None = None, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.vocab, self.d, self.n_ctx = vocab, d, n_ctx
        self.n_heads = n_heads
        self.d_ff = d_ff or 4 * d
        self.emb = Tensor(rng.normal(0, 0.02, (vocab, d)))
        self.pos = Tensor(rng.normal(0, 0.01, (n_ctx, d)))
        self.layers = []
        for _ in range(n_layers):
            s = d ** -0.5
            self.layers.append({
                "Wq": Tensor(rng.normal(0, s, (d, d))),
                "Wk": Tensor(rng.normal(0, s, (d, d))),
                "Wv": Tensor(rng.normal(0, s, (d, d))),
                "Wo": Tensor(rng.normal(0, s, (d, d))),
                "W1": Tensor(rng.normal(0, (d * self.d_ff) ** -0.5, (d, self.d_ff))),
                "W2": Tensor(rng.normal(0, (d * self.d_ff) ** -0.5, (self.d_ff, d))),
                "n1": RMSNorm(d), "n2": RMSNorm(d),
            })
        self.norm = RMSNorm(d)

    def forward(self, idx: np.ndarray) -> Tensor:
        idx = np.asarray(idx, dtype=np.int64)
        n = idx.shape[0]
        h = self.emb[idx] + self.pos[:n]
        return self.forward_from_h(h)

    def forward_from_h(self, h: Tensor, return_hidden: bool = False):
        """Continue the forward pass from a (possibly MODIFIED) residual stream.

        This is the hook a grafted memory layer needs: the branch rewrites `h`
        and the frozen base model processes the rest.  Splitting `forward` into
        "embed" + "forward_from_h" is what makes a *training-free* attachment
        to an already-trained model possible at all.
        """
        n = h.data.shape[0]
        causal = np.triu(np.ones((n, n), dtype=bool), k=1)
        H, dh = self.n_heads, self.d // self.n_heads
        for L in self.layers:
            x = L["n1"](h)
            q = x @ L["Wq"]
            k = x @ L["Wk"]
            v = x @ L["Wv"]
            outs = []
            for hh in range(H):
                sl = slice(hh * dh, (hh + 1) * dh)
                qh = q[:, sl]
                kh = k[:, sl]
                vh = v[:, sl]
                sc = (qh @ kh.T) * (dh ** -0.5)
                a = softmax_rows(sc, mask=causal)
                outs.append(a @ vh)
            o = np.concatenate([o.data for o in outs], axis=1)
            oT = Tensor(o)
            h = h + oT @ L["Wo"]
            y = L["n2"](h)
            h = h + (y @ L["W1"]).gelu() @ L["W2"]
        h = self.norm(h)
        logits = h @ self.emb.T
        return (logits, h) if return_hidden else logits

    def forward_batch(self, x):
        outs = [self.forward(x[b]) for b in range(x.shape[0])]
        return concat(outs, axis=0)

    def params(self) -> dict:
        p = {}
        _collect(self, "emb", p)
        _collect(self, "pos", p)
        for i, L in enumerate(self.layers):
            for k, v in L.items():
                if isinstance(v, Tensor):
                    p[f"L{i}.{k}"] = v
            p[f"L{i}.n1.g"] = L["n1"].g
            p[f"L{i}.n2.g"] = L["n2"].g
        _collect(self.norm, "norm", p)
        return p