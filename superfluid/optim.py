"""
superfluid.optim — Adam, implemented locally (no torch in this environment).
"""

from __future__ import annotations
import numpy as np


class Adam:
    """Adam with decoupled weight decay and gradient clipping."""

    def __init__(self, params: dict, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0, clip: float = 1.0):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.clip = clip
        self.t = 0
        # params: name -> Tensor  (Tensor has .data and .grad)
        self.m = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.v = {k: np.zeros_like(v.data) for k, v in params.items()}

    def zero_grad(self):
        for p in self.params.values():
            p.grad = np.zeros_like(p.data)

    def step(self, lr: float | None = None):
        self.t += 1
        lr = self.lr if lr is None else lr
        b1c = 1 - self.b1 ** self.t
        b2c = 1 - self.b2 ** self.t
        total_sq = 0.0
        for p in self.params.values():
            g = p.grad
            total_sq += float(np.sum(g * g))
        norm = np.sqrt(total_sq)
        scale = 1.0
        if self.clip and norm > self.clip:
            scale = self.clip / (norm + 1e-12)
        for k, p in self.params.items():
            g = p.grad * scale
            if self.wd:
                g = g + self.wd * p.data
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / b1c
            vhat = self.v[k] / b2c
            p.data -= lr * mhat / (np.sqrt(vhat) + self.eps)

    def lr_at(self, step: int, total: int, warmup: int = 200, floor: float = 0.1):
        if step < warmup:
            return self.lr * (step + 1) / warmup
        p = (step - warmup) / max(1, total - warmup)
        return self.lr * (floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * min(p, 1.0))))
