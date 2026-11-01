"""
superfluid.torch_graft — the same memory layer, in PyTorch, for a REAL LLM.
This is the GPU-scale twin of `superfluid/graft.py`.  It exists because the
numpy implementation cannot be attached to a pretrained transformer: it
needs a tied embedding table to turn a stored token id back into a
vector, and it needs the model's own scales.
WHAT IS THE SAME AS THE NUMPY VERSION
the two tiers: a lossy (diffusion) state and an exact, content-keyed
store of token ids;
the attach-is-a-no-op property: `Wo_exact` and `Wo_lossy` start at ZERO,
so attaching the branch leaves the logits bit-identical;
the read gain and the contrast transform (`pc1`), because with a tied
readout a raw embedding injection boosts the token's embedding-space
neighbours too;
causality: the store is written from the pair (context ending at t-1,
token at t), so a token is never read back before it was written.
WHAT IS DIFFERENT
the state lives in a torch tensor (it can be a parameter, a buffer, or
just an attribute);
the writes are done under `no_grad` -- the layer is a MEMORY, and the
exactness of the tier does not depend on differentiating through it;
the contrast transform is computed once from the model's embedding table
(on the right device) and can be refreshed after the model is trained.
NOT EXECUTED IN THIS SANDBOX
This module has NOT been run here: the development environment has no
torch, no GPU and 3 GB of RAM.  It is written to be read, reviewed and
run on a GPU box.  `experiments/gpu_graft.py` is the driver.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SAABranchTorch", "ctx_hash"]


def ctx_hash(tokens) -> int:
    """FNV-1a 64-bit hash of a token sequence (the content key)."""
    h = 0xCBF29CE484222325
    for t in tokens:
        h ^= int(t) & 0xFFFFFFFFFFFFFFFF
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


class SAABranchTorch(nn.Module):
    """The attachable memory layer (torch).

    Parameters
     ----------
     M, r, d   : anchors, rank, model width
     emb       : the model's input embedding matrix (tied or not).  Used to
                 turn a stored token id back into a vector.
     ctx_len   : how many trailing tokens form the content key
     capacity  : max entries in the exact store
     read_gain : the scalar gain of the exact injection (CALIBRATE it; see
                 the gain sweep in results/e6_graft.json)
     transform : "raw" | "center" | "pc1" -- the selectivity of the injection
     """

    def __init__(self, M: int = 64, r: int = 8, d: int = 64, emb=None,
                 ctx_len: int = 1, capacity: int = 8192, read_gain: float = 300.0,
                 transform: str = "pc1", use_exact: bool = True,
                 use_lossy: bool = False, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.M, self.r, self.d = int(M), int(r), int(d)
        self.ctx_len = int(ctx_len)
        self.capacity = int(capacity)
        self.read_gain = float(read_gain)
        self.transform = str(transform)
        self.use_exact, self.use_lossy = bool(use_exact), bool(use_lossy)
        self.protos = nn.Parameter(torch.randn(M, d, generator=g) * d ** -0.5)
        self.Wq = nn.Parameter(torch.randn(d, d, generator=g) * d ** -0.5)
        self.Wk = nn.Parameter(torch.randn(d, d, generator=g) * d ** -0.5)
        self.Wv = nn.Parameter(torch.randn(d, r, generator=g) * d ** -0.5)
        self.Wr = nn.Parameter(torch.randn(r, d, generator=g) * d ** -0.5)
        # ZERO => attaching the branch is an EXACT identity
        self.Wo_exact = nn.Parameter(torch.zeros(d, d))
        self.Wo_lossy = nn.Parameter(torch.zeros(d, d))
        self.emb = emb          # NOT a parameter: it belongs to the model
        self.state = torch.zeros(M, r)
        self.store: dict[int, int] = {}
        self.store_order: list[int] = []
        self._buf: list[int] = []
        self._contrast = None

    # ------------------------------------------------------------- lifecycle
    def reset(self):
        self.state = torch.zeros_like(self.state)
        self.store.clear()
        self.store_order.clear()
        self._buf.clear()

    def refresh_contrast(self):
        """Recompute the (optionally transformed) embedding table."""
        self._contrast = None

    def _contrast_table(self, device, dtype):
        if self._contrast is None:
            E = self.emb.weight if hasattr(self.emb, "weight") else self.emb
            E = E.detach().to(device=device, dtype=torch.float32)
            t = self.transform
            if t == "center":
                E = E - E.mean(0, keepdim=True)
            elif t.startswith("pc"):
                k = int(t[2:] or 1)
                # the SVD is done ONCE, on the right device
                _, _, Vt = torch.linalg.svd(E, full_matrices=False)
                for j in range(min(k, Vt.shape[0])):
                    p = Vt[j]
                    E = E - torch.outer(E @ p, p)
            self._contrast = E.to(dtype=dtype)
        return self._contrast

    def state_bytes(self) -> int:
        return self.state.numel() * 4 + len(self.store) * 20

    # ----------------------------------------------------------------- maths
    def _route(self, content: torch.Tensor):
        """content (T, d) -> soft top-k weights (T, M) and the argmax (T,)."""
        q = content @ self.Wq
        kk = self.protos @ self.Wk
        sc = (q @ kk.T) / (self.d ** 0.5)
        k = min(4, self.M)
        top = sc.topk(k, dim=-1).indices
        w = torch.zeros_like(sc)
        s = sc.gather(1, top)
        s = s - s.max(dim=-1, keepdim=True).values
        e = torch.softmax(s, dim=-1)
        w.scatter_(1, top, e)
        return w, top[:, 0]

    def _key(self, buf):
        if len(buf) < self.ctx_len:
            return None
        if self.ctx_len == 1:
            return int(buf[-1])
        return ctx_hash(buf[-self.ctx_len:])

    def _put(self, key: int, value: int):
        if key in self.store:
            return
        if len(self.store) >= self.capacity:
            self.store.pop(self.store_order.pop(0), None)
        self.store[key] = int(value)
        self.store_order.append(int(key))

    # ---------------------------------------------------------------- forward
    def forward(self, h: torch.Tensor, content: torch.Tensor,
                token_ids: torch.Tensor):
        """h, content: (T, d);  token_ids: (T,) int64."""
        T = token_ids.shape[0]
        ids = token_ids.detach().cpu().tolist()
        out_e = None
        hits = 0
        if self.use_exact and self.emb is not None:
            C = self._contrast_table(h.device, h.dtype)
            rows = torch.zeros(T, self.d, device=h.device, dtype=h.dtype)
            buf = list(self._buf)
            for t in range(T):
                prev_key = self._key(buf)
                buf.append(int(ids[t]))
                if len(buf) > self.ctx_len:
                    buf = buf[-self.ctx_len:]
                key = self._key(buf)
                v = self.store.get(key, -1)
                if v >= 0:
                    rows[t] = C[v] * self.read_gain
                    hits += 1
                if prev_key is not None:
                    self._put(prev_key, int(ids[t]))
            self._buf = buf
            out_e = rows

        if self.use_lossy:
            with torch.no_grad():
                w, _ = self._route(content.detach())
                self.state = self.state + w.T @ (h.detach() @ self.Wv.detach())
                self.state = 0.9 * self.state        # a simple decay
            out_l = (w @ self.state) @ self.Wr
        else:
            out_l = None

        h_out = h
        if out_e is not None:
            h_out = h_out + out_e @ self.Wo_exact
        if out_l is not None:
            h_out = h_out + out_l @ self.Wo_lossy
        return h_out, {"hits": hits}
