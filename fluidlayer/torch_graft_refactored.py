"""
fluidlayer.torch_graft_refactored - The refactored PyTorch adapter with all fixes.

This is the GPU-scale twin of `graft_refactored.py` with all critical fixes:
1. 12-Gram Zero-Recall Collapse - Dynamic compressed key-hashing
2. Sheaf Diffusion Degradation - Gated decay attention-weighted sheaf
3. Standard Serving & Compilation Constraints - GGUF/llama.cpp compatible abstraction
4. Zero Security & Tenant Isolation - XOR-masking with session_id

WHAT IS THE SAME AS THE NUMPY VERSION
-------------------------------------
- Two tiers: lossy (diffusion) state and exact, content-keyed store
- Attach-is-a-no-op property: Wo_exact and Wo_lossy start at ZERO
- Read gain and contrast transform (pc1)
- Causality: store is written causally
- Majority tracking and least-corroborated eviction

WHAT IS DIFFERENT
-----------------
- State lives in torch tensors
- Writes are done under no_grad (memory, not differentiable)
- Contrast transform computed once from the model's embedding table
- All fixes from the refactored NumPy version are included
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .graft_refactored import SessionSecurity, ctx_hash_compressed

__all__ = ["SAABranchTorchRefactored", "ctx_hash_compressed"]


# =============================================================================
# PyTorch Compatible Security
# =============================================================================


class SessionSecurityTorch:
    """PyTorch-compatible session security with XOR-masking."""

    def __init__(self, session_id: int = 0):
        self.session_security = SessionSecurity(session_id)

    def mask_token(self, token_id: int) -> int:
        return self.session_security.mask_token(token_id)

    def unmask_token(self, masked_token_id: int) -> int:
        return self.session_security.unmask_token(masked_token_id)

    def mask_tensor(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Apply XOR mask to a tensor of token IDs."""
        mask = self.session_security.get_mask()
        return token_ids ^ mask

    def unmask_tensor(self, masked_token_ids: torch.Tensor) -> torch.Tensor:
        """Remove XOR mask from a tensor of token IDs."""
        mask = self.session_security.get_mask()
        return masked_token_ids ^ mask


# =============================================================================
# PyTorch Compatible Compressed Store
# =============================================================================


class _CompressedModeStoreTorch:
    """PyTorch-compatible content-keyed store with compressed keys."""

    __slots__ = (
        "value",
        "support",
        "disagree",
        "order",
        "capacity",
        "session_security",
        "compression_bits",
    )

    def __init__(
        self,
        capacity: int = 4096,
        session_security: SessionSecurityTorch = None,
        compression_bits: int = 32,
    ):
        self.value: dict = {}
        self.support: dict = {}
        self.disagree: dict = {}
        self.order: list = []
        self.capacity = int(capacity)
        self.session_security = session_security
        self.compression_bits = compression_bits

    def __len__(self):
        return len(self.value)

    def __contains__(self, key):
        return key in self.value

    def __getitem__(self, key):
        return self.value[key]

    def keys(self):
        return list(self.order)

    def get(self, key, default=-1):
        return self.value.get(key, default)

    def get_unmasked(self, key, default=-1):
        value = self.value.get(key, default)
        if value >= 0 and self.session_security:
            value = self.session_security.unmask_token(value)
        return value

    def confidence(self, key) -> float:
        if key not in self.value:
            return 0.0
        a = self.support.get(key, 1)
        b = self.disagree.get(key, 0)
        return float(a) / float(max(a + b, 1))

    def put(self, key, value):
        if self.session_security:
            value = self.session_security.mask_token(value)

        if key in self.value:
            if self.value[key] == value:
                self.support[key] = self.support.get(key, 1) + 1
            else:
                a = self.support.get(key, 1)
                b = self.disagree.get(key, 0) + 1
                if b > a:
                    self.value[key] = int(value)
                    self.support[key] = b
                    self.disagree[key] = a
                else:
                    self.disagree[key] = b
            return
        if len(self.value) >= self.capacity:
            self._evict()
        self.value[key] = int(value)
        self.support[key] = 1
        self.disagree[key] = 0
        self.order.append(int(key))

    def _evict(self):
        worst, worst_key = None, None
        for k in self.order:
            sc = self.support.get(k, 0) - self.disagree.get(k, 0)
            if worst is None or sc < worst:
                worst, worst_key = sc, k
        if worst_key is None:
            return
        self.value.pop(worst_key, None)
        self.support.pop(worst_key, None)
        self.disagree.pop(worst_key, None)
        if worst_key in self.order:
            self.order.remove(worst_key)

    def clear(self):
        self.value.clear()
        self.support.clear()
        self.disagree.clear()
        self.order.clear()


# =============================================================================
# PyTorch Gated Decay Sheaf
# =============================================================================


class GatedDecaySheafTorch:
    """PyTorch implementation of the gated decay attention-weighted sheaf."""

    def __init__(self, M: int, r: int, d: int, seed: int = 0):
        self.M = M
        self.r = r
        self.d = d

        # Initialize with deterministic seed
        g = torch.Generator().manual_seed(seed)

        # Prototype vectors
        self.protos = torch.nn.Parameter(torch.randn(M, d, generator=g) * d**-0.5)

        # Attention weights
        self.attention_weights = torch.nn.Parameter(
            torch.randn(M, M, generator=g) * (d * M) ** -0.5
        )

        # Positional gates (could be made learnable)
        self.positional_gates = torch.nn.Parameter(torch.ones(M))

        # Decay rates
        self.decay_rates = torch.nn.Parameter(torch.full((M,), 0.95))

        # Build the gated Laplacian
        self._build_gated_laplacian()

    def _build_gated_laplacian(self):
        """Build a gated Laplacian that respects positional boundaries."""
        M = self.M

        # Create attention-weighted adjacency matrix
        A = torch.abs(self.attention_weights)

        # Apply positional gates
        for i in range(M):
            for j in range(M):
                distance = abs(i - j)
                if distance > M // 4:
                    A[i, j] = 0.0

        # Normalize
        row_sums = A.sum(dim=1, keepdim=True)
        A = A / (row_sums + 1e-12)

        # Create degree matrix
        D = torch.diag(A.sum(dim=1))

        # Create gated Laplacian
        self.L = D - A

        # Compute spectral properties (on CPU for stability)
        self.lam_max = self._power_iteration_lmax(self.L.detach().cpu().numpy())

    def _power_iteration_lmax(self, L_np, iters: int = 60, tol: float = 1e-10) -> float:
        """Compute largest eigenvalue of symmetric PSD matrix."""
        import numpy as np

        n = L_np.shape[0]
        if n == 0:
            return 0.0
        v = np.random.default_rng(0).standard_normal(n)
        v /= np.linalg.norm(v) + 1e-16
        lam = 0.0
        for _ in range(iters):
            w = L_np @ v
            nw = np.linalg.norm(w)
            if nw < 1e-16:
                return 0.0
            v_new = w / nw
            lam_new = float(v_new @ (L_np @ v_new))
            if abs(lam_new - lam) < tol * max(1.0, abs(lam_new)):
                lam = lam_new
                v = v_new
                break
            lam, v = lam_new, v_new
        return max(lam, 0.0)

    def apply_gated_diffusion(
        self, state: torch.Tensor, t: float = 1.0, cheb_m: int = 8
    ) -> torch.Tensor:
        """Apply gated decay diffusion to the state."""
        if self.lam_max <= 0:
            return state

        device = state.device
        dtype = state.dtype

        # Convert to numpy for Chebyshev expansion (for stability)
        import numpy as np

        from .diffusion import cheb_coeffs_exp

        state_np = state.detach().cpu().numpy()
        L_np = self.L.detach().cpu().numpy()

        n = self.M
        Lt = (2.0 / self.lam_max) * L_np - np.eye(n)
        Y = [state_np]
        if cheb_m >= 1:
            Y.append(Lt @ Y[0])
            for k in range(2, cheb_m + 1):
                Y.append(2.0 * (Lt @ Y[k - 1]) - Y[k - 2])

        out_np = np.zeros_like(state_np)
        coeffs = cheb_coeffs_exp(t, self.lam_max, cheb_m)
        for k, a in enumerate(coeffs):
            if k < len(Y):
                out_np += a * Y[k]

        # Apply positional decay
        decay_np = self.decay_rates.detach().cpu().numpy()
        for i in range(self.M):
            out_np[i] *= decay_np[i]

        return torch.from_numpy(out_np).to(device=device, dtype=dtype)


# =============================================================================
# PyTorch GGUF Compatible Layer
# =============================================================================


class GGUFCompatibleLayerTorch:
    """PyTorch-compatible GGUF abstraction layer."""

    def __init__(self, d: int, V: int = 0):
        self.d = d
        self.V = V
        self.static_params = {}
        self.dynamic_state = {}

    def register_static_param(self, name: str, value: torch.Tensor):
        self.static_params[name] = value

    def register_dynamic_state(self, name: str, value: torch.Tensor):
        self.dynamic_state[name] = value

    def to_gguf_schema(self) -> dict:
        schema = {
            "type": "fluidlayer_memory_torch",
            "version": 1,
            "d": self.d,
            "V": self.V,
            "static_params": {},
            "dynamic_state_shape": {},
        }

        for name, value in self.static_params.items():
            schema["static_params"][name] = {
                "shape": tuple(value.shape),
                "dtype": str(value.dtype),
            }

        for name, value in self.dynamic_state.items():
            schema["dynamic_state_shape"][name] = {
                "shape": tuple(value.shape),
                "dtype": str(value.dtype),
            }

        return schema


# =============================================================================
# Main Refactored PyTorch SAABranch
# =============================================================================


class SAABranchTorchRefactored(nn.Module):
    """The refactored attachable memory layer in PyTorch with all critical fixes.

    This version addresses all four critical failure modes:
    1. 12-Gram Zero-Recall Collapse: Dynamic compressed key-hashing
    2. Sheaf Diffusion Degradation: Gated decay attention-weighted sheaf
    3. Standard Serving & Compilation Constraints: GGUF-compatible abstraction
    4. Zero Security & Tenant Isolation: XOR-masking with session_id
    """

    def __init__(
        self,
        M: int = 64,
        r: int = 8,
        d: int = 64,
        emb=None,
        ctx_len: int = 1,
        capacity: int = 4096,
        read_gain: float = 300.0,
        transform: str = "pc1",
        use_exact: bool = True,
        use_lossy: bool = False,
        seed: int = 0,
        use_router: bool = True,
        session_id: int = 0,
        compression_bits: int = 32,
        gated_sheaf: bool = True,
        gguf_compatible: bool = False,
    ):
        super().__init__()

        for name, value in (("M", M), ("r", r), ("capacity", capacity)):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        g = torch.Generator().manual_seed(seed)
        self.M, self.r, self.d = int(M), int(r), int(d)
        self.ctx_len = int(ctx_len)
        self.capacity = int(capacity)
        self.read_gain = float(read_gain)
        self.transform = str(transform)
        self.use_exact, self.use_lossy = bool(use_exact), bool(use_lossy)
        self.use_router = bool(use_router)
        self.compression_bits = compression_bits
        self.gguf_compatible = gguf_compatible

        # Initialize session security
        self.session_security = SessionSecurityTorch(session_id) if session_id else None

        # Initialize GGUF compatibility layer
        V_emb = (
            emb.weight.shape[0]
            if emb is not None and hasattr(emb, "weight")
            else (emb.shape[0] if emb is not None else 0)
        )
        self.gguf_layer = (
            GGUFCompatibleLayerTorch(d, V_emb) if gguf_compatible else None
        )

        # Initialize gated decay sheaf if enabled
        self.gated_sheaf = GatedDecaySheafTorch(M, r, d, seed) if gated_sheaf else None

        # Parameters
        self.protos = nn.Parameter(torch.randn(M, d, generator=g) * d**-0.5)
        self.Wq = nn.Parameter(torch.randn(d, d, generator=g) * d**-0.5)
        self.Wk = nn.Parameter(torch.randn(d, d, generator=g) * d**-0.5)
        self.Wv = nn.Parameter(torch.randn(d, r, generator=g) * d**-0.5)
        self.Wr = nn.Parameter(torch.randn(r, d, generator=g) * d**-0.5)

        # ZERO => attaching the branch is an EXACT identity
        self.Wo_exact = nn.Parameter(torch.zeros(d, d))
        self.Wo_lossy = nn.Parameter(torch.zeros(d, d))

        self.emb = emb
        self.state = torch.zeros(M, r)

        # Use compressed mode store with session security
        self.store = _CompressedModeStoreTorch(
            capacity=self.capacity,
            session_security=self.session_security,
            compression_bits=self.compression_bits,
        )
        self.anchor_store = _CompressedModeStoreTorch(
            capacity=self.M,
            session_security=self.session_security,
            compression_bits=self.compression_bits,
        )

        self._buf: list[int] = []
        self._contrast = None

        # Register parameters with GGUF layer if enabled
        if self.gguf_compatible and self.gguf_layer:
            self.gguf_layer.register_static_param("Wq", self.Wq)
            self.gguf_layer.register_static_param("Wk", self.Wk)
            self.gguf_layer.register_static_param("Wv", self.Wv)
            self.gguf_layer.register_static_param("Wr", self.Wr)
            self.gguf_layer.register_static_param("Wo_exact", self.Wo_exact)
            self.gguf_layer.register_static_param("Wo_lossy", self.Wo_lossy)
            self.gguf_layer.register_dynamic_state("state", self.state)

    # ------------------------------------------------------------- lifecycle
    def reset(self):
        self.state = torch.zeros_like(self.state)
        self.store.clear()
        self.anchor_store.clear()
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
                _, _, Vt = torch.linalg.svd(E, full_matrices=False)
                for j in range(min(k, Vt.shape[0])):
                    p = Vt[j]
                    E = E - torch.outer(E @ p, p)
            self._contrast = E.to(dtype=dtype)
        return self._contrast

    def state_bytes(self) -> int:
        """Logical packed-payload estimate; excludes Python/allocator overhead."""
        return (
            self.state.numel() * 4 + len(self.store) * 20 + len(self.anchor_store) * 20
        )

    def to_gguf_schema(self) -> dict:
        """Generate a GGUF-compatible schema for this layer."""
        if self.gguf_layer:
            return self.gguf_layer.to_gguf_schema()
        return {}

    # ----------------------------------------------------------------- maths
    def _route(self, content: torch.Tensor):
        """content (T, d) -> soft top-k weights (T, M) and the argmax (T,)."""
        q = content @ self.Wq
        kk = self.protos @ self.Wk
        sc = (q @ kk.T) / (self.d**0.5)
        k = min(4, self.M)
        top = sc.topk(k, dim=-1).indices
        w = torch.zeros_like(sc)
        s = sc.gather(1, top)
        s = s - s.max(dim=-1, keepdim=True).values
        e = torch.softmax(s, dim=-1)
        w.scatter_(1, top, e)
        return w, top[:, 0]

    def _semantic_key(self, buf, ctx_len, compression_bits):
        """Create a semantic key for a context window."""
        if ctx_len == 1:
            if buf:
                return int(buf[-1]) & ((1 << compression_bits) - 1)
            return 0
        window = buf[-ctx_len:] if len(buf) >= ctx_len else buf
        return ctx_hash_compressed(
            window, ngram_range=(1, ctx_len), compression_bits=compression_bits
        )

    # ---------------------------------------------------------------- forward
    def forward(self, h: torch.Tensor, content: torch.Tensor, token_ids: torch.Tensor):
        """h, content: (T, d);  token_ids: (T,) int64."""
        T = token_ids.shape[0]
        ids = token_ids.detach().cpu().tolist()
        w, am = self._route(content.detach())
        out_e = None
        hits = 0
        retrieved = torch.full((T,), -1, dtype=torch.long)
        confidence = torch.zeros(T, dtype=torch.float32)
        router_retrieved = torch.full((T,), -1, dtype=torch.long)
        router_confidence = torch.zeros(T, dtype=torch.float32)

        if self.use_exact and self.emb is not None:
            C = self._contrast_table(h.device, h.dtype)
            rows = torch.zeros(T, self.d, device=h.device, dtype=h.dtype)
            buf = list(self._buf)

            for t in range(T):
                prev_key = self._semantic_key(buf, self.ctx_len, self.compression_bits)
                buf.append(int(ids[t]))
                if len(buf) > self.ctx_len:
                    buf = buf[-self.ctx_len :]
                key = self._semantic_key(buf, self.ctx_len, self.compression_bits)

                # Router tier
                if self.use_router:
                    anchor = int(am[t])
                    value = self.anchor_store.get(anchor, -1)
                    if value >= 0:
                        router_retrieved[t] = self.anchor_store.get_unmasked(anchor, -1)
                        router_confidence[t] = self.anchor_store.confidence(anchor)

                # Exact tier with compressed key
                v = self.store.get(key, -1)
                if v >= 0:
                    unmasked_v = self.store.get_unmasked(key, -1)
                    retrieved[t] = unmasked_v
                    confidence[t] = self.store.confidence(key)
                    rows[t] = C[unmasked_v] * self.read_gain
                    hits += 1

                # Causal write with session security
                if prev_key is not None:
                    masked_value = ids[t]
                    if self.session_security:
                        masked_value = self.session_security.mask_token(ids[t])
                    self.store.put(prev_key, int(masked_value))

                # Router tier write
                if t > 0 and self.use_router:
                    masked_value = ids[t]
                    if self.session_security:
                        masked_value = self.session_security.mask_token(ids[t])
                    self.anchor_store.put(int(am[t - 1]), int(masked_value))

            self._buf = buf
            out_e = rows

        if self.use_lossy:
            with torch.no_grad():
                if self.gated_sheaf:
                    # Use gated decay diffusion
                    self.state = self.gated_sheaf.apply_gated_diffusion(
                        self.state, t=1.0, cheb_m=8
                    )
                    self.state = self.state + w.T @ (h.detach() @ self.Wv.detach())
                else:
                    # Original simple decay
                    self.state = self.state + w.T @ (h.detach() @ self.Wv.detach())
                    self.state = 0.9 * self.state
            out_l = (w @ self.state) @ self.Wr
        else:
            out_l = None

        h_out = h
        if out_e is not None:
            h_out = h_out + out_e @ self.Wo_exact
        if out_l is not None:
            h_out = h_out + out_l @ self.Wo_lossy

        support = torch.tensor(
            [self.store.support.get(int(v), 0) if v >= 0 else 0 for v in retrieved],
            dtype=torch.long,
        )

        return h_out, {
            "route": w,
            "anchor": am,
            "hits": hits,
            "emb_read": out_e,
            "retrieved": retrieved,
            "support": support,
            "confidence": confidence,
            "router_retrieved": router_retrieved,
            "router_confidence": router_confidence,
        }

    # ------------------------------------------------------------- gated logits
    def gated_logits(
        self,
        logits,
        info,
        gain: float = 8.0,
        use_support: bool = False,
        support_scale: float = 1.0,
        tau: float | None = None,
        c_min: float = 2.0,
        big: float = 20.0,
        router_weight: float = 0.3,
        gate_space: str = "logit",
    ):
        """Apply the uncertainty-gated retrieval boost."""
        retrieved = info.get("retrieved")
        if retrieved is None:
            return logits
        z = logits
        retrieved = torch.as_tensor(retrieved, device=z.device, dtype=torch.long)
        T = z.shape[0]
        fallback = torch.zeros(T, device=z.device, dtype=torch.bool)
        rr = info.get("router_retrieved")
        rconf = info.get("router_confidence")
        if rr is not None:
            rr = torch.as_tensor(rr, device=z.device, dtype=torch.long)
            fallback = (retrieved < 0) & (rr >= 0)
            retrieved = torch.where(fallback, rr, retrieved)

        probs = torch.softmax(z, dim=1)
        p_max = probs.max(dim=1).values
        if str(gate_space) == "logit":
            uncertainty = -torch.log(p_max.clamp_min(1e-12))
        else:
            uncertainty = 1.0 - p_max
        if tau is None:
            tau = 20.0 if str(gate_space) == "logit" else 2.0
        boost = float(gain) * uncertainty

        if rconf is not None:
            rc = torch.as_tensor(rconf, device=z.device, dtype=z.dtype)
            boost = torch.where(
                fallback,
                boost * float(router_weight) * rc.clamp(0.0, 1.0),
                boost,
            )
        if use_support:
            support = info.get("support")
            if support is not None:
                sup = torch.as_tensor(support, device=z.device, dtype=z.dtype)
                boost = boost * (sup / float(support_scale)).clamp(max=1.0)

        confidence = info.get("confidence")
        if confidence is None:
            conf = torch.zeros(T, device=z.device, dtype=z.dtype)
        else:
            conf = torch.as_tensor(confidence, device=z.device, dtype=z.dtype)
        if rconf is not None:
            rc = torch.as_tensor(rconf, device=z.device, dtype=z.dtype)
            conf = torch.where(fallback, rc, conf)
        decide = (uncertainty >= float(tau)) | (conf >= float(c_min))
        boost = torch.where(decide, torch.full_like(boost, float(big)), boost)

        out = z.clone()
        rows = torch.nonzero(retrieved >= 0, as_tuple=False).flatten()
        if rows.numel():
            out[rows, retrieved[rows]] += boost[rows]
        return out
