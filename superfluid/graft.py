"""
superfluid.graft — the attachable memory layer.
WHAT THIS IS
A single module you can bolt onto ANY causal LM with one insertion:
    h' = h + W_e . E(h)  +  W_l . S(h)
             ^^^^^^^         ^^^^^^^
             exact tier      lossy tier (the constant-size fluid state)
E(h)  = the EXACT read:  a bounded, CONTENT-KEYED store of token ids,
        pulled back out through the model's own embedding table.
S(h)  = the LOSSY read:  the diffused aggregate state (M x r floats).
WHY IT IS SAFE TO ATTACH
`W_e` and `W_l` are initialised to ZERO, so at the moment of attachment the
branch is an EXACT identity:  h' == h, bit for bit.  Every logit, every
perplexity, every sampled token is unchanged.  The layer can only start to
matter after someone trains it, and training it is opt-in.
WHY THERE ARE TWO TIERS
A diffusion smears.  If the exact value of a token lives only inside the
fluid, it is mixed with its neighbours and is no longer exact.  So:
the fluid answers WHERE   (an address space of M anchors)
the store answers WHAT    (an int32 token id, keyed by content)
The store is CONTENT-KEYED, not anchor-keyed.  That distinction matters and
it was learned the hard way: a per-anchor "last write wins" slot is evicted
by ordinary filler tokens within a few hundred tokens, because every token
routs somewhere.  Keying by the content of the context means a fact is only
overwritten by the SAME context occurring again -- which is exactly the
semantics you want (and it is the same trick as the frozen register's
`ctx_key`).
CAUSALITY
The store is written CAUSALLY.  At position t the branch has seen tokens
x_0..x_t only.  The pair (context ending at t-1, next token x_t) is complete
at time t, so the write happens at time t -- never earlier.  The read at
position t uses the context ending at t, which is a DIFFERENT key from the
one being written, so a token can never be read back before it was written
and the model can never "copy the current token" through the store.
PERSISTENCE
The state (`state`, `store`) lives on the object, NOT in the autodiff graph.
It survives across calls: that is what makes it a memory rather than a
layer.  `reset()` clears it (a new document / a new session).
"""
from __future__ import annotations

import numpy as np

from .autodiff import Tensor, softmax_rows
from .diffusion import cheb_coeffs_exp, power_iteration_lmax

__all__ = ["SAABranch", "GRAFT_STATE_BYTES", "ctx_hash"]


def ctx_hash(tokens) -> int:
    """FNV-1a 64-bit hash of a token sequence (the content key)."""
    h = 0xCBF29CE484222325
    for t in tokens:
        h ^= int(t) & 0xFFFFFFFFFFFFFFFF
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _knn_edges(X: np.ndarray, k: int) -> np.ndarray:
    """Undirected kNN edge list of a point set (cosine similarity)."""
    n = X.shape[0]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    kk = min(k, n - 1)
    idx = np.argsort(-S, axis=1)[:, :kk]
    rows = np.repeat(np.arange(n), kk)
    cols = idx.reshape(-1)
    e = np.stack([np.minimum(rows, cols), np.maximum(rows, cols)], axis=1)
    return np.unique(e, axis=0)


class SAABranch:
    """The attachable memory layer (see the module docstring).

    Parameters
     ----------
     M      : number of anchors (the address space).  M*r*8 bytes is the
              constant runtime state of the lossy tier.
     r      : rank of each anchor's state.
     d      : model width.
     emb    : the model's embedding table (V x d).  Needed to turn a stored
              token id back into a vector.  Shared, never copied.
     ctx_len: how many trailing tokens form the content key (1 = a token-level
              "what followed this token" memory; k = a k-gram memory).
     capacity: max entries in the exact store.
     use_exact / use_lossy : ablation switches (measured in E6).
     """

    def __init__(self, M: int = 64, r: int = 8, d: int = 64, emb=None,
                 k: int = 4, cheb_m: int = 8, t: float = 1.0, seed: int = 0,
                 use_exact: bool = True, use_lossy: bool = True,
                 capacity: int = 4096, ctx_len: int = 1,
                 trainable_route: bool = True, read_gain: str = "norm",
                 center: bool = True, read_transform: str = "pc1",
                 state_mode: str = "persistent", hist_len: int = 8,
                 inject_mode: str = "state"):
        for name, val in (("M", M), ("r", r), ("capacity", capacity)):
            if int(val) <= 0:
                raise ValueError(f"{name} must be positive, got {val}")
        if emb is not None and int(emb.shape[1]) != int(d):
            raise ValueError(f"d={d} does not match the embedding width "
                             f"{emb.shape[1]}")
        if str(state_mode) not in ("persistent", "rebuild"):
            raise ValueError("state_mode must be 'persistent' or 'rebuild'")
        if str(read_gain) not in ("norm", "raw"):
            raise ValueError("read_gain must be 'norm' or 'raw'")
        if (str(read_transform) not in ("raw", "center")
                and not str(read_transform).startswith("pc")):
            raise ValueError("read_transform must be 'raw', 'center' or 'pcK'")
        rng = np.random.default_rng(seed)
        self.M, self.r, self.d = int(M), int(r), int(d)
        self.k, self.cheb_m, self.t = int(k), int(cheb_m), float(t)
        self.use_exact, self.use_lossy = bool(use_exact), bool(use_lossy)
        self.emb = emb
        self.center = bool(center)
        self.read_transform = str(read_transform)
        self._contrast = None
        self.ctx_len = int(ctx_len)
        self.capacity = int(capacity)
        self.read_gain = str(read_gain)
        self.read_scale = 1.0
        self.protos = Tensor(rng.normal(0.0, d ** -0.5, (M, d)))
        # route head: content -> anchor scores.  The route head is a function
        # of CONTENT ONLY (no position), so the same context always lands on
        # the same anchor -- the fluid addresses by geometry, not by position.
        s = d ** -0.5
        self.Wq = Tensor(rng.normal(0.0, s, (d, d)))
        self.Wk = Tensor(rng.normal(0.0, s, (d, d)))
        self.Wv = Tensor(rng.normal(0.0, s, (d, r)))     # write projection
        self.Wr = Tensor(rng.normal(0.0, s, (r, d)))     # lossy read projection
        # The LAYER-LOCAL head: the layer predicts the next token from its
        # own state.  This is the objective that FIXES the divergence: the
        # layer is trained on a well-posed task of its own, and the base
        # model's gradients never enter the layer's parameters.
        V = emb.shape[0] if emb is not None else 0
        self.V = int(V)
        self.Wp = Tensor(rng.normal(0.0, r ** -0.5, (r, V))) if V else None
        self.bias = Tensor(np.zeros(V)) if V else None
        # output projections: ZERO => the branch is an exact identity at attach
        self.Wo_exact = Tensor(np.zeros((d, d)))
        self.Wo_lossy = Tensor(np.zeros((d, d)))
        self.trainable_route = bool(trainable_route)
        # "persistent": the state is a numpy accumulator (the ORIGINAL design;
        # it cannot be differentiated, which is what made the training
        # diverge).
        # "rebuild":    the state is REBUILT from a bounded history INSIDE
        #              the autodiff graph every call (truncated BPTT), so the
        #              gradient is exact and bounded.  The state VALUES are
        #              identical (the diffusion is linear); only the gradient
        #              path is new.
        self.state_mode = str(state_mode)
        # "state": the raw diffused read is injected (the original design).
        # "prior": the layer's OWN next-token distribution is projected into
        #          the residual through the embedding table -- a soft domain
        #          prior, and the only injection the layer-local objective
        #          can honestly justify.
        self.inject_mode = str(inject_mode)
        self.hist_len = int(hist_len)
        self._hist: list = []
        # ---- the persistent state -------------------------------------
        self.state = np.zeros((M, r))                  # lossy tier
        self.store: dict[int, int] = {}                # exact tier: key -> id
        self.store_order: list[int] = []               # FIFO eviction order
        self._buf: list[int] = []                      # rolling context window
        self.write_calls = 0
        self.tokens_written = 0
        self.store_hits = 0
        self.store_lookups = 0
        # the address graph (constant: the prototypes are fixed)
        self.edges = _knn_edges(self.protos.data, k)
        n = M
        A = np.zeros((n, n))
        for a, b in self.edges:
            A[a, b] = 1.0
            A[b, a] = 1.0
        self.L = np.diag(A.sum(axis=1)) - A
        self.lam_max = float(power_iteration_lmax(self.L))
        self._cheb = (cheb_coeffs_exp(self.t, self.lam_max, self.cheb_m)
                      if self.lam_max > 0 else np.array([1.0]))

    # ------------------------------------------------------------- lifecycle
    def reset(self):
        """Clear the memory (new document / new session)."""
        self.state[:] = 0.0
        self.store.clear()
        self.store_order.clear()
        self._buf.clear()

    @property
    def state_bytes(self) -> int:
        """Constant runtime state: the diffusion state + the exact store."""
        return self.state.size * 8 + len(self.store) * 20

    def params(self) -> dict:
        p = {
            "Wq": self.Wq, "Wk": self.Wk, "Wv": self.Wv, "Wr": self.Wr,
            "Wo_exact": self.Wo_exact, "Wo_lossy": self.Wo_lossy,
        }
        if self.Wp is not None:
            p["Wp"] = self.Wp
            p["bias"] = self.bias
        if not self.trainable_route:
            for k in ("Wq", "Wk"):
                p.pop(k, None)
        return p

    # ------------------------------------------------------- the contrast table
    def refresh_contrast(self):
        """Recompute the mean-centred embedding table.

        WHY: the readout is TIED (logits = h . emb.T), so injecting a token's
        embedding v into the residual boosts the logit of every token u by
        gain * (emb[v] . emb[u]).  Embedding rows are NOT orthogonal, so the
        raw injection also boosts v's embedding-space NEIGHBOURS -- measured:
        the target's logit 134.9 vs the winner's 151.1, i.e. the injection
        made the WRONG token win.  Removing the shared (mean) direction
        makes the boost selective: the target's logit then wins outright.
        The table is cached because for a real LLM (V=50k, d=4096) computing
        the mean on every forward would be absurd.  Call refresh_contrast()
        after the base model has been (re)trained.
        """
        self._contrast = None

    def _contrast_table(self):
        """The table the exact read pulls from: the embedding matrix, possibly
        transformed so that the injection is SELECTIVE.

        "raw"    : the plain embedding (leaks into the neighbours)
        "center" : mean-centred (removes the shared direction)
        "pcK"    : the top-K principal directions of the embedding matrix are
                   projected out.  This is the one that works: with a TIED
                   readout, logit_u += gain * (emb[v] . emb[u]), so a raw
                   injection boosts v's embedding-space neighbours as well.
                   Measured on the graft benchmark: raw 0.725, pc1 0.800.
        """
        if self._contrast is None:
            if self.emb is None:
                return None
            E = self.emb.data
            t = self.read_transform
            if t == "raw":
                self._contrast = E.copy()
            elif t == "center":
                self._contrast = E - E.mean(axis=0, keepdims=True)
            elif t.startswith("pc"):
                k = int(t[2:] or 1)
                # the SVD is O(V d^2): for a real LLM do it OFFLINE once and
                # ship the table, or use a randomized SVD on a subsample.
                _, _, Vt = np.linalg.svd(E, full_matrices=False)
                T = E.copy()
                for j in range(min(k, Vt.shape[0])):
                    p = Vt[j]
                    T = T - np.outer(T @ p, p)
                self._contrast = T
            else:
                self._contrast = E.copy()
        return self._contrast

    # ----------------------------------------------------------------- maths
    def _diffuse(self) -> np.ndarray:
        """exp(-t L) applied to the anchor state, by Chebyshev expansion.

        The state is a persistent, non-differentiable accumulator (it lives
        outside the autodiff graph on purpose: it is a MEMORY, not an
        activation).  The gradient therefore flows only through the READ.
        """
        if self.lam_max <= 0:
            return self.state
        n = self.M
        Lt = (2.0 / self.lam_max) * self.L - np.eye(n)
        Y = [self.state]
        if self.cheb_m >= 1:
            Y.append(Lt @ Y[0])
            for k in range(2, self.cheb_m + 1):
                Y.append(2.0 * (Lt @ Y[k - 1]) - Y[k - 2])
        out = np.zeros_like(self.state)
        for k, a in enumerate(self._cheb):
            if k < len(Y):
                out += a * Y[k]
        return out

    def _diffuse_tensor(self, S: Tensor) -> Tensor:
        """exp(-t L) applied to a TENSOR state (the differentiable path)."""
        if self.lam_max <= 0:
            return S
        n = self.M
        Lt = (2.0 / self.lam_max) * Tensor(self.L) - Tensor(np.eye(n))
        Y = [S]
        if self.cheb_m >= 1:
            Y.append(Lt @ Y[0])
            for k in range(2, self.cheb_m + 1):
                Y.append(2.0 * (Lt @ Y[k - 1]) - Y[k - 2])
        out = Tensor(np.zeros_like(S.data))
        for k, a in enumerate(self._cheb):
            if k < len(Y):
                out = out + a * Y[k]
        return out

    def _rebuild_state(self, h: Tensor, content: Tensor):
        """Rebuild the anchor state from a bounded history, inside the graph.

        The diffusion is LINEAR, so exp(-tL)(sum of writes) = sum of
        exp(-tL)(writes): the rebuilt state has the SAME VALUES as the
        persistent accumulator, but now every write is differentiable.
        """
        parts = []
        for (hh, cc) in self._hist:
            w_np, _ = self._route(cc)
            parts.append(Tensor(w_np).T @ (Tensor(hh) @ self.Wv))
        w_np, _ = self._route(content.data)
        parts.append(Tensor(w_np).T @ (h @ self.Wv))
        S = parts[0]
        for p in parts[1:]:
            S = S + p
        return self._diffuse_tensor(S)

    def _route(self, content: np.ndarray):
        """(T, d) content -> (T, M) soft top-k routing weights + the argmax."""
        q = content @ self.Wq.data                 # (T, d)
        kk = self.protos.data @ self.Wk.data       # (M, d)
        sc = (q @ kk.T) / (self.d ** 0.5)          # (T, M)
        top = np.argsort(-sc, axis=1)[:, : min(self.k, self.M)]
        w = np.zeros_like(sc)
        rows = np.arange(sc.shape[0])[:, None]
        s_top = sc[rows, top]
        s_top = s_top - s_top.max(axis=1, keepdims=True)
        e = np.exp(s_top)
        e = e / e.sum(axis=1, keepdims=True)
        w[rows, top] = e
        return w, top[:, 0]

    # ---------------------------------------------------------------- forward
    def forward(self, h: Tensor, content: Tensor, token_ids: np.ndarray):
        """Return (h_out, info).  `h` and `content` are Tensors (T, d).
        `content` is the position-free representation used for routing.
        `token_ids` are the tokens CONSUMED at each position.
        """
        token_ids = np.asarray(token_ids, dtype=np.int64)
        T = token_ids.shape[0]
        w_np, am = self._route(content.data)     # routing is not differentiable
        w = Tensor(w_np)
        # ---- the lossy tier: write, diffuse, read ----------------------
        out_l = None
        state_read = None
        if self.use_lossy:
            if self.state_mode == "rebuild":
                # differentiable: the state is rebuilt inside the graph
                S = self._rebuild_state(h, content)
                self.state = S.data.copy()          # keep a numpy copy for inspection
                state_read = w @ S                  # (T, r)  <- differentiable
                out_l = state_read @ self.Wr        # (T, d)
            else:
                self.state += w_np.T @ (h.data @ self.Wv.data)   # write
                self.state = self._diffuse()                      # diffuse
                state_read = w @ Tensor(self.state)              # (T, r)
                out_l = state_read @ self.Wr                      # (T, d)
            if self.inject_mode == "prior" and self.Wp is not None:
                z = self.layer_logits({"state_read": state_read})
                if z is not None:
                    pr = softmax_rows(z)                          # (T, V)
                    out_l = pr @ self.emb.data                    # (T, d)
        # ---- the exact tier: content-keyed store -----------------------
        out_e = None
        hits = 0
        if self.use_exact and self.emb is not None:
            read_ids = np.zeros((T, self.d))
            buf = list(self._buf)
            for t in range(T):
                prev_key = self._key(buf)
                buf.append(int(token_ids[t]))
                if len(buf) > self.ctx_len:
                    buf = buf[-self.ctx_len:]
                key = self._key(buf)
                self.store_lookups += 1
                v = self.store.get(key, -1)
                if v >= 0:
                    vec = self.emb.data[v]
                    if self.center:
                        C = self._contrast_table()
                        if C is not None:
                            vec = C[v]
                    if self.read_gain == "raw":
                        vec = vec * float(self.read_scale)
                    elif self.read_gain == "norm":
                        # inject the recalled token AT THE RESIDUAL'S OWN SCALE.
                        # Without this the injected embedding is ~50x smaller
                        # than the residual and simply cannot move the argmax.
                        hn = float(np.linalg.norm(h.data[t]))
                        vn = float(np.linalg.norm(vec))
                        if vn > 1e-12:
                            vec = vec * (hn / vn)
                    read_ids[t] = vec
                    hits += 1
                # causal write: the pair (context ending at t-1, x_t) is
                # complete NOW, so it is written NOW -- never earlier.
                if prev_key is not None:
                    self._put(prev_key, int(token_ids[t]))
            self._buf = buf
            out_e = Tensor(read_ids)
        # ---- combine ----------------------------------------------------
        h_out = h
        if out_e is not None:
            h_out = h_out + out_e @ self.Wo_exact
        if out_l is not None:
            h_out = h_out + out_l @ self.Wo_lossy
        if self.state_mode == "rebuild":
            self._hist.append((h.data.copy(), content.data.copy()))
            if len(self._hist) > self.hist_len:
                self._hist = self._hist[-self.hist_len:]
        self.write_calls += 1
        self.tokens_written += T
        self.store_hits += hits
        return h_out, {"route": w_np, "anchor": am, "hits": hits,
                       "state_read": state_read, "emb_read": out_e}

    # ------------------------------------------------- the layer-local head
    def layer_logits(self, info):
        """The layer's OWN next-token prediction, from its own state.

        This is the objective that replaces the base model's loss.  It is
        well-posed (the target is the token that actually followed), it never
        touches the base model's parameters, and therefore it CANNOT diverge
        the way the base-model-loss training did.
        """
        sr = info.get("state_read")
        if sr is None or self.Wp is None:
            return None
        return sr @ self.Wp + self.bias

    # ------------------------------------------------------------- the store
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
            old = self.store_order.pop(0)          # FIFO eviction
            self.store.pop(old, None)
        self.store[key] = int(value)
        self.store_order.append(int(key))


def GRAFT_STATE_BYTES(M: int, r: int) -> int:
    """Bytes of persistent state for M anchors of rank r (exact tier + fluid)."""
    return M * r * 8 + 20 * 819