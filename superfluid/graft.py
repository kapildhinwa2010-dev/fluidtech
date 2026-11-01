"""
superfluid.graft — the attachable memory layer.

WHAT THIS IS
------------
A single module you can bolt onto ANY causal LM with one insertion:

        h' = h + W_e . E(h)  +  W_l . S(h)
                 ^^^^^^^         ^^^^^^^
                 exact tier      lossy tier (the constant-size fluid state)

    E(h)  = the EXACT read:  a bounded, CONTENT-KEYED store of token ids,
            pulled back out through the model's own embedding table.
    S(h)  = the LOSSY read:  the diffused aggregate state (M x r floats).

WHY IT IS SAFE TO ATTACH
------------------------
`W_e` and `W_l` are initialised to ZERO, so at the moment of attachment the
branch is an EXACT identity:  h' == h, bit for bit.  Every logit, every
perplexity, every sampled token is unchanged.  The layer can only start to
matter after someone trains it, and training it is opt-in.

WHY THERE ARE TWO TIERS
-----------------------
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
---------
The store is written CAUSALLY.  At position t the branch has seen tokens
x_0..x_t only.  The pair (context ending at t-1, next token x_t) is complete
at time t, so the write happens at time t -- never earlier.  The read at
position t uses the context ending at t, which is a DIFFERENT key from the
one being written, so a token can never be read back before it was written
and the model can never "copy the current token" through the store.

PERSISTENCE
-----------
The state (`state`, `store`) lives on the object, NOT in the autodiff graph.
It survives across calls: that is what makes it a *memory* rather than a
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


class _ModeStore:
    """A content-keyed store that keeps the MAJORITY value and a confidence.

    The counts fit in the frozen register's existing `meta uint16`, so a
    packed slot stays 20 bytes.
    """

    __slots__ = ("value", "support", "disagree", "order", "capacity")

    def __init__(self, capacity: int = 4096):
        self.value: dict = {}
        self.support: dict = {}
        self.disagree: dict = {}
        self.order: list = []
        self.capacity = int(capacity)

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

    def confidence(self, key) -> float:
        if key not in self.value:
            return 0.0
        a = self.support.get(key, 1)
        b = self.disagree.get(key, 0)
        return float(a) / float(max(a + b, 1))

    def put(self, key, value):
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
        """Evict the LEAST corroborated entry, not merely the oldest."""
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
                 inject_mode: str = "state", use_router: bool = True):
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
        self.use_router = bool(use_router)
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
        # the EXACT tier: a content-keyed store (a precise answer, a bounded
        # reach -- only the last ~capacity contexts)
        self.store = _ModeStore(capacity=capacity)
        # the ROUTER tier: ONE entry per anchor of the fluid.  The fluid's
        # geometry decides WHICH anchor a context belongs to, so this tier
        # has the WHOLE stream's reach (M entries, never evicted) at a
        # cluster-level precision.  "the fluid answers WHERE, the register
        # answers WHAT" -- made literal.
        self.anchor_store = _ModeStore(capacity=M)
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
        self.anchor_store.clear()
        self._buf.clear()

    def support_snapshot(self, retrieved):
        """The confirmation count of each retrieved value (0 = never seen)."""
        return np.array([self.store.support.get(int(r), 0) if r >= 0 else 0
                         for r in retrieved], dtype=np.int64)

    def confidence_snapshot(self, retrieved, keys=None):
        """The corroboration ratio of each retrieved value."""
        out = np.zeros(len(retrieved))
        for i, r in enumerate(retrieved):
            if r < 0:
                continue
            if keys is not None:
                out[i] = self.confidence(int(keys[i]))
            else:
                out[i] = 1.0
        return out

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
        retrieved = np.full(T, -1, dtype=np.int64)
        conf_arr = np.zeros(T, dtype=float)
        router_retrieved = np.full(T, -1, dtype=np.int64)
        router_conf = np.zeros(T, dtype=float)
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
                # the ROUTER tier first (it has the whole-stream reach),
                # then the exact tier (a precise answer, a bounded reach)
                if self.use_router:
                    av = self.anchor_store.get(int(am[t]), -1)
                    if av >= 0:
                        router_retrieved[t] = av
                        router_conf[t] = self.anchor_store.confidence(int(am[t]))
                v = self.store.get(key, -1)
                if v >= 0:
                    retrieved[t] = v
                    conf_arr[t] = self.confidence(key)
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
                # the ROUTER tier: the pair is filed under the ANCHOR of the
                # previous position -- the fluid's geometry decides which
                # anchor a context belongs to, so this tier never evicts and
                # its reach is the WHOLE stream.
                if t > 0 and self.use_router:
                    self.anchor_store.put(int(am[t - 1]), int(token_ids[t]))
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
                       "state_read": state_read, "emb_read": out_e,
                       "retrieved": retrieved,
                       "support": self.support_snapshot(retrieved),
                       "confidence": conf_arr,
                       "router_retrieved": router_retrieved,
                       "router_confidence": router_conf}

    # ------------------------------------------------------------- the gate
    def gated_logits(self, logits, info, gain: float = 8.0,
                     use_support: bool = False, support_scale: float = 1.0,
                     tau: float | None = None, c_min: float = 2.0,
                     big: float = 20.0, router_weight: float = 0.3,
                     gate_space: str = "logit"):
        """Apply the retrieval as a GATED override, in logit space.

        THE PROBLEM THIS SOLVES.  The injection gain that maximises RECALL
        (300) and the one that maximises GENERATION quality (0.3) are three
        orders of magnitude apart: a hard override must be strong enough to
        win the argmax, but a strong override forces a wrong retrieval into
        the model and ruins it (+1.08 nats).

        THE GATE.  The base model's OWN uncertainty at that position:

            u_t = 1 - max_v p(v)

        and the retrieved token's logit is boosted by  gain * u_t.

        WHY ONE KNOB SERVES BOTH.  In the recall task the base model has no
        idea what comes next (u ~ 1), so the retrieval decides.  In the
        generation task the base model is usually confident (u ~ 0), so the
        retrieval is ignored.  The gate is not a second hyper-parameter --
        it is the model's own confidence.

        TWO RULES make this exact rather than merely safe:

        RULE 1 (the clueless rule) -- **OFF BY DEFAULT** (`tau=2`, i.e. it
        never fires).  If the base model is clueless (u_t >= tau), the
        retrieval decides outright.  It is off because it was measured to
        be DANGEROUS: on real text a character-level model's top-1
        probability is often below 0.5, so the rule fired constantly and
        forced wrong retrievals into the model (+0.65 nats, E11c).

        It is also unnecessary: the SOFT GATE ALREADY delivers the full
        recall.  At u ~ 0.99 the boost is gain * u ~ 8, which is enough to
        win the argmax, and a wrong override then costs ~8 nats instead
        of ~60.  Measured: the soft gate alone reaches a recall of
        **1.000** (E11a).

        RULE 2 (the trusted-memory rule) -- **OFF BY DEFAULT** (`c_min=2`,
        i.e. it never fires).  If the retrieval has never been contradicted
        (its confidence >= c_min), it would decide as well.  It is off
        because it was measured to be dangerous: on real text many
        contexts have a deterministic-looking continuation that is still
        wrong for a new occurrence, and forcing those overrides made the
        base model's loss explode (E11c).  The base model's own
        uncertainty (RULE 1) is the reliable signal.

        Otherwise: the soft gate, boost = gain * u_t.

        `logits` is (T, V) numpy or a Tensor; the result is the same type.
        """
        from .autodiff import Tensor as _T
        is_tensor = isinstance(logits, _T)
        z = logits.data if is_tensor else np.asarray(logits)
        retrieved = info.get("retrieved")
        if retrieved is None:
            return logits
        retrieved = np.asarray(retrieved)
        T, V = z.shape
        # THE FALLBACK CHAIN.  The exact tier answers when it has the
        # context (a precise answer, a bounded reach).  When it does not --
        # the context is older than the store's reach -- the ROUTER tier
        # answers instead: the fluid's geometry puts the context in a
        # cluster, and the cluster's majority continuation is the best
        # available answer.  It gets a smaller boost (router_weight),
        # because it is a cluster-level guess, not a fact.
        rr = info.get("router_retrieved")
        rconf = info.get("router_confidence")
        if rr is not None:
            rr = np.asarray(rr)
            fallback = (retrieved < 0) & (rr >= 0)
            retrieved = np.where(fallback, rr, retrieved)
        else:
            fallback = np.zeros(T, dtype=bool)
        # THE MODEL'S OWN UNCERTAINTY.
        # "logit" space (the default):  u = -log p_max, i.e. how many nats
        # the model is away from being sure.  This is SCALE-FREE: the same
        # gain works on a 131k-parameter model (whose logits span +-2) and on
        # GPT-2 (whose logits span +-20).  Measured on real GPT-2: the
        # probability-space gate needed a gain of 50+ and hurt the
        # perplexity; the logit-space gate works at a gain of ~5.
        # "prob" space:  u = 1 - p_max  (the original formulation).
        # The "never fires" threshold depends on the gate's units: in
        # probability space u <= 1, in logit space u is unbounded (nats).
        if tau is None:
            tau = 2.0 if str(gate_space) != "logit" else 20.0
        e = np.exp(z - z.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        if str(gate_space) == "logit":
            u = -np.log(np.clip(p.max(axis=1), 1e-12, None))
        else:
            u = 1.0 - p.max(axis=1)
        boost = gain * u
        # the router tier's answer is a CLUSTER-LEVEL guess: a smaller boost,
        # and scaled by how much the cluster actually agrees with itself.
        # Without the confidence scaling the fallback overrules the base
        # model on contexts where the cluster's answer is a coin flip
        # (measured: the fact-task recall fell from 1.000 to 0.625).
        if rconf is not None:
            rc = np.asarray(rconf, dtype=float)
            boost = np.where(fallback,
                             boost * float(router_weight) * np.clip(rc, 0.0, 1.0),
                             boost)
        if use_support:
            sup = info.get("support")
            if sup is not None:
                conf = np.minimum(1.0, np.asarray(sup, dtype=float) / support_scale)
                boost = boost * conf
        # RULE 2: the retrieval's own confidence
        conf = info.get("confidence")
        if conf is None:
            conf = np.zeros(T)
        else:
            conf = np.asarray(conf, dtype=float)
        if rconf is not None:
            conf = np.where(fallback, np.asarray(rconf, dtype=float), conf)
        decide = (u >= tau) | (conf >= c_min)
        boost = np.where(decide, big, boost)
        out = z.copy()
        rows = np.nonzero(retrieved >= 0)[0]
        if len(rows):
            out[rows, retrieved[rows]] += boost[rows]
        return _T(out) if is_tensor else out

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
        """Record an occurrence: (context -> the token that followed it).

        The store keeps the MAJORITY continuation, not the first one.  The
        first-write-wins policy was exact but blind: a context that occurs
        everywhere kept only its first continuation, so the statistically
        correct answer was lost (measured: 0/3).  An online mode tracker
        keeps the statistically correct answer AND counts the
        corroboration, which gives a CONFIDENCE for free.
        """
        self.store.put(key, value)

    def confidence(self, key: int) -> float:
        """How often the stored value has been corroborated (1.0 = never
        contradicted).  0.0 for an unknown context."""
        return self.store.confidence(key)


def GRAFT_STATE_BYTES(M: int, r: int) -> int:
    """Bytes of persistent state for M anchors of rank r (exact tier + fluid)."""
    return M * r * 8 + 20 * 819
