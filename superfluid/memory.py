"""
superfluid.memory — the FROZEN REGISTER and the geometry-addressed memory graft.
This module implements the two ideas that make the fluid trustworthy:
A frozen register (L1): an append-only, write-once table of EXACT
facts.  A fact's value is stored as an INTEGER token id, so a retrieved
value is bit-identical to the source -- not float-approximate.  The
register is never diffused, so the fluid's smoothing cannot destroy it.
A sheaf memory adapter (L2, the "graft"): a constant-size state
S in R^{M x r} plus a static prototype codebook.  The adapter routes
(geometry decides WHERE) and the register stores (content decides
WHAT).  That split is what makes exactness possible at all: a diffusion
smears, so an exact value must never live inside one.
An external tier (L3): a pointer into the original source.  A
pointer is 6 bytes and is unbounded, so it is the fallback whenever the
register cannot answer.
Slot layout (20 bytes, little-endian):
key     uint64   8 B   64-bit context hash (the OCCURRENCE identity)
val     int32    4 B   the exact token id
sig     uint32   4 B   4 x uint8 embedding signature (collision detector)
anchor  uint16   2 B   the anchor the fact was routed to when promoted
meta    uint16   2 B   doc_id (12 bits) | trust (2 bits) | reserved (2)
Why 20 bytes and not the 14 quoted in DESIGN-pinned-memory.md: the 4-byte
embedding signature was added so that a 64-bit hash collision is DETECTABLE
rather than silent ("fail loud, never silently wrong").  The cost of that
guarantee is 6 bytes/fact, i.e. 819 facts in 16 KB instead of 1,170.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "FNV_OFFSET", "FNV_PRIME", "MASK64", "fnv1a64", "ctx_key",
    "FrozenRegister", "SheafMemoryAdapter", "logit_override",
    "budget_split", "embedding_signature", "SLOT_BYTES",
]

# ------------------------------------------------------------------ hashing
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1


def fnv1a64(values) -> int:
    """Deterministic 64-bit FNV-1a hash over a sequence of integers.

    Pure python, no dependencies, and stable across runs/platforms (unlike
    Python's salted `hash()`), so a register written today can be verified
    tomorrow.
    """
    h = FNV_OFFSET
    for v in values:
        h ^= int(v) & MASK64
        h = (h * FNV_PRIME) & MASK64
    return h


def ctx_key(window, doc_id: int = 0) -> int:
    """64-bit identity of a CONTEXT OCCURRENCE (doc_id + token window).

    The same content in two different documents gets two different keys, so
    a register can be told apart per source; the fact-level grouping (for the
    corroboration quorum) is handled by the caller, which passes the same
    `window` with different `doc_id`s.
    """
    if isinstance(window, (int, np.integer)):
        window = (int(window),)
    vals = [int(doc_id) & 0xFFF]
    vals.extend(int(w) & MASK64 for w in window)
    return fnv1a64(vals)


def embedding_signature(vec: np.ndarray, dims: int = 4) -> int:
    """Quantise the first `dims` coordinates of a unit vector to a uint32.

    Used as a cheap second channel: if the 64-bit key collides, the
    signature almost certainly will not, and the read fails loud instead of
    returning a wrong exact value.
    """
    v = np.asarray(vec, dtype=np.float64).ravel()
    if v.size == 0:
        return 0
    n = np.linalg.norm(v[:dims]) + 1e-12
    q = np.clip(np.round(v[:dims] / n * 127.0) + 128.0, 0, 255).astype(np.uint8)
    out = 0
    for i in range(dims):
        out |= int(q[i]) << (8 * i)
    return out


# ------------------------------------------------------------------ register
SLOT_BYTES = 20


class FrozenRegister:
    """Append-only, write-once table of EXACT facts.

    Guarantees
     ----------
     * **Invariance.**  A slot that holds (key, val) at time t still holds
       (key, val) at every later time, bit-for-bit.  Nothing in the write
       path can modify an existing slot.
     * **Integer exactness.**  `val` is an int32 token id, so a successful
       recall returns a value that is *equal*, not approximately equal.
     * **Fail-loud.**  A recall verifies both the 64-bit key and the 4-byte
       embedding signature.  On mismatch the register reports "not found"
       instead of returning a value that may belong to another fact.
    """

    SLOT_BYTES = SLOT_BYTES

    def __init__(self, capacity: int = 1024):
        self.capacity = int(capacity)
        self.key = np.zeros(self.capacity, dtype=np.uint64)
        self.val = np.zeros(self.capacity, dtype=np.int32)
        self.sig = np.zeros(self.capacity, dtype=np.uint32)
        self.anchor = np.zeros(self.capacity, dtype=np.uint16)
        self.meta = np.zeros(self.capacity, dtype=np.uint16)
        self.used = np.zeros(self.capacity, dtype=bool)
        self.n = 0
        self._by_key: dict[int, list[int]] = {}

    # ------------------------------------------------------------- encoding
    @staticmethod
    def _pack_meta(doc_id: int, trust: int) -> int:
        return (int(doc_id) & 0x0FFF) | ((int(trust) & 0x3) << 12)

    @staticmethod
    def _unpack_meta(meta: int):
        return int(meta) & 0x0FFF, (int(meta) >> 12) & 0x3

    # ---------------------------------------------------------------- write
    def promote(self, val: int, key: int, sig: int = 0, anchor: int = 0,
                doc_id: int = 0, trust: int = 0) -> int:
        """Append a NEW slot.  Returns the slot index.

        Append-only means a fact that is promoted twice gets two slots and
        the *latest* wins on recall: that is the versioning mechanism, and it
        is also what makes the register an append-only audit log.
        """
        if self.n >= self.capacity:
            raise MemoryError(
                f"frozen register is full ({self.capacity} slots x {SLOT_BYTES} B "
                f"= {self.capacity * SLOT_BYTES} B).  Overflow is REFUSED on purpose: "
                f"a register that silently grows becomes a KV cache.")
        s = self.n
        self.key[s] = np.uint64(key & MASK64)
        self.val[s] = np.int32(val)
        self.sig[s] = np.uint32(sig & 0xFFFFFFFF)
        self.anchor[s] = np.uint16(anchor)
        self.meta[s] = np.uint16(self._pack_meta(doc_id, trust))
        self.used[s] = True
        self.n += 1
        self._by_key.setdefault(int(key), []).append(s)
        return s

    def set_trust(self, slot: int, trust: int = 1) -> None:
        """Raise the trust bit of an existing slot (corroboration gate)."""
        doc, _ = self._unpack_meta(int(self.meta[slot]))
        self.meta[slot] = np.uint16(self._pack_meta(doc, trust))

    # ----------------------------------------------------------------- read
    def slots_for(self, key: int) -> list[int]:
        return self._by_key.get(int(key), [])

    def latest(self, key: int):
        """Most recent slot holding `key`, or None."""
        sl = self._by_key.get(int(key))
        return sl[-1] if sl else None

    def verify(self, slot: int, key: int, sig: int | None = None) -> bool:
        """True iff the slot really holds this occurrence."""
        if not (0 <= slot < self.n) or not self.used[slot]:
            return False
        if int(self.key[slot]) != (int(key) & MASK64):
            return False
        if sig is not None and int(self.sig[slot]) != (int(sig) & 0xFFFFFFFF):
            return False
        return True

    def recall(self, key: int, sig: int | None = None):
        """Return (slot, exact_value) or None.

        `sig` is the embedding signature of the query.  When supplied, a
        signature mismatch makes the recall FAIL (return None) instead of
        returning another fact's value -- this is the fail-loud path.
        """
        s = self.latest(key)
        if s is None:
            return None
        if sig is not None and int(self.sig[s]) != (int(sig) & 0xFFFFFFFF):
            return None            # hash collision or wrong occurrence -> fail loud
        return s, int(self.val[s])

    def read_slot(self, slot: int):
        """Raw read of a slot (no verification) -> (key, val, sig, anchor, trust)."""
        if not (0 <= slot < self.n) or not self.used[slot]:
            return None
        doc, trust = self._unpack_meta(int(self.meta[slot]))
        return (int(self.key[slot]), int(self.val[slot]), int(self.sig[slot]),
                int(self.anchor[slot]), doc, trust)

    def trust_of(self, slot: int) -> int:
        return self._unpack_meta(int(self.meta[slot]))[1]

    def doc_of(self, slot: int) -> int:
        return self._unpack_meta(int(self.meta[slot]))[0]

    def quorum(self, key: int) -> int:
        """Number of DISTINCT sources that promoted this key.

        The corroboration gate: a fact promoted by two independent documents
        is "trusted"; a fact promoted once stays "suspect".  Because the key
        includes doc_id, two different documents produce different keys --
        so the caller must pass the doc-independent fact key.  We therefore
        count distinct doc_ids stored under keys that share the same
        *occurrence signature* is not possible here; instead we count the
        distinct doc_ids across all slots of the given key, which is exact
        for a fixed (window, doc) pair and conservative otherwise.
        """
        sl = self._by_key.get(int(key), [])
        return len({self.doc_of(s) for s in sl})

    # ------------------------------------------------------------- accounting
    def nbytes(self) -> int:
        """Bytes of *state* held (the serialised slot table)."""
        return self.capacity * SLOT_BYTES

    def used_bytes(self) -> int:
        return self.n * SLOT_BYTES

    def stats(self) -> dict:
        trusts = [self.trust_of(s) for s in range(self.n)] if self.n else []
        return {
            "capacity_slots": self.capacity,
            "used_slots": int(self.n),
            "slot_bytes": SLOT_BYTES,
            "state_bytes": self.nbytes(),
            "used_bytes": self.used_bytes(),
            "n_trusted": int(sum(1 for t in trusts if t)),
            "fill_fraction": float(self.n / max(self.capacity, 1)),
        }

    def to_bytes(self) -> bytes:
        return (self.key.tobytes() + self.val.tobytes() + self.sig.tobytes() +
                self.anchor.tobytes() + self.meta.tobytes() +
                self.used.tobytes())

    @classmethod
    def from_bytes(cls, blob: bytes, capacity: int) -> FrozenRegister:
        r = cls(capacity=capacity)
        n = capacity
        p = 0

        def take(arr, dt):
            nonlocal p
            out = np.frombuffer(blob[p:p + n * np.dtype(dt).itemsize], dtype=dt).copy()
            p += n * np.dtype(dt).itemsize
            return out

        r.key = take(None, np.uint64)
        r.val = take(None, np.int32)
        r.sig = take(None, np.uint32)
        r.anchor = take(None, np.uint16)
        r.meta = take(None, np.uint16)
        r.used = take(None, np.uint8).astype(bool)
        r.n = int(r.used.sum())
        r._by_key = {}
        for s in range(r.n):
            r._by_key.setdefault(int(r.key[s]), []).append(s)
        return r


# ------------------------------------------------------------------ override
def logit_override(logits: np.ndarray, token_ids) -> np.ndarray:
    """Force `token_ids` to win the argmax, deterministically.

    This is the difference between "the model was told the answer" and "the
    model cannot disagree":  a tool that merely *returns* a token can be
    outvoted by the fluid's own logits, so the extraction would be
    probabilistic.  Overriding the logit makes the next token provably the
    recalled one.

    Accepts logits of shape (V,) or (B, V), as a plain ndarray **or** as a
    framework Tensor.  `token_ids` is an int or a sequence of ints (one per
    row for the 2-D case).
    """
    # accept a framework Tensor as well as a plain ndarray
    if hasattr(logits, "data"):
        logits = logits.data
    out = np.array(logits, dtype=np.float64, copy=True)
    if out.ndim == 1:
        if out.shape[0] == 0:
            raise ValueError("empty logits")
        out[int(token_ids)] = np.inf
        return out
    if out.ndim == 2:
        ids = np.atleast_1d(np.asarray(token_ids, dtype=np.int64))
        if ids.size == 1:
            ids = np.full(out.shape[0], ids[0], dtype=np.int64)
        if ids.size != out.shape[0]:
            raise ValueError("token_ids length must match the batch")
        rows = np.arange(out.shape[0])
        out[rows, ids] = np.inf
        return out
    raise ValueError("logits must be 1-D or 2-D")


# ------------------------------------------------------------------- budget
def budget_split(total_bytes: int, M: int, r: int, slot_bytes: int = SLOT_BYTES,
                 codebook_bytes: int = 0) -> dict:
    """Split a RUNTIME byte budget between the fluid state and the frozen register.

    `total_bytes` is the *runtime* budget (what a tenant pays per session).
    `codebook_bytes` is the STATIC artefact (prototypes + value projection),
    which is shared across tenants and therefore reported, not subtracted.

    The design law from DESIGN-pinned-memory.md: **naming one anchor out of M
    needs log2(M) bits, while an anchor holds r*16 bits.**  So r is grossly
    over-provisioned for addressing, and a bigger budget should be spent on
    MORE PINS, not on more anchors.  Growing the fluid raises the per-token
    compute linearly in M; growing the register costs nothing at inference.

    If the requested M does not fit, M is reduced to the largest value that
    does -- and the reduction is reported, because silently over-committing
    the budget is how a "16 KB" claim turns into a 32 KB claim.
    """
    total = int(total_bytes)
    r = int(r)
    M_max = total // max(r * 2, 1)
    M_used = int(min(int(M), M_max))
    fluid = M_used * r * 2
    rest = max(total - fluid, 0)
    pins = rest // int(slot_bytes)
    return {
        "total_bytes": total,
        "M_requested": int(M),
        "M_used": M_used,
        "M_reduced": bool(M_used < int(M)),
        "r": r,
        "fluid_bytes": fluid,
        "register_bytes": int(pins) * int(slot_bytes),
        "pins": int(pins),
        "codebook_bytes": int(codebook_bytes),
        "artefact_bytes": int(total) + int(codebook_bytes),
        "fluid_pct": 100.0 * fluid / max(total, 1),
    }


# ------------------------------------------------------------------ adapter
class SheafMemoryAdapter:
    """Constant-size, geometry-addressed memory with an exact frozen register.

    Layout
     ------
     * `protos`  (M, d)  static prototype codebook -- part of the *prior
       artefact*, not the runtime state.
     * `anchors` (M, r)  the runtime fluid state (the "16 KB").
     * `register`         the exact tier (append-only).
     * `W_in`   (d, r)    value projection, part of the codebook.

    Read path:  query -> nearest prototype (address) -> anchor content (gist)
    Exact path: query -> content key -> register slot -> INTEGER token id.

    The two paths are deliberately separate.  A diffusion smears, so the
    value that must be exact never enters the diffusion.
    """

    def __init__(self, M: int = 1024, r: int = 8, d: int = 64,
                 capacity: int = 1024, seed: int = 0):
        self.M, self.r, self.d = int(M), int(r), int(d)
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.protos = None
        self.anchors = np.zeros((self.M, self.r), dtype=np.float64)
        self.register = FrozenRegister(capacity=capacity)
        self.W_in = rng.normal(0.0, self.d ** -0.5, (self.d, self.r))
        self._L = None
        self._lam_max = None
        self.write_calls = 0
        self.tokens_written = 0

    # ------------------------------------------------------------- codebook
    def set_prototypes(self, protos: np.ndarray) -> None:
        """Install the static codebook (M, d) and rebuild the anchor graph."""
        P = np.asarray(protos, dtype=np.float64)
        if P.shape != (self.M, self.d):
            raise ValueError(f"prototypes must be ({self.M}, {self.d}), got {P.shape}")
        self.protos = P
        self._L = None

    def anchor_graph(self, k: int = 8):
        """k-NN graph over the prototypes -> signed sheaf Laplacian (numpy)."""
        if self.protos is None:
            raise RuntimeError("set_prototypes() first")
        from .sheaf import build_laplacian
        Pn = self.protos / (np.linalg.norm(self.protos, axis=1, keepdims=True) + 1e-12)
        S = Pn @ Pn.T
        np.fill_diagonal(S, -np.inf)
        nbr = np.argpartition(-S, kth=min(k, self.M - 1) - 1, axis=1)[:, :k]
        rows = np.repeat(np.arange(self.M), k)
        cols = nbr.reshape(-1)
        u = np.minimum(rows, cols)
        v = np.maximum(rows, cols)
        key = u.astype(np.int64) * self.M + v
        _, keep = np.unique(key, return_index=True)
        edges = np.stack([u[keep], v[keep]], axis=1)
        alpha = np.maximum(S[edges[:, 0], edges[:, 1]], 0.0) + 1e-3
        return edges, alpha, build_laplacian(self.M, edges, alpha)

    # --------------------------------------------------------------- routing
    def route(self, q: np.ndarray, k: int = 1):
        """Nearest prototype(s) for a query.  Returns (idx, scores)."""
        if self.protos is None:
            raise RuntimeError("set_prototypes() first")
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        Pn = self.protos / (np.linalg.norm(self.protos, axis=1, keepdims=True) + 1e-12)
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        sc = qn @ Pn.T
        if k == 1:
            idx = np.argmax(sc, axis=1)
            return idx, sc[np.arange(len(idx)), idx]
        order = np.argsort(-sc, axis=1)[:, :k]
        rows = np.arange(sc.shape[0])[:, None]
        return order, sc[rows, order]

    # ----------------------------------------------------------------- write
    def write(self, X: np.ndarray, t: float = 1.0, cheb_m: int = 12,
              diffuse: bool = True) -> None:
        """Streaming write of a chunk of embeddings (n, d).

        The diffusion is applied AFTER the scatter, which is exact for a
        linear operator:  exp(-tL)(S + Bx) = exp(-tL)S + exp(-tL)Bx.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.d:
            raise ValueError(f"expected (n, {self.d}) embeddings, got {X.shape}")
        if X.shape[0] == 0:
            return
        idx, _ = self.route(X, k=1)
        proj = X @ self.W_in
        np.add.at(self.anchors, idx, proj)
        self.write_calls += 1
        self.tokens_written += int(X.shape[0])
        if diffuse:
            self.diffuse(t=t, cheb_m=cheb_m)

    def diffuse(self, t: float = 1.0, cheb_m: int = 12) -> None:
        """Apply exp(-t L) to the anchor state (the 'freezing' step)."""
        from .diffusion import cheb_basis, cheb_coeffs_exp, power_iteration_lmax
        if self.protos is None:
            return
        if self._L is None:
            self._L = self.anchor_graph(k=8)
        edges, alpha, L = self._L
        if self._lam_max is None:
            self._lam_max = power_iteration_lmax(L)
        lam = self._lam_max
        if lam <= 1e-8:
            return
        Y = cheb_basis(L, lam, self.anchors, cheb_m)
        a = cheb_coeffs_exp(t, lam, cheb_m)
        self.anchors = sum(a[k] * Y[k] for k in range(cheb_m + 1))

    # ------------------------------------------------------------------ read
    def read(self, q: np.ndarray, k: int = 1):
        """Geometry read: returns (anchor_idx, scores, content (r,))."""
        idx, sc = self.route(q, k=k)
        if k == 1:
            content = self.anchors[idx[0]]
            return idx, sc, content
        w = np.exp(sc - sc.max(axis=1, keepdims=True))
        w = w / w.sum(axis=1, keepdims=True)
        content = (self.anchors[idx] * w[:, :, None]).sum(axis=1)
        return idx, sc, content

    # ---------------------------------------------------------------- exact
    def promote(self, vec: np.ndarray, value_token_id: int, window,
                doc_id: int = 0, trust: int = 0):
        """Promote a fact into the register, anchored to its nearest prototype.

        Returns the slot index.  The fact's value NEVER enters the fluid, so
        it can never be smeared.
        """
        idx, _ = self.route(np.atleast_2d(np.asarray(vec, dtype=np.float64)), k=1)
        anchor = int(idx[0])
        sig = embedding_signature(vec)
        key = ctx_key(window, doc_id=doc_id)
        return self.register.promote(int(value_token_id), key, sig=sig,
                                     anchor=anchor, doc_id=doc_id, trust=trust)

    def recall(self, window, vec: np.ndarray | None = None):
        """Exact recall.  Returns (slot, token_id) or None.

        Verification: the 64-bit context key must match, and -- if `vec` is
        given -- the 4-byte embedding signature must match too.  A mismatch
        returns None (fail loud) instead of a wrong value.
        """
        key = ctx_key(window)
        sig = embedding_signature(vec) if vec is not None else None
        out = self.register.recall(key, sig=sig)
        if out is None:
            return None
        return out

    # ----------------------------------------------------------- accounting
    def state_bytes(self) -> int:
        """Runtime state only: anchors + register.  Excludes the codebook."""
        return int(self.anchors.size * 2) + self.register.nbytes()

    def codebook_bytes(self) -> int:
        """Static artefact: prototypes + value projection."""
        if self.protos is None:
            return int(self.W_in.size * 2)
        return int((self.protos.size + self.W_in.size) * 2)

    def stats(self) -> dict:
        s = self.register.stats()
        s.update({
            "M": self.M, "r": self.r, "d": self.d,
            "state_bytes": self.state_bytes(),
            "codebook_bytes": self.codebook_bytes(),
            "tokens_written": self.tokens_written,
            "write_calls": self.write_calls,
        })
        return s
