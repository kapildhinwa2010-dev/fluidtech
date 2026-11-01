"""
superfluid.autodiff — a tiny reverse-mode automatic differentiation engine.

Pure NumPy, ~1 op set, enough to train the Superfluid stack end-to-end inside
this sandbox (no torch available).  Every op has a hand-written backward and
is verified against finite differences in `selftest.py`.

Design: each Tensor holds `data`, `grad`, its parent Tensors and a closure
`_backward` that pushes gradient contributions into the parents.  The graph is
rebuilt every forward pass; parameter leaves accumulate `.grad` across the
traversal (zero them between steps).
"""

from __future__ import annotations

import numpy as np

_SQRT_PI = np.sqrt(np.pi)


def _unbroadcast(g: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduce a broadcasted gradient back to `shape`."""
    if g.shape == shape:
        return g
    # sum over broadcast axes
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g.reshape(shape)


class Tensor:
    __slots__ = ("data", "grad", "parents", "op", "_backward", "requires_grad")

    def __init__(self, data, parents=(), op="", requires_grad=True):
        self.data = np.array(data, dtype=np.float64, copy=True)
        self.grad = np.zeros_like(self.data)
        self.parents = tuple(parents)
        self.op = op
        self.requires_grad = requires_grad
        self._backward = None

    # -------------------------------------------------------------- plumbing
    @property
    def shape(self):
        return self.data.shape

    def _apply(self, data, op, parents, backward):
        out = Tensor(data, parents=parents, op=op)
        out._backward = backward
        return out

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self.op})"

    # ------------------------------------------------------------ arithmetic
    def __add__(self, other):
        o = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = self._apply(self.data + o.data, "+", (self, o), lambda: None)

        def bw():
            g = out.grad
            if self.requires_grad:
                self.grad += _unbroadcast(g, self.data.shape)
            if o.requires_grad:
                o.grad += _unbroadcast(g, o.data.shape)
        out._backward = bw
        return out

    __radd__ = __add__

    def __sub__(self, other):
        o = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        neg = o.data
        out = self._apply(self.data - neg, "-", (self, o), lambda: None)

        def bw():
            g = out.grad
            if self.requires_grad:
                self.grad += _unbroadcast(g, self.data.shape)
            if o.requires_grad:
                o.grad += _unbroadcast(-g, o.data.shape)
        out._backward = bw
        return out

    def __rsub__(self, other):
        o = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return o - self

    def __mul__(self, other):
        o = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = self._apply(self.data * o.data, "*", (self, o), lambda: None)

        def bw():
            g = out.grad
            if self.requires_grad:
                self.grad += _unbroadcast(g * o.data, self.data.shape)
            if o.requires_grad:
                o.grad += _unbroadcast(g * self.data, o.data.shape)
        out._backward = bw
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = self._apply(self.data / o.data, "/", (self, o), lambda: None)

        def bw():
            g = out.grad
            if self.requires_grad:
                self.grad += _unbroadcast(g / o.data, self.data.shape)
            if o.requires_grad:
                o.grad += _unbroadcast(-g * self.data / (o.data ** 2), o.data.shape)
        out._backward = bw
        return out

    def __rtruediv__(self, other):
        o = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return o / self

    def __neg__(self):
        return self * -1.0

    def __pow__(self, p):
        p = float(p)
        out = self._apply(self.data ** p, "pow", (self,), lambda: None)

        def bw():
            if self.requires_grad:
                self.grad += out.grad * p * (self.data ** (p - 1))
        out._backward = bw
        return out

    def __matmul__(self, other):
        o = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = self._apply(self.data @ o.data, "@", (self, o), lambda: None)

        def bw():
            g = out.grad
            if self.requires_grad:
                self.grad += g @ o.data.T
            if o.requires_grad:
                o.grad += self.data.T @ g
        out._backward = bw
        return out

    # --------------------------------------------------------------- shapes
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        old = self.data.shape
        out = self._apply(self.data.reshape(shape), "reshape", (self,), lambda: None)

        def bw():
            self.grad += out.grad.reshape(old)
        out._backward = bw
        return out

    @property
    def T(self):
        return self.transpose()

    def transpose(self):
        out = self._apply(self.data.T, "T", (self,), lambda: None)

        def bw():
            self.grad += out.grad.T
        out._backward = bw
        return out

    def __getitem__(self, key):
        out = self._apply(self.data[key], "getitem", (self,), lambda: None)

        def bw():
            g = out.grad
            src = np.zeros_like(self.data)
            k = key
            # --- strip newaxis (None) entries: they only add output axes ---
            if isinstance(k, tuple):
                pos = [i for i, kk in enumerate(k) if kk is None]
                for p in reversed(pos):
                    g = np.squeeze(g, axis=p)
                k = tuple(kk for kk in k if kk is not None)
                if len(k) == 1:
                    k = k[0]
            if isinstance(k, tuple):
                has_arr = any(isinstance(kk, np.ndarray) for kk in k)
                if not has_arr:
                    src[k] += g                      # pure basic indexing
                else:
                    arrs = [kk for kk in k if isinstance(kk, np.ndarray)]
                    bshape = np.broadcast_shapes(*[kk.shape for kk in arrs])
                    idx_arrays = []
                    for _, kk in enumerate(k):
                        if isinstance(kk, np.ndarray):
                            idx_arrays.append(np.broadcast_to(kk, bshape))
                        else:
                            idx_arrays.append(kk)    # slices/ints kept as-is
                    np.add.at(src, tuple(idx_arrays), g)
            elif isinstance(k, np.ndarray):
                np.add.at(src, k, g)
            elif k is None:                      # x[None]
                src[...] = g.reshape(self.data.shape)
            else:
                src[k] += g
            self.grad += src
        out._backward = bw
        return out

    def gather_rows(self, idx):
        """out[i, j] = self[idx[i, j]]   (self is (n, d), idx is (n, k) -> (n, k, d))"""
        out = self._apply(
            self.data[idx], "gather_rows", (self,), lambda: None
        )

        def bw():
            np.add.at(self.grad, idx, out.grad)
        out._backward = bw
        return out

    # ------------------------------------------------------------ reductions
    def sum(self, axis=None, keepdims=False):
        out = self._apply(
            self.data.sum(axis=axis, keepdims=keepdims),
            "sum",
            (self,),
            lambda: None,
        )

        def bw():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.broadcast_to(g, self.data.shape).copy()
        out._backward = bw
        return out

    def max(self, axis=None, keepdims=False):
        am = np.argmax(self.data, axis=axis, keepdims=True)
        out = self._apply(
            np.max(self.data, axis=axis, keepdims=keepdims),
            "max",
            (self,),
            lambda: None,
        )
        mask = np.zeros_like(self.data, dtype=bool)
        np.put_along_axis(mask, am, True, axis=axis)

        def bw():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += g * mask
        out._backward = bw
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        s = self.sum(axis=axis, keepdims=keepdims)
        return s * (1.0 / n)          # do NOT override s._backward: the mul
                                       # backward already routes out.grad -> s.grad

    def square(self):
        return self * self

    def abs(self):
        out = self._apply(np.abs(self.data), "abs", (self,), lambda: None)

        def bw():
            self.grad += out.grad * np.sign(self.data)
        out._backward = bw
        return out

    def exp(self):
        out = self._apply(np.exp(self.data), "exp", (self,), lambda: None)

        def bw():
            self.grad += out.grad * out.data
        out._backward = bw
        return out

    def log(self):
        out = self._apply(np.log(self.data), "log", (self,), lambda: None)

        def bw():
            self.grad += out.grad / self.data
        out._backward = bw
        return out

    def sqrt(self):
        out = self._apply(np.sqrt(self.data), "sqrt", (self,), lambda: None)

        def bw():
            self.grad += out.grad * 0.5 / out.data
        out._backward = bw
        return out

    def tanh(self):
        out = self._apply(np.tanh(self.data), "tanh", (self,), lambda: None)

        def bw():
            self.grad += out.grad * (1 - out.data ** 2)
        out._backward = bw
        return out

    def erf(self):
        out = self._apply(0.5 * (1 + _erf_np(self.data)), "erf", (self,), lambda: None)

        def bw():
            self.grad += out.grad * (2.0 / _SQRT_PI) * np.exp(-self.data ** 2)
        out._backward = bw
        return out

    def relu(self):
        out = self._apply(np.maximum(self.data, 0.0), "relu", (self,), lambda: None)

        def bw():
            self.grad += out.grad * (self.data > 0)
        out._backward = bw
        return out

    def gelu(self):
        out = self._apply(
            0.5 * self.data * (1 + _erf_np(self.data / np.sqrt(2.0))),
            "gelu",
            (self,),
            lambda: None,
        )

        def bw():
            x = self.data
            cdf = 0.5 * (1 + _erf_np(x / np.sqrt(2.0)))
            pdf = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
            self.grad += out.grad * (cdf + x * pdf)
        out._backward = bw
        return out

    # --------------------------------------------------------------- scatter
    def index_add(self, idx, src):
        """out = zeros_like(self.data) with src scattered into rows idx.

        The scatter MUST happen in the forward pass too -- an earlier version
        returned zeros here, which silently turned every Laplacian built with
        this op into the zero matrix (and the whole fluid into a no-op).
        """
        out_data = np.zeros_like(self.data)
        np.add.at(out_data, idx, src.data)
        out = self._apply(out_data, "index_add", (self, src), lambda: None)

        def bw():
            # out = base + scatter(src, idx)  =>  d/d(src) = out.grad[idx]
            src.grad += out.grad[idx]
            self.grad += out.grad             # d/d(base): base is the zero base
        out._backward = bw
        return out

    def gather(self, idx):
        """out[i] = self[idx[i]]"""
        out = self._apply(self.data[idx], "gather", (self,), lambda: None)

        def bw():
            src = np.zeros_like(self.data)
            np.add.at(src, idx, out.grad)
            self.grad += src
        out._backward = bw
        return out

    # ----------------------------------------------------------------- misc
    def pow(self, p):
        return self ** float(p)

    def backward(self, seed=None):
        """Reverse-mode backward pass.

        If `seed` is given it overwrites the output gradient.  Otherwise the
        output gradient is seeded with ones *only if it is still all zeros*
        (so an externally supplied gradient, e.g. dCE/dlogits, is respected).
        """
        if seed is not None:
            self.grad = np.array(seed, dtype=np.float64, copy=True)
        elif not np.any(self.grad):
            self.grad = np.ones_like(self.data)
        topo, seen = [], set()

        def build(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t.parents:
                build(p)
            topo.append(t)

        build(self)
        for t in reversed(topo):
            if t._backward is not None:
                t._backward()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)


def softplus(t: Tensor) -> Tensor:
    """log(1 + exp(t)), numerically stable for large |t|."""
    return (1.0 + t.exp()).log()


def _erf_np(x: np.ndarray) -> np.ndarray:
    """Vectorised erf (Abramowitz & Stegun 7.1.26 is too weak; use tanh form)."""
    sign = np.sign(x)
    ax = np.abs(x)
    p = 0.3275911
    a1, a2, a3, a4, a5 = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    )
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-ax * ax)
    return sign * y


def concat(tensors, axis=0):
    datas = [t.data for t in tensors]
    out = Tensor(np.concatenate(datas, axis=axis), parents=tuple(tensors), op="concat")

    def bw():
        sizes = [d.shape[axis] for d in datas]
        idx = 0
        for t, s in zip(tensors, sizes):
            sl = [slice(None)] * out.data.ndim
            sl[axis] = slice(idx, idx + s)
            t.grad += out.grad[tuple(sl)]
            idx += s
    out._backward = bw
    return out


def stack(tensors, axis=0):
    datas = [t.data for t in tensors]
    out = Tensor(np.stack(datas, axis=axis), parents=tuple(tensors), op="stack")

    def bw():
        for i, t in enumerate(tensors):
            sl = [slice(None)] * out.data.ndim
            sl[axis] = i
            t.grad += out.grad[tuple(sl)]
    out._backward = bw
    return out


def softmax_rows(x: Tensor, mask: np.ndarray | None = None) -> Tensor:
    """Row-wise softmax with optional additive -1e9 mask (mask=True -> masked)."""
    z = x.data if mask is None else np.where(mask, -1e9, x.data)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    if mask is not None:
        e = np.where(mask, 0.0, e)
    s = e.sum(axis=-1, keepdims=True)
    p = e / s
    out = x._apply(p, "softmax", (x,), lambda: None)

    def bw():
        g = out.grad
        dot = (g * p).sum(axis=-1, keepdims=True)
        x.grad += p * (g - dot)
    out._backward = bw
    return out
