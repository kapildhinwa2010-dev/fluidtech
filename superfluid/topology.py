"""
superfluid.topology — building the data's geometry in one forward pass.

This module implements the "instant geometry" of the proposal:

  1. content routing      -> semantic edges (top-k over QK similarity)
  2. temporal chain      -> order edges (position as 1-dimensional topology)
  3. 0-D persistence     -> exact barcode via union-find (Kruskal filtration)
  4. Betti numbers      -> beta_0 (components), beta_1 (cycle rank)

(3) and (4) are *exact*, not approximated, and cost O(m log m) — which is why
the geometry can be recomputed every forward pass instead of being cached
offline.  This is the concrete answer to "jam the data's geometry in one pass".
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "route_topk", "edges_from_topk", "chain_edges", "unique_edges",
    "kruskal_barcode", "betti_numbers", "topo_descriptor",
    "connected_components",
    "hodge_relevance", "hodge_embedding", "hodge_edge_participation",
    "local_density", "persistence_landmarks",
]


# ------------------------------------------------------- selection criteria
def hodge_embedding(n: int, edges: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """The Hodge embedding Psi = (L + (1/n) 11^T)^-1 of the sheaf Laplacian."""
    from .sheaf import build_laplacian

    laplacian = build_laplacian(n, edges, np.asarray(alpha, dtype=np.float64))
    constant = np.ones((n, 1)) / np.sqrt(max(n, 1))
    matrix = laplacian + constant @ constant.T
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)


def hodge_edge_participation(
    Psi: np.ndarray, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Return normalized Hodge participation for a set of node pairs."""
    Psi = np.asarray(Psi, dtype=np.float64)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    numerator = np.abs(Psi[a, b])
    denominator = np.sqrt(
        np.maximum(np.abs(Psi[a, a]), 1e-18)
        * np.maximum(np.abs(Psi[b, b]), 1e-18)
    )
    return np.clip(numerator / denominator, 0.0, 1.0)


def hodge_relevance(n: int, edges: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Return per-node harmonic participation scores for the sheaf Laplacian."""
    embedding = hodge_embedding(n, edges, alpha)
    participation = np.abs(embedding).sum(axis=1) - np.abs(np.diag(embedding))
    return np.clip(participation, 0.0, None)


def local_density(
    protos: np.ndarray, radius: float | None = None, k: int = 8
) -> np.ndarray:
    """Return prototype neighborhood counts or effective top-k density."""
    prototypes = np.asarray(protos, dtype=np.float64)
    n = prototypes.shape[0]
    if radius is not None:
        squared_norms = (prototypes**2).sum(1)
        distances = np.sqrt(
            np.maximum(
                squared_norms[:, None]
                + squared_norms[None, :]
                - 2.0 * (prototypes @ prototypes.T),
                0.0,
            )
        )
        return (distances <= radius).sum(axis=1).astype(np.float64)
    normalized = prototypes / (
        np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-12
    )
    similarities = normalized @ normalized.T
    np.fill_diagonal(similarities, -np.inf)
    k = int(min(k, n - 1)) if n > 1 else 1
    nearest = np.partition(similarities, -k, axis=1)[:, -k:]
    maximum = np.abs(nearest).max(axis=1, keepdims=True) + 1e-12
    return (np.abs(nearest) / maximum).sum(axis=1)


def persistence_landmarks(
    n: int, edges: np.ndarray, weights: np.ndarray, k: int = 10
) -> np.ndarray:
    """Return node indices whose topological classes persist the longest."""
    edges = np.asarray(edges, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if edges.shape[0] == 0:
        return np.arange(min(k, n), dtype=np.int64)
    parent = list(range(n))
    members = {i: [i] for i in range(n)}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    death = np.full(n, -np.inf, dtype=np.float64)
    order = np.argsort(weights, kind="stable")
    for edge_index in order:
        u = int(edges[edge_index, 0])
        v = int(edges[edge_index, 1])
        weight = float(weights[edge_index])
        root_u, root_v = find(u), find(v)
        if root_u == root_v:
            continue
        merged_nodes = members[root_u] + members[root_v]
        for node in merged_nodes:
            death[node] = weight
        if len(members[root_u]) < len(members[root_v]):
            root_u, root_v = root_v, root_u
        parent[root_v] = root_u
        members[root_u] = members[root_u] + members[root_v]
        del members[root_v]
    for root in list(members):
        for node in members[root]:
            death[node] = np.inf
    order_nodes = np.lexsort((np.arange(n), -death))
    return order_nodes[:min(k, n)].astype(np.int64)


def connected_components(n: int, edges: np.ndarray):
    """Return connected-component labels and the number of components."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for edge in edges:
        root_u, root_v = find(int(edge[0])), find(int(edge[1]))
        if root_u != root_v:
            parent[root_v] = root_u
    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64), int(labels.max()) + 1 if n else 0


# ------------------------------------------------------------------ routing
def route_topk(Q: np.ndarray, K: np.ndarray, k: int, causal: bool = True):
    """Return each query's k most similar keys, masking future keys if causal."""
    n = Q.shape[0]
    scores = Q @ K.T / np.sqrt(Q.shape[1])
    if causal:
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)
    k = int(min(k, n))
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row = np.arange(n)[:, None]
    logits = scores[row, idx]
    order = np.argsort(-logits, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    logits = np.take_along_axis(logits, order, axis=1)
    return idx, logits


def edges_from_topk(idx: np.ndarray) -> np.ndarray:
    """Return unique undirected edges from a (n, k) neighbor-index array."""
    n, k = idx.shape
    u = np.repeat(np.arange(n), k)
    v = idx.reshape(-1)
    mask = u < v
    u, v = u[mask], v[mask]
    if u.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    key = u.astype(np.int64) * n + v
    _, idx2 = np.unique(key, return_index=True)
    return np.stack([u[idx2], v[idx2]], axis=1)


def chain_edges(n: int) -> np.ndarray:
    """Return temporal-chain edges (i, i+1)."""
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64)
    return np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)


def unique_edges(*edge_sets):
    """Return the union of edge lists with duplicates removed."""
    nonempty = [edges for edges in edge_sets if edges.size]
    if not nonempty:
        return np.zeros((0, 2), dtype=np.int64)
    all_edges = np.concatenate(nonempty, axis=0)
    u = np.minimum(all_edges[:, 0], all_edges[:, 1])
    v = np.maximum(all_edges[:, 0], all_edges[:, 1])
    key = u.astype(np.int64) * (int(max(u.max(), v.max())) + 1) + v
    _, keep = np.unique(key, return_index=True)
    return np.stack([u[keep], v[keep]], axis=1)


# ------------------------------------------------------------- persistence
def kruskal_barcode(n: int, edges: np.ndarray, weights: np.ndarray):
    """Return the exact 0-dimensional persistence barcode and beta_0."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    m = edges.shape[0]
    order = np.argsort(weights) if m else ()
    birth = [0.0] * n
    pairs = []
    for edge_index in order:
        u = int(edges[edge_index, 0])
        v = int(edges[edge_index, 1])
        weight = float(weights[edge_index])
        root_u, root_v = find(u), find(v)
        if root_u == root_v:
            continue
        birth_value = min(birth[root_u], birth[root_v])
        pairs.append((birth_value, weight))
        if birth[root_u] > birth[root_v]:
            root_u, root_v = root_v, root_u
        parent[root_v] = root_u
        birth[root_u] = birth_value
    roots = {find(i) for i in range(n)}
    for root in roots:
        pairs.append((birth[root], float("inf")))
    return pairs, len(roots)


def betti_numbers(n: int, edges: np.ndarray) -> dict:
    """Return beta_0, beta_1, and graph size for a 1-dimensional complex."""
    m = edges.shape[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for edge in edges:
        root_u, root_v = find(int(edge[0])), find(int(edge[1]))
        if root_u != root_v:
            parent[root_v] = root_u
    components = len({find(i) for i in range(n)})
    beta1 = m - n + components
    return {
        "beta0": components,
        "beta1": int(max(beta1, 0)),
        "m": int(m),
        "n": int(n),
    }


def topo_descriptor(n: int, edges: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return a fixed-length topological signature of the routed complex."""
    betti = betti_numbers(n, edges)
    pairs, _ = kruskal_barcode(n, edges, weights)
    finite = [(birth, death) for birth, death in pairs if np.isfinite(death)]
    lifetimes = np.array(
        [death - birth for birth, death in finite], dtype=np.float64
    )
    if lifetimes.size:
        maximum = float(lifetimes.max())
        total = float(lifetimes.sum())
        mean = float(lifetimes.mean())
        longer_quarter = int((lifetimes > 0.25 * maximum).sum())
        longer_half = int((lifetimes > 0.50 * maximum).sum())
    else:
        maximum = total = mean = 0.0
        longer_quarter = longer_half = 0
    return np.array(
        [
            betti["beta0"],
            betti["beta1"],
            total,
            maximum,
            mean,
            longer_quarter,
            longer_half,
            betti["m"] / max(n, 1),
            np.log1p(betti["m"]),
        ],
        dtype=np.float64,
    )
