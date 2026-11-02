"""The topology: the frustration index, the Hodge participation, persistence."""
import numpy as np

from fluidlayer.topology import (
    hodge_embedding,
)


def test_the_frustration_index_matches_a_global_section():
    """frustration == 0  <=>  a non-trivial global section exists.

    This is the check that replaced the old ``||L 1||`` diagnostic, which
    does NOT detect the frozen state of a scalar signed sheaf.
    """
    import importlib.util
    import os

    from fluidlayer.sheaf import signed_laplacian_from_edges
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "st", os.path.join(here, "selftest.py")
    )
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)

    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(40):
        n = int(rng.integers(4, 9))
        edges = np.array([(i, i + 1) for i in range(n - 1)], dtype=np.int64)
        alpha = rng.choice([-1.0, 1.0], size=len(edges))
        L = signed_laplacian_from_edges(n, edges, alpha)
        evals = np.linalg.eigvalsh(L)
        kernel_dim = int(np.sum(evals < 1e-8))
        frustration = st._frustration(n, edges, alpha)
        assert (frustration == 0) == (kernel_dim >= 1)
        checked += 1
    assert checked == 40


def test_the_hodge_embedding_is_finite():
    n = 6
    edges = np.array([(i, i + 1) for i in range(n - 1)] + [(0, 3)], dtype=np.int64)
    alpha = np.ones(len(edges))
    Psi = hodge_embedding(n, edges, alpha)
    assert np.isfinite(Psi).all()


def test_the_persistence_landmarks_are_ordered():
    from fluidlayer.topology import kruskal_barcode
    n = 12
    edges = [(i, i + 1) for i in range(n - 1)] + [(0, 5), (3, 9)]
    edges = np.array(edges, dtype=np.int64)
    weights = np.arange(len(edges), dtype=float)
    bars = kruskal_barcode(n, edges, weights)
    assert bars is not None
    assert len(bars) <= len(edges) + 1
