"""
superfluid — a topological "fluid" architecture for sequence models.
Core idea
Replace the transformer's dense attention with a cellular sheaf diffusion:
X(t) = exp(-t L_F) X(0),      L_F = delta^T delta   (sheaf Laplacian)
The fluid "freezes" the data's geometry onto the space of global sections
(the Hodge limit).  A multi-scale band filter bank recovers the transient,
a top-k route head restores sharp retrieval, and a single pointwise
nonlinearity keeps the model from collapsing to a linear map.
Exact token IDs live outside the diffused state. The attachable `SAABranch`
uses a bounded, majority-tracking exact store alongside its constant-size
fluid state. The low-level `FrozenRegister` helper separately provides an
append-only 20-byte slot format with explicit serialization. These logical
slot sizes exclude Python object overhead in the attachable reference store.
"""
from .autodiff import Tensor, concat, softmax_rows, softplus
from .blocks import RMSNorm, SuperfluidBlock
from .diffusion import (
                    apply_bands,
                    band_energies,
                    band_times,
                    cheb_basis,
                    cheb_coeffs_exp,
                    cheb_degree,
                    exact_exp_action,
                    jam_time,
                    power_iteration_lmax,
                    spectral_gap,
                    spectral_summary,
)
from .graft import GRAFT_STATE_BYTES, SAABranch, ctx_hash
from .losses import (
                    band_decorrelation,
                    cross_entropy,
                    cross_entropy_grad,
                    cross_entropy_tensor,
)
from .memory import (
                    SLOT_BYTES,
                    FrozenRegister,
                    SheafMemoryAdapter,
                    budget_split,
                    ctx_key,
                    embedding_signature,
                    fnv1a64,
                    logit_override,
)
from .model import SuperfluidLM, TinyTransformer
from .sheaf import (
                    balance_residual,
                    build_laplacian,
                    component_balance,
                    is_balanced,
                    laplacian_from_dense_A,
                    sheaf_energy,
)
from .topology import (
                    betti_numbers,
                    chain_edges,
                    connected_components,
                    edges_from_topk,
                    hodge_edge_participation,
                    hodge_embedding,
                    hodge_relevance,
                    kruskal_barcode,
                    local_density,
                    persistence_landmarks,
                    route_topk,
                    topo_descriptor,
                    unique_edges,
)

__version__ = "1.2.0"
version = __version__

__all__ = [
    "SAABranch", "GRAFT_STATE_BYTES", "ctx_hash",
    "Tensor", "concat", "softmax_rows", "softplus",
    "build_laplacian", "sheaf_energy", "balance_residual", "is_balanced",
    "laplacian_from_dense_A", "component_balance",
    "power_iteration_lmax", "cheb_coeffs_exp", "cheb_basis",
    "apply_bands", "band_energies", "spectral_gap", "jam_time",
    "cheb_degree", "exact_exp_action", "spectral_summary",
    "band_times",
    "route_topk", "edges_from_topk", "chain_edges", "unique_edges",
    "kruskal_barcode", "betti_numbers", "topo_descriptor",
    "connected_components",
    "hodge_relevance", "hodge_embedding", "hodge_edge_participation",
    "local_density", "persistence_landmarks",
    "SuperfluidBlock", "RMSNorm", "SuperfluidLM", "TinyTransformer",
    "cross_entropy", "cross_entropy_grad", "cross_entropy_tensor",
    "band_decorrelation",
    # --- the exact tier -------------------------------------------------
    "FrozenRegister", "SheafMemoryAdapter", "logit_override", "budget_split",
    "ctx_key", "fnv1a64", "embedding_signature", "SLOT_BYTES",
]
