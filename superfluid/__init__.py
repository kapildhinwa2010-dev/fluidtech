"""
superfluid — a topological "fluid" architecture for sequence models.
Core idea
Replace the transformer's dense attention with a cellular sheaf diffusion:
X(t) = exp(-t L_F) X(0),      L_F = delta^T delta   (sheaf Laplacian)
The fluid "freezes" the data's geometry onto the space of global sections
(the Hodge limit).  A multi-scale band filter bank recovers the transient,
a top-k route head restores sharp retrieval, and a single pointwise
nonlinearity keeps the model from collapsing to a linear map.
Second idea: the EXACT tier
A diffusion smears, so an exact value must never live inside one.  The
`memory` module splits the memory in two:
L1  FROZEN REGISTER -- an append-only, write-once table of exact facts
(20 bytes each).  The value is an INTEGER token id, so a recall is
bit-exact rather than approximate.
L2  SUPERFLUID STATE -- a constant-size (M x r) state that carries the
aggregate.  It never grows, no matter how much is streamed through it.
L3  an external source -- a 6-byte pointer, unbounded, costs a call.
The two tiers are separate on purpose: the fluid answers WHERE, the register
answers WHAT.
"""
from .autodiff import Tensor, concat, softmax_rows, softplus
from .sheaf import (build_laplacian, sheaf_energy, balance_residual,
                    is_balanced, laplacian_from_dense_A, component_balance)
from .diffusion import (power_iteration_lmax, cheb_coeffs_exp, cheb_basis,
                        apply_bands, band_energies, spectral_gap, jam_time,
                        cheb_degree, exact_exp_action, spectral_summary,
                        band_times)
from .topology import (route_topk, edges_from_topk, chain_edges, unique_edges,
                       kruskal_barcode, betti_numbers, topo_descriptor,
                       connected_components, hodge_relevance, hodge_embedding,
                       hodge_edge_participation, local_density,
                       persistence_landmarks)
from .blocks import SuperfluidBlock, RMSNorm
from .model import SuperfluidLM, TinyTransformer
from .losses import (cross_entropy, cross_entropy_grad, cross_entropy_tensor,
                     band_decorrelation)
from .graft import SAABranch, GRAFT_STATE_BYTES, ctx_hash
from .memory import (FrozenRegister, SheafMemoryAdapter, logit_override,
                     budget_split, ctx_key, fnv1a64, embedding_signature,
                     SLOT_BYTES)

version = "1.2.0"

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