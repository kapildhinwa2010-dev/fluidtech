"""The graft: attaching it must be a no-op, and the exact tier must be exact."""
import numpy as np
import pytest

from superfluid import SAABranch, TinyTransformer
from superfluid.graft import ctx_hash


@pytest.fixture
def model():
    return TinyTransformer(32, d=32, n_layers=2, n_ctx=32, seed=0)


def test_attaching_is_bit_identical(model):
    br = SAABranch(M=16, r=4, d=32, emb=model.emb, ctx_len=1, seed=0)
    tok = np.array([3, 7, 11, 5, 9, 2, 8, 13], dtype=np.int64)
    lg0 = model.forward(tok)
    h0 = model.emb[tok] + model.pos[:len(tok)]
    h1, _ = br.forward(h0, model.emb[tok], tok)
    lg1 = model.forward_from_h(h1)
    assert np.array_equal(lg0.data, lg1.data)


def test_the_state_stays_bounded(model):
    br = SAABranch(M=16, r=4, d=32, emb=model.emb, ctx_len=1, capacity=64,
                   seed=0)
    tok = np.array([3, 7, 11, 5, 9, 2, 8, 13], dtype=np.int64)
    h0 = model.emb[tok] + model.pos[:len(tok)]
    sizes = []
    for _ in range(20):
        br.forward(h0, model.emb[tok], tok)
        sizes.append(br.state_bytes)
    assert max(sizes) <= 16 * 4 * 8 + 64 * 20 + 1024


def test_the_exact_tier_returns_the_stored_id(model):
    br = SAABranch(M=16, r=4, d=32, emb=model.emb, ctx_len=1, seed=0)
    seq = np.array([4, 21, 6, 30, 12, 17, 2, 25], dtype=np.int64)
    br.reset()
    br.forward(model.emb[seq] + model.pos[:len(seq)], model.emb[seq], seq)
    # the store maps a CONTEXT to the token that followed it
    assert br.store[int(seq[-2])] == int(seq[-1])


def test_the_store_tracks_the_majority_continuation(model):
    br = SAABranch(M=16, r=4, d=32, emb=model.emb, ctx_len=1, seed=0)
    br.reset()
    seq = np.array([9, 3, 9, 7, 9, 7], dtype=np.int64)
    br.forward(model.emb[seq] + model.pos[:len(seq)], model.emb[seq], seq)
    assert br.store[9] == 7


def test_a_reset_clears_the_memory(model):
    br = SAABranch(M=16, r=4, d=32, emb=model.emb, ctx_len=1, seed=0)
    seq = np.array([4, 21, 6, 30], dtype=np.int64)
    br.forward(model.emb[seq] + model.pos[:len(seq)], model.emb[seq], seq)
    assert len(br.store) > 0
    br.reset()
    assert len(br.store) == 0
    assert not br.state.any()


def test_the_layer_local_head_produces_logits(model):
    br = SAABranch(M=16, r=4, d=32, emb=model.emb, ctx_len=1,
                   state_mode="rebuild", hist_len=4, seed=0)
    tok = np.array([1, 2, 3, 4], dtype=np.int64)
    h0 = model.emb[tok] + model.pos[:len(tok)]
    _, info = br.forward(h0, model.emb[tok], tok)
    z = br.layer_logits(info)
    assert z is not None and z.data.shape == (4, 32)


def test_the_rebuild_state_is_the_diffusion_of_the_history_writes(model):
    """The rebuild mode must be exactly exp(-tL) applied to the writes of
    the window -- computed independently here, by hand.

    (It is NOT the same value as the persistent accumulator: the persistent
    mode diffuses after every write, so it is an exponentially-weighted sum
    of the past writes, while the rebuild is the diffusion of the LAST
    hist_len windows' writes.  Two different estimators of the same
    aggregate; the rebuild one is the differentiable one.)
    """
    tok = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    h0 = model.emb[tok] + model.pos[:len(tok)]
    br = SAABranch(M=16, r=4, d=32, emb=model.emb, ctx_len=1,
                   state_mode="rebuild", hist_len=8, seed=0)
    br.forward(h0, model.emb[tok], tok)
    # the write of this window, by hand
    w_np, _ = br._route(model.emb[tok].data)
    write = w_np.T @ (h0.data @ br.Wv.data)
    # exp(-tL) by the same Chebyshev expansion, by hand
    n = br.M
    Lt = (2.0 / br.lam_max) * br.L - np.eye(n)
    Y = [write]
    if br.cheb_m >= 1:
        Y.append(Lt @ Y[0])
        for k in range(2, br.cheb_m + 1):
            Y.append(2.0 * (Lt @ Y[k - 1]) - Y[k - 2])
    expect = np.zeros_like(write)
    for k, a in enumerate(br._cheb):
        if k < len(Y):
            expect += a * Y[k]
    assert np.allclose(br.state, expect, atol=1e-9)


def test_bad_arguments_are_refused():
    """A production layer refuses a misconfiguration loudly."""
    from superfluid import TinyTransformer
    m = TinyTransformer(32, d=32, n_layers=1, n_ctx=8, seed=0)
    for kwargs in ({"M": 0}, {"r": 0}, {"capacity": 0},
                   {"state_mode": "nonsense"}, {"read_gain": "nonsense"},
                   {"read_transform": "nonsense"}, {"d": 999}):
        try:
            SAABranch(emb=m.emb, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"SAABranch accepted a bad argument: {kwargs}")


def test_ctx_hash_is_stable():
    assert ctx_hash([1, 2, 3]) == ctx_hash([1, 2, 3])
    assert ctx_hash([1, 2, 3]) != ctx_hash([3, 2, 1])
