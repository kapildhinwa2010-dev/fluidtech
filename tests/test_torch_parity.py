"""The numpy layer and the torch twin must agree.

This is the test that the "the torch version is just a port" claim rests on.
It is SKIPPED when torch is not installed -- the repository stays
numpy-only by default -- but it runs wherever torch is available.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _make_pair(seed=0, M=16, d=32, r=4, capacity=4096):
    from fluidlayer import SAABranch
    from fluidlayer.model import TinyTransformer
    from fluidlayer.torch_graft import SAABranchTorch
    m = TinyTransformer(32, d=d, n_layers=2, n_ctx=32, seed=seed)
    np_br = SAABranch(M=M, r=r, d=d, emb=m.emb, ctx_len=1,
                      capacity=capacity, seed=seed)
    th_br = SAABranchTorch(M=M, r=r, d=d, emb=torch.nn.Parameter(
        torch.tensor(m.emb.data, dtype=torch.float32),
        requires_grad=False), ctx_len=1, capacity=capacity, seed=seed)
    # copy the route head so both routers agree
    with torch.no_grad():
        th_br.protos.data.copy_(torch.tensor(np_br.protos.data))
        th_br.Wq.data.copy_(torch.tensor(np_br.Wq.data))
        th_br.Wk.data.copy_(torch.tensor(np_br.Wk.data))
    return m, np_br, th_br


def test_the_two_routers_agree():
    m, np_br, th_br = _make_pair()
    x = np.array([4, 21, 6, 30, 12, 17, 2, 25], dtype=np.int64)
    h = m.emb[x] + m.pos[:len(x)]
    np_br.reset()
    th_br.reset()
    np_br.forward(h, m.emb[x], x)
    # the content is the POSITION-FREE embedding in both implementations
    th = torch.tensor(m.emb[x].data, dtype=torch.float32)
    th_br.forward(torch.tensor(h.data, dtype=torch.float32), th,
                  torch.tensor(x))
    assert np_br.store.get(2) == th_br.store.get(2), "the stores disagree"
    assert len(np_br.store) == len(th_br.store)


def test_the_gated_override_agrees():
    m, np_br, th_br = _make_pair()
    x = np.array([7, 19, 2], dtype=np.int64)
    h = m.emb[x] + m.pos[:len(x)]
    np_br.reset()
    th_br.reset()
    seq = np.array([4, 21, 6, 30, 12, 17, 2, 25], dtype=np.int64)
    np_br.forward(m.emb[seq] + m.pos[:len(seq)], m.emb[seq], seq)
    th = torch.tensor(m.emb[seq].data, dtype=torch.float32)
    th_br.forward(torch.tensor(m.emb[seq].data + m.pos[:len(seq)].data,
                               dtype=torch.float32), th, torch.tensor(seq))
    _, info_n = np_br.forward(h, m.emb[x], x)
    _, info_t = th_br.forward(torch.tensor(h.data, dtype=torch.float32),
                              torch.tensor(m.emb[x].data, dtype=torch.float32),
                              torch.tensor(x))
    lg_n = np_br.gated_logits(np.zeros((3, 32)), info_n, gain=8.0)
    lg_t = th_br.gated_logits(torch.zeros(3, 32), info_t, gain=8.0)
    assert np.allclose(lg_n, lg_t.numpy(), atol=1e-5), "the gated overrides disagree"


def test_majority_tracking_and_evidence_eviction_agree():
    m, np_br, th_br = _make_pair(capacity=2)
    x = np.array([1, 9, 1, 9, 1, 9, 10, 11, 12, 13, 14, 15],
                 dtype=np.int64)
    h = m.emb[x]
    np_br.forward(h, h, x)
    th = torch.tensor(h.data, dtype=torch.float32)
    th_br.forward(th, th, torch.tensor(x))

    assert np_br.store.value == th_br.store.value
    assert np_br.store.support == th_br.store.support
    assert np_br.store.disagree == th_br.store.disagree
    assert np_br.store.get(1) == th_br.store.get(1) == 9
    assert len(np_br.store) == len(th_br.store) == 2
