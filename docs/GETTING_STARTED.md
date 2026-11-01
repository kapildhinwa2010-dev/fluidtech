Getting started
Install
pip install .            # from a checkout
pip install -e ".[dev]"  # for the tests
The only runtime dependency is numpy.  There is no torch, no GPU and no
download: the whole project runs on a laptop.
python selftest.py                 # 38 machine-checked claims
python examples/quickstart.py      # the 60-second tour
python examples/graft_demo.py      # the capability, end to end
Attaching the layer to a model
The layer is one module with one insertion.  For a HuggingFace model:
import torch
from superfluid.torch_graft import SAABranchTorch
model = ...                                     # any causal LM
emb = model.get_input_embeddings()
branch = SAABranchTorch(M=64, r=8, d=emb.weight.shape[1], emb=emb,
                        ctx_len=1, read_gain=300.0, transform="pc1",
                        use_exact=True, use_lossy=False).to(device)
def forward_with_memory(model, branch, input_ids):
    h = model.get_input_embeddings()(input_ids)
    h, info = branch(h[0], h[0], input_ids[0])
    return model(inputs_embeds=h.unsqueeze(0), attention_mask=...)
Two rules, and both matter:
Attach with `Wo_exact = 0` (the default).  Then the branch is an
exact identity and the model's logits are bit-identical to before.
You cannot regress the model by attaching it.
Arm the exact tier deliberately, when you want retrieval:
branch.Wo_exact.data = torch.eye(d)     # the identity map
branch.read_scale = 300.0                # a CALIBRATED scalar
The gain is not a guess — it is calibrated against the model's own logit
gap (see the sweep in `results/e6_graft.json`).  Past a certain point the
curve saturates; injecting more only makes the model more confident, not
more correct.
Streaming a session
The layer is a memory: its state persists across calls, and it is the
content of the stream that it remembers, not the position.
branch.reset()                      # a new session
for window in windows:              # 128-token windows, in order
    h = model.get_input_embeddings()(window)
    h, _ = branch.forward(h, window)
    ...                              # run the model on h
`branch.reset()` starts a new session.  Do this per user / per session:
one branch instance = one memory.
The store is bounded (`capacity`, 20 B per cached fact) and evicts the
oldest entry when it is full.  The register refuses to overflow rather
than growing without limit.
The store is first-write-wins: a fact that is written twice keeps its
first value, and a later write of the same context is ignored.  That
makes the memory an append-only audit trail.
What it will and will not do
Read `docs/GRAFT.md` before you build on it.  In one line:
it will let a frozen model quote something it saw earlier, even
outside its attention window, bit-exactly;
it will not make the model better at language, extend its attention
window, or let it reason over what it retrieves.
Serving notes
One `SAABranch` per session.  The state is a plain object; there is no
shared global state.
The state is not encrypted and not tenant-isolated: if you serve many
users, give each its own instance.
Retrieved content is untrusted.  See `SECURITY.md`.
Using the numpy implementation
The numpy implementation is the reference; the torch one mirrors it.
from superfluid import SAABranch, TinyTransformer
model = TinyTransformer(vocab=32, d=32, n_layers=2, n_ctx=32, seed=0)
branch = SAABranch(M=16, r=4, d=32, emb=model.emb, ctx_len=1)
h = model.emb[tok] + model.pos[:len(tok)]
h, info = branch.forward(h, model.emb[tok], tok)
logits = model.forward_from_h(h)      # <- the frozen model continues
`TinyTransformer.forward` was split into `forward` (embed + continue) and
`forward_from_h` (continue from a given residual stream) precisely so that a
frozen model can be driven from a modified stream.