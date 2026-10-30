API reference
Everything below is importable from the package root:
from superfluid import SAABranch, FrozenRegister, TinyTransformer, ctx_key
`SAABranch` — the attachable memory layer
SAABranch(M=64, r=8, d=None, emb=None, k=4, cheb_m=8, t=1.0, seed=0,
          use_exact=True, use_lossy=False, capacity=4096, ctx_len=1,
          trainable_route=True, read_gain="norm", center=True,
          read_transform="pc1", state_mode="persistent", hist_len=8,
          inject_mode="state")
| argument|what it does|
| ---|---|
| M ,  r|the anchors and their rank.  The lossy tier's state is  M × r  floats —  constant size , whatever you stream through it.|
| emb|the model's embedding table.  Needed to turn a stored token id back into a vector.  Shared, never copied.|
| ctx_len|how many trailing tokens form the content key.   1  =  "what followed this token ";  k  = a k-gram memory.|
| capacity|max entries in the exact store (20 B each).  The oldest is evicted.|
| read_gain|"norm"  (match the residual's scale) or  "raw"  (a fixed scalar, used with  read_scale ).|
| read_transform|"raw" ,  "center"  or  "pc1"  — how selective the injection is.   "pc1"  is the one that works  with a tied readout.|
| state_mode|"persistent"  (a numpy accumulator, not differentiable) or  "rebuild"  (rebuilt inside the graph from the last  hist_len  windows — differentiable).|
| inject_mode|"state"  (the raw diffused read) or  "prior"  (the layer's own predicted distribution, projected through the embedding table).|
| use_exact ,  use_lossy|the two tiers, independently switchable.   use_lossy=False  is the recommended setting  — see  docs/NEGATIVE-RESULTS.md  N6/N7.|
Methods
| method|what it does|
| ---|---|
| forward(h, content, token_ids)|returns  (h_out, info) .   h  is the residual stream,  content  the  position-free  representation used for routing,  token_ids  the tokens consumed.|
| reset()|clears the memory (a new session).|
| refresh_contrast()|recompute the transformed embedding table.   Call it after the model is trained , or after the embeddings change.|
| layer_logits(info)|the layer's own next-token prediction (the layer-local objective).|
| state_bytes|the size of the persistent state.|
| params()|the trainable parameters (for the layer-local training).|
Properties
Attaching is a no-op.`Wo_exact` and `Wo_lossy` start at zero, so
`h_out == h` bit for bit until you train something.
Causality.  The store is written from the pair (context ending at
t-1, token at t), which is complete only at time t.  A token can never
be read back before it was written, and the model can never "copy the
current token" through the store.
First-write-wins.  A context that is seen twice keeps its first value.
The memory is an append-only audit trail.
`FrozenRegister` — the exact store on its own
reg = FrozenRegister(capacity=1024)
reg.promote(value_token_id, key, sig=0, anchor=0, doc_id=0, trust=0)
slot, value = reg.recall(key, sig=None)
a value is stored as an int32 — bit-exact, never approximate;
`SLOT_BYTES == 20`: `key uint64 | val int32 | sig uint32 | anchor uint16 | meta uint16`;
`recall(key, sig)`fails loud: if the signature does not match, it
returns `None` instead of another fact's value;
`promote`refuses to overflow (`MemoryError`) rather than growing —
a register that silently grows becomes a KV cache;
`to_bytes()` / `from_bytes(blob, capacity)` for a durable store.
`budget_split(total_bytes, M, r, slot_bytes)`
Splits a byte budget between the fluid state and the register, never
over-committing: if the requested `M` does not fit, `M` is reduced and
`M_reduced` is set.
`ctx_key(tokens)` / `ctx_hash(tokens)`
A 64-bit FNV-1a hash of a token sequence — the content key of the exact
store.
`logit_override(logits, token_ids)`
Forces `token_ids` to win the argmax, deterministically.  Accepts a numpy
array or a framework `Tensor`.  This is the difference between "the
model was told the answer" and "the model cannot disagree".
`TinyTransformer.forward_from_h(h, return_hidden=False)`
Continues the forward pass from a (possibly modified) residual stream.  This
is the hook a grafted memory layer needs.​a value is stored as an int32 — bit-exact, never approximate;  
​SLOT_BYTES == 20: key uint64 | val int32 | sig uint32 | anchor uint16 | meta uint16;  
​recall(key, sig) fails loud: if the signature does not match, it
returns None instead of another fact's value;  
​promote refuses to overflow (MemoryError) rather than growing —
a register that silently grows becomes a KV cache;  
​to_bytes() / from_bytes(blob, capacity) for a durable store.  
​budget_split(total_bytes, M, r, slot_bytes)
​Splits a byte budget between the fluid state and the register, never
over-committing: if the requested M does not fit, M is reduced and
M_reduced is set.  
​ctx_key(tokens) / ctx_hash(tokens)
​A 64-bit FNV-1a hash of a token sequence — the content key of the exact
store.  
​logit_override(logits, token_ids)
​Forces token_ids to win the argmax, deterministically.  Accepts a numpy
array or a framework Tensor.  This is the difference between "the
model was told the answer" and "the model cannot disagree".  
​TinyTransformer.forward_from_h(h, return_hidden=False)
​Continues the forward pass from a (possibly modified) residual stream.  This
is the hook a grafted memory layer needs.  