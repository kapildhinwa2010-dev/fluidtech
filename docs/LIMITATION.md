Limitations
This is the honest list.  It is long on purpose: the point of the project is
that the limits are measured, not implied.
What the layer does NOT do
It does not make the model better at language.  It makes the model
better at remembering what it has already seen.  The base model's own
modelling quality is untouched.
It does not extend the attention window.  The window is still 128
tokens.  The layer can quote something from outside it; it cannot
reason over what it retrieves.
It cannot compose.  The store is a cache of raw continuations.  It
does not answer questions about what it holds, does not infer, and does
not do arithmetic.
It is not a KV cache replacement.  The KV cache is position-indexed;
the layer is content-indexed.  They are complementary (see
`docs/COMPARISON.md`).
Measured limits
| limit|the number|
| ---|---|
| the cross-window recall ceiling|0.800 , not 1.0 — the injection is not perfectly selective with a tied readout|
| the diffusion tier's effect on a frozen model|+0.81 nats  if trained through the base model's loss;  0.0000  after the fix; and  no upside  either way|
| the domain prior's benefit|−0.0059 nats  on a well-trained model (it shrinks as the model improves)|
| the layer's memory|6.3 KB for  M=64, r=8 , plus 20 B per cached fact|
| the exact tier's training-free ceiling|a  scalar  gain, calibrated; past ~30 it saturates|
The design limits that are known and accepted
The store is a cache, not a database.  Bounded, first-write-wins,
FIFO eviction.  It quotes; it does not query.
The injection is not selective by default.  With a tied readout,
injecting a token's embedding boosts its embedding-space neighbours too.
The `pc1` transform projects out the dominant shared direction and buys
+0.075 of recall (0.725 → 0.800).  A perfect one-hot would need the
pseudo-inverse of the embedding matrix, which is numerically unusable.
The persistent state cannot be differentiated.  That is the whole
reason the diffusion tier's training diverged (N6/N7).  The `rebuild`
mode is the workaround, and it is a truncated estimator.
The route head is not differentiable through its hard top-k; only the
soft weights carry a gradient.
The measurement limits
Everything here was run on CPU, at small scale, on a character-level
or 512-token subword corpus, with a 131,392-parameter model.  The
subword repeat (`--tokenizer bpe`) is the closest thing to a realistic
tokenizer in this repository.
The GPU run was not executed here.`experiments/gpu_graft.py` is
written to be run on a GPU box; nothing in this repository should be read
as a claim about a 1B+ model until that run happens.
The synthetic corpus flatters the layer.  It is a Markov chain with
70 % self-loops, so "what followed this context" is right far more often
than in real text.  The real-corpus numbers are the ones to quote.
Known bugs this project found in its own code
Ten of them, all found by running the code, all fixed, all documented in
`docs/NEGATIVE-RESULTS.md` and `CHANGELOG.md`.  Among them: a Chebyshev
recurrence missing a factor of two, a scatter operation that never
scattered, a gradient that was zeroed after the backward pass, an `exp(+α)`
overflow, a balance diagnostic that did not detect the frozen state, a
per-anchor store that was evicted by ordinary filler tokens, and an
injection that made the wrong token win.