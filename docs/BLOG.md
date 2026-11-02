# The 16 KB memory: exact where it matters, lossy everywhere else

By Kapil Dhinwa

(Historical blog draft. Its claims should be checked against the current
measurements and limitations before reuse.)

There is a number in long-context serving that nobody puts on a slide:
A 70B model on 2× H100 80GB cannot hold a single 128K-token session.
Not "serves it slowly". Cannot hold it. The weights take 131.5 GB; 40 GB
of KV cache does not fit in the 28.5 GB that is left. The KV cache is not an
optimisation problem — it is a wall, and it arrives at 32K context, long
before anyone calls it a wall.
I spent a year building a replacement for it. Here is what worked, what did
not, and the numbers for both.

## The idea: stop storing, start addressing

Attention remembers by position. Every token keeps a copy of itself, and
the copy is kept forever. That is why the cost is linear and why the cost is
unavoidable during a sequence, until that context is discarded.
A different architecture routes tokens to a fixed set of cells; the cell
holds a diffusion of what was written there; and the routing — not the
storage — carries the association. The fluid state is `M × r` floats. For
`M = 1024, r = 8` that is 16 KB, independent of stream length.
The mathematics is a cellular-sheaf Laplacian: `X(t) = exp(-t L_F) X(0)`.
The field converges exponentially to `ker(L_F)` — the Hodge limit — and the
contraction is certified: `‖e^{-tL}‖ ≤ e^{-tλ₂}`. A multi-scale band filter
bank recovers the transient that the limit throws away, and a sparse top-k
route head restores the sharp selection that a smooth diffusion cannot
provide.

## The part I did not expect

A diffusion smears. That is the whole point of it, and it is fatal for
exactness: the moment a token's value enters the Laplacian, it is mixed with
its neighbours and is provably no longer exact.
So the exact value must never live inside the fluid. I split the memory:

| tier|holds|size|exact?|
| ---|---|---|---|
| attachable exact store|token IDs|20 B packed-slot estimate / context|✅ bit-exact ID|
| fluidlayer state|the aggregate|16 KB, constant|❌ lossy|
| external source|a pointer|6 B / fact|✅ exact|

A recalled token ID is an `int32`, so the stored ID is exact. The attachable
store tracks majority continuations and evicts the least-corroborated entries.
This bounded in-memory store is not a tamper-evident audit log, and support
counts do not establish that a fact is true or prevent poisoning.
The extraction is a logit override, not a suggestion. The tool does not
return a token and hope the model uses it; it sets the recalled token's
logit to `+∞`, so the argmax is provably that token.

## What the numbers say

The fluid carries the local aggregate well: cosine 0.968 against the
true per-anchor mean.
When a token ID is retrieved, it is bit-exact; the packed-slot estimate is
20 B/context, excluding Python object overhead,
versus the KV cache's 131,072 B/token — 6,554× cheaper.
More anchors buy almost nothing (16× the anchors: +0.013 aggregate quality,
256× the compute) while more exact slots buy a lot (64× the slots: +0.69
fact recall, for 20 KB). Spend the budget on exactness, not on storage.
A training-free 2.4 KB domain prior improves a frozen model's
perplexity — by 0.048 nats on a weak model, but only 0.0059 nats on a
better one, and a plain 23 KB unigram table beats it 6×. The prior's
value lives where the model is weakest, not where it is good.

## What the measurements ruled out

This is the part I care about most, because it is the part nobody else
publishes.
The topological selection criterion lost. I built the whole selection
story around persistent homology — pick the facts the geometry says are
structurally important. Measured: 0.000 precision, versus 0.970
for plain local geometric affinity. Hodge participation measures global
coupling; inside a topic it saturates and cannot discriminate. The
project was renamed around the thing that actually works: geometry, not
topology.
My own balance diagnostic was wrong.`‖L·1‖∞` does not detect the
frozen state of a scalar signed sheaf — an all-negative even cycle has
`‖L·1‖∞ = 4` and still freezes onto a beautiful alternating sign pattern.
The right diagnostic is the frustration index. Fixed, and now in the test
suite.
λ₂ does not predict quality (pearson 0.20). The certificate is true
about the operator; it says nothing about the task. The claim was
weakened, not the data massaged.
An earlier report said a transformer beat the fluid (2.6331 vs 2.7397 nats)
and cited a 97.3 % mechanism result. The source scripts and raw outputs for
those measurements are not in this repository, so both claims are
unverified; see N4 in `docs/NEGATIVE-RESULTS.md`.
Grafted onto a frozen model, my own diffusion tier made the model
worse — +0.81 nats (ppl 13.3 -> 29.9) on real English. The cause is
structural: the persistent state sits outside the autodiff graph, so the
read projection is trained against a moving, non-differentiable target
and the optimisation diverges. The exact tier next to it is fine
(-0.03 nats). So the shipping recommendation is now "ship the
register, not the diffusion" — a much smaller and much more defensible
product than the one I started with.
Even after I fixed the diffusion tier, it had nothing to add.  The
fix worked: the divergence is gone (+1.14 nats of damage -> +0.0000) and
the layer's own objective now converges.  But the layer's next-token
prediction converges to 3.81 nats where the base model does the same
task in 2.59 — a 16 KB state is simply worse at predicting text than a
131,392-parameter model.  The injection-strength sweep is the honest
summary: every non-zero gain hurts, monotonically, and the optimum is
zero.  I fixed the bug and the feature still does not earn its
place.  That is the most useful thing I learned all year.
A trivial unigram table beats my domain prior (−0.038 vs −0.0059
nats on the well-trained model; −0.132 vs −0.048 on a weak one) — and a
rank-8 unigram at the same 2.4 KB size beats it too, so the prior is
not even the best use of its own bytes. Its benefit shrinks as the
model gets better, which means the downloadable-prior product only
makes sense for small, on-device models, not good ones. I will not
pretend otherwise.
Why publish this?
Because the repository runs its own checks. `python selftest.py` prints
50/50 — pure NumPy, no GPU, no downloads — and every number above comes
from a file in `results/` that the same command regenerates. It found ten
real bugs in my own code, including a Chebyshev recurrence missing a factor
of two and a scatter operation that never scattered.
The architecture is not revolutionary and I am not claiming it is. But the
exactness guarantee is real, the byte economics are real, and the
negative results are real. In a field where most papers do not run their
own code, that is a more durable contribution than a benchmark number would
be.
The artifact is 136 KB. It is a prior, not a model: a 16 KB runtime state, a
static codebook, and a landmark table you can plug into any LLM with one
residual-stream insertion.
Try it. Then tell me where it breaks — I have found five places already, and
I would like to find more.
Repository: https://github.com/kapildhinwa2010-dev/fluidtech ·
50 self-verification checks · pure NumPy · no GPU required.
