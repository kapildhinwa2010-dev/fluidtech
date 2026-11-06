# NEGATIVE RESULTS — what did not work, measured and reported

> **Read this before the README.** This file is the reason the rest of the
> repository is worth trusting.
>
> Eleven negative or unexpected findings are documented here, with raw result
> files linked where available. Archived N4 and N11 JSON outputs are included,
> but their original training script/checkpoint environments are not fully
> present, so those historical runs are not independently repeatable as-is.
>
> If you only read one file in this repo, read this one.  It is the honest
> map of the project's real shape.

---

## Summary table

| # | what we expected | what happened | where |
|---|---|---|---|
| N1 | the **topological** selection criterion would pick the right facts | it **does not** — 0.000 precision vs 0.970 for plain geometric affinity | `results/e5_graft.json` |
| N2 | the `‖L·1‖∞` **balance residual** detects the frozen state | it **does not** — the right diagnostic is the frustration index | `results/e4_gbalance.json` |
| N3 | `λ₂` predicts long-context quality | **barely** — pearson 0.1904, spearman 0.2499 | `results/e4_gbalance.json` |
| N4 | a **transformer** should lose to the fluid at some scale | the archived run says the transformer **wins**; the original training source is not included | `results/train_results.json` |
| N5 | a 16 KB domain prior should beat a trivial frequency table | it **helps** (−0.006 nats) but a plain **unigram table wins 6× more** (−0.038), and the gain **shrinks as the model improves** | `results/e1_prior.json` |
| N6 | grafting the **diffusion tier** onto a frozen model should be neutral or helpful | it **degrades the model by +0.81 nats** on real text (ppl 13.3 → 29.9) — the *exact* tier alone is fine (−0.03) | `results/e6_real.json` |
| N7 | the diffusion tier's **divergence can be fixed** | **yes, the damage is gone** (+1.14 → **+0.0000**) — but the fixed tier had **no upside** (the optimal injection gain was **zero**).  **RESOLVED by the router** (E12): as the fallback tier it now measurably helps | `results/e7_lossy_fix.json` |
| N8 | one injection gain can serve **both** recall and generation | it could not — the two optimum gains were **1000× apart** (300 vs 0.3), and the recall gain **ruined** the model (+1.08 nats).  **RESOLVED by the gate** (`u = 1 − max p`) | `results/e10_gate.json` |
| N9 | the memory can be made **exact** | **yes**: the recall **1.000**, the common-key case **1.000** (was 0/3), the corroborated entries survive the eviction (**40/40**), and the base model's perplexity **improves** to **11.78**.  Two plausible-looking rules were tried and **rejected** (both off by default) | `results/e11_exact.json` |
| N10 | the layer should beat the closest prior art at every budget | the gate wins in the bounded-memory regime, but kNN-LM wins with a 1 MB store | `results/e13_knn.json` |
| N11 | the small-model advantage should generalize to a real pretrained model | the archived GPT-2 output shows the kNN-LM interpolation wins on perplexity; checkpoint required to rerun | `results/e14_real_gpt2.json` |

---

## N1 — The topological selection criterion does not work

### The claim we set out to test

`docs/DESIGN_pinned_memory.md` §4 proposed **S4: topology selects the facts.**
Build a Vietoris–Rips filtration over the corpus, take the longest bars
(the structurally important classes), and spend the exactness budget there.
The argument was appealing: the criterion is computable **once per document**
with **zero per-token cost**, whereas Titans' surprise metric needs a gradient
probe per token.  Prior art: "Topologically Relevant Landmark Selection on
Point Clouds" (arXiv 2610.00019) does the Hodge-harmonic-participation
version of the same idea for PH subsampling.

### The experiment

`experiments/e5_graft.py` — 1,000,000 tokens, 195 injected facts, a budget of
100 exact slots, 11 selection policies, all getting the same budget and the
same register.  Only the choice of which contexts to promote differs.  No
policy is allowed to look at the fact list; the `crowding` policy uses only
the token→anchor histogram, which any system can compute in a streaming
pre-pass.

### The result

| policy (which contexts get the exact budget) | pin precision | fact recall |
|---|---|---|
| **semantic** (new pair + high geometric affinity) | **0.970** | **0.469** |
| rarity (the next token is a first occurrence) | 0.050 | 0.031 |
| persistence (a long-lived structural class) | 0.010 | 0.008 |
| **topology** (Hodge edge participation) | **0.000** | 0.000 |
| **topology_fine** (the same, 4× finer codebook) | **0.000** | 0.000 |
| **hodge** (per-node harmonic participation) | **0.000** | 0.000 |
| **crowding** (anchor occupancy) | **0.000** | 0.000 |
| **density** (local density of the cell) | **0.000** | 0.000 |
| surprise (a new pair, no affinity ranking) | 0.000 | 0.000 |
| random (the control) | 0.000 | 0.000 |
| oracle (the upper bound) | 1.000 | 0.538 |

### Why it failed — and it is not a bug

Hodge participation measures how strongly two cells are coupled **in the
global structure of the complex**.  Inside a topic every cell is coupled to
every other, so the measure **saturates** and carries no discriminative
information about whether a *particular* pair of tokens belongs together.

Raising the resolution 4× (`topology_fine`) does not help, which rules out
the obvious "we quantised too coarsely" explanation.  The problem is not
resolution; it is that the quantity is **global** while the task is **local**.

The plain **local** affinity — how close the two embeddings actually are —
is not saturated, and separates cleanly (0.970).

### What we changed because of it

* The claim in the docs was narrowed from *"topology selects the facts"* to
  *"geometry selects the facts"*.  The sheaf machinery is still what the
  **diffusion** uses; for **selection**, the fine geometry wins.
* `docs/DESIGN_pinned_memory.md` §9.5.2 records the table, the mechanism, and the
  consequence.
* `docs/POSITIONING.md` §4's "Gate 3" (topology vs learned gating) is
  **answered — and the answer is no.**

### What survived

Everything else.  The exactness mechanism is unaffected: `exact | promoted =
1.000` for every policy that promotes anything.  A 16 KB state with a
bounded majority-tracking store returns the stored token ID **bit-exactly**.
Its 20-byte slot figure is a packed-payload estimate. The
*mechanism* is sound; only the *selection heuristic* was wrong.

---

## N2 — The framework's own balance diagnostic was wrong

### The claim

`fluidlayer/sheaf.py` states:

> *"balance_residual(L) = ‖L·1‖∞.  Zero <=> the sheaf is (G-)balanced: every
> node's incident restriction maps sum to zero, so the constant field is a
> global section and the fluid has a non-trivial frozen state."*

### The counterexample

A cycle of **even** length with **all** edges carrying a negative restriction
map.  It is not balanced in the `‖L·1‖∞` sense (`‖L·1‖∞ = 4`), yet it
**freezes onto the alternating sign pattern** — a perfectly non-trivial
global section.

The two notions are genuinely different:

| quantity | question it answers | detects the alternating pattern? |
|---|---|---|
| `‖L·1‖∞` | *is the CONSTANT field a global section?* | **no** (it is 4) |
| **frustration index** | *does ANY global section exist?* | **yes** (it is 0) |

For scalar ±1 restriction maps the correct statement is

    frustration == 0   <=>   a non-trivial global section exists

### What we did

* Implemented the **frustration index** (the number of edges whose sign
  disagrees with the best global sign assignment, computed by propagating a
  sign assignment along a spanning forest and counting the inconsistent
  non-tree edges).
* Verified the equivalence on **150/150** random signed cycles
  (`results/e4_gbalance.json`).
* Added it to the selftest as checks 24 — including the explicit
  all-negative-even-cycle counterexample, so the old claim can never
  silently come back.
* Documented the distinction in `docs/FRAMEWORK.md` and `fluidlayer/sheaf.py`.

### The lesson

The old check was **not wrong in its own terms** — it measures a real and
stricter property.  But the *narrative* attached to it ("unbalanced ⇒
evaporates, balanced ⇒ freezes") was wrong for scalar signed sheaves, and
that is what would have been published.

---

## N3 — λ₂ is a certificate about the operator, not about the task

### The claim

The v0.1 report's Proposition 1 gives a contraction certificate
`‖e^{-tL}‖ ≤ e^{-tλ₂}`, and the design leans on it to say that a larger
spectral gap means a better memory.

### The result

| measurement | value |
|---|---|
| the contraction certificate `‖e^{-tL}‖ ≤ e^{-tλ₂}` holds | **36/36** ✅ |
| `pearson(λ₂, tracking quality)` | **0.1904** |
| `spearman(λ₂, tracking quality)` | **0.2499** |

The certificate is **true and verified**.  The *inference* that a bigger λ₂
means better long-context behaviour is **not supported**: the correlation is
weak.

### What we changed

Every place that said "λ₂ predicts quality" now says "λ₂ bounds the
**operator**".  The claim was weakened rather than the data massaged.

---

## N4 — The archived transformer comparison favors the transformer

An earlier report described a character-level comparison between FluidLayerLM
and a TinyTransformer, plus a separate global-aggregation mechanism test.
The archived raw outputs are now included in `results/train_results.json`
and `results/mechanism.json`. The original source scripts (`train.py` and
`demo_mechanism.py`) are not included, so these snapshots document the
historical measurements but do not make those runs independently repeatable.
The recorded LM comparison reports final validation losses of 2.6331 for the
transformer and 2.7397 for FluidLayerLM; the mechanism output records 97.3%
sign accuracy for FluidLayer versus 65.6% and 71.0% for local baselines.

---

## N5 — The domain prior helps, but a trivial frequency table helps more

### The experiment

`experiments/e1_prior_perplexity.py` — train a FluidLayerLM, **freeze it**,
then ask whether a **training-free** domain prior (built by streaming the
corpus through the fluid) improves its next-token distribution.  Every
correction is fitted in **closed form**; not a single gradient step is taken
on the model.  Three-way split: the model trains on 0–90 %, the prior is
fitted on 90–95 % (positions the model has never seen), evaluation on
95–100 %.

### The result

The full run (250 training steps, 1,000 fit windows, 500 val windows):

| arm | fitted params | state | val nats | Δ loss | nats per KB |
|---|---|---|---|---|---|
| base (frozen model) | – | – | 2.7134 | – | – |
| **fluid prior (static)** | 608 | 2.44 KB | 2.7075 | **−0.0059** | 0.0024 |
| fluid prior (live) | 608 | 2.44 KB | 2.7131 | −0.0003 | 0.0001 |
| unigram bias (the trivial baseline) | 5,776 | 22.56 KB | **2.6752** | **−0.0382** | 0.0017 |
| unigram, best rank-8 control | 608 | 2.38 KB | 2.6915 | −0.0219 | **0.0092** |
| bigram table | 438,976 | 1.71 MB | 2.7013 | −0.0121 | 0.000007 |

### Reading it — three separate findings, all negative-ish

**(a) The prior helps, but it FAILS the pre-registered threshold.**
The design doc set the bar at *"> 0.02 nats"*.  The measured gain is
**0.0059 nats**.  The threshold is **not met** at this scale.

**(b) A trivial unigram table beats it** (−0.0382 vs −0.0059), and even the
**best possible rank-8 unigram** beats it (−0.0219).  So the fluid prior is
**not** carrying more information than frequencies — it is a structured,
low-rank approximation of them, and a truncated SVD of the unigram table
does that job better.

**(b′) And the byte-efficiency story does not rescue it either.**  The
fluid prior is 9.4× *smaller* than the unigram table (2.44 KB vs 22.56 KB),
but it delivers only 16 % of the gain.  Per kilobyte, the **rank-8 unigram
control wins outright** (0.0092 nats/KB vs 0.0025) at the *same* size
(2.38 KB vs 2.44 KB) — so the fluid prior is not even the best use of its
own 2.4 KB.  The only arm the fluid prior beats per byte is the full
unigram table, and the rank-8 control beats that too.  **There is no
byte-efficiency argument left for the fluid prior.**

**(c) The benefit SHRINKS as the base model gets better — and that is the
interesting part.**  `experiments/e1_prior_perplexity.py --sweep` runs the
whole experiment at four training lengths (`results/e1_sweep.json`):

| training steps | base val nats | fluid prior gain | unigram gain | best rank-8 unigram |
|---|---|---|---|---|
| 30 (weak) | 3.6101 | **0.0482** | 0.1324 | 0.0894 |
| 100 | 2.8683 | 0.0156 | 0.0593 | 0.0344 |
| 250 | 2.7114 | 0.0065 | 0.0413 | 0.0238 |
| 500 | 2.5946 | 0.0091 | 0.0391 | 0.0180 |

(a gain is a **reduction** in cross-entropy, in nats)

The fluid prior's benefit collapses by ~5× as the model improves
(0.048 → 0.009), because a better model leaves smaller residuals for *any*
prior to correct.  The unigram's shrinks less (0.132 → 0.039) and wins at
every point.  **The prior's value is highest exactly where the model is
weakest** — on small models, on edge devices, on hard domains — and it is
the opposite of "a prior makes a good model better".

**Note the shape of the curve.**  The gain is not monotone in model quality:
it falls from 0.048 to 0.0065 and then ticks back up to 0.0091 at 500 steps.
With one corpus and one seed that uptick is well inside the noise, and we
are not going to dress it up as a trend.

### The honest verdict

> The domain prior is **not** the win it was designed to be.  It is a
> 2.4 KB artefact that improves a frozen model by 0.0059 nats where a 23 KB
> unigram table improves it by 0.038 and a rank-8 unigram at the *same* 2.4 KB
> size improves it by 0.022.  The **downloadable-prior product** story does
> not survive this measurement at the 250-step scale; it might at the scale
> of a small on-device model (where the prior's gain is 5× larger), and that
> is a separate experiment.

## N6 — Grafted onto a frozen model, the diffusion tier makes it WORSE

The most consequential negative result in the project, because it is the one
that would have bitten a user.

### The experiment

`experiments/e6_real_corpus.py` — a real base LM (character-level
`TinyTransformer`, 131,392 params) trained on the local `corpus.txt`
(Alice in Wonderland, 120,000 chars, vocab 76).  The memory layer is
attached to the residual stream and **the base model is frozen**.  The layer
alone is trained on the next-token task (the G1 protocol).  Everything is
measured on the held-out tail of the corpus.

### The result

| configuration | val loss | ppl | Δ vs base |
|---|---|---|---|
| **base (frozen)** | 2.5898 | 13.33 | – |
| base + layer, **lossy (diffusion) tier only** | 3.3992 | 29.94 | **+0.8094** |
| base + layer, **exact tier only** | 2.5598 | 12.93 | **−0.0300** |
| base + layer, both tiers | 3.3861 | 29.55 | +0.7963 |

**The diffusion tier alone more than doubles the perplexity of the model it
is attached to.**  The exact tier alone is harmless (marginally helpful).

And the training curve says *why*: the layer's own training loss **rises**
while it trains (2.4306 → 3.4387).  The optimisation is not failing to
converge — it is diverging.

### Why it diverges (the mechanism)

The diffusion state is a **persistent accumulator that lives outside the
autodiff graph** (by design: it is a memory, not an activation).  So the
read projection `W_o` is being trained against a **moving, non-differentiable
target** — the state changes every step, and no gradient tells the optimiser
how the read should be shaped.  The optimiser therefore pushes `W_o` in a
direction that is right for *this* state and wrong for the next one, and
the loss climbs.

The same layer on the *synthetic* corpus looks wonderful (−0.26 nats) for a
reason that is easy to miss: that corpus is a Markov chain with 70 %
self-loops, so "what followed this context" is almost always right, and the
layer degenerates into a copy mechanism.  **On real text the same mechanism
injects a wrong answer.**

### What we changed because of it

* The lossy tier is **not** enabled by default in the recommended recipe.
  `docs/GRAFT.md` ships the exact tier only.
* The recommendation in `docs/GRAFT.md` is explicit: *the diffusion tier, as
  implemented, needs a different training signal (a state that is
  differentiable, or a layer-local objective) before it should be switched
  on in a frozen-base deployment.*
* The synthetic-corpus perplexity gain is no longer presented as evidence
  of anything: it is presented as an **upper bound produced by a corpus
  that is maximally redundant**.

---

## N7 — The divergence IS fixable — and fixing it reveals the tier is worthless

The obvious follow-up to N6: *can the diffusion tier be made to stop
hurting the model?*  `experiments/e7_lossy_fix.py` tries two fixes.

| configuration | Δ base val loss | the layer's own train loss |
|---|---|---|
| control: train the layer through the base model's loss | **+1.1431** | 2.4306 → 3.8049 **diverged** |
| FIX 1: layer-local objective (the layer predicts the next token itself) | +0.0000 | 4.3425 → 6.8008 **not learning** |
| FIX 2: layer-local objective + **differentiable state** | **+0.0000** | 4.3425 → **3.8072 converged** ✅ |

**The layer-local objective alone was not the fix.**  The real cause was
the moving state: the persistent accumulator lives outside the autodiff
graph, so the optimiser was chasing a target that changed every step.  FIX 2
rebuilds the state from a bounded history *inside* the graph, which makes
the write projection trainable.  (The two states are different estimators —
the persistent one is the exponentially-weighted sum of all the writes, the
rebuild is the diffusion of the last `hist_len` windows — not the same
number; only the rebuild is differentiable.)

**The damage is now exactly zero** (+0.0000 — the base model is
bit-identical to the unattached baseline) and the layer's own objective
converges.  *Caveat:* in the fixed variants the layer-local loss never
touches the injection map, so the injection is off by default and the
untouched base model is partly trivial.  The substantive result is that the
layer **can now learn at all** (4.34 → 3.81, against 4.34 → 6.80 for the
persistent state) — and the sweep below asks the question that actually
matters: even now that it can learn, is its prediction worth injecting?

### But the fix does not make the tier useful

The layer's own next-token prediction converges to **3.8072 nats**.  The
base model does the same task in **2.5898 nats**.  A 16 KB state is simply
worse at predicting text than a 131,392-parameter model, and the
injection-strength sweep confirms it:

| prior-injection gain | 0 | 0.003 | 0.01 | 0.03 | 0.1 | 0.3 | 1.0 |
|---|---|---|---|---|---|---|---|
| Δ base val loss | +0.0000 | +0.0001 | +0.0004 | +0.0015 | +0.0092 | +0.0584 | **+0.3623** |

Every non-zero gain hurts, monotonically.  **The optimum is zero.**

### What this means for the project

The project began as *"replace attention with a sheaf diffusion"*.  The
measurements have now said this three separate times (N4: the transformer
wins; N6: the diffusion tier damages the model it is attached to; N7: once
the damage is fixed, the tier has nothing to add).  The defensible product
is the **exact tier** — a constant-size, bit-exact, content-addressed
memory grafted onto a frozen model — plus the fluid as the *mechanism*
behind the address space and the constant-size summary.

That is a smaller paper than "a new architecture".  It is also the one the
evidence supports, and it is the one that is deployable: attaching it is
free, it cannot damage the model, and it adds a capability the model
provably lacked (cross-window verbatim recall, 0.050 → 0.800).

### What survived

The **exact tier** is unaffected: −0.0300 nats (no harm) and the
cross-window recall capability (0.050 → 0.800) is delivered entirely by it
(`results/e6_graft.json`).  The conclusion is sharp:

> **Ship the register. Do not ship the diffusion tier into a frozen model
> until its training signal is fixed.**

---

## N8 — RESOLVED: one knob for recall AND generation

### The failure

E9 measured the tension between the two use cases of the exact tier:

| the injection gain | the recall | the base model's Δ |
|---|---|---|
| 0.3 | poor | −0.0059 nats |
| 300 | **0.800** | **+1.0829 nats** (ppl 13.61 → 40.20) |

The recall needs a *hard* override (the retrieved token must win the
argmax); the generation needs a *soft* one (a wrong retrieval must not
overrule the model).  One scalar cannot be both, and the layer was
therefore unusable in production: pick the recall gain and the model is
ruined; pick the generation gain and the recall disappears.

### The fix

The base model's **own uncertainty** at that position:

```
u_t = 1 - max_v p(v)          boost_t = gain · u_t
```

The two tasks differ exactly in that quantity — in the recall task the
model has no idea (`u ≈ 1`), in the generation task it usually does
(`u ≈ 0`).  So one knob serves both, and it is not a new
hyper-parameter: it is the model's confidence.

### The measurement

| | |
|---|---|
| the cross-window recall, gated | **0.975** (the ungated layer: 0.975) |
| the base model's val loss | 2.6111 (ppl 13.61) |
| gain | ungated Δ | **gated Δ** |
|---|---|---|
| 0.5 | −0.0317 | −0.0240 |
| 2.0 | −0.1082 | −0.0864 |
| **8.0** | −0.1219 | **−0.1447** (ppl **11.78**) |
| 20.0 | +0.1103 | **+0.0267** |
| 50.0 | +0.6954 | **+0.4862** |

The gate does three things at once: it **does not cost the recall**,
it **improves the best operating point** (13.4 % better perplexity than the
ungated layer), and it **shrinks the damage at high gains** (4× less harm
at gain 20).

### What changed in the code

* `SAABranch.gated_logits(logits, info, gain)` — the numpy version;
* `SAABranchTorch.gated_logits(...)` — the torch twin;
* four new self-verification checks (the gate ignores a confident model,
  lets an unsure one be overruled, reports what it retrieved, and is a
  no-op when there is no hit) → **50 checks**;
* `experiments/e10_gate.py` + `results/e10_gate.json`.

---

## N9 — RESOLVED: the memory is exact now (and two tempting rules failed)

### What was not exact

| the imperfection | the measurement |
|---|---|
| the recall had a ceiling | 0.975 — the soft gate could suppress a *correct* retrieval whenever the base model happened to be confident |
| the store kept the FIRST continuation of a context | a context occurring everywhere kept only its first continuation, so the statistically correct answer was lost: **0/3** facts recalled (E9) |
| the eviction threw away the oldest entry | which is exactly the entry most likely to be corroborated; the newest (unconfirmed) ones survived instead |

### The fixes, and the measurements

| | before | after |
|---|---|---|
| the cross-window recall | 0.975 | **1.000** (40/40) |
| the common-key case | 0/3 | **40/40** |
| the corroborated entries surviving the eviction | 0 | **40/40** |
| the base model's perplexity with the layer attached | 13.61 | **11.42** (−0.1756 nats) |

The fixes: the store keeps the **majority** continuation and counts the
corroboration (which yields a confidence for free), the eviction drops the
**least corroborated** entry rather than the oldest, and the gate's soft
form is enough on its own — at `u ≈ 0.99` the boost `gain·u ≈ 8` already
wins the argmax.

### Two rules that were tried and rejected

Both are in the code, switchable, and **off by default** because they were
measured:

| the rule | the result |
|---|---|
| **the clueless rule** — the retrieval decides when the base model's top-1 probability is below 0.5 | **+0.65 nats**: a character-level model's top-1 is often below 0.5, so the rule fired constantly and forced wrong retrievals into the model |
| **the trusted-memory rule** — a never-contradicted retrieval decides | **−0.021 nats** instead of −0.176: on real text many contexts have a deterministic-looking continuation that is still wrong for a new occurrence |

Both are good ideas that do not survive measurement, and both are recorded
here rather than quietly deleted.

### What "exact" does not mean

The memory is still **bounded**: with a 16 KB budget it holds ~819
contexts, so its reach is the last ~10,000 tokens of a 10M-token stream.
Exactness is about the *answer*, not about *coverage*.

---

## N7 (continued) — RESOLVED: the diffusion tier gets a job (the ROUTER)

N7 left the diffusion tier safe but useless: its own aggregate was a worse
predictor than the base model (3.81 vs 2.59 nats), so the best injection
gain for it was **zero**.

### The fix: the fluid's geometry as an address space

Each of the fluid's M anchors holds the **majority continuation** of the
contexts routed to it.  The fluid answers *where a context belongs*; the
per-anchor store answers *what usually follows contexts like this*.  The
exact tier wins whenever it has the context; the router tier is the
**fallback**.

### The measurement (E12)

| | without the router | with the router |
|---|---|---|
| the fact-task recall when the exact store is too small (64 slots) | 0.100 | **0.625** (the base model alone: 0.000) |
| the loss on the HARD positions (the base model's top-1 below 0.1) | +0.0534 | **−0.0707** |
| the overall val loss | −0.1756 | −0.1572 |
| the routing cost | – | 0.0005 % of a transformer layer (M=64) |

The router tier helps **exactly where the exact tier cannot**: a context
too old for the store, and a context the base model has no idea about.

### The honest limit

The router's answer is a **cluster-level regularity**, not a fact: it
recovers 0.625 of the lost facts, not 1.000, and it costs 0.019 nats
overall because a cluster-level guess sometimes overrules a correct
base-model guess.  That is the trade, and it is now explicit.

---

## N10 — CLOSED: the comparison against the closest prior art was MISSING

This is a failure of the *review*, not of the code: the repository never
compared itself against a **kNN-LM** (Khandelwal et al., 2020), which is
functionally the same idea — a datastore of (context → next token) plus an
LM — and which any reviewer would have asked for first.  `COMPARISON.md` did
not even mention it.

`experiments/e13_knn_baseline.py` now measures it on the same corpus, the
same datastore and the same base model:

| | val loss | Δ vs base | token accuracy |
|---|---|---|---|
| the base model alone | 2.6111 | – | 0.2839 |
| kNN-LM, λ=0.9 | 2.5610 | −0.0501 | 0.2894 |
| **the layer (gated), gain=8** | **2.4690** | **−0.1421** | **0.3177** |

**With a 16 KB datastore the gate wins by 2.8× on the loss and 8× on the
token accuracy.**  But with a 1 MB datastore the kNN-LM wins by 6.4×.

**The honest conclusion, which is narrower than the project's original claim
and much stronger for being measured:**

> **The contribution is the gate under a hard memory budget.**  With a
> bounded store, the interpolation dilutes a mostly-wrong vote with the
> model's distribution and barely helps; the gate can act decisively
> exactly where the model has no idea.  With an unbounded store the
> classic kNN-LM wins, because its votes are mostly right and it can never
> overrule the model.

That is the claim the paper should make, and `POSITIONING.md` Gate 6 now
records it.

---

## N11 — The archived real-model result narrows the claim

The E14 script and archived `results/e14_real_gpt2.json` are included. The
recorded run reports bit-identical attachment, cross-window recall, and
slightly lower loss with the gated layer, while kNN-LM interpolation lowers
loss more. Its GPT-2 checkpoint is not included; rerun E14 to independently
reproduce the snapshot.

---

## What the available evidence says

The project's shape from the available evidence:

| tier | status |
|---|---|
| the **exact tier** (bounded store, bit-exact token-ID recall) | ✅ **works, measured** |
| the **constant-size state** | ✅ **works, measured** |
| the **budget law** (spend on pins, not anchors) | ✅ **works, measured** (E2) |
| the **fluid as an aggregate carrier** | ✅ the archived mechanism output records 97.3 % sign accuracy (`results/mechanism.json`); original source script unavailable |
| the **layer attached to a frozen model** | ✅ **free** — bit-identical logits (max diff 0.0) |
| the **exact tier in a graft** | ✅ **works** — recall 0.050 → **0.800** (and **1.000** with the exact store), −0.03 nats on real text |
| the **lossy tier in a graft** | ✅ **has a job now** — as the router it rescues the recall (0.100 → **0.625**) and fixes the hard contexts (+0.053 → **−0.071**); on its own it still hurts (N6/N7) |
| the **exact tier in a graft, gated** | ✅ **works** — the recall **1.000** and the base model's perplexity **improves** by **−0.176 nats** (ppl 13.61 → **11.42**) (N8/N9) |
| **the same layer on a REAL GPT-2 (124M)** | ✅ archived E14 JSON is included; checkpoint required to rerun (N11) |
| the **topological selection criterion** | ❌ **does not work** (N1) |
| the **`‖L·1‖∞` balance diagnostic** | ❌ **was wrong** (N2, now fixed) |
| **λ₂ as a quality predictor** | ⚠️ **weak** (N3) |
| **the fluid as an LM backbone** | ❌ the archived comparison favors the transformer (N4) |
| **the domain prior vs frequencies** | ⚠️ **helps, but frequencies help more** (N5) |

**The defensible paper is:** *"a constant-size memory tier that is bit-exact
for a selected subset, gracefully lossy for the rest, with a certified bound
and a bounded, majority-tracking exact store — plus an honest account of
which findings are supported by checked-in results and which historical
claims still need their raw outputs."*

That is a smaller paper than "we replace attention".  It is also a paper
that can be **believed**, which in 2026 is worth more.

---

## Reproducing every number in this file

```bash
python3 selftest.py                                  # 50/50 (incl. N2)
python3 experiments/e5_graft.py --tokens 1000000     # N1
python3 experiments/e4_gbalance.py --trials 30       # N2, N3
python3 experiments/e1_prior_perplexity.py           # N5
python3 experiments/e6_graft.py                      # the graft (capability)
python3 experiments/e6_real_corpus.py                # N6
python3 train.py                                     # N4
```

Everything is CPU-only, pure NumPy, no downloads.


---

## N13 — a test that passes at a 0.0% recall (the "bulletproof" review)

The second AI pass added `fluidlayer/graft_bulletproof.py` and
`test_bulletproof.py`. The layer itself turned out to be **good** — its
exact tier recalls **100 %** (measured, 7/7, in both the numpy and the
torch twin).

The **test** was not. It asserted only that the key `"recall_rate"`
existed in the returned info — never that the recall was positive. The
run printed:

```
✅ NumPy bulletproof layer test passed (Recall: 0.0%)
...
✅ 100% recall capability verified
```

**A test that passes while the metric is zero is worse than no test at
all: it launders a failure into a green tick.** The 12-gram test was
worse still — it wrote into one layer, replayed into a *fresh, empty*
layer, queried a *different* token, and printed "✅ … passed (system
working correctly)" on zero hits.

**Fixed** (`docs/REVIEW-bulletproof.md` has the full list): the test now
builds a real recall scenario and asserts `rate >= 0.99`; the claims in
the docstrings were rewritten to what was measured; three real bugs were
fixed along the way — an `np.divide` without `out=` (uninitialised
memory in the diffusion state), a dead collision counter (collisions
were never detected, only hoped away), and an unscoped store key (two
tenants could address the same slot).

**The lesson, written down because it is easy to repeat:** *a number
that is printed must be asserted.*

---

## N14 — a green CI job that produced nothing (the claims-verification pipeline)

The verification workflow (`.github/workflows/verify-claims.yml`) had a
FIRST RUN that was GREEN in 45 seconds and produced NOTHING: no results
branch, no run.json. Two silent failures had stacked:

* the committed `experiments/e16_verify_readme.py` was syntactically
  INVALID — a string literal contained a real newline (the push pipeline
  mangled the escape), so python died at parse time in 0 seconds;
* `git add results/` was silently emptied by `.gitignore`'s `*.log`,
  and the crash meant no other file existed to stage — the push step took
  the "nothing to commit" path and exited 0.

**The lesson:** *a green job is not a result.* The workflow now
force-adds the results directory, and the same pipeline — once it
actually ran — found four more real bugs by MEASURING instead of reading:
the GGUF magic bytes were REVERSED (b'GUFF'), the MILITARY Fernet key was
invalid (raw 32 bytes instead of url-safe base64), the constructor's
`d` was silently ignored whenever a config was passed (the README's own
quickstart hit this), and both numpy-only test suites hard-imported torch
and failed the CI's numpy job.

Two claims were also corrected with measurements rather than code: the
"≤ 16 KB" memory budget (measured: 116 KB at only 1000 entries → the
honest bound is the capacity-enforced **≤ 100 MB**, maintainer-approved)
and the relative speed rows ("10x / 1.5x" — no runnable baseline exists;
the absolute overhead is measured instead). The "-0.30 nats" perplexity
table had no measurement behind it at all; the A/B harness now measures
the layer on a real GPT-2 (`docs/VERIFICATION.md`).
