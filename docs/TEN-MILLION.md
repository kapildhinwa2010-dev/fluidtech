# What happens at ten million tokens

> *"Give a model 10M tokens with this layer attached.  What does it
> remember?  And how does the performance change — before and after?"*

Measured by `experiments/e9_ten_million.py` (10,000,000 tokens, CPU). Every
number below is in `results/e9_ten_million.json`. To run the long experiment
in two stages:

```bash
python experiments/e9_ten_million.py --phase stream
python experiments/e9_ten_million.py --phase performance
```

---

## The footprint, first

| | 10M tokens |
|---|---|
| **the layer's state** | **16 KB** — and it does not grow |
| the same tokens as a KV cache (8B model, fp16, GQA) | **1.31 TB** |
| the same tokens as a KV cache (70B model) | **3.28 TB** |
| ratio | **80.0 million ×** (8B) / **200.2 million ×** (70B) smaller |

The layer's state is constant: it is the same 16 KB after 10 tokens and
after 10 billion.  That is the entire economic argument, and it is
arithmetic rather than a claim.

---

## Filling the memory is *model-free*

The exact tier's write is *"the content key → the token that followed it"*.
It needs no forward pass.  So filling a 10M-token memory is a dictionary
update per token, not a model call:

```
10,000,000 tokens streamed in 12.1 s   (0.83M tokens/s, no model forward)
```

That is the difference between this layer and the KV cache: the KV cache
*is* the model's forward pass, kept around.  This one runs beside it.

---

## What it actually remembers

Two different answers, because the answer depends on how distinctive the
context is.

### Distinctive keys → it remembers, all the way down

Facts planted at six depths, queried after the whole 10M-token stream:

| the fact was planted at token | recalled? |
|---|---|
| 100,000 | ✅ |
| 1,000,000 | ✅ |
| 3,000,000 | ✅ |
| 5,000,000 | ✅ |
| 7,000,000 | ✅ |
| 9,000,000 | ✅ |

**6/6, at every depth.**  A 16 KB memory reached all the way back through
10M tokens.

**But the control matters more:** facts planted with *common* keys (a key
that occurs all over the stream) were recalled **0/3**. The store is
majority-tracking, so a context with many conflicting continuations does
not identify one planted answer.

> **The layer does not remember "facts". It remembers a continuation
> associated with a context.** The attachable store tracks majority support
> and evicts the least-corroborated entry when full.

### Realistic keys (12-grams) → it remembers only the last ~10,000 tokens

A 12-gram context is distinctive, but there are millions of them. With a
16 KB budget (819 slots), the bounded store evicts entries:

| the fact was planted at token | recalled? |
|---|---|
| 100,000 … 9,000,000 | ❌ **0/6** |

The reported reach was roughly the last **10,000 tokens** of this 10M-token
stream — 0.1 % of it. This is a measurement of that run, not a guarantee
that the store is a FIFO window; current eviction is evidence-based.

> **A constant-size memory does not scale to a long stream.**  It scales in
> *bytes*; its *reach* is bounded by the number of distinct contexts it can
> hold.  At 10M tokens the exact tier's reach is the last 0.1 % of the
> stream — unless the context is rare enough that the store never fills.

This is the single most useful thing the 10M experiment says, and it is a
**budget** fact, not an architecture fact: give the store 10× the slots
and its reach grows 10×.

---

## The performance delta: before vs. after

The base model's next-token loss on held-out text, with the layer attached
and the memory filled with 10M tokens of *other* text (so the retrievals
are often wrong — the realistic case):

| the injection gain | val loss | ppl | Δ vs base |
|---|---|---|---|
| **0** (the layer off) | 2.6111 | 13.61 | +0.0000 |
| 0.1 | 2.6081 | 13.57 | -0.0030 |
| **0.3 ← the optimum** | **2.6054** | **13.54** | **-0.0057** |
| 1.0 | 2.6236 | 13.79 | +0.0125 |
| 3.0 | 2.7458 | 15.58 | +0.1347 |
| 10.0 | 3.0720 | 21.59 | +0.4610 |
| 30.0 | 3.4374 | 31.11 | +0.8263 |
| 100.0 | 3.6353 | 37.91 | +1.0242 |
| 300 | 3.6939 | 40.20 | **+1.0829** |

### The finding that matters

**The gain that maximises RECALL is not the gain that maximises
GENERATION — they are three orders of magnitude apart.**

* The recall needs a **big** gain (300): the recalled token must *win* the
  argmax, so it has to be forced.
* The generation needs a **tiny** gain (0.3): the retrieval is a *hint*,
  and a hint that is wrong must not be allowed to overrule the model.

At the recall-optimal gain, the layer **more than triples the model's
perplexity** (13.61 → 40.20) whenever the retrieved continuation is wrong.
At the generation-optimal gain it is a **small but real improvement**
(13.61 → 13.54, −0.0057 nats).

So the honest answer to "performance pehle ke aur ab ke beech kitna antar
hoga" is:

> **With the layer switched off: nothing changes (bit-identical).
> With the layer armed at the right gain: −0.006 nats (a small, real
> improvement).  With the layer armed at the gain you would naively pick
> from the recall benchmark: +1.10 nats, i.e. the model is ruined.**
>
> **And with the gate (E10): −0.145 nats (ppl 13.61 → 11.78) at the same
> time as the full recall.**  The gain is no longer a choice between the
> two use cases — the model's own confidence picks.

And the practical rule that falls out: **the injection gain must be
calibrated for the use case, and a single knob cannot serve both.**
A production layer needs a *gate* on the injection (apply it only when the
retrieval is confident), not one global scalar.  That is the next thing to
build.

---

## The three-tier picture, at 10M tokens

| what | how deep does it reach? | exact? |
|---|---|---|
| **the exact tier**, distinctive keys | the whole stream (6/6 facts at 10M depth) | ✅ bit-exact |
| **the exact tier**, 12-gram keys, 16 KB | the last ~10,000 tokens (0/6 facts) | ✅ bit-exact |
| **the diffusion tier** (the 16 KB aggregate) | the whole stream | ❌ no exact recall at all |
| the KV cache | the whole stream | ✅ but 1.31–3.28 **terabytes** |

The diffusion tier is the only tier that "remembers everything" — and it
remembers it as a **summary**, which cannot answer "what exactly was
said".  The exact tier can answer exactly, but only about what fits in its
budget.

---

## Reproducing

```bash
python experiments/e9_ten_million.py --tokens 10000000
```

~13 minutes on one CPU core (the 10M-token fill is 12 s; the 12-gram eviction test is the slow part).  `--tokens 200000` gives the same structure in
30 seconds, if you want a quick look.
