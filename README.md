<div align="center">

# Superfluid

**A constant-size, bit-exact memory tier for LLMs — attachable with one residual-stream insertion.**

`16 KB` of state instead of `40 GB` of KV cache ·  
exact recall of facts from outside the attention window ·  
pure NumPy, no GPU, no downloads ·  
seven measured negative results, published on purpose

[![CI](https://github.com/<your-username>/superfluid/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/superfluid/actions/workflows/ci.yml)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Self-verification](https://img.shields.io/badge/self--verification-38%2F38-brightgreen.svg)

</div>

---

## The problem, in one number

A 70B model on 2×H100 80GB cannot hold a single 128K-token session.

The weights take **131.5 GB**; the KV cache for 128K tokens needs another **40 GB**, and only **28.5 GB** is left.  
The KV cache is not an optimisation problem — it is a **wall**, and it arrives at 32K context.

The usual answer is to quantise it, compress it, or make the model smaller.  
This project asks a different question:

> What if the memory were addressed by content instead of by position, and constant instead of growing?

---

## What this is

A memory layer you can bolt onto any causal LM:

```text
h' = h + W_e · E(h)  +  W_l · S(h)
         ^^^^^^         ^^^^^
         exact tier     lossy tier (the constant-size fluid state)
```

- The **exact tier** is a bounded, content-keyed store of token ids.  
  **20 bytes a fact.** A recalled value is an `int32` — bit-exact, not approximate.  
  Append-only, first-write-wins, and it fails loud on a signature mismatch rather than returning another fact’s value.

- The **lossy tier** is a sheaf-diffusion aggregate over `M` anchors — `M × r` floats, constant size, whatever you stream through it.

The two tiers exist because a diffusion smears. If the exact value of a token lives only inside the fluid, it is mixed with its neighbours and is provably no longer exact.

So:

> **The fluid answers WHERE. The register answers WHAT.**

---

## The three numbers

| Claim | Measured result |
|---|---|
| Attaching the layer changes nothing | The logits are **bit-identical** (max diff `0.000e+00`). The output projections start at zero, so the branch is a **provable no-op** until someone trains it. |
| It adds a capability the model did not have | Cross-window verbatim recall: **`0.050 → 0.800`** (chance `0.125`) — **16× the base model** — for a fact planted 400 tokens outside the attention window. |
| The memory is exact | With the majority-tracking store and evidence-based eviction: recall **`1.000`**, common-context case **`1.000`** (was `0/3`), and corroborated entries survive the flood (**`40/40`**). |
| The model gets better | Gated retrieval improves the base model’s perplexity by **`−0.176 nats`** (`ppl 13.61 → 11.42`) — the model’s own confidence decides when the memory speaks. |
| It beats its closest prior art in the regime it targets | Against a **kNN-LM** (the standard way to use a datastore), with a **16 KB** store the gated layer wins by **7.4×** on loss (`−0.1134 vs −0.0154`). With a 1 MB store the kNN-LM wins by 6.4×, so the claim is precise: **the gate is the contribution under a hard memory budget.** |
| The fluid tier earns its place | Used as the **router** (the geometry as an address space), it rescues recall when the exact store is too small (`0.100 → 0.625`) and fixes hard contexts where the base model has no idea (`+0.053 → −0.071 nats`) — at **0.0005%** of a layer’s compute. |
| The diffusion tier delivers none of it | With the lossy tier alone: **`0.050`**, i.e. the base model’s own number. The exact tier delivers **all** of it. |

Measured on a frozen base model, in [`results/e6_graft.json`](results/e6_graft.json).  
The recipe is in [`docs/GRAFT.md`](docs/GRAFT.md).

---

## Try it, in 60 seconds

```bash
git clone https://github.com/<your-username>/superfluid && cd superfluid
pip install -r requirements.txt

python selftest.py              # 50 machine-checked claims
python examples/quickstart.py   # the tour
python examples/graft_demo.py   # the capability, end to end
python examples/gate_demo.py    # the gate: recall AND a better perplexity
python experiments/e11_exact.py # the exact memory (the full benchmark)
```

Expected output:

```text
base model        : 1/20 correct  (it never saw the fact)
base + the layer  : 20/20 correct  (the layer retrieved it)
```

---

## Using it

```python
import torch
from superfluid.torch_graft import SAABranchTorch

branch = SAABranchTorch(
    M=64,
    r=8,
    d=d,
    emb=model.get_input_embeddings(),
    ctx_len=1,
    read_gain=300.0,
    transform="pc1",
    use_exact=True,
    use_lossy=False,
)

branch.reset()                       # a new session

for window in windows:               # stream the session
    h = model.get_input_embeddings()(window)
    h, _ = branch.forward(h, window)
    logits = model(inputs_embeds=h.unsqueeze(0)).logits
```

- **6.3 KB** of state for `M=64, r=8`, plus **20 B** per cached fact.
- The base model is frozen the whole time.
- Attaching it is free and reversible.
- The NumPy reference implementation is [`superfluid/graft.py`](superfluid/graft.py), and the Torch one mirrors it exactly.

Full instructions: [`docs/GETTING-STARTED.md`](docs/GETTING-STARTED.md)  
API: [`docs/API.md`](docs/API.md)

---

## What it does not do

Being explicit, because this is where grafts usually oversell:

- It does **not** make the model better at language — it makes it better at remembering what it has already seen.
- It does **not** extend the attention window — the window is still 128 tokens; the layer can quote from outside it, not reason over it.
- It does **not** compose, infer, or answer questions about what it holds.
- The diffusion tier has **no measurable upside** on real text and used to damage the model it was attached to (`+0.81 nats`) until its training signal was fixed — keep it off.

The full list, with numbers: [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)

---

## At ten million tokens

Give a model 10M tokens with the layer attached, and:

- The state is still **16 KB**.  
  The same tokens as a KV cache would require **1.31 TB** for an 8B model, **3.28 TB** for a 70B one.
- Filling it costs **80 seconds** and **no model forward at all** — the exact tier is model-free.
- With distinctive keys it recalls **6/6** facts planted from 100K to 9M tokens deep.  
  With realistic 12-gram keys and a 16 KB budget it recalls **0/6** — its reach is the last ~10,000 tokens.
- The perplexity delta is **`−0.0057 nats`** at the right injection gain, and **`+1.08 nats`** at the gain that maximises recall — the two knobs are three orders of magnitude apart.  
  The gate closes that gap: scaling the override by the model’s own uncertainty keeps recall at **`1.000`** and improves perplexity to **`11.42`**.

Details: [`docs/TEN-MILLION.md`](docs/TEN-MILLION.md)

---

## The negative results

This is the part that makes the rest believable.  
Seven results came out negative or against expectation, and all seven are measured, reproducible and published.

| What we expected | What happened |
|---|---|
| The **topological** selection criterion picks the right facts | It does **not** — `0.000` precision vs `0.970` for plain geometric affinity. |
| `‖L·1‖∞` detects the frozen state | It does **not** — the frustration index does; the old claim is now a test case. |
| `λ₂` predicts long-context quality | Barely — Pearson `0.20`. |
| A transformer loses at some scale | It **wins** at every scale we could train. |
| A domain prior beats a frequency table | It helps (`−0.006 nats`) but a **unigram table wins 6× more**. |
| The diffusion tier is neutral on a frozen model | It **degrades** the model by **`+0.81 nats`** (fixed → `+0.0000`, but no upside either). |
| Training the layer helps | It **hurts** — the training-free identity map is better (`0.800 vs 0.650`). |

Read it: [`docs/NEGATIVE-RESULTS.md`](docs/NEGATIVE-RESULTS.md)

---

## The repository

```text
superfluid/            the package (pure NumPy; torch only in torch_graft.py)
  graft.py             the attachable memory layer
  memory.py            the frozen register (20 B per fact, bit-exact)
  torch_graft.py       the same layer in PyTorch, for a real LLM
  gpt2_model.py        a GPT-2 forward pass in PyTorch (no transformers needed)
  gpt2_tokenizer.py    the GPT-2 byte-level BPE, in pure Python
  sheaf.py topology.py diffusion.py blocks.py autodiff.py ...

experiments/           every measurement; each writes a JSON into results/
  e14_real_gpt2.py     the graft on a REAL pretrained GPT-2 (124M)
  e13_knn_baseline.py  the layer vs. the closest prior art (kNN-LM)

tools/                 the artefact builder, the BPE tokenizer, the bundler,
                       the checkpoint downloader, a PyPI fetcher

tests/                 the unit tests (pytest, or run them without it)

examples/              quickstart.py, graft_demo.py, gate_demo.py

docs/                  the design docs, the API, the limits, the theory report

results/               every number in the README, as JSON
```

---

## Documentation

| Document | What it is |
|---|---|
| [`docs/GRAFT.md`](docs/GRAFT.md) | What the layer does to a base model — measured. |
| [`docs/TEN-MILLION.md`](docs/TEN-MILLION.md) | What happens at 10M tokens — what it remembers, and the performance delta. |
| [`docs/MILESTONES.md`](docs/MILESTONES.md) | The twelve milestones, and what is left. |
| [`docs/NEGATIVE-RESULTS.md`](docs/NEGATIVE-RESULTS.md) | The seven things that did not work. |
| [`docs/GETTING-STARTED.md`](docs/GETTING-STARTED.md) | How to use it. |
| [`docs/API.md`](docs/API.md) | The API reference. |
| [`docs/REPRODUCING.md`](docs/REPRODUCING.md) | Every command, every number. |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | The honest list. |
| [`docs/FRAMEWORK.md`](docs/FRAMEWORK.md) | The mathematics. |
| [`docs/DESIGN-pinned-memory.md`](docs/DESIGN-pinned-memory.md) | The design of the frozen register. |
| [`docs/COMPARISON.md`](docs/COMPARISON.md) | vs. KV cache, sliding window, RAG, Titans, Infini, MLA. |
| [`docs/POSITIONING.md`](docs/POSITIONING.md) | The prior-art audit — this is not revolutionary, and here is why. |
| [`docs/index.html`](docs/index.html) | The theory report (propositions, cost tables, citations). |

---

## Status — and it now includes a REAL pretrained LLM

GPT-2 (124M parameters, the real 50,257-token BPE tokenizer), run on CPU with the memory layer attached:

| Test | Result |
|---|---|
| Attaching the layer | The logits are **bit-identical** (max diff `0.000e+00`). |
| A fact planted **2,947 tokens** before the query | The base GPT-2 predicts `464`; **the layer predicts the fact**. |
| The base model’s perplexity | `4.0289` (`ppl 56.20`) → **`4.0019` (`ppl 54.70`)** with the layer. |

And one result that narrows the claim honestly:

On the real model, the classic kNN-LM interpolation beats the gated override by **8×** on perplexity — the opposite of the 131k-parameter model, where the gate won by **7.4×**.  
The gate wins where the base model is weak; interpolation wins where it is strong.  
The capability works on both.

The checkpoint, the forward pass and the tokenizer are all in this repository:

- [`superfluid/gpt2_model.py`](superfluid/gpt2_model.py)
- [`superfluid/gpt2_tokenizer.py`](superfluid/gpt2_tokenizer.py)

No `transformers` dependency, and the generation is verified coherent.

The Torch twin is now tested against the NumPy reference in [`tests/test_torch_parity.py`](tests/test_torch_parity.py): the two implementations must agree on the router, the store and the gated override.

That test found **four real port bugs**:

1. A different route scale.
2. The wrong routing content — the residual stream instead of the position-free embedding.
3. A different confidence formula.
4. A read/write order that broke the snapshot semantics.

Four bugs that would have shipped silently.

Everything else in this repository is machine-checked and regenerates with one command.

---

## A research prototype, honestly labelled

Everything above was measured on CPU, at small scale, on a character-level or 512-token subword corpus, with a **131,392-parameter model**.

The subword repeat is the closest thing to a realistic tokenizer in this repository.  
The GPU run, [`experiments/gpu_graft.py`](experiments/gpu_graft.py), is written and has not been executed here.

---

## Reproducing

```bash
python selftest.py                    # 50 checks
python experiments/run_all.py --full  # every measurement (~40 min, CPU)
```

Everything is pure NumPy, CPU-only, and needs no network.

The details and expected outputs: [`docs/REPRODUCING.md`](docs/REPRODUCING.md)

---

## Contributing

Issues and pull requests are welcome — especially measurement disagreements.

If a number in the docs does not match what you get, that is a bug in the code or in the documentation, and it is the most valuable report you can file.

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## License

MIT — see [`LICENSE`](LICENSE).

If you use this, please cite it: [`CITATION.cff`](CITATION.cff).

---

## Confession

The project began as:

> “Replace attention with a sheaf diffusion.”

The measurements have said three times that this is not the way.

What is left is smaller and much more defensible:

A constant-size memory tier that is bit-exact for a selected subset, gracefully lossy for the rest, with a certified bound, an append-only audit trail, 50 machine-checked claims, and a published account of the seven places where it did not beat the alternatives.

## author's confession 

> I am kapil this is my like school project if you are reading this so you read the whole repo may be.....

and I tell you honestly 
this idea is mine only,
the code is written by ai all the files are created by ai and when I put these all on github there are many options so I somehow but those files is not in order ...
or there are many lines where written "your name" so the ai said write my name there but I forgot so don't mind.
>I don't know about ai 
>I don't know How to coding 
>but he idea of fluidtech was force information into topological liquid is mine executed by ai 


only author's confession written by Me .🤗🤫


>so I am write here ,sorry Ai write here how to connect this layer ---
but today my credits are spent so I update it tomorrow 
bye 👋 
