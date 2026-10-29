<div align="center">
Superfluid
A constant-size, bit-exact memory tier for LLMs — attachable with one
residual-stream insertion.
`16 KB` of state instead of `40 GB` of KV cache ·
exact recall of facts from outside the attention window ·
pure NumPy, no GPU, no downloads ·
seven measured negative results, published on purpose
https://github.com/<your-username>/superfluid/actions/workflows/ci.yml/badge.svg
https://img.shields.io/badge/python-3.9%2B-blue.svg
https://img.shields.io/badge/license-MIT-green.svg
https://img.shields.io/badge/self--verification-38%2F38-brightgreen.svg
</div>
The problem, in one number
A 70B model on 2×H100 80GB cannot hold a single 128K-token session.
The weights take 131.5 GB; the KV cache for 128K tokens needs another
40 GB, and only 28.5 GB is left.  The KV cache is not an optimisation
problem — it is a wall, and it arrives at 32K context.
The usual answer is to quantise it, compress it, or make the model
smaller.  This project asks a different question:
What if the memory were addressed by content instead of by
position, and constant instead of growing?
What this is
A memory layer you can bolt onto any causal LM:
h' = h + W_e · E(h)  +  W_l · S(h)
         ^^^^^^         ^^^^^
         exact tier      lossy tier (the constant-size fluid state)
the exact tier is a bounded, content-keyed store of token ids.
20 bytes a fact.  A recalled value is an `int32` — bit-exact, not
approximate.  Append-only, first-write-wins, and it fails loud on a
signature mismatch rather than returning another fact's value.
the lossy tier is a sheaf-diffusion aggregate over `M` anchors —
`M × r` floats, constant size, whatever you stream through it.
The two tiers exist because a diffusion smears.  If the exact value of
a token lives only inside the fluid, it is mixed with its neighbours and is
provably no longer exact.  So:
the fluid answers WHERE, the register answers WHAT.
The three numbers
|  | |
| ---|---|
| attaching the layer changes nothing|the logits are  bit-identical  (max diff  0.000e+00 ).  The output projections start at zero, so the branch is a  provable  no-op until someone trains it|
| it adds a capability the model did not have|cross-window verbatim recall:  0.050 → 0.800  (chance 0.125) —  16× the base model  — for a fact planted 400 tokens outside the attention window|
| the diffusion tier delivers none of it|with the lossy tier alone:  0.050 , i.e. the base model's own number.  The exact tier delivers  all  of it|
Measured on a frozen base model, in `results/e6_graft.json`.  The recipe is
in `[docs/GRAFT.md](docs/GRAFT.md)`.
Try it, in 60 seconds
git clone https://github.com/<your-username>/superfluid && cd superfluid
pip install -r requirements.txt
python selftest.py            # 38 machine-checked claims
python examples/quickstart.py # the tour
python examples/graft_demo.py # the capability, end to end
   base model        : 1/20 correct  (it never saw the fact)
   base + the layer  : 20/20 correct  (the layer retrieved it)
Using it
import torch
from superfluid.torch_graft import SAABranchTorch
branch = SAABranchTorch(M=64, r=8, d=d, emb=model.get_input_embeddings(),
                        ctx_len=1, read_gain=300.0, transform="pc1",
                        use_exact=True, use_lossy=False)
branch.reset()                       # a new session
for window in windows:               # stream the session
    h = model.get_input_embeddings()(window)
    h, _ = branch.forward(h, window)
    logits = model(inputs_embeds=h.unsqueeze(0)).logits
6.3 KB of state for `M=64, r=8`, plus 20 B per cached fact;
the base model is frozen the whole time;
attaching it is free and reversible;
the numpy reference implementation is `superfluid/graft.py`, and the
torch one mirrors it exactly.
Full instructions: `[docs/GETTING-STARTED.md](docs/GETTING-STARTED.md)` ·
API: `[docs/API.md](docs/API.md)`
What it does not do
Being explicit, because this is where grafts usually oversell:
it does not make the model better at language — it makes it better at
remembering what it has already seen;
it does not extend the attention window — the window is still 128
tokens; the layer can quote from outside it, not reason over it;
it does not compose, infer, or answer questions about what it holds;
the diffusion tier has no measurable upside on real text and used to
damage the model it was attached to (+0.81 nats) until its training
signal was fixed — keep it off.
The full list, with numbers: `[docs/LIMITATIONS.md](docs/LIMITATIONS.md)`
The negative results
This is the part that makes the rest believable.  Seven results came out
negative or against expectation, and all seven are measured, reproducible
and published:
| what we expected|what happened|
| --- | ---|
| the  topological  selection criterion picks the right facts|it does  not  — 0.000 precision vs 0.970 for plain geometric affinity|
| ‖L·1‖∞  detects the frozen state|it does  not  — the frustration index does; the old claim is now a test case|
| λ₂ predicts long-context quality|barely — pearson 0.20|
| a transformer loses at some scale|it  wins  at every scale we could train|
| a domain prior beats a frequency table|it helps (−0.006 nats) but a  unigram table wins 6× more|
| the diffusion tier is neutral on a frozen model|it  degrades  the model by  +0.81 nats  (fixed → +0.0000, but no upside either)|
| training the layer helps|it  hurts  — the training-free identity map is better (0.800 vs 0.650)|
Read it: `[docs/NEGATIVE-RESULTS.md](docs/NEGATIVE-RESULTS.md)`
The repository
superfluid/            the package (pure NumPy; torch only in torch_graft.py)
  graft.py             the attachable memory layer
  memory.py            the frozen register (20 B per fact, bit-exact)
  sheaf.py topology.py diffusion.py blocks.py autodiff.py ...
experiments/           every measurement; each writes a JSON into results/
tools/                 the artefact builder, the BPE tokenizer, the bundler
tests/                 the unit tests (pytest, or run them without it)
examples/              quickstart.py, graft_demo.py
docs/                  the design docs, the API, the limits, the theory report
results/               every number in the README, as JSON
| document|what it is|
| --- | ---|
| [[docs/GRAFT.md](docs/GRAFT.md)](docs/GRAFT.md)|what the layer does to a base model  — measured|
| [[docs/NEGATIVE-RESULTS.md](docs/NEGATIVE-RESULTS.md)](docs/NEGATIVE-RESULTS.md)|the seven things that did not work|
| [[docs/GETTING-STARTED.md](docs/GETTING-STARTED.md)](docs/GETTING-STARTED.md)|how to use it|
| [[docs/API.md](docs/API.md)](docs/API.md)|the API reference|
| [[docs/REPRODUCING.md](docs/REPRODUCING.md)](docs/REPRODUCING.md)|every command, every number|
| [[docs/LIMITATIONS.md](docs/LIMITATIONS.md)](docs/LIMITATIONS.md)|the honest list|
| [[docs/FRAMEWORK.md](docs/FRAMEWORK.md)](docs/FRAMEWORK.md)|the mathematics|
| [[docs/DESIGN-pinned-memory.md](docs/DESIGN-pinned-memory.md)](docs/DESIGN-pinned-memory.md)|the design of the frozen register|
| [[docs/COMPARISON.md](docs/COMPARISON.md)](docs/COMPARISON.md)|vs. KV cache, sliding window, RAG, Titans, Infini, MLA|
| [[docs/POSITIONING.md](docs/POSITIONING.md)](docs/POSITIONING.md)|the prior-art audit —  this is not revolutionary, and here is why|
| [[docs/index.html](docs/index.html)](docs/index.html)|the theory report (propositions, cost tables, citations)|
Status
A research prototype, honestly labelled.  Everything above was measured
on CPU, at small scale, on a character-level or 512-token subword corpus,
with a 131,392-parameter model.  The subword repeat is the closest thing
to a realistic tokenizer in this repository; the GPU run
(`experiments/gpu_graft.py`) is written and has not been executed here.
The project began as "replace attention with a sheaf diffusion".  The
measurements have said three times that this is not the way.  What is left
is smaller and much more defensible:
A constant-size memory tier that is bit-exact for a selected subset,
gracefully lossy for the rest, with a certified bound, an append-only
audit trail, 38 machine-checked claims, and a published account of the
seven places where it did not beat the alternatives.
Reproducing
python selftest.py                    # 38 checks
python experiments/run_all.py --full  # every measurement (~40 min, CPU)
Everything is pure NumPy, CPU-only, and needs no network.  The details and
the expected outputs: `[docs/REPRODUCING.md](docs/REPRODUCING.md)`
Contributing
Issues and pull requests are welcome — especially measurement
disagreements: if a number in the docs does not match what you get, that
is a bug in the code or in the documentation, and it is the most valuable
report you can file.  See `[CONTRIBUTING.md](CONTRIBUTING.md)`.
License
MIT — see `[LICENSE](LICENSE)`.  If you use this, please cite it
(`[CITATION.cff](CITATION.cff)`).