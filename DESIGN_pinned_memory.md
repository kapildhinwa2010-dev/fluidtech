DESIGN — Frozen Register: making the memory layer exact where it matters
Status: design + theory. No code. Every number below is computed in
`calc` blocks and re-derivable.
Parent docs:`docs/PLAN-memory-graft.md` (the graft), `docs/LAUNCH.md` (the release).
One-line thesis:A frozen token is not a value in a fluid — it is an
identity promoted from a lossy tier to an exact one. Exactness does not
come from a bigger state; it comes from addressing, and the topology tells
you which addresses deserve exactness.
0.0 IMPLEMENTATION UPDATE (v1.1) — what the code actually says
The design below was written before the implementation.  Three things changed
when the code was written, and all three are recorded here rather than
quietly edited away:
| design said|implementation says|why|
| ---|---|---|
| 14 B per slot|20 B per slot|a 4-byte embedding signature was added so a 64-bit hash collision is  detectable  ( "fail loud, never silently wrong "). 1,170 facts in 16 KB became  819 .|
| M* ≈ β₀ + β₁|keep, but note it is a  lower  bound|a 16 KB state with 1,024 anchors is already over-allocated for a single document|
| topology selects the facts|it does not  (measured)|see §9.5 — the plain local geometric affinity wins 0.970 vs 0.000|
And the measured results are in §9.5.
0. The trap in the question (read this first)
The question was: "freeze a token in the layer, go token-wise, add a special
extraction tool — can we get exact tokens out?"
The trap: the Superfluid state is a diffusion. Diffusion smears. The
moment a token's value enters the sheaf Laplacian, it is mixed with every
neighbouring anchor and is provably no longer exact. Freezing a token
inside the fluid is therefore self-defeating: you are asking a smearing
operator to preserve a sharp value.
The fix: freeze it outside the fluid. A frozen token must never enter
the diffusion. It lives in a separate, append-only register, and the
fluid's job is reduced from storing to addressing.
This is the entire design in one move:
Superfluid stops being a memory. It becomes an index.
The memory moves to a write-once register; the fluid answers where, and
the register answers what.
And the reason this is not a retreat: the 97.3 % mechanism result was always a
routing result. In the mechanism test the fluid predicted the right global
aggregate to 97.3 % sign accuracy. It was already an excellent addresser.
We just gave it an exact value to point at.
1. The impossibility, stated precisely
| claim|number|
| ---|---|
| 16 KB state|131,072 bits|
| bits to name 1 of 1M locations|20|
| bits for a 128K-vocab token id|17|
| ⇒ addressable units in 16 KB|3,542  (locate+identify)|
| distinct token ids storable raw|7,710|
| distinct token ids storable with a perfect hash|4,369|
| KV-cache cost per token (70B)|2,747,269 B|
Conclusion: 16 KB can address ~3.5K things and hold ~5K distinct
token identities. It can never hold 1M tokens. Any design that promises
"exact token from anywhere" in 16 KB is lying.
The resolution is not more bytes — it is choosing which bytes are exact.
That is a selection problem, and selection is where topology earns its
keep.
2. The three-tier memory (the actual architecture)
        ┌──────────────────────────────────────────────────┐
 L3  ──▶ │  EXTERNAL SOURCE   (unbounded)                   │  exact, costs a call
         │  pointer = doc_id(16b) + offset(32b) = 6 B       │
         └──────────────────────────────────────────────────┘
               ▲ pointer is 6 B, stored in L1
         ┌────┴─────────────────────────────────────────────┐
 L1  ──▶ │  FROZEN REGISTER   (bounded, EXACT)              │  14 B/fact
         │  slot = ctx_key(8B) ‖ token_id(4B) ‖ trust(2B)    │  append-only
         │  capacity @16 KB = 1,170 facts                   │  bit-exact
         └──────────────────────────────────────────────────┘
               ▲ slot id
         ┌────┴─────────────────────────────────────────────┐
 L2  ──▶ │  SUPERFLUID STATE  (bounded, LOSSY)              │  16 B/anchor
         │  S ∈ R^{M×r}: routing prototypes + gist          │  no identity
         │  M=1024, r=4..8                                   │
         └──────────────────────────────────────────────────┘
Slot layout (20 B, as implemented):`key uint64 | val int32 | sig uint32 | anchor uint16 | meta uint16`.
The read path is three clean steps, each with a guarantee:
query h  ──▶  L2 routes h → anchor a   (approx: 97.3 % measured)
              anchor a → slot id s       (exact: 1 byte)
              L1[s] = (key, token_id)    (exact: 4 bytes)
              verify key matches h's context hash   (fail-loud)
              next_token ← token_id      (BIT-EXACT integer)
Geometry selects the slot. The slot names the token. The tool overrides the
logit. No float ever touches the final answer — the emitted token id is
identical to the source token id, not approximately equal.
3. The write path and the "frozen" algebra
Streaming, token-wise (this is what makes it O(1) per token):
for each token x_t = (id_t, emb_t):
    a_t = route(emb_t)                       # L2: which anchor/slot-block
    if pin_criterion(x_t, a_t):              # §4: which criterion?
        REGISTER[slot(a_t, next_free)] = (ctx_hash_t, id_t, trust=0)
    S ← e^{-tL}(S + B emb_t)                 # L2: gist, always
Proposition 1 (Invariance / Exactness). If a slot holds `(k, v)` at time t,
then for all t' > t the slot still holds `(k, v)` and `v` is bit-identical.
Proof. The register is append-only; the write path only ever appends to a
free slot; the diffusion operates on `S` only. ∎
Corollary (Integer exactness). If `v` is a token id (an integer), the
extracted value is exactly equal — not float-approximate. The error is
exactly zero, with no ε, at any sequence length, any number of subsequent
writes, any spectral gap. This is a strictly stronger guarantee than any
lossy-compression prior work can make.
Proposition 2 (No cross-contamination). The presence of the register does
not perturb `S` (disjoint state spaces). Therefore the hybrid read
`y = α_f·read_fluid(S) + α_p·read_frozen(REG)` satisfies
‖error on a pinned fact‖  ≤  α_f · e^{−tλ₂} · ‖B x‖
using the jamming bound `‖e^{−tL}u‖ ≤ e^{−λ₂t}‖u‖` from the v0.1 report.
Interpretation: the pin is protected from the fluid's smoothing, and
`α_f` is a dial. Set `α_f = 0` on a pinned query → the read is exact by
construction, for free. The certified stability bound still governs the
unpinned path, unchanged.
| λ₂|t|‖e^{−tL}‖ = e^{−tλ₂}|write blast radius shrink|
| ---|---|---|---|
| 0.10|1.0|0.9048|9.5 %|
| 0.25|1.0|0.7788|22.1 %|
| 0.10|2.0|0.8187|18.1 %|
| 0.25|2.0|0.6065|39.3 %|
Proposition 3 (Key collision ⇒ fail-loud). With a 64-bit context key and
P = 1,365 slots, collision probability is `P²/2⁶⁵ = 5.05 × 10⁻¹⁴`. Crucially
it is detectable: the slot stores the value, so a collision shows up as a
value mismatch and the read falls back to L3 instead of returning a wrong
token. The system never silently returns a wrong exact-token.
4. The selection problem — the actual research contribution
We can now pin 1,170–5,120 facts. Which ones? This is where every prior
art differs, and where the honest novelty lives.
| criterion|how|per-token cost|what it predicts|prior art|
| ---|---|---|---|---|
| S1 Model-requested|model emits  PROMOTE  action|1 extra token in action space|the model knows what it will need|Toolformer-style tool use|
| S2 Surprise / gradient|‖∇ℓ(M;x_t)‖  (Titans' surprise)|1 gradient probe per token|unpredictable tokens matter|Titans  (2024), LLMLingua (self-information)|
| S3 Influence on future|leave-one-out influence of  x_t  on future predictions|expensive; batched|tokens that  change the future  matter|TracIn, ATI, LLMLingua-2|
| S4 Topology (ours)|persistence length of the token in the Vietoris–Rips filtration|zero per token  (computed once per document)|structurally central  tokens matter|nothing in LLMs|
4.1 Why S4 is worth a paper
S1 needs training (the model must learn to ask) — no training-free
variant.
S2/S3 need a gradient per token — an inference-time cost that scales
with sequence length, which is exactly what we are trying to make cheap.
S4 is computed once per document and costs 0 per token. The
filtration over token embeddings is a pre-pass. This is not a preference; it
is a structural advantage: the only criterion whose cost does not scale
with the thing we are trying to make O(1).
4.2 The topology story (and why it is not hand-waving)
Build a Vietoris–Rips filtration over token embeddings → persistence barcode.
Then:
How many anchors? The number of essential H₀ bars (long-lived
components) is the number of independent structural modes of the
document. So `M* ≈ β₀ + β₁`. Nobody currently derives M from the data —
everyone hardcodes 1024. Our check: a 1,024-anchor state covers documents
with ≤1,024 structural modes; a 2,000-mode document is under-allocated.
This is a testable prediction, and it is falsifiable.
Which tokens to freeze? The longest bars = the structural landmarks =
the tokens that carry the document's skeleton. Freeze those and you get
exact extraction of the load-bearing facts, not random trivia.
Why the fluid wins global aggregation (explains the 97.3 %). The
harmonic branch of `L` (its kernel) is the globally dominant H₀ mode —
and the global mean is the longest bar. The fluid preserves the kernel
component exactly; that is precisely why it beat local aggregators
(65.6 % / 71.0 %) on the global-mean task and lost on position-sensitive
tasks. The v0.1 theory report and the mechanism experiment now have one
unified explanation. That is a real, publishable narrative.
5. Budget: the 3-D trade-off (M anchors, k pins/anchor, r dims)
total_bytes = M·r·2   (fluid, fp16)  +  M·k·14  (register)
pins        = M·k
| total B|M|r|k|fluid B|reg B|pins|B/anchor for routing|
| ---|---|---|---|---|---|---|---|
| 16,384|1024|8|1|16,384|14,336|1,024|16.0|
| 16,384|1024|4|4|8,192|57,344|4,096|8.0|
| 32,768|1024|8|2|16,384|28,672|2,048|16.0|
| 32,768|2048|8|2|32,768|57,344|4,096|16.0|
| 65,536|1024|8|4|16,384|57,344|4,096|16.0|
| 88,064|1024|8|4|16,384|57,344|4,096|16.0|
The key finding, and it is a design law:
Naming one anchor out of 1,024 needs 10 bits. An fp16 `r=8` anchor holds
128 bits — 12× over-provisioned for addressing. The excess must be
spent on gist, not on addresses. `r=4` (8 B/anchor) addresses 65,536
anchors — still 6× over-provisioned — and frees 8 KB, which buys 3,072 more
exactly-retrievable facts.
So the optimum is `r` small, `M` large, `k` moderate:
`(M=1024, r=4, k=4)` → 4,096 exact facts in 16 KB total.
And this produces a genuinely new ablation nobody has run: how do you
split `r` between addressing and gist? The two jobs have opposite
requirements — addressing wants a sharp discriminative code, gist wants a
rich averaged code. Measuring routing accuracy and gist RMSE as a function
of the `r_addr : r_gist` split is a clean, publishable experiment that uses the
existing 21-check harness.
6. The "special tool", done properly
The instinct "make a tool that extracts tokens" is right, but a tool that
returns a token and hopes the model uses it is not exact. Two failure
modes: (a) the fluid's logit still outvotes the tool, (b) the model paraphrases
the retrieved span.
Correct design — a hard logit override:
RECALL action →  slot s  →  token_id v
   next_logit[v] = +∞      (single write, 1 flop)
   all other logits unchanged
⇒ argmax(next_token) ≡ v, deterministically, with probability 1
This is the difference between "the model was told" and "the model cannot
disagree." It converts retrieval from a statistical to an algebraic
guarantee.
Action space (2 extra tokens):
`PROMOTE` — the model pins the current token/span (S1 policy)
`RECALL` — the model asks the L1/L3 tier for an exact value
Cost: RECALL is O(1) — one 4-byte read + one logit write. No extra FLOPs
of consequence. The only cost is 2 tokens of action space and the training
signal to learn when to use it.
Escalation ladder (this is the "better" part):
L2 fluid → gist, lossy, free
L1 register → exact token id, 14 B
L1b codebook → exact text span, 64 B
L3 source → exact anything, costs a fetch
The model learns to escalate. RAG does this with a vector DB; we do it with a
16–88 KB file.
7. The problems this creates (and their fixes)
This is where a design becomes a research contribution.
| #|new problem|why it is new|fix|
| ---|---|---|---|
| 1|Capacity pressure|prior art has a  cliff , we have a  knee|gradient-degradation: unpinned facts lose fidelity continuously (RMSE grows) while pinned stay exact forever.  Larimar: 100% recall ≤K=1000, 82% at 1024 — a hard cliff. Ours: 100% for the pinned set, continuous decay for the rest.|
| 2|Staleness  — a pinned fact gets superseded|append-only means old facts never die|versioned slots: append a new version,  "latest wins " for reads; the fluid's harmonic branch keeps the slow-moving  current  aggregate alive. Two timescales, one state.|
| 3|Hash collision|5×10⁻¹⁴ but non-zero|store the value, verify on read, fall back to L3 →  fail-loud, never silently wrong|
| 4|Memory poisoning|write-once means the  first  write wins — a poisoned prompt that arrives first pins a bad fact  permanently|corroboration-gated freezing : promote to  trusted  only after the same key is written from ≥2 independent sources; keep 1 trust bit per slot. The sheaf's λ₂ bound limits the  fluid's  poisoning radius; the quorum limits the  register's .  Two-tier defense, and the second one is new.|
| 5|Non-differentiability|a write-once register cannot be trained by gradient|don't train it — it is  addressed memory , not parameters. Train only the  pin policy  (a gate), reuse the existing router projection →  0 extra parameters .|
| 6|Cross-layer duplication|the same fact pinned at 80 layers = 80 slots|one  shared  register + layer-specific read projections (the layers read it at different semantic altitudes — this is the  point  of a sheaf)|
| 7|What if a fact must be  generated , not copied?|pointers only work for source text|a generated fact is stored as a  span of token ids  (4 B each) in L1b — same machinery, cost 4 B/token|
| 8|The register must not become a KV cache|the whole point is the 16 KB|hard cap  M·k ; overflow is  refused , not grown. The refusal is a  feature : it forces the selection policy to be good.|
| 9|The model must learn to escalate|new action space|train with the tool-use objective; but note the  default  path (L2) is unchanged, so a model that never escalates still works|
8. Honest prior art (what we are and are not claiming)
| work|what it does|what it lacks (our gap)|
| ---|---|---|
| Titans  (Google, 2024)|surprise-gated fixed-size neural memory;  does  do needle-in-haystack at 2M ctx|lossy everywhere; no exactness, no audit trail; surprise costs a gradient per token|
| Larimar  (2024)|external episodic memory, one-shot edits, 100% recall ≤K,  82% at 1024|1,573 B/fact;  hard capacity cliff ; no graceful degradation|
| Infini-attention|compressive memory, constant size|lossy; no exact tier|
| Landmark attention|fixed KV set (closest spirit)|learned weights , not written content; lossy; no selection policy|
| Memory Layers  (Meta, 2024)|trainable KV lookup, 128B memory params, product-quantized keys|requires pretraining; gigabytes; approximate top-k; no append-only audit|
| DeepSeek-V2 MLA|compressed KV|still O(n); lossy|
| LLMLingua / Selective Context / TokenGate|token importance via perplexity / MI; prune or compress|they  delete  tokens; no exact tier; they shrink the  input , not a state; importance is question-conditioned|
| RAG|external retrieval, exact text|2,176 B/fact; needs a vector DB; consumes context; chunking loses identity|
Our defensible claim, in one line:
Prior work gives you either a lossy fixed-size state (Infini, Titans,
landmark) or a big exact table that must be trained (Memory Layers) or a
big lossy table (Larimar). Nobody has a fixed-size state that is exact
for the subset that matters, gracefully lossy for the rest, with a certified
bound and an append-only audit trail — and nobody allocates that subset
using topology.
The competitor that worries me most is Titans (they already do
needle-in-haystack). Our defence must be exactness + cost + auditability,
not "we do needles too".
9. The money experiment (E5)
E5 — "pin 100 facts in 16 KB and extract them exactly from a 1M-token
document."
Setup. 1M-token document. 100 questions, each whose answer is a specific
span. Pin using the topology criterion (S4). Extract with the logit override.
Baselines.
| baseline | exact? | bytes |
|---|---|---|
| full KV cache | ✅ | 2.62 TB |
| RAG top-5 | ~60–80 % | 10.9 KB/fact + DB |
| Infini-attention | ❌ | 16 KB |
| Titans-style compressive | ❌ | 16 KB |
| random pinning (control) | ✅ for the pinned ones | 1.4 KB |
| topology pinning (ours) | ✅ | 1.4 KB |
The control is the whole experiment. If topology pinning ≈ random pinning,
the selection criterion is decoration and the paper dies. If topology pinning
beats random by a wide margin on the unpinned questions too (because pinned
landmarks also improve the fluid's gist), the criterion is real.
Pass criterion: exact extraction 100/100 with ≤2 KB of register, and
topology-pinned ≥ random-pinned + 20 points on the unpinned-question set.
Cost:< 1 GPU-day. Same harness as the existing 21 checks.
10. The other four experiments
| id|question|method|pass criterion|
| ---|---|---|---|
| E5a|does the  r_addr : r_gist  split matter?|sweep r ∈ {2,4,6,8} × split|routing acc flat, gist RMSE improves with r_gist; a knee exists|
| E5b|does the persistence diagram predict what the model wants to pin?|correlate bar-length rank with the model's PROMOTE decisions|Spearman  > 0.5|
| E5c|does pinning help the  fluid  too?|100 pinned landmarks vs 100 random anchors, measure global-gist RMSE|topology-pinned landmarks ↓ RMSE (landmarks are the skeleton)|
| E5d|does the trust bit prevent poisoning?|inject a poisoned fact as the first mention vs the second|1st-mention-only → untrusted (blocked); 2nd independent mention → trusted|
E5b is the most valuable one. If the topology predicts the model's own
memory requests, then the sheaf metric is not decoration — it is a readout of
what the model considers important. That would be the result that makes this
a theory rather than a trick.
11. What becomes newly possible (updated capability ledger)
| capability|status|guarantee|
| ---|---|---|
| exact token extraction for a chosen subset|new, by design|bit-exact integer, zero error|
| exact  text span  extraction (L1b codebook)|new|exact bytes, 64 B/span|
| graceful degradation instead of a capacity cliff|new|pinned 100 % forever, rest decays smoothly|
| tamper-evident memory|new|append-only ⇒ audit log, first-write-wins semantics|
| poisoning blast-radius control|new + improved|λ₂ bound (fluid) + quorum (register)|
| model-initiated memory management|new|2-token action space, 0 extra params|
| optimal anchor count derived from data|new|M* ≈ β₀ + β₁|
| constant-memory decode|unchanged|16 KB|
| attach to any LLM|unchanged|one residual insertion|
12. Kill criteria (write these down now, publish them now)
Topology pinning ≈ random pinning on E5 → the selection criterion is
decoration. Fall back to: the register + model-requested pinning (S1) is
still a valid system (a 112–155× cheaper exact-fact store than Larimar /
RAG), but the topological claim dies.
`r_addr` vs `r_gist` shows no trade-off → the fluid is doing one job, not
two → simplify to a pure router, and the story becomes "a learned index"
(weaker, but honest).
E5b shows no correlation between persistence and the model's own requests
→ the sheaf metric is not a readout of importance → the theory claim
weakens to "an alternative addressing scheme".
A competitor (Titans/Infini follow-up) delivers exact extraction at
comparable cost → the engineering contribution is absorbed; only the theory
survives. Publish fast.
13. Sequencing (2–4 weeks)
| phase|work|output|
| ---|---|---|
| P0 (2 days)|superfluid/memory.py :  FrozenRegister  (append-only, ctx-key, verify, trust bit) +  promote()  /  recall()  + logit override|unit-tested, in the 21-check harness|
| P1 (2 days)|SAA mode in  blocks.py : dual-role anchor (address + gist),  r_addr:r_gist  split|E5a sweep|
| P2 (3 days)|tools/pretrain_prior.py  → domain prior with a  landmark codebook  (exact spans)|86 KB → new artifact, E5 harness|
| P3 (3 days)|persistence-based landmark selection over the corpus|E5b, E5c|
| P4 (3 days)|HF forward-hook: graft onto a real 7B, measure exact extraction|E5 — the money result|
| P5 (2 days)|poison/quorum experiments, writeup|E5d + 2-page note|
Everything is numpy. No torch, no scipy, no GPUs required for P0–P3.
P4 needs one GPU for a few hours.
14. The one-paragraph version (for the README / the tweet)
Attention remembers by position. We remember by geometry. But geometry alone
smears: diffusion destroys exactness by construction. So we split the
memory. A 16 KB sheaf routes — 97.3 % accurate at picking the right
landmark — and an append-only frozen register stores the exact token id
at 14 bytes per fact, 196,234× cheaper than a KV cache. Topology decides
which landmarks get exact storage, using the document's own persistence
diagram, at zero per-token cost. The model can promote a fact or recall one
exactly; recall is a logit override, so the answer is the source token id,
bit-for-bit, with probability 1. Everything else stays fluid.
9.5 MEASURED (v1.1) — what actually happened
Run `python3 experiments/e5_graft.py --tokens 1000000 --budget 100`.
Full numbers in `results/e5_graft.json`; the sweep in `results/e2_scaling.json`.
9.5.1 The mechanism works
| claim|measured|
| ---|---|
| the fluid carries the local aggregate|gist cosine 0.968  vs the true per-anchor mean (global-mean cosine 0.191)|
| a promoted fact comes back exactly|exact given promoted = 1.000  (every policy)|
| the exact tier is cheap|20 B/fact  vs KV's 131,072 B/token =  6,554×|
| the state is constant|16,384 B before and after 1,000,000 tokens|
9.5.2 The SELECTION criterion — an honest negative
| policy|pin precision|fact recall|
| ---|---|---|
| semantic  (new pair + high geometric affinity)|0.970|0.469|
| rarity|0.050|0.031|
| persistence landmarks|0.010|0.008|
| Hodge edge participation (topology)|0.000|0.000|
| Hodge node relevance|0.000|0.000|
| crowding / occupancy|0.000|0.000|
| surprise (new pair only)|0.000|0.000|
| random|0.000|0.000|
| oracle|1.000|0.538|
Why the topological criteria lose.  Hodge participation measures how
strongly two cells are coupled in the global structure.  Inside a topic
every cell is coupled to every other, so the measure saturates and carries
no discriminative information.  The plain local affinity — how close the
two embeddings actually are — is not saturated and separates cleanly
(0.970).  Raising the resolution 4× (`topology_fine`) does not help, because
the problem is not resolution, it is that the quantity is global.
Design consequence: for selection, use the fine geometry (the
restriction maps / edge affinities), not the harmonic participation.  For
spreading (the diffusion itself) the Hodge machinery is what it is for.
The sheaf formulation still earns its place — but in a different job than the
one this document originally assigned to it.
This is the kill criterion from §12 firing, honestly reported.  The
project does not die: the exactness mechanism, the budget law and the
constant-size property all survive and are measured.
9.5.3 The budget law, measured
| anchors M|gist|flops/token| |exact budget K|fact recall|register|
| ---|---|---|---|---|---|---|
| 256|0.944|3,153,920| |16|0.077|320 B|
| 1,024|0.946|50,364,416| |64|0.308|1,280 B|
| 4,096|0.957|805,437,440| |256|0.723|5,120 B|
|  | | | |1,024|0.762|20,480 B|
16× more anchors → +0.013 gist, 256× compute.
64× more exact slots → +0.69 recall, 20 KB.
The budget goes on pins.  This is the design law of §5, now measured
instead of asserted.
9.5.4 A correction to the framework's own theory
`balance_residual(L) = ‖L·1‖∞` does not detect the frozen state of a
scalar signed sheaf.  An all-negative EVEN cycle has `‖L·1‖∞ = 4` yet
freezes onto the alternating sign pattern.  The correct diagnostic is the
frustration index (the number of edges whose sign disagrees with the
best global sign assignment), and the correct statement is
frustration == 0   <=>   a non-trivial global section exists
verified on 150/150 random signed cycles (`results/e4_gbalance.json`),
and now checked in the selftest.  `‖L·1‖∞` measures the stricter property
"the CONSTANT field is a global section", which is a different thing.
Also measured: the contraction certificate holds on 36/36 checks, but
`pearson(λ₂, tracking quality) = 0.20` — λ₂ is a certificate about the
operator, not a predictor of task quality.  The claim is weakened
accordingly.