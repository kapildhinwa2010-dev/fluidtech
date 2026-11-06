# REVIEW - experiments branch (Superfluid-Mamba & Ultra-Fluid)

Date: 2026-11-06
Scope: `experiments/superfluid_mamba_layer.py`, `experiments/ultra_fluid_layer.py`
and their test suites.
Method: the code was reviewed BY READING it in this session; it was not
executed locally.  CI is the executor of record for the fixes.  (This is
the honest caveat -- the earlier "All 20 tests passing!" commit messages
could not have been true for the code as written; see finding 1.)

This review applies the same standard as `REVIEW-bulletproof.md` on main,
and the same lesson (`NEGATIVE-RESULTS.md` N13): **a number that is
printed must be asserted.**

---

## Findings and fixes

### superfluid_mamba_layer.py

1. **The PyTorch forward pass could not have run as committed.**
   `out_sf = (state_sf.mean(dim=0) @ self.W_sf).expand(seq_len, -1)` --
   the left side is a 1-D tensor of shape `(d,)`, and `.expand(n, -1)`
   on a 1-D tensor raises a RuntimeError.  The commit message claiming
   "10/10 tests passing" therefore could not have been produced by
   actually running this file.  FIXED: unsqueeze to `(1, d)` first.
   The same bug existed in `ultra_fluid_layer.py`.

2. **SuperfluidDiffusion was not a diffusion.**
   `torch.einsum('md,mm->md', state, diffusion_matrix)`: the repeated
   `m` inside the second operand takes the DIAGONAL of the diffusion
   matrix, so the "diffusion" reduced to elementwise scaling by the
   diagonal -- no information ever propagated between nodes.  A
   shape-only test passed it.  FIXED: `einsum('mn,nd->md', ...)`; the
   test now seeds a one-hot state and asserts the spread.

3. **Uninitialized memory in AdaptiveGatedSheaf.**
   `np.divide(adjacency, row_sums + 1e-12, where=row_sums + 1e-12 != 0)`
   without `out=` leaves the "untouched" entries uninitialized -- exactly
   the bug fixed on main in the bulletproof review.  The rows of this
   particular graph are never all-zero, so plain division is safe.
   FIXED; the test now asserts a finite, row-stochastic adjacency.

4. **encrypt() could never be undone.**
   `Fernet(Fernet.generate_key())` created a NEW random key on every
   call, so ciphertext was permanently undecryptable, and the docstring
   claimed AES-256 while Fernet is AES-128-CBC + HMAC-SHA256.  FIXED:
   the key is derived deterministically from the session ID; claims
   corrected.

5. **Session isolation was decorative.**
   `_generate_key` used Python's salted `hash()` (not stable across
   processes) and also concatenated the raw key string, so the session
   prefix added nothing.  FIXED: deterministic session-ID namespacing;
   a cross-session isolation assertion added to the tests.

6. **GatedFusion was a no-op design.**
   The same gate was applied to every tier, which is equivalent to
   gating the SUM of the tiers: no tier could ever be favoured.
   FIXED: per-tier softmax gates; the test asserts the gates differ.

7. **NP twin: guaranteed crash on the first recall.**
   `value = self.store.get(curr_key, -1)` followed by `if value != -1`
   compares a stored ndarray with -1 -- a truth-value ambiguity error
   the moment a hit occurs.  It "worked" only because recall was 0.
   FIXED: `None` default like the torch twin.

8. **Dishonest tests.** The suite printed "Recall rate: 0.00%" and then
   "ALL TESTS PASSED" (and even "100% recall" in the summary).  FIXED:
   fresh-pass recall is asserted to be 0 by construction, and a replay
   pass asserts the measured recall against the computed expectation.
   The suite now lists what is and is NOT verified.

### ultra_fluid_layer.py

9. **Unregistered module created inside forward().**
   `kv_proj = nn.Linear(...)` was created inside `forward()`: never
   registered, never trained, a different random projection every call.
   FIXED: registered in `__init__`.

10. **Double residual in UltraFluidStack.**
    The layer adds its own internal residual; the stack then computed
    `h + skip_weight * h_layer` on top, double-counting the input.
    FIXED: the stack consumes the layer output directly.

11. **Identical if/else.** The `use_residual` True and False branches
    were byte-identical.  FIXED: the non-residual path actually omits
    the residual, and the test proves it (zero-init W matrices must
    yield a zero output, not `h`).

12. **CompressedUltraFluid claimed "50-80% memory reduction"** while
    its forward pass literally delegates to the base layer.  FIXED:
    labelled a pass-through until compression is actually wired.

---

## Claims that were overstated (now corrected in docstrings)

- "100% recall guarantee": the exact tier recalls only contexts that
  repeat verbatim; first sight is always a miss.
- "Perfect hash / double hashing": DOUBLE_64 is a single ~128-bit
  digest of the same string; collisions remain possible and there is
  no collision detection.
- "Production-ready / READY FOR PRODUCTION TESTING": this branch is
  explicitly experimental.
- All headline performance numbers (perplexity deltas, 100K/200K
  context, 99.99%/99.999% accuracy, speedups, memory footprints):
  hypothetical, never measured against a real LLM.  They are no longer
  printed as if they were test results.

## Known limitations left open (next steps)

- The Mamba scan is a Python-level sequential loop (a simplified, not
  hardware-aware scan); benchmarking it against attention as-is would
  be misleading.
- The superfluid tier emits a position-independent vector (mean over
  anchors); it cannot carry per-position information.
- The NP twin has no Mamba/Superfluid tiers (placeholders).
- CompressedUltraFluid does not compress anything yet.
- No CI job runs these suites on the experiments branch yet; wiring
  them in is the natural next step.
