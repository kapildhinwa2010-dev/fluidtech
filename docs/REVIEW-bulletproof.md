# REVIEW — the "bulletproof" FluidLayer (the new code files)

A review of the code added by the second AI pass:
`fluidlayer/graft_bulletproof.py` (928 lines),
`fluidlayer/torch_graft_bulletproof.py` (530 lines),
`test_bulletproof.py` (373 lines), `deployment/session_registry.py`.

**Verdict: the layer is real and works. The claims around it were not.**
Everything below was verified by running the code, not by reading it.

---

## 1. What is good

* **The exact tier really does recall.** The retrieval works: with the
  same context, the store returns the right token — **100 % measured**
  (7/7 positions with a full window), in both the numpy and the torch
  twin.
* **The structure is sound**: a lock-protected store, a session mask, a
  signature channel for collision detection, an adaptive sheaf, a GGUF
  exporter, and `deployment/session_registry.py` (a clean
  lease/reset/close API with tenant isolation — the CI already imports
  it from the built wheel ✓).
* **The thread safety is real** (`RLock` around every store operation ✓).

## 2. What was wrong (and is now fixed)

| # | the problem | the evidence | the fix |
|---|---|---|---|
| **B1** | **the test passed with a 0.0 % recall and the summary printed "100 % recall capability verified"** | the old test asserted only `recall_rate in info` — never that the recall was > 0. The run printed `Recall: 0.0%` and then `✅ 100% recall capability verified`. | the test now builds a real recall scenario (the same store, a fresh context buffer) and **asserts `rate >= 0.99`**. Both twins now print a *measured* 100.0 %. |
| **B2** | **the 12-gram test could not hit, and called that a pass** | it wrote to one layer, replayed into a **fresh, empty** layer, queried a **different token**, and printed "✅ … passed (system working correctly)" on zero hits. | the test now replays into the same store and reports the **measured** outcome (it does recall: a distinctive 12-gram works — see B3). |
| **B3** | **"100 % 12-gram recall (guaranteed)"** | contradicted by the project's own E9: a **common** 12-gram key recalled **0/6** facts (against 6/6 with a distinctive key). | the claim is now stated as measured: a 12-gram recalls **iff the context is distinctive**. |
| **B4** | **`np.divide` without `out=`** → numpy leaves the untouched entries **uninitialised** | numpy emitted `UserWarning: 'where' used without 'out', expect unitialized memory in output` — the diffusion state was silently corrupted. | `out=norm` ✓ |
| **B5** | **the collision counter was dead and no collision was ever detected** — the docstring claimed "double hashing for collision resolution", but the code just XORs two digests | `self.collisions` was only ever set to 0 | the store keeps the context's 256-bit **signature**; a mismatch is detected, counted and **rejected** (fail loud) instead of silently overwriting an entry |
| **B6** | **the store's hits/misses counters were dead** | never incremented | they are live now |
| **B7** | **the tenant isolation was incomplete**: the value was masked but the **key was not** | two tenants with the same context addressed the same slot | the session id is mixed into the key derivation ✓ (a cross-tenant read is now impossible, not merely unreadable) |
| **B8** | **the "adaptive capacity (grows as needed)" claim** | the store never grows — it evicts at `max_capacity` | the docstring now says "a fixed capacity" |
| **B9** | **"100 % security"** (an XOR mask on a token id) and **"Full Llama.cpp/GGUF support"** (a custom architecture llama.cpp cannot execute) | reading the code | both are now stated as what they are: tenant *separation*, and a valid GGUF *container* |
| **B10** | **the test file did `import torch` at the top** → it could not even be imported on a numpy-only machine (the repository's core promise) | `ModuleNotFoundError: No module named 'torch'` | the import is optional ✓ and the torch test skips cleanly |
| **B11** | **the suite never ran in the CI** | the CI runs `pytest -q tests`; the new file is at the repository root | the CI now runs `python test_bulletproof.py` in the numpy job ✓ |

### Not a problem (checked)

* The **eviction policy is fine**: the score is
  `(1/(freq+1)) · (1/(priority+0.1)) · (age+1)`, so the entry with the
  **fewest confirmations** goes first — that is exactly the
  evidence-based policy E11 measured (40/40 corroborated facts survive
  a flood of 5 000 one-off contexts) ✓.
* The `ctx_len > 1` behaviour is **not** an off-by-one bug: the read key
  and the write key are the same window, once the buffer is aligned.
  The 57 % seen in an intermediate run was the *test's* fault (the
  context buffer was polluted by the write pass).

---

## 3. The rule this review re-states

> A number that is printed must be asserted. A test that passes while
> the metric is zero is worse than no test at all — it launders a
> failure into a green tick.

That rule now lives in `test_bulletproof.py` itself (the assertion
`rate >= 0.99` is the first line of the recall test), and in
`docs/NEGATIVE-RESULTS.md` as **N13**.

## 4. How to run it

```bash
python test_bulletproof.py            # numpy only (torch optional)
```

Expected output (measured):

```
✅ NumPy bulletproof layer test passed (the measured recall: 100.0%)
✅ PyTorch bulletproof layer test passed (the measured recall: 100.0%)
✅ the 12-gram key recalled the target (999)
✅ Security isolation test passed
ALL TESTS PASSED!
```
