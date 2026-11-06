# FluidTech Codespec — standing instructions for Copilot

These rules apply to ALL work Copilot does in this repository. They are
not suggestions. The repository's history is full of "100% recall /
production-ready / all tests passing" claims that were never measured or
run; the reviews in docs/REVIEW-bulletproof.md, docs/REVIEW-experiments.md
and docs/NEGATIVE-RESULTS.md (N13) are the standard to uphold.

## 1. Run it, don't read it

- Code is "verified" only by EXECUTING it. Never claim a test passed
  unless you actually ran it and can paste the output.
- Use `python run_without_pytest.py` (the pytest-free runner) and/or
  `python -m pytest` for the suites. Run them at the repo root.
- Run the relevant suite after EVERY change, before committing.

## 2. A number that is printed must be asserted

- Any metric a test or example prints (recall, perplexity, speed,
  memory) must be backed by an `assert` with a justified bound —
  not merely displayed and declared a success.
- If the honest result is 0 or negative, report 0 or negative. Do not
  tune the scenario until it looks good.

## 3. Claims must be measured

- README.md, IMPLEMENTATION_SUMMARY.md and every doc may only contain
  performance numbers that were actually measured, with the method
  recorded (model, dataset, seed, hardware, commit SHA).
- Unmeasured goals must be labelled "hypothetical / not measured".
  Words like "guaranteed", "bulletproof", "100%", "production-ready",
  "military-grade" are banned unless a test proves them.

## 4. Real-model verification

- Perplexity / accuracy claims must be measured against a real model.
  The repo already has the harness for this:
  - `experiments/e14_real_gpt2.py`  (GPT-2 wiring)
  - `experiments/e15_scale_model.py` (scaled experiment)
  - `tools/download_hf.py` (checkpoint downloader with size verify)
  - `corpus.txt` (evaluation corpus)
- Prefer the smallest model that can test the claim (GPT-2 124M).
  A fresh perplexity delta must come from an actual A/B run:
  baseline model vs model + FluidLayer, same corpus, same seed.
- Write the numbers and the methodology into
  `docs/VERIFICATION.md` — every README claim: measured value,
  verdict (CONFIRMED / REFUTED / UNVERIFIABLE), and how it was tested.

## 5. Bug-fix discipline

- Fix the root cause, not the symptom; never weaken a test to make it
  pass; never delete a failing assertion.
- New code must come with tests that would have caught the bug it fixes.
- Follow the existing style: NumPy core stays NumPy-only (no torch
  import at module top level in numpy modules), torch code optional.
- Keep the `run_without_pytest.py` shim working (it does not support
  all pytest features; check it still runs all tests).

## 6. Reporting

- Commit messages: state what you RAN and what it printed, and what
  you only read.
- PR descriptions must include: commands executed, their exit codes,
  and the measured numbers backing any claim in the PR.
- When a claim fails verification, log it in
  `docs/NEGATIVE-RESULTS.md` with the next free N-number and update
  the README accordingly in the same PR.
