# VERIFICATION — every README claim, measured in CI

This repository does not trust its own README. Every performance and
security claim is **measured** on every push to `main` by
[`experiments/e16_verify_readme.py`](../experiments/e16_verify_readme.py),
run by [`.github/workflows/verify-claims.yml`](../.github/workflows/verify-claims.yml).
The raw verdicts are published to the `verification-results` branch:

* `results/verification/run.json` — machine-readable verdicts
* `results/verification/run.md` — the same, as a table + the full log
* `results/logs/*.log` — the test suites' output, the claim-run log and the ruff lint log

Verdicts: **CONFIRMED** (the measurement matched the claim, and where a
bound is meaningful the number is *asserted*, not merely printed) /
**REFUTED** / **UNVERIFIABLE** (no runnable way to measure).

---

## The claims, as measured (run on commit 2d511480 — 9/9 CONFIRMED)

| Claim | Verdict | Measured |
|---|---|---|
| 100% recall for previously seen contexts (1/4/8/12-gram) | CONFIRMED | replay recall **1.000** at every ctx_len; never-seen text recalls 0.0000 (12-gram) — the store does not fabricate recall |
| Never-seen contexts recall ~0 (no fabricated recall) | CONFIRMED | fresh 12-gram recall 0.0000 over 6000 corpus words |
| The layer attaches to a real GPT-2 and runs (A/B, untrained) | CONFIRMED | sshleifer/tiny-gpt2, 69 chunks × 128 tokens: baseline **10.8249** nats/token, with layer **10.8254** (Δ ≈ +0.0005 — attaching is essentially free when nothing is recalled) |
| GGUF export writes a valid GGUF container | CONFIRMED | magic **b'GGUF'**; llama.cpp cannot EXECUTE a custom architecture without a backend patch |
| MILITARY level: Fernet encrypt/decrypt round-trip | CONFIRMED | round-trip ok at STANDARD and MILITARY; Fernet is AES-128-CBC + HMAC-SHA256 (not AES-256) |
| Tenant isolation: no cross-session reads | CONFIRMED | tenant_2 replays tenant_1's tokens on a fresh layer: **0 hits** |
| Measured overhead (forward < 50 ms/256 tok, keygen < 50 µs, lookup < 50 µs) | CONFIRMED | forward **4.6–9.4 ms**/256 tokens (~18–37 µs/token), keygen **3–6 µs**, lookup **3–5 µs** (CI runner CPU; varies run to run) |
| Bounded memory: state + capacity-bounded store stays under 100 MB | CONFIRMED | state **4096 B**; store ~**112 B/entry**; worst case at max_capacity 100000 ≈ **11.2 MB** — far below the 100 MB ceiling (the old "≤ 16 KB" row was REFUTED: 116 KB at only 1000 entries) |
| Thread safety (8 concurrent threads) | CONFIRMED | 8 threads × 50 forwards, **0 errors**, 6400 tokens |

---

## Bugs the verification found (and fixed)

The runs of this pipeline found **six real bugs** — which is the
point of running claims instead of reading them:

1. **The GGUF magic was b'GUFF'** — `0x46465547` packs the bytes in the
   wrong order; every exported file failed the container check. Fixed to
   `0x46554747` (b'GGUF').
2. **MILITARY encryption crashed** — Fernet was given a raw 32-byte key,
   which is invalid (it needs 32 url-safe-base64-encoded bytes). Fixed by
   deriving the key with sha256 + urlsafe_b64encode. Also: the README
   said AES-256; Fernet is AES-128-CBC + HMAC-SHA256. Both now say what
   the code does.
3. **`SAABranchBulletproof(d=768, config=...)` silently ignored `d`** —
   the constructor's M/r/d only applied when no config was passed, so a
   config with d left at its default (64) won and the layer crashed deep
   inside `_route` with a confusing matmul error. The README's own
   quickstart hit exactly this. Fixed: an explicit non-default constructor
   value now wins over the config.
4. **The verification itself produced nothing on its first run** — a
   green 45-second job with zero results: the committed e16 script was
   syntactically invalid (a string literal contained a real newline), and
   `git add results/` was silently emptied by `.gitignore`'s `*.log`.
   Fixed (chr(10) literals; `git add -f`). Lesson: *a green job is not
   a result* (N14 in the negative-results log).
5. **torch was a hard import in the numpy-only suites** —
   test_bulletproof.py and test_refactored_simple.py crashed the CI's
   numpy-only jobs. torch is now optional; the torch tests skip loudly.
6. **The CI lint job had been red since Nov 5** — 335 ruff violations
   (over-long lines, imports in the middle of the file, unused imports,
   unsorted import blocks, trailing whitespace) were hiding behind the
   red X while the other jobs stayed green. Fixed: the long lines and
   the import placement by hand, the mechanical rest by
   `ruff --fix --unsafe-fixes`, committed by this very workflow
   (the published lint log now ends with "All checks passed!").

## Honest corrections, made with the maintainer's approval

* the **"≤ 16 KB" memory budget** was REFUTED (the exact store grows
  with entries: 116 KB at 1000 entries). 100 MB is a hard **ceiling,
  not a target** — 4 KB state + a store that can never exceed
  `max_capacity` entries (measured worst case ~11.2 MB at the
  defaults, far below the ceiling).
* the **relative speed claims** (10x hash, 1.5x faster) were UNVERIFIABLE:
  after the review fixes there is no runnable "original" baseline. The
  README states the measured absolute overhead instead.
* the **"-0.30 nats" perplexity table** had no measurement behind it.
  The A/B harness now measures the layer on a real GPT-2; the layer is
  untrained, so no improvement is claimed without training.

## Run history

| Run | Commit | Outcome |
|---|---|---|
| 1 | dfc032fe | green in 45 s, ZERO output (bug 4) |
| 2 | 466e94ea | first real measurement: recall/GGUF-fix-needed, security/threads CONFIRMED, memory REFUTED, 3 harness errors |
| 3 | 795dbe5b | recall 100% replay CONFIRMED; GGUF + MILITARY CONFIRMED after bugs 1–3 |
| 4 | 6027fe61 | 8/9 CONFIRMED (memory ≤ 100 MB + isolation + speed in); perplexity harness error (hook returned a tuple) |
| 5 | 311e3184 | **9/9 CONFIRMED** — the table above |
| 6 | dc2b32fd | 9/9 CONFIRMED again (docs-only honesty pass) |
| 7 | 340bd5d2 | memory claim reworded ("< 100 MB ceiling, not a target"); 28 over-long lines + mid-file imports + B904 + F841 fixed by hand |
| 8 | 2d511480 | **9/9 CONFIRMED + lint clean** — the workflow autofixed the 304 remaining mechanical violations, committed them, and verified the result (ruff exit 0) |

## Rerun it

Any push to `main` runs the pipeline (it also has a
`workflow_dispatch` trigger). The results land on the
`verification-results` branch within a couple of minutes:

```bash
python experiments/e16_verify_readme.py --sha HEAD --out results/verification
```
