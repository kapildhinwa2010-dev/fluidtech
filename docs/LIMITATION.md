# Limitations

FluidLayer is a research prototype. The measurements are useful for
understanding the mechanism, but they do not establish production
performance or safety.

## What it does not do

- It does not improve general language modeling. In the measured setup, it
  can help a model recall previously observed continuations.
- It does not enlarge the model's attention window. Retrieved content is
  not equivalent to reasoning over the original context.
- It does not compose facts, answer arbitrary questions, or perform
  arithmetic over stored content.
- It is not a drop-in replacement for a KV cache. The mechanisms are
  complementary: the KV cache is position-indexed, while this memory is
  content-indexed.

## Measured limits

| Limit | Measurement |
|---|---|
| Cross-window recall | The reported benchmark reaches `0.800`, not `1.000`; tied embeddings make injection imperfectly selective. |
| Diffusion-tier behavior | An earlier training setup caused `+0.81` nats of loss. A fix removed the measured damage but demonstrated no standalone upside on that task. |
| Domain-prior benefit | The benefit shrinks on a stronger base model; a simple frequency baseline performed better in the reported experiment. |
| State accounting | The NumPy `state_bytes` figure is a logical payload estimate, not actual process memory. Python object and allocator overhead are additional. |
| Exact memory reach | The store is bounded. Its reachable history depends on capacity, key repetition, and the eviction policy. |

## Implementation constraints

- The exact store is a cache, not a database. It tracks majority
  continuations and evicts the least-corroborated entry; it does not answer
  arbitrary queries.
- A bit-exact token ID means the stored integer is returned exactly. It does
  not mean the value is correct, safe, or authenticated.
- The tied-readout injection is not perfectly selective. The `pc1`
  transform improved recall in the reported benchmark but did not make it
  perfect.
- Persistent diffusion state is outside the autodiff graph. Rebuild mode
  uses a bounded history and is a truncated estimator.
- The route head uses hard top-k selection; gradients do not pass through
  that discrete selection.

## Evaluation limits

Most experiments were run on CPU with small models and character-level or
small subword corpora. The included pretrained-model results cover one
specific model and task. The GPU-scale driver has not been validated across
production models, hardware, workloads, or concurrent sessions.

The synthetic corpus is especially favorable to continuation memory. Use
the real-text measurements and the [negative results](NEGATIVE-RESULTS.md)
when assessing claims.

## Production use

No production-readiness or security guarantee is made. The project does
not provide encryption, tenant isolation, authenticated provenance,
prompt-injection protection, or poisoning defenses by default. Review the
[security policy](../SECURITY.md) before using it with real users or data.
