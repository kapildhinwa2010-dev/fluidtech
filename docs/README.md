# Documentation

This page is the guide to the project's Markdown documentation. Start with
the quickstart if you want to run the code, or use the topic map below to
find a specific answer.

## Start here

1. [Project overview](../README.md) — what the project does and the measured
   limits of the current research prototype.
2. [Getting started](GETTING_STARTED.md) — installation and first-use
   examples.
3. [API reference](API.md) — supported classes, methods, and parameters.
4. [Limitations](LIMITATION.md) — what the current implementation does not
   do and what has not been validated.

## Find a topic

| If you want to… | Read… |
|---|---|
| Attach the memory layer to a model | [Graft measurements and recipe](GRAFT.md) |
| Understand the mathematical framework | [Framework specification](FRAMEWORK.md) |
| Reproduce tests and research results | [Reproduction guide](REPRODUCING.md) |
| See the 10-million-token experiment | [Ten-million-token results](TEN-MILLION.md) |
| Review failed hypotheses and measured limits | [Negative results](NEGATIVE-RESULTS.md) |
| Compare with other memory approaches | [Comparison](COMPARISON.md) |
| Review the design history | [Pinned-memory design (historical)](DESIGN_pinned_memory.md) |
| Understand the project's research positioning | [Positioning (historical)](POSITIONING.md) |
| See a summary of project milestones | [Milestones](MILESTONES.md) |
| Read the release history | [Changelog](CHANGELOG.md) |
| Read the project blog draft | [Blog draft](BLOG.md) |
| Read the generated theory report | [Theory report](index.html) |

## How to read the research claims

- Results under `results/` are outputs from the experiments. Use the
  reproduction guide to regenerate them.
- Measured results on small CPU models do not establish performance on
  production-scale language models.
- `DESIGN_pinned_memory.md` and `POSITIONING.md` are historical research
  notes, not specifications of current behavior.
- The project is a research prototype. Review [Limitations](LIMITATION.md)
  and the repository's [security policy](../SECURITY.md) before adapting it
  for real users or sensitive data.

## Documentation contributions

Use the [contribution guide](../CONTRIBUTING.md). Keep instructions
copyable, link to related pages, and distinguish measured results from
projections or historical proposals.
