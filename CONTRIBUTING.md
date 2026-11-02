# Contributing

Thanks for your interest in improving Superfluid. Contributions should be
measurable, reproducible, and consistent with the project's research scope.

## Development setup

```bash
git clone https://github.com/kapildhinwa2010-dev/fluidtech.git
cd fluidtech
python -m pip install -e ".[dev]"
make test
make quick
```

`make test` runs the self-verification, bundled self-test, and unit tests.
`make quick` runs the resource-bounded experiment smoke checks.

## Contribution principles

- **Measure behavior changes.** If a change affects a claim, include the
  before/after result and the command that reproduces it.
- **Keep negative results visible.** Update
  [`docs/NEGATIVE-RESULTS.md`](docs/NEGATIVE-RESULTS.md) when a result
  worsens or an assumption is disproven.
- **Add a check for new behavior.** Extend the tests or self-verification
  rather than relying on a manual claim.
- **Avoid new core dependencies.** The runtime package is intended to remain
  NumPy-based; PyTorch support is optional.
- **Fail explicitly.** Do not hide invalid input, overflow, or failed
  operations behind success-shaped defaults.
- **Keep documentation usable.** Use headings, fenced code blocks, working
  links, and clearly label measured results, projections, and historical
  notes.

## A good first contribution

1. Add a focused test for a property or bug.
2. Add or fix the implementation.
3. Run the relevant tests and `make lint`.
4. Update the documentation and changelog if user-visible behavior changed.

## Style

- Python: 4 spaces, 88 columns, and project Ruff rules.
- Use docstrings to explain intent or non-obvious constraints.
- Keep comments focused on the reason for a measurement or constraint.

## Report a bug

Open an issue with the smallest reproduction you can provide. Include the
command, expected result, actual result, and relevant environment details.
For security issues, follow the private reporting instructions in
[`SECURITY.md`](SECURITY.md) rather than filing a public issue.

## Code of conduct

Participation in the project is covered by the
[Code of Conduct](CODE_OF_CONDUCT.md).
