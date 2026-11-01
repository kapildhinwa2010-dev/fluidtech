## What this changes

<!-- one paragraph -->

## The measurement

<!-- REQUIRED for anything that changes behaviour: the before/after numbers
     and the command that produces them. A PR without a measurement cannot be
     reviewed, because nothing else in this project can be reviewed either. -->
before:
after:
command:
## The self-verification

- [ ] `python selftest.py` passes
- [ ] `python superfluid_single.py --selftest` passes
- [ ] `python -m pytest -q tests` passes
- [ ] a new check exists for any new behaviour

## If this changes a claim in the docs

- [ ] the number in `docs/` was updated
- [ ] `docs/NEGATIVE-RESULTS.md` was updated if a result got worse
- [ ] `docs/CHANGELOG.md` has an entry

## Dependencies

- [ ] no new runtime dependency (numpy only)
