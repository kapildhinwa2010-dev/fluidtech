# Superfluid -- a constant-size exact memory tier for LLMs.
#
#   make test        self-verification and unit tests
#   make quick       a fast smoke run of every experiment
#   make full        every experiment, at full size (slow: ~40 min, CPU)
#   make bundle      rebuild the single-file bundle
#   make lint        ruff + a compile check
#   make release     build the sdist and the wheel

PY ?= python3
RESULTS := results

.PHONY: help test quick full bundle lint clean release artifacts help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

test: ## run the self-verification checks
	$(PY) selftest.py
	$(PY) superfluid_single.py --selftest
	$(PY) -m pytest -q tests

quick: ## a fast smoke run of every experiment
	$(PY) experiments/run_all.py

full: ## every experiment at full size
	$(PY) experiments/run_all.py --full

bundle: ## rebuild the single-file bundle
	$(PY) tools/bundle_single_file.py

lint: ## compile check + ruff (if installed)
	$(PY) -m compileall -q superfluid experiments tools tests examples
	$(PY) -m ruff check superfluid experiments tools tests examples

artifacts: ## rebuild the downloadable prior artifact
	$(PY) experiments/pretrain_prior.py --synthetic --tokens 200000

release: ## build the sdist and the wheel
	$(PY) -m build

clean: ## remove build artefacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
