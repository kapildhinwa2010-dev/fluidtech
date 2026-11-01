/usr/bin/env bash
# Check that everything the README claims is actually in the tree.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
fail=0

need() { if [ ! -e "$1" ]; then echo "MISSING: $1"; fail=1; fi; }

for f in README.md LICENSE CITATION.cff CHANGELOG.md CONTRIBUTING.md \
         CODE_OF_CONDUCT.md SECURITY.md pyproject.toml requirements.txt \
         Makefile Dockerfile .gitignore .pre-commit-config.yaml \
         selftest.py superfluid_single.py examples/quickstart.py \
         docs/GETTING-STARTED.md docs/API.md docs/REPRODUCING.md \
         docs/LIMITATIONS.md docs/GRAFT.md docs/NEGATIVE-RESULTS.md \
         .github/workflows/ci.yml .github/workflows/nightly.yml \
         .github/workflows/release.yml .github/PULL_REQUEST_TEMPLATE.md \
         .github/ISSUE_TEMPLATE/bug_report.md \
         prior/domain_prior.npz prior/manifest.json \
         results/e6_graft.json results/e6_real.json results/e6_real_bpe.json \
         results/e7_lossy_fix.json; do
  need "$f"
done

echo "==> the checks must pass"
python3 selftest.py > /dev/null && echo "  selftest: ok" || { echo "  selftest: FAILED"; fail=1; }
python3 superfluid_single.py --selftest > /dev/null && echo "  bundle: ok" || { echo "  bundle: FAILED"; fail=1; }
python3 tests/run_without_pytest.py > /dev/null && echo "  tests: ok" || { echo "  tests: FAILED"; fail=1; }

echo "==> the JSON must parse"
for f in results/*.json prior/manifest.json; do
  python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$f" || { echo "  BAD JSON: $f"; fail=1; }
done
echo "  all results parse"

[ $fail -eq 0 ] && echo "ALL GOOD" || { echo "SOMETHING IS MISSING"; exit 1; }
