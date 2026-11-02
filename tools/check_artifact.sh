#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
fail=0

need() {
  if [[ ! -e "$1" ]]; then
    printf 'MISSING: %s\n' "$1"
    fail=1
  fi
}

for path in \
  README.md LICENSE CITATION.cff CONTRIBUTING.md CODE_OF_CONDUCT.md \
  SECURITY.md pyproject.toml requirements.txt Makefile Dockerfile \
  .gitignore .pre-commit-config.yaml selftest.py superfluid_single.py \
  superfluid/py.typed superfluid/topology.py examples/quickstart.py \
  docs/GETTING_STARTED.md docs/CHANGELOG.md docs/API.md docs/REPRODUCING.md \
  docs/LIMITATION.md docs/GRAFT.md docs/NEGATIVE-RESULTS.md \
  .github/workflows/ci.yml .github/workflows/nightly.yml \
  .github/workflows/release.yml .github/PULL_REQUEST_TEMPLATE.md \
  .github/ISSUE_TEMPLATE/bug_report.md \
  results/e6_graft.json results/e6_real.json results/e6_real_bpe.json \
  results/e7_lossy_fix.json; do
  need "$path"
done

if (( fail != 0 )); then
  exit 1
fi

printf '==> Running verification\n'
python3 selftest.py
python3 superfluid_single.py --selftest
python3 run_without_pytest.py
python3 -m ruff check superfluid experiments tools tests examples selftest.py run_without_pytest.py

printf '==> Validating result JSON\n'
shopt -s nullglob
result_files=(results/*.json)
if (( ${#result_files[@]} == 0 )); then
  echo "MISSING: results/*.json"
  exit 1
fi
for path in "${result_files[@]}"; do
  python3 -c 'import json,sys; json.load(open(sys.argv[1], encoding="utf-8"))' "$path"
done

echo "All repository checks passed."
