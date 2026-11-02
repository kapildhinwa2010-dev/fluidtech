#!/usr/bin/env bash
# Release the project: verify, build, tag, publish.
#
#   tools/release.sh 2.0.0
#
# The checks run first and ABORT the release if they fail.  That is the
# point: a tag must never point at a broken tree.
set -euo pipefail

VERSION="${1:-}"
if [ -z "$VERSION" ]; then
  echo "usage: tools/release.sh <version>   (e.g. 2.0.0)" >&2
  exit 2
fi

PY="${PY:-python3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> the checks"
"$PY" selftest.py
"$PY" fluidlayer_single.py --selftest
"$PY" run_without_pytest.py

echo "==> the version must match"
grep -q "version = \"$VERSION\"" pyproject.toml || {
  echo "pyproject.toml does not say $VERSION" >&2; exit 1; }

echo "==> building"
rm -rf dist build
"$PY" -m build

echo "==> the artefacts"
ls -la dist/
sha256sum dist/* | tee dist/SHA256SUMS

echo
echo "Now, by hand:"
echo "  git tag -a v$VERSION -m 'v$VERSION'"
echo "  git push origin v$VERSION"
echo "  # the release workflow publishes to PyPI and attaches files to GitHub Releases"
