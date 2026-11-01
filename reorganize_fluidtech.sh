#!/usr/bin/env bash
set -u

# FluidTech repository reorganization script
# -----------------------------------------------------------------------------
# Purpose:
#   - Create the required project folder structure
#   - Move root-level files into their logical directories using git mv
#   - Handle typos and rename edge cases safely
#   - Remove duplicate files
#   - Verify the final repository state with git status
#
# Safety:
#   - Safe to re-run
#   - Skips missing files and existing destinations
#   - Preserves Git history by using git mv
# -----------------------------------------------------------------------------

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: This script must be run from inside a Git repository."
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT" || exit 1

log() {
  printf '%s\n' "$1"
}

warn() {
  printf 'WARN: %s\n' "$1" >&2
}

git_move() {
  local src="$1"
  local dst="$2"

  if [[ ! -e "$src" && ! -L "$src" ]]; then
    warn "Source missing; skipping: $src"
    return 0
  fi

  if [[ -e "$dst" || -L "$dst" ]]; then
    warn "Destination already exists; skipping: $dst"
    return 0
  fi

  mkdir -p -- "$(dirname "$dst")"

  log "Moving: $src -> $dst"
  git mv -- "$src" "$dst"
}

# -----------------------------------------------------------------------------
# 1) Create required directories
# -----------------------------------------------------------------------------
log "==> Creating required directories..."
mkdir -p \
  superfluid \
  experiments \
  results \
  tests \
  examples \
  docs \
  tools \
  .github/workflows \
  .github/ISSUE_TEMPLATE

# -----------------------------------------------------------------------------
# 2) Move files into superfluid/
# -----------------------------------------------------------------------------
log ""
log "==> Moving core package files into superfluid/"
git_move "__init__.py" "superfluid/__init__.py"
git_move "graft.py" "superfluid/graft.py"
git_move "memory.py" "superfluid/memory.py"
git_move "gpt2_model.py" "superfluid/gpt2_model.py"
git_move "gpt2_tokenizer.py" "superfluid/gpt2_tokenizer.py"
git_move "sheaf.py" "superfluid/sheaf.py"
git_move "diffusion.py" "superfluid/diffusion.py"
git_move "blocks.py" "superfluid/blocks.py"
git_move "autodiff.py" "superfluid/autodiff.py"
git_move "data.py" "superfluid/data.py"
git_move "losses.py" "superfluid/losses.py"
git_move "optim.py" "superfluid/optim.py"
git_move "bpe.py" "superfluid/bpe.py"

# -----------------------------------------------------------------------------
# 3) Move files into experiments/
# -----------------------------------------------------------------------------
log ""
log "==> Moving experiment files into experiments/"
git_move "e1_prior_perplexity.py" "experiments/e1_prior_perplexity.py"
git_move "e2_scaling.py" "experiments/e2_scaling.py"
git_move "e4_gbalance.py" "experiments/e4_gbalance.py"
git_move "e5_graft.py" "experiments/e5_graft.py"
git_move "e6_graft.py" "experiments/e6_graft.py"
git_move "e6_real_corpus.py" "experiments/e6_real_corpus.py"
git_move "e7_lossy_fix.py" "experiments/e7_lossy_fix.py"
git_move "e9_ten_million.py" "experiments/e9_ten_million.py"
git_move "e10_gate.py" "experiments/e10_gate.py"
git_move "e11_exact.py" "experiments/e11_exact.py"
git_move "e12_router.py" "experiments/e12_router.py"
git_move "e13_knn_baseline.py" "experiments/e13_knn_baseline.py"
git_move "e14_real_gpt2.py" "experiments/e14_real_gpt2.py"
git_move "run_all.py" "experiments/run_all.py"
git_move "pretrain_prior.py" "experiments/pretrain_prior.py"
git_move "gpu_graft.py" "experiments/gpu_graft.py"

# -----------------------------------------------------------------------------
# 4) Move files into results/
# -----------------------------------------------------------------------------
log ""
log "==> Moving result artifacts into results/"
git_move "e1_prior.json" "results/e1_prior.json"
git_move "e5_graft.json" "results/e5_graft.json"
git_move "e6_graft.json" "results/e6_graft.json"
git_move "e6_real.json" "results/e6_real.json"
git_move "e6_real_bpe.json" "results/e6_real_bpe.json"
git_move "e7_lossy_fix.json" "results/e7_lossy_fix.json"
git_move "e10_gate.json" "results/e10_gate.json"
git_move "e11_exact.json" "results/e11_exact.json"
git_move "e12_router.json" "results/e12_router.json"
git_move "e13_knn.json" "results/e13_knn.json"

# -----------------------------------------------------------------------------
# 5) Move files into tests/
# -----------------------------------------------------------------------------
log ""
log "==> Moving test files into tests/"
git_move "conftest.py" "tests/conftest.py"
git_move "test_bpe.py" "tests/test_bpe.py"
git_move "test_graft.py" "tests/test_graft.py"
git_move "test_memory.py" "tests/test_memory.py"
git_move "test_selftest.py" "tests/test_selftest.py"
git_move "test_topology.py" "tests/test_topology.py"
git_move "test_torch_parity.py" "tests/test_torch_parity.py"

# -----------------------------------------------------------------------------
# 6) Move files into examples/
# -----------------------------------------------------------------------------
log ""
log "==> Moving example scripts into examples/"
git_move "quickstart.py" "examples/quickstart.py"
git_move "graft_demo.py" "examples/graft_demo.py"
git_move "gate_demo.py" "examples/gate_demo.py"
git_move "gate_demo-1.py" "examples/gate_demo-1.py"

# -----------------------------------------------------------------------------
# 7) Move files into docs/
# -----------------------------------------------------------------------------
log ""
log "==> Moving documentation into docs/"
git_move "API.md" "docs/API.md"
git_move "BLOG.md" "docs/BLOG.md"
git_move "CHANGELOG.md" "docs/CHANGELOG.md"
git_move "COMPARISON.md" "docs/COMPARISON.md"
git_move "DESIGN_pinned_memory.md" "docs/DESIGN_pinned_memory.md"
git_move "FRAMEWORK.md" "docs/FRAMEWORK.md"
git_move "GRAFT.md" "docs/GRAFT.md"
git_move "MILESTONES.md" "docs/MILESTONES.md"
git_move "NEGATIVE-RESULTS.md" "docs/NEGATIVE-RESULTS.md"
git_move "POSITIONING.md" "docs/POSITIONING.md"
git_move "REPRODUCING.md" "docs/REPRODUCING.md"
git_move "TEN-MILLION.md" "docs/TEN-MILLION.md"
git_move "index.html" "docs/index.html"

# -----------------------------------------------------------------------------
# 8) Move files into tools/
# -----------------------------------------------------------------------------
log ""
log "==> Moving utility scripts into tools/"
git_move "bundle_single_file.py" "tools/bundle_single_file.py"
git_move "download_gpt2.py" "tools/download_gpt2.py"
git_move "fetch_pypi.py" "tools/fetch_pypi.py"
git_move "check_artifact.sh" "tools/check_artifact.sh"
git_move "release.sh" "tools/release.sh"

# -----------------------------------------------------------------------------
# 9) Move files into .github/workflows/
# -----------------------------------------------------------------------------
log ""
log "==> Moving GitHub workflow files"
git_move "ci.yml" ".github/workflows/ci.yml"
git_move "nightly.yml" ".github/workflows/nightly.yml"
git_move "release.yml" ".github/workflows/release.yml"

# -----------------------------------------------------------------------------
# 10) Move issue templates and repo metadata files
# -----------------------------------------------------------------------------
log ""
log "==> Moving GitHub config files"
git_move "bug_report.md" ".github/ISSUE_TEMPLATE/bug_report.md"
git_move "dependabot.yml" ".github/dependabot.yml"
git_move "PULL_REQUEST_TEMPLATE.md" ".github/PULL_REQUEST_TEMPLATE.md"

# -----------------------------------------------------------------------------
# 11) Handle rename and typo corrections
# -----------------------------------------------------------------------------
log ""
log "==> Handling special rename and cleanup cases"

if [[ -e "torch_ graft.py" || -L "torch_ graft.py" ]]; then
  if [[ ! -e "superfluid/torch_graft.py" && ! -L "superfluid/torch_graft.py" ]]; then
    log "Renaming: 'torch_ graft.py' -> 'superfluid/torch_graft.py'"
    git mv -- "torch_ graft.py" "superfluid/torch_graft.py"
  else
    warn "superfluid/torch_graft.py already exists; skipping"
  fi
fi

git_move "GETTING_STRATED.md" "docs/GETTING_STARTED.md"
git_move "LIMATATION.md" "docs/LIMITATION.md"

if [[ -e "license " || -L "license " ]]; then
  if [[ ! -e "LICENSE" && ! -L "LICENSE" ]]; then
    log "Renaming: 'license ' -> 'LICENSE'"
    git mv -- "license " "LICENSE"
  else
    warn "LICENSE already exists; skipping"
  fi
fi

git_move "modle.py" "model.py"
git_move "siperfluid_single.py" "superfluid_single.py"

# -----------------------------------------------------------------------------
# 12) Delete duplicate documentation file
# -----------------------------------------------------------------------------
if [[ -e "NEGATIVE-RESULT.md" || -L "NEGATIVE-RESULT.md" ]]; then
  log "Deleting duplicate file: NEGATIVE-RESULT.md"
  git rm --ignore-unmatch -- "NEGATIVE-RESULT.md" >/dev/null 2>&1 || rm -f -- "NEGATIVE-RESULT.md"
fi

# -----------------------------------------------------------------------------
# 13) Final verification
# -----------------------------------------------------------------------------
log ""
log "==> Verifying final repository state"
git add -A
git status --short

log ""
log "Done. Review the git status output above to confirm the final structure."
