#!/usr/bin/env bash
# Bootstrap the cyber/ project on a fresh clone.
#
# Detects `uv` (https://docs.astral.sh/uv/) and uses it preferentially;
# falls back to plain `python -m venv` + pip when uv is not available.
# Works on Windows (Git Bash / MSYS), macOS, and Linux/WSL.
#
# - Creates a Python venv at ./.venv
# - Installs schmidt_demos and its deps in editable mode
# - Clones CybORG++ at a pinned commit into ./third_party/
# - Runs the MiniCAGE smoke test
#
# Demo A runs without third_party/. Demos B, C, D require it.
#
# Idempotent: skips steps that are already done.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# CybORG++ pinned commit (2026-01-07: "Add MIT License to the project")
CYBORG_PP_REPO="https://github.com/alan-turing-institute/CybORG_plus_plus.git"
CYBORG_PP_COMMIT="c343476cb4490ba3e850561e0d11e7036ce50822"
CYBORG_PP_DIR="third_party/CybORG_plus_plus"

# Detect uv
HAS_UV=0
if command -v uv >/dev/null 2>&1; then
  HAS_UV=1
  echo "[bootstrap] uv detected: $(uv --version)"
fi

# 1. venv
if [ ! -d .venv ]; then
  if [ "$HAS_UV" -eq 1 ]; then
    echo "[bootstrap] creating venv at ./.venv via uv"
    uv venv .venv
  else
    echo "[bootstrap] creating venv at ./.venv via python -m venv"
    python -m venv .venv
  fi
fi

# Resolve the venv python (cross-platform)
if [ -x ".venv/Scripts/python.exe" ]; then
  PYTHON=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  echo "[bootstrap] could not find venv python" >&2
  exit 1
fi

# 2. editable install
echo "[bootstrap] installing schmidt_demos (editable) + deps"
if [ "$HAS_UV" -eq 1 ]; then
  uv pip install --python "$PYTHON" --quiet -e .
else
  "$PYTHON" -m pip install --quiet --upgrade pip
  "$PYTHON" -m pip install --quiet -e .
fi

# 3. CybORG++ at a pinned commit (vendored, not pip-installed)
if [ ! -d "$CYBORG_PP_DIR" ]; then
  echo "[bootstrap] cloning CybORG++ -> $CYBORG_PP_DIR"
  mkdir -p third_party
  git clone --quiet "$CYBORG_PP_REPO" "$CYBORG_PP_DIR"
fi
( cd "$CYBORG_PP_DIR" && \
    git fetch --quiet --depth 1 origin "$CYBORG_PP_COMMIT" && \
    git checkout --quiet "$CYBORG_PP_COMMIT" )

# 4. smoke test
echo "[bootstrap] running MiniCAGE smoke test"
"$PYTHON" scripts/smoke_test_minicage.py

# 5. Optional: warn if .env / ANTHROPIC_API_KEY missing (Demo D)
if [ ! -f .env ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "[bootstrap] note: .env not found and ANTHROPIC_API_KEY not set."
  echo "[bootstrap]       Demos A/B/C run without it; Demo D needs Anthropic API access."
  echo "[bootstrap]       Place ANTHROPIC_API_KEY=... in $ROOT/.env to enable Demo D."
fi

echo "[bootstrap] done."
