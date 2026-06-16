#!/usr/bin/env bash
# Run a script with the project venv (see scripts/setup-python.sh).
set -euo pipefail

root="${PIXI_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
venv_py="${root}/.venv/bin/python"

if [[ ! -x "${venv_py}" ]]; then
  echo "Project venv not found. Run: pixi run setup-python" >&2
  echo "  or: bash scripts/setup-python.sh" >&2
  exit 1
fi

exec "${venv_py}" "$@"
