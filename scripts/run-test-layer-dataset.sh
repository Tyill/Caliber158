#!/usr/bin/env bash
# CAL158L layer dataset roundtrip (not in make test).
set -euo pipefail

root="${PIXI_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${root}"

if [[ ! -x "${root}/.venv/bin/python" ]]; then
  echo "Project venv not found. Run: make setup-python" >&2
  exit 1
fi

"${root}/.venv/bin/pip" install -q pytest 2>/dev/null || true

export PYTHONPATH="${root}/python:${PYTHONPATH:-}"
"${root}/.venv/bin/python" -m pytest tests/test_layer_dataset.py tests/test_layer_scales.py -q
