#!/usr/bin/env bash
# Smoke: Qwen3.6 MoE config + optional from_pretrained (set CALIBER158_MOE_SMOKE_LOAD=1).
set -euo pipefail

root="${PIXI_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${root}"

if [[ ! -x "${root}/.venv/bin/python" ]]; then
  echo "Project venv not found. Run: make setup-python" >&2
  exit 1
fi

export PYTHONPATH="${root}/python:${PYTHONPATH:-}"
exec "${root}/.venv/bin/python" "${root}/python/smoke_moe_model.py" "$@"
