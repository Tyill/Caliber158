#!/usr/bin/env bash
# Export variables from project .env for pixi run / pixi shell.
root="${PIXI_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
env_file="${root}/.env"

if [[ -f "${env_file}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${env_file}"
  set +a
fi

# Prefer project venv over system Python when present.
if [[ -x "${root}/.venv/bin" ]]; then
  export PATH="${root}/.venv/bin:${PATH}"
fi

# Keep HuggingFace downloads inside the repo (not ~/.cache).
models_dir="${CALIBER158_MODELS_DIR:-models}"
if [[ "${models_dir}" != /* ]]; then
  models_dir="${root}/${models_dir}"
fi
hf_home="${models_dir}/huggingface"
export HF_HOME="${HF_HOME:-${hf_home}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${hf_home}/hub}"
