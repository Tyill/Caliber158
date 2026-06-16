#!/usr/bin/env bash
# Export variables from project .env for pixi run / pixi shell.
# Does not override variables already set in the parent environment.
root="${PIXI_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
env_file="${root}/.env"

if [[ -f "${env_file}" ]]; then
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    [[ -z "${line}" ]] && continue
    key="${line%%=*}"
    val="${line#*=}"
    key="${key%"${key##*[![:space:]]}"}"
    val="${val#"${val%%[![:space:]]*}"}"
    if [[ -z "${!key+x}" ]]; then
      export "${key}=${val}"
    fi
  done < "${env_file}"
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
