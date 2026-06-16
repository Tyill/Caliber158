#!/usr/bin/env bash
# Create a project-local Python venv and install teacher-extraction deps.
set -euo pipefail

root="${PIXI_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
venv="${root}/.venv"
req="${root}/python/requirements.txt"

if [[ ! -f "${req}" ]]; then
  echo "missing ${req}" >&2
  exit 1
fi

if [[ ! -d "${venv}" ]]; then
  echo "creating ${venv}"
  if python3 -m venv "${venv}" 2>/dev/null; then
    :
  elif [[ -x "${root}/.pixi/envs/default/bin/python" ]]; then
    echo "system python3-venv missing; using pixi Python"
    "${root}/.pixi/envs/default/bin/python" -m venv "${venv}"
  else
    echo "failed to create venv. Install python3-venv or run: pixi install" >&2
    exit 1
  fi
fi

"${venv}/bin/pip" install --upgrade pip

if [[ "${CALIBER158_TORCH:-cuda}" == "cpu" ]]; then
  echo "installing CPU-only PyTorch (CALIBER158_TORCH=cpu)"
  "${venv}/bin/pip" uninstall -y torch torchvision torchaudio 2>/dev/null || true
  "${venv}/bin/pip" install -r "${root}/python/requirements-cpu.txt"
else
  echo "installing CUDA PyTorch (cu130; set CALIBER158_TORCH=cpu for CPU-only wheel)"
  "${venv}/bin/pip" uninstall -y torch torchvision torchaudio 2>/dev/null || true
  "${venv}/bin/pip" install -r "${req}"
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "note: NVIDIA driver not active — PyTorch will use CPU until you install nvidia-driver-580"
fi

echo "Python env ready: ${venv}/bin/python"
echo "HuggingFace cache: ${root}/models/huggingface (set CALIBER158_MODELS_DIR in .env)"
