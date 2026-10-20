#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-python3}"
VENV="${ROOT}/.venv"

if [ ! -d "${VENV}" ]; then
    echo "==> Creating virtual environment at ${VENV}"
    "${PYTHON}" -m venv "${VENV}"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

echo "==> Installing Python dependencies"
pip install --upgrade pip
pip install -r "${ROOT}/python/requirements.txt"
pip install pybind11

echo "==> Python environment ready (activate with: source ${VENV}/bin/activate)"
