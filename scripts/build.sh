#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "${ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${ROOT}/.venv/bin/activate"
fi

ORT="${ONNXRUNTIME_ROOT:-${ROOT}/third_party/onnxruntime}"
if [ ! -d "${ORT}/include" ]; then
    echo "error: ONNX Runtime not found at ${ORT}. Run scripts/download_onnxruntime.sh first." >&2
    exit 1
fi

PYBIND_DIR="$(python -m pybind11 --cmakedir)"

echo "==> Configuring (ONNXRUNTIME_ROOT=${ORT})"
cmake -B "${ROOT}/build" -S "${ROOT}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DONNXRUNTIME_ROOT="${ORT}" \
    -Dpybind11_DIR="${PYBIND_DIR}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo "==> Building"
cmake --build "${ROOT}/build" -j

ln -sfn "${ROOT}/build/compile_commands.json" "${ROOT}/compile_commands.json"
echo "==> Build complete: ${ROOT}/build"
