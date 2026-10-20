#!/usr/bin/env bash
# Clean checkout to a fully built, artifact-loaded, verified engine, in one command.
#
#   bash scripts/setup_all.sh
#
# Steps: download dependencies -> download and export models + corpus -> build -> verify.
# Each underlying script is idempotent and can be re-run on its own.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Checking prerequisites"
missing=0
for tool in curl tar cmake python3; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "  missing required tool: ${tool}" >&2
        missing=1
    fi
done
if [ "${missing}" -ne 0 ]; then
    echo "Install the missing tools and re-run." >&2
    exit 1
fi
command -v nvcc >/dev/null 2>&1 \
    || echo "  warning: nvcc not found — the CUDA Toolkit is required to build the kernels"
command -v nvidia-smi >/dev/null 2>&1 \
    || echo "  warning: nvidia-smi not found — an NVIDIA GPU is required at runtime"

bash "${ROOT}/scripts/download_onnxruntime.sh"
bash "${ROOT}/scripts/setup_python.sh"
bash "${ROOT}/scripts/build.sh"
bash "${ROOT}/scripts/generate_artifacts.sh"
bash "${ROOT}/scripts/verify.sh"

echo
echo "==> Everything is set up and verified."
echo "    Try:  bash scripts/run_demo.sh"
echo "    Bench: bash scripts/bench_all.sh"
