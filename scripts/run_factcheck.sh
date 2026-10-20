#!/usr/bin/env bash
# Fact-checks scientific claims end to end: zero-copy CUDA retrieval over SciFact, then a
# Llama 3.2 verdict with calibrated probabilities and a cited explanation.
#
#   bash scripts/run_factcheck.sh
#   CLAIM="Statins increase the risk of type 2 diabetes." bash scripts/run_factcheck.sh
#   LLAMA=models/llama/Llama-3.2-1B-Instruct bash scripts/run_factcheck.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "${ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${ROOT}/.venv/bin/activate"
fi

ORT="${ONNXRUNTIME_ROOT:-${ROOT}/third_party/onnxruntime}"
export LD_LIBRARY_PATH="${ORT}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ROOT}/build/bindings:${ROOT}/python:${PYTHONPATH:-}"

LLAMA="${LLAMA:-${ROOT}/models/llama/Llama-3.2-3B-Instruct}"
[ -d "${LLAMA}" ] || LLAMA="meta-llama/Llama-3.2-3B-Instruct"

CLAIMS=(
    "${CLAIM:-Taking aspirin daily reduces the risk of colorectal cancer.}"
)

for claim in "${CLAIMS[@]}"; do
    python "${ROOT}/python/examples/fact_check.py" \
        --config "${ROOT}/configs/scifact.json" --llama "${LLAMA}" --claim "${claim}"
done
