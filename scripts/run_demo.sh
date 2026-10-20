#!/usr/bin/env bash
# Runs one query through every configured variant of the engine so the differences between
# them are visible side by side.
#
#   bash scripts/run_demo.sh
#   QUERY="how does crispr gene editing work" bash scripts/run_demo.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "${ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${ROOT}/.venv/bin/activate"
fi

ORT="${ONNXRUNTIME_ROOT:-${ROOT}/third_party/onnxruntime}"
export LD_LIBRARY_PATH="${ORT}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ROOT}/build/bindings:${ROOT}/python:${PYTHONPATH:-}"

DATASET="${DATASET:-scifact}"
QUERY="${QUERY:-what causes type 2 diabetes}"
TOP_K="${TOP_K:-5}"

run_variant() {
    local title="$1"
    local config="$2"
    shift 2
    if [ ! -f "${config}" ]; then
        echo
        echo "── ${title}: skipped (no ${config})"
        return
    fi
    echo
    echo "=============================================================="
    echo " ${title}"
    echo "=============================================================="
    python "${ROOT}/python/examples/quickstart.py" \
        --config "${config}" --query "${QUERY}" --top-k "${TOP_K}" "$@"
}

run_variant "float32 baseline"            "${ROOT}/configs/${DATASET}.json"
run_variant "int8 storage (4x smaller)"   "${ROOT}/configs/${DATASET}_int8.json"
run_variant "binary + on-device rescore"  "${ROOT}/configs/${DATASET}_binary_rescore.json"
run_variant "hybrid BM25 + dense"         "${ROOT}/configs/${DATASET}_hybrid.json" --hybrid

echo
echo "=============================================================="
echo " C++ CLI (same query, no Python in the loop)"
echo "=============================================================="
"${ROOT}/build/apps/zcgvs_search_cli" \
    --embeddings "${ROOT}/data/${DATASET}.bin" \
    --metadata "${ROOT}/data/${DATASET}.meta" \
    --vocab "${ROOT}/models/vocab.txt" \
    --encoder "${ROOT}/models/encoder/model.onnx" \
    --reranker "${ROOT}/models/reranker/model.onnx" \
    --query "${QUERY}" --top-k "${TOP_K}" --stats
