#!/usr/bin/env bash
# Reproduces every number quoted in docs/BENCHMARKS.md.
#
#   bash scripts/bench_all.sh
#
# The kernel benchmark runs on synthetic data and needs no artifacts at all, so it works
# straight after a build. The end-to-end and quality benchmarks need models and a corpus.

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
# Host memory for the synthetic corpus is BENCH_DOCS * BENCH_DIM * 4 bytes; raise it to push
# the scan harder if you have the RAM and VRAM.
BENCH_DOCS="${BENCH_DOCS:-500000}"
BENCH_DIM="${BENCH_DIM:-384}"
MAX_QUERIES="${MAX_QUERIES:-100}"

echo "=============================================================="
echo " 1. kernel microbenchmark (synthetic, no artifacts needed)"
echo "=============================================================="
"${ROOT}/build/apps/zcgvs_bench_kernels" \
    --docs "${BENCH_DOCS}" --dim "${BENCH_DIM}" --k 100 --markdown 1

if [ ! -f "${ROOT}/data/${DATASET}.bin" ]; then
    echo
    echo "==> Skipping the end-to-end benchmarks: no data/${DATASET}.bin."
    echo "    Run scripts/generate_artifacts.sh first."
    exit 0
fi

echo
echo "=============================================================="
echo " 2. end-to-end latency and throughput"
echo "=============================================================="
RESCORE_ARGS=()
if [ -f "${ROOT}/data/${DATASET}.binary.bin" ]; then
    RESCORE_ARGS=(--rescore-embeddings "${ROOT}/data/${DATASET}.bin" --rescore-candidates 512)
fi
LEXICAL_ARGS=()
if [ -f "${ROOT}/data/${DATASET}.lex" ]; then
    LEXICAL_ARGS=(--lexical "${ROOT}/data/${DATASET}.lex")
fi

python "${ROOT}/python/benchmarks/benchmark_e2e.py" \
    --embeddings "${ROOT}/data/${DATASET}.bin" \
    --metadata "${ROOT}/data/${DATASET}.meta" \
    --vocab "${ROOT}/models/vocab.txt" \
    --encoder "${ROOT}/models/encoder/model.onnx" \
    --reranker "${ROOT}/models/reranker/model.onnx" \
    "${LEXICAL_ARGS[@]+"${LEXICAL_ARGS[@]}"}"

echo
echo "=============================================================="
echo " 3. retrieval quality across the storage ladder"
echo "=============================================================="
CONFIGS=("${ROOT}/configs/${DATASET}.json")
for variant in int8 binary_rescore hybrid; do
    candidate="${ROOT}/configs/${DATASET}_${variant}.json"
    [ -f "${candidate}" ] && CONFIGS+=("${candidate}")
done

python "${ROOT}/python/benchmarks/eval_quality.py" \
    --config "${CONFIGS[@]}" \
    --dataset "${DATASET}" \
    --max-queries "${MAX_QUERIES}"

echo
echo "==> Paste the tables above into docs/BENCHMARKS.md."
