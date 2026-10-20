#!/usr/bin/env bash
# Runs every check the project has, in dependency order, and prints a pass/fail summary.
#
# Tiers, so a partial setup still verifies what it can:
#   1. formats   pure Python, no GPU, no build           (always runs)
#   2. kernels   ctest CUDA harnesses                    (needs a build)
#   3. engine    end-to-end query through the bindings   (needs models + corpus)
#   4. variants  quantized / rescore / hybrid configs    (needs the derived artifacts)
#
# Skip a tier explicitly with SKIP_KERNELS=1, SKIP_ENGINE=1 or SKIP_VARIANTS=1.

set -uo pipefail

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

PASSED=0
FAILED=0
SKIPPED=0
SUMMARY=()

record() {
    local status="$1"
    local name="$2"
    case "${status}" in
        pass) PASSED=$((PASSED + 1)); SUMMARY+=("  PASS     ${name}") ;;
        fail) FAILED=$((FAILED + 1)); SUMMARY+=("  FAIL     ${name}") ;;
        skip) SKIPPED=$((SKIPPED + 1)); SUMMARY+=("  SKIPPED  ${name}") ;;
    esac
}

run_step() {
    local name="$1"
    shift
    echo
    echo "── ${name} ──────────────────────────────────────────"
    if "$@"; then
        record pass "${name}"
    else
        record fail "${name}"
    fi
}

echo "=============================================================="
echo " zcgvs verification"
echo "=============================================================="

# ---- tier 1: wire formats (no GPU, no build) ----------------------------------------
run_step "formats round-trip" python "${ROOT}/tests/python/test_formats.py"
run_step "llama logic (calibration, fusion)" python "${ROOT}/tests/python/test_factcheck.py"

# ---- tier 2: CUDA kernels ------------------------------------------------------------
if [ "${SKIP_KERNELS:-0}" = "1" ]; then
    record skip "CUDA kernel tests (SKIP_KERNELS=1)"
elif [ ! -d "${ROOT}/build" ]; then
    record skip "CUDA kernel tests (no build/ — run scripts/build.sh)"
else
    run_step "CUDA kernel tests" ctest --test-dir "${ROOT}/build" --output-on-failure
fi

# ---- tier 3: engine end to end -------------------------------------------------------
ARTIFACTS_READY=1
for required in "${ROOT}/data/${DATASET}.bin" "${ROOT}/data/${DATASET}.meta" \
                "${ROOT}/models/vocab.txt" "${ROOT}/models/encoder/model.onnx"; do
    [ -f "${required}" ] || ARTIFACTS_READY=0
done

if [ "${SKIP_ENGINE:-0}" = "1" ]; then
    record skip "end-to-end query (SKIP_ENGINE=1)"
elif [ "${ARTIFACTS_READY}" -ne 1 ]; then
    record skip "end-to-end query (missing artifacts — run scripts/generate_artifacts.sh)"
elif [ ! -d "${ROOT}/build/bindings" ]; then
    record skip "end-to-end query (no python bindings — run scripts/build.sh)"
else
    run_step "end-to-end query" python "${ROOT}/python/examples/quickstart.py" \
        --embeddings "${ROOT}/data/${DATASET}.bin" \
        --metadata "${ROOT}/data/${DATASET}.meta" \
        --vocab "${ROOT}/models/vocab.txt" \
        --encoder "${ROOT}/models/encoder/model.onnx" \
        --reranker "${ROOT}/models/reranker/model.onnx" \
        --query "${QUERY}"
fi

# ---- tier 4: the optional stages ------------------------------------------------------
verify_config() {
    local label="$1"
    local artifact="$2"
    local config="$3"
    shift 3
    if [ ! -f "${artifact}" ]; then
        record skip "${label} (no ${artifact} — run scripts/generate_artifacts.sh)"
        return
    fi
    if [ ! -f "${config}" ]; then
        record skip "${label} (no ${config})"
        return
    fi
    run_step "${label}" python "${ROOT}/python/examples/quickstart.py" \
        --config "${config}" --query "${QUERY}" "$@"
}

if [ "${SKIP_VARIANTS:-0}" = "1" ] || [ "${ARTIFACTS_READY}" -ne 1 ]; then
    record skip "storage / rescore / hybrid variants"
else
    verify_config "int8 storage" \
        "${ROOT}/data/${DATASET}.int8.bin" "${ROOT}/configs/${DATASET}_int8.json"
    verify_config "binary + on-device rescore" \
        "${ROOT}/data/${DATASET}.binary.bin" "${ROOT}/configs/${DATASET}_binary_rescore.json"
    verify_config "hybrid BM25 + dense" \
        "${ROOT}/data/${DATASET}.lex" "${ROOT}/configs/${DATASET}_hybrid.json" --hybrid
fi

echo
echo "=============================================================="
echo " summary"
echo "=============================================================="
for line in "${SUMMARY[@]}"; do
    echo "${line}"
done
echo
echo "  ${PASSED} passed, ${FAILED} failed, ${SKIPPED} skipped"

[ "${FAILED}" -eq 0 ]
