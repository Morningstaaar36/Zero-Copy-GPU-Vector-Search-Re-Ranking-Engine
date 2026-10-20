#!/usr/bin/env bash
# Fetches the Meta Llama 3.2 Instruct checkpoints used by the reasoning layer, plus the
# SciFact claim-verification labels used to evaluate it.
#
#   bash scripts/download_llama.sh
#
# The official meta-llama/* repos are gated: accept the license on the model page and run
# `hf auth login` once. Without access, the byte-identical unsloth/* mirror of the same
# release is used instead (redistribution is permitted by the Llama 3.2 Community License).
#
# Overrides:
#   LLAMA_MODELS="Llama-3.2-3B-Instruct Llama-3.2-1B-Instruct"   which checkpoints to fetch

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "${ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${ROOT}/.venv/bin/activate"
fi

MODELS="${LLAMA_MODELS:-Llama-3.2-3B-Instruct Llama-3.2-1B-Instruct}"

for model in ${MODELS}; do
    target="${ROOT}/models/llama/${model}"
    echo "==> ${model} -> ${target}"
    if hf download "meta-llama/${model}" --include "*.json" "*.safetensors" \
        --local-dir "${target}" >/dev/null 2>&1; then
        echo "    from meta-llama/${model}"
    else
        echo "    no access to meta-llama/${model}; using the unsloth/${model} mirror"
        hf download "unsloth/${model}" --include "*.json" "*.safetensors" \
            --local-dir "${target}" >/dev/null
    fi
done

echo "==> SciFact claim-verification labels"
PYTHONPATH="${ROOT}/python" python -c \
    "from zcgvs_tools import scifact_claims; print('   ', scifact_claims.ensure_release())"
