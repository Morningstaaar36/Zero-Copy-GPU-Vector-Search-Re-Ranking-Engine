#!/usr/bin/env bash
# Fetches every external dependency the project needs and nothing else: the ONNX Runtime C++
# SDK, the Python packages, the two model checkpoints and the BEIR corpus. Nothing is
# trained; every artifact is a download or a deterministic transform of one.
#
#   bash scripts/download_all.sh
#
# Overrides:
#   ONNXRUNTIME_VERSION=1.19.2   pick a different ONNX Runtime release
#   DATASET=nfcorpus             embed a different BEIR subset
#   CORPUS_LIMIT=500             embed only the first N documents (quick smoke run)
#   EMBED_DEVICE=cpu             embed without a GPU
#   SKIP_ARTIFACTS=1             download dependencies only, skip models and corpus

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> [1/3] ONNX Runtime C++ SDK"
bash "${ROOT}/scripts/download_onnxruntime.sh"

echo
echo "==> [2/3] Python environment"
bash "${ROOT}/scripts/setup_python.sh"

if [ "${SKIP_ARTIFACTS:-0}" = "1" ]; then
    echo
    echo "==> Skipping model and corpus artifacts (SKIP_ARTIFACTS=1)"
    exit 0
fi

echo
echo "==> [3/3] Models and corpus artifacts"
bash "${ROOT}/scripts/generate_artifacts.sh"

echo
echo "==> Everything needed to run is on disk:"
echo "    third_party/onnxruntime   ONNX Runtime C++ SDK"
echo "    .venv                     Python environment"
echo "    models/                   exported ONNX encoder + reranker + vocab.txt"
echo "    data/                     corpus matrices, metadata, lexical index"
echo
echo "    Next: bash scripts/build.sh && bash scripts/verify.sh"
