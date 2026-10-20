#!/usr/bin/env bash
# Produces every offline artifact the engine loads: ONNX models, the WordPiece vocabulary,
# the float32 corpus matrix, its float16/int8/binary variants, and the BM25 inverted index.
#
# No training happens here. The models are downloaded checkpoints exported to ONNX; the
# quantized corpora are deterministic transforms of the float32 one.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "${ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${ROOT}/.venv/bin/activate"
fi

DATASET="${DATASET:-scifact}"
EMBED_DEVICE="${EMBED_DEVICE:-cuda}"
LIMIT="${CORPUS_LIMIT:-}"
SKIP_LEXICAL="${SKIP_LEXICAL:-0}"
SKIP_QUANTIZED="${SKIP_QUANTIZED:-0}"

cd "${ROOT}/python"

echo "==> Exporting the bi-encoder and cross-encoder to ONNX"
python -m zcgvs_tools.export_models --output-dir "${ROOT}/models"

echo
echo "==> Embedding BEIR/${DATASET} (device=${EMBED_DEVICE})"
if [ -n "${LIMIT}" ]; then
    python -m zcgvs_tools.embed_corpus --dataset "${DATASET}" \
        --output-dir "${ROOT}/data" --device "${EMBED_DEVICE}" --limit "${LIMIT}"
else
    python -m zcgvs_tools.embed_corpus --dataset "${DATASET}" \
        --output-dir "${ROOT}/data" --device "${EMBED_DEVICE}"
fi

if [ "${SKIP_QUANTIZED}" != "1" ]; then
    echo
    echo "==> Deriving float16 / int8 / binary corpora (no re-embedding)"
    python -m zcgvs_tools.quantize --input "${ROOT}/data/${DATASET}.bin" \
        --dtype float16 int8 binary
fi

if [ "${SKIP_LEXICAL}" != "1" ]; then
    echo
    echo "==> Building the BM25 inverted index"
    python -m zcgvs_tools.build_lexical \
        --metadata "${ROOT}/data/${DATASET}.meta" \
        --vocab "${ROOT}/models/vocab.txt" \
        --tokenizer-dir "${ROOT}/models/encoder" \
        --output "${ROOT}/data/${DATASET}.lex"
fi

echo
echo "==> Artifacts written to ${ROOT}/models and ${ROOT}/data"
ls -lh "${ROOT}/data" | sed 's/^/    /'
