#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ONNX Runtime ships one GPU build per CUDA major version, and a build linked against CUDA 12
# will not load on a CUDA 13 toolkit. Pick the pairing from the installed nvcc rather than
# hardcoding one: releases from 1.24 onward carry the CUDA major in the package name.
CUDA_MAJOR="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p' | head -1)"

if [ "${CUDA_MAJOR:-12}" -ge 13 ] 2>/dev/null; then
    VERSION="${ONNXRUNTIME_VERSION:-1.28.0}"
    DEFAULT_PKG="onnxruntime-linux-x64-gpu_cuda13-${VERSION}"
else
    VERSION="${ONNXRUNTIME_VERSION:-1.20.1}"
    DEFAULT_PKG="onnxruntime-linux-x64-gpu-${VERSION}"
fi

PKG="${ONNXRUNTIME_PKG:-${DEFAULT_PKG}}"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${PKG}.tgz"
DEST="${ROOT}/third_party"

mkdir -p "${DEST}"

if [ ! -d "${DEST}/${PKG}" ]; then
    echo "==> Downloading ONNX Runtime ${VERSION}"
    curl -L --fail --retry 3 -o "${DEST}/${PKG}.tgz" "${URL}"
    echo "==> Extracting"
    tar -xzf "${DEST}/${PKG}.tgz" -C "${DEST}"
    rm -f "${DEST}/${PKG}.tgz"
else
    echo "==> ONNX Runtime ${VERSION} already present"
fi

ln -sfn "${DEST}/${PKG}" "${DEST}/onnxruntime"
echo "==> ONNX Runtime ready at ${DEST}/onnxruntime"
