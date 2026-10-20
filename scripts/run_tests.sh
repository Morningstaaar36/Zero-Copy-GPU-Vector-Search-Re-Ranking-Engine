#!/usr/bin/env bash
# Thin wrapper kept for compatibility: the full tiered check now lives in scripts/verify.sh.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec bash "${ROOT}/scripts/verify.sh" "$@"
