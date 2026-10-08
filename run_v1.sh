#!/usr/bin/env bash
# Explicit fallback: needs only the original v1 runtime; no SynthStrip dependency.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for arg in "$@"; do
  if [[ "$arg" == --skull-strip || "$arg" == --skull-strip=* ]]; then
    echo 'run_v1.sh always disables postprocessing; use run.sh for --skull-strip.' >&2
    exit 2
  fi
done
exec bash "$ROOT/run.sh" "$@" --skull-strip off
