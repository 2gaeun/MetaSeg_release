#!/usr/bin/env bash
# Portable entrypoint; use inside the bundled Docker/Python environment.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export nnUNet_compile=false
exec "${PYTHON:-python}" -m bms_deploy.cli \
  --weights "${ROOT}/weights/segmentation_131329" --da-weights "${ROOT}/weights" "$@"
