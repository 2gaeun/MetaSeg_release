#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
docker build -f "$ROOT/docker/Dockerfile" -t "${IMAGE_REF:-brainmetaseg-da-deploy:0.2.0}" "$ROOT"
