#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec docker build -f "$ROOT/docker/Dockerfile" -t "${METASEG_IMAGE:-metaseg:2.0}" "$ROOT"
