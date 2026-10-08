#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VERSION=${1:-v2}
case "$VERSION" in
  v1) FILE="$ROOT/docker/Dockerfile" ;;
  v2)
    FILE="$ROOT/docker/Dockerfile.v2"
    docker image inspect brainmetaseg-da-deploy:v1 >/dev/null || {
      echo 'Load the tested v1 backup first: bash docker/load.sh v1' >&2; exit 1;
    }
    ;;
  *) echo 'Usage: bash docker/build.sh [v1|v2]' >&2; exit 2 ;;
esac
if ! docker build -f "$FILE" -t "${IMAGE_REF:-brainmetaseg-da-deploy:$VERSION}" "$ROOT"; then
  echo 'Build failed. The v1 backup remains available; load v1 and use run_v1.sh. Postprocessing is not silently skipped.' >&2
  exit 1
fi
