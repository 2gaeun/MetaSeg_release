#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VERSION=${1:-v2}
case "$VERSION" in v1|v2) ;; *) echo 'Usage: bash docker/load.sh [v1|v2]' >&2; exit 2 ;; esac
IMAGE_REF=${IMAGE_REF:-brainmetaseg-da-deploy:$VERSION}
ARCHIVE=${ARCHIVE:-$ROOT/docker/brainmetaseg-da-deploy_${VERSION}.tar.gz}
if [[ -f "$ARCHIVE.sha256" ]]; then
  (cd -- "$(dirname -- "$ARCHIVE")" && sha256sum -c "$(basename -- "$ARCHIVE").sha256")
fi
gzip -dc "$ARCHIVE" | docker load
# Podman-built archives can carry a localhost/ prefix; standard Docker needs an explicit alias.
if ! docker image inspect "$IMAGE_REF" >/dev/null 2>&1; then
  docker image inspect "localhost/$IMAGE_REF" >/dev/null
  docker tag "localhost/$IMAGE_REF" "$IMAGE_REF"
fi
echo "Image ready: $IMAGE_REF"
