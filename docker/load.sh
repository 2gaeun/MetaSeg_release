#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE_REF=${IMAGE_REF:-brainmetaseg-da-deploy:0.2.0}
ARCHIVE=${ARCHIVE:-$ROOT/docker/brainmetaseg-da-deploy_0.2.0.tar.gz}
gzip -dc "$ARCHIVE" | docker load
# Podman-built archives can carry a localhost/ prefix; standard Docker needs an explicit alias.
if ! docker image inspect "$IMAGE_REF" >/dev/null 2>&1; then
  docker image inspect "localhost/$IMAGE_REF" >/dev/null
  docker tag "localhost/$IMAGE_REF" "$IMAGE_REF"
fi
echo "Image ready: $IMAGE_REF"
