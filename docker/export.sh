#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE_REF=${IMAGE_REF:-brainmetaseg-da-deploy:0.2.0}
ARCHIVE=${ARCHIVE:-$ROOT/docker/brainmetaseg-da-deploy_0.2.0.tar.gz}
docker image inspect "$IMAGE_REF" >/dev/null
[[ ! -e "$ARCHIVE" && ! -e "$ARCHIVE.partial" ]] || { echo "Archive already exists: $ARCHIVE" >&2; exit 1; }
docker save "$IMAGE_REF" | gzip -1c > "$ARCHIVE.partial"
gzip -t "$ARCHIVE.partial"
mv -- "$ARCHIVE.partial" "$ARCHIVE"
echo "Image exported: $ARCHIVE"
