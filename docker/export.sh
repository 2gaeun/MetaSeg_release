#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VERSION=${1:-v2}
case "$VERSION" in v1|v2) ;; *) echo 'Usage: bash docker/export.sh [v1|v2]' >&2; exit 2 ;; esac
IMAGE_REF=${IMAGE_REF:-brainmetaseg-da-deploy:$VERSION}
ARCHIVE=${ARCHIVE:-$ROOT/docker/brainmetaseg-da-deploy_${VERSION}.tar.gz}
docker image inspect "$IMAGE_REF" >/dev/null
[[ ! -e "$ARCHIVE" && ! -e "$ARCHIVE.partial" ]] || { echo "Archive already exists: $ARCHIVE" >&2; exit 1; }
docker save "$IMAGE_REF" | gzip -1c > "$ARCHIVE.partial"
gzip -t "$ARCHIVE.partial"
mv -- "$ARCHIVE.partial" "$ARCHIVE"
(cd -- "$(dirname -- "$ARCHIVE")" && sha256sum "$(basename -- "$ARCHIVE")") > "$ARCHIVE.sha256"
echo "Image exported: $ARCHIVE"
