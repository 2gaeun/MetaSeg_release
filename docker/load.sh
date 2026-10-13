#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ARCHIVE=${1:-$ROOT/docker/MetaSeg_image.tar.gz}
IMAGE=${METASEG_IMAGE:-metaseg:2.0}
[[ -f "$ARCHIVE" ]] || { echo "이미지 파일이 없습니다: $ARCHIVE" >&2; exit 1; }
if [[ -f "$ARCHIVE.sha256" ]]; then
 read -r expected _ < "$ARCHIVE.sha256"
 [[ "$expected" =~ ^[[:xdigit:]]{64}$ ]] || { echo '체크섬 파일 형식을 확인하세요.' >&2; exit 1; }
 actual=$(sha256sum "$ARCHIVE"); actual=${actual%% *}
 [[ "$actual" == "${expected,,}" ]] || { echo '이미지 파일이 손상되었거나 체크섬과 다릅니다.' >&2; exit 1; }
fi
gzip -dc "$ARCHIVE" | docker load
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
 for ref in "localhost/$IMAGE" brainmetaseg-da-deploy:v2 localhost/brainmetaseg-da-deploy:v2; do
  if docker image inspect "$ref" >/dev/null 2>&1; then docker tag "$ref" "$IMAGE"; break; fi
 done
fi
docker image inspect "$IMAGE" >/dev/null
echo '이미지 준비 완료'
