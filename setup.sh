#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
command -v docker >/dev/null || { echo '먼저 README의 Docker 설치 순서를 확인하세요.' >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo 'Docker에 연결할 수 없습니다. README의 실행 권한 설정과 Docker 시작 여부를 확인하세요.' >&2; exit 1; }
IMAGE=${METASEG_IMAGE:-metaseg:2.0}
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
 echo '실행 환경이 이미 준비되어 있습니다.'
elif [[ -f "$ROOT/docker/MetaSeg_image.tar.gz" ]]; then
 bash "$ROOT/docker/load.sh" "$ROOT/docker/MetaSeg_image.tar.gz"
elif [[ -f "$ROOT/docker/brainmetaseg-da-deploy_v2.tar.gz" ]]; then
 bash "$ROOT/docker/load.sh" "$ROOT/docker/brainmetaseg-da-deploy_v2.tar.gz"
else
 echo '실행 환경을 내려받고 준비합니다. 최초 실행에는 시간이 걸립니다.'
 bash "$ROOT/docker/build.sh"
fi
echo '환경 준비 완료. input 폴더에 영상과 series.csv를 준비한 뒤 bash run.sh 를 실행하세요.'
