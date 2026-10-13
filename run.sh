#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${METASEG_IN_CONTAINER:-0}" == 1 ]]; then
 export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
 export nnUNet_compile=false
 if [[ $# -eq 0 ]]; then exec python -m bms_deploy.launcher --bundle "$ROOT"; fi
 exec python -m bms_deploy.cli --weights "$ROOT/weights/segmentation_131329" --da-weights "$ROOT/weights" "$@"
fi
if [[ $# -ne 0 ]]; then
 echo '일반 실행: input/series.csv와 영상을 준비한 뒤 bash run.sh 를 실행하세요.'
 exit 2
fi
command -v docker >/dev/null || { echo 'Docker가 없습니다. README의 설치 순서를 확인하세요.' >&2; exit 1; }
[[ -f "$ROOT/input/series.csv" ]] || { echo 'input/series.csv가 없습니다.' >&2; exit 1; }
for file in weights/t1ce2bb_112788_step0175000/ema.pt weights/bb2t1ce_114599_step0150000/ema.pt weights/segmentation_131329/checkpoint_epoch_0950.pth; do
 [[ -f "$ROOT/$file" ]] || { echo "모델 파일이 없습니다: $file (담당자에게 받은 weights를 복사하세요.)" >&2; exit 1; }
done
IMAGE=${METASEG_IMAGE:-metaseg:2.0}
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo 'Docker 이미지가 준비되지 않았습니다. 먼저 bash setup.sh 를 실행하세요.' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'NVIDIA 드라이버를 확인하세요. nvidia-smi 명령이 필요합니다.' >&2; exit 1; }
selection=${METASEG_GPU:-${SLURM_JOB_GPUS:-0}}
selection=${selection%%,*}
uuid=$(nvidia-smi -i "$selection" --query-gpu=uuid --format=csv,noheader)
[[ "$uuid" =~ ^GPU-[[:xdigit:]-]+$ ]] || { echo '사용할 GPU를 확인할 수 없습니다.' >&2; exit 1; }
gpu_arg="device=$uuid"
user_args=(--user "$(id -u):$(id -g)")
# Rootless Podman already maps container root to the invoking host user.
podman_rootless=$(docker info --format '{{.Host.Security.Rootless}}' 2>/dev/null || true)
case "$podman_rootless" in
 true) gpu_arg=$uuid; user_args=() ;;
 false) gpu_arg=$uuid ;;
esac
mkdir -p "$ROOT/output"
# Host-owned output and a private temporary HOME; expose just one selected GPU.
exec docker run --rm --network none --gpus "$gpu_arg" --shm-size=8g \
 "${user_args[@]}" \
 --mount "type=bind,src=$ROOT,dst=/bundle,readonly" \
 --mount "type=bind,src=$ROOT/output,dst=/bundle/output" \
 -e METASEG_IN_CONTAINER=1 -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
 -e OMP_NUM_THREADS=4 -e nnUNet_compile=false -w /bundle "$IMAGE" bash /bundle/run.sh
