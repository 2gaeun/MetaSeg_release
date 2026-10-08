# v1 실행 안내 — 기존 병변 분할 파이프라인

v1은 입력 MRI에서 반대 시퀀스의 synthetic을 생성하고, 두 채널 분할 후 원본 geometry의
병변 mask를 출력한다. **SynthStrip 후처리를 적용하지 않는다.**
현재 번들에서는 `run_v1.sh`를 사용하며, v2를 사용할 수 없을 때의 복구 실행 경로이기도 하다.

[버전 선택](README.md) · [v2 실행 안내](README_v2.md)

## 실행에 필요한 파일과 환경

이 문서는 `v2-synthstrip` 브랜치의 배포 번들을 기준으로 한다. Linux amd64용 Docker 환경과,
GPU 실행 시 NVIDIA 드라이버 및 Docker의 GPU 접근 설정이 필요하다.
Slurm에서는 별도 할당받은 job 안에서 실행하고, 할당된 GPU만 컨테이너에 노출한다.

Git 저장소에는 소스와 설정 파일이 들어 있다. **Git clone만으로는 추론할 수 없다.**
모델 바이너리와 Docker 이미지 아카이브는 별도로 전달받아 아래 위치에 두어야 한다.
샘플 데이터와 과거 `outputs`는 실행에 필요하지 않다.

```text
dist/
  run.sh, run_v1.sh
  src/
  docker/
  weights/
    t1ce2bb_112788_step0175000/    # T1CE → BB DA 모델
    bb2t1ce_114599_step0150000/    # BB → T1CE DA 모델
    segmentation_131329/         # checkpoint_epoch_0950.pth, plans.json, dataset.json 등
```

전달받은 `weights`의 디렉터리 구조와 부속 파일을 유지한다.
Segmentation은 Dataset510 / 131329 Focal 모델의 `checkpoint_epoch_0950.pth`를 사용한다.
모델 weight는 Docker 이미지 안에 들어 있지 않으며 `/bundle/weights`에서 읽는다.
`run.sh`가 번들의 소스와 weight를 지정하므로, 반드시 번들을 mount하고 해당 스크립트로 실행한다.

## 1. v1 Docker 이미지 준비

이미지 아카이브와 체크섬을 다음 위치에 둔다.

```text
docker/brainmetaseg-da-deploy_v1.tar.gz
docker/brainmetaseg-da-deploy_v1.tar.gz.sha256
```

호스트에서 실행한다.

```bash
cd /path/to/dist
bash docker/load.sh v1
```

로드되는 이미지 이름은 `brainmetaseg-da-deploy:v1`이다.
이 아카이브는 기존 `brainmetaseg-da-deploy:0.2.0`과 동일한 이미지 ID를 가진 백업이다.
파일을 전달받은 환경에서는 인터넷 없이 로드할 수 있다.

아카이브 없이 환경을 새로 만들 경우에는 아래 명령을 사용한다. 인터넷 접근이 필요하며,
기존 환경의 정확한 복원이 목적이면 새 빌드 대신 위 아카이브를 로드한다.

```bash
bash docker/build.sh v1
```

### Docker 보조 스크립트

| 명령 | 역할 |
|---|---|
| `bash docker/load.sh v1` 또는 `v2` | 아카이브 체크섬을 검사하고 이미지를 로드한다. |
| `bash docker/build.sh v1` 또는 `v2` | 해당 버전의 Dockerfile로 이미지를 새로 빌드한다. |
| `bash docker/export.sh v1` 또는 `v2` | 로컬 이미지를 tar.gz와 `.sha256`으로 저장한다. 기존 아카이브는 덮어쓰지 않는다. |

세 스크립트 모두 버전을 생략하면 `v2`를 선택하므로, v1 사용 시에는 `v1`을 명시한다.
필요하면 환경변수 `IMAGE_REF`로 이미지 이름을 바꿀 수 있다.
Load/export의 아카이브 위치는 `ARCHIVE`로 바꾼다. 기본 위치는
`docker/brainmetaseg-da-deploy_<버전>.tar.gz`다.

## 입력 영상과 CSV

`--input-root`는 **폴더 경로**다. CSV는 첫 컬럼에 입력 ID, 둘째 컬럼에 `T1CE` 또는 `BB`를 적는다.
시퀀스의 대소문자는 구분하지 않는다. Header는 생략할 수 있으며, 사용할 때는 아래 예시를 따른다.
한 번의 실행에서 DICOM과 NIfTI를 섞지 않는다.

### NIfTI 입력

```text
input/
  case001.nii.gz
  case002.nii
```

```csv
file_id,sequence
case001,T1CE
case002,BB
```

- `--input-format nifti`를 사용한다.
- CSV에는 `.nii` 또는 `.nii.gz`를 제외한 file ID만 적는다. 폴더 경로를 적지 않는다.
- 입력 폴더 하위도 검색한다. 같은 basename의 파일이 여러 개면 오류다.
  예를 들어 `case001.nii`와 `case001.nii.gz`가 함께 있으면 구분할 수 없다.

### DICOM 입력

```text
input/
  series001/
    slice001.dcm
    slice002.dcm
    ...
  series002/
    ...
```

```csv
folder,sequence
series001,T1CE
series002,BB
```

- `--input-format dicom`을 사용한다. `dcm`은 옵션 값으로 지원하지 않는다.
- CSV 첫 컬럼은 `--input-root` 기준 series 폴더명 또는 상대 경로다.
- 각 CSV 폴더에는 하나의 SeriesInstanceUID만 있어야 한다.
- Classic single-frame MR을 지원한다. Multiframe, 여러 echo/timepoint 혼합,
  비균일 slice 간격 등 geometry가 불명확한 입력은 오류로 처리한다.

입력 영상은 미리 skull stripping하지 않는다. 반대 시퀀스의 실제 영상이나 GT는 필요하지 않다.
DA가 반대 시퀀스의 synthetic을 생성하고, 분할 입력 채널은 항상 ch0=T1CE-like / ch1=BB-like다.

## 2. NIfTI 입력으로 실행

```bash
# 호스트에서 실행한다. 아래 경로와 GPU UUID는 실제 환경에 맞게 수정한다.
cd /path/to/dist
BUNDLE_DIR="$(pwd)"
INPUT_DIR="/absolute/path/to/input"
CSV_PATH="/absolute/path/to/input.csv"
OUTPUT_DIR="/absolute/path/to/results"
GPU_DEVICE="GPU-여기에-사용할-GPU-UUID"
mkdir -p "$OUTPUT_DIR"

# 사용 가능한 GPU UUID는 nvidia-smi -L로 확인한다.
# Slurm에서는 이 job에 할당된 GPU의 UUID만 사용한다.
docker run --rm --network none \
  --gpus "device=${GPU_DEVICE}" \
  --mount "type=bind,src=${BUNDLE_DIR},dst=/bundle,readonly" \
  --mount "type=bind,src=${INPUT_DIR},dst=/input,readonly" \
  --mount "type=bind,src=${CSV_PATH},dst=/input.csv,readonly" \
  --mount "type=bind,src=${OUTPUT_DIR},dst=/results" \
  -w /bundle brainmetaseg-da-deploy:v1 \
  bash /bundle/run_v1.sh \
    --input-format nifti \
    --input-root /input --csv /input.csv \
    --output /results/v1_run_001 \
    --device cuda
```

`INPUT_DIR`, `CSV_PATH`는 실제로 존재해야 하며, mount에는 필요한 폴더/파일만 지정한다.
`OUTPUT_DIR`는 생성해도 되지만, 그 아래 실행 폴더는 프로그램이 만들므로 미리 만들지 않는다.
다시 실행할 때는 `--output`의 실행 폴더명을 바꾼다. 기존 결과를 덮어쓰거나 자동 이어서 실행하지 않는다.

- **DICOM 실행:** 위 명령의 `--input-format nifti`를 `dicom`으로 바꾸고 DICOM용 폴더와 CSV를 지정한다.
- **Debug 실행:** 명령 마지막에 `--debug`를 추가한다.
- **입력만 검사:** `--validate-inputs-only`를 추가하고 새 `--output` 경로를 사용한다.
  모델 추론을 하지 않으므로 Docker의 `--gpus`는 생략할 수 있다.
- **CPU 추론:** Docker의 `--gpus`를 제거하고 `--device cpu`로 바꾼다.
  전체 DA/병변 분할의 CPU 처리속도는 이번 검증에서 측정하지 않았다.

`run_v1.sh`는 항상 후처리를 끈다. `--skull-strip`을 직접 전달하면 오류다.
동일한 OFF 경로를 `run.sh ... --skull-strip off`로 실행할 수도 있다.

## 3. 실행 옵션

아래 기본값은 번들의 `run.sh`/`run_v1.sh`로 실행할 때의 값이다.

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--input-root PATH` | 필수 | 컨테이너에서 보이는 입력 폴더. 단일 파일 경로가 아니다. |
| `--csv PATH` | 필수 | 위의 두 컬럼 입력 목록 CSV. |
| `--input-format FORMAT` | `dicom` | `dicom`, `nifti`, 검증용 `nifti-pairs`. NIfTI라면 명시해야 한다. |
| `--output PATH` | 필수 | 실행 결과를 저장할 **새 폴더**. 이미 존재하면 오류다. |
| `--device cuda\|cpu` | `cuda` | DA와 병변 분할의 실행 장치. CUDA가 없다고 CPU로 자동 전환하지 않는다. |
| `--threads N` | `2` | PyTorch CPU thread 수. v2 후처리 ON이면 SynthStrip에도 전달한다. 1 이상이어야 한다. |
| `--debug` | OFF | 지정하면 중간 영상, probability, geometry 기록을 추가 저장한다. 끌 때는 옵션을 생략한다. |
| `--da-batch-size N` | `8` | DA slice 추론 batch 크기(1 이상). 분할 모델의 patch 크기와는 다른 옵션이다. |
| `--da-nfe N` | `50` | DA sampling step 수(1~999). 변경하면 synthetic과 최종 예측이 달라질 수 있다. |
| `--seed 0` | `0` | DA seed. 현재 배포 코드는 0만 허용하며 볼륨마다 초기화한다. |
| `--validate-inputs-only` | OFF | 입력 목록과 영상 로딩을 확인한다. 모델 추론이나 마스크 생성은 하지 않는다. 결과 기록용 `--output`은 필요하다. |
| `--weights PATH` | 번들의 `weights/segmentation_131329` | 분할 weight와 plans 등이 있는 폴더. `run.sh`가 자동 전달하므로 일반 실행에서는 생략한다. |
| `--da-weights PATH` | 번들의 `weights` | 두 방향 DA weight의 상위 폴더. 일반 실행에서는 생략한다. |
| `--synthetic-root PATH` | 미사용 | 저장된 **pseudo-space** synthetic을 재사용하는 고급 검증 옵션. 아래 설명 참고. |
| `-h`, `--help` | — | CLI 도움말을 출력한다. 입력 경로 인자 없이 사용할 수 있다. |

`--skull-strip on|off`, `--synthstrip-device cpu|cuda`, `--synthstrip-home PATH`는
`run.sh`에 있는 v2 관련 옵션이다. `run_v1.sh`에서는 후처리가 강제로 OFF이며,
SynthStrip 장치/경로 설정도 사용하지 않는다. 자세한 설명은 [v2 옵션](README_v2.md#3-실행-옵션)을 참고한다.

도움말은 번들이 mount된 컨테이너 안에서 다음처럼 확인한다.

```bash
bash /bundle/run_v1.sh --help
```

## 4. 결과 확인

`--output /results/RUN_NAME`을 지정하면 호스트의 `${OUTPUT_DIR}/RUN_NAME`에 저장된다.
현재 CLI의 `<output_id>`는 입력 ID를 파일명에 맞게 정리한 문자열과 10자리 해시의 조합이다.
따라서 CSV의 `case001`이 그대로 `case001.nii.gz`가 되지는 않는다.
입력 ID와 출력 ID의 대응은 `manifest.json`의 `file_id` 또는 `folder`, `id`로 확인한다.

```text
RUN_NAME/
  <output_id>.nii.gz             # 최종 native-space binary 병변 mask
  manifest.json                 # 실행 상태, 입력 ID ↔ 출력 ID, 결과 요약
  manifest.partial.json         # 실행 중 누적 기록
  errors.json                   # 사례 오류가 발생한 경우
  <output_id>/logs/
    result.json
    da_result.json              # DA를 실제 실행한 경우
    transform.json
    crop.json
    debug/                      # --debug를 지정한 경우
      real_pseudo.nii.gz, synthetic_pseudo.nii.gz
      da_input_network.nii.gz, da_output_network.nii.gz, da_geometry.json
      seg_input_0000.nii.gz, seg_input_0001.nii.gz
      model_input_0000.nii.gz, model_input_0001.nii.gz
      model_mask.nii.gz, model_probability.nii.gz
      pseudo_mask.nii.gz, pseudo_probability.nii.gz
      native_mask.nii.gz, native_probability.nii.gz, geometry.json
```

DA debug 파일은 DA를 실행한 경우에 생성된다. `model_*`는 nnU-Net 내부 전처리 grid,
`pseudo_*`는 crop을 복원한 전체 pseudo grid, `native_*`는 원본 영상의 grid다.
Debug OFF여도 JSON 로그는 남으며, 영상 출력은 최종 native mask다.

최종 mask는 후처리 전 병변 예측이며, 원본 영상과 shape·geometry가 일치한다.
기본 foreground probability 기준은 `>= 0.5`다. 별도의 skull stripping이나 크기 필터를 적용하지 않는다.

## 저장된 synthetic/pair를 사용하는 검증 옵션

일반적인 신규 영상 추론에서는 아래 옵션을 사용하지 않는다.

- `--synthetic-root PATH`: 원본 NIfTI/DICOM에서 real pseudo 영상과 transform을 만들고,
  지정 폴더의 `<output_id>.nii.gz`를 synthetic으로 읽는다. DA 추론을 생략한다.
  Synthetic은 Dataset510의 전체 pseudo grid여야 하며 shape와 geometry가 맞아야 한다.
  일반 native-space synthetic을 넣는 옵션이 아니다. 외부 폴더라면 Docker mount도 추가해야 한다.
- `--input-format nifti-pairs`: 각 CSV 폴더에 `real.nii.gz`, `synthetic.nii.gz`, `transform.json`을 둔다.
  CSV header는 DICOM 예시와 같은 `folder,sequence`를 사용한다.
  두 영상은 전체 pseudo grid이며 transform schema는 `i2sb-pseudo-1mm-transform-v1`이어야 한다.
  Dataset505 native pair나 Dataset506 physical-space transform은 사용할 수 없다.
  이 모드에는 원본 native MRI가 없으므로 **SynthStrip 후처리 ON은 허용하지 않는다.**

Dataset510 pseudo grid를 실제 physical 1mm 영상으로 해석하지 않는다.
전처리·복원 규칙과 모델 출처는 [HANDOFF.md](HANDOFF.md)를 참고한다.

## 오류 확인과 v1 백업

- `--output`이 이미 존재하면 새로운 실행 폴더명을 지정한다.
- NIfTI ID 중복/누락, CSV 컬럼 또는 시퀀스 오류는 입력 파일과 CSV를 수정한다.
- Geometry 불일치는 자동 보간하지 않고 오류를 낸다. 입력이나 transform을 확인한다.
- 개별 사례 실패는 `errors.json`에 기록되며 전체 실행은 종료 코드 1을 반환한다.
  실패 전에 완료된 사례 결과가 남을 수 있으므로 `manifest.json`과 함께 확인한다.
- v2 빌드가 실패해도 v1 아카이브를 로드한 뒤 이 문서의 명령으로 실행할 수 있다.
- 후처리 추가 전 소스 스냅샷은 전달 번들의 `backups/v1/source.tar.gz`에 있다.
  이 소스 스냅샷에는 모델 바이너리가 포함되지 않으며 Git 저장소에도 업로드하지 않는다.
