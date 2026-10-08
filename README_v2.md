# v2 실행 안내 — SynthStrip 1.8 선택적 후처리

v2는 기존 DA → 병변 분할 뒤에 SynthStrip 1.8 후처리를 추가한 버전이다.
**원본 MRI에서 brain mask를 생성하고, 최종 native 병변 mask와 AND하여 뇌 밖 예측을 제거한다.**
입력 영상과 DA/분할 입력은 skull stripping하지 않는다. v2 이미지에서도 후처리 기본값은 OFF이므로,
적용하려면 `--skull-strip on`을 명시해야 한다.

[버전 선택](README.md) · [v1 실행 안내](README_v1.md)

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

## 1. v2 Docker 이미지 준비

이미지 아카이브와 체크섬을 다음 위치에 둔다.

```text
docker/brainmetaseg-da-deploy_v2.tar.gz
docker/brainmetaseg-da-deploy_v2.tar.gz.sha256
```

호스트에서 실행한다.

```bash
cd /path/to/dist
bash docker/load.sh v2
```

이미지 이름은 `brainmetaseg-da-deploy:v2`다. SynthStrip script/weight와 추가 의존성이 포함되어 있다.
파일을 전달받은 환경에서는 인터넷 없이 로드할 수 있다.

직접 빌드할 경우에는 검증된 v1을 먼저 로드한다. v2 빌드는 공식 SynthStrip 이미지에 접근해야 한다.

```bash
bash docker/load.sh v1
bash docker/build.sh v2
```

v2는 기존 PyTorch/nnU-Net 환경을 유지하면서 공식 SynthStrip 1.8 구성요소를 추가한다.
빌드 실패 시의 v1 복구 절차는 아래에 정리했다.

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

## 2. NIfTI 입력으로 후처리까지 실행

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
  -w /bundle brainmetaseg-da-deploy:v2 \
  bash /bundle/run.sh \
    --input-format nifti \
    --input-root /input --csv /input.csv \
    --output /results/v2_run_001 \
    --device cuda \
    --skull-strip on --synthstrip-device cpu
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

`--device cuda`는 DA와 병변 분할에 적용되고, `--synthstrip-device cpu`는 후처리에 적용된다.
SynthStrip은 기본 CPU로 별도 프로세스에서 실행하므로 기존 모델의 GPU 상주 메모리와 경쟁하지 않는다.
GPU 후처리는 `--synthstrip-device cuda`로 선택하지만, 모델 동시 상주 상태의 추가 VRAM은 아직 검증하지 않았다.
CPU만 사용할 때는 `--device cpu --synthstrip-device cpu`를 사용한다.

## 3. 실행 옵션

### 후처리 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--skull-strip on\|off` | `off` | `on`이면 native 병변 mask와 SynthStrip brain mask를 AND한다. `off`이면 기존 v1의 분할 동작이다. |
| `--synthstrip-device cpu\|cuda` | `cpu` | SynthStrip 실행 장치. `--device`와 별개이며 후처리 ON일 때만 사용한다. |
| `--synthstrip-home PATH` | 환경변수 `BMS_SYNTHSTRIP_HOME`, 없으면 `/opt/synthstrip/1.8` | SynthStrip script/weight 위치. v2 이미지에 설정되어 있으므로 일반 실행에서는 생략한다. 공식 파일 해시 검사를 하며 임의 모델 교체 옵션은 아니다. |

SynthStrip은 기본 border=1mm, `--no-csf` 미사용으로 실행한다. 이 조건을 바꾸는 CLI 옵션은 제공하지 않는다.
Brain mask dilation, 병변 연결요소 제거, 작은 병변 제거를 추가로 수행하지 않는다.
후처리가 OFF이면 SynthStrip 관련 장치/경로 옵션은 사용하지 않는다.

### 공통 옵션

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

도움말은 번들이 mount된 컨테이너 안에서 확인한다.

```bash
bash /bundle/run.sh --help
```

`--validate-inputs-only`는 SynthStrip 설치/추론까지 검사하는 옵션은 아니다.
후처리 ON 상태의 `nifti-pairs`는 입력 검사 모드에서도 허용하지 않는다.
DA seed, sampling, 모델 및 전처리 조건은 후처리 추가로 변경되지 않았다.

## 4. 결과와 debug 파일

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

후처리 ON이면 최상위 `<output_id>.nii.gz`는 **후처리 완료된 binary 병변 mask**다.
후처리 OFF이면 v1과 같은 미필터링 병변 mask다. Metadata의 `pipeline_version`은 ON이면 `v2`, OFF이면 `v1`이다.

후처리 ON에서 추가되는 기록:

```text
<output_id>/logs/
  postprocessing.json                  # 버전, 장치, brain/병변 voxel 수, 시간, 파일 해시
  synthstrip.log                       # SynthStrip 표준 출력과 오류
  debug/                               # --debug를 지정한 경우
    brain_mask.nii.gz                  # 원본 MRI로 생성한 brain mask
    native_mask_before_skullstrip.nii.gz
    native_probability_before_skullstrip.nii.gz
    native_mask.nii.gz                 # 필터링된 mask, 최상위 최종 mask와 동일
    native_probability.nii.gz          # brain mask 밖을 0으로 만든 probability
```

Pseudo/model-space debug mask는 병변 분할의 원래 결과를 유지하며 native-space에서만 후처리한다.
Debug OFF이면 임시 brain mask와 입력 파일을 정리하고 최종 native mask 및 텍스트/JSON 로그를 남긴다.
출력은 사용자가 지정한 `--output`으로 저장하며 연구용 NAS 경로가 코드에 고정되어 있지 않다.

Brain/병변/원본의 shape·geometry가 맞지 않거나 brain mask가 비어 있으면 오류다.
조용히 보간하거나 후처리를 생략하지 않는다. 후처리에 실패하면 해당 사례는 failed로 기록하고,
최종 출력 위치에 미처리 mask를 대신 저장하지 않는다. 진단용 미처리 mask는
`logs/native_mask_pending_postprocess.nii.gz`에 남을 수 있다.
`errors.json`, `logs/result.json`, `logs/synthstrip.log`를 함께 확인한다.

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

## 5. v2를 사용할 수 없을 때 v1으로 실행

1. 호스트에서 `bash docker/load.sh v1`을 실행한다.
2. [v1 실행 예시](README_v1.md#2-nifti-입력으로-실행)대로 이미지 `brainmetaseg-da-deploy:v1`과
   `bash /bundle/run_v1.sh`를 사용한다.
3. 기존에 시도한 v2 출력 폴더와 다른 새 `--output` 경로를 지정한다.

이 경로는 **SynthStrip 없이 기존 병변 분할을 실행**한다. v1 이미지에서 `--skull-strip on`을
요청하면 SynthStrip 부재를 오류로 알리며 자동 OFF 전환은 하지 않는다.
이미 v2 이미지가 정상 동작하고 후처리만 끄고 싶다면 `run.sh ... --skull-strip off`를 사용한다.

## 6. 현재 검증 범위

- v1/v2에서 단위 테스트 각각 28개 통과.
- 기존 병변 예측을 활용한 실제 6건의 brain extraction/후처리 검증 완료.
- 저장된 brain mask와 달랐던 3건은 공식 SynthStrip 1.8 재실행 출력과 v2 출력이 완전히 일치.
- 기존 reference 중 한 사례는 T1CE·BB brain mask 합집합이다. v2는 단일 입력 MRI에서 생성한
  brain mask를 사용하므로 이 reference와 동일한 동작으로 간주하지 않는다.
- 전체 78개 DA→병변 분할→후처리 통합 추론, 실제 DICOM의 SynthStrip 검증,
  GPU 동시 상주 VRAM 검증은 아직 수행하지 않았다.

수치, reference 비교의 제한과 이미지 복원 검증은 [HANDOFF.md](HANDOFF.md)에 기록되어 있다.
