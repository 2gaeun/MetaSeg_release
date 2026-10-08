# BrainMetaSeg 배포 인수인계

현재 계약: 2026-10-08. **v1은 기존 파이프라인**, **v2는 선택적 SynthStrip 1.8 후처리**다.
기본값은 후처리 OFF다. 모든 입력 영상과 DA/병변 분할 입력은 skull stripping하지 않는다.
ON일 때만 원본 native 영상의 brain mask와 최종 native 병변 mask를 AND한다.
병변의 연결요소 제거, 크기 필터, brain mask dilation은 추가하지 않는다.

## v1 백업과 v2 실행

| 구분 | 이미지 | 오프라인 아카이브 |
|---|---|---|
| v1: 검증된 기존 환경 | `brainmetaseg-da-deploy:v1` | `docker/brainmetaseg-da-deploy_v1.tar.gz` |
| v2: SynthStrip 1.8 포함 | `brainmetaseg-da-deploy:v2` | `docker/brainmetaseg-da-deploy_v2.tar.gz` |

v1은 기존 `0.2.0` 이미지와 동일 ID
`41be76fc731d4b783029637a12aff46f8791d6b479e5ca18c84595123479612d`다.
기존 `brainmetaseg-da-deploy_0.2.0.tar.gz`도 보존했다. 각 새 아카이브의 `.sha256`을 함께 전달한다.
후처리 추가 전 코드(commit `fe01793`)는 `backups/v1/source.tar.gz`와 provenance에 보존했다.
Weights는 기존 `weights/`에 그대로 있으며 source archive에는 모델 바이너리가 포함되지 않는다.

```bash
# 그대로 사용할 이미지 로드: 실제 배포 환경에서 재빌드할 필요가 없다.
bash docker/load.sh v1
bash docker/load.sh v2

# v2를 직접 빌드해야 할 때: 검증된 v1을 먼저 로드한다.
bash docker/load.sh v1
bash docker/build.sh v2

# 각 이미지 안에서 dist를 /bundle에 mount한 뒤 실행한다.
bash /bundle/run.sh --input-format nifti --input-root /input --csv /input.csv \
  --output /output/v2 --skull-strip on --debug
bash /bundle/run.sh --input-format nifti --input-root /input --csv /input.csv \
  --output /output/v1 --skull-strip off
# v1 전용 실행 파일도 제공한다.
bash /bundle/run_v1.sh --input-format nifti --input-root /input --csv /input.csv \
  --output /output/v1_backup
```

v2 빌드가 실패하면 v1 이미지와 아카이브는 그대로 남는다. **v1을 로드해 run_v1.sh로 명시적으로 실행**한다.
빌드 실패나 SynthStrip 오류를 이유로 요청된 후처리를 조용히 생략하지 않는다.
v1 이미지에서도 현재 소스의 OFF 경로가 동작하며 SynthStrip 의존성이 필요 없다.
`docker/build.sh v1`은 Dockerfile로 환경을 새로 만드는 명령이므로, 정확한 백업 복원에는 `load.sh v1`을 쓴다.

`Dockerfile.v2`는 공식 `freesurfer/synthstrip:1.8` digest에서 원본 script, model-v1 weights와
Surfa/xxhash/importlib_resources를 복사한다. 기존 torch/numpy/nnU-Net은 업그레이드하지 않는다.
출처·버전·파일 해시는 `docker/synthstrip1.8.json`에 있다. 현재 빌드/검증 플랫폼은 Linux amd64, Python 3.10이다.

## v2 후처리 계약

- `--skull-strip on|off`: 기본 `off`. 결과 metadata의 pipeline_version은 ON이면 v2, OFF이면 v1이다.
- `--synthstrip-device cpu|cuda`: 기본 `cpu`. 별도 프로세스에서 실행하여 부모의 DA/분할 설정에 영향을 주지 않는다.
  CUDA는 선택할 수 있으나 기존 모델 상주 상태의 추가 VRAM 요구량은 아직 검증하지 않았다.
- NIfTI는 원본 파일을 그대로 SynthStrip에 전달한다. DICOM은 읽은 native 영상을 임시 NIfTI로 전달한다.
  DA normalization, synthetic, pseudo grid를 brain extraction 입력으로 사용하지 않는다.
- 기본 SynthStrip border=1mm, `--no-csf` 미사용. 기존 reference 생성 설정을 유지한다.
- 원본/brain/병변 mask의 shape·affine을 검사하고, 불일치 시 보간하거나 조정하지 않고 오류를 낸다.
- 최종 출력은 `(native lesion != 0) AND (native brain != 0)`의 uint8 binary mask다.
  후처리가 성공해야 최종 파일을 공개 경로에 저장한다. 실패 시 미처리 mask는 logs의 pending 파일로 남고,
  해당 case는 failed로 기록된다.
- pseudo/model-space mask는 기존 분할 결과를 유지한다. 필터링은 native-space에서만 적용한다.
- Debug ON에서는 기존 native mask/probability를 `*_before_skullstrip.nii.gz`로 보존하고,
  `brain_mask.nii.gz`, 필터링한 `native_mask.nii.gz`와 `native_probability.nii.gz`를 저장한다.
- Debug OFF에서는 성공한 최종 mask만 영상으로 남는다. `logs/postprocessing.json`, `logs/synthstrip.log`로
  버전·조건·foreground voxel 수·소요시간·오류 출처를 확인할 수 있다.
- 원본 native 영상이 없는 `nifti-pairs` 모드에는 ON을 허용하지 않는다.
- 출력 경로는 기존 `--output` 인자를 사용하며 NAS 검증 경로를 배포 코드에 고정하지 않는다.

후처리 구현 검증 결과와 범위는 문서 마지막의 v2 검증 항목을 참고한다.

## 모델과 독립 실행

- Segmentation: **Dataset510 / 학습 job 131329**.
- Dataset: `Dataset510_BrainMet_Synthetic_2ch_Pseudo1mm_RealChannelCrop_DHW112x128x160`.
- Trainer: `nnUNetTrainerBrainMetaFocalLR3e3Components`.
- 파일: `weights/segmentation_131329/checkpoint_epoch_0950.pth`.
  950은 sweep에서 선택한 **파일명 label**이다. 저장된 `current_epoch`는 기록만 하며
  그 값으로 다른 체크포인트를 선택하지 않는다. 지정 파일의 저장된 current_epoch는 **952**다.
- Dataset510의 `plans.json`, `dataset.json`을 그대로 번들에 복사했다. Dataset505 plans를 사용하지 않는다.
- 체크포인트 SHA256: `8aa7987162b9fd91e318fe49c232eb1205e36564d7208b0f6ab0de4a9d6c2d85`.
- `weights/segmentation_131329/provenance.json`에 복사한 파일의 해시를 기록했다.
- Focal trainer 계층은 `build_network_architecture`를 override하지 않는다.
  `bms_deploy/trainer.py`는 같은 nnU-Net 기본 network builder를 상속하는 **추론 전용 adapter**다.
  Focal loss·학습 logger는 추론에서 필요하지 않다. trainer 원본 및 loss/mixin은
  `weights/segmentation_131329/trainer_source/`에 출처 기록으로 포함했다.
- 모델 state dict는 strict load한다. Feature fusion이 아니며 채널을 `[real, synthetic]`으로 재배열하지 않는다.
- 연구용 BrainMetaSeg/I2SB 소스, GT, NAS 절대경로, 기존 test manifest는 운영 의존성이 아니다.
  `run.sh`가 자신의 위치에서 `src`와 `weights`를 찾는다. dist를 복사하여 사용할 수 있다.
  남아 있는 `weights/segmentation_123195`는 과거 모델이며 현재 실행 경로에서 사용하지 않는다.

## 실행과 입력

```bash
cd /path/to/dist
bash run.sh --input-format nifti --input-root /input --csv /input.csv \
  --output /output/new_run --debug
bash run.sh --input-format dicom --input-root /dicom_parent --csv /input.csv \
  --output /output/new_run
```

CSV는 2컬럼이며 둘째 컬럼은 `T1CE` 또는 `BB`(대소문자 정규화)다. Header는 선택사항이다.

- `dicom`(기본): 첫 컬럼은 input-root 아래 series 폴더명/상대 경로다.
  한 폴더는 정확히 하나의 SeriesInstanceUID를 가져야 한다. Classic single-frame MR만 지원한다.
  다중 series, 비균일 slice spacing, mixed echo/time, 불명확한 geometry, shear를 거부한다.
- `nifti`: 첫 컬럼은 확장자 없는 file ID다. `{ID}.nii` 또는 `{ID}.nii.gz`를 하위 폴더까지 검색한다.
  중복 basename/확장자, 누락, root 밖 경로는 오류다.
- `--validate-inputs-only`는 모델을 실행하지 않고 입력 계약만 검사한다.
- 기존 output 경로를 덮어쓰지 않는다. 개별 사례 실패는 `errors.json`에 기록하고 최종 exit 1을 반환한다.
- 기본 `--device cuda`, `--threads 2`, `--da-batch-size 8`, `--da-nfe 50`, `--seed 0`.
  seed는 0만 허용하고 **각 볼륨마다 초기화**한다. 이번 Dataset510 전환에서 DA 난수 설정을 바꾸지 않았다.
- Debug 기본 OFF. CPU는 `--device cpu`로 명시해야 하며 자동 fallback은 없다.

저장 pair 검증용 `--input-format nifti-pairs`는 `folder,sequence` CSV를 사용하며 각 폴더에
`real.nii.gz`, `synthetic.nii.gz`, `transform.json`을 요구한다. **두 영상 모두 전체 pseudo grid**여야 한다.
과거 Dataset505 native pair는 이 모드에 사용할 수 없다. Transform은 아래 전용 schema여야 한다.
`--synthetic-root`는 `<output_id>.nii.gz` 형태의 **pseudo synthetic**을 읽는다.
이때 native 입력에서 real pseudo와 transform을 직접 생성하고 synthetic과 geometry를 엄격히 비교한다.

## DA → pseudo grid

| 입력 | 반대 시퀀스 | DA checkpoint |
|---|---|---|
| T1CE | BB | 112788 / step 175000 / EMA |
| BB | T1CE | 114599 / step 150000 / EMA |

1. native 입력을 보간 없이 RAS로 재배열한다. Oblique direction을 유지한다.
2. RAS z spacing이 1mm와 `1e-4`보다 크게 다르면 z만 SITK linear로 resample한다.
   depth=`round(Nz*sz)`, XY size/spacing 및 origin/direction 유지, 바깥 값=0.
3. 전체 작업 볼륨의 0.1–99.9 percentile clipping → `[-1,1]` 정규화. 상수 볼륨은 0.
4. axial slice마다 torch bilinear(`align_corners=False`)로 256×256 resize한다.
5. 기존 2.5D I2SB: 중심 endpoint 1채널, `[z-1,z,z+1]` condition 3채널, 끝 index clamp.
   EMA float32, interval=1000, beta_max=0.3, stochastic sampling, denoised clipping 없음.
   방향별 모델은 첫 사용에 로드하여 재사용하고 전체 실행 종료에 해제한다.
6. 출력은 **256×256×depth를 유지한 채 `[-1,1]`로 clip**한다.
   같은 전처리를 거친 real과 함께 사용한다. Synthetic을 native XY로 확대하지 않는다.
7. pseudo spacing=`(1,1,work_z_spacing)`, size=`(256,256,work_depth)`.
   direction은 work와 같다. Work의 physical center를 계산하고 새 nominal grid의 center가
   같은 위치가 되도록 origin을 계산한다. Identity affine이나 spacing만 교체하는 방식이 아니다.

이는 `data_synthetic/model-space`의 legacy `make_network_space_reference` 방식이다.
**Nominal pseudo-1mm이며 physical 1mm가 아니다.** `data_synthetic_v2`, Dataset506 변환과 혼용하지 않는다.
원본 `run_part_*/selection_summary_*.json`에서도 두 checkpoint 파일, RAS, resize-slices,
keep_network_resolution=true를 확인했다. 생성 스크립트 기본 sampling은 batch8/NFE50이며
원본 generator에는 현재 배포의 볼륨별 seed0 재초기화가 없다. 이번 변경은 배포의 seed0,
sampling 및 normalization을 유지하므로 저장 synthetic과 E2E voxel 완전 일치를 가정하지 않는다.
Transform schema는 `i2sb-pseudo-1mm-transform-v1`이다. original/ras/work_before_xy_resize/
pseudo_model_space의 size·spacing·origin·direction·affine, z resampling, 입력 sequence와
real_channel_index를 신규 입력에서 직접 생성한다. GT는 생성 과정에 필요하지 않다.

## Segmentation과 native 복원

- 입력 shape 및 전체 geometry가 서로/transform과 일치해야 한다. 불일치는 명시적 오류다.
- 채널은 항상 **ch0=T1CE-like, ch1=BB-like**:
  T1CE 입력은 `[real T1CE, synthetic BB]`, BB 입력은 `[synthetic T1CE, real BB]`.
- real normalized background `-1`(absolute tolerance `1e-6`, rtol=0)을 제외한 bbox를 구해
  두 채널에 동일 적용한다. bbox와 real-channel provenance를 기록한다.
- nnU-Net 공식 전처리(축 transpose, 내부 nonzero crop, 채널별 Z-score, mask=false)를 사용한다.
  Dataset510 목표 spacing `[1,1,1]`이며 같은 shape의 spacing resampling은 no-op다.
  전처리 전후 shape가 달라지면 오류를 내며 추가 spacing 보간을 도입하지 않는다.
- patch DHW=`112×128×160`, sliding-window step=0.5, Gaussian 병합,
  checkpoint에 저장된 mirroring 축을 적용한다.
- logits → softmax foreground probability → nnU-Net 내부 crop 및 transpose 복원 →
  export bbox 복원(바깥 0) → **전체 pseudo probability >=0.5**로 이진화한다.
- XY를 transform의 work_before_xy_resize 크기로 **torch `F.interpolate(mode="nearest")`** 복원한다.
  floor indexing이며 nearest-exact/linear/bilinear가 아니다.
- work geometry를 부여하고 original reference로 **SimpleITK nearest** resample한다.
  원본 shape·spacing·origin·direction을 모두 검사한다. Orientation을 별도로 중복 복원하지 않는다.
- Debug native probability도 같은 두 단계 nearest를 사용한다.
  따라서 probability threshold와 binary mask 복원 순서가 교환 가능하다.
- pseudo grid를 physical 영상으로 직접 native에 resample하거나 단일 3D resize로 대체하지 않는다.

## 출력

```text
output/
  manifest.json, manifest.partial.json
  errors.json                         # 실패 시
  <output_id>.nii.gz                   # 최종 native binary mask; ON이면 후처리 적용
  <output_id>/logs/
    result.json                       # dataset510, job131329, filename/stored epoch
    da_result.json                    # DA 수행 시, seed0/sampling/transform
    transform.json                    # 전용 pseudo transform
    crop.json                         # export bbox + nnU-Net properties + channel provenance
    postprocessing.json, synthstrip.log # 후처리 ON
    debug/                            # --debug ON일 때만
      real_pseudo.nii.gz, synthetic_pseudo.nii.gz
      da_input_network.nii.gz, da_output_network.nii.gz, da_geometry.json
      seg_input_0000.nii.gz, seg_input_0001.nii.gz
      model_input_0000.nii.gz, model_input_0001.nii.gz
      model_mask.nii.gz, model_probability.nii.gz
      pseudo_mask.nii.gz, pseudo_probability.nii.gz
      native_mask.nii.gz, native_probability.nii.gz, geometry.json
      brain_mask.nii.gz                 # 후처리 ON
      native_mask_before_skullstrip.nii.gz, native_probability_before_skullstrip.nii.gz # ON
```

`model_*`는 내부 nnU-Net crop/transpose 전처리 grid, `pseudo_*`는 crop을 되돌린 전체 nominal grid다.
`da_output_network`는 clip 전 sampling 출력, `synthetic_pseudo`는 clip 후 segmentation 입력이다.
Pseudo voxel 수를 physical mm³로 해석하지 않는다. Debug OFF의 영상 출력은 최종 native mask뿐이다.

## 환경·검증

`docker/requirements.txt`의 torch2.3.1, CUDA11.8/cuDNN8 runtime, nnunetv2==2.8.1을 유지한다.
기존 `brainmetaseg-da-deploy:0.2.0` 이미지를 dependency runtime으로 사용할 수 있다.
실행 시 반드시 번들을 mount하여 `bash /bundle/run.sh`로 호출한다. 이미지 내부의 과거 설치 코드 대신
run.sh가 지정한 현재 bundle/src가 사용된다. 소스 package version은 2.0.0이다.
오프라인 이미지 아카이브는 `docker/load.sh v1` 또는 `docker/load.sh v2`로 로드한다.

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' --verbose
python tools/refresh_assets.py
python tools/refresh_assets.py --verify
```

학습 allocation에 attach하지 않는다. 검증 컨테이너는 독립 sbatch로 제출하고 할당 GPU UUID만 노출한다.
광범위 NAS root mount와 `srun --overlap docker run`은 금지한다.
샘플 영상과 outputs는 연구/검증용이며 환자 데이터 없는 배포 패키지에서는 제외한다.
패키지 구성에 맞게 assets.json 목록도 갱신해야 한다. 기존 I2SB/guided-diffusion 라이선스를 유지한다.

변경 파일: `run.sh`, `src/pyproject.toml`, `src/bms_deploy/{cli,da_adapter,segment,pseudo,trainer,geometry}.py`,
`tests/{test_pseudo,verify_dataset510,verify_da_pipeline}.py`, `tests/dataset510_validation_results.json`,
`tools/refresh_assets.py`, 본 문서, 신규 segmentation weights와 assets.json.
과거 source는 outputs의 migration backup에 보존했다. 기존 연구 결과와 실행 중인 학습 job은 변경하지 않았다.

### Dataset510 전환 검증 결과 (2026-10-06, v1)

검증 출력: `outputs/dataset510_migration_20261006_144603/`.
참조: `test_seed42_dataset510_focal131329_epoch0950_job132408/evaluation_job132409`의 skull stripping 미적용 결과.

- CPU job 134640: 회귀 20개 및 strict checkpoint loading 통과. 연구 소스/NAS 데이터 없이 실행했다.
- 저장 pair GPU job 134638 (RTX8000): 6개 사례의 채널 순서, export bbox,
  nnU-Net normalization/transpose/crop 배열이 기존 exported pair 전처리와 **정확히 일치**했다.
- 참조 model probability를 새 crop/nearest 복원 코드에 직접 입력하면 **6개 모두 pseudo/native mask 불일치 0**.
- 새 GPU 추론 결과의 native mask는 **6개 모두 불일치 0**.
  Pseudo mask는 5개 일치, BrainMet_test_00048에서 1 voxel 불일치.
  해당 확률은 기존 0.5001220703 / 신규 0.4998168945로 0.5를 서로 다르게 통과했다.
  이 사례의 전체 model probability 최대 절대 차이는 0.001438886, MAE 2.7434e-9다.
  복원 단계 차이와 구분되는 추론 수치 차이며, GPU/cuDNN 등의 개별 원인은 분리하지 않았다.
  따라서 pseudo mask 전체의 bitwise 동등성은 주장하지 않는다.

| Case | 입력 | native XYZ / orientation | z spacing | pseudo 불일치 voxel | native 불일치 voxel |
|---|---|---|---:|---:|---:|
| 00001 | T1CE | 1024×1024×150 / LPS(oblique) | 1 | 0 | 0 |
| 00006 | BB | 512×512×160 / LPS | 1 | 0 | 0 |
| 00011 | T1CE | 480×480×150 / LAS | 1 | 0 | 0 |
| 00018 | T1CE | 512×512×115 / LPS | 1.5 | 0 | 0 |
| 00048 | BB | 240×240×176 / SAR | 1 | 1 | 0 |
| 00059 | BB | 180×256×256 / RAS | 1 | 0 | 0 |

E2E job 134642(T1CE)와 134646(BB), 후속 CPU 평가 134650을 완료했다.
134642는 추론 자체가 완료된 뒤 검증 도구의 output_id 경로 처리에서 실패했으며,
수정된 평가 도구로 저장 결과를 읽어 134650에서 평가했다. DA 추론은 반복하지 않았다.
두 실제 볼륨 모두 입력에서 생성한 transform의 geometry가 기존 transform과 일치했고,
native mask의 원본 geometry도 검증했다. 채널 순서와 export bbox, nnU-Net 내부 crop/shape도
참조와 일치했다. Sampling은 기존 seed0 / batch8 / NFE50이다.

| Case | DA real 최대 절대 차이 | DA synthetic MAE vs 저장 pair | pseudo 불일치 voxel | native 불일치 voxel | native 예측 간 Dice |
|---|---:|---:|---:|---:|---:|
| BrainMet_test_00018 | 1.7881393e-07 | 0.016391567 | 92 | 244 | 0.99470991 |
| BrainMet_test_00048 | 1.1920929e-07 | 0.021163022 | 21 | 10 | 0.93902439 |

위 Dice는 기존 예측과의 일치도이며 GT에 대한 정확도 DSC가 아니다.
저장 synthetic의 과거 생성 난수 상태를 새 DA 실행과 동일하다고 가정하지 않았다.
과거 생성 코드는 원본 pixel type으로 z resample 후 float32로 변환하며, 현재 배포는 입력부터
float32를 사용한다. 이 순서도 이번 변경에서 유지했으며 저장 real과의 작은 수치 차이를 기록했다.
고정 pair 검증과 분리했으므로 E2E의 차이를 segmentation/복원 코드 차이로 단정하지 않는다.
- BrainMet_test_00018: 과거 seed0 raw DA 출력 대비 MAE 0.014252424, 최대 절대 차이 1.6436873.
- BrainMet_test_00048: 과거 seed0 raw DA 출력 대비 MAE 0.018442868, 최대 절대 차이 1.5915836.

과거 seed0 결과와도 완전 일치하지 않았다. 두 사례의 DA 입력 텐서는 과거 seed0 실행과
정확히 일치하지만 raw 출력은 다르다. 이전 A6000 실행과 이번 RTX8000 실행은 환경이 다르며,
GPU/cuDNN 조합 등 구체적 원인은 이 비교만으로 확정할 수 없다.

추가로 job 134655에서 동일 RTX8000에 변경 전후 DA 코드를 각각 로드해 대조했다.
원본 z spacing 1.5mm 영상의 3개 slice(전처리 후 depth 4)를 seed0/batch8/NFE50으로 실행했으며,
**T1CE→BB와 BB→T1CE 모두 입력 및 clip 전 raw synthetic이 정확히 일치(불일치 0, 최대 차이 0)**했다.
이는 이번 연결부 변경이 동일 환경에서 DA 출력을 바꾸지 않았다는 소규모 대조 결과이며,
서로 다른 GPU의 전체 볼륨 결과에 대한 bitwise 재현성 보장은 아니다.
Native probability의 >=0.5 결과와 native binary mask의 일치도 추가 확인했다.

GPU job 134646의 별도 컨테이너는 **bundle과 writable 검증 output만 mount**하여 실행했다.
NAS 입력 데이터/연구 소스/GT 없이 생성한 DICOM 2개를 debug ON/OFF로 실행하고,
같은 영상을 `.nii`/`.nii.gz`로 읽은 결과를 비교했다. Native mask·synthetic이 일치했고,
debug OFF의 영상 출력은 최종 native mask뿐이었다.

Geometry 비교 허용 오차는 affine atol=1e-4, rtol=0이며 shape는 동일해야 한다.
00048에서 다른 pseudo voxel XYZ=(142,63,72)는 규정된 XY nearest 복원에서 선택되지 않아
native mask에는 차이가 없었다. 이 규칙을 보정하거나 nearest-exact로 변경하지 않았다.

재현 결과 요약은 `tests/dataset510_validation_results.json`, 상세 결과는 위 migration output의
`validation_summary.json`, `pairs_134638`, `e2e_134642`, `e2e_134646`, `portable_smoke_134646`에 있다.
78개 전수 재추론 및 GT 성능 평가는 이번 검증 범위에 포함하지 않았다.


`tests/verify_dataset510.py`가 이번 비교용 entrypoint다. 연구 참조 경로는 인자로 전달한다.
`tests/verify_reference.py`와 `tools/bundle_assets.py`는 과거 Dataset505용이며 현재 bundle에 실행하지 않는다.
과거 tests/*validation_results.json은 당시 결과를 그대로 보존한다.


## v2 후처리 검증 (2026-10-08)

검증은 기존에 저장한 78개 결과 중 6개 native lesion mask와 원본 MRI를 이용했다.
**DA 및 병변 분할 모델은 새로 실행하지 않았다.** 기존 입력/학습 job/연구 결과는 변경하지 않았다.
신규 검증 로그와 brain mask, 후처리 전후 mask/probability는 사용자 지정 NAS의
`logs/nnUNet_test_inference/_dist_v2/postprocess_validation_20261008_122415/`에 별도로 저장했다.
이는 검증 기록 위치이며 배포 실행 코드의 기본 출력 경로가 아니다.

- 기존 20개 + 신규 후처리 8개 단위 테스트가 **v1/v2 각각 28개 통과**했다(job 134962).
  CLI 연결 테스트의 DA/Segmenter는 mock이며 실제 전체 추론 검증과 구분한다.
- 실제 SynthStrip 및 후처리 검증은 CPU 2 threads, 독립 Slurm CPU job 134963에서 수행했다.
  T1CE/BB, XY 1024/512/480/240/180, z spacing 1/1.5mm, LPS/LAS/SAR/RAS를 포함한다.
- 6개 모두 brain/native mask의 원본 shape·geometry 일치, binary 값, 병변 AND 연산을 확인했다.
  Debug probability와 최종 mask의 일치 및 실패 시 최종 파일을 남기지 않는 동작은 단위 테스트로 검증했다.
- v1/v2 이미지 모두 tar.gz 저장 후 SHA256 검사와 실제 `docker load`를 수행했고,
  재로드 전후 image ID가 동일했다(job 134966). v1에서 `run_v1.sh --help`도 확인했다.
- v1 tar.gz SHA256: `925cb8dcde66d178cc81b4d7a3f95d366720e41dc78bcdc3dcf08e86fe2c07fa`.
- v2 tar.gz SHA256: `153ecfbc0faefc3911f1756128350658fb4e61bc43b07fc52c780cdda25f2b5a`.
- v2 image ID: `8a2713adfcf53cf69d2ce98ec7ab6567a00eba2fa1e70e2e6f8647470efa5e28`.

기존 `data/brainmask_SynthStrip/masks_binary`와의 비교:

| Case | Sequence | brain 불일치 voxel | brain Dice | 기존 brain으로 필터링한 병변과 불일치 voxel |
|---|---|---:|---:|---:|
| BrainMet_test_00001 | T1CE | 0 | 1 | 0 |
| BrainMet_test_00006 | BB | 1 | 0.9999999383 | 0 |
| BrainMet_test_00011 | T1CE | 0 | 1 | 0 |
| BrainMet_test_00018 | T1CE | 0 | 1 | 0 |
| BrainMet_test_00048 | BB | 20,571 | 0.9934044413 | 0 |
| BrainMet_test_00059 | BB | 1 | 0.9999996919 | 0 |

표의 Dice는 저장된 brain mask와의 일치도이며 GT에 대한 병변 DSC가 아니다.
Brain mask 자체의 완전 일치는 3/6이다. 저장 brain과 새 brain을 적용한 병변 결과는 6/6 완전 일치했다.
후처리 전후 병변 voxel은 00001에서 42,016→42,007, 00018에서 23,060→23,046이고 나머지는 동일했다.
요약은 `tests/postprocess_validation_results.json`, 상세 비교는 검증 폴더의 `comparison.json`에 있다.

### 저장 reference 차이의 대조 확인

차이가 난 00006/00048/00059를 공식 `freesurfer/synthstrip:1.8` 이미지에서 동일 원본,
CPU 2 threads로 재실행했다(job 134977). **세 사례 모두 공식 이미지와 v2 brain mask가 voxel 단위로 완전히 일치**했다.
공식 환경은 torch 2.1.2+cpu, v2는 기존 torch 2.3.1 환경이며 이번 비교에서는 결과 차이가 없었다.

00048은 공식 이미지로 T1CE를 추가 실행한 뒤 BB brain mask와 OR했다(job 134978).
이 **두 시퀀스 합집합은 저장 reference와 불일치 0 voxel**이었다. 저장된 T1CE/BB reference도
서로 완전히 같고, 기존 생성 로그에 `paired_t1ce_or_bb` / `t1ce_bb_union` 기록이 있다.
따라서 이 사례의 20,571 voxel 차이는 단일 BB brain mask와 과거 합집합 reference의 차이로 확인했다.
v2는 주어진 단일 MRI에서 brain mask를 만들며, 별도 시퀀스나 연구용 reference를 요구하지 않는다.
배포 구현에 reference union을 추가하지 않았다.

00006/00059의 과거 reference 대비 1 voxel 차이는 공식 이미지에서도 재현되었다.
이 두 차이의 과거 생성 환경/수치 연산 원인은 확정하지 않았다. 완전 일치로 간주하지 않는다.
대조 기록은 검증 폴더의 `official_comparison.json`, `official_pair_comparison.json`에 있다.
최초 공식 이미지 대조 제출 134976은 컨테이너 namespace 오류로 추론 전에 실패했다.
이미지 복원 작업 종료 후 독립 job 134977로 성공했으며 실행 중 컨테이너를 변경하거나 migrate하지 않았다.

남은 범위: 전체 78개 DA→병변 분할→후처리 통합 추론, 실제 DICOM 원본의 SynthStrip 검증,
GPU 후처리의 모델 동시 상주 VRAM 및 실행속도 검증은 수행하지 않았다.
후처리를 끈 경로의 계산은 기존과 동일하며, 이번 실제 영상 검증은 후처리 단계에 한정된다.
