# BrainMetaSeg 배포 파이프라인

Brain MRI와 시퀀스 정보(`T1CE` 또는 `BB`)를 입력받아 synthetic 반대 시퀀스를 생성하고,
Dataset510 / 131329 Focal 모델로 metastasis 병변 mask를 예측한다.

## 한국어 실행 안내

| 실행 방식 | 이미지 | 실행 파일/옵션 | 안내 |
|---|---|---|---|
| v1: 기존 병변 분할 | `brainmetaseg-da-deploy:v1` | `run_v1.sh` | [v1 README](README_v1.md) |
| v2: SynthStrip 1.8 후처리 | `brainmetaseg-da-deploy:v2` | `run.sh --skull-strip on` | [v2 README](README_v2.md) |

두 안내는 현재 `v2-synthstrip` 브랜치의 번들을 기준으로 한다.
v2 이미지에서도 후처리는 기본 OFF다. v1은 후처리를 끈 기존 동작과 백업 실행을 위한 경로다.
이 README의 실행 파일과 버전별 Docker 스크립트는 이전 `main` 스냅샷과 다를 수 있다.

**실행하려면 별도로 전달받은 모델 weights가 필요하다.** Docker 아카이브도 Git에 포함되지 않는다.
각 안내에 파일 준비, 이미지 로드/빌드, NIfTI/DICOM CSV, Docker 실행 예시,
옵션, 출력 파일 및 오류 확인 방법을 정리했다.

모델 출처, 전처리·복원 계약과 검증 결과는 [HANDOFF.md](HANDOFF.md)를 참고한다.
