# MetaSeg 사용 안내

뇌 MRI를 넣으면 병변 마스크 파일을 만들어 주는 프로그램입니다.
**영상의 종류(BB 또는 T1CE)는 사용자가 목록에 적어 주어야 합니다. 영상 종류를 자동으로 분류하는 프로그램은 아닙니다.**

처음 한 번만 설치하면, 다음부터는 `input`에 영상을 넣고 **`bash run.sh`**만 실행하면 됩니다.
결과는 `output`에 저장됩니다.

## 1. 사용할 컴퓨터를 준비하세요

설치를 담당하는 분께 아래 표를 전달하세요. 이 안내는 **Ubuntu가 설치된 NVIDIA GPU 서버**를 기준으로 합니다.

| 항목 | 준비 기준 |
|---|---|
| 운영체제 | Ubuntu 22.04 LTS 또는 24.04 LTS, 64비트 x86 컴퓨터 |
| GPU | NVIDIA GPU 1개, 전용 GPU 메모리 24GB 이상 권장 |
| 시스템 메모리(RAM) | 64GB 이상 권장, 여유가 있으면 96GB 이상 |
| CPU | 사용 가능한 CPU 8개 이상 권장 |
| 저장 공간 | 비어 있는 공간 100GB 이상 권장. 영상 수가 많으면 추가 공간 필요 |
| 인터넷 | 처음 설치하거나 실행 환경을 만드는 동안 필요 |
| 설치 권한 | 관리자 권한 또는 서버 관리자의 도움 필요 |

위 표는 환경을 준비하기 위한 권장 기준입니다. 모든 조합에서 보장되는 최저 사양은 아닙니다.
전체 78개 영상은 A6000 48GB, 할당 CPU 14개, RAM 요청 약 94GB 환경에서 처리했습니다.
그 실행에서 GPU 메모리는 최대 약 13.7GB로 관측됐습니다.
다른 작업이 같은 GPU를 사용하지 않는 환경을 준비하세요.
Windows와 Mac에서의 직접 실행은 이 안내에 포함하지 않습니다.

## 2. NVIDIA 드라이버와 Docker를 설치하세요 — 처음 한 번

Docker는 이 프로그램에 필요한 실행 환경을 준비해 주는 도구입니다.
이미 설치되어 있는 서버라면 담당 관리자에게 확인하고 **3번부터 진행**하세요.
아래 설치 명령은 새 Ubuntu 서버용입니다. 공용 서버에서는 관리자가 진행해야 합니다.

### 2-1. NVIDIA GPU가 인식되는지 확인하기

1. Ubuntu의 **터미널**을 엽니다. 바탕화면에서 `Ctrl` + `Alt` + `T`를 눌러도 됩니다.
2. 아래 한 줄을 복사하고 Enter를 누릅니다.

```bash
nvidia-smi
```

GPU 이름이 적힌 표가 나오면 다음으로 이동합니다.
명령을 찾을 수 없거나 오류가 나오면, 관리자에게 **NVIDIA 드라이버 설치 및 재부팅**을 요청하세요.
관리자용 공식 안내: [Ubuntu NVIDIA 드라이버 설치](https://ubuntu.com/server/docs/nvidia-drivers-installation/).

### 2-2. Docker 설치하기

아래 블록을 터미널에 붙여 넣습니다. 비밀번호를 물으면 Ubuntu 로그인 비밀번호를 입력합니다.
입력 중 글자가 보이지 않는 것은 정상입니다.

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo docker run --rm hello-world
```

`Hello from Docker!`가 나오면 Docker 설치가 완료된 것입니다.
위 순서는 [Docker 공식 Ubuntu 설치 안내](https://docs.docker.com/engine/install/ubuntu/)를 기준으로 합니다.
설치 충돌이 나면 기존 Docker/Podman 등을 임의로 삭제하지 말고 관리자에게 확인하세요.

### 2-3. Docker에서 GPU를 사용할 수 있게 설정하기

아래 블록을 이어서 실행합니다.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo usermod -aG docker "$USER"
```

이 단계는 [NVIDIA 공식 설치 안내](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)를 기준으로 합니다.
Docker 실행 권한은 관리자 수준의 권한이므로 서버 관리자가 허용한 계정에만 부여하세요.

이제 **Ubuntu에서 로그아웃했다가 다시 로그인**하세요. 터미널 창만 닫았다 여는 것으로는 부족할 수 있습니다.
다시 터미널을 열어 아래를 실행합니다.

```bash
docker run --rm hello-world
```

`sudo` 없이도 `Hello from Docker!`가 나오면 준비 완료입니다.

## 3. 프로그램과 모델 파일을 받으세요

1. [프로그램 저장소](https://github.com/2gaeun/MetaSeg_release)를 엽니다.
2. 브랜치가 **`main`**인지 확인합니다.
3. 초록색 **Code** 버튼을 누릅니다.
4. **Download ZIP**을 누릅니다.
5. 받은 ZIP 파일의 압축을 풉니다.
6. 압축을 푼 `MetaSeg_release-main` 폴더를 엽니다. 그 안에 `run.sh`, `setup.sh`, `input`, `output`, `weights`가 보여야 합니다.
7. 배포 담당자에게 **모델 파일 묶음**을 별도로 받아 압축을 풉니다. GitHub 다운로드에는 큰 모델 파일이 포함되어 있지 않습니다.
8. 받은 `weights` 폴더의 내용을 프로그램의 `weights` 폴더에 복사합니다. 폴더를 한 겹 더 넣지 마세요.

복사가 끝나면 다음 파일들이 보여야 합니다. 같이 받은 나머지 설정 파일도 유지하세요.

```text
MetaSeg_release-main/
  run.sh
  setup.sh
  input/
  output/
  weights/
    t1ce2bb_112788_step0175000/
      ema.pt
    bb2t1ce_114599_step0150000/
      ema.pt
    segmentation_131329/
      checkpoint_epoch_0950.pth
```

`weights/weights/...`가 되면 잘못 복사한 것입니다.

## 4. 처리할 영상을 input에 넣으세요

처음에는 영상 한 개로 시작하는 것을 권장합니다. **NIfTI 파일 또는 DICOM 폴더를 사용할 수 있습니다.**
한 번에 두 종류를 섞지는 마세요. 영상 파일이나 폴더의 이름은 영문·숫자·밑줄로 간단히 정하면 편합니다.

### NIfTI 파일을 받은 경우

1. `.nii.gz` 또는 `.nii` 파일을 프로그램의 `input` 폴더에 복사합니다.
2. 아래 예시에서는 파일 이름을 `case01.nii.gz`, `case02.nii.gz`라고 하겠습니다.
3. `input/series.csv`를 텍스트 편집기로 엽니다.
4. 예시 내용을 실제 파일에 맞게 고친 뒤 저장합니다.

```text
input/
  series.csv
  case01.nii.gz
  case02.nii.gz
```

`series.csv` 내용:

```csv
id,sequence
case01,T1CE
case02,BB
```

첫 번째 열에는 **파일 이름에서 `.nii.gz` 또는 `.nii`를 뺀 이름**을 적습니다.
두 번째 열에는 영상의 종류인 `T1CE` 또는 `BB`를 적습니다.
영상 한 개만 넣었다면 CSV에도 해당 영상 한 줄만 남기세요. 예시 행을 그대로 남겨두지 마세요.
`NonBB`는 지원하는 입력 값이 아닙니다. 영상이 실제 T1CE인지 확인한 뒤 적어 주세요.

### DICOM 파일을 받은 경우

1. 한 series의 DICOM 파일들을 한 폴더에 모읍니다.
2. 그 폴더를 통째로 `input` 아래에 넣습니다. 예: `case01` 폴더.
3. `series.csv` 첫 번째 열에는 그 **폴더 이름**을 적습니다.

```text
input/
  series.csv
  case01/
    slice001.dcm
    slice002.dcm
    ...
```

```csv
id,sequence
case01,BB
```

한 폴더에 여러 series를 섞지 마세요. 폴더와 파일 형식은 프로그램이 확인하므로 실행 옵션은 바꾸지 않습니다.

## 5. 실행 환경을 준비하세요 — 처음 한 번

1. `run.sh`가 보이는 프로그램 폴더의 빈 공간에서 마우스 오른쪽 버튼을 누릅니다.
2. **터미널에서 열기(Open in Terminal)**를 선택합니다.
3. 아래 한 줄을 입력하고 Enter를 누릅니다.

```bash
bash setup.sh
```

필요한 파일을 내려받고 Docker 이미지를 만듭니다. 처음에는 시간이 걸릴 수 있으므로 창을 닫지 마세요.
**“환경 준비 완료”**가 나올 때까지 기다립니다.

배포 담당자가 Docker 이미지 파일을 함께 전달했다면, 실행 전에 그 파일을
`docker/MetaSeg_image.tar.gz`에 두세요. 체크섬 파일도 받았다면 바로 옆에 둡니다.
이 경우 같은 `bash setup.sh` 명령으로 받은 이미지를 불러오므로 새로 만들 필요가 없습니다.

## 6. 분석을 시작하세요

같은 터미널에서 아래 한 줄을 실행합니다. **옵션을 붙일 필요가 없습니다.**

```bash
bash run.sh
```

프로그램이 Docker 실행 환경을 열고 영상 분석을 시작합니다.
진행 중인 영상 번호가 표시됩니다. 수 분 이상 걸릴 수 있으므로 **“모든 처리가 완료되었습니다”**가 나올 때까지 기다리세요.
화면에 글자가 잠시 추가되지 않아도 처리 중일 수 있습니다.

`run.sh`를 더블클릭하면 컴퓨터 설정에 따라 편집기가 열릴 수 있습니다.
위처럼 터미널에서 실행하는 방법을 사용하세요.

## 7. output에서 결과를 확인하세요

1. 프로그램 폴더의 `output`을 엽니다.
2. 그 안에서 이번 실행 때 생긴 날짜·시간 이름의 폴더를 엽니다.
3. 입력 이름에 해당하는 `.nii.gz` 파일이 병변 마스크입니다.

```text
output/
  20261013_153000_123456/
    case01.nii.gz
    case02.nii.gz
    logs/
```

`case01.nii.gz`는 원본 영상이 아니라 **병변으로 예측한 부분을 표시한 마스크**입니다.
뇌 바깥 영역을 제외한 최종 결과가 저장됩니다.
기존에 사용하는 NIfTI 뷰어에서 원본 MRI와 함께 열어 확인하세요.
`logs`에는 오류 확인용 기록이 들어 있으므로 문제가 있을 때 담당자에게 전달할 수 있습니다.

다른 영상을 처리할 때는 `input`의 영상과 `series.csv`를 바꾸고 **6번만 다시 실행**합니다.
매번 새 결과 폴더가 생기므로 이전 결과를 덮어쓰지 않습니다.

## 잘 안 될 때

| 화면에 나온 내용 | 확인할 것 |
|---|---|
| 모델 파일이 없습니다 | 담당자에게 받은 모델 파일이 `weights` 바로 아래의 올바른 폴더에 있는지 확인하세요. |
| 해당 영상을 하나만 찾을 수 있어야 합니다 | CSV의 이름과 실제 파일 이름을 맞추세요. 같은 ID의 `.nii`와 `.nii.gz`를 둘 다 두지 마세요. |
| sequence는 T1CE 또는 BB로 적으세요 | CSV 두 번째 열을 확인하세요. |
| Docker에 연결할 수 없습니다 | 2-3의 로그아웃/로그인을 했는지, Docker가 실행 중인지 관리자에게 확인하세요. |
| Docker 이미지가 준비되지 않았습니다 | 먼저 `bash setup.sh`를 실행하세요. |
| GPU 또는 NVIDIA 관련 오류 | 관리자에게 `nvidia-smi` 결과와 오류 메시지를 전달하세요. |
| 일부 또는 전체 처리가 실패했습니다 | 안내된 결과 폴더의 `logs/run.log`를 담당자에게 전달하세요. |

입력 영상과 출력 마스크를 이 공개 GitHub 저장소에 올리지 마세요.
영상 파일과 분석 결과는 Git 업로드 대상에서 제외되어 있습니다.
`series.csv`는 예제 파일로 버전 관리되므로, 실제 ID로 수정한 내용은 공개 저장소에 올리지 마세요.
