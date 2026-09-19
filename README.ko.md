# Wiggle-3D

[English](README.md) | [한국어](README.ko.md)

3D 다중 렌즈 필름 카메라(RETO 3D, Nimslo, Nishika 등)의 스캔 이미지를 자동으로 분할하고 정밀하게 정렬하여 흔들림 없는 Wiggle 3D 입체 루프 GIF를 생성하는 고성능 병렬 Rust 엔진입니다.

---

## Wiggle-3D란 무엇인가요?

3D 필름 카메라(RETO 3D, Nimslo 3D, Nishika N8000/N9000 등)는 표준 35mm 필름에 3개 또는 4개의 하프프레임 렌즈를 통해 찰나의 순간을 동시에 포착합니다. 현상소에서 스캔하면 이 다중 렌즈 프레임들이 하나의 가로로 긴 필름 스트립 이미지로 출력됩니다.

자연스러운 입체 시차(Wiggle Stereogram) 애니메이션을 만들기 위해서는 다음 과정이 필요합니다:
1. 스캔 원본에서 각 하위 프레임(Sub-frame) 영역을 찾아 분할 잘라내기(Crop)를 수행해야 합니다.
2. 각 프레임 간의 주요 피사체(인물, 얼굴, 전경 객체) 초점 위치를 서브픽셀 단위로 정밀하게 정렬하여 어지러운 수직 흔들림과 급격한 수평 이탈을 제거해야 합니다.
3. 정렬된 프레임들을 매끄러운 핑퐁 루프(`0 → 1 → 2 → 1`)와 균일한 색상 양자화를 통해 깜빡임 없는 단일 GIF로 합성해야 합니다.

**Wiggle-3D**는 이 모든 복잡한 처리 과정을 안전하고 빠른 고성능 CLI 도구 및 Rust 라이브러리로 자동화합니다.

---

## 샘플 결과 (Sample)

원본 3구 렌즈 필름 스캔 스트립(좌측)을 자동으로 분할하고 인물 초점 고정 및 정밀 정렬을 거쳐 합성된 3D 입체 루프 애니메이션(우측)입니다:

<p align="center">
  <img src="assets/sample_scan.jpg" alt="3구 렌즈 필름 스캔 스트립" width="560" />
  &nbsp;&nbsp;
  <img src="assets/sample_wiggle.gif" alt="Wiggle 3D 입체 루프" width="179" />
</p>

---

## 주요 특장점 (Featured Advantages)

- **완전 병렬화된 고성능 아키텍처 (Fully Parallelized Architecture)**: 다단계 스트리밍 파이프라인과 멀티코어 병렬 처리(`rayon` + 비동기 파일 I/O)를 기반으로 구축되었습니다. 고해상도 필름 스캔 디렉터리 전체를 메모리 급증 없이 최대 처리량으로 고속 일괄 변환합니다.
- **머신러닝 기반 정밀 정렬 (ML-Based Alignment)**: 딥러닝 특징점 검출 및 에피폴라 시차 추정을 통해 다각도 렌즈 프레임 간의 시각적 패턴을 추적하고 수직/수평 흔들림을 완벽히 보정하여 부드러운 3D 입체감을 완성합니다.
- **사용자 관점의 인물 얼굴 감지 및 초점 고정 (Face Focal Locking)**: 사용자 시선이 집중되는 인물 피사체와 얼굴 랜드마크를 자동으로 감지하고, 얼굴을 기준으로 입체 초점 평면을 고정하여 입체 루프 재생 시 인물이 항상 선명하고 안정감 있게 유지됩니다.
- **자동 서브프레임 분할 (Automated Frame Splitting)**: 원본 필름 스트립에서 프레임 경계를 자동으로 감지하고 분할하여 번거로운 수동 자르기 작업을 없앱니다.
- **깜빡임 없는 통합 팔레트 양자화 (Flicker-Free Unified Palette)**: 모든 프레임에서 전역 256색 팔레트를 균일하게 연산하고 Floyd-Steinberg 디더링을 적용하여 프레임 전환 시 발생하는 색상 깜빡임을 제거합니다.
- **진단용 시각화 오버레이 내장 (Diagnostic Visual Overlays)**: `--debug` 옵션 실행 시 RoI 경계 박스, 얼굴 랜드마크 포인트, 에피폴라 매칭 벡터가 표시된 디버그 이미지 아티팩트를 자동 생성합니다.

---

## 빠른 시작 (Quick Start)

### 설치 방법

#### 1-라인 자동 설치 (권장)

**Linux / macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/phd-eokho/wiggle-3d/main/install.sh | sh
```

NVIDIA GPU (CUDA) 가속 사용 시:
```bash
curl -fsSL https://raw.githubusercontent.com/phd-eokho/wiggle-3d/main/install.sh | WIGGLE3D_CUDA=1 sh
```

*삭제(Uninstall) 시 (Linux / macOS):*
```bash
curl -fsSL https://raw.githubusercontent.com/phd-eokho/wiggle-3d/main/install.sh | sh -s -- --uninstall
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/phd-eokho/wiggle-3d/main/install.ps1 | iex
```

NVIDIA GPU (CUDA) 가속 사용 시:
```powershell
$env:WIGGLE3D_CUDA = "1"; irm https://raw.githubusercontent.com/phd-eokho/wiggle-3d/main/install.ps1 | iex
```

*삭제(Uninstall) 시 (Windows):*
```powershell
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/phd-eokho/wiggle-3d/main/install.ps1))) -Uninstall
```

설치 스크립트는 최적화된 바이너리(`reto-cli` / `reto-cli.exe`)를 `~/.local/bin/`에 설치하고, 필요한 ONNX Runtime 동적 라이브러리를 자동 설정합니다.

#### 소스 코드에서 직접 빌드 (개발자용)

```bash
git clone https://github.com/phd-eokho/wiggle-3d.git
cd wiggle-3d
cargo build --release
```

### 기본 사용법

#### 단일 스캔 파일 처리
```bash
reto-cli --input samples/film_strip_01.jpg --output output_dir/
```

#### 디렉터리 내 전체 스캔 일괄 배치 처리
```bash
reto-cli --input /path/to/scans/ --output /path/to/results/
```

#### 디버그 시각화 및 상세 진단 로그 활성화
```bash
reto-cli --input samples/ --output results/ --debug -v
```

---

## CLI 사용법 레퍼런스 (CLI Usage Reference)

### 명령어 형식

```bash
reto-cli [OPTIONS] --input <PATH> --output <DIR>
```

### 옵션 및 플래그 상세

| 옵션 / 플래그 | 타입 / 기본값 | 설명 |
| :--- | :--- | :--- |
| `-i, --input <PATH>` | `Path` (필수) | 단일 스캔 이미지 파일 경로 또는 배치 처리 대상 디렉터리 경로. |
| `-o, --output <DIR>` | `Path` (필수) | 출력 GIF 및 진단 아티팩트가 저장될 대상 디렉터리. |
| `--debug` | `bool` (플래그) | 중간 디버그 시각화 출력 활성화 (RoI 오버레이, 특징점 매칭 벡터, 시차 로그 등). |
| `--gif-delay <MS>` | `u32` (기본값: `100`) | 프레임 간 애니메이션 지연 시간 (밀리초 단위, 100ms = 10 fps). |
| `--no-dither` | `bool` (플래그) | NeuQuant 색상 양자화 시 Floyd-Steinberg 디더링 비활성화. |
| `--no-progress` | `bool` (플래그) | 대화형 터미널 진행률 표시줄 비활성화 (CI/CD 환경 및 로그 수집에 권장). |
| `--log-file <FILE>` | `Path` (선택) | 상세 실행 및 진단 로그가 기록될 커스텀 로그 파일 경로. |
| `-q, --quiet` | `bool` (플래그) | 치명적 오류를 제외한 모든 콘솔 출력 억제. |
| `-v, -vv` | `Count` (플래그) | 로그 상세 수준 증가 (`-v`: DEBUG, `-vv`: TRACE). |

### 지원 이미지 포맷

`reto-cli`는 다음 확장자를 가진 이미지 파일을 자동으로 탐색하고 처리합니다:
- JPEG: `.jpg`, `.jpeg`
- PNG: `.png`
- TIFF: `.tiff`, `.tif`
- Bitmap: `.bmp`
- WebP: `.webp`

---

## 개발자 가이드 (Developer Guide)

### 시스템 아키텍처

프로젝트는 명확한 관심사 분리를 위해 두 개의 크레이트로 구성되어 있습니다:

```
wiggle-3d/
├── reto-core/      # 순수 컴퓨터 비전 라이브러리 및 신경망 추론 엔진
└── reto-cli/       # CLI 프론트엔드, 듀얼 싱크 로깅 및 진행률 리포터
```

- **`reto-core`**: CLI 의존성이 전혀 없는 독립형 Rust 라이브러리로, RoI 분할, 신경망 특징점 및 얼굴 감지, 시차 정렬 및 GIF 인코딩을 제공합니다.
- **`reto-cli`**: 인자 검증, 파일 탐색, 터미널 진행률 시각화(`indicatif`), 듀얼 싱크 로깅(`tracing-subscriber`)을 담당하는 경량 CLI 프론트엔드입니다.

### 빌드 및 테스트

#### 사전 요구사항
- Rust 1.75+ (2021 Edition)
- 표준 C/C++ 툴체인 (ONNX Runtime 네이티브 바인딩에 필요)

#### 전체 단위 및 통합 테스트 실행
```bash
cargo test --all-targets --all-features
```

#### 문서화 테스트 실행
```bash
cargo test --doc
```

#### 린터 및 포맷 검사
```bash
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

---

## 라이선스 및 서드파티 고지 (License & Third-Party Notice)

본 프로젝트의 소스코드는 [MIT 라이선스](LICENSE)에 따라 배포됩니다.

### 서드파티 딥러닝 모델 가중치

정렬 과정에서 사용되는 사전 학습된 신경망 모델 가중치는 런타임에 로드되며 각각의 원저작자 라이선스를 따릅니다:
- **SuperPoint**: Copyright (c) Magic Leap, Inc. [Magic Leap Non-Commercial Research License](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE)에 따라 배포됩니다. SuperPoint 모델 가중치는 비상업적 연구 및 개인적 평가 목적으로만 사용이 제한됩니다.
- **RetinaFace**: MobileNet0.25 백본 사전 학습 가중치는 MIT 라이선스 및 학술적 허용 라이선스 조건에 따라 제공됩니다.

자세한 내용은 [LICENSE](LICENSE) 파일을 참조하십시오.
