# Wiggle-3D

[English](README.md) | [한국어](README.ko.md)

3D 다중 렌즈 필름 카메라(RETO 3D, Nimslo, Nishika 등)의 스캔 이미지를 자동으로 분할하고 정밀하게 정렬하여 흔들림 없는 Wiggle 3D 입체 루프 GIF를 생성하는 고성능 병렬 Rust 엔진입니다.

---

## Wiggle-3D란 무엇인가요?

3D 필름 카메라(RETO 3D, Nimslo 3D, Nishika N8000/N9000 등)는 표준 35mm 필름에 3개 또는 4개의 하프프레임 렌즈를 통해 찰나의 순간을 동시에 포착합니다. 현상소에서 스캔하면 이 다중 렌즈 프레임들이 하나의 가로로 긴 필름 스트립 이미지로 출력됩니다.

자연스러운 입체 시차(Wiggle Stereogram) 애니메이션을 만들기 위해서는 다음 과정이 필요합니다:
1. 필름 스캔 원본에서 각 서브프레임(Sub-frame) 영역을 감지하여 분할(Crop)합니다.
2. 주요 피사체(인물, 얼굴, 전경)를 기준으로 서브픽셀 단위 정밀 정렬을 수행하고, 물리적인 카메라 섀시 처짐(Center Sag) 및 렌즈 간 extrinsic 오차를 보정합니다.
3. 정렬된 프레임들을 적응형 모션 타이밍과 통합 색상 양자화가 적용된 매끄러운 핑퐁 루프(`0 → 1 → 2 → 1`) GIF로 합성합니다.

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

- **완전 병렬화된 스트리밍 파이프라인 (Fully Parallelized Architecture)**: 다단계 파이프라인과 멀티스레드 병렬 처리(`rayon` + 비동기 파일 I/O)를 통해 대용량 필름 스캔 디렉터리 전체를 메모리 낭비 없이 최고 속도로 일괄 처리합니다.
- **계층적 6-DoF Extrinsic Bundle Adjustment**: Dyadic Reduction Tree와 Huber Loss 기반 최적화로 다중 렌즈 간의 결합 카메라 포즈 및 섀시 중심 처짐(Center Sag, $\Delta y$)을 보정하여, 어지러운 수직 떨림을 없애고 reprojection error를 60% 이상 줄입니다.
- **Subpixel Refinement & Drift Gating**: 2D Structure Tensor 기반으로 신경망 특징점 좌표를 서브픽셀 단위로 정밀 보정하며, 렌즈 왜곡이 큰 가장자리 영역의 drift를 방어적으로 차단하여 수렴 안정성을 유지합니다.
- **$\mathrm{SE}(3)$ 기반 적응형 등속 모션 타이밍 (Adaptive Motion Timing)**: 카메라 궤적의 $\mathrm{SE}(3)$(Special Euclidean Group, 6자유도 강체 변환) geodesic 이동 거리를 계산하여 프레임 딜레이를 동적으로 분배함으로써, 렌즈 간격이 비대칭인 카메라에서도 일정한 속도감의 자연스러운 3D 입체 루프를 생성합니다.
- **인물 얼굴 감지 및 초점 고정 (Face Focal Locking)**: 시선이 집중되는 인물과 얼굴 랜드마크를 자동 감지하고, 해당 위치에 초점 평면(Focal Plane)을 고정하여 피사체가 흔들림 없이 또렷하게 유지되도록 합니다.
- **자동 서브프레임 분할 (Automated Frame Splitting)**: 원본 필름 스트립에서 각 렌즈 프레임의 경계를 자동으로 감지하고 분할(Crop)합니다.
- **깜빡임 없는 통합 팔레트 양자화 (Flicker-Free Unified Palette)**: 모든 프레임에 걸쳐 전역 256색 팔레트와 Floyd-Steinberg dithering을 적용하여 프레임 전환 시 발생하는 색상 깜빡임을 제거합니다.
- **네이티브 24-bit TrueColor HEVC MP4 비디오 출력**: 표준 GIF 애니메이션과 함께 $\mathrm{SE}(3)$ 시차 타이밍이 보존된 고품질 24-bit TrueColor H.265 (HEVC) MP4 비디오를 하드웨어 가속으로 생성합니다.
- **진단용 시각화 오버레이 (Diagnostic Visual Overlays)**: `--debug` 옵션 실행 시 RoI 경계 박스, 얼굴 랜드마크, 특징점 매칭 벡터가 시각화된 디버그 이미지를 생성합니다.

---

## 빠른 시작 (Quick Start)

### 설치 방법

#### 1-라인 자동 설치 (권장)

**Linux / macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/phd-eokho/wiggle-3d/main/install.sh | sh
```

NVIDIA GPU (CUDA & NVENC) 가속 사용 시:
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

NVIDIA GPU (CUDA & NVENC) 가속 사용 시:
```powershell
$env:WIGGLE3D_CUDA = "1"; irm https://raw.githubusercontent.com/phd-eokho/wiggle-3d/main/install.ps1 | iex
```

*삭제(Uninstall) 시 (Windows):*
```powershell
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/phd-eokho/wiggle-3d/main/install.ps1))) -Uninstall
```

설치 스크립트는 최적화된 바이너리(Linux/macOS: `~/.local/bin/reto-cli`, Windows: `%LOCALAPPDATA%\Programs\reto3d\reto-cli.exe`)를 설치하고, 필요한 ONNX Runtime 동적 라이브러리를 자동 설정합니다.

#### 소스 코드에서 직접 빌드 (개발자용)

```bash
git clone https://github.com/phd-eokho/wiggle-3d.git
cd wiggle-3d
cargo build --release
```

### 기본 사용법

#### 단일 스캔 파일 처리 (기본 GIF 출력)
```bash
reto-cli --input samples/film_strip_01.jpg --output output_dir/
```

#### 24-bit TrueColor HEVC MP4 비디오 함께 생성
```bash
reto-cli --input samples/ --output results/ --enable-mp4
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
| `-o, --output <DIR>` | `Path` (필수) | 출력 GIF, MP4 비디오 및 진단 아티팩트가 저장될 대상 디렉터리. |
| `--debug` | `bool` (플래그) | 중간 디버그 시각화 출력 활성화 (RoI 오버레이, 특징점 매칭 벡터, 시차 로그 등). |
| `--gif-delay <MS>` | `u32` (기본값: `100`) | 프레임 간 애니메이션 지연 시간 (밀리초 단위, 100ms = 10 fps). |
| `--no-dither` | `bool` (플래그) | NeuQuant 색상 양자화 시 Floyd-Steinberg 디더링 비활성화. |
| `--enable-mp4` | `bool` (플래그) | GIF와 함께 24-bit TrueColor HEVC (H.265) MP4 비디오 생성 활성화. |
| `--enable-nvenc` | `bool` (플래그) | HEVC MP4 비디오 생성 시 NVIDIA NVENC 하드웨어 가속 강제 사용. |
| `--mp4-loops <COUNT>` | `usize` (기본값: `4`) | MP4 비디오 내 인코딩될 핑퐁 루프 반복 횟수. |
| `--mp4-crf <CRF>` | `u32` (기본값: `18`) | HEVC 비디오 인코딩 품질/압축률 (0–51, 낮을수록 고화질). |
| `--no-progress` | `bool` (플래그) | 대화형 터미널 진행률 표시줄 비활성화 (CI/CD 환경 및 로그 수집에 권장). |
| `--log-file <FILE>` | `Path` (선택) | 상세 실행 및 진단 로그가 기록될 커스텀 로그 파일 경로. |
| `-q, --quiet` | `bool` (플래그) | 치명적 오류를 제외한 모든 콘솔 출력 억제. |
| `-v, -vv` | `Count` (플래그) | 로그 상세 수준 증가 (`-v`: DEBUG, `-vv`: TRACE). |

> [!WARNING]
> **하드웨어 비디오 인코더 지원 현황 안내:**
> - **NVIDIA NVENC (Linux / Windows)**: 실제 GPU 하드웨어에서 완벽하게 테스트 및 검증 완료.
> - **Linux VA-API, macOS VideoToolbox, Windows Media Foundation**: 아키텍처 및 플랫폼 인터페이스가 구현되어 있으나, **실제 하드웨어 실기 테스트는 아직 진행되지 않았습니다 (실험적 단계)**. `--enable-mp4` 옵션 사용 시, CLI는 배치 처리를 시작하기 전에 사용 가능한 하드웨어 백엔드를 사전 검사(pre-flight check)하며 작동 가능한 하드웨어 인코더가 없을 경우 즉시 실패(fail-fast)합니다.

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
