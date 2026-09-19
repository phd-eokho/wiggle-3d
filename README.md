# Wiggle-3D

[English](README.md) | [한국어](README.ko.md)

A high-performance, parallelized Rust engine for splitting 3D multi-lens film camera scans (RETO 3D, Nimslo, Nishika) and generating synchronized, jitter-free Wiggle 3D stereoscopic loop GIFs.

---

## What is Wiggle-3D?

3D film cameras (such as the RETO 3D, Nimslo 3D, and Nishika N8000/N9000) capture a single moment across 3 or 4 half-frame lenses simultaneously on standard 35mm film. When scanned, these multi-lens frames appear side-by-side on a single wide film strip.

To produce a natural stereoscopic "wiggle" animation:
1. Each sub-frame must be located and cropped from the film scan.
2. The focal subject (people, faces, foreground objects) across all frames must be aligned with sub-pixel precision to eliminate disorienting vertical shake and jerky horizontal jumps.
3. The aligned frames are assembled into a smooth ping-pong loop (`0 → 1 → 2 → 1`) with unified color quantization.

**Wiggle-3D** automates this entire workflow into a fast, memory-safe, and parallelized CLI tool and Rust library.

---

## Sample

A raw 3-lens film scan (*left*) automatically partitioned, stabilized with sub-pixel focal locking, and synthesized into a smooth stereoscopic Wiggle 3D loop (*right*):

<p align="center">
  <img src="assets/sample_scan.jpg" alt="Raw 3-Lens Film Strip Scan" width="560" />
  &nbsp;&nbsp;
  <img src="assets/sample_wiggle.gif" alt="Wiggle 3D Animation" width="179" />
</p>

---

## Featured Advantages

- **Fully Parallelized Architecture**: Built with a multi-stage streaming pipeline and multi-core parallel execution (`rayon` + asynchronous file I/O). Processes entire directories of high-resolution film scans at maximum throughput with low, bounded memory usage.
- **Machine Learning-Based Alignment**: Uses deep learning keypoint detection and epipolar motion estimation to track visual features across all lens angles, stabilizing parallax shifts for silky-smooth 3D depth.
- **Intelligent Facial Detection & Focal Locking**: Automatically identifies human subjects and facial landmarks from the user's perspective, locking the stereoscopic focal plane onto faces so the main subject remains sharp, stable, and perfectly anchored.
- **Automated Frame Splitting**: Automatically detects sub-frame boundaries on raw film strips, eliminating tedious manual cropping.
- **Flicker-Free Unified Palette Quantization**: Computes a global 256-color palette across all frames with Floyd-Steinberg dithering to prevent color flashing between loop frames.
- **Diagnostic Visual Overlays**: Generates visual debug artifacts including RoI boundary boxes, facial landmark points, and epipolar match vectors when run with `--debug`.

---

## Quick Start

### Installation

#### 1-Line Standalone Installer (Recommended)

**Linux / macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/phd-eokho/wiggle-3d/main/install.sh | sh
```

For NVIDIA GPU acceleration (CUDA):
```bash
curl -fsSL https://raw.githubusercontent.com/phd-eokho/wiggle-3d/main/install.sh | WIGGLE3D_CUDA=1 sh
```

*To uninstall (Linux / macOS):*
```bash
curl -fsSL https://raw.githubusercontent.com/phd-eokho/wiggle-3d/main/install.sh | sh -s -- --uninstall
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/phd-eokho/wiggle-3d/main/install.ps1 | iex
```

For NVIDIA GPU acceleration (CUDA):
```powershell
$env:WIGGLE3D_CUDA = "1"; irm https://raw.githubusercontent.com/phd-eokho/wiggle-3d/main/install.ps1 | iex
```

*To uninstall (Windows):*
```powershell
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/phd-eokho/wiggle-3d/main/install.ps1))) -Uninstall
```

The installer downloads the pre-built binary (`reto-cli` / `reto-cli.exe`) to `~/.local/bin/` and automatically configures the required ONNX Runtime dynamic libraries.

#### Build from Source (Developers)

```bash
git clone https://github.com/phd-eokho/wiggle-3d.git
cd wiggle-3d
cargo build --release
```

### Basic Usage

#### Process a Single Scan File
```bash
reto-cli --input samples/film_strip_01.jpg --output output_dir/
```

#### Batch Process an Entire Directory
```bash
reto-cli --input /path/to/scans/ --output /path/to/results/
```

#### Run with Debug Visualizations and Diagnostic Logs
```bash
reto-cli --input samples/ --output results/ --debug -v
```

---

## CLI Usage Reference

### Command Syntax

```bash
reto-cli [OPTIONS] --input <PATH> --output <DIR>
```

### Options and Flags

| Option / Flag | Type / Default | Description |
| :--- | :--- | :--- |
| `-i, --input <PATH>` | `Path` (Required) | Path to an individual scan file or a directory of scans. |
| `-o, --output <DIR>` | `Path` (Required) | Destination directory for output GIFs and diagnostic artifacts. |
| `--debug` | `bool` (Flag) | Enables intermediate debug outputs (RoI overlays, matched features, disparity logs). |
| `--gif-delay <MS>` | `u32` (Default: `100`) | Inter-frame animation delay in milliseconds (100ms = 10 fps). |
| `--no-dither` | `bool` (Flag) | Disables Floyd-Steinberg dithering during NeuQuant color quantization. |
| `--no-progress` | `bool` (Flag) | Disables interactive terminal progress bar (recommended for CI/headless logs). |
| `--log-file <FILE>` | `Path` (Optional) | Custom file destination for detailed trace and diagnostic logs. |
| `-q, --quiet` | `bool` (Flag) | Suppresses all console output except fatal errors. |
| `-v, -vv` | `Count` (Flag) | Increases verbosity level (`-v` for DEBUG, `-vv` for TRACE). |

### Supported Image Formats

`reto-cli` automatically scans and processes files matching the following extensions:
- JPEG: `.jpg`, `.jpeg`
- PNG: `.png`
- TIFF: `.tiff`, `.tif`
- Bitmap: `.bmp`
- WebP: `.webp`

---

## Developer Guide

### Architecture

The project is structured into two main crates:

```
wiggle-3d/
├── reto-core/      # Computer vision library and neural inference engine
└── reto-cli/       # CLI front-end, dual-sink logging, and progress reporting
```

- **`reto-core`**: A standalone, zero-CLI-dependency Rust library providing RoI extraction, neural feature and face detection, parallax alignment, and GIF encoding.
- **`reto-cli`**: A lightweight CLI front-end handling argument validation, file discovery, terminal progress visualization (`indicatif`), and dual-sink logging (`tracing-subscriber`).

### Building and Testing

#### Prerequisites
- Rust 1.75+ (2021 Edition)
- Standard C/C++ toolchain (required by ONNX Runtime native bindings)

#### Run All Unit and Integration Tests
```bash
cargo test --all-targets --all-features
```

#### Run Documentation Tests
```bash
cargo test --doc
```

#### Run Linter and Format Checks
```bash
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

---

## License & Third-Party Notice

This project is licensed under the [MIT License](LICENSE).

### Third-Party Neural Models

The neural network model weights used by Wiggle-3D for alignment are loaded at runtime and governed by their respective licenses:
- **SuperPoint**: Copyright (c) Magic Leap, Inc. Licensed under the [Magic Leap Non-Commercial Research License](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE). Use of SuperPoint weights is restricted to non-commercial research and personal evaluation.
- **RetinaFace**: MobileNet0.25 backbone pretrained weights licensed under the MIT License / academic permissive research terms.

For more details, see the [LICENSE](LICENSE) file.
