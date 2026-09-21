#!/usr/bin/env bash
set -euo pipefail

# Download official Microsoft ONNX Runtime release (x86_64 linux, built against glibc 2.28+)
ORT_VERSION="${ORT_VERSION:-1.19.2}"
USE_GPU="${ORT_GPU:-0}"
TARGET_DIR=".cache_reto3d/onnxruntime"

for arg in "$@"; do
    case "$arg" in
        --gpu)
            USE_GPU=1
            ;;
        --cpu)
            USE_GPU=0
            ;;
        *)
            TARGET_DIR="$arg"
            ;;
    esac
done

OS="$(uname -s)"
ARCH="$(uname -m)"

mkdir -p "$TARGET_DIR"

if [ "$OS" = "Darwin" ]; then
    PACKAGE_TYPE="cpu"
    if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
        TAR_NAME="onnxruntime-osx-arm64-${ORT_VERSION}.tgz"
        EXTRACT_DIR="${TARGET_DIR}/onnxruntime-osx-arm64-${ORT_VERSION}"
    else
        TAR_NAME="onnxruntime-osx-x86_64-${ORT_VERSION}.tgz"
        EXTRACT_DIR="${TARGET_DIR}/onnxruntime-osx-x86_64-${ORT_VERSION}"
    fi
    LIB_NAME="libonnxruntime.dylib"
else
    if [ "$ARCH" != "x86_64" ]; then
        echo "Warning: Current architecture is $ARCH. Official release linux-x64 will only work on x86_64." >&2
    fi

    if [ "$USE_GPU" = "1" ]; then
        PACKAGE_TYPE="gpu"
        TAR_NAME="onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz"
        EXTRACT_DIR="${TARGET_DIR}/onnxruntime-linux-x64-gpu-${ORT_VERSION}"
    else
        PACKAGE_TYPE="cpu"
        TAR_NAME="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
        EXTRACT_DIR="${TARGET_DIR}/onnxruntime-linux-x64-${ORT_VERSION}"
    fi
    LIB_NAME="libonnxruntime.so"
fi

DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${TAR_NAME}"
TARGET_TAR="${TARGET_DIR}/${TAR_NAME}"

if [ ! -f "${EXTRACT_DIR}/lib/${LIB_NAME}" ]; then
    echo "Downloading ONNX Runtime (${PACKAGE_TYPE}) v${ORT_VERSION} from ${DOWNLOAD_URL}..."
    curl -fSL --progress-bar "$DOWNLOAD_URL" -o "$TARGET_TAR"
    echo "Extracting ${TAR_NAME} to ${TARGET_DIR}..."
    tar -xzf "$TARGET_TAR" -C "$TARGET_DIR"
    rm -f "$TARGET_TAR"
fi

LIB_PATH="${EXTRACT_DIR}/lib/${LIB_NAME}"
LIB_DIR="$(cd "${EXTRACT_DIR}/lib" && pwd)"
mkdir -p ".cache_reto3d"
ln -sfn "${LIB_DIR}" ".cache_reto3d/lib"

echo "ONNX Runtime (${PACKAGE_TYPE}) shared library ready at: ${LIB_PATH}"
echo "Canonical symlink created at: .cache_reto3d/lib"
echo "To compile with standard dynamic linking, export:"
echo "  export ORT_LIB_LOCATION=\"${LIB_DIR}\""
echo "  export ORT_PREFER_DYNAMIC_LINK=1"
if [ "$OS" = "Darwin" ]; then
    echo "To run binaries with dynamic linking, export:"
    echo "  export DYLD_LIBRARY_PATH=\"${LIB_DIR}:\${DYLD_LIBRARY_PATH:-}\""
    echo "  export DYLD_FALLBACK_LIBRARY_PATH=\"${LIB_DIR}:\${DYLD_FALLBACK_LIBRARY_PATH:-}\""
else
    echo "To run binaries with dynamic linking, export:"
    echo "  export LD_LIBRARY_PATH=\"${LIB_DIR}:\${LD_LIBRARY_PATH:-}\""
fi

if [ -n "${GITHUB_ENV:-}" ]; then
    echo "ORT_LIB_LOCATION=${LIB_DIR}" >> "$GITHUB_ENV"
    echo "ORT_PREFER_DYNAMIC_LINK=1" >> "$GITHUB_ENV"
    if [ "$OS" = "Darwin" ]; then
        echo "DYLD_LIBRARY_PATH=${LIB_DIR}:${DYLD_LIBRARY_PATH:-}" >> "$GITHUB_ENV"
        echo "DYLD_FALLBACK_LIBRARY_PATH=${LIB_DIR}:${DYLD_FALLBACK_LIBRARY_PATH:-}" >> "$GITHUB_ENV"
    else
        echo "LD_LIBRARY_PATH=${LIB_DIR}:${LD_LIBRARY_PATH:-}" >> "$GITHUB_ENV"
    fi
fi
