#!/usr/bin/env sh
# Wiggle-3D Standalone 1-Line Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/phd-eokho/wiggle-3d/main/install.sh | sh

set -eu

REPO="${WIGGLE3D_REPO:-phd-eokho/wiggle-3d}"
INSTALL_DIR="${WIGGLE3D_INSTALL_DIR:-$HOME/.local/bin}"
LIB_DIR="${WIGGLE3D_LIB_DIR:-$(dirname "$INSTALL_DIR")/lib}"
STATE_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/reto3d"
ORT_MANIFEST="${STATE_DIR}/onnxruntime_files.txt"
VERSION="${WIGGLE3D_VERSION:-latest}"

# Standardized ANSI Logging
NC='\033[0m'

log_debug() {
    local cyan='\033[0;36m'
    printf "%b[DEBUG]%b %s\n" "$cyan" "$NC" "$1"
}

log_info() {
    local green='\033[0;32m'
    printf "%b[INFO]%b %s\n" "$green" "$NC" "$1"
}

log_warn() {
    local yellow='\033[1;33m'
    printf "%b[WARN]%b %s\n" "$yellow" "$NC" "$1" >&2
}

log_error() {
    local red='\033[0;31m'
    printf "%b[ERROR]%b %s\n" "$red" "$NC" "$1" >&2
}

# 1. Parse Arguments & Handle Uninstallation
ACTION="${1:-install}"
case "$ACTION" in
    --uninstall|-u|uninstall)
        log_info "Uninstalling Wiggle-3D..."
        if [ -f "${INSTALL_DIR}/reto-cli" ]; then
            rm -f "${INSTALL_DIR}/reto-cli"
            log_info "Removed binary: ${INSTALL_DIR}/reto-cli"
        else
            log_warn "Binary ${INSTALL_DIR}/reto-cli was not found."
        fi

        # Remove ONNX Runtime shared libraries only if provisioned by this script
        if [ -f "$ORT_MANIFEST" ]; then
            xargs rm -f < "$ORT_MANIFEST" 2>/dev/null || true
            rm -f "$ORT_MANIFEST"
            log_info "Removed installer-provisioned ONNX Runtime libraries from ${LIB_DIR}"
        fi

        CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/reto3d"
        if [ -d "$CACHE_DIR" ]; then
            rm -rf "$CACHE_DIR"
            log_info "Cleaned model cache: ${CACHE_DIR}"
        fi

        if [ -d "$STATE_DIR" ]; then
            rm -rf "$STATE_DIR"
        fi

        log_info "Wiggle-3D has been successfully uninstalled."
        exit 0
        ;;
    --help|-h|help)
        printf "Wiggle-3D Standalone Installer / Uninstaller\n\n"
        printf "Usage:\n"
        printf "  install.sh [ACTION]\n\n"
        printf "Actions:\n"
        printf "  install             Install Wiggle-3D CLI (default)\n"
        printf "  uninstall, -u       Uninstall Wiggle-3D binary and model caches\n"
        printf "  --help, -h          Show this help message\n\n"
        printf "Environment Overrides:\n"
        printf "  WIGGLE3D_INSTALL_DIR  Target install directory (default: %s)\n" "$INSTALL_DIR"
        printf "  WIGGLE3D_LIB_DIR      Target library directory (default: %s)\n" "$LIB_DIR"
        printf "  WIGGLE3D_VERSION      Target release version (default: %s)\n" "$VERSION"
        printf "  WIGGLE3D_REPO         GitHub repository (default: %s)\n" "$REPO"
        printf "  WIGGLE3D_CUDA         Force CUDA GPU runtime (1 or true)\n"
        printf "  WIGGLE3D_AUTO_ORT     Auto-confirm ONNX Runtime download (1 or true)\n"
        exit 0
        ;;
esac

# 2. Detect Operating System & Architecture
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux)
        OS_TAG="unknown-linux-gnu"
        ;;
    Darwin)
        OS_TAG="apple-darwin"
        ;;
    *)
        log_error "Unsupported operating system: $OS. Please build from source via cargo."
        exit 1
        ;;
esac

case "$ARCH" in
    x86_64|amd64)
        ARCH_TAG="x86_64"
        ;;
    aarch64|arm64)
        ARCH_TAG="aarch64"
        ;;
    *)
        log_error "Unsupported architecture: $ARCH. Please build from source via cargo."
        exit 1
        ;;
esac

TARGET_TRIPLE="${ARCH_TAG}-${OS_TAG}"

log_info "Detected target platform: ${TARGET_TRIPLE}"

# 3. Resolve Release Tag
if [ "$VERSION" = "latest" ]; then
    RELEASE_TAG="$(curl -sSL "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name":' | head -n 1 | sed -E 's/.*"([^"]+)".*/\1/' || true)"
    if [ -z "$RELEASE_TAG" ]; then
        log_warn "Could not fetch latest release tag via GitHub API; defaulting to v0.1.0."
        RELEASE_TAG="v0.1.0"
    fi
else
    RELEASE_TAG="$VERSION"
fi

TAR_NAME="wiggle-3d-${RELEASE_TAG}-${TARGET_TRIPLE}.tar.gz"
DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${TAR_NAME}"
CHECKSUM_URL="${DOWNLOAD_URL}.sha256"

# 4. Create Temporary Workspace
TMP_DIR="$(mktemp -d 2>/dev/null || mktemp -d -t 'wiggle3d')"
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT INT TERM

log_info "Downloading Wiggle-3D (${RELEASE_TAG}) from ${DOWNLOAD_URL}..."

if ! curl -fSL --progress-bar "$DOWNLOAD_URL" -o "${TMP_DIR}/${TAR_NAME}"; then
    log_error "Failed to download release archive from ${DOWNLOAD_URL}"
    log_error "Please check available releases at https://github.com/${REPO}/releases"
    exit 1
fi

# Optional Checksum verification
if curl -fsSL "$CHECKSUM_URL" -o "${TMP_DIR}/${TAR_NAME}.sha256" 2>/dev/null; then
    log_info "Verifying SHA-256 checksum..."
    (
        cd "$TMP_DIR"
        if command -v sha256sum >/dev/null 2>&1; then
            sha256sum -c "${TAR_NAME}.sha256" >/dev/null 2>&1 || {
                log_error "SHA-256 checksum verification failed!"
                exit 1
            }
        elif command -v shasum >/dev/null 2>&1; then
            shasum -a 256 -c "${TAR_NAME}.sha256" >/dev/null 2>&1 || {
                log_error "SHA-256 checksum verification failed!"
                exit 1
            }
        fi
    )
fi

# 5. Extract Archive
tar -xzf "${TMP_DIR}/${TAR_NAME}" -C "$TMP_DIR"

BIN_SRC="$(find "$TMP_DIR" -maxdepth 2 -type f -name "reto-cli" | head -n 1)"
[ -n "$BIN_SRC" ] && [ -f "$BIN_SRC" ] || { log_error "Binary 'reto-cli' was not found in the downloaded archive."; exit 1; }

# 6. Install Binary
mkdir -p "$INSTALL_DIR"
mv "$BIN_SRC" "${INSTALL_DIR}/reto-cli"
chmod +x "${INSTALL_DIR}/reto-cli"

log_info "Successfully installed 'reto-cli' to ${INSTALL_DIR}/reto-cli"

# 7. Check PATH
case ":$PATH:" in
    *":${INSTALL_DIR}:"*)
        ;;
    *)
        log_warn "${INSTALL_DIR} is not in your PATH."
        printf "  Add the following line to your shell configuration (~/.bashrc, ~/.zshrc):\n\n"
        printf "    export PATH=\"%s:\$PATH\"\n\n" "$INSTALL_DIR"
        ;;
esac

# 8. Verify ONNX Runtime Dependency & Interactive Setup
ONNX_VERSION="1.19.2"
if ! "${INSTALL_DIR}/reto-cli" --help >/dev/null 2>&1; then
    log_warn "ONNX Runtime (v${ONNX_VERSION}) was not detected on your system."
    printf "  'reto-cli' requires ONNX Runtime dynamic libraries for neural vision alignment.\n"
    printf "  Because RPATH is configured to \$ORIGIN/../lib, placing the libraries in '%s' satisfies this dependency.\n\n" "$LIB_DIR"

    # Detect CUDA GPU availability
    HAS_CUDA=false
    if command -v nvidia-smi >/dev/null 2>&1 || [ -e /dev/nvidia0 ] || [ -d /usr/local/cuda ]; then
        HAS_CUDA=true
    fi

    case "$OS" in
        Linux)
            case "$ARCH" in
                x86_64|amd64)
                    if [ "${WIGGLE3D_CUDA:-}" = "1" ] || [ "${WIGGLE3D_CUDA:-}" = "true" ] || { [ -z "${WIGGLE3D_CUDA:-}" ] && [ "$HAS_CUDA" = "true" ]; }; then
                        ORT_TAR="onnxruntime-linux-x64-gpu-${ONNX_VERSION}.tgz"
                        log_info "Detected NVIDIA CUDA GPU. Selected runtime: GPU package (${ORT_TAR})"
                    else
                        ORT_TAR="onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
                        log_info "Selected runtime: CPU package (${ORT_TAR})"
                    fi
                    ;;
                aarch64|arm64)
                    ORT_TAR="onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz"
                    log_info "Selected runtime: CPU package (${ORT_TAR})"
                    ;;
                *)
                    ORT_TAR="onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
                    ;;
            esac
            ;;
        Darwin)
            ORT_TAR="onnxruntime-osx-universal2-${ONNX_VERSION}.tgz"
            log_info "Selected runtime: macOS Universal package (${ORT_TAR})"
            ;;
    esac

    ORT_DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${ORT_TAR}"

    # Prompt user for automatic download if interactive or auto-confirmed
    DO_DOWNLOAD=false
    if [ "${WIGGLE3D_AUTO_ORT:-}" = "1" ] || [ "${WIGGLE3D_AUTO_ORT:-}" = "true" ]; then
        DO_DOWNLOAD=true
    elif [ -r /dev/tty ]; then
        printf "\n  Would you like to download and install ONNX Runtime now? [Y/n]: "
        read -r REPLY </dev/tty || REPLY="n"
        case "$REPLY" in
            [yY][eE][sS]|[yY]|"")
                DO_DOWNLOAD=true
                ;;
            *)
                DO_DOWNLOAD=false
                ;;
        esac
    fi

    if [ "$DO_DOWNLOAD" = "true" ]; then
        log_info "Downloading ONNX Runtime v${ONNX_VERSION} from ${ORT_DOWNLOAD_URL}..."
        ORT_TMP="${TMP_DIR}/ort"
        mkdir -p "$ORT_TMP"
        mkdir -p "$LIB_DIR"
        mkdir -p "$STATE_DIR"
        if curl -fSL --progress-bar "$ORT_DOWNLOAD_URL" -o "${TMP_DIR}/${ORT_TAR}"; then
            tar -xzf "${TMP_DIR}/${ORT_TAR}" -C "$ORT_TMP"
            : > "$ORT_MANIFEST"
            case "$OS" in
                Linux)
                    for f in "${ORT_TMP}"/onnxruntime-*/lib/libonnxruntime*; do
                        base_name="$(basename "$f")"
                        cp -P "$f" "${LIB_DIR}/"
                        echo "${LIB_DIR}/${base_name}" >> "$ORT_MANIFEST"
                    done
                    ;;
                Darwin)
                    for f in "${ORT_TMP}"/onnxruntime-*/lib/libonnxruntime*.dylib; do
                        base_name="$(basename "$f")"
                        cp -P "$f" "${LIB_DIR}/"
                        echo "${LIB_DIR}/${base_name}" >> "$ORT_MANIFEST"
                    done
                    ;;
            esac
            log_info "Successfully installed ONNX Runtime libraries to ${LIB_DIR}."
        else
            log_error "Failed to download ONNX Runtime."
        fi
    else
        printf "\n  Manual Installation Command:\n"
        case "$OS" in
            Linux)
                printf "    mkdir -p \"%s\" && curl -fsSL \"%s\" | tar -xz -C /tmp && cp -P /tmp/onnxruntime-*/lib/libonnxruntime* \"%s/\" && rm -rf /tmp/onnxruntime-*\n\n" \
                    "$LIB_DIR" "$ORT_DOWNLOAD_URL" "$LIB_DIR"
                ;;
            Darwin)
                printf "    mkdir -p \"%s\" && curl -fsSL \"%s\" | tar -xz -C /tmp && cp -P /tmp/onnxruntime-*/lib/libonnxruntime*.dylib \"%s/\" && rm -rf /tmp/onnxruntime-*\n\n" \
                    "$LIB_DIR" "$ORT_DOWNLOAD_URL" "$LIB_DIR"
                ;;
        esac
    fi
fi

log_info "Wiggle-3D installation completed! Run 'reto-cli --help' to get started."
