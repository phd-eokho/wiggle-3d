//! NVIDIA NVENC hardware-accelerated HEVC (H.265) video frame encoder.
//!
//! Dynamically loads `libnvidia-encode.so.1` / `nvEncodeAPI64.dll` and `libcuda.so.1` / `nvcuda.dll`
//! at runtime to provide maximum hardware throughput on NVIDIA Turing, Ampere, Ada Lovelace, and Blackwell GPUs.

#![allow(
    unsafe_code,
    non_snake_case,
    non_camel_case_types,
    clippy::used_underscore_binding,
    clippy::manual_c_str_literals,
    clippy::cast_possible_truncation,
    clippy::cast_ptr_alignment,
    clippy::unreadable_literal,
    clippy::doc_markdown,
    clippy::borrow_as_ptr,
    clippy::ptr_as_ptr,
    clippy::cast_possible_wrap,
    clippy::missing_const_for_fn,
    clippy::upper_case_acronyms,
    clippy::items_after_statements,
    clippy::if_not_else,
    clippy::missing_transmute_annotations,
    clippy::non_send_fields_in_send_ty
)]

use super::mp4_muxer::{parse_annex_b_nalus, HevcNalUnit};
use super::{HevcEncoderConfig, HevcFrameEncoder, VideoError};
use crate::color::Yuv420PlanarFrame;
use std::ffi::{c_char, c_void, CStr, CString};

type NVENCSTATUS = u32;
const NV_ENC_SUCCESS: NVENCSTATUS = 0;

/// Windows/COM-style 128-bit globally unique identifier for NVENC presets and codecs.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GUID {
    /// Low 32 bits of the GUID.
    pub data1: u32,
    /// Next 16 bits.
    pub data2: u16,
    /// Next 16 bits.
    pub data3: u16,
    /// High 64 bits.
    pub data4: [u8; 8],
}

/// GUID for the HEVC (H.265) video codec in NVENC.
pub const NV_ENC_CODEC_HEVC_GUID: GUID = GUID {
    data1: 0x790cdc88,
    data2: 0x4522,
    data3: 0x4d7b,
    data4: [0x94, 0x25, 0xbd, 0xa9, 0x97, 0x5f, 0x76, 0x03],
};

/// GUID for H.264 video codec.
pub const NV_ENC_CODEC_H264_GUID: GUID = GUID {
    data1: 0x6bc82762,
    data2: 0x4e63,
    data3: 0x4ca4,
    data4: [0xaa, 0x85, 0x1e, 0x50, 0xf3, 0x21, 0xf6, 0xbf],
};

/// GUID for the P7 preset (highest visual quality).
pub const NV_ENC_PRESET_P7_GUID: GUID = GUID {
    data1: 0x84848c12,
    data2: 0x6f71,
    data3: 0x4c13,
    data4: [0x93, 0x1b, 0x53, 0xe2, 0x83, 0xf5, 0x79, 0x74],
};

/// GUID for the P4 preset (medium / balanced).
pub const NV_ENC_PRESET_P4_GUID: GUID = GUID {
    data1: 0x90a7b826,
    data2: 0xdf06,
    data3: 0x4862,
    data4: [0xb9, 0xd2, 0xcd, 0x6d, 0x73, 0xa0, 0x86, 0x81],
};

/// GUID for the default NVENC preset.
pub const NV_ENC_PRESET_DEFAULT_GUID: GUID = GUID {
    data1: 0x771966a7,
    data2: 0x7464,
    data3: 0x4e63,
    data4: [0x8e, 0x60, 0x80, 0xab, 0x55, 0x4a, 0x69, 0xc3],
};

/// Buffer format constant for Planar YUV420 (IYUV).
pub const NV_ENC_BUFFER_FORMAT_IYUV: u32 = 0x00000100;
/// Buffer format constant for Semi-Planar NV12.
pub const NV_ENC_BUFFER_FORMAT_NV12: u32 = 0x00000001;
/// Device type for CUDA context.
pub const NV_ENC_DEVICE_TYPE_CUDA: u32 = 1;

/// Tuning info for high visual quality.
pub const NV_ENC_TUNING_INFO_HIGH_QUALITY: u32 = 1;

#[repr(C)]
struct NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS {
    version: u32,
    deviceType: u32,
    device: *mut c_void,
    customExtension: *mut c_void,
    apiVersion: u32,
    reserved1: [u32; 253],
    reserved2: [*mut c_void; 64],
}

#[repr(C)]
struct NV_ENC_INITIALIZE_PARAMS {
    version: u32,
    encodeGUID: GUID,
    presetGUID: GUID,
    encodeWidth: u32,
    encodeHeight: u32,
    darWidth: u32,
    darHeight: u32,
    frameRateNum: u32,
    frameRateDen: u32,
    enableEncodeAsync: u32,
    enablePTD: u32,
    bitfields: u32,
    privDataSize: u32,
    reserved: u32,
    privData: *mut c_void,
    encodeConfig: *mut c_void,
    maxEncodeWidth: u32,
    maxEncodeHeight: u32,
    maxMEHintCountsPerBlock: [u32; 8], // 32 bytes
    tuningInfo: u32,
    bufferFormat: u32,
    numStateBuffers: u32,
    outputStatsLevel: u32,
    reserved1: [u32; 284],
    reserved2: [*mut c_void; 64],
}

#[repr(C)]
struct NV_ENC_CONFIG {
    version: u32,
    profileGUID: GUID,
    gopLength: u32,
    frameIntervalP: i32,
    monoChromeEncoding: u32,
    frameFieldMode: u32,
    mvPrecision: u32,
    _pad_middle: [u8; 128],
    _hevc_level: u32,
    _hevc_tier: u32,
    _hevc_min_cu: u32,
    _hevc_max_cu: u32,
    _hevc_intra_pred: u32,
    hevc_idr_period: u32,
    _pad_end: [u8; 3392],
}

#[repr(C)]
struct NV_ENC_PRESET_CONFIG {
    version: u32,
    _pad: u32,
    presetCfg: NV_ENC_CONFIG,
    reserved1: [u32; 255],
    reserved2: [*mut c_void; 64],
}

#[repr(C)]
struct NV_ENC_CREATE_INPUT_BUFFER {
    version: u32,
    width: u32,
    height: u32,
    memoryHeap: u32,
    bufferFmt: u32,
    reserved: u32,
    inputBuffer: *mut c_void,
    pSysMemBuffer: *mut c_void,
    reserved1: [u32; 58],
    reserved2: [*mut c_void; 63],
}

#[repr(C)]
struct NV_ENC_CREATE_BITSTREAM_BUFFER {
    version: u32,
    size: u32,
    memoryHeap: u32,
    reserved: u32,
    bitstreamBuffer: *mut c_void,
    bitstreamBufferPtr: *mut c_void,
    reserved1: [u32; 58],
    reserved2: [*mut c_void; 64],
}

#[repr(C)]
struct NV_ENC_LOCK_INPUT_BUFFER {
    version: u32,
    doNotWait: u32,
    inputBuffer: *mut c_void,
    bufferDataPtr: *mut c_void,
    pitch: u32,
    reserved1: [u32; 251],
    reserved2: [*mut c_void; 64],
}

#[repr(C)]
struct NV_ENC_LOCK_BITSTREAM {
    version: u32,
    bitfields: u32,
    outputBitstream: *mut c_void,
    sliceOffsets: *mut u32,
    frameIdx: u32,
    hwEncodeStatus: u32,
    numSlices: u32,
    bitstreamSizeInBytes: u32,
    outputTimeStamp: u64,
    outputDuration: u64,
    bitstreamBufferPtr: *mut c_void,
    pictureType: u32,
    pictureStruct: u32,
    frameAvgQP: u32,
    frameSatd: u32,
    ltrFrameIdx: u32,
    ltrFrameBitmap: u32,
    temporalId: u32,
    intraMBCount: u32,
    interMBCount: u32,
    averageMVX: i32,
    averageMVY: i32,
    alphaLayerSizeInBytes: u32,
    outputStatsPtrSize: u32,
    reserved: u32,
    outputStatsPtr: *mut c_void,
    frameIdxDisplay: u32,
    reserved1: [u32; 219],
    reserved2: [*mut c_void; 63],
    reservedInternal: [u32; 8],
}

#[repr(C)]
struct NV_ENC_PIC_PARAMS {
    version: u32,
    inputWidth: u32,
    inputHeight: u32,
    inputPitch: u32,
    encodePicFlags: u32,
    frameIdx: u32,
    inputTimeStamp: u64,
    inputDuration: u64,
    inputBuffer: *mut c_void,
    outputBitstream: *mut c_void,
    completionEvent: *mut c_void,
    bufferFmt: u32,
    pictureStruct: u32,
    pictureType: u32,
    codecPicParams: [u8; 1040],
    meHintCountsPerBlock: [u32; 8],
    meExternalHints: *mut c_void,
    reserved2: [u32; 7],
    reserved5: [*mut c_void; 2],
    qpDeltaMap: *mut i8,
    qpDeltaMapSize: u32,
    reservedBitFields: u32,
    meHintRefPicDist: [u16; 2],
    diffPicNumHint: i32,
    alphaBuffer: *mut c_void,
    meExternalSbHints: *mut c_void,
    meSbHintsCount: u32,
    stateBufferIdx: u32,
    outputReconBuffer: *mut c_void,
    reserved3: [u32; 284],
    reserved6: [*mut c_void; 57],
}

#[repr(C)]
struct NV_ENCODE_API_FUNCTION_LIST {
    version: u32,
    reserved: u32,
    nvEncOpenEncodeSession:
        Option<unsafe extern "C" fn(*mut c_void, u32, *mut *mut c_void) -> NVENCSTATUS>,
    nvEncGetEncodeGUIDCount: Option<unsafe extern "C" fn(*mut c_void, *mut u32) -> NVENCSTATUS>,
    nvEncGetEncodeProfileGUIDCount:
        Option<unsafe extern "C" fn(*mut c_void, GUID, *mut u32) -> NVENCSTATUS>,
    nvEncGetEncodeProfileGUIDs:
        Option<unsafe extern "C" fn(*mut c_void, GUID, *mut GUID, u32, *mut u32) -> NVENCSTATUS>,
    nvEncGetEncodeGUIDs:
        Option<unsafe extern "C" fn(*mut c_void, *mut GUID, u32, *mut u32) -> NVENCSTATUS>,
    nvEncGetInputFormatCount:
        Option<unsafe extern "C" fn(*mut c_void, GUID, *mut u32) -> NVENCSTATUS>,
    nvEncGetInputFormats:
        Option<unsafe extern "C" fn(*mut c_void, GUID, *mut u32, u32, *mut u32) -> NVENCSTATUS>,
    nvEncGetEncodeCaps: *mut c_void,
    nvEncGetEncodePresetCount:
        Option<unsafe extern "C" fn(*mut c_void, GUID, *mut u32) -> NVENCSTATUS>,
    nvEncGetEncodePresetGUIDs:
        Option<unsafe extern "C" fn(*mut c_void, GUID, *mut GUID, u32, *mut u32) -> NVENCSTATUS>,
    nvEncGetEncodePresetConfig:
        Option<unsafe extern "C" fn(*mut c_void, GUID, GUID, *mut c_void) -> NVENCSTATUS>,
    nvEncInitializeEncoder:
        Option<unsafe extern "C" fn(*mut c_void, *mut NV_ENC_INITIALIZE_PARAMS) -> NVENCSTATUS>,
    nvEncCreateInputBuffer:
        Option<unsafe extern "C" fn(*mut c_void, *mut NV_ENC_CREATE_INPUT_BUFFER) -> NVENCSTATUS>,
    nvEncDestroyInputBuffer: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    nvEncCreateBitstreamBuffer: Option<
        unsafe extern "C" fn(*mut c_void, *mut NV_ENC_CREATE_BITSTREAM_BUFFER) -> NVENCSTATUS,
    >,
    nvEncDestroyBitstreamBuffer:
        Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    nvEncEncodePicture:
        Option<unsafe extern "C" fn(*mut c_void, *mut NV_ENC_PIC_PARAMS) -> NVENCSTATUS>,
    nvEncLockBitstream:
        Option<unsafe extern "C" fn(*mut c_void, *mut NV_ENC_LOCK_BITSTREAM) -> NVENCSTATUS>,
    nvEncUnlockBitstream: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    nvEncLockInputBuffer:
        Option<unsafe extern "C" fn(*mut c_void, *mut NV_ENC_LOCK_INPUT_BUFFER) -> NVENCSTATUS>,
    nvEncUnlockInputBuffer: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    nvEncGetEncodeStats: *mut c_void,
    nvEncGetSequenceParams: *mut c_void,
    nvEncRegisterAsyncEvent: *mut c_void,
    nvEncUnregisterAsyncEvent: *mut c_void,
    nvEncMapInputResource: *mut c_void,
    nvEncUnmapInputResource: *mut c_void,
    nvEncDestroyEncoder: Option<unsafe extern "C" fn(*mut c_void) -> NVENCSTATUS>,
    nvEncInvalidateRefFrames: *mut c_void,
    nvEncOpenEncodeSessionEx: Option<
        unsafe extern "C" fn(
            *mut NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS,
            *mut *mut c_void,
        ) -> NVENCSTATUS,
    >,
    nvEncRegisterResource: *mut c_void,
    nvEncUnregisterResource: *mut c_void,
    nvEncReconfigureEncoder: *mut c_void,
    reserved1: *mut c_void,
    nvEncCreateMVBuffer: *mut c_void,
    nvEncDestroyMVBuffer: *mut c_void,
    nvEncRunMotionEstimationOnly: *mut c_void,
    nvEncGetLastErrorString: Option<unsafe extern "C" fn(*mut c_void) -> *const c_char>,
    nvEncSetIOCudaStreams: *mut c_void,
    nvEncGetEncodePresetConfigEx:
        Option<unsafe extern "C" fn(*mut c_void, GUID, GUID, u32, *mut c_void) -> NVENCSTATUS>,
    nvEncGetSequenceParamEx: *mut c_void,
    nvEncRestoreEncoderState: *mut c_void,
    nvEncLookaheadPicture: *mut c_void,
    reserved2: [*mut c_void; 275],
}

type NvEncodeAPICreateInstanceFn =
    unsafe extern "C" fn(*mut NV_ENCODE_API_FUNCTION_LIST) -> NVENCSTATUS;
type NvEncodeAPIGetMaxSupportedVersionFn = unsafe extern "C" fn(*mut u32) -> NVENCSTATUS;

type CuInitFn = unsafe extern "C" fn(u32) -> u32;
type CuDeviceGetFn = unsafe extern "C" fn(*mut i32, i32) -> u32;
type CuCtxCreateV2Fn = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> u32;
type CuCtxDestroyV2Fn = unsafe extern "C" fn(*mut c_void) -> u32;

/// Dynamic loader for CUDA Driver API and NVENC API.
struct DynamicNvencDriver {
    _cuda_lib: *mut c_void,
    _nvenc_lib: *mut c_void,
    cu_ctx: *mut c_void,
    cu_ctx_destroy: Option<CuCtxDestroyV2Fn>,
    funcs: NV_ENCODE_API_FUNCTION_LIST,
    api_version: u32,
}

#[cfg(target_os = "windows")]
extern "system" {
    fn LoadLibraryA(lpLibFileName: *const c_char) -> *mut c_void;
    fn GetProcAddress(hModule: *mut c_void, lpProcName: *const c_char) -> *mut c_void;
    fn FreeLibrary(hLibModule: *mut c_void) -> std::ffi::c_int;
}

#[cfg(not(target_os = "windows"))]
unsafe fn dyn_load_lib(name: *const c_char) -> *mut c_void {
    unsafe { libc::dlopen(name, libc::RTLD_NOW | libc::RTLD_LOCAL) }
}

#[cfg(not(target_os = "windows"))]
unsafe fn dyn_load_sym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void {
    unsafe { libc::dlsym(handle, symbol) }
}

#[cfg(not(target_os = "windows"))]
unsafe fn dyn_close_lib(handle: *mut c_void) {
    if !handle.is_null() {
        unsafe {
            libc::dlclose(handle);
        }
    }
}

#[cfg(target_os = "windows")]
unsafe fn dyn_load_lib(name: *const c_char) -> *mut c_void {
    unsafe { LoadLibraryA(name) }
}

#[cfg(target_os = "windows")]
unsafe fn dyn_load_sym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void {
    unsafe { GetProcAddress(handle, symbol) }
}

#[cfg(target_os = "windows")]
unsafe fn dyn_close_lib(handle: *mut c_void) {
    if !handle.is_null() {
        unsafe {
            FreeLibrary(handle);
        }
    }
}

impl DynamicNvencDriver {
    #[allow(clippy::too_many_lines)]
    fn load() -> Result<Self, VideoError> {
        unsafe {
            // 1. Load CUDA Driver library
            let cuda_names = [
                #[cfg(target_os = "windows")]
                "nvcuda.dll",
                #[cfg(not(target_os = "windows"))]
                "libcuda.so.1",
                #[cfg(not(target_os = "windows"))]
                "libcuda.so",
            ];

            let mut cuda_lib = std::ptr::null_mut();
            for name in cuda_names {
                let cname =
                    CString::new(name).map_err(|e| VideoError::NvencUnavailable(e.to_string()))?;
                let handle = dyn_load_lib(cname.as_ptr());
                if !handle.is_null() {
                    cuda_lib = handle;
                    break;
                }
            }

            if cuda_lib.is_null() {
                return Err(VideoError::NvencUnavailable(
                    "CUDA driver library (libcuda.so.1 / nvcuda.dll) not found".into(),
                ));
            }

            // Load CUDA symbols
            let sym_init = dyn_load_sym(cuda_lib, b"cuInit\0".as_ptr().cast());
            let sym_dev_get = dyn_load_sym(cuda_lib, b"cuDeviceGet\0".as_ptr().cast());
            let sym_ctx_create = dyn_load_sym(cuda_lib, b"cuCtxCreate_v2\0".as_ptr().cast());
            let sym_ctx_create_fallback = if sym_ctx_create.is_null() {
                dyn_load_sym(cuda_lib, b"cuCtxCreate\0".as_ptr().cast())
            } else {
                sym_ctx_create
            };
            let sym_ctx_destroy = dyn_load_sym(cuda_lib, b"cuCtxDestroy_v2\0".as_ptr().cast());
            let sym_ctx_destroy_fallback = if sym_ctx_destroy.is_null() {
                dyn_load_sym(cuda_lib, b"cuCtxDestroy\0".as_ptr().cast())
            } else {
                sym_ctx_destroy
            };

            if sym_init.is_null() || sym_dev_get.is_null() || sym_ctx_create_fallback.is_null() {
                dyn_close_lib(cuda_lib);
                return Err(VideoError::NvencUnavailable(
                    "Missing required CUDA entrypoint symbols".into(),
                ));
            }

            let cu_init: CuInitFn = std::mem::transmute(sym_init);
            let cu_dev_get: CuDeviceGetFn = std::mem::transmute(sym_dev_get);
            let cu_ctx_create: CuCtxCreateV2Fn = std::mem::transmute(sym_ctx_create_fallback);
            let cu_ctx_destroy: Option<CuCtxDestroyV2Fn> = if !sym_ctx_destroy_fallback.is_null() {
                Some(std::mem::transmute(sym_ctx_destroy_fallback))
            } else {
                None
            };

            if cu_init(0) != 0 {
                dyn_close_lib(cuda_lib);
                return Err(VideoError::NvencUnavailable("cuInit(0) failed".into()));
            }

            let mut device: i32 = 0;
            if cu_dev_get(&mut device, 0) != 0 {
                dyn_close_lib(cuda_lib);
                return Err(VideoError::NvencUnavailable(
                    "cuDeviceGet(&dev, 0) failed".into(),
                ));
            }

            let sym_ctx_set = dyn_load_sym(cuda_lib, b"cuCtxSetCurrent\0".as_ptr().cast());

            let mut cu_ctx: *mut c_void = std::ptr::null_mut();
            if cu_ctx_create(&mut cu_ctx, 0, device) != 0 || cu_ctx.is_null() {
                dyn_close_lib(cuda_lib);
                return Err(VideoError::NvencUnavailable("cuCtxCreate failed".into()));
            }

            if !sym_ctx_set.is_null() {
                let cu_ctx_set: unsafe extern "C" fn(*mut c_void) -> u32 =
                    std::mem::transmute(sym_ctx_set);
                let _ = cu_ctx_set(cu_ctx);
            }

            // 2. Load NVENC library
            let nvenc_names = [
                #[cfg(target_os = "windows")]
                "nvEncodeAPI64.dll",
                #[cfg(not(target_os = "windows"))]
                "libnvidia-encode.so.1",
                #[cfg(not(target_os = "windows"))]
                "libnvidia-encode.so",
            ];

            let mut nvenc_lib = std::ptr::null_mut();
            for name in nvenc_names {
                let cname =
                    CString::new(name).map_err(|e| VideoError::NvencUnavailable(e.to_string()))?;
                let handle = dyn_load_lib(cname.as_ptr());
                if !handle.is_null() {
                    nvenc_lib = handle;
                    break;
                }
            }

            if nvenc_lib.is_null() {
                if let Some(destroy) = cu_ctx_destroy {
                    destroy(cu_ctx);
                }
                dyn_close_lib(cuda_lib);
                return Err(VideoError::NvencUnavailable(
                    "NVIDIA NVENC library (libnvidia-encode.so.1 / nvEncodeAPI64.dll) not found"
                        .into(),
                ));
            }

            let create_instance_sym =
                dyn_load_sym(nvenc_lib, b"NvEncodeAPICreateInstance\0".as_ptr().cast());
            let get_max_ver_sym = dyn_load_sym(
                nvenc_lib,
                b"NvEncodeAPIGetMaxSupportedVersion\0".as_ptr().cast(),
            );

            if create_instance_sym.is_null() {
                if let Some(destroy) = cu_ctx_destroy {
                    destroy(cu_ctx);
                }
                dyn_close_lib(nvenc_lib);
                dyn_close_lib(cuda_lib);
                return Err(VideoError::NvencUnavailable(
                    "NvEncodeAPICreateInstance symbol not found".into(),
                ));
            }

            let mut max_ver: u32 = 0;
            if !get_max_ver_sym.is_null() {
                let get_max_ver: NvEncodeAPIGetMaxSupportedVersionFn =
                    std::mem::transmute(get_max_ver_sym);
                let _ = get_max_ver(&mut max_ver);
            }

            let create_instance: NvEncodeAPICreateInstanceFn =
                std::mem::transmute(create_instance_sym);
            let mut funcs: NV_ENCODE_API_FUNCTION_LIST = std::mem::zeroed();

            const fn make_nvenc_struct_ver(major: u32, minor: u32, ver: u32) -> u32 {
                let api_ver = major | (minor << 24);
                api_ver | (ver << 16) | (0x7 << 28)
            }

            let test_api_pairs = [
                (13, 0),
                (13, 1),
                (12, 2),
                (12, 1),
                (12, 0),
                (11, 1),
                (11, 0),
                (max_ver >> 4, max_ver & 0xF),
                (max_ver & 0xFF, (max_ver >> 24) & 0xFF),
            ];

            let mut success = false;
            let mut last_status = 15;
            let mut active_major = 13;
            let mut active_minor = 0;

            for &(maj, min) in &test_api_pairs {
                funcs.version = make_nvenc_struct_ver(maj, min, 2);
                let status = create_instance(&mut funcs);
                if status == NV_ENC_SUCCESS {
                    success = true;
                    active_major = maj;
                    active_minor = min;
                    break;
                }
                last_status = status;
            }

            if !success {
                if let Some(destroy) = cu_ctx_destroy {
                    destroy(cu_ctx);
                }
                dyn_close_lib(nvenc_lib);
                dyn_close_lib(cuda_lib);
                return Err(VideoError::NvencError(format!(
                    "NvEncodeAPICreateInstance returned error code: {last_status}"
                )));
            }

            Ok(Self {
                _cuda_lib: cuda_lib,
                _nvenc_lib: nvenc_lib,
                cu_ctx,
                cu_ctx_destroy,
                funcs,
                api_version: active_major | (active_minor << 24),
            })
        }
    }

    fn get_last_error(&self, encoder: *mut c_void) -> String {
        if let Some(get_err) = self.funcs.nvEncGetLastErrorString {
            unsafe {
                let ptr = get_err(encoder);
                if !ptr.is_null() {
                    return CStr::from_ptr(ptr).to_string_lossy().into_owned();
                }
            }
        }
        "Unknown NVENC error".into()
    }
}

impl Drop for DynamicNvencDriver {
    fn drop(&mut self) {
        unsafe {
            if let Some(destroy) = self.cu_ctx_destroy {
                if !self.cu_ctx.is_null() {
                    destroy(self.cu_ctx);
                }
            }
            if !self._nvenc_lib.is_null() {
                dyn_close_lib(self._nvenc_lib);
            }
            if !self._cuda_lib.is_null() {
                dyn_close_lib(self._cuda_lib);
            }
        }
    }
}

/// Hardware NVENC HEVC video frame encoder.
pub struct NvencHevcEncoder {
    driver: Option<DynamicNvencDriver>,
    encoder: *mut c_void,
    config: Option<HevcEncoderConfig>,
    input_buffer: *mut c_void,
    bitstream_buffer: *mut c_void,
    frame_index: u32,
}

unsafe impl Send for NvencHevcEncoder {}

impl NvencHevcEncoder {
    /// Creates a new uninitialized `NvencHevcEncoder`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            driver: None,
            encoder: std::ptr::null_mut(),
            config: None,
            input_buffer: std::ptr::null_mut(),
            bitstream_buffer: std::ptr::null_mut(),
            frame_index: 0,
        }
    }
}

impl Default for NvencHevcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl HevcFrameEncoder for NvencHevcEncoder {
    #[allow(clippy::too_many_lines)]
    fn initialize(&mut self, config: &HevcEncoderConfig) -> Result<(), VideoError> {
        let driver = DynamicNvencDriver::load()?;

        unsafe {
            let open_session_ex = driver.funcs.nvEncOpenEncodeSessionEx.ok_or_else(|| {
                VideoError::NvencUnavailable(
                    "nvEncOpenEncodeSessionEx function pointer missing".into(),
                )
            })?;

            let struct_ver = |ver: u32| -> u32 { driver.api_version | (ver << 16) | (0x7 << 28) };

            let mut session_params: NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS = std::mem::zeroed();
            session_params.version = struct_ver(1);
            session_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
            session_params.device = driver.cu_ctx;
            session_params.apiVersion = driver.api_version;

            let mut encoder_ptr: *mut c_void = std::ptr::null_mut();
            let mut session_status = open_session_ex(&mut session_params, &mut encoder_ptr);

            if session_status != NV_ENC_SUCCESS || encoder_ptr.is_null() {
                if let Some(open_session) = driver.funcs.nvEncOpenEncodeSession {
                    encoder_ptr = std::ptr::null_mut();
                    let status =
                        open_session(driver.cu_ctx, NV_ENC_DEVICE_TYPE_CUDA, &mut encoder_ptr);
                    if status == NV_ENC_SUCCESS && !encoder_ptr.is_null() {
                        session_status = NV_ENC_SUCCESS;
                    }
                }
            }

            if session_status != NV_ENC_SUCCESS || encoder_ptr.is_null() {
                return Err(VideoError::NvencError(format!(
                    "nvEncOpenEncodeSessionEx failed with status: {session_status}"
                )));
            }

            // Query preset GUIDs for HEVC
            let mut preset_count = 0u32;
            let mut selected_preset = NV_ENC_PRESET_P7_GUID;
            if let Some(get_preset_count) = driver.funcs.nvEncGetEncodePresetCount {
                let status =
                    get_preset_count(encoder_ptr, NV_ENC_CODEC_HEVC_GUID, &mut preset_count);
                if status == NV_ENC_SUCCESS && preset_count > 0 {
                    let mut presets = vec![
                        GUID {
                            data1: 0,
                            data2: 0,
                            data3: 0,
                            data4: [0; 8]
                        };
                        preset_count as usize
                    ];
                    let mut actual = 0u32;
                    if let Some(get_presets) = driver.funcs.nvEncGetEncodePresetGUIDs {
                        let _ = get_presets(
                            encoder_ptr,
                            NV_ENC_CODEC_HEVC_GUID,
                            presets.as_mut_ptr(),
                            preset_count,
                            &mut actual,
                        );
                        if !presets.is_empty() {
                            selected_preset = presets
                                .iter()
                                .copied()
                                .find(|p| *p == NV_ENC_PRESET_P7_GUID)
                                .unwrap_or(presets[0]);
                        }
                    }
                }
            }

            // Query preset config via nvEncGetEncodePresetConfigEx
            let mut preset_cfg: NV_ENC_PRESET_CONFIG = std::mem::zeroed();
            preset_cfg.version = struct_ver(5) | (1 << 31);
            preset_cfg.presetCfg.version = struct_ver(9) | (1 << 31);
            let mut has_preset_cfg = false;

            if let Some(get_preset_cfg_ex) = driver.funcs.nvEncGetEncodePresetConfigEx {
                let status = get_preset_cfg_ex(
                    encoder_ptr,
                    NV_ENC_CODEC_HEVC_GUID,
                    selected_preset,
                    NV_ENC_TUNING_INFO_HIGH_QUALITY,
                    std::ptr::addr_of_mut!(preset_cfg).cast(),
                );
                if status == NV_ENC_SUCCESS {
                    has_preset_cfg = true;
                }
            }

            let mut init_params: NV_ENC_INITIALIZE_PARAMS = std::mem::zeroed();
            init_params.version = struct_ver(7) | (1 << 31);
            init_params.encodeGUID = NV_ENC_CODEC_HEVC_GUID;
            init_params.presetGUID = selected_preset;
            init_params.encodeWidth = config.width;
            init_params.encodeHeight = config.height;
            init_params.darWidth = config.width;
            init_params.darHeight = config.height;
            init_params.frameRateNum = config.fps.max(1);
            init_params.frameRateDen = 1;
            init_params.enablePTD = 1;
            init_params.tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;
            if has_preset_cfg {
                preset_cfg.presetCfg.gopLength = config.gop_size.max(1);
                preset_cfg.presetCfg.frameIntervalP = 1; // IPPP (No B-frames)
                preset_cfg.presetCfg.hevc_idr_period = config.gop_size.max(1);
                init_params.encodeConfig = std::ptr::addr_of_mut!(preset_cfg.presetCfg).cast();
            }

            let init_encoder = driver.funcs.nvEncInitializeEncoder.ok_or_else(|| {
                VideoError::NvencUnavailable("nvEncInitializeEncoder missing".into())
            })?;

            let mut status = init_encoder(encoder_ptr, &mut init_params);
            if status != NV_ENC_SUCCESS {
                init_params.encodeConfig = std::ptr::null_mut();
                status = init_encoder(encoder_ptr, &mut init_params);
            }
            if status != NV_ENC_SUCCESS {
                let err_str = driver.get_last_error(encoder_ptr);
                return Err(VideoError::NvencError(format!(
                    "nvEncInitializeEncoder failed (status {status}): {err_str}"
                )));
            }

            // Create input buffer (YUV420 Planar)
            let mut create_in: NV_ENC_CREATE_INPUT_BUFFER = std::mem::zeroed();
            create_in.version = struct_ver(2);
            create_in.width = config.width;
            create_in.height = config.height;
            create_in.bufferFmt = NV_ENC_BUFFER_FORMAT_IYUV;

            let create_input_buf = driver.funcs.nvEncCreateInputBuffer.ok_or_else(|| {
                VideoError::NvencUnavailable("nvEncCreateInputBuffer missing".into())
            })?;

            let mut status = create_input_buf(encoder_ptr, &mut create_in);
            if status != NV_ENC_SUCCESS {
                create_in.version = struct_ver(1);
                status = create_input_buf(encoder_ptr, &mut create_in);
            }
            if status != NV_ENC_SUCCESS {
                let err_str = driver.get_last_error(encoder_ptr);
                return Err(VideoError::NvencError(format!(
                    "nvEncCreateInputBuffer failed (status {status}): {err_str}"
                )));
            }

            // Create bitstream output buffer
            let mut create_bs: NV_ENC_CREATE_BITSTREAM_BUFFER = std::mem::zeroed();
            create_bs.version = struct_ver(1);

            let create_bs_buf = driver.funcs.nvEncCreateBitstreamBuffer.ok_or_else(|| {
                VideoError::NvencUnavailable("nvEncCreateBitstreamBuffer missing".into())
            })?;

            let status = create_bs_buf(encoder_ptr, &mut create_bs);
            if status != NV_ENC_SUCCESS {
                let err_str = driver.get_last_error(encoder_ptr);
                return Err(VideoError::NvencError(format!(
                    "nvEncCreateBitstreamBuffer failed (status {status}): {err_str}"
                )));
            }

            self.driver = Some(driver);
            self.encoder = encoder_ptr;
            self.config = Some(config.clone());
            self.input_buffer = create_in.inputBuffer;
            self.bitstream_buffer = create_bs.bitstreamBuffer;
            self.frame_index = 0;

            tracing::info!(
                width = config.width,
                height = config.height,
                fps = config.fps,
                crf = config.crf_or_bitrate,
                "Initialized hardware NVENC HEVC encoder session successfully"
            );

            Ok(())
        }
    }

    #[allow(clippy::too_many_lines)]
    fn encode_frame(
        &mut self,
        frame: &Yuv420PlanarFrame,
        is_keyframe: bool,
    ) -> Result<Vec<HevcNalUnit>, VideoError> {
        let driver = self
            .driver
            .as_ref()
            .ok_or_else(|| VideoError::NvencError("Encoder not initialized".into()))?;

        unsafe {
            let struct_ver = |ver: u32| -> u32 { driver.api_version | (ver << 16) | (0x7 << 28) };

            // Lock input buffer
            let mut lock_in: NV_ENC_LOCK_INPUT_BUFFER = std::mem::zeroed();
            lock_in.version = struct_ver(1);
            lock_in.inputBuffer = self.input_buffer;

            let lock_input = driver
                .funcs
                .nvEncLockInputBuffer
                .ok_or_else(|| VideoError::NvencError("nvEncLockInputBuffer missing".into()))?;

            let status = lock_input(self.encoder, &mut lock_in);
            if status != NV_ENC_SUCCESS {
                let err_str = driver.get_last_error(self.encoder);
                return Err(VideoError::NvencError(format!(
                    "nvEncLockInputBuffer failed ({status}): {err_str}"
                )));
            }

            let dst_ptr = lock_in.bufferDataPtr as *mut u8;
            let pitch = lock_in.pitch as usize;
            let width = frame.width as usize;
            let height = frame.height as usize;
            let half_h = height / 2;
            let half_w = width / 2;

            // Copy Y plane
            for row in 0..height {
                let src_offset = row * frame.y_stride;
                let dst_offset = row * pitch;
                std::ptr::copy_nonoverlapping(
                    frame.y_plane.as_ptr().add(src_offset),
                    dst_ptr.add(dst_offset),
                    width,
                );
            }

            // Copy U & V planes (Planar IYUV layout: Y, then U, then V)
            let u_plane_dst = dst_ptr.add(pitch * height);
            let uv_pitch = (pitch / 2).max(half_w);
            for row in 0..half_h {
                let src_offset = row * frame.uv_stride;
                let dst_offset = row * uv_pitch;
                std::ptr::copy_nonoverlapping(
                    frame.u_plane.as_ptr().add(src_offset),
                    u_plane_dst.add(dst_offset),
                    half_w,
                );
            }

            let v_plane_dst = u_plane_dst.add(uv_pitch * half_h);
            for row in 0..half_h {
                let src_offset = row * frame.uv_stride;
                let dst_offset = row * uv_pitch;
                std::ptr::copy_nonoverlapping(
                    frame.v_plane.as_ptr().add(src_offset),
                    v_plane_dst.add(dst_offset),
                    half_w,
                );
            }

            let unlock_input = driver
                .funcs
                .nvEncUnlockInputBuffer
                .ok_or_else(|| VideoError::NvencError("nvEncUnlockInputBuffer missing".into()))?;
            unlock_input(self.encoder, self.input_buffer);

            // Encode Picture
            let mut pic_params: NV_ENC_PIC_PARAMS = std::mem::zeroed();
            pic_params.version = struct_ver(7) | (1 << 31);
            pic_params.inputWidth = frame.width;
            pic_params.inputHeight = frame.height;
            pic_params.inputPitch = lock_in.pitch;
            pic_params.inputBuffer = self.input_buffer;
            pic_params.outputBitstream = self.bitstream_buffer;
            pic_params.bufferFmt = NV_ENC_BUFFER_FORMAT_IYUV;
            pic_params.pictureStruct = 1; // Progressive
            pic_params.frameIdx = self.frame_index;
            if is_keyframe || self.frame_index == 0 {
                pic_params.encodePicFlags = 0x00000002 /* FORCEIDR */ | 0x00000004 /* OUTPUT_SPSPPS */;
            }

            let encode_pic = driver
                .funcs
                .nvEncEncodePicture
                .ok_or_else(|| VideoError::NvencError("nvEncEncodePicture missing".into()))?;

            let mut status = encode_pic(self.encoder, &mut pic_params);
            if status == 17
            /* NV_ENC_ERR_NEED_MORE_INPUT */
            {
                self.frame_index += 1;
                return Ok(Vec::new());
            }
            if status != NV_ENC_SUCCESS {
                pic_params.version = struct_ver(4) | (1 << 31);
                status = encode_pic(self.encoder, &mut pic_params);
                if status == 17 {
                    self.frame_index += 1;
                    return Ok(Vec::new());
                }
            }
            if status != NV_ENC_SUCCESS {
                let err_str = driver.get_last_error(self.encoder);
                return Err(VideoError::NvencError(format!(
                    "nvEncEncodePicture failed ({status}): {err_str}"
                )));
            }

            // Lock bitstream to read encoded NAL units
            let mut lock_bs: NV_ENC_LOCK_BITSTREAM = std::mem::zeroed();
            lock_bs.version = struct_ver(2);
            lock_bs.outputBitstream = self.bitstream_buffer;

            let lock_bitstream = driver
                .funcs
                .nvEncLockBitstream
                .ok_or_else(|| VideoError::NvencError("nvEncLockBitstream missing".into()))?;

            let mut status = lock_bitstream(self.encoder, &mut lock_bs);
            if status != NV_ENC_SUCCESS {
                lock_bs.version = struct_ver(1);
                status = lock_bitstream(self.encoder, &mut lock_bs);
            }
            if status != NV_ENC_SUCCESS {
                let err_str = driver.get_last_error(self.encoder);
                return Err(VideoError::NvencError(format!(
                    "nvEncLockBitstream failed ({status}): {err_str}"
                )));
            }

            let nalus = if lock_bs.bitstreamSizeInBytes > 0 && !lock_bs.bitstreamBufferPtr.is_null()
            {
                let bitstream_slice = std::slice::from_raw_parts(
                    lock_bs.bitstreamBufferPtr as *const u8,
                    lock_bs.bitstreamSizeInBytes as usize,
                );
                parse_annex_b_nalus(bitstream_slice)
            } else {
                Vec::new()
            };

            let unlock_bitstream = driver
                .funcs
                .nvEncUnlockBitstream
                .ok_or_else(|| VideoError::NvencError("nvEncUnlockBitstream missing".into()))?;
            unlock_bitstream(self.encoder, self.bitstream_buffer);

            self.frame_index += 1;
            Ok(nalus)
        }
    }

    fn flush(&mut self) -> Result<Vec<HevcNalUnit>, VideoError> {
        let Some(driver) = self.driver.as_ref() else {
            return Ok(Vec::new());
        };

        unsafe {
            let struct_ver = |ver: u32| -> u32 { driver.api_version | (ver << 16) | (0x7 << 28) };

            let mut pic_params: NV_ENC_PIC_PARAMS = std::mem::zeroed();
            pic_params.version = struct_ver(7) | (1 << 31);
            pic_params.encodePicFlags = 0x00000008; // NV_ENC_PIC_FLAG_EOS

            if let Some(encode_pic) = driver.funcs.nvEncEncodePicture {
                let _ = encode_pic(self.encoder, &mut pic_params);
            }

            Ok(Vec::new())
        }
    }
}

impl Drop for NvencHevcEncoder {
    fn drop(&mut self) {
        if let Some(driver) = self.driver.take() {
            unsafe {
                if !self.bitstream_buffer.is_null() {
                    if let Some(destroy_bs) = driver.funcs.nvEncDestroyBitstreamBuffer {
                        destroy_bs(self.encoder, self.bitstream_buffer);
                    }
                    self.bitstream_buffer = std::ptr::null_mut();
                }

                if !self.input_buffer.is_null() {
                    if let Some(destroy_in) = driver.funcs.nvEncDestroyInputBuffer {
                        destroy_in(self.encoder, self.input_buffer);
                    }
                    self.input_buffer = std::ptr::null_mut();
                }

                if !self.encoder.is_null() {
                    if let Some(destroy_enc) = driver.funcs.nvEncDestroyEncoder {
                        destroy_enc(self.encoder);
                    }
                    self.encoder = std::ptr::null_mut();
                }
            }
        }
    }
}
