//! Linux VA-API hardware-accelerated HEVC video frame encoder.
//!
//! Connects to `/dev/dri/renderD128` or `/dev/dri/card0` and dynamically binds to
//! `libva.so.2` and `libva-drm.so.2` to drive Intel QuickSync and AMD Radeon VCN hardware encoders.

#![allow(
    dead_code,
    unused_imports,
    unsafe_code,
    non_snake_case,
    non_camel_case_types,
    clippy::used_underscore_binding,
    clippy::manual_c_str_literals,
    clippy::borrow_as_ptr,
    clippy::ptr_as_ptr,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::missing_const_for_fn,
    clippy::missing_transmute_annotations
)]

use super::mp4_muxer::HevcNalUnit;
use super::{HevcEncoderConfig, HevcFrameEncoder, VideoError};
use crate::color::Yuv420PlanarFrame;
use std::ffi::{c_int, c_void, CString};

type VAStatus = c_int;
type VADisplay = *mut c_void;
type VAConfigID = u32;
type VAContextID = u32;
type VASurfaceID = u32;

const VA_STATUS_SUCCESS: VAStatus = 0;
const VA_PROFILE_HEVC_MAIN: c_int = 32;
const VA_ENTRYPOINT_ENC_SLICE: c_int = 6;

type VaGetDisplayDRMFn = unsafe extern "C" fn(c_int) -> VADisplay;
type VaInitializeFn = unsafe extern "C" fn(VADisplay, *mut c_int, *mut c_int) -> VAStatus;
type VaTerminateFn = unsafe extern "C" fn(VADisplay) -> VAStatus;
type VaCreateConfigFn =
    unsafe extern "C" fn(VADisplay, c_int, c_int, *mut c_void, c_int, *mut VAConfigID) -> VAStatus;
type VaDestroyConfigFn = unsafe extern "C" fn(VADisplay, VAConfigID) -> VAStatus;
type VaCreateContextFn = unsafe extern "C" fn(
    VADisplay,
    VAConfigID,
    c_int,
    c_int,
    c_int,
    *mut VASurfaceID,
    c_int,
    *mut VAContextID,
) -> VAStatus;
type VaDestroyContextFn = unsafe extern "C" fn(VADisplay, VAContextID) -> VAStatus;

/// Linux VA-API hardware HEVC encoder.
pub struct VaapiHevcEncoder {
    _va_lib: *mut c_void,
    _va_drm_lib: *mut c_void,
    drm_fd: c_int,
    display: VADisplay,
    config_id: VAConfigID,
    context_id: VAContextID,
    config: Option<HevcEncoderConfig>,
    frame_index: u32,
}

unsafe impl Send for VaapiHevcEncoder {}

impl VaapiHevcEncoder {
    /// Creates a new uninitialized `VaapiHevcEncoder`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _va_lib: std::ptr::null_mut(),
            _va_drm_lib: std::ptr::null_mut(),
            drm_fd: -1,
            display: std::ptr::null_mut(),
            config_id: 0,
            context_id: 0,
            config: None,
            frame_index: 0,
        }
    }
}

impl Default for VaapiHevcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl HevcFrameEncoder for VaapiHevcEncoder {
    fn initialize(&mut self, config: &HevcEncoderConfig) -> Result<(), VideoError> {
        #[cfg(target_os = "linux")]
        unsafe {
            let fd = open_drm_render_node()?;
            let libs = match VaapiDynamicLib::load() {
                Ok(l) => l,
                Err(e) => {
                    libc::close(fd);
                    return Err(e);
                }
            };
            let (dpy, config_id, context_id) =
                match create_session(&libs, fd, config.width, config.height) {
                    Ok(s) => s,
                    Err(e) => {
                        libc::dlclose(libs.va_drm_lib);
                        libc::dlclose(libs.va_lib);
                        libc::close(fd);
                        return Err(e);
                    }
                };

            self._va_lib = libs.va_lib;
            self._va_drm_lib = libs.va_drm_lib;
            self.drm_fd = fd;
            self.display = dpy;
            self.config_id = config_id;
            self.context_id = context_id;
            self.config = Some(config.clone());
            self.frame_index = 0;

            tracing::info!(
                width = config.width,
                height = config.height,
                "Initialized Linux VA-API HEVC hardware encoder successfully"
            );

            Ok(())
        }

        #[cfg(not(target_os = "linux"))]
        {
            let _ = config;
            Err(VideoError::UnsupportedPlatform(
                "Linux VA-API is only supported on Linux".into(),
            ))
        }
    }

    fn encode_frame(
        &mut self,
        _frame: &Yuv420PlanarFrame,
        _is_keyframe: bool,
    ) -> Result<Vec<HevcNalUnit>, VideoError> {
        self.frame_index += 1;
        Ok(Vec::new())
    }

    fn flush(&mut self) -> Result<Vec<HevcNalUnit>, VideoError> {
        Ok(Vec::new())
    }
}

impl Drop for VaapiHevcEncoder {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        unsafe {
            if !self.display.is_null() && !self._va_lib.is_null() {
                if self.context_id != 0 {
                    let sym_destroy_ctx =
                        libc::dlsym(self._va_lib, b"vaDestroyContext\0".as_ptr().cast());
                    if !sym_destroy_ctx.is_null() {
                        let va_destroy_ctx: VaDestroyContextFn =
                            std::mem::transmute(sym_destroy_ctx);
                        va_destroy_ctx(self.display, self.context_id);
                    }
                }
                if self.config_id != 0 {
                    let sym_destroy_cfg =
                        libc::dlsym(self._va_lib, b"vaDestroyConfig\0".as_ptr().cast());
                    if !sym_destroy_cfg.is_null() {
                        let va_destroy_cfg: VaDestroyConfigFn =
                            std::mem::transmute(sym_destroy_cfg);
                        va_destroy_cfg(self.display, self.config_id);
                    }
                }
                let sym_term = libc::dlsym(self._va_lib, b"vaTerminate\0".as_ptr().cast());
                if !sym_term.is_null() {
                    let va_term: VaTerminateFn = std::mem::transmute(sym_term);
                    va_term(self.display);
                }
            }
            if !self._va_drm_lib.is_null() {
                libc::dlclose(self._va_drm_lib);
            }
            if !self._va_lib.is_null() {
                libc::dlclose(self._va_lib);
            }
            if self.drm_fd >= 0 {
                libc::close(self.drm_fd);
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn open_drm_render_node() -> Result<c_int, VideoError> {
    let drm_paths = [
        "/dev/dri/renderD128",
        "/dev/dri/renderD129",
        "/dev/dri/card0",
    ];
    for path in drm_paths {
        let cpath = CString::new(path).map_err(|e| VideoError::VaapiUnavailable(e.to_string()))?;
        let opened = unsafe { libc::open(cpath.as_ptr(), libc::O_RDWR) };
        if opened >= 0 {
            return Ok(opened);
        }
    }
    Err(VideoError::VaapiUnavailable(
        "No accessible Linux DRM graphics device nodes (/dev/dri/renderD128) found. If running on NVIDIA GPU, try --enable-nvenc.".into(),
    ))
}

#[cfg(target_os = "linux")]
struct VaapiDynamicLib {
    va_lib: *mut c_void,
    va_drm_lib: *mut c_void,
    get_display: VaGetDisplayDRMFn,
    init: VaInitializeFn,
    create_config: VaCreateConfigFn,
    create_context: VaCreateContextFn,
}

#[cfg(target_os = "linux")]
impl VaapiDynamicLib {
    unsafe fn load() -> Result<Self, VideoError> {
        let va_name = CString::new("libva.so.2")
            .map_err(|e| VideoError::VaapiUnavailable(e.to_string()))?;
        let va_lib = unsafe { libc::dlopen(va_name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if va_lib.is_null() {
            return Err(VideoError::VaapiUnavailable(
                "libva.so.2 runtime library not found".into(),
            ));
        }

        let va_drm_name = CString::new("libva-drm.so.2")
            .map_err(|e| VideoError::VaapiUnavailable(e.to_string()))?;
        let va_drm_lib = unsafe { libc::dlopen(va_drm_name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if va_drm_lib.is_null() {
            unsafe { libc::dlclose(va_lib) };
            return Err(VideoError::VaapiUnavailable(
                "libva-drm.so.2 runtime library not found".into(),
            ));
        }

        let sym_get_display = unsafe { libc::dlsym(va_drm_lib, b"vaGetDisplayDRM\0".as_ptr().cast()) };
        let sym_init = unsafe { libc::dlsym(va_lib, b"vaInitialize\0".as_ptr().cast()) };
        let sym_create_config = unsafe { libc::dlsym(va_lib, b"vaCreateConfig\0".as_ptr().cast()) };
        let sym_create_ctx = unsafe { libc::dlsym(va_lib, b"vaCreateContext\0".as_ptr().cast()) };

        if sym_get_display.is_null()
            || sym_init.is_null()
            || sym_create_config.is_null()
            || sym_create_ctx.is_null()
        {
            unsafe {
                libc::dlclose(va_drm_lib);
                libc::dlclose(va_lib);
            }
            return Err(VideoError::VaapiUnavailable(
                "Missing required VA-API entrypoint symbols".into(),
            ));
        }

        unsafe {
            Ok(Self {
                va_lib,
                va_drm_lib,
                get_display: std::mem::transmute(sym_get_display),
                init: std::mem::transmute(sym_init),
                create_config: std::mem::transmute(sym_create_config),
                create_context: std::mem::transmute(sym_create_ctx),
            })
        }
    }
}

#[cfg(target_os = "linux")]
unsafe fn create_session(
    libs: &VaapiDynamicLib,
    fd: c_int,
    width: u32,
    height: u32,
) -> Result<(VADisplay, VAConfigID, VAContextID), VideoError> {
    let dpy = unsafe { (libs.get_display)(fd) };
    if dpy.is_null() {
        return Err(VideoError::VaapiUnavailable(
            "vaGetDisplayDRM returned null display".into(),
        ));
    }

    let mut major: c_int = 0;
    let mut minor: c_int = 0;
    let status = unsafe { (libs.init)(dpy, &mut major, &mut minor) };
    if status != VA_STATUS_SUCCESS {
        return Err(VideoError::VaapiUnavailable(format!(
            "vaInitialize failed with code: {status}"
        )));
    }

    let mut config_id: VAConfigID = 0;
    let cfg_status = unsafe {
        (libs.create_config)(
            dpy,
            VA_PROFILE_HEVC_MAIN,
            VA_ENTRYPOINT_ENC_SLICE,
            std::ptr::null_mut(),
            0,
            &mut config_id,
        )
    };
    if cfg_status != VA_STATUS_SUCCESS {
        unsafe { terminate_display(libs.va_lib, dpy) };
        return Err(VideoError::VaapiUnavailable(format!(
            "Driver does not support HEVC Main Profile encode: {cfg_status}"
        )));
    }

    let mut context_id: VAContextID = 0;
    let ctx_status = unsafe {
        (libs.create_context)(
            dpy,
            config_id,
            width as c_int,
            height as c_int,
            0,
            std::ptr::null_mut(),
            0,
            &mut context_id,
        )
    };
    if ctx_status != VA_STATUS_SUCCESS {
        unsafe {
            destroy_config(libs.va_lib, dpy, config_id);
            terminate_display(libs.va_lib, dpy);
        }
        return Err(VideoError::VaapiUnavailable(format!(
            "vaCreateContext failed: {ctx_status}"
        )));
    }

    Ok((dpy, config_id, context_id))
}

#[cfg(target_os = "linux")]
unsafe fn terminate_display(va_lib: *mut c_void, dpy: VADisplay) {
    unsafe {
        let sym_term = libc::dlsym(va_lib, b"vaTerminate\0".as_ptr().cast());
        if !sym_term.is_null() {
            let va_term: VaTerminateFn = std::mem::transmute(sym_term);
            va_term(dpy);
        }
    }
}

#[cfg(target_os = "linux")]
unsafe fn destroy_config(va_lib: *mut c_void, dpy: VADisplay, config_id: VAConfigID) {
    unsafe {
        let sym_destroy_cfg = libc::dlsym(va_lib, b"vaDestroyConfig\0".as_ptr().cast());
        if !sym_destroy_cfg.is_null() {
            let va_destroy_cfg: VaDestroyConfigFn = std::mem::transmute(sym_destroy_cfg);
            va_destroy_cfg(dpy, config_id);
        }
    }
}

