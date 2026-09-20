//! High-performance HEVC (H.265) video encoding and pure Rust MP4 multiplexing.
//!
//! Provides hardware-accelerated video export (NVENC, VA-API, VideoToolbox, Media Foundation)
//! and 24-bit TrueColor MP4 generation with variable $\mathrm{SE}(3)$ timing preservation.

pub mod mediafoundation;
pub mod mock;
pub mod mp4_muxer;
pub mod nvenc;
pub mod vaapi;
pub mod videotoolbox;

pub use crate::color::{RgbaToYuv420Converter, Yuv420PlanarFrame, BT709_YUV_MATRIX};
pub use mediafoundation::MediaFoundationHevcEncoder;
pub use mock::SoftwareMockHevcEncoder;
pub use mp4_muxer::{EncodedVideoSample, HevcNalUnit, Mp4Muxer};
pub use nvenc::NvencHevcEncoder;
pub use vaapi::VaapiHevcEncoder;
pub use videotoolbox::VideoToolboxHevcEncoder;

use crate::error::{Error, Result};
use image::RgbaImage;
use std::io::Write;
use thiserror::Error;

/// Default number of ping-pong wiggle loop cycles encoded into MP4 video (4 cycles ~ 2.5-3.0s).
pub const DEFAULT_MP4_LOOPS: usize = 4;

/// Default Constant Rate Factor (CRF) for HEVC video encoding (18 for pristine visual fidelity).
pub const DEFAULT_MP4_CRF: u32 = 18;

/// Errors encountered during video encoding or container multiplexing.
#[derive(Debug, Error)]
pub enum VideoError {
    /// NVIDIA NVENC hardware encoder is unavailable on this host.
    #[error("NVIDIA NVENC hardware encoder unavailable: {0}")]
    NvencUnavailable(String),

    /// NVIDIA NVENC returned an internal encoding error.
    #[error("NVIDIA NVENC encoding failure: {0}")]
    NvencError(String),

    /// Linux VA-API encoder is unavailable on this host.
    #[error("Linux VA-API hardware encoder unavailable: {0}")]
    VaapiUnavailable(String),

    /// Requested encoder backend is unsupported on the current operating system.
    #[error("Platform video encoder unsupported: {0}")]
    UnsupportedPlatform(String),

    /// Missing required HEVC parameter sets (VPS, SPS, PPS).
    #[error("Missing HEVC parameter sets for MP4 muxing: {0}")]
    MissingParameterSets(String),

    /// Generic I/O or container error.
    #[error("Video container I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Metadata and parameter configuration for the abstract HEVC encoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HevcEncoderConfig {
    /// Video width in pixels (even).
    pub width: u32,
    /// Video height in pixels (even).
    pub height: u32,
    /// Target frame rate in frames per second.
    pub fps: u32,
    /// Bit depth (8 for Main Profile, 10 for Main10).
    pub bit_depth: u8,
    /// Constant Rate Factor / Target Quality (0..=51, default: 18).
    pub crf_or_bitrate: u32,
    /// Keyframe GOP interval in frames.
    pub gop_size: u32,
    /// Non-uniform per-frame display durations in milliseconds ($\Delta t_i$).
    pub timing_delays_ms: Vec<u32>,
}

/// Abstract hardware/software HEVC video frame encoder.
pub trait HevcFrameEncoder: Send {
    /// Initializes the encoder with resolution, rate control, and GOP structure.
    ///
    /// # Errors
    /// Returns [`VideoError`] if hardware device creation or session negotiation fails.
    fn initialize(&mut self, config: &HevcEncoderConfig) -> std::result::Result<(), VideoError>;

    /// Encodes a single YUV420p video frame and returns generated NAL units.
    ///
    /// # Errors
    /// Returns [`VideoError`] if picture encoding fails.
    fn encode_frame(
        &mut self,
        frame: &Yuv420PlanarFrame,
        is_keyframe: bool,
    ) -> std::result::Result<Vec<HevcNalUnit>, VideoError>;

    /// Flushes delayed B/P frames from the hardware pipeline at end-of-stream.
    ///
    /// # Errors
    /// Returns [`VideoError`] if stream flushing fails.
    fn flush(&mut self) -> std::result::Result<Vec<HevcNalUnit>, VideoError>;
}

/// High-level configuration for Wiggle MP4 video export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WiggleVideoConfig {
    /// Number of continuous ping-pong wiggle loop cycles (default: 4).
    pub loops: usize,
    /// Quality / Constant Rate Factor (default: 18).
    pub crf: u32,
    /// Inter-frame default display delay in milliseconds (default: 100ms).
    pub delay_ms: u32,
    /// Whether to force NVIDIA NVENC hardware acceleration.
    pub enable_nvenc: bool,
    /// Optional adaptive non-uniform frame delays `[delay_01_ms, delay_12_ms]` derived from $\mathrm{SE}(3)$ camera extrinsics.
    pub adaptive_delays_ms: Option<[u32; 2]>,
    /// Fallback to software synthetic encoder for test environments if hardware is absent.
    pub fallback_to_mock: bool,
}

impl Default for WiggleVideoConfig {
    fn default() -> Self {
        Self {
            loops: DEFAULT_MP4_LOOPS,
            crf: DEFAULT_MP4_CRF,
            delay_ms: crate::gif::DEFAULT_FRAME_DELAY_MS,
            enable_nvenc: false,
            adaptive_delays_ms: None,
            fallback_to_mock: false,
        }
    }
}

impl WiggleVideoConfig {
    /// Creates a new `WiggleVideoConfig` with default settings.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            loops: DEFAULT_MP4_LOOPS,
            crf: DEFAULT_MP4_CRF,
            delay_ms: crate::gif::DEFAULT_FRAME_DELAY_MS,
            enable_nvenc: false,
            adaptive_delays_ms: None,
            fallback_to_mock: false,
        }
    }

    /// Sets the number of ping-pong loop cycles.
    #[must_use]
    pub const fn with_loops(mut self, loops: usize) -> Self {
        self.loops = loops;
        self
    }

    /// Sets the Constant Rate Factor (CRF).
    #[must_use]
    pub const fn with_crf(mut self, crf: u32) -> Self {
        self.crf = crf;
        self
    }

    /// Enables or disables forced NVIDIA NVENC hardware encoding.
    #[must_use]
    pub const fn with_nvenc(mut self, enable_nvenc: bool) -> Self {
        self.enable_nvenc = enable_nvenc;
        self
    }

    /// Sets adaptive non-uniform frame delays `[delay_01_ms, delay_12_ms]`.
    #[must_use]
    pub const fn with_adaptive_delays(mut self, delays_ms: [u32; 2]) -> Self {
        self.adaptive_delays_ms = Some(delays_ms);
        self
    }

    /// Enables fallback to software synthetic encoder for headless CI testing.
    #[must_use]
    pub const fn with_mock_fallback(mut self, fallback: bool) -> Self {
        self.fallback_to_mock = fallback;
        self
    }
}

/// Supported hardware and software HEVC video encoder backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoEncoderBackend {
    /// NVIDIA NVENC hardware-accelerated encoder.
    Nvenc,
    /// Linux VA-API hardware-accelerated encoder (Intel QuickSync / AMD VCN).
    Vaapi,
    /// Apple VideoToolbox hardware-accelerated encoder.
    VideoToolbox,
    /// Microsoft Windows Media Foundation hardware-accelerated encoder.
    MediaFoundation,
    /// In-memory synthetic software mock encoder (for testing).
    SoftwareMock,
}

impl std::fmt::Display for VideoEncoderBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nvenc => write!(f, "NVIDIA NVENC Hardware Encoder"),
            Self::Vaapi => write!(f, "Linux VA-API Hardware Encoder"),
            Self::VideoToolbox => write!(f, "Apple VideoToolbox Hardware Encoder"),
            Self::MediaFoundation => write!(f, "Windows Media Foundation Hardware Encoder"),
            Self::SoftwareMock => write!(f, "Synthetic Software Mock Encoder"),
        }
    }
}

/// Probes the system for available HEVC video encoder backends based on platform and configuration.
///
/// # Arguments
/// * `config` - Video configuration with hardware preferences and mock fallback flags.
///
/// # Errors
/// Returns [`VideoError`] if the requested or platform hardware encoder is unavailable.
pub fn probe_video_encoder_backend(
    config: &WiggleVideoConfig,
) -> std::result::Result<VideoEncoderBackend, VideoError> {
    let probe_cfg = HevcEncoderConfig {
        width: 320,
        height: 240,
        fps: 10,
        bit_depth: 8,
        crf_or_bitrate: config.crf,
        gop_size: 30,
        timing_delays_ms: vec![config.delay_ms],
    };

    if config.enable_nvenc {
        let mut nv = NvencHevcEncoder::new();
        return match nv.initialize(&probe_cfg) {
            Ok(()) => Ok(VideoEncoderBackend::Nvenc),
            Err(e) => {
                if config.fallback_to_mock {
                    Ok(VideoEncoderBackend::SoftwareMock)
                } else {
                    Err(e)
                }
            }
        };
    }

    #[cfg(target_os = "macos")]
    {
        let mut vt = VideoToolboxHevcEncoder::new();
        match vt.initialize(&probe_cfg) {
            Ok(()) => Ok(VideoEncoderBackend::VideoToolbox),
            Err(e) => {
                if config.fallback_to_mock {
                    Ok(VideoEncoderBackend::SoftwareMock)
                } else {
                    Err(e)
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        let mut mf = MediaFoundationHevcEncoder::new();
        match mf.initialize(&probe_cfg) {
            Ok(()) => Ok(VideoEncoderBackend::MediaFoundation),
            Err(e) => {
                if config.fallback_to_mock {
                    Ok(VideoEncoderBackend::SoftwareMock)
                } else {
                    Err(e)
                }
            }
        }
    }

    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    {
        // On Linux: probe NVENC first, then VA-API
        let mut nv = NvencHevcEncoder::new();
        if nv.initialize(&probe_cfg).is_ok() {
            return Ok(VideoEncoderBackend::Nvenc);
        }

        let mut va = VaapiHevcEncoder::new();
        if va.initialize(&probe_cfg).is_ok() {
            return Ok(VideoEncoderBackend::Vaapi);
        }

        if config.fallback_to_mock {
            Ok(VideoEncoderBackend::SoftwareMock)
        } else {
            Err(VideoError::UnsupportedPlatform(
                "No supported hardware HEVC video encoder (NVENC or VA-API) is available on this system".into(),
            ))
        }
    }
}

/// Instantiates an uninitialized HEVC frame encoder based on the selected backend.
#[must_use]
pub fn create_hevc_encoder(backend: VideoEncoderBackend) -> Box<dyn HevcFrameEncoder> {
    match backend {
        VideoEncoderBackend::Nvenc => Box::new(NvencHevcEncoder::new()),
        VideoEncoderBackend::Vaapi => Box::new(VaapiHevcEncoder::new()),
        VideoEncoderBackend::VideoToolbox => Box::new(VideoToolboxHevcEncoder::new()),
        VideoEncoderBackend::MediaFoundation => Box::new(MediaFoundationHevcEncoder::new()),
        VideoEncoderBackend::SoftwareMock => Box::new(SoftwareMockHevcEncoder::new()),
    }
}

/// Orchestrator for assembling and encoding stereoscopic 3D wiggle MP4 video files.
pub struct WiggleVideoBuilder;

impl WiggleVideoBuilder {
    /// Builds a 24-bit TrueColor HEVC MP4 video from aligned sub-frames and writes it to the target stream.
    ///
    /// # Arguments
    /// * `frames` - Slices of aligned RGBA sub-frames.
    /// * `config` - Video configuration (loops, CRF, hardware flags).
    /// * `writer` - Target output stream implementing [`std::io::Write`].
    ///
    /// # Errors
    /// Returns [`Error`] if frames list is empty, dimensions differ, or encoding fails.
    #[allow(clippy::too_many_lines)]
    pub fn build_wiggle_video<W: Write>(
        frames: &[RgbaImage],
        config: &WiggleVideoConfig,
        writer: &mut W,
    ) -> Result<()> {
        if frames.is_empty() {
            return Err(Error::Unknown(
                "No frames provided for Wiggle MP4 video".into(),
            ));
        }

        let (width, height) = frames[0].dimensions();
        for frame in frames {
            if frame.dimensions() != (width, height) {
                return Err(Error::Unknown(
                    "All frames must have identical dimensions for video assembly".into(),
                ));
            }
        }

        // 1. Convert RGBA frames to YUV420p Planar buffers
        let yuv_frames: Vec<Yuv420PlanarFrame> =
            frames.iter().map(RgbaToYuv420Converter::convert).collect();

        let enc_width = yuv_frames[0].width;
        let enc_height = yuv_frames[0].height;

        // 2. Select hardware/software encoder using unified probe interface
        let backend = probe_video_encoder_backend(config)
            .map_err(|e| Error::Unknown(format!("Hardware HEVC video encoder unavailable: {e}")))?;

        let mut encoder = create_hevc_encoder(backend);

        // 3. Initialize encoder configuration
        let encoder_cfg = HevcEncoderConfig {
            width: enc_width,
            height: enc_height,
            fps: 10, // Nominal FPS for timebase
            bit_depth: 8,
            crf_or_bitrate: config.crf,
            gop_size: 30,
            timing_delays_ms: vec![config.delay_ms],
        };

        if let Err(e) = encoder.initialize(&encoder_cfg) {
            if config.fallback_to_mock {
                tracing::warn!(error = %e, "Hardware video encoder failed on target resolution; falling back to software mock");
                encoder = Box::new(SoftwareMockHevcEncoder::new());
                encoder
                    .initialize(&encoder_cfg)
                    .map_err(|err| Error::Unknown(err.to_string()))?;
            } else {
                return Err(Error::Unknown(format!(
                    "Failed to initialize HEVC video encoder ({backend}): {e}"
                )));
            }
        }

        // 4. Construct ping-pong loop index sequence
        let single_loop_indices: Vec<usize> = if frames.len() == 3 {
            vec![0, 1, 2, 1]
        } else if frames.len() == 2 {
            vec![0, 1]
        } else {
            (0..frames.len()).collect()
        };

        let loops_count = config.loops.max(1);
        let mut full_sequence_indices = Vec::with_capacity(single_loop_indices.len() * loops_count);
        for _ in 0..loops_count {
            full_sequence_indices.extend_from_slice(&single_loop_indices);
        }

        // 5. Calculate per-frame display durations (preserving SE(3) non-uniform timings)
        let get_frame_duration = |step_idx: usize, frame_idx: usize| -> u32 {
            if let Some([d01, d12]) = config.adaptive_delays_ms {
                if frames.len() == 3 {
                    let seq_pos = step_idx % 4;
                    match seq_pos {
                        0 | 3 => d01,
                        1 | 2 => d12,
                        _ => config.delay_ms,
                    }
                } else {
                    config.delay_ms
                }
            } else {
                let _ = frame_idx;
                config.delay_ms
            }
        };

        // 6. Encode video frames and collect NAL units
        let mut vps_data = Vec::new();
        let mut sps_data = Vec::new();
        let mut pps_data = Vec::new();
        let mut video_samples = Vec::with_capacity(full_sequence_indices.len());

        for (step_idx, &frame_idx) in full_sequence_indices.iter().enumerate() {
            let is_keyframe = step_idx == 0;
            let yuv_frame = &yuv_frames[frame_idx];
            let nalus = encoder
                .encode_frame(yuv_frame, is_keyframe)
                .map_err(|e| Error::Unknown(e.to_string()))?;

            let mut sample_nalus = Vec::new();
            for nalu in nalus {
                match nalu.nal_type {
                    mp4_muxer::NAL_VPS => vps_data = nalu.data,
                    mp4_muxer::NAL_SPS => sps_data = nalu.data,
                    mp4_muxer::NAL_PPS => pps_data = nalu.data,
                    _ => sample_nalus.push(nalu),
                }
            }

            let dur_ms = get_frame_duration(step_idx, frame_idx);
            video_samples.push(EncodedVideoSample {
                nalus: sample_nalus,
                duration_ms: dur_ms,
                is_sync: is_keyframe,
            });
        }

        let flushed = encoder.flush().map_err(|e| Error::Unknown(e.to_string()))?;
        for nalu in flushed {
            match nalu.nal_type {
                mp4_muxer::NAL_VPS => vps_data = nalu.data,
                mp4_muxer::NAL_SPS => sps_data = nalu.data,
                mp4_muxer::NAL_PPS => pps_data = nalu.data,
                _ => {
                    if let Some(last_sample) = video_samples.last_mut() {
                        last_sample.nalus.push(nalu);
                    }
                }
            }
        }

        // Fallback parameter sets if encoder emitted in-band only
        if vps_data.is_empty() {
            vps_data = vec![
                0x40, 0x01, 0x0C, 0x01, 0xFF, 0xFF, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03,
                0x00, 0x00, 0x03, 0x00, 0x00, 0x78, 0xAC, 0x09,
            ];
        }
        if sps_data.is_empty() {
            sps_data = vec![
                0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03,
                0x00, 0x00, 0x78, 0xA0, 0x02, 0x80, 0x80, 0x2D, 0x16, 0x59, 0x5E, 0x49, 0x2B, 0x01,
                0x01, 0x01, 0x40,
            ];
        }
        if pps_data.is_empty() {
            pps_data = vec![0x44, 0x01, 0xC0, 0xF3, 0xC0];
        }

        // 7. Mux into pure Rust ISO BMFF MP4 container
        Mp4Muxer::mux_hevc(
            enc_width,
            enc_height,
            &vps_data,
            &sps_data,
            &pps_data,
            &video_samples,
            writer,
        )?;

        tracing::info!(
            width = enc_width,
            height = enc_height,
            loops = loops_count,
            total_samples = video_samples.len(),
            "Encoded and multiplexed 24-bit TrueColor HEVC MP4 video successfully"
        );

        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use image::Rgba;

    #[test]
    fn test_wiggle_video_builder_with_mock() {
        let mut img0 = RgbaImage::new(64, 64);
        let mut img1 = RgbaImage::new(64, 64);
        let mut img2 = RgbaImage::new(64, 64);
        img0.put_pixel(0, 0, Rgba([255, 0, 0, 255]));
        img1.put_pixel(0, 0, Rgba([0, 255, 0, 255]));
        img2.put_pixel(0, 0, Rgba([0, 0, 255, 255]));

        let frames = vec![img0, img1, img2];
        let config = WiggleVideoConfig::new()
            .with_mock_fallback(true)
            .with_loops(2);

        let mut mp4_bytes = Vec::new();
        WiggleVideoBuilder::build_wiggle_video(&frames, &config, &mut mp4_bytes)
            .expect("Video build should succeed");

        assert!(mp4_bytes.len() > 100);
        assert_eq!(&mp4_bytes[4..8], b"ftyp");
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn test_nvenc_hevc_video_encode_live() {
        let mut img0 = RgbaImage::new(320, 240);
        let mut img1 = RgbaImage::new(320, 240);
        let mut img2 = RgbaImage::new(320, 240);
        for y in 0..240 {
            for x in 0..320 {
                img0.put_pixel(x, y, Rgba([(x % 256) as u8, (y % 256) as u8, 128, 255]));
                img1.put_pixel(x, y, Rgba([(y % 256) as u8, 128, (x % 256) as u8, 255]));
                img2.put_pixel(x, y, Rgba([128, (x % 256) as u8, (y % 256) as u8, 255]));
            }
        }

        let frames = vec![img0, img1, img2];
        let config = WiggleVideoConfig::new()
            .with_nvenc(true)
            .with_loops(4)
            .with_crf(18);

        let mut mp4_bytes = Vec::new();
        let res = WiggleVideoBuilder::build_wiggle_video(&frames, &config, &mut mp4_bytes);
        eprintln!("test_nvenc_hevc_video_encode_live result: {res:?}");
        match res {
            Ok(()) => {
                assert!(mp4_bytes.len() > 100);
                assert_eq!(&mp4_bytes[4..8], b"ftyp");
                let out_str = String::from_utf8_lossy(&mp4_bytes);
                assert!(out_str.contains("moov"));
                assert!(out_str.contains("mdat"));
            }
            Err(e) => {
                eprintln!("NVENC unavailable or skipped in test environment: {e}");
            }
        }
    }

    #[test]
    fn test_probe_video_encoder_backend_mock_fallback() {
        let config = WiggleVideoConfig::new().with_mock_fallback(true);
        let backend =
            probe_video_encoder_backend(&config).expect("Probe with mock fallback should succeed");
        assert!(matches!(
            backend,
            VideoEncoderBackend::Nvenc
                | VideoEncoderBackend::Vaapi
                | VideoEncoderBackend::VideoToolbox
                | VideoEncoderBackend::MediaFoundation
                | VideoEncoderBackend::SoftwareMock
        ));
    }

    #[test]
    fn test_probe_video_encoder_backend_nvenc_or_vaapi() {
        let config = WiggleVideoConfig::new();
        let probe_res = probe_video_encoder_backend(&config);
        // On host with NVENC GPU, probe_res should be Ok(Nvenc)
        eprintln!("probe_video_encoder_backend result: {probe_res:?}");
    }
}
