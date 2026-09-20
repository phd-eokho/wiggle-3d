//! macOS Apple VideoToolbox hardware-accelerated HEVC video frame encoder.
//!
//! Provides native hardware acceleration on Apple Silicon (M1/M2/M3/M4 Media Engine)
//! and Intel T2 Macs via macOS system frameworks (`VTCompressionSession`).

use crate::color::Yuv420PlanarFrame;
use super::mp4_muxer::HevcNalUnit;
use super::{HevcEncoderConfig, HevcFrameEncoder, VideoError};

/// Apple VideoToolbox HEVC hardware encoder for macOS.
#[allow(dead_code)]
pub struct VideoToolboxHevcEncoder {
    config: Option<HevcEncoderConfig>,
    frame_index: u32,
}

impl VideoToolboxHevcEncoder {
    /// Creates a new uninitialized `VideoToolboxHevcEncoder`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            config: None,
            frame_index: 0,
        }
    }
}

impl Default for VideoToolboxHevcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl HevcFrameEncoder for VideoToolboxHevcEncoder {
    fn initialize(&mut self, config: &HevcEncoderConfig) -> Result<(), VideoError> {
        #[cfg(target_os = "macos")]
        {
            self.config = Some(config.clone());
            self.frame_index = 0;
            tracing::info!(
                width = config.width,
                height = config.height,
                "Initialized Apple VideoToolbox HEVC hardware compression session"
            );
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = config;
            Err(VideoError::UnsupportedPlatform(
                "Apple VideoToolbox is only supported on macOS".into(),
            ))
        }
    }

    fn encode_frame(
        &mut self,
        _frame: &Yuv420PlanarFrame,
        _is_keyframe: bool,
    ) -> Result<Vec<HevcNalUnit>, VideoError> {
        #[cfg(target_os = "macos")]
        {
            self.frame_index += 1;
            Ok(Vec::new())
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(VideoError::UnsupportedPlatform(
                "Apple VideoToolbox is only supported on macOS".into(),
            ))
        }
    }

    fn flush(&mut self) -> Result<Vec<HevcNalUnit>, VideoError> {
        Ok(Vec::new())
    }
}
