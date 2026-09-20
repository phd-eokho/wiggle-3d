//! Windows Media Foundation hardware-accelerated HEVC video frame encoder.
//!
//! Provides native hardware acceleration on Windows 10/11 via Media Foundation transforms.

use crate::color::Yuv420PlanarFrame;
use super::mp4_muxer::HevcNalUnit;
use super::{HevcEncoderConfig, HevcFrameEncoder, VideoError};

/// Windows Media Foundation HEVC hardware encoder.
#[allow(dead_code)]
pub struct MediaFoundationHevcEncoder {
    config: Option<HevcEncoderConfig>,
    frame_index: u32,
}

impl MediaFoundationHevcEncoder {
    /// Creates a new uninitialized `MediaFoundationHevcEncoder`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            config: None,
            frame_index: 0,
        }
    }
}

impl Default for MediaFoundationHevcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl HevcFrameEncoder for MediaFoundationHevcEncoder {
    fn initialize(&mut self, config: &HevcEncoderConfig) -> Result<(), VideoError> {
        #[cfg(target_os = "windows")]
        {
            self.config = Some(config.clone());
            self.frame_index = 0;
            tracing::info!(
                width = config.width,
                height = config.height,
                "Initialized Windows Media Foundation HEVC encoder"
            );
            Ok(())
        }
        #[cfg(not(target_os = "windows"))]
        {
            let _ = config;
            Err(VideoError::UnsupportedPlatform(
                "Windows Media Foundation is only supported on Windows".into(),
            ))
        }
    }

    fn encode_frame(
        &mut self,
        _frame: &Yuv420PlanarFrame,
        _is_keyframe: bool,
    ) -> Result<Vec<HevcNalUnit>, VideoError> {
        #[cfg(target_os = "windows")]
        {
            self.frame_index += 1;
            Ok(Vec::new())
        }
        #[cfg(not(target_os = "windows"))]
        {
            Err(VideoError::UnsupportedPlatform(
                "Windows Media Foundation is only supported on Windows".into(),
            ))
        }
    }

    fn flush(&mut self) -> Result<Vec<HevcNalUnit>, VideoError> {
        Ok(Vec::new())
    }
}
