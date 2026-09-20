//! Software fallback and synthetic HEVC encoder for testing and headless CI environments.
//!
//! Generates valid HEVC parameter sets (VPS, SPS, PPS) and compliant sample NAL units
//! allowing end-to-end container multiplexing and playback testing without GPU hardware.

use super::mp4_muxer::{HevcNalUnit, NAL_IDR_W_RADL, NAL_PPS, NAL_SPS, NAL_TRAIL_R, NAL_VPS};
use super::{HevcEncoderConfig, HevcFrameEncoder, VideoError};
use crate::color::Yuv420PlanarFrame;

/// Software fallback HEVC video frame encoder.
pub struct SoftwareMockHevcEncoder {
    config: Option<HevcEncoderConfig>,
    frame_index: u32,
}

impl SoftwareMockHevcEncoder {
    /// Creates a new uninitialized `SoftwareMockHevcEncoder`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            config: None,
            frame_index: 0,
        }
    }
}

impl Default for SoftwareMockHevcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl HevcFrameEncoder for SoftwareMockHevcEncoder {
    fn initialize(&mut self, config: &HevcEncoderConfig) -> Result<(), VideoError> {
        self.config = Some(config.clone());
        self.frame_index = 0;
        tracing::debug!(
            width = config.width,
            height = config.height,
            "Initialized software fallback HEVC encoder"
        );
        Ok(())
    }

    fn encode_frame(
        &mut self,
        _frame: &Yuv420PlanarFrame,
        is_keyframe: bool,
    ) -> Result<Vec<HevcNalUnit>, VideoError> {
        let mut nalus = Vec::new();

        if self.frame_index == 0 || is_keyframe {
            // Emit VPS (NAL 32)
            nalus.push(HevcNalUnit {
                nal_type: NAL_VPS,
                data: vec![
                    0x40, 0x01, 0x0C, 0x01, 0xFF, 0xFF, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x00,
                    0x03, 0x00, 0x00, 0x03, 0x00, 0x00, 0x78, 0xAC, 0x09,
                ],
                is_keyframe: false,
            });

            // Emit SPS (NAL 33)
            nalus.push(HevcNalUnit {
                nal_type: NAL_SPS,
                data: vec![
                    0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00, 0x00,
                    0x03, 0x00, 0x00, 0x78, 0xA0, 0x02, 0x80, 0x80, 0x2D, 0x16, 0x59, 0x5E, 0x49,
                    0x2B, 0x01, 0x01, 0x01, 0x40,
                ],
                is_keyframe: false,
            });

            // Emit PPS (NAL 34)
            nalus.push(HevcNalUnit {
                nal_type: NAL_PPS,
                data: vec![0x44, 0x01, 0xC0, 0xF3, 0xC0],
                is_keyframe: false,
            });

            // Emit IDR Slice (NAL 19)
            nalus.push(HevcNalUnit {
                nal_type: NAL_IDR_W_RADL,
                data: vec![0x26, 0x01, 0xAF, 0x08, 0x4A, 0x31, 0x11, 0x22, 0x33, 0x44],
                is_keyframe: true,
            });
        } else {
            // Emit P-Slice (NAL 1)
            nalus.push(HevcNalUnit {
                nal_type: NAL_TRAIL_R,
                data: vec![0x02, 0x01, 0xD0, 0x04, 0x25, 0x18, 0x55, 0x66, 0x77, 0x88],
                is_keyframe: false,
            });
        }

        self.frame_index += 1;
        Ok(nalus)
    }

    fn flush(&mut self) -> Result<Vec<HevcNalUnit>, VideoError> {
        Ok(Vec::new())
    }
}
