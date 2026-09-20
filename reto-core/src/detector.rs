//! `RoI` detection traits, configuration, diagnostic taps, and baseline detectors.

use crate::color::Bt709LumaConverter;
use crate::error::RoiError;
use crate::geom::{FrameRoi, FrameRoiSet, NormalizedRect, StripOrientation};
use crate::luma::{ScaledLumaImage, DEFAULT_INVERSE_GAMMA, PROJECTION_MAX_DIMENSION};
use crate::stats::AxisStatisticsProfile;
use image::GenericImageView;
use serde::{Deserialize, Serialize};

/// Diagnostic event hook for intermediate algorithm telemetry.
///
/// Allows external observers (debug loggers, UI visualizers, profilers) to intercept
/// key pipeline stages during frame detection without modifying algorithm internals.
pub trait RoiDiagnosticTap: Send + Sync {
    /// Called when the scaled luma representation is constructed during preprocessing.
    fn on_luma_image(&self, _luma: &ScaledLumaImage) {}

    /// Backward-compatible event forwarder for [`RoiDiagnosticTap::on_luma_image`].
    fn on_grayscale_strip(&self, luma: &ScaledLumaImage) {
        self.on_luma_image(luma);
    }

    /// Called when the 2D edge-preserving denoised luma image is computed.
    ///
    /// *Note: Retained for optional developer debug visualization; not dispatched during standard pipeline execution.*
    #[allow(dead_code)]
    fn on_denoised_luma(&self, _luma: &ScaledLumaImage) {}

    /// Backward-compatible event forwarder for [`RoiDiagnosticTap::on_denoised_luma`].
    #[allow(dead_code)]
    fn on_denoised_strip(&self, luma: &ScaledLumaImage) {
        self.on_denoised_luma(luma);
    }

    /// Called when per-pixel statistics along the stacking axis are extracted.
    fn on_axis_statistics(&self, _stats: &AxisStatisticsProfile) {}

    /// Called when projection profiles along stacking direction are computed.
    fn on_projection_profile(&self, _profile: &[f32], _orientation: StripOrientation) {}

    /// Called when gutter candidates or divider coordinates are found.
    fn on_gutter_candidates(&self, _gutters: &[f32]) {}

    /// Called when the finalized `RoI` set has been extracted.
    fn on_rois_detected(&self, _rois: &FrameRoiSet) {}
}

/// Zero-cost No-Op observer tap.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoOpDiagnosticTap;

impl RoiDiagnosticTap for NoOpDiagnosticTap {}

/// Multi-tap dispatcher allowing multiple [`RoiDiagnosticTap`] plugins to observe pipeline events.
#[derive(Default)]
pub struct CompositeDiagnosticTap<'a> {
    taps: Vec<&'a dyn RoiDiagnosticTap>,
}

impl<'a> CompositeDiagnosticTap<'a> {
    /// Creates a new empty `CompositeDiagnosticTap`.
    #[must_use]
    pub const fn new() -> Self {
        Self { taps: Vec::new() }
    }

    /// Adds a diagnostic tap plugin to the composite list.
    pub fn add(&mut self, tap: &'a dyn RoiDiagnosticTap) {
        self.taps.push(tap);
    }
}

impl RoiDiagnosticTap for CompositeDiagnosticTap<'_> {
    fn on_luma_image(&self, luma: &ScaledLumaImage) {
        for tap in &self.taps {
            tap.on_luma_image(luma);
        }
    }

    fn on_denoised_luma(&self, luma: &ScaledLumaImage) {
        for tap in &self.taps {
            tap.on_denoised_luma(luma);
        }
    }

    fn on_axis_statistics(&self, stats: &AxisStatisticsProfile) {
        for tap in &self.taps {
            tap.on_axis_statistics(stats);
        }
    }

    fn on_projection_profile(&self, profile: &[f32], orientation: StripOrientation) {
        for tap in &self.taps {
            tap.on_projection_profile(profile, orientation);
        }
    }

    fn on_gutter_candidates(&self, gutters: &[f32]) {
        for tap in &self.taps {
            tap.on_gutter_candidates(gutters);
        }
    }

    fn on_rois_detected(&self, rois: &FrameRoiSet) {
        for tap in &self.taps {
            tap.on_rois_detected(rois);
        }
    }
}

/// Diagnostic tap plugin that saves intermediate scaled grayscale / luma images to disk.
#[derive(Debug, Clone)]
pub struct SaveLumaDiagnosticTap {
    output_path: std::path::PathBuf,
}

impl SaveLumaDiagnosticTap {
    /// Creates a new `SaveLumaDiagnosticTap`.
    #[must_use]
    pub const fn new(output_path: std::path::PathBuf) -> Self {
        Self { output_path }
    }
}

impl RoiDiagnosticTap for SaveLumaDiagnosticTap {
    fn on_luma_image(&self, luma: &ScaledLumaImage) {
        let gray_img = luma.to_gray_image();
        let _ = gray_img.save(&self.output_path);
    }
}

/// Diagnostic tap plugin that saves intermediate denoised grayscale / luma images to disk.
///
/// *Note: Retained for optional developer debug visualization.*
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SaveDenoisedLumaDiagnosticTap {
    output_path: std::path::PathBuf,
}

#[allow(dead_code)]
impl SaveDenoisedLumaDiagnosticTap {
    /// Creates a new `SaveDenoisedLumaDiagnosticTap`.
    #[must_use]
    pub const fn new(output_path: std::path::PathBuf) -> Self {
        Self { output_path }
    }
}

impl RoiDiagnosticTap for SaveDenoisedLumaDiagnosticTap {
    fn on_denoised_luma(&self, luma: &ScaledLumaImage) {
        let gray_img = luma.to_gray_image();
        let _ = gray_img.save(&self.output_path);
    }
}

/// Configuration options for `RoI` detection algorithms.
///
/// # Examples
/// ```
/// use reto_core::RoiDetectionConfig;
///
/// let config = RoiDetectionConfig::with_expected_frames(3);
/// assert_eq!(config.expected_frames, 3);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoiDetectionConfig {
    /// Expected number of frames (default: 3 for RETO3D Classic).
    pub expected_frames: usize,
    /// Minimum expected frame aspect ratio along stacking direction.
    pub min_frame_aspect_ratio: f32,
    /// Maximum expected frame aspect ratio.
    pub max_frame_aspect_ratio: f32,
}

impl Default for RoiDetectionConfig {
    fn default() -> Self {
        Self {
            expected_frames: 3,
            min_frame_aspect_ratio: 0.4,
            max_frame_aspect_ratio: 1.2,
        }
    }
}

impl RoiDetectionConfig {
    /// Helper to construct detection options specifying expected frame count.
    ///
    /// # Arguments
    /// * `expected_frames` - Number of frames expected in the scanned image layout.
    ///
    /// # Examples
    /// ```
    /// use reto_core::RoiDetectionConfig;
    ///
    /// let config = RoiDetectionConfig::with_expected_frames(4);
    /// assert_eq!(config.expected_frames, 4);
    /// ```
    #[must_use]
    pub const fn with_expected_frames(expected_frames: usize) -> Self {
        Self {
            expected_frames,
            min_frame_aspect_ratio: 0.4,
            max_frame_aspect_ratio: 1.2,
        }
    }
}

/// Core trait for `RoI` detection algorithms.
pub trait RoiDetector: Send + Sync {
    /// Detects sub-frame regions of interest directly from a pre-scaled luma image.
    ///
    /// # Arguments
    /// * `luma` - Pre-computed scaled luma image buffer.
    /// * `config` - Detection parameters.
    /// * `diagnostic_tap` - Optional diagnostic observer tap.
    ///
    /// # Errors
    /// Returns [`RoiError`] if detection fails or configuration parameters are invalid.
    fn detect_luma(
        &self,
        luma: &ScaledLumaImage,
        config: &RoiDetectionConfig,
        diagnostic_tap: Option<&dyn RoiDiagnosticTap>,
    ) -> Result<FrameRoiSet, RoiError>;

    /// Backward-compatible alias for [`RoiDetector::detect_luma`].
    ///
    /// # Errors
    /// Returns [`RoiError`] if detection fails or configuration parameters are invalid.
    fn detect_strip(
        &self,
        luma: &ScaledLumaImage,
        config: &RoiDetectionConfig,
        diagnostic_tap: Option<&dyn RoiDiagnosticTap>,
    ) -> Result<FrameRoiSet, RoiError> {
        self.detect_luma(luma, config, diagnostic_tap)
    }

    /// Detects sub-frame regions of interest on a generic image view by converting it to a luma image.
    ///
    /// # Arguments
    /// * `image` - Source image view.
    /// * `config` - Detection parameters.
    /// * `diagnostic_tap` - Optional diagnostic observer tap.
    ///
    /// # Errors
    /// Returns [`RoiError`] if detection fails, image dimensions are invalid, or frame count is zero.
    fn detect<I: GenericImageView + Sync>(
        &self,
        image: &I,
        config: &RoiDetectionConfig,
        diagnostic_tap: Option<&dyn RoiDiagnosticTap>,
    ) -> Result<FrameRoiSet, RoiError> {
        let luma_img = ScaledLumaImage::from_image(
            image,
            &Bt709LumaConverter::new(),
            PROJECTION_MAX_DIMENSION,
        )?;
        self.detect_luma(&luma_img, config, diagnostic_tap)
    }
}

/// Simple baseline detector that divides the image into `N` equal segments along the stacking axis.
///
/// Serves as the reference/test implementation for the [`RoiDetector`] interface.
///
/// # Examples
/// ```
/// use reto_core::{EvenSplitDetector, RoiDetectionConfig, RoiDetector};
/// use image::{Rgba, RgbaImage};
///
/// let img = RgbaImage::from_pixel(300, 100, Rgba([255, 255, 255, 255]));
/// let detector = EvenSplitDetector::new();
/// let config = RoiDetectionConfig::default();
/// let rois = detector.detect(&img, &config, None).expect("Detection succeeds");
/// assert_eq!(rois.len(), 3);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct EvenSplitDetector;

impl EvenSplitDetector {
    /// Creates a new `EvenSplitDetector`.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl RoiDetector for EvenSplitDetector {
    #[allow(clippy::cast_precision_loss)]
    fn detect_luma(
        &self,
        luma: &ScaledLumaImage,
        config: &RoiDetectionConfig,
        diagnostic_tap: Option<&dyn RoiDiagnosticTap>,
    ) -> Result<FrameRoiSet, RoiError> {
        let size = luma.size();
        let orientation = luma.orientation;
        let delegator = orientation.delegator();
        let n = config.expected_frames;
        if n == 0 {
            return Err(RoiError::ZeroExpectedFrames(0));
        }

        let step = 1.0_f32 / n as f32;
        let mut frames = Vec::with_capacity(n);

        for i in 0..n {
            let stack_min = (i as f32) * step;
            let stack_len = if i == n - 1 {
                1.0_f32 - stack_min
            } else {
                step
            };

            let rect = delegator.build_rect(stack_min, stack_len, 0.0_f32, 1.0_f32);
            let validated_rect = NormalizedRect::new(rect.x, rect.y, rect.width, rect.height)?;

            frames.push(FrameRoi {
                index: i,
                bounds: validated_rect,
                confidence: 1.0_f32,
            });
        }

        let roi_set = FrameRoiSet::new(size.width, size.height, orientation, frames);

        if let Some(tap) = diagnostic_tap {
            tap.on_rois_detected(&roi_set);
        }

        Ok(roi_set)
    }
}

/// Working prototype `RoI` detector shell for pillar and projection statistics algorithm development.
///
/// Currently initializes with an even-separation baseline body to enable incremental step-by-step
/// refinement and visual evaluation.
///
/// # Examples
/// ```
/// use reto_core::{PillarStatsDetector, RoiDetectionConfig, RoiDetector};
/// use image::{Rgba, RgbaImage};
///
/// let img = RgbaImage::from_pixel(300, 100, Rgba([200, 200, 200, 255]));
/// let detector = PillarStatsDetector::new();
/// let config = RoiDetectionConfig::default();
/// let rois = detector.detect(&img, &config, None).expect("Detection succeeds");
/// assert_eq!(rois.len(), 3);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct PillarStatsDetector;

impl PillarStatsDetector {
    /// Creates a new `PillarStatsDetector`.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl RoiDetector for PillarStatsDetector {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::suboptimal_flops,
        clippy::too_many_lines
    )]
    #[tracing::instrument(skip(self, luma, config, diagnostic_tap), level = "debug")]
    fn detect_luma(
        &self,
        luma: &ScaledLumaImage,
        config: &RoiDetectionConfig,
        diagnostic_tap: Option<&dyn RoiDiagnosticTap>,
    ) -> Result<FrameRoiSet, RoiError> {
        let width = luma.width;
        let height = luma.height;
        let orientation = luma.orientation;
        let delegator = orientation.delegator();

        let n = config.expected_frames;
        if n == 0 {
            return Err(RoiError::ZeroExpectedFrames(0));
        }

        // Broadcast intermediate luma image to diagnostic observer plugins
        if let Some(tap) = diagnostic_tap {
            tap.on_luma_image(luma);
        }

        // Apply in-place 2D 3x3 branchless sorting network median denoising
        let mut denoised_luma = luma.clone();
        denoised_luma.median_filter_3x3();

        // Perform baseline subtraction and linear contrast expansion on denoised luma using min_x(P5)
        denoised_luma.inverse_gamma_stretch(DEFAULT_INVERSE_GAMMA);

        // Extract per-pixel cross-axis statistics from the denoised + gamma stretched luma image
        let axis_stats = AxisStatisticsProfile::compute(&denoised_luma);

        // Broadcast contrast spread projection profile to diagnostic taps
        let diff_profile: Vec<f32> = axis_stats
            .p98_minus_p5_series()
            .iter()
            .map(|&v| f32::from(v))
            .collect();
        if let Some(tap) = diagnostic_tap {
            tap.on_projection_profile(&diff_profile, orientation);
        }

        // Evaluate prioritized partition strategies (1. Threshold -> 2. Optimal Grid -> 3. Even Split)
        let partition =
            crate::partition::PrioritizedPartitionEngine::standard().execute(&axis_stats, n);

        if let Some(tap) = diagnostic_tap {
            let gutter_f32: Vec<f32> = partition
                .gutter_centers
                .iter()
                .map(|&c| c as f32 / denoised_luma.major_len as f32)
                .collect();
            tap.on_gutter_candidates(&gutter_f32);
        }

        let total_major = denoised_luma.major_len as f32;
        let mut frames = Vec::with_capacity(n);

        for (i, &(f_start, f_len)) in partition.frame_spans.iter().enumerate() {
            let norm_start = f_start as f32 / total_major;
            let norm_len = (f_len as f32 / total_major).min(1.0_f32 - norm_start);

            let rect = delegator.build_rect(norm_start, norm_len, 0.0_f32, 1.0_f32);
            let validated_rect = NormalizedRect::new(
                rect.x.clamp(0.0, 1.0),
                rect.y.clamp(0.0, 1.0),
                rect.width.clamp(0.0, 1.0),
                rect.height.clamp(0.0, 1.0),
            )?;

            frames.push(FrameRoi {
                index: i,
                bounds: validated_rect,
                confidence: partition.confidence,
            });
        }

        tracing::debug!(
            strategy = partition.strategy_name,
            confidence = format_args!("{:.2}", partition.confidence),
            frame_count = frames.len(),
            spans = ?partition.frame_spans,
            "Configured active frame RoIs"
        );

        let roi_set = FrameRoiSet::new(width, height, orientation, frames);

        if let Some(tap) = diagnostic_tap {
            tap.on_rois_detected(&roi_set);
        }

        Ok(roi_set)
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::suboptimal_flops
)]
mod tests {
    use super::*;
    use crate::geom::Size2D;
    use image::{Rgba, RgbaImage};
    use std::sync::atomic::{AtomicBool, Ordering};

    struct TestTap {
        called: AtomicBool,
    }

    impl RoiDiagnosticTap for TestTap {
        fn on_rois_detected(&self, _rois: &FrameRoiSet) {
            self.called.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn test_even_split_horizontal_default_3() {
        // Horizontally-long strip: 300x100 (aspect ratio 3.0 > 1.0 -> Horizontal)
        let img = RgbaImage::from_pixel(300, 100, Rgba([255, 255, 255, 255]));
        let detector = EvenSplitDetector::new();
        let config = RoiDetectionConfig::default(); // N = 3
        let tap = TestTap {
            called: AtomicBool::new(false),
        };

        let result = detector
            .detect(&img, &config, Some(&tap))
            .expect("Detection should succeed");

        assert_eq!(result.orientation, StripOrientation::Horizontal);
        assert_eq!(result.len(), 3);
        assert!(tap.called.load(Ordering::SeqCst));

        // Frame 0: x in [0.0, 1/3], width = 1/3, y = 0.0, height = 1.0
        let f0 = &result.frames[0];
        assert_eq!(f0.index, 0);
        assert!((f0.bounds.x - 0.0).abs() < 1e-6);
        assert!((f0.bounds.width - (1.0 / 3.0)).abs() < 1e-6);
        assert_eq!(f0.bounds.y, 0.0);
        assert_eq!(f0.bounds.height, 1.0);

        let px0 = f0.bounds.to_pixel_rect(Size2D::new(300, 100));
        assert_eq!(px0.x, 0);
        assert_eq!(px0.width, 100);
        assert_eq!(px0.y, 0);
        assert_eq!(px0.height, 100);

        // Frame 1: x in [1/3, 2/3]
        let f1 = &result.frames[1];
        assert_eq!(f1.index, 1);
        let px1 = f1.bounds.to_pixel_rect(Size2D::new(300, 100));
        assert_eq!(px1.x, 100);
        assert_eq!(px1.width, 100);

        // Frame 2: x in [2/3, 1.0]
        let f2 = &result.frames[2];
        assert_eq!(f2.index, 2);
        let px2 = f2.bounds.to_pixel_rect(Size2D::new(300, 100));
        assert_eq!(px2.x, 200);
        assert_eq!(px2.width, 100);

        // Test zero-copy sub_image view slicing
        let sub0 = result.sub_image(&img, 0).expect("Sub-image 0 should slice");
        assert_eq!(sub0.dimensions(), (100, 100));
        let sub1 = result.sub_image(&img, 1).expect("Sub-image 1 should slice");
        assert_eq!(sub1.dimensions(), (100, 100));
        let sub2 = result.sub_image(&img, 2).expect("Sub-image 2 should slice");
        assert_eq!(sub2.dimensions(), (100, 100));
    }

    #[test]
    fn test_even_split_vertical_n4() {
        // Vertically-long strip: 100x400 (aspect ratio < 1.0 -> Vertical)
        let img = RgbaImage::from_pixel(100, 400, Rgba([255, 255, 255, 255]));
        let detector = EvenSplitDetector::new();
        let config = RoiDetectionConfig::with_expected_frames(4);

        let result = detector
            .detect(&img, &config, None)
            .expect("Detection should succeed");

        assert_eq!(result.orientation, StripOrientation::Vertical);
        assert_eq!(result.len(), 4);

        for (i, frame) in result.frames.iter().enumerate() {
            assert_eq!(frame.index, i);
            assert_eq!(frame.bounds.x, 0.0);
            assert_eq!(frame.bounds.width, 1.0);
            assert!((frame.bounds.y - (i as f32 * 0.25)).abs() < 1e-6);
            assert!((frame.bounds.height - 0.25).abs() < 1e-6);

            let px = frame.bounds.to_pixel_rect(Size2D::new(100, 400));
            assert_eq!(px.x, 0);
            assert_eq!(px.width, 100);
            assert_eq!(px.y, i as u32 * 100);
            assert_eq!(px.height, 100);
        }
    }

    #[test]
    fn test_even_split_square_rejected() {
        let img = RgbaImage::from_pixel(200, 200, Rgba([0, 0, 0, 255]));
        let detector = EvenSplitDetector::new();
        let config = RoiDetectionConfig::default();

        let err = detector.detect(&img, &config, None).unwrap_err();
        assert_eq!(
            err,
            RoiError::SquareImageNotSupported {
                width: 200,
                height: 200
            }
        );
    }
}
