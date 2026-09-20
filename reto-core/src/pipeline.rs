//! Core pipeline orchestrator for processing verified image batches.
//!
//! Exposes supported format metadata and batch processing operations while leaving
//! front-end file discovery and path verification to callers (CLI, TUI).

use crate::color::Bt709LumaConverter;
use crate::detector::{PillarStatsDetector, RoiDetectionConfig, RoiDetector};
use crate::error::{Error, Result};
use crate::face::RetinaFaceDetector;
use crate::feature::AlignmentDiagnosticTap;
use crate::geom::FrameRoiSet;
use crate::luma::{ScaledLumaImage, PROJECTION_MAX_DIMENSION};
use crate::visualizer::RoiVisualizer;
use image::{DynamicImage, Rgba};
use std::path::{Path, PathBuf};

/// Default supported input image file extensions for Reto-Split batch scanning.
pub const SUPPORTED_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"];

/// Checks if the given path has a supported image extension according to core format support.
///
/// # Arguments
/// * `path` - The file path to test.
///
/// # Examples
/// ```
/// use reto_core::is_supported_image;
/// use std::path::Path;
///
/// assert!(is_supported_image(Path::new("scan.JPG")));
/// assert!(!is_supported_image(Path::new("scan.txt")));
/// ```
#[must_use]
pub fn is_supported_image<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref()
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            SUPPORTED_EXTENSIONS
                .iter()
                .any(|&supported| ext.eq_ignore_ascii_case(supported))
        })
}

/// Independent processing context for a single film scan image.
///
/// Encapsulates per-image source metadata, decoded buffers, and intermediate
/// analysis state across processing pipeline stages to ensure thread safety
/// and enable parallel execution without data races.
///
/// # Examples
/// ```
/// use reto_core::ImageItemContext;
/// use std::path::PathBuf;
///
/// let item = ImageItemContext::new(PathBuf::from("scan.jpg"), PathBuf::from("output"));
/// assert_eq!(item.file_stem(), "scan");
/// ```
#[derive(Debug, Clone)]
pub struct ImageItemContext {
    /// Path to the source scan image file.
    source_path: PathBuf,
    /// Destination directory for intermediate artifacts and results.
    output_dir: PathBuf,
    /// Loaded source image buffer.
    image: Option<DynamicImage>,
    /// Intermediate detection results (frame regions of interest).
    rois: Option<FrameRoiSet>,
    /// Scaled single-channel luma image buffer for reusable vision analysis.
    luma_image: Option<ScaledLumaImage>,
    /// Extracted feature keypoints for each detected sub-frame.
    features: Option<Vec<crate::feature::FeatureFrame>>,
    /// Target execution device for model inference.
    device: crate::feature::BackendDevice,
    /// Configuration for Wiggle GIF generation.
    gif_config: crate::gif::WiggleGifConfig,
    /// Optional configuration for HEVC MP4 video generation.
    video_config: Option<crate::video::WiggleVideoConfig>,
    /// Whether debug visualization mode is enabled.
    debug: bool,
}

impl ImageItemContext {
    /// Initializes a new image item context from a source path and destination directory.
    ///
    /// # Arguments
    /// * `source_path` - Path to the input image file.
    /// * `output_dir` - Destination directory for output artifacts.
    #[must_use]
    pub const fn new(source_path: PathBuf, output_dir: PathBuf) -> Self {
        Self {
            source_path,
            output_dir,
            image: None,
            rois: None,
            luma_image: None,
            features: None,
            device: crate::feature::BackendDevice::Auto,
            gif_config: crate::gif::WiggleGifConfig::new(crate::gif::DEFAULT_FRAME_DELAY_MS),
            video_config: None,
            debug: false,
        }
    }

    /// Returns the source file path.
    #[must_use]
    pub fn source_path(&self) -> &Path {
        &self.source_path
    }

    /// Returns the output directory path.
    #[must_use]
    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }

    /// Returns the file stem of the source image for naming outputs.
    #[must_use]
    pub fn file_stem(&self) -> &str {
        self.source_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("strip")
    }

    /// Sets the decoded image buffer.
    pub fn set_image(&mut self, img: DynamicImage) {
        self.image = Some(img);
    }

    /// Reference to the decoded image if loaded.
    #[must_use]
    pub const fn image(&self) -> Option<&DynamicImage> {
        self.image.as_ref()
    }

    /// Takes the decoded image buffer, leaving `None` in its place.
    pub const fn take_image(&mut self) -> Option<DynamicImage> {
        self.image.take()
    }

    /// Sets the detected frame `RoIs`.
    pub fn set_rois(&mut self, rois: FrameRoiSet) {
        self.rois = Some(rois);
    }

    /// Reference to detected `RoIs` if present.
    #[must_use]
    pub const fn rois(&self) -> Option<&FrameRoiSet> {
        self.rois.as_ref()
    }

    /// Takes the detected frame `RoIs`, leaving `None` in its place.
    pub const fn take_rois(&mut self) -> Option<FrameRoiSet> {
        self.rois.take()
    }

    /// Sets the scaled luma image buffer.
    pub fn set_luma_image(&mut self, luma: ScaledLumaImage) {
        self.luma_image = Some(luma);
    }

    /// Backward-compatible alias for [`ImageItemContext::set_luma_image`].
    pub fn set_luma_strip(&mut self, luma: ScaledLumaImage) {
        self.set_luma_image(luma);
    }

    /// Reference to the scaled luma image if generated.
    #[must_use]
    pub const fn luma_image(&self) -> Option<&ScaledLumaImage> {
        self.luma_image.as_ref()
    }

    /// Backward-compatible alias for [`ImageItemContext::luma_image`].
    #[must_use]
    pub const fn luma_strip(&self) -> Option<&ScaledLumaImage> {
        self.luma_image()
    }

    /// Takes the scaled luma image buffer, leaving `None` in its place.
    pub const fn take_luma_image(&mut self) -> Option<ScaledLumaImage> {
        self.luma_image.take()
    }

    /// Backward-compatible alias for [`ImageItemContext::take_luma_image`].
    pub const fn take_luma_strip(&mut self) -> Option<ScaledLumaImage> {
        self.take_luma_image()
    }

    /// Sets the extracted feature frames.
    pub fn set_features(&mut self, features: Vec<crate::feature::FeatureFrame>) {
        self.features = Some(features);
    }

    /// Reference to extracted feature frames if present.
    #[must_use]
    pub fn features(&self) -> Option<&[crate::feature::FeatureFrame]> {
        self.features.as_deref()
    }

    /// Returns the target execution device for model inference.
    #[must_use]
    pub const fn device(&self) -> crate::feature::BackendDevice {
        self.device
    }

    /// Sets the target execution device for model inference.
    pub const fn set_device(&mut self, device: crate::feature::BackendDevice) {
        self.device = device;
    }

    /// Returns the Wiggle GIF generation configuration.
    #[must_use]
    pub const fn gif_config(&self) -> crate::gif::WiggleGifConfig {
        self.gif_config
    }

    /// Sets the Wiggle GIF generation configuration.
    pub const fn set_gif_config(&mut self, gif_config: crate::gif::WiggleGifConfig) {
        self.gif_config = gif_config;
    }

    /// Returns the optional HEVC MP4 video generation configuration.
    #[must_use]
    pub const fn video_config(&self) -> Option<crate::video::WiggleVideoConfig> {
        self.video_config
    }

    /// Sets the HEVC MP4 video generation configuration.
    pub const fn set_video_config(
        &mut self,
        video_config: Option<crate::video::WiggleVideoConfig>,
    ) {
        self.video_config = video_config;
    }

    /// Builder method to set HEVC MP4 video generation configuration.
    #[must_use]
    pub const fn with_video_config(
        mut self,
        video_config: Option<crate::video::WiggleVideoConfig>,
    ) -> Self {
        self.video_config = video_config;
        self
    }

    /// Returns whether debug visualization mode is enabled.
    #[must_use]
    pub const fn debug(&self) -> bool {
        self.debug
    }

    /// Sets whether debug visualization mode is enabled.
    pub const fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Builder method to set debug visualization mode.
    #[must_use]
    pub const fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }
}

/// Progress event notification emitted during batch processing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgressEvent<'a> {
    /// Processing started for an image item.
    ItemStarted {
        /// File stem of the image item being processed.
        file_stem: &'a str,
        /// Current 1-based index in the batch.
        index: usize,
        /// Total number of items in the batch.
        total: usize,
    },
    /// Processing completed for an image item.
    ItemCompleted {
        /// File stem of the image item processed.
        file_stem: &'a str,
        /// Current 1-based index in the batch.
        index: usize,
        /// Total number of items in the batch.
        total: usize,
        /// Whether processing succeeded without error.
        success: bool,
    },
}

/// Thread-safe observer trait for receiving pipeline execution progress.
pub trait ProgressObserver: std::fmt::Debug + Send + Sync {
    /// Handles an incoming pipeline progress event.
    fn on_progress(&self, event: ProgressEvent<'_>);
}

/// A validated batch request containing verified file paths ready for processing.
///
/// # Examples
/// ```
/// use reto_core::{BackendDevice, BatchProcessingRequest};
/// use std::path::PathBuf;
///
/// let req = BatchProcessingRequest::new(vec![PathBuf::from("img.png")], PathBuf::from("dist"), true);
/// assert!(req.debug);
/// assert_eq!(req.device, BackendDevice::Auto);
/// ```
#[derive(Debug, Clone)]
pub struct BatchProcessingRequest {
    /// Verified image file paths.
    pub files: Vec<PathBuf>,
    /// Destination directory for output results and visual debug artifacts.
    pub output_dir: PathBuf,
    /// Whether debug visualization mode is enabled.
    pub debug: bool,
    /// Compute device / execution provider for model inference.
    pub device: crate::feature::BackendDevice,
    /// Configuration for Wiggle GIF generation.
    pub gif_config: crate::gif::WiggleGifConfig,
    /// Optional configuration for HEVC MP4 video generation.
    pub video_config: Option<crate::video::WiggleVideoConfig>,
    /// Optional observer for receiving progress notifications.
    pub progress_observer: Option<std::sync::Arc<dyn ProgressObserver>>,
}

impl BatchProcessingRequest {
    /// Creates a new `BatchProcessingRequest`.
    ///
    /// # Arguments
    /// * `files` - List of verified image file paths to process.
    /// * `output_dir` - Destination directory for output artifacts.
    /// * `debug` - Flag to enable debug visualization output.
    #[must_use]
    pub const fn new(files: Vec<PathBuf>, output_dir: PathBuf, debug: bool) -> Self {
        Self {
            files,
            output_dir,
            debug,
            device: crate::feature::BackendDevice::Auto,
            gif_config: crate::gif::WiggleGifConfig::new(crate::gif::DEFAULT_FRAME_DELAY_MS),
            video_config: None,
            progress_observer: None,
        }
    }

    /// Sets the compute device for inference.
    #[must_use]
    pub const fn with_device(mut self, device: crate::feature::BackendDevice) -> Self {
        self.device = device;
        self
    }

    /// Sets the Wiggle GIF generation configuration.
    #[must_use]
    pub const fn with_gif_config(mut self, gif_config: crate::gif::WiggleGifConfig) -> Self {
        self.gif_config = gif_config;
        self
    }

    /// Sets the HEVC MP4 video generation configuration.
    #[must_use]
    pub const fn with_video_config(
        mut self,
        video_config: Option<crate::video::WiggleVideoConfig>,
    ) -> Self {
        self.video_config = video_config;
        self
    }

    /// Returns the HEVC MP4 video generation configuration, if enabled.
    #[must_use]
    pub const fn video_config(&self) -> Option<crate::video::WiggleVideoConfig> {
        self.video_config
    }

    /// Attaches a progress observer to receive batch processing execution events.
    #[must_use]
    pub fn with_progress_observer(
        mut self,
        observer: std::sync::Arc<dyn ProgressObserver>,
    ) -> Self {
        self.progress_observer = Some(observer);
        self
    }

    /// Creates independent per-image processing contexts from this batch request.
    #[must_use]
    pub fn create_item_contexts(&self) -> Vec<ImageItemContext> {
        self.files
            .iter()
            .map(|path| {
                let mut ctx = ImageItemContext::new(path.clone(), self.output_dir.clone());
                ctx.set_device(self.device);
                ctx.set_gif_config(self.gif_config);
                ctx.set_video_config(self.video_config);
                ctx.set_debug(self.debug);
                ctx
            })
            .collect()
    }
}

/// Outcome status of processing an individual image item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemProcessingOutcome {
    /// Item was processed successfully.
    Success,
    /// Item processing encountered an error.
    Failure,
}

/// Execution summary of batch processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProcessSummary {
    /// Number of verified input image files received.
    pub total_input: usize,
    /// Number of images successfully processed.
    pub successful_count: usize,
    /// Number of images that encountered errors.
    pub failed_count: usize,
}

/// Processes a single image item context through detection, visualization, and crop export stages.
///
/// # Arguments
/// * `item` - The image item context to process.
/// * `config` - Detection parameters.
///
/// # Errors
/// Returns [`Error`] if image decoding, `RoI` detection, or file saving fails.
#[tracing::instrument(level = "debug", skip_all)]
fn load_image(item: &mut ImageItemContext) -> Result<DynamicImage> {
    match item.image.take() {
        Some(img) => Ok(img),
        None => Ok(image::open(item.source_path())?),
    }
}

#[tracing::instrument(level = "debug", skip_all)]
fn prepare_luma(dynamic_img: &DynamicImage) -> Result<ScaledLumaImage> {
    ScaledLumaImage::from_image(
        dynamic_img,
        &Bt709LumaConverter::new(),
        PROJECTION_MAX_DIMENSION,
    )
    .map_err(Into::into)
}

#[tracing::instrument(level = "debug", skip(luma, config, tap))]
fn detect_rois(
    luma: &ScaledLumaImage,
    config: &RoiDetectionConfig,
    tap: Option<&dyn crate::detector::RoiDiagnosticTap>,
) -> Result<FrameRoiSet> {
    let detector = PillarStatsDetector::new();
    let rois = detector.detect_luma(luma, config, tap)?;
    tracing::debug!(
        orientation = ?rois.orientation,
        frame_count = rois.len(),
        "Detected frame RoIs"
    );
    let bounds: Vec<_> = rois.frames.iter().map(|f| &f.bounds).collect();
    tracing::debug!(
        bounds = ?bounds,
        "Frame RoI bounds configured"
    );
    Ok(rois)
}

#[tracing::instrument(level = "debug", skip(dynamic_img, rois))]
fn render_roi_overlay(
    dynamic_img: &DynamicImage,
    rois: &FrameRoiSet,
    output_dir: &Path,
    file_stem: &str,
) -> (PathBuf, DynamicImage) {
    let border_color = Rgba([0, 255, 128, 255]); // High-contrast emerald green
    let roi_overlay = RoiVisualizer::render_overlay(dynamic_img, rois, border_color, 4);
    let overlay_path = output_dir.join(format!("{file_stem}_roi_overlay.png"));
    (overlay_path, DynamicImage::ImageRgba8(roi_overlay))
}

#[allow(clippy::type_complexity)]
#[tracing::instrument(level = "debug", skip_all)]
fn match_features(
    file_stem: &str,
    features: &[crate::feature::FeatureFrame],
    orientation: crate::geom::StripOrientation,
    rois: &FrameRoiSet,
    overlay_path: Option<PathBuf>,
    roi_overlay: Option<DynamicImage>,
    frame_faces: &[crate::visualizer::FrameFaceRecord],
) -> Result<(
    Vec<crate::feature::FeatureTriplet>,
    Vec<(crate::feature::FramePair, Vec<crate::feature::FeatureMatch>)>,
)> {
    use crate::feature::{FeatureMatcher, SuperPointDescriptorMatcher, TripletConsistencyConfig};

    let mut extracted_triplets = Vec::new();
    let mut extracted_pairs = Vec::new();

    // If at least 2 frames exist, match across frames
    if features.len() >= 2 {
        let matcher = SuperPointDescriptorMatcher::default();
        let consistency_config = TripletConsistencyConfig::with_orientation(orientation);

        let tap = match (overlay_path, roi_overlay) {
            (Some(path), Some(overlay)) => {
                let t = crate::visualizer::SaveMatchesDiagnosticTap::new(Some(path), None, overlay)
                    .with_rois(rois.clone());
                t.add_faces(frame_faces);
                for (idx, frame) in features.iter().enumerate() {
                    t.on_features_extracted(idx, frame);
                }
                Some(t)
            }
            _ => None,
        };

        if features.len() >= 3 {
            match matcher.extract_consistent_triplets(
                features,
                &consistency_config,
                tap.as_ref().map(|t| t as &dyn AlignmentDiagnosticTap),
            ) {
                Ok(triplets) => {
                    tracing::info!(
                        file = %file_stem,
                        triplet_count = triplets.len(),
                        "Extracted depth-consistent feature triplets across 3 views"
                    );
                    extracted_triplets = triplets;
                }
                Err(e) => {
                    tracing::warn!(file = %file_stem, error = %e, "Failed to extract feature triplets; falling back to pairwise matching");
                }
            }
        } else {
            let pair_matches = matcher.match_pair_bidirectional(&features[0], &features[1])?;
            if let Some(ref t) = tap {
                t.on_matches_found(pair_matches.pair, &pair_matches.matches);
            }
            extracted_pairs.push((pair_matches.pair, pair_matches.matches));
        }

        if let Some(t) = tap {
            t.finish()?;
        }
    } else if let (Some(path), Some(overlay)) = (overlay_path, roi_overlay) {
        // Fallback: draw keypoints on overlay if fewer than 2 frames and debug overlay is requested
        let feat_tap = crate::visualizer::SaveFeaturesDiagnosticTap::new(
            path,
            overlay,
            Rgba([255, 64, 128, 255]),
        )
        .with_rois(rois.clone());
        for (idx, frame) in features.iter().enumerate() {
            feat_tap.on_features_extracted(idx, frame);
        }
        feat_tap.finish()?;
    }

    Ok((extracted_triplets, extracted_pairs))
}

/// Intermediate payload produced by Stage 1 (image loading, scaling, RoI detection, and sub-frame extraction).
struct VisionStagePayload {
    item: ImageItemContext,
    file_stem: String,
    output_dir: PathBuf,
    device: crate::feature::BackendDevice,
    dynamic_img: DynamicImage,
    luma_image: ScaledLumaImage,
    rois: FrameRoiSet,
    sub_frame_crops: Vec<DynamicImage>,
    overlay_path: Option<PathBuf>,
    roi_overlay: Option<DynamicImage>,
}

/// Intermediate payload produced by Stage 2 (parallel neural inference, feature matching, and alignment cropping).
struct AlignedStagePayload {
    item: ImageItemContext,
    file_stem: String,
    output_dir: PathBuf,
    dynamic_img: DynamicImage,
    rois: FrameRoiSet,
    aligned_frames: Option<Vec<image::RgbaImage>>,
}

/// Intermediate payload produced by Stage 3 (color quantization, GIF encoding, and optional HEVC MP4 video generation).
struct EncodedStagePayload {
    item: ImageItemContext,
    gif_output: Option<(PathBuf, Vec<u8>)>,
    video_output: Option<(PathBuf, Vec<u8>)>,
}

/// Stage 1: Ingestion, format decoding, scaled luma generation, and RoI detection.
fn stage_decode_and_roi(
    mut item: ImageItemContext,
    config: &RoiDetectionConfig,
) -> Result<VisionStagePayload> {
    let file_stem = item.file_stem().to_string();
    let output_dir = item.output_dir().to_path_buf();
    let device = item.device();

    let dynamic_img = load_image(&mut item)?;
    let luma_image = prepare_luma(&dynamic_img)?;
    let rois = detect_rois(&luma_image, config, None)?;

    let sub_frame_crops = RoiVisualizer::extract_frame_images(&dynamic_img, &rois)?;

    let (overlay_path, roi_overlay) = if item.debug() {
        let (path, overlay) = render_roi_overlay(&dynamic_img, &rois, &output_dir, &file_stem);
        (Some(path), Some(overlay))
    } else {
        (None, None)
    };

    Ok(VisionStagePayload {
        item,
        file_stem,
        output_dir,
        device,
        dynamic_img,
        luma_image,
        rois,
        sub_frame_crops,
        overlay_path,
        roi_overlay,
    })
}

fn detect_faces_for_sub_frames(
    crops: &[DynamicImage],
    file_stem: &str,
) -> Vec<crate::visualizer::FrameFaceRecord> {
    if crops.is_empty() {
        return Vec::new();
    }
    match RetinaFaceDetector::default_engine() {
        Ok(face_detector) => match face_detector.detect_faces_for_rois(crops) {
            Ok(records) => {
                let total_faces: usize = records.iter().map(|(_, list, _)| list.len()).sum();
                if total_faces > 0 {
                    tracing::debug!(
                        file = %file_stem,
                        faces = total_faces,
                        "Detected faces across sub-frames"
                    );
                }
                records
            }
            Err(e) => {
                tracing::warn!(file = %file_stem, error = %e, "Failed to run face detection on sub-frames");
                Vec::new()
            }
        },
        Err(e) => {
            tracing::warn!(file = %file_stem, error = %e, "Failed to initialize RetinaFace detector");
            Vec::new()
        }
    }
}

fn extract_dominant_face_bbox(
    frame_faces: &[crate::visualizer::FrameFaceRecord],
) -> Option<crate::geom::NormalizedRect> {
    frame_faces
        .iter()
        .find(|(f_idx, _, dom_idx)| *f_idx == 1 && dom_idx.is_some())
        .and_then(|(_, detections, dom_idx)| {
            dom_idx.and_then(|idx| detections.get(idx).map(|f| f.bbox))
        })
        .or_else(|| {
            frame_faces.iter().find_map(|(_, detections, dom_idx)| {
                dom_idx.and_then(|idx| detections.get(idx).map(|f| f.bbox))
            })
        })
}

fn compute_alignment_shifts(
    features: &[crate::feature::FeatureFrame],
    triplets: &[crate::feature::FeatureTriplet],
    pairs: &[(crate::feature::FramePair, Vec<crate::feature::FeatureMatch>)],
    dominant_face_bbox: Option<crate::geom::NormalizedRect>,
) -> [(f32, f32); 3] {
    if !triplets.is_empty() {
        crate::gif::WiggleAligner::compute_depth_surface_shifts_from_triplets_with_face_priority(
            features,
            triplets,
            dominant_face_bbox,
            crate::gif::DEFAULT_DISPARITY_BIN_SIZE_PX,
            crate::gif::DEFAULT_CLUSTER_TOLERANCE_PX,
        )
    } else if !pairs.is_empty() {
        crate::gif::WiggleAligner::compute_depth_surface_shifts_from_pairs_with_face_priority(
            features,
            pairs,
            dominant_face_bbox,
            crate::gif::DEFAULT_DISPARITY_BIN_SIZE_PX,
            crate::gif::DEFAULT_CLUSTER_TOLERANCE_PX,
        )
    } else {
        [(0.0, 0.0); 3]
    }
}

#[allow(clippy::cast_precision_loss)]
fn align_and_crop_payload(
    crops: &[DynamicImage],
    orig_w: u32,
    orig_h: u32,
    luma_w: u32,
    luma_h: u32,
    shifts: &[(f32, f32); 3],
) -> Result<Option<Vec<image::RgbaImage>>> {
    if crops.is_empty() {
        return Ok(None);
    }
    let scale_x = orig_w as f32 / luma_w as f32;
    let scale_y = orig_h as f32 / luma_h as f32;
    let scaled_shifts: Vec<(f32, f32)> = shifts
        .iter()
        .map(|&(dx, dy)| (dx * scale_x, dy * scale_y))
        .collect();
    let aligned = crate::gif::WiggleAligner::align_and_crop(crops, &scaled_shifts)?;
    Ok(Some(aligned))
}

/// Stage 2: Concurrent neural face/feature detection, descriptor matching, and sub-frame alignment.
fn stage_vision_and_align(mut payload: VisionStagePayload) -> Result<AlignedStagePayload> {
    // Run Face Detection and SuperPoint Feature Extraction concurrently via rayon::join
    let (frame_faces, feature_frames_res) = rayon::join(
        || detect_faces_for_sub_frames(&payload.sub_frame_crops, &payload.file_stem),
        || {
            use crate::feature::{PointDetector, SuperPointConfig, SuperPointDetector};
            let point_detector = SuperPointDetector::new(SuperPointConfig {
                device: payload.device,
                ..Default::default()
            });
            point_detector.detect_luma_all(&payload.luma_image, &payload.rois.frames, None)
        },
    );

    let features = feature_frames_res?;
    let counts: Vec<usize> = features
        .iter()
        .map(crate::feature::FeatureFrame::len)
        .collect();
    tracing::info!(
        file = %payload.file_stem,
        keypoints = ?counts,
        "Detected frame keypoints"
    );

    let dominant_face_bbox = extract_dominant_face_bbox(&frame_faces);

    let (triplets, pairs) = match_features(
        &payload.file_stem,
        &features,
        payload.luma_image.orientation,
        &payload.rois,
        payload.overlay_path,
        payload.roi_overlay,
        &frame_faces,
    )?;

    let shifts = compute_alignment_shifts(&features, &triplets, &pairs, dominant_face_bbox);
    let aligned_frames = align_and_crop_payload(
        &payload.sub_frame_crops,
        payload.dynamic_img.width(),
        payload.dynamic_img.height(),
        payload.luma_image.width,
        payload.luma_image.height,
        &shifts,
    )?;

    payload.item.set_luma_image(payload.luma_image);
    payload.item.set_features(features);

    Ok(AlignedStagePayload {
        item: payload.item,
        file_stem: payload.file_stem,
        output_dir: payload.output_dir,
        dynamic_img: payload.dynamic_img,
        rois: payload.rois,
        aligned_frames,
    })
}

/// Stage 3: In-memory NeuQuant color quantization and GIF byte serialization, or MP4 video encoding.
fn stage_quantize_and_encode(mut payload: AlignedStagePayload) -> Result<EncodedStagePayload> {
    let video_output = if let (Some(ref aligned_frames), Some(video_cfg)) =
        (&payload.aligned_frames, payload.item.video_config())
    {
        let video_path = payload
            .output_dir
            .join(format!("{}_wiggle.mp4", payload.file_stem));
        let mut video_bytes = Vec::new();
        crate::video::WiggleVideoBuilder::build_wiggle_video(
            aligned_frames,
            &video_cfg,
            &mut video_bytes,
        )?;
        Some((video_path, video_bytes))
    } else {
        None
    };

    let gif_output = if video_output.is_none() {
        if let Some(ref aligned_frames) = payload.aligned_frames {
            let gif_path = payload
                .output_dir
                .join(format!("{}_wiggle.gif", payload.file_stem));
            let mut gif_bytes = Vec::new();
            crate::gif::WiggleGifBuilder::build_wiggle_gif(
                aligned_frames,
                &payload.item.gif_config(),
                &mut gif_bytes,
            )?;
            Some((gif_path, gif_bytes))
        } else {
            None
        }
    } else {
        None
    };

    payload.item.set_image(payload.dynamic_img);
    payload.item.set_rois(payload.rois);

    Ok(EncodedStagePayload {
        item: payload.item,
        gif_output,
        video_output,
    })
}

/// Stage 4: Disk writer that persists serialized GIF / MP4 bytes and completes image item processing.
fn stage_write_output(payload: EncodedStagePayload) -> Result<ImageItemContext> {
    if let Some((gif_path, gif_bytes)) = payload.gif_output {
        std::fs::write(&gif_path, gif_bytes)?;
        tracing::info!(gif_path = ?gif_path, "Saved Wiggle 3D GIF");
    }
    if let Some((video_path, video_bytes)) = payload.video_output {
        std::fs::write(&video_path, video_bytes)?;
        tracing::info!(video_path = ?video_path, "Saved Wiggle 3D TrueColor HEVC MP4 video");
    }
    tracing::info!(file = ?payload.item.source_path(), "Processed image item successfully");
    Ok(payload.item)
}

/// Processes a single image item context through detection, visualization, and feature extraction.
///
/// # Arguments
/// * `item` - The image item context to process.
/// * `config` - Detection parameters.
///
/// # Errors
/// Returns [`Error`] if image loading, `RoI` detection, or feature extraction fails.
#[tracing::instrument(level = "debug", skip_all)]
pub fn process_item(
    item: ImageItemContext,
    config: &RoiDetectionConfig,
) -> Result<ImageItemContext> {
    let vision_payload = stage_decode_and_roi(item, config)?;
    let aligned_payload = stage_vision_and_align(vision_payload)?;
    let encoded_payload = stage_quantize_and_encode(aligned_payload)?;
    stage_write_output(encoded_payload)
}

/// Processes a single film strip image: detects `RoIs`, generates visual overlay, and exports sub-frame crops.
///
/// # Arguments
/// * `file_path` - Path to the input film strip scan image.
/// * `output_dir` - Destination directory where overlay and cropped frame files are written.
/// * `config` - Detection parameters.
///
/// # Errors
/// Returns [`Error`] if image decoding, `RoI` detection, or file saving fails.
pub fn process_single_image(
    file_path: &Path,
    output_dir: &Path,
    config: &RoiDetectionConfig,
) -> Result<FrameRoiSet> {
    let item = ImageItemContext::new(file_path.to_path_buf(), output_dir.to_path_buf());
    let mut processed = process_item(item, config)?;
    processed.take_rois().ok_or_else(|| {
        Error::Unknown("Missing RoI results in processed context".to_string())
    })
}

/// Drives processing for a verified batch of image files.
///
/// Executes per-item processing via a 4-stage pipelined streaming dataflow with bounded
/// channel backpressure to overlap disk I/O, neural inference, and color quantization.
/// Assumes file list verification has already been conducted by the front-end.
///
/// # Arguments
/// * `request` - Batch processing specifications including files, destination, and debug flags.
///
/// # Errors
/// Returns [`Error::Io`] if output directory creation fails.
type Stage1Result = std::result::Result<(usize, String, VisionStagePayload), (usize, String, Error)>;
type Stage2Result = std::result::Result<(usize, String, AlignedStagePayload), (usize, String, Error)>;
type Stage3Result = std::result::Result<(usize, String, EncodedStagePayload), (usize, String, Error)>;

fn spawn_ingestion_stage(
    items: Vec<ImageItemContext>,
    config: RoiDetectionConfig,
    observer: Option<std::sync::Arc<dyn ProgressObserver>>,
    tx: crossbeam_channel::Sender<Stage1Result>,
) {
    let total = items.len();
    std::thread::spawn(move || {
        for (idx, item) in items.into_iter().enumerate() {
            let file_stem = item.file_stem().to_string();
            if let Some(ref obs) = observer {
                obs.on_progress(ProgressEvent::ItemStarted {
                    file_stem: &file_stem,
                    index: idx + 1,
                    total,
                });
            }

            let result = stage_decode_and_roi(item, &config)
                .map(|payload| (idx + 1, file_stem.clone(), payload))
                .map_err(|e| (idx + 1, file_stem, e));

            if tx.send(result).is_err() {
                break;
            }
        }
    });
}

fn spawn_vision_workers(
    rx: &crossbeam_channel::Receiver<Stage1Result>,
    tx: &crossbeam_channel::Sender<Stage2Result>,
    concurrency_limit: usize,
) {
    for _ in 0..concurrency_limit {
        let rx_worker = rx.clone();
        let tx_worker = tx.clone();
        std::thread::spawn(move || {
            while let Ok(msg) = rx_worker.recv() {
                let result = match msg {
                    Ok((idx, stem, payload)) => match stage_vision_and_align(payload) {
                        Ok(aligned) => Ok((idx, stem, aligned)),
                        Err(e) => Err((idx, stem, e)),
                    },
                    Err(err) => Err(err),
                };
                if tx_worker.send(result).is_err() {
                    break;
                }
            }
        });
    }
}

fn spawn_encoding_workers(
    rx: &crossbeam_channel::Receiver<Stage2Result>,
    tx: &crossbeam_channel::Sender<Stage3Result>,
    concurrency_limit: usize,
) {
    for _ in 0..concurrency_limit {
        let rx_worker = rx.clone();
        let tx_worker = tx.clone();
        std::thread::spawn(move || {
            while let Ok(msg) = rx_worker.recv() {
                let result = match msg {
                    Ok((idx, stem, payload)) => match stage_quantize_and_encode(payload) {
                        Ok(encoded) => Ok((idx, stem, encoded)),
                        Err(e) => Err((idx, stem, e)),
                    },
                    Err(err) => Err(err),
                };
                if tx_worker.send(result).is_err() {
                    break;
                }
            }
        });
    }
}

fn drain_and_write_outputs(
    rx: &crossbeam_channel::Receiver<Stage3Result>,
    total: usize,
    observer: Option<&std::sync::Arc<dyn ProgressObserver>>,
) -> (usize, usize) {
    let mut successful_count = 0;
    let mut failed_count = 0;

    while let Ok(msg) = rx.recv() {
        match msg {
            Ok((idx, stem, payload)) => match stage_write_output(payload) {
                Ok(_) => {
                    if let Some(obs) = observer {
                        obs.on_progress(ProgressEvent::ItemCompleted {
                            file_stem: &stem,
                            index: idx,
                            total,
                            success: true,
                        });
                    }
                    successful_count += 1;
                }
                Err(e) => {
                    tracing::error!(file = %stem, error = ?e, "Failed writing output files");
                    if let Some(obs) = observer {
                        obs.on_progress(ProgressEvent::ItemCompleted {
                            file_stem: &stem,
                            index: idx,
                            total,
                            success: false,
                        });
                    }
                    failed_count += 1;
                }
            },
            Err((idx, stem, err)) => {
                tracing::error!(file = %stem, error = ?err, "Failed processing image");
                if let Some(obs) = observer {
                    obs.on_progress(ProgressEvent::ItemCompleted {
                        file_stem: &stem,
                        index: idx,
                        total,
                        success: false,
                    });
                }
                failed_count += 1;
            }
        }
    }
    (successful_count, failed_count)
}

/// Drives processing for a verified batch of image files.
///
/// Executes per-item processing via a 4-stage pipelined streaming dataflow with bounded
/// channel backpressure to overlap disk I/O, neural inference, and color quantization.
/// Assumes file list verification has already been conducted by the front-end.
///
/// # Arguments
/// * `request` - Batch processing specifications including files, destination, and debug flags.
///
/// # Errors
/// Returns [`Error`] if output directory creation fails or an encoder cannot be initialized.
#[tracing::instrument(level = "debug", skip_all)]
pub fn run_batch(request: &BatchProcessingRequest) -> Result<ProcessSummary> {
    if request.files.is_empty() {
        tracing::warn!("Batch request contains no files to process");
        return Ok(ProcessSummary::default());
    }

    std::fs::create_dir_all(&request.output_dir)?;

    let config = RoiDetectionConfig::default();
    let total_input = request.files.len();

    // Early pre-flight availability check for MP4 video encoder backend if video export is requested
    if let Some(ref video_cfg) = request.video_config() {
        let backend = crate::video::probe_video_encoder_backend(video_cfg).map_err(|e| {
            Error::Unknown(format!(
                "MP4 video export requested, but no available HEVC video encoder backend was found on this host: {e}"
            ))
        })?;
        tracing::info!(backend = %backend, "Validated hardware HEVC MP4 video encoder backend before batch processing");
    }

    tracing::info!(
        batch_size = total_input,
        expected_frames = config.expected_frames,
        output_dir = ?request.output_dir,
        "Starting parallel batch RoI processing"
    );

    let items = request.create_item_contexts();
    let total = items.len();

    let concurrency_limit = rayon::current_num_threads().clamp(2, 8);
    let (tx_vision, rx_vision) = crossbeam_channel::bounded(concurrency_limit);
    let (tx_aligned, rx_aligned) = crossbeam_channel::bounded(concurrency_limit);
    let (tx_encoded, rx_encoded) = crossbeam_channel::bounded(concurrency_limit);

    spawn_ingestion_stage(items, config, request.progress_observer.clone(), tx_vision);
    spawn_vision_workers(&rx_vision, &tx_aligned, concurrency_limit);
    drop(tx_aligned);
    spawn_encoding_workers(&rx_aligned, &tx_encoded, concurrency_limit);
    drop(tx_encoded);

    let (successful_count, failed_count) =
        drain_and_write_outputs(&rx_encoded, total, request.progress_observer.as_ref());

    let summary = ProcessSummary {
        total_input,
        successful_count,
        failed_count,
    };

    tracing::info!(
        total = summary.total_input,
        succeeded = summary.successful_count,
        failed = summary.failed_count,
        "Batch processing finished"
    );

    Ok(summary)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::redundant_clone)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};

    #[test]
    fn test_is_supported_image() {
        assert!(is_supported_image(Path::new("test.jpg")));
        assert!(is_supported_image(Path::new("test.PNG")));
        assert!(is_supported_image(Path::new("test.webp")));
        assert!(!is_supported_image(Path::new("test.txt")));
        assert!(!is_supported_image(Path::new("test")));
    }

    #[test]
    fn test_run_batch_on_generated_image() {
        let temp_dir = std::env::temp_dir().join("reto_core_test_batch");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        let input_path = temp_dir.join("sample_strip.png");
        let output_path = temp_dir.join("output");

        // Create a 300x100 synthetic strip
        let img = RgbaImage::from_pixel(300, 100, Rgba([200, 200, 200, 255]));
        img.save(&input_path).unwrap();

        // Non-debug run: verify wiggle GIF is saved, but roi_overlay is NOT saved
        let request_non_debug =
            BatchProcessingRequest::new(vec![input_path.clone()], output_path.clone(), false);
        let summary = run_batch(&request_non_debug).expect("Non-debug batch should run");
        assert_eq!(summary.total_input, 1);
        assert_eq!(summary.successful_count, 1);
        assert!(!output_path.join("sample_strip_roi_overlay.png").exists());
        assert!(output_path.join("sample_strip_wiggle.gif").exists());

        // Debug run: verify roi_overlay is saved when debug is enabled
        let request_debug =
            BatchProcessingRequest::new(vec![input_path.clone()], output_path.clone(), true);
        let summary_debug = run_batch(&request_debug).expect("Debug batch should run");
        assert_eq!(summary_debug.successful_count, 1);
        assert!(output_path.join("sample_strip_roi_overlay.png").exists());

        // Video run: verify wiggle MP4 is saved and GIF is NOT saved when video config is enabled
        let video_output_path = temp_dir.join("video_output");
        let video_config = crate::video::WiggleVideoConfig::new().with_mock_fallback(true);
        let request_video = BatchProcessingRequest::new(vec![input_path], video_output_path.clone(), false)
            .with_video_config(Some(video_config));
        let summary_video = run_batch(&request_video).expect("Video batch should run");
        assert_eq!(summary_video.successful_count, 1);
        assert!(video_output_path.join("sample_strip_wiggle.mp4").exists());
        assert!(!video_output_path.join("sample_strip_wiggle.gif").exists());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[derive(Debug, Default)]
    struct MockProgressObserver {
        started: std::sync::Mutex<Vec<(String, usize, usize)>>,
        completed: std::sync::Mutex<Vec<(String, usize, usize, bool)>>,
    }

    impl ProgressObserver for MockProgressObserver {
        fn on_progress(&self, event: ProgressEvent<'_>) {
            match event {
                ProgressEvent::ItemStarted {
                    file_stem,
                    index,
                    total,
                } => {
                    self.started
                        .lock()
                        .unwrap()
                        .push((file_stem.to_string(), index, total));
                }
                ProgressEvent::ItemCompleted {
                    file_stem,
                    index,
                    total,
                    success,
                } => {
                    self.completed.lock().unwrap().push((
                        file_stem.to_string(),
                        index,
                        total,
                        success,
                    ));
                }
            }
        }
    }

    #[test]
    fn test_run_batch_with_progress_observer() {
        let temp_dir = std::env::temp_dir().join("reto_core_test_progress_observer");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        let input_path = temp_dir.join("obs_strip.png");
        let output_path = temp_dir.join("output");

        let img = RgbaImage::from_pixel(300, 100, Rgba([200, 200, 200, 255]));
        img.save(&input_path).unwrap();

        let observer = std::sync::Arc::new(MockProgressObserver::default());
        let request = BatchProcessingRequest::new(vec![input_path], output_path, false)
            .with_progress_observer(observer.clone());

        let summary = run_batch(&request).expect("Batch with observer should succeed");
        assert_eq!(summary.total_input, 1);
        assert_eq!(summary.successful_count, 1);

        let started = observer.started.lock().unwrap().clone();
        assert_eq!(started.len(), 1);
        assert_eq!(started[0], ("obs_strip".to_string(), 1, 1));

        let completed = observer.completed.lock().unwrap().clone();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0], ("obs_strip".to_string(), 1, 1, true));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
