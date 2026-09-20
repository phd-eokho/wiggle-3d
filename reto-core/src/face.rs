//! Lightweight face detection, landmark extraction, and dominant portrait focal plane scoring.
//!
//! Provides the [`FaceDetector`] trait, [`FaceDetection`] representation, and a high-performance
//! `RetinaFace-MobileNet0.25` ONNX inference engine for subject-aware parallax wiggle anchoring.

use crate::error::{FaceError, Result};
use crate::feature::{ensure_model_cached, init_ort_environment_if_needed};
use crate::geom::{NormalizedRect, Point2D, Size2D};
use image::RgbImage;
use ndarray::{Array4, ArrayViewD};
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};

/// Default image canvas width for RetinaFace input tensor.
pub const RETINAFACE_CANVAS_WIDTH: u32 = 640;

/// Default image canvas height for RetinaFace input tensor.
pub const RETINAFACE_CANVAS_HEIGHT: u32 = 640;

/// Default remote URL for the `RetinaFace-MobileNet0.25` ONNX pretrained model weights.
pub const DEFAULT_RETINAFACE_MODEL_URL: &str =
    "https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv1_0.25.onnx";

/// Fallback mirror URL for the `RetinaFace-MobileNet0.25` ONNX pretrained model weights.
pub const FALLBACK_RETINAFACE_MODEL_URL: &str =
    "https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv1_0.25.onnx";

/// Default SHA256 hexadecimal digest for the `RetinaFace-MobileNet0.25` ONNX weights.
pub const DEFAULT_RETINAFACE_MODEL_SHA256: &str =
    "b7a7acab55e104dce6f32cdfff929bd83946da5cd869b9e2e9bdffafd1b7e4a5";

/// Default confidence threshold for retaining candidate face detections.
pub const DEFAULT_FACE_CONFIDENCE_THRESHOLD: f32 = 0.80;

/// Default Intersection-over-Union (IoU) threshold for Non-Maximum Suppression (NMS).
pub const DEFAULT_FACE_NMS_THRESHOLD: f32 = 0.40;

/// Default standard deviation for Gaussian center-proximity salience weighting.
pub const DEFAULT_CENTER_SALIENT_SIGMA: f32 = 0.50;

/// 5-point canonical facial landmarks: `[Left Eye, Right Eye, Nose Tip, Left Mouth Corner, Right Mouth Corner]`.
pub type FacialLandmarks = [Point2D<f32>; 5];

/// Detected human face with bounding geometry, confidence score, and 5-point facial landmarks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaceDetection {
    /// Normalized sub-frame bounding box `[0.0, 1.0]^2`.
    pub bbox: NormalizedRect,
    /// Model detection confidence score `[0.0, 1.0]`.
    pub score: f32,
    /// 5 canonical facial landmark points in normalized sub-frame coordinates.
    pub landmarks: FacialLandmarks,
}

impl FaceDetection {
    /// Constructs a new `FaceDetection`.
    #[inline]
    #[must_use]
    pub const fn new(bbox: NormalizedRect, score: f32, landmarks: FacialLandmarks) -> Self {
        Self {
            bbox,
            score,
            landmarks,
        }
    }

    /// Calculates the subject salience score with default Gaussian center-bias ($\sigma = 0.50$).
    ///
    /// Evaluates:
    /// $$S(j) = c_j \cdot \sqrt{a_j} \cdot \exp\left(-\frac{d_{\text{center}}(j)^2}{2 \sigma_{\text{center}}^2}\right)$$
    ///
    /// # Examples
    /// ```
    /// use reto_core::{FaceDetection, NormalizedRect, Point2D};
    ///
    /// let bbox = NormalizedRect::new(0.4, 0.4, 0.2, 0.2).unwrap();
    /// let landmarks = [Point2D::new(0.5, 0.5); 5];
    /// let face = FaceDetection::new(bbox, 0.95, landmarks);
    /// assert!(face.salience_score() > 0.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn salience_score(&self) -> f32 {
        self.salience_score_with_sigma(DEFAULT_CENTER_SALIENT_SIGMA)
    }

    /// Calculates the subject salience score with custom Gaussian center-bias $\sigma$.
    #[must_use]
    pub fn salience_score_with_sigma(&self, sigma: f32) -> f32 {
        let center_x = self.bbox.width.mul_add(0.5, self.bbox.x);
        let center_y = self.bbox.height.mul_add(0.5, self.bbox.y);

        // Normalized distance from center (0.5, 0.5) mapped to [-1, 1] range
        let dx = (center_x - 0.5) / 0.5;
        let dy = (center_y - 0.5) / 0.5;
        let d_center_sq = dx.mul_add(dx, dy * dy);

        let area = (self.bbox.width * self.bbox.height).max(0.0);
        let center_penalty = (-d_center_sq / (2.0 * sigma * sigma)).exp();

        self.score * area.sqrt() * center_penalty
    }
}

/// Abstract contract for face detection engines.
pub trait FaceDetector: Send + Sync {
    /// Detects candidate faces in the provided RGB sub-frame image.
    ///
    /// # Errors
    /// Returns [`Error::Face`] if preprocessing, inference, or tensor decoding fails.
    fn detect_faces(&self, image: &RgbImage) -> Result<Vec<FaceDetection>>;
}

/// Diagnostic observation tap for intermediate face detection artifacts.
pub trait FaceDiagnosticTap: Send + Sync {
    /// Emits intermediate visualization of detected faces, landmarks, and salience scores.
    fn on_faces_detected(
        &self,
        frame_idx: usize,
        image: &RgbImage,
        detections: &[FaceDetection],
        dominant_idx: Option<usize>,
    );
}

/// No-op diagnostic tap that ignores all face observation events.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoOpFaceDiagnosticTap;

impl FaceDiagnosticTap for NoOpFaceDiagnosticTap {
    #[inline]
    fn on_faces_detected(
        &self,
        _frame_idx: usize,
        _image: &RgbImage,
        _detections: &[FaceDetection],
        _dominant_idx: Option<usize>,
    ) {
    }
}

/// Selects the dominant portrait subject index from a collection of candidate face detections.
///
/// Uses center-biased Gaussian salience scoring to prioritize large, centered faces over
/// peripheral passersby or small background faces.
///
/// # Returns
/// Returns `Some(index)` of the highest-scoring dominant face, or `None` if `faces` is empty.
///
/// # Examples
/// ```
/// use reto_core::{select_dominant_face, FaceDetection, NormalizedRect, Point2D};
///
/// let lm = [Point2D::new(0.5, 0.5); 5];
/// let background_face = FaceDetection::new(
///     NormalizedRect::new(0.05, 0.05, 0.08, 0.08).unwrap(),
///     0.90,
///     lm,
/// );
/// let dominant_face = FaceDetection::new(
///     NormalizedRect::new(0.35, 0.35, 0.30, 0.30).unwrap(),
///     0.95,
///     lm,
/// );
///
/// let faces = vec![background_face, dominant_face];
/// assert_eq!(select_dominant_face(&faces), Some(1));
/// ```
#[must_use]
pub fn select_dominant_face(faces: &[FaceDetection]) -> Option<usize> {
    if faces.is_empty() {
        return None;
    }
    faces
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.salience_score()
                .partial_cmp(&b.salience_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
}

/// Configuration parameters for the `RetinaFace` detection engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetinaFaceConfig {
    /// Minimum confidence threshold for candidate detections.
    pub confidence_threshold: f32,
    /// Intersection-over-Union (IoU) threshold for Non-Maximum Suppression (NMS).
    pub nms_threshold: f32,
    /// Optional local path to the ONNX model binary.
    pub model_path: Option<PathBuf>,
    /// Remote URL to download the model from if missing.
    pub model_url: String,
    /// Expected SHA-256 digest of the ONNX model weights.
    pub model_sha256: Option<String>,
    /// Model input dimensions (width, height).
    pub input_size: Size2D<u32>,
    /// Box and landmark decoding variances (standard default: `[0.1, 0.2]`).
    pub variances: [f32; 2],
}

impl Default for RetinaFaceConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: DEFAULT_FACE_CONFIDENCE_THRESHOLD,
            nms_threshold: DEFAULT_FACE_NMS_THRESHOLD,
            model_path: None,
            model_url: DEFAULT_RETINAFACE_MODEL_URL.to_string(),
            model_sha256: Some(DEFAULT_RETINAFACE_MODEL_SHA256.to_string()),
            input_size: Size2D::new(RETINAFACE_CANVAS_WIDTH, RETINAFACE_CANVAS_HEIGHT),
            variances: [0.1, 0.2],
        }
    }
}

/// Returns the default local cache path for the `RetinaFace` ONNX model.
#[must_use]
pub fn get_default_retinaface_cache_path() -> PathBuf {
    std::env::var("RETO3D_CACHE_DIR").map_or_else(
        |_| {
            PathBuf::from(".cache_reto3d")
                .join("models")
                .join("retinaface_mobilenet0.25.onnx")
        },
        |dir| {
            PathBuf::from(dir)
                .join("models")
                .join("retinaface_mobilenet0.25.onnx")
        },
    )
}

/// Generates normalized prior / anchor boxes `[cx, cy, s_x, s_y]` for RetinaFace at input resolution.
///
/// Feature strides: `[8, 16, 32]`
/// Min sizes: `[[16.0, 32.0], [64.0, 128.0], [256.0, 512.0]]`
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
pub fn generate_retinaface_anchors(width: u32, height: u32) -> Vec<[f32; 4]> {
    let strides = [8u32, 16, 32];
    let min_sizes = [[16.0f32, 32.0], [64.0, 128.0], [256.0, 512.0]];
    let width_f = width as f32;
    let height_f = height as f32;

    let mut anchors = Vec::with_capacity(16800);

    for (stride, sizes) in strides.into_iter().zip(min_sizes) {
        let feature_h = height.div_ceil(stride);
        let feature_w = width.div_ceil(stride);
        let stride_f = stride as f32;

        for y in 0..feature_h {
            let cy = ((y as f32) + 0.5) * stride_f / height_f;
            for x in 0..feature_w {
                let cx = ((x as f32) + 0.5) * stride_f / width_f;
                for &min_sz in &sizes {
                    let sx = min_sz / width_f;
                    let sy = min_sz / height_f;
                    anchors.push([cx, cy, sx, sy]);
                }
            }
        }
    }

    anchors
}

/// Shared `RetinaFace` ONNX runtime instance.
pub(crate) struct SharedRetinaFaceModel {
    session: Mutex<Session>,
    anchors: Vec<[f32; 4]>,
}

type SharedRetinaFaceHandle = Arc<SharedRetinaFaceModel>;
type FaceResult<T> = std::result::Result<T, FaceError>;
type SingleFlightBarrier = Arc<(Mutex<Option<FaceResult<SharedRetinaFaceHandle>>>, Condvar)>;

#[derive(Clone)]
enum CacheSlot {
    Ready(SharedRetinaFaceHandle),
    Pending(SingleFlightBarrier),
}

static GLOBAL_RETINAFACE_CACHE: Mutex<Option<HashMap<PathBuf, CacheSlot>>> = Mutex::new(None);

/// Clears the global `RetinaFace` model cache from memory.
pub fn clear_retinaface_model_cache() {
    if let Ok(mut guard) = GLOBAL_RETINAFACE_CACHE.lock() {
        *guard = None;
    }
}

/// Retrieves or compiles the shared `RetinaFace` ONNX model session.
#[allow(clippy::significant_drop_tightening)]
pub(crate) fn get_or_load_retinaface_model(
    config: &RetinaFaceConfig,
) -> Result<SharedRetinaFaceHandle> {
    init_ort_environment_if_needed();

    let model_path = config
        .model_path
        .clone()
        .unwrap_or_else(get_default_retinaface_cache_path);

    let (barrier, is_initiator) = {
        let mut guard = GLOBAL_RETINAFACE_CACHE.lock().map_err(|e| {
            FaceError::Inference(format!("Global RetinaFace cache lock poisoned: {e}"))
        })?;
        let cache = guard.get_or_insert_with(HashMap::new);

        match cache.get(&model_path) {
            Some(CacheSlot::Ready(model)) => return Ok(Arc::clone(model)),
            Some(CacheSlot::Pending(barrier)) => (Arc::clone(barrier), false),
            None => {
                let barrier = Arc::new((Mutex::new(None), Condvar::new()));
                cache.insert(model_path.clone(), CacheSlot::Pending(Arc::clone(&barrier)));
                (barrier, true)
            }
        }
    };

    if !is_initiator {
        let (lock, cvar) = &*barrier;
        let mut guard = lock
            .lock()
            .map_err(|e| FaceError::Inference(format!("RetinaFace barrier lock poisoned: {e}")))?;
        while guard.is_none() {
            guard = cvar.wait(guard).map_err(|e| {
                FaceError::Inference(format!("RetinaFace barrier condvar wait failed: {e}"))
            })?;
        }
        return match guard.as_ref() {
            Some(Ok(handle)) => Ok(Arc::clone(handle)),
            Some(Err(err)) => Err(err.clone().into()),
            None => Err(FaceError::ModelLoad("Single-flight failed".into()).into()),
        };
    }

    let compile_result: FaceResult<SharedRetinaFaceHandle> =
        (|| -> FaceResult<SharedRetinaFaceHandle> {
            ensure_model_cached(
                &model_path,
                &config.model_url,
                config.model_sha256.as_deref(),
            )
            .map_err(|e| FaceError::ModelLoad(format!("Failed to ensure RetinaFace model: {e}")))?;

            let session = Session::builder()
                .map_err(|e| {
                    FaceError::ModelLoad(format!("Failed to create RetinaFace SessionBuilder: {e}"))
                })?
                .commit_from_file(&model_path)
                .map_err(|e| {
                    FaceError::ModelLoad(format!(
                        "Failed to load RetinaFace ONNX model at {}: {e}",
                        model_path.display()
                    ))
                })?;

            let anchors =
                generate_retinaface_anchors(config.input_size.width, config.input_size.height);

            Ok(Arc::new(SharedRetinaFaceModel {
                session: Mutex::new(session),
                anchors,
            }))
        })();

    {
        let mut guard = GLOBAL_RETINAFACE_CACHE.lock().map_err(|e| {
            FaceError::Inference(format!("Global RetinaFace cache lock poisoned: {e}"))
        })?;
        if let Some(cache) = guard.as_mut() {
            if let Ok(ref model) = compile_result {
                cache.insert(model_path, CacheSlot::Ready(Arc::clone(model)));
            } else {
                cache.remove(&model_path);
            }
        }
    }

    let (lock, cvar) = &*barrier;
    if let Ok(mut result_guard) = lock.lock() {
        *result_guard = Some(compile_result.clone());
        cvar.notify_all();
    }

    compile_result.map_err(Into::into)
}

/// Pure-Rust CPU / ONNX face detector backed by `RetinaFace-MobileNet0.25`.
#[derive(Clone)]
pub struct RetinaFaceDetector {
    config: RetinaFaceConfig,
    model: SharedRetinaFaceHandle,
}

impl RetinaFaceDetector {
    /// Constructs a new `RetinaFaceDetector` with specified configuration.
    ///
    /// # Errors
    /// Returns [`Error::Face`] if model loading or compilation fails.
    pub fn new(config: RetinaFaceConfig) -> Result<Self> {
        let model = get_or_load_retinaface_model(&config)?;
        Ok(Self { config, model })
    }

    /// Constructs a new `RetinaFaceDetector` with default parameters.
    ///
    /// # Errors
    /// Returns [`Error::Face`] if default model loading fails.
    pub fn default_engine() -> Result<Self> {
        Self::new(RetinaFaceConfig::default())
    }

    /// Preprocesses the source image into normalized letterboxed BGR tensor of size `[1, 3, H, W]`.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::suboptimal_flops
    )]
    fn preprocess_image(
        image: &RgbImage,
        target_w: u32,
        target_h: u32,
    ) -> (Array4<f32>, (f32, f32, f32, f32, f32)) {
        let (src_w, src_h) = image.dimensions();
        let scale = ((target_w as f32) / (src_w as f32)).min((target_h as f32) / (src_h as f32));
        let resized_w = ((src_w as f32) * scale).round() as u32;
        let resized_h = ((src_h as f32) * scale).round() as u32;

        let resized = if resized_w == src_w && resized_h == src_h {
            image.clone()
        } else {
            image::imageops::resize(
                image,
                resized_w.max(1),
                resized_h.max(1),
                image::imageops::FilterType::Triangle,
            )
        };

        let pad_x = ((target_w - resized_w) / 2) as usize;
        let pad_y = ((target_h - resized_h) / 2) as usize;

        // BGR mean subtraction constants for RetinaFace
        let mean_b = 104.0f32;
        let mean_g = 117.0f32;
        let mean_r = 123.0f32;

        let mut tensor = Array4::<f32>::zeros((1, 3, target_h as usize, target_w as usize));

        for y in 0..resized_h as usize {
            for x in 0..resized_w as usize {
                let pixel = resized.get_pixel(x as u32, y as u32);
                let r = f32::from(pixel[0]);
                let g = f32::from(pixel[1]);
                let b = f32::from(pixel[2]);

                let dst_y = y + pad_y;
                let dst_x = x + pad_x;

                // Channel 0: B, Channel 1: G, Channel 2: R
                tensor[[0, 0, dst_y, dst_x]] = b - mean_b;
                tensor[[0, 1, dst_y, dst_x]] = g - mean_g;
                tensor[[0, 2, dst_y, dst_x]] = r - mean_r;
            }
        }

        (
            tensor,
            (
                scale,
                pad_x as f32 / target_w as f32,
                pad_y as f32 / target_h as f32,
                resized_w as f32 / target_w as f32,
                resized_h as f32 / target_h as f32,
            ),
        )
    }

    /// Computes Intersection over Union (IoU) between two bounding boxes.
    fn calculate_iou(a: NormalizedRect, b: NormalizedRect) -> f32 {
        let left = a.x.max(b.x);
        let top = a.y.max(b.y);
        let right = (a.x + a.width).min(b.x + b.width);
        let bottom = (a.y + a.height).min(b.y + b.height);

        let inter_w = (right - left).max(0.0);
        let inter_h = (bottom - top).max(0.0);
        let inter_area = inter_w * inter_h;

        let area_a = a.width * a.height;
        let area_b = b.width * b.height;
        let union_area = area_a + area_b - inter_area;

        if union_area <= 1e-6 {
            0.0
        } else {
            inter_area / union_area
        }
    }

    /// Applies greedy Non-Maximum Suppression over candidate detections.
    #[must_use]
    pub fn apply_nms(mut candidates: Vec<FaceDetection>, iou_thresh: f32) -> Vec<FaceDetection> {
        candidates.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut kept: Vec<FaceDetection> = Vec::with_capacity(candidates.len());
        let mut suppressed = vec![false; candidates.len()];

        for i in 0..candidates.len() {
            if suppressed[i] {
                continue;
            }
            let current_bbox = candidates[i].bbox;
            kept.push(candidates[i].clone());

            for j in (i + 1)..candidates.len() {
                if !suppressed[j]
                    && Self::calculate_iou(current_bbox, candidates[j].bbox) > iou_thresh
                {
                    suppressed[j] = true;
                }
            }
        }

        kept
    }

    /// Detects faces across a set of sub-frame images extracted from frame ROIs.
    ///
    /// # Arguments
    /// * `crops` - Slice of extracted sub-frame images.
    ///
    /// # Errors
    /// Returns [`Error::Face`] if preprocessing, inference, or tensor decoding fails.
    pub fn detect_faces_for_rois(
        &self,
        crops: &[image::DynamicImage],
    ) -> Result<Vec<crate::visualizer::FrameFaceRecord>> {
        let mut records = Vec::with_capacity(crops.len());
        for (idx, crop) in crops.iter().enumerate() {
            let rgb = crop.to_rgb8();
            let faces = self.detect_faces(&rgb)?;
            let dominant_idx = select_dominant_face(&faces);
            records.push((idx, faces, dominant_idx));
        }
        Ok(records)
    }
}

struct AnchorDecodeContext {
    pad_x_norm: f32,
    pad_y_norm: f32,
    content_scale_x: f32,
    content_scale_y: f32,
    var_0: f32,
    var_1: f32,
}

#[inline]
fn compute_face_score(c0: f32, c1: f32) -> f32 {
    if (c0 + c1 - 1.0).abs() < 1e-3 && c0 >= 0.0 && c1 >= 0.0 {
        c1
    } else {
        let max_c = c0.max(c1);
        let exp0 = (c0 - max_c).exp();
        let exp1 = (c1 - max_c).exp();
        let sum_exp = exp0 + exp1;
        if sum_exp > 0.0 {
            exp1 / sum_exp
        } else {
            0.0
        }
    }
}

#[allow(clippy::suboptimal_flops, clippy::similar_names)]
fn decode_candidate_at_anchor(
    i: usize,
    anchor: [f32; 4],
    loc: &ArrayViewD<'_, f32>,
    landmarks: &ArrayViewD<'_, f32>,
    score: f32,
    ctx: &AnchorDecodeContext,
) -> Option<FaceDetection> {
    let anchor_cx = anchor[0];
    let anchor_cy = anchor[1];
    let anchor_sx = anchor[2];
    let anchor_sy = anchor[3];

    let dx = loc[[0, i, 0]];
    let dy = loc[[0, i, 1]];
    let dw = loc[[0, i, 2]];
    let dh = loc[[0, i, 3]];

    let box_cx = anchor_cx + dx * ctx.var_0 * anchor_sx;
    let box_cy = anchor_cy + dy * ctx.var_0 * anchor_sy;
    let box_w = anchor_sx * (dw * ctx.var_1).exp();
    let box_h = anchor_sy * (dh * ctx.var_1).exp();

    let box_x = box_cx - (box_w * 0.5);
    let box_y = box_cy - (box_h * 0.5);

    let norm_x = ((box_x - ctx.pad_x_norm) * ctx.content_scale_x).clamp(0.0, 1.0);
    let norm_y = ((box_y - ctx.pad_y_norm) * ctx.content_scale_y).clamp(0.0, 1.0);
    let norm_w = (box_w * ctx.content_scale_x).min(1.0 - norm_x);
    let norm_h = (box_h * ctx.content_scale_y).min(1.0 - norm_y);

    if norm_w <= 1e-4 || norm_h <= 1e-4 {
        return None;
    }

    let bbox = NormalizedRect::new(norm_x, norm_y, norm_w, norm_h).ok()?;

    let mut face_landmarks = [Point2D::new(0.0, 0.0); 5];
    for k in 0..5 {
        let lx = landmarks[[0, i, k * 2]];
        let ly = landmarks[[0, i, (k * 2) + 1]];

        let lm_x = anchor_cx + lx * ctx.var_0 * anchor_sx;
        let lm_y = anchor_cy + ly * ctx.var_0 * anchor_sy;

        let unpad_lx = ((lm_x - ctx.pad_x_norm) * ctx.content_scale_x).clamp(0.0, 1.0);
        let unpad_ly = ((lm_y - ctx.pad_y_norm) * ctx.content_scale_y).clamp(0.0, 1.0);
        face_landmarks[k] = Point2D::new(unpad_lx, unpad_ly);
    }

    Some(FaceDetection::new(bbox, score, face_landmarks))
}

impl FaceDetector for RetinaFaceDetector {
    #[allow(
        clippy::significant_drop_tightening,
        clippy::similar_names,
        clippy::cast_precision_loss,
        clippy::suboptimal_flops
    )]
    fn detect_faces(&self, image: &RgbImage) -> Result<Vec<FaceDetection>> {
        let (input_tensor, (_scale, pad_x_norm, pad_y_norm, content_w_norm, content_h_norm)) =
            Self::preprocess_image(
                image,
                self.config.input_size.width,
                self.config.input_size.height,
            );

        let mut session_guard =
            self.model.session.lock().map_err(|e| {
                FaceError::Inference(format!("RetinaFace session mutex poisoned: {e}"))
            })?;

        let tensor_val = Tensor::from_array(input_tensor).map_err(|e| {
            FaceError::TensorLayout(format!("Failed to build RetinaFace input tensor: {e}"))
        })?;

        let input_name = session_guard
            .inputs()
            .first()
            .map_or("input0", |i| i.name())
            .to_string();

        let outputs = session_guard
            .run(ort::inputs![input_name.as_str() => tensor_val])
            .map_err(|e| FaceError::Inference(format!("RetinaFace inference failed: {e}")))?;

        let (loc_arr, conf_arr, land_arr) = if outputs.len() >= 3 {
            (
                outputs[0].try_extract_array::<f32>(),
                outputs[1].try_extract_array::<f32>(),
                outputs[2].try_extract_array::<f32>(),
            )
        } else {
            return Err(FaceError::Inference(
                "RetinaFace model outputs missing required tensors".to_string(),
            )
            .into());
        };

        let (Ok(loc), Ok(conf), Ok(landmarks)) = (loc_arr, conf_arr, land_arr) else {
            return Err(FaceError::Inference(
                "Failed to extract float arrays from RetinaFace outputs".to_string(),
            )
            .into());
        };

        let num_anchors = self.model.anchors.len().min(loc.shape()[1]);
        let mut candidates = Vec::new();
        let var_0 = self.config.variances[0];
        let var_1 = self.config.variances[1];

        let decode_ctx = AnchorDecodeContext {
            pad_x_norm,
            pad_y_norm,
            content_scale_x: if content_w_norm > 0.0 {
                1.0 / content_w_norm
            } else {
                1.0
            },
            content_scale_y: if content_h_norm > 0.0 {
                1.0 / content_h_norm
            } else {
                1.0
            },
            var_0,
            var_1,
        };

        for i in 0..num_anchors {
            let c0 = conf[[0, i, 0]];
            let c1 = conf[[0, i, 1]];

            let score = compute_face_score(c0, c1);
            if score < self.config.confidence_threshold {
                continue;
            }

            if let Some(candidate) = decode_candidate_at_anchor(
                i,
                self.model.anchors[i],
                &loc,
                &landmarks,
                score,
                &decode_ctx,
            ) {
                candidates.push(candidate);
            }
        }

        Ok(Self::apply_nms(candidates, self.config.nms_threshold))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_anchor_generation_counts() {
        let anchors = generate_retinaface_anchors(640, 640);
        // Stride 8: 80x80x2 = 12800
        // Stride 16: 40x40x2 = 3200
        // Stride 32: 20x20x2 = 800
        // Total = 16800
        assert_eq!(anchors.len(), 16800);

        for anchor in &anchors {
            assert!(anchor[0] >= 0.0 && anchor[0] <= 1.0);
            assert!(anchor[1] >= 0.0 && anchor[1] <= 1.0);
            assert!(anchor[2] > 0.0 && anchor[3] > 0.0);
        }
    }

    #[test]
    fn test_salience_scoring_prioritizes_centered_face() {
        let lm = [Point2D::new(0.5, 0.5); 5];

        let centered = FaceDetection::new(
            NormalizedRect::new(0.35, 0.35, 0.30, 0.30).unwrap(),
            0.90,
            lm,
        );

        let corner = FaceDetection::new(
            NormalizedRect::new(0.02, 0.02, 0.30, 0.30).unwrap(),
            0.90,
            lm,
        );

        assert!(centered.salience_score() > corner.salience_score());
    }

    #[test]
    fn test_dominant_face_selector() {
        let lm = [Point2D::new(0.5, 0.5); 5];
        let passerby = FaceDetection::new(
            NormalizedRect::new(0.01, 0.10, 0.05, 0.05).unwrap(),
            0.95,
            lm,
        );
        let subject = FaceDetection::new(
            NormalizedRect::new(0.30, 0.20, 0.40, 0.50).unwrap(),
            0.88,
            lm,
        );

        let faces = vec![passerby, subject];
        assert_eq!(select_dominant_face(&faces), Some(1));
    }

    #[test]
    fn test_nms_suppression() {
        let lm = [Point2D::new(0.5, 0.5); 5];
        let box1 = FaceDetection::new(
            NormalizedRect::new(0.20, 0.20, 0.30, 0.30).unwrap(),
            0.95,
            lm,
        );
        // Overlapping box with lower score
        let box2 = FaceDetection::new(
            NormalizedRect::new(0.21, 0.21, 0.29, 0.29).unwrap(),
            0.80,
            lm,
        );
        // Disjoint box
        let box3 = FaceDetection::new(
            NormalizedRect::new(0.70, 0.70, 0.20, 0.20).unwrap(),
            0.90,
            lm,
        );

        let suppressed = RetinaFaceDetector::apply_nms(vec![box1, box2, box3], 0.40);
        assert_eq!(suppressed.len(), 2);
        assert_eq!(suppressed[0].score, 0.95);
        assert_eq!(suppressed[1].score, 0.90);
    }
}
