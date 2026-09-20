//! Feature extraction, keypoint representation, and backend abstraction for multi-frame parallax alignment.
//!
//! Provides the [`FeatureExtractor`] trait, `SuperPoint` detector/descriptor extraction,
//! and integration with [`crate::luma::ScaledGrayscaleStrip`] for high-performance sub-view alignment.

use crate::error::AlignmentError;
use crate::geom::{
    FrameRoi, NormalizedRect, OrientationDelegator, Point2D, Size2D, StripOrientation,
};
use crate::luma::ScaledLumaImage;
use image::{GenericImageView, Pixel};
use ndarray::{Array4, ArrayViewD, Axis};
use num_traits::ToPrimitive;
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};

/// Strong handle to a shared `SuperPoint` ONNX runtime instance.
pub(crate) type SharedModelHandle = Arc<SharedSuperPointModel>;

/// Single-flight synchronization barrier for concurrent model compilation.
type SingleFlightBarrier = Arc<(Mutex<Option<AlignmentResult<SharedModelHandle>>>, Condvar)>;

/// Cache slot holding either a ready shared model reference or an in-flight compilation barrier.
#[derive(Clone)]
enum CacheSlot {
    Ready(SharedModelHandle),
    Pending(SingleFlightBarrier),
}

/// Thread-safe reference registry mapping model file paths to cached neural networks.
type ModelCacheMap = HashMap<PathBuf, CacheSlot>;

/// Global singleton mutex guarding the `SuperPoint` runtime cache.
type GlobalModelCache = Mutex<Option<ModelCacheMap>>;

/// Thread-safe local cache slot for the shared model reference.
type ModelSlot = Arc<Mutex<Option<SharedModelHandle>>>;

/// Dynamic multidimensional float tensor view.
type TensorView<'a> = ArrayViewD<'a, f32>;

/// 2D keypoint score candidate pair (sub-pixel coordinate and confidence score).
pub type KeypointScore = (Point2D<f32>, f32);

/// Pair of (Index A, Index B) frame identifiers for pairwise alignment.
pub type FramePair = (usize, usize);

/// Matrix representation of 3x3 3D rotation tensor.
pub type RotationMatrix3x3 = [[f32; 3]; 3];

/// 3D translation vector.
pub type TranslationVector3 = [f32; 3];

/// Alignment specialized result type.
pub type AlignmentResult<T> = Result<T, AlignmentError>;

/// Shared `SuperPoint` ONNX runnable session.
pub(crate) struct SharedSuperPointModel {
    session: Mutex<Session>,
}

impl SharedSuperPointModel {
    /// Executes model inference on the canvas input tensor and decodes keypoints.
    #[allow(clippy::significant_drop_tightening)]
    pub(crate) fn infer_and_decode(
        &self,
        input: Array4<f32>,
        ctx: &SuperPointDecodeContext<'_>,
    ) -> AlignmentResult<Vec<KeyPoint>> {
        let mut session_guard = self.session.lock().map_err(|e| {
            AlignmentError::Inference(format!("SuperPoint session mutex poisoned: {e}"))
        })?;
        let tensor = Tensor::from_array(input).map_err(|e| {
            AlignmentError::TensorLayout(format!("Failed to build input tensor: {e}"))
        })?;
        let outputs = session_guard
            .run(ort::inputs!["image" => tensor])
            .map_err(|e| AlignmentError::Inference(format!("Model inference failed: {e}")))?;

        let (kpts_res, scs_res, descs_res) = if outputs.len() >= 3 {
            (
                outputs[0].try_extract_array::<f32>(),
                outputs[1].try_extract_array::<f32>(),
                outputs[2].try_extract_array::<f32>(),
            )
        } else if outputs.contains_key("keypoints")
            && outputs.contains_key("scores")
            && outputs.contains_key("descriptors")
        {
            (
                outputs["keypoints"].try_extract_array::<f32>(),
                outputs["scores"].try_extract_array::<f32>(),
                outputs["descriptors"].try_extract_array::<f32>(),
            )
        } else {
            return Err(AlignmentError::Inference(
                "SuperPoint outputs missing expected tensors".to_string(),
            ));
        };

        match (kpts_res, scs_res, descs_res) {
            (Ok(kpts), Ok(scs), Ok(descs)) => {
                SuperPointDetector::decode_superpoint_vectorized(kpts, scs, descs, ctx)
            }
            _ => Err(AlignmentError::Inference(
                "Failed to extract float array views from model outputs".to_string(),
            )),
        }
    }
}

static GLOBAL_SUPERPOINT_CACHE: GlobalModelCache = Mutex::new(None);

/// Loads or retrieves a shared `SuperPoint` ONNX model reference from the global runtime cache.
///
/// The model is compiled once on first use and retained in memory across batch processing runs.
/// Concurrent callers for the same model path coordinate via a single-flight barrier so only
/// the first thread compiles the model once.
#[allow(clippy::significant_drop_tightening)]
pub(crate) fn get_or_load_superpoint_model(
    config: &SuperPointConfig,
) -> AlignmentResult<SharedModelHandle> {
    let model_path = config
        .model_path
        .clone()
        .unwrap_or_else(get_default_model_cache_path);

    let (barrier, is_initiator) = {
        let mut guard = GLOBAL_SUPERPOINT_CACHE.lock().map_err(|e| {
            AlignmentError::Inference(format!("Global model cache lock poisoned: {e}"))
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
        let mut result_guard = lock
            .lock()
            .map_err(|e| AlignmentError::Inference(format!("Single-flight lock poisoned: {e}")))?;
        while result_guard.is_none() {
            result_guard = cvar.wait(result_guard).map_err(|e| {
                AlignmentError::Inference(format!("Single-flight condvar poisoned: {e}"))
            })?;
        }
        return result_guard.as_ref().cloned().unwrap_or_else(|| {
            Err(AlignmentError::Inference(
                "Missing model result".to_string(),
            ))
        });
    }

    let compile_result = (|| -> AlignmentResult<SharedModelHandle> {
        ensure_model_cached(
            &model_path,
            &config.model_url,
            config.expected_sha256.as_deref(),
        )?;

        init_ort_environment_if_needed();

        tracing::info!(
            backend = "ort",
            engine = "ONNX Runtime (Dynamic Linking)",
            device = ?config.device,
            model_path = %model_path.display(),
            "Loading shared SuperPoint inference model"
        );

        let (session, _device) = create_ort_session(&model_path, config.device)?;

        Ok(Arc::new(SharedSuperPointModel {
            session: Mutex::new(session),
        }))
    })();

    if let Ok(mut guard) = GLOBAL_SUPERPOINT_CACHE.lock() {
        if let Some(ref mut cache) = *guard {
            match &compile_result {
                Ok(shared) => {
                    cache.insert(model_path, CacheSlot::Ready(Arc::clone(shared)));
                }
                Err(_) => {
                    cache.remove(&model_path);
                }
            }
        }
    }

    let (lock, cvar) = &*barrier;
    if let Ok(mut result_guard) = lock.lock() {
        *result_guard = Some(compile_result.clone());
        cvar.notify_all();
    }

    compile_result
}

/// Creates an ONNX Runtime session for the model at `model_path` targeting `device`.
fn create_ort_session(
    model_path: &Path,
    device: BackendDevice,
) -> AlignmentResult<(Session, BackendDevice)> {
    let (session, actual_device) = match device {
        BackendDevice::Auto => {
            let try_cuda = (|| -> ort::Result<Session> {
                Session::builder()?
                    .with_execution_providers([ort::ep::CUDA::default().build()])?
                    .commit_from_file(model_path)
            })();

            match try_cuda {
                Ok(session) => {
                    tracing::info!("Successfully initialized CUDA GPU execution provider");
                    (session, BackendDevice::Cuda)
                }
                Err(e) => {
                    tracing::warn!(
                        gpu_error = %e,
                        "Failed to initialize CUDA execution provider; falling back to CPU"
                    );
                    let session = Session::builder()
                        .map_err(|e| {
                            AlignmentError::ModelLoad(format!(
                                "Failed to create SessionBuilder: {e}"
                            ))
                        })?
                        .commit_from_file(model_path)
                        .map_err(|e| {
                            AlignmentError::ModelLoad(format!(
                                "Failed to load ONNX model on CPU at {}: {e}",
                                model_path.display()
                            ))
                        })?;
                    (session, BackendDevice::Cpu)
                }
            }
        }
        BackendDevice::Cuda => {
            let session = Session::builder()
                .map_err(|e| {
                    AlignmentError::ModelLoad(format!("Failed to create SessionBuilder: {e}"))
                })?
                .with_execution_providers([ort::ep::CUDA::default().build()])
                .map_err(|e| {
                    AlignmentError::ModelLoad(format!(
                        "Failed to configure CUDA execution provider: {e}"
                    ))
                })?
                .commit_from_file(model_path)
                .map_err(|e| {
                    AlignmentError::ModelLoad(format!(
                        "Failed to load ONNX model at {}: {e}",
                        model_path.display()
                    ))
                })?;
            (session, BackendDevice::Cuda)
        }
        BackendDevice::Cpu => {
            let session = Session::builder()
                .map_err(|e| {
                    AlignmentError::ModelLoad(format!("Failed to create SessionBuilder: {e}"))
                })?
                .commit_from_file(model_path)
                .map_err(|e| {
                    AlignmentError::ModelLoad(format!(
                        "Failed to load ONNX model at {}: {e}",
                        model_path.display()
                    ))
                })?;
            (session, BackendDevice::Cpu)
        }
    };

    tracing::info!(
        active_device = ?actual_device,
        "SuperPoint model session initialized"
    );

    Ok((session, actual_device))
}

/// Clears the global `SuperPoint` model cache from memory.
pub fn clear_superpoint_model_cache() {
    if let Ok(mut guard) = GLOBAL_SUPERPOINT_CACHE.lock() {
        *guard = None;
    }
}

/// Default remote URL for the `SuperPoint` ONNX pretrained model weights.
pub const DEFAULT_SUPERPOINT_MODEL_URL: &str =
    "https://github.com/fettahyildizz/superpoint_lightglue_tensorrt/raw/main/weights/superpoint.onnx";

/// Fallback mirror URL for the `SuperPoint` ONNX pretrained model weights.
pub const FALLBACK_SUPERPOINT_MODEL_URL: &str =
    "https://github.com/fettahyildizz/superpoint_lightglue_tensorrt/raw/main/weights/superpoint.onnx";

/// Default SHA256 hexadecimal digest for the `SuperPoint` ONNX pretrained model weights.
pub const DEFAULT_SUPERPOINT_MODEL_SHA256: &str =
    "86708faa8daca9ca51a5673a66d55fe1b79cae9708b07dd8faf440fbab3c8e55";

/// Computes the hexadecimal SHA-256 digest of a file.
///
/// # Arguments
/// * `path` - Path to the file.
///
/// # Errors
/// Returns [`std::io::Error`] if opening or reading the file fails.
pub fn compute_file_sha256(path: &Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    let result = hasher.finalize();
    Ok(format!("{result:x}"))
}

/// Returns the default local cache path for the `SuperPoint` ONNX model.
///
/// Checks the `RETO3D_CACHE_DIR` environment variable, otherwise defaults to `./.cache_reto3d/models/superpoint.onnx`.
#[must_use]
pub fn get_default_model_cache_path() -> PathBuf {
    std::env::var("RETO3D_CACHE_DIR").map_or_else(
        |_| {
            PathBuf::from(".cache_reto3d")
                .join("models")
                .join("superpoint.onnx")
        },
        |dir| PathBuf::from(dir).join("models").join("superpoint.onnx"),
    )
}

/// Discovers the location of the dynamic ONNX Runtime shared library (`libonnxruntime.so`).
///
/// Searches `ORT_DYLIB_PATH`, standard system paths, and local cached release installations.
#[must_use]
pub fn get_default_ort_dylib_path() -> Option<PathBuf> {
    if let Ok(path_str) = std::env::var("ORT_DYLIB_PATH") {
        let p = PathBuf::from(path_str);
        if p.exists() {
            return Some(p);
        }
    }

    let candidates = [
        PathBuf::from(
            ".cache_reto3d/onnxruntime/onnxruntime-linux-x64-gpu-1.19.2/lib/libonnxruntime.so",
        ),
        PathBuf::from(
            ".cache_reto3d/onnxruntime/onnxruntime-linux-x64-1.19.2/lib/libonnxruntime.so",
        ),
        PathBuf::from("/usr/local/lib/libonnxruntime.so"),
        PathBuf::from("/usr/lib/libonnxruntime.so"),
    ];

    candidates.into_iter().find(|candidate| candidate.exists())
}

/// Initializes the ONNX Runtime environment if needed.
pub fn init_ort_environment_if_needed() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = ort::init().commit();
    });
}

/// Ensures the `SuperPoint` model is cached locally, downloading it from `url` if missing
/// or if the existing file hash does not match `expected_sha256`.
///
/// # Arguments
/// * `model_path` - Local file path where the model binary should reside.
/// * `url` - Remote URL to download from if missing.
/// * `expected_sha256` - Optional expected SHA-256 digest to verify against.
///
/// # Errors
/// Returns [`AlignmentError::ModelLoad`] if directory creation, download, hash verification, or file writing fails.
pub fn ensure_model_cached(
    model_path: &Path,
    url: &str,
    expected_sha256: Option<&str>,
) -> AlignmentResult<()> {
    if model_path.exists() {
        if let Some(expected) = expected_sha256 {
            match compute_file_sha256(model_path) {
                Ok(actual) if actual.eq_ignore_ascii_case(expected) => {
                    tracing::debug!(
                        path = %model_path.display(),
                        sha256 = %actual,
                        "Model cache hit with verified hash"
                    );
                    return Ok(());
                }
                Ok(actual) => {
                    tracing::warn!(
                        path = %model_path.display(),
                        expected = %expected,
                        actual = %actual,
                        "Cached model hash mismatch; re-downloading model"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        path = %model_path.display(),
                        error = %e,
                        "Failed to compute hash of cached model; re-downloading model"
                    );
                }
            }
        } else {
            return Ok(());
        }
    }

    if let Some(parent) = model_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            AlignmentError::ModelLoad(format!(
                "Failed to create cache directory {}: {e}",
                parent.display()
            ))
        })?;
    }
    tracing::info!(
        url = %url,
        target = %model_path.display(),
        "Downloading SuperPoint ONNX model..."
    );

    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(4))
        .timeout_read(std::time::Duration::from_secs(10))
        .redirects(5)
        .build();
    let resp = match agent.get(url).call() {
        Ok(resp) => resp,
        Err(e) => {
            tracing::warn!(
                primary_error = %e,
                fallback_url = %FALLBACK_SUPERPOINT_MODEL_URL,
                "Primary SuperPoint download failed; attempting fallback mirror"
            );
            agent
                .get(FALLBACK_SUPERPOINT_MODEL_URL)
                .call()
                .map_err(|e| {
                    AlignmentError::ModelLoad(format!("Failed to download model from {url}: {e}"))
                })?
        }
    };

    let mut reader = resp.into_reader();
    let temp_path = model_path.with_extension("tmp");
    let mut file = std::fs::File::create(&temp_path).map_err(|e| {
        AlignmentError::ModelLoad(format!("Failed to create temporary model file: {e}"))
    })?;
    std::io::copy(&mut reader, &mut file)
        .map_err(|e| AlignmentError::ModelLoad(format!("Failed to write model file: {e}")))?;

    if let Some(expected) = expected_sha256 {
        let actual = compute_file_sha256(&temp_path).map_err(|e| {
            let _ = std::fs::remove_file(&temp_path);
            AlignmentError::ModelLoad(format!("Failed to compute hash of downloaded model: {e}"))
        })?;
        if !actual.eq_ignore_ascii_case(expected) {
            let _ = std::fs::remove_file(&temp_path);
            return Err(AlignmentError::ModelLoad(format!(
                "Downloaded model hash mismatch: expected {expected}, got {actual}"
            )));
        }
    }

    std::fs::rename(&temp_path, model_path)
        .map_err(|e| AlignmentError::ModelLoad(format!("Failed to finalize model file: {e}")))?;
    tracing::info!(
        target = %model_path.display(),
        "SuperPoint ONNX model cached successfully"
    );
    Ok(())
}

/// Target compute device / execution provider for model inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum BackendDevice {
    /// Automatically attempts GPU (CUDA) execution provider first, falling back to CPU if unavailable.
    #[default]
    Auto,
    /// Explicit host CPU execution with SIMD acceleration (AVX2, AVX512, NEON via ONNX Runtime).
    Cpu,
    /// Explicit NVIDIA CUDA GPU execution provider via ONNX Runtime.
    Cuda,
}

impl BackendDevice {
    /// Selects the default available execution device for the ONNX Runtime engine.
    ///
    /// Defaults to [`BackendDevice::Auto`], which attempts GPU acceleration with CPU fallback.
    #[must_use]
    pub const fn default_available() -> Self {
        Self::Auto
    }
}

/// Status of localized sub-pixel gradient tensor refinement for a keypoint.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum SubpixelStatus {
    /// Initial neural coordinate without refinement attempt.
    #[default]
    NeuralOnly,
    /// Successfully converged to a sub-pixel location.
    Refined {
        /// Sub-pixel displacement vector (dx, dy) in pixels.
        delta: (f32, f32),
        /// Number of gradient descent iterations performed.
        iterations: usize,
    },
    /// Sub-pixel refinement exceeded maximum drift threshold; reverted to neural coordinate.
    DriftRejected {
        /// Attempted displacement magnitude in pixels that caused rejection.
        drift_px: f32,
    },
    /// Structure tensor was ill-conditioned (1D aperture edge or low texture); skipped refinement.
    PoorConditioning {
        /// Minimum eigenvalue of the spatial structure tensor.
        min_eigenvalue: f32,
        /// Conditioning ratio lambda_min / lambda_max.
        cond_ratio: f32,
    },
    /// Keypoint lies too close to image boundary to extract full patch.
    BoundarySkipped,
}

/// A 2D feature keypoint with sub-pixel coordinates, confidence score, and optional descriptor vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyPoint {
    /// Sub-pixel 2D coordinates in local sub-frame pixel space.
    pub point: Point2D<f32>,
    /// Detection confidence score in `[0.0, 1.0]`.
    pub score: f32,
    /// High-dimensional descriptor vector (e.g. 256 dimensions for `SuperPoint`).
    pub descriptor: Option<Vec<f32>>,
    /// Status and metadata of localized sub-pixel gradient refinement.
    pub subpixel_status: SubpixelStatus,
}

impl KeyPoint {
    /// Constructs a new `KeyPoint` with point position, confidence score, and optional descriptor.
    #[inline]
    #[must_use]
    pub const fn new(point: Point2D<f32>, score: f32, descriptor: Option<Vec<f32>>) -> Self {
        Self {
            point,
            score,
            descriptor,
            subpixel_status: SubpixelStatus::NeuralOnly,
        }
    }

    /// Constructs a new `KeyPoint` with explicit sub-pixel refinement status.
    #[inline]
    #[must_use]
    pub const fn with_subpixel_status(
        point: Point2D<f32>,
        score: f32,
        descriptor: Option<Vec<f32>>,
        subpixel_status: SubpixelStatus,
    ) -> Self {
        Self {
            point,
            score,
            descriptor,
            subpixel_status,
        }
    }
}

/// Extracted sparse keypoints and local descriptors for an individual sub-frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureFrame {
    /// 0-indexed frame number within the film strip scan.
    pub frame_index: usize,
    /// Bounding rectangle of the sub-frame in normalized unit space.
    pub roi_bounds: NormalizedRect,
    /// Dimensions (width, height) of the processed frame image.
    pub image_size: Size2D<u32>,
    /// Detected keypoints.
    pub keypoints: Vec<KeyPoint>,
}

impl FeatureFrame {
    /// Creates a new `FeatureFrame` descriptor container.
    #[must_use]
    pub const fn new(
        frame_index: usize,
        roi_bounds: NormalizedRect,
        image_size: Size2D<u32>,
        keypoints: Vec<KeyPoint>,
    ) -> Self {
        Self {
            frame_index,
            roi_bounds,
            image_size,
            keypoints,
        }
    }

    /// Returns the number of extracted keypoints.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.keypoints.len()
    }

    /// Checks if no keypoints were extracted.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.keypoints.is_empty()
    }
}

/// Detailed descriptor matching score and geometric agreement metadata.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MatchScore {
    /// Raw descriptor cosine similarity / correlation score in `[-1.0, 1.0]`.
    pub similarity: f32,
    /// Normalized matching confidence score in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Nearest neighbor distance ratio (Lowe's ratio test: d1 / d2), if computed.
    pub distance_ratio: Option<f32>,
}

impl MatchScore {
    /// Constructs a `MatchScore` from a normalized confidence score.
    #[inline]
    #[must_use]
    pub const fn from_confidence(confidence: f32) -> Self {
        Self {
            similarity: confidence,
            confidence,
            distance_ratio: None,
        }
    }

    /// Constructs a detailed `MatchScore` with similarity and distance ratio.
    #[inline]
    #[must_use]
    pub const fn new(similarity: f32, confidence: f32, distance_ratio: Option<f32>) -> Self {
        Self {
            similarity,
            confidence,
            distance_ratio,
        }
    }
}

/// Match correspondence between two keypoint indices with structured confidence.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FeatureMatch {
    /// Index of the keypoint in frame A.
    pub index_a: usize,
    /// Index of the keypoint in frame B.
    pub index_b: usize,
    /// Matching confidence or similarity score.
    pub confidence: f32,
    /// Detailed matching score metadata.
    pub score: MatchScore,
}

impl FeatureMatch {
    /// Constructs a new `FeatureMatch` correspondence pair.
    #[inline]
    #[must_use]
    pub const fn new(index_a: usize, index_b: usize, confidence: f32) -> Self {
        Self {
            index_a,
            index_b,
            confidence,
            score: MatchScore::from_confidence(confidence),
        }
    }

    /// Constructs a `FeatureMatch` with detailed `MatchScore`.
    #[inline]
    #[must_use]
    pub const fn with_score(index_a: usize, index_b: usize, score: MatchScore) -> Self {
        Self {
            index_a,
            index_b,
            confidence: score.confidence,
            score,
        }
    }
}

/// Diagnostic observation tap for feature extraction and correspondence matching.
pub trait AlignmentDiagnosticTap: Send + Sync {
    /// Invoked when keypoints are extracted from an individual sub-frame.
    fn on_features_extracted(&self, _frame_idx: usize, _features: &FeatureFrame) {}

    /// Invoked when pairwise correspondences are established between two frames.
    fn on_matches_found(&self, _pair: FramePair, _matches: &[FeatureMatch]) {}

    /// Invoked when verified feature triplets satisfying the cascaded boundary condition are extracted.
    fn on_triplets_verified(&self, _triplets: &[FeatureTriplet]) {}

    /// Invoked when relative camera pose $(R, \mathbf{t})$ has been estimated.
    fn on_pose_estimated(&self, _pair: FramePair, _r: &RotationMatrix3x3, _t: &TranslationVector3) {
    }
}

/// Match direction between two frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatchDirection {
    /// Directed correspondence from Frame A to Frame B.
    Forward,
    /// Directed correspondence from Frame B to Frame A.
    Reverse,
    /// Verified mutual bidirectional consensus between Frame A and Frame B.
    Mutual,
}

/// Pairwise frame match result container.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairwiseMatchSet {
    /// Frame pair (`index_a`, `index_b`).
    pub pair: FramePair,
    /// Identified correspondences.
    pub matches: Vec<FeatureMatch>,
    /// Agreement status across forward/reverse searches.
    pub direction: MatchDirection,
}

impl PairwiseMatchSet {
    /// Creates a new `PairwiseMatchSet`.
    #[inline]
    #[must_use]
    pub const fn new(
        pair: FramePair,
        matches: Vec<FeatureMatch>,
        direction: MatchDirection,
    ) -> Self {
        Self {
            pair,
            matches,
            direction,
        }
    }

    /// Number of matched pairs.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.matches.len()
    }

    /// Whether the match set is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.matches.is_empty()
    }
}

/// A 3-view feature track corresponding to a single physical 3D scene point across all 3 frames.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FeatureTriplet {
    /// Keypoint index in Frame 0.
    pub index_0: usize,
    /// Keypoint index in Frame 1.
    pub index_1: usize,
    /// Keypoint index in Frame 2.
    pub index_2: usize,
    /// Geometric confidence score (geometric mean of pairwise confidence).
    pub confidence: f32,
    /// Relative horizontal disparity across baseline 0-1 (x0 - x1).
    pub disparity_01: f32,
    /// Relative horizontal disparity across baseline 1-2 (x1 - x2).
    pub disparity_12: f32,
    /// Cascaded disparity additivity error `|disp_02 - (disp_01 + disp_12)|`.
    pub cascade_error: f32,
}

/// Boundary condition configuration for rigid chassis cascaded transform and depth consistency.
///
/// # TODO (Lens Distortion Modeling)
/// Uncalibrated optical distortion from low-cost multi-lens plastic toy cameras (such as RETO 3D,
/// Nimslo, Nishika) causes non-linear peripheral curvature mismatch near outer boundaries.
/// When multi-view reprojection residual errors across outer frame boundaries exceed `0.5 px`,
/// incorporate per-lens radial ($k_1, k_2$) and tangential ($p_1, p_2$) distortion parameters
/// into joint bundle adjustment self-calibration:
/// - $x_d = x(1 + k_1 r^2 + k_2 r^4) + 2 p_1 x y + p_2(r^2 + 2 x^2)$
/// - $y_d = y(1 + k_1 r^2 + k_2 r^4) + p_1(r^2 + 2 y^2) + 2 p_2 x y$
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TripletConsistencyConfig {
    /// Nominal baseline ratio $B_{01} / B_{12}$ (1.0 for symmetric baseline).
    pub baseline_ratio: f32,
    /// Maximum allowed relative depth discrepancy between 0-1 and 1-2 disparities.
    pub max_disparity_ratio_deviation: f32,
    /// Maximum allowed cross-baseline jitter tolerance in pixels (vertical for horizontal strips, horizontal for vertical strips).
    pub max_cross_baseline_jitter_px: f32,
    /// Maximum allowed cascaded re-projection error in pixels `|disp_02 - (disp_01 + disp_12)|`.
    pub max_cascade_error_px: f32,
    /// Minimum disparity in pixels to avoid division by zero on points at optical infinity.
    pub min_disparity_px: f32,
    /// Radial boundary slack scaling factor for outer image regions (e.g. 1.5 relaxes tolerance up to 2.5x at corners).
    pub radial_slack_factor: f32,
    /// Strip orientation (horizontal vs vertical frame layout).
    pub orientation: StripOrientation,
}

impl Default for TripletConsistencyConfig {
    fn default() -> Self {
        Self {
            baseline_ratio: 1.0,
            max_disparity_ratio_deviation: 0.35,
            max_cross_baseline_jitter_px: 25.0,
            max_cascade_error_px: 5.0,
            min_disparity_px: 0.1,
            radial_slack_factor: 1.5,
            orientation: StripOrientation::Horizontal,
        }
    }
}

impl TripletConsistencyConfig {
    /// Creates a new `TripletConsistencyConfig` for the given strip orientation.
    #[inline]
    #[must_use]
    pub const fn with_orientation(orientation: StripOrientation) -> Self {
        Self {
            baseline_ratio: 1.0,
            max_disparity_ratio_deviation: 0.35,
            max_cross_baseline_jitter_px: 25.0,
            max_cascade_error_px: 5.0,
            min_disparity_px: 0.1,
            radial_slack_factor: 1.5,
            orientation,
        }
    }

    /// Verifies if a candidate feature triplet satisfies the rigid cascaded transform boundary condition.
    ///
    /// # Arguments
    /// * `pt0` - 2D coordinate in Frame 0.
    /// * `pt1` - 2D coordinate in Frame 1.
    /// * `pt2` - 2D coordinate in Frame 2.
    ///
    /// # Returns
    /// `Some((disparity_01, disparity_12, cascade_error))` if inlier, else `None`.
    #[inline]
    #[must_use]
    pub fn verify_triplet(
        &self,
        pt0: Point2D<f32>,
        pt1: Point2D<f32>,
        pt2: Point2D<f32>,
    ) -> Option<(f32, f32, f32)> {
        self.verify_triplet_with_bounds(pt0, pt1, pt2, None)
    }

    /// Verifies if a candidate feature triplet satisfies the rigid cascaded transform boundary condition,
    /// with optional radius-dependent boundary slack expansion for uncalibrated wide-angle distortion.
    ///
    /// # Arguments
    /// * `pt0` - 2D coordinate in Frame 0.
    /// * `pt1` - 2D coordinate in Frame 1.
    /// * `pt2` - 2D coordinate in Frame 2.
    /// * `frame_size` - Optional dimensions of the sub-frame to compute normalized radius.
    ///
    /// # Returns
    /// `Some((disparity_01, disparity_12, cascade_error))` if inlier, else `None`.
    #[must_use]
    #[allow(clippy::suboptimal_flops, clippy::cast_precision_loss)]
    pub fn verify_triplet_with_bounds(
        &self,
        pt0: Point2D<f32>,
        pt1: Point2D<f32>,
        pt2: Point2D<f32>,
        frame_size: Option<Size2D<u32>>,
    ) -> Option<(f32, f32, f32)> {
        // Project onto baseline axis (parallax) and orthogonal cross-baseline axis (epipolar jitter)
        let (p0_base, p0_cross, p1_base, p1_cross, p2_base, p2_cross) = match self.orientation {
            StripOrientation::Horizontal => (pt0.x, pt0.y, pt1.x, pt1.y, pt2.x, pt2.y),
            StripOrientation::Vertical => (pt0.y, pt0.x, pt1.y, pt1.x, pt2.y, pt2.x),
        };

        // Compute radial boundary slack multiplier if frame size is provided
        let slack = frame_size.map_or(1.0_f32, |sz| {
            let cx = sz.width as f32 * 0.5;
            let cy = sz.height as f32 * 0.5;
            let r_max_sq = cx * cx + cy * cy;
            if r_max_sq > 1e-3 {
                let dx = pt0.x - cx;
                let dy = pt0.y - cy;
                let r_norm_sq = (dx * dx + dy * dy) / r_max_sq;
                1.0 + self.radial_slack_factor * r_norm_sq.min(1.0)
            } else {
                1.0
            }
        });

        let eff_max_cross_jitter = self.max_cross_baseline_jitter_px * slack;
        let eff_max_cascade_err = self.max_cascade_error_px * slack;

        // 1. Cross-baseline jitter bound across all view pairs
        let d_cross_01 = (p0_cross - p1_cross).abs();
        let d_cross_12 = (p1_cross - p2_cross).abs();
        let d_cross_02 = (p0_cross - p2_cross).abs();
        if d_cross_01 > eff_max_cross_jitter
            || d_cross_12 > eff_max_cross_jitter
            || d_cross_02 > eff_max_cross_jitter
        {
            return None;
        }

        // 2. Collinear along-baseline disparity calculation across all 3 view combinations
        let raw_disp_01 = p0_base - p1_base;
        let raw_disp_12 = p1_base - p2_base;
        let raw_disp_02 = p0_base - p2_base;

        let disp_01 = raw_disp_01.abs();
        let disp_12 = raw_disp_12.abs();
        let disp_02 = raw_disp_02.abs();

        // Direction check across all pairs: enforce sign consistency when above unrectified mounting jitter
        let noise_floor = 4.0_f32;
        if disp_01 > noise_floor
            && disp_12 > noise_floor
            && (raw_disp_01.signum() - raw_disp_12.signum()).abs() > 0.01
        {
            return None;
        }
        if disp_02 > noise_floor
            && disp_01 > noise_floor
            && (raw_disp_02.signum() - raw_disp_01.signum()).abs() > 0.01
        {
            return None;
        }

        // 3. Disparity ratio consistency check across baselines
        if disp_01 >= self.min_disparity_px && disp_12 >= self.min_disparity_px {
            let ratio = disp_01 / disp_12;
            let ratio_dev = (ratio - self.baseline_ratio).abs() / self.baseline_ratio;
            if ratio_dev > self.max_disparity_ratio_deviation {
                return None;
            }
        }

        // 4. Cascaded baseline transform additivity & long-baseline anchor consistency
        // Expected displacement on baseline 1-2 from baseline 0-1 scaled by baseline_ratio
        let expected_raw_disp_12 = raw_disp_01 / self.baseline_ratio;
        let cascade_error = (raw_disp_12 - expected_raw_disp_12).abs();
        if cascade_error > eff_max_cascade_err {
            return None;
        }

        Some((disp_01, disp_12, cascade_error))
    }
}

/// A dyadic interval span in the hierarchical reduction tree across 1D adjacent camera frames.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DyadicSpan {
    /// Starting frame index $i$.
    pub start_frame: usize,
    /// Ending frame index $k$.
    pub end_frame: usize,
    /// Midpoint partition frame index $j$ where the span is recursively decomposed.
    pub mid_frame: usize,
    /// Tree depth level (0 for adjacent leaves with stride 1, 1 for stride 2, etc.).
    pub level: usize,
    /// Observed relative translation `(along_baseline_dx, cross_baseline_dy)` from direct feature matches in pixels.
    pub observed_translation: (f32, f32),
    /// Composed relative translation `(along_baseline_dx, cross_baseline_dy)` from child spans in pixels.
    pub composed_translation: (f32, f32),
    /// Hierarchical chord closure residual error in pixels `|observed - composed|`.
    pub chord_residual_px: f32,
    /// Number of pairwise feature correspondences anchoring this span.
    pub match_count: usize,
}

/// Detailed diagnostic report comparing camera array extrinsics and residual errors before and after hierarchical optimization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchicalOptimizationReport {
    /// Number of camera frames in the 1D sequential array.
    pub num_cameras: usize,
    /// Root-mean-square chord consistency error in pixels before joint optimization.
    pub rmse_before_px: f32,
    /// Root-mean-square residual error in pixels after joint optimization.
    pub rmse_after_px: f32,
    /// Relative residual error reduction percentage: `(rmse_before - rmse_after) / rmse_before * 100.0`.
    pub relative_improvement_pct: f32,
    /// Optimized camera positions `(along_baseline_x, cross_baseline_y)` relative to Camera 0 at `(0, 0)`.
    pub camera_positions: Vec<(f32, f32)>,
    /// Optimized relative translations between adjacent lenses `(0->1, 1->2, ..., N-2->N-1)`.
    pub adjacent_translations: Vec<(f32, f32)>,
    /// Vertical sag of interior lenses relative to the straight chord connecting Camera 0 and Camera N-1.
    pub center_sags_px: Vec<f32>,
    /// Per-level root-mean-square residual errors in pixels across reduction tree levels.
    pub level_rmse_px: Vec<f32>,
    /// Diagnostic records for all dyadic spans evaluated in the reduction tree.
    pub dyadic_spans: Vec<DyadicSpan>,
}

/// Configuration for the hierarchical reduction tree and joint extrinsic optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HierarchicalExtrinsicsConfig {
    /// Huber loss threshold for robust outlier downweighting in pixels (e.g. 2.0 px).
    pub huber_threshold_px: f32,
    /// Regularization weight for hierarchical chord closure consistency constraints.
    pub chord_consistency_weight: f32,
    /// Maximum optimization iterations for IRLS (Iteratively Reweighted Least Squares).
    pub max_iterations: usize,
    /// Convergence tolerance on parameter update norm.
    pub convergence_epsilon: f32,
}

impl Default for HierarchicalExtrinsicsConfig {
    fn default() -> Self {
        Self {
            huber_threshold_px: 2.0,
            chord_consistency_weight: 1.0,
            max_iterations: 10,
            convergence_epsilon: 1e-4,
        }
    }
}

/// Solves a small dense linear system `A * x = b` using Gaussian elimination with partial pivoting.
#[inline]
#[allow(clippy::needless_range_loop)]
fn solve_linear_system_dynamic(
    n: usize,
    a_in: &[Vec<f32>],
    b_in: &[f32],
) -> Option<Vec<f32>> {
    if n == 0 || a_in.len() < n || b_in.len() < n {
        return None;
    }
    let mut a = a_in.to_vec();
    let mut b = b_in.to_vec();

    for i in 0..n {
        let mut max_row = i;
        let mut max_val = a[i][i].abs();
        for k in (i + 1)..n {
            let val = a[k][i].abs();
            if val > max_val {
                max_val = val;
                max_row = k;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        if max_row != i {
            a.swap(i, max_row);
            b.swap(i, max_row);
        }
        let pivot = a[i][i];
        for k in (i + 1)..n {
            let factor = a[k][i] / pivot;
            for j in i..n {
                let subtrahend = factor * a[i][j];
                a[k][j] -= subtrahend;
            }
            let b_sub = factor * b[i];
            b[k] -= b_sub;
            a[k][i] = 0.0;
        }
    }

    let mut x = vec![0.0_f32; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }
    Some(x)
}

/// Hierarchical binary reduction tree for $O(N \log N)$ multi-lens camera array extrinsic optimization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchicalReductionTree {
    /// Number of camera frames in the sequential 1D array.
    pub num_frames: usize,
    /// Tree levels containing dyadic spans grouped by stride ($2^h$).
    pub spans: Vec<DyadicSpan>,
}

impl HierarchicalReductionTree {
    /// Builds a hierarchical reduction tree from observed pairwise matches and frame coordinates.
    ///
    /// # Arguments
    /// * `frames` - Slice of feature frames ($N \ge 2$).
    /// * `match_sets` - Slice of pairwise match sets between frame pairs.
    /// * `orientation` - Scan strip layout orientation.
    ///
    /// # Returns
    /// Constructed `HierarchicalReductionTree` with initial observation stats and child compositions.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::similar_names,
        clippy::too_many_lines,
        clippy::imprecise_flops
    )]
    pub fn build_from_match_sets(
        frames: &[FeatureFrame],
        match_sets: &[PairwiseMatchSet],
        orientation: StripOrientation,
    ) -> Self {
        let n = frames.len();
        if n < 2 {
            return Self {
                num_frames: n,
                spans: Vec::new(),
            };
        }

        // Map pairwise matches by (min(a,b), max(a,b)) -> ((med_base, med_cross), count)
        let mut obs_map: HashMap<(usize, usize), ((f32, f32), usize)> = HashMap::new();
        for ms in match_sets {
            let (f_a, f_b) = ms.pair;
            if f_a >= n || f_b >= n || f_a == f_b {
                continue;
            }
            let (src, dst) = if f_a < f_b { (f_a, f_b) } else { (f_b, f_a) };
            let mut d_base = Vec::with_capacity(ms.matches.len());
            let mut d_cross = Vec::with_capacity(ms.matches.len());

            for m in &ms.matches {
                let (idx_src, idx_dst) = if f_a < f_b {
                    (m.index_a, m.index_b)
                } else {
                    (m.index_b, m.index_a)
                };
                if idx_src < frames[src].keypoints.len() && idx_dst < frames[dst].keypoints.len() {
                    let ps = frames[src].keypoints[idx_src].point;
                    let pd = frames[dst].keypoints[idx_dst].point;
                    let (ps_b, ps_c, pd_b, pd_c) = match orientation {
                        StripOrientation::Horizontal => (ps.x, ps.y, pd.x, pd.y),
                        StripOrientation::Vertical => (ps.y, ps.x, pd.y, pd.x),
                    };
                    d_base.push(ps_b - pd_b);
                    d_cross.push(pd_c - ps_c);
                }
            }

            if !d_base.is_empty() {
                let med_b = compute_median(&mut d_base);
                let med_c = compute_median(&mut d_cross);
                obs_map.insert((src, dst), ((med_b, med_c), d_base.len()));
            }
        }

        // Generate dyadic spans across levels
        let mut spans = Vec::new();
        let mut stride = 1usize;
        let mut level = 0usize;

        while stride < n {
            for i in 0..n {
                let k = i + stride;
                if k >= n {
                    break;
                }
                let mid = i + stride / 2;
                let (obs, count) = obs_map.get(&(i, k)).copied().unwrap_or(((0.0, 0.0), 0));
                spans.push(DyadicSpan {
                    start_frame: i,
                    end_frame: k,
                    mid_frame: mid,
                    level,
                    observed_translation: obs,
                    composed_translation: (0.0, 0.0),
                    chord_residual_px: 0.0,
                    match_count: count,
                });
            }
            stride *= 2;
            level += 1;
        }

        // Ensure root span (0, n-1) is included
        if n > 2 && !spans.iter().any(|s| s.start_frame == 0 && s.end_frame == n - 1) {
            let mid = n / 2;
            let (obs, count) = obs_map.get(&(0, n - 1)).copied().unwrap_or(((0.0, 0.0), 0));
            spans.push(DyadicSpan {
                start_frame: 0,
                end_frame: n - 1,
                mid_frame: mid,
                level,
                observed_translation: obs,
                composed_translation: (0.0, 0.0),
                chord_residual_px: 0.0,
                match_count: count,
            });
        }

        // Compute bottom-up composed translations and chord residuals
        let mut span_val_map: HashMap<(usize, usize), (f32, f32)> = HashMap::new();
        // First populate level 0 (leaves)
        for s in &mut spans {
            if s.level == 0 {
                s.composed_translation = s.observed_translation;
                span_val_map.insert((s.start_frame, s.end_frame), s.observed_translation);
            }
        }

        // Then compute higher levels
        for s in &mut spans {
            if s.level > 0 {
                let t_left = span_val_map
                    .get(&(s.start_frame, s.mid_frame))
                    .copied()
                    .unwrap_or_default();
                let t_right = span_val_map
                    .get(&(s.mid_frame, s.end_frame))
                    .copied()
                    .unwrap_or_default();
                let comp = (t_left.0 + t_right.0, t_left.1 + t_right.1);
                s.composed_translation = comp;
                span_val_map.insert((s.start_frame, s.end_frame), comp);

                if s.match_count > 0 {
                    let d0 = s.observed_translation.0 - comp.0;
                    let d1 = s.observed_translation.1 - comp.1;
                    s.chord_residual_px = d0.hypot(d1);
                } else {
                    s.observed_translation = comp;
                    s.chord_residual_px = 0.0;
                }
            }
        }

        Self {
            num_frames: n,
            spans,
        }
    }

    /// Optimizes camera array extrinsics and chord consistency using robust Iteratively Reweighted Least Squares (IRLS).
    ///
    /// # Arguments
    /// * `config` - Optimization parameters and Huber loss gating settings.
    ///
    /// # Returns
    /// A [`HierarchicalOptimizationReport`] containing before/after RMSE, relative improvement,
    /// optimized camera positions, and center sags.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::similar_names,
        clippy::suboptimal_flops,
        clippy::too_many_lines
    )]
    pub fn optimize(&self, config: &HierarchicalExtrinsicsConfig) -> HierarchicalOptimizationReport {
        let n = self.num_frames;
        if n < 2 || self.spans.is_empty() {
            return HierarchicalOptimizationReport {
                num_cameras: n,
                rmse_before_px: 0.0,
                rmse_after_px: 0.0,
                relative_improvement_pct: 0.0,
                camera_positions: vec![(0.0, 0.0); n],
                adjacent_translations: vec![(0.0, 0.0); n.saturating_sub(1)],
                center_sags_px: vec![0.0; n.saturating_sub(2)],
                level_rmse_px: Vec::new(),
                dyadic_spans: self.spans.clone(),
            };
        }

        // 1. Evaluate pre-optimization RMSE across non-leaf dyadic spans
        let mut pre_sq_err = 0.0_f32;
        let mut pre_count = 0usize;
        let mut max_level = 0usize;

        for s in &self.spans {
            max_level = max_level.max(s.level);
            if s.level > 0 && s.match_count > 0 {
                pre_sq_err += s.chord_residual_px * s.chord_residual_px;
                pre_count += 1;
            }
        }

        let rmse_before_px = if pre_count > 0 {
            (pre_sq_err / pre_count as f32).sqrt()
        } else {
            0.0
        };

        // 2. Solve linear least squares for camera positions [t_1 ... t_{n-1}] (t_0 = 0)
        let num_vars = n - 1;
        let mut pos_x = vec![0.0_f32; n];
        let mut pos_y = vec![0.0_f32; n];

        // Initialize from adjacent level 0 leaves
        for s in &self.spans {
            if s.level == 0 && s.start_frame + 1 == s.end_frame {
                pos_x[s.end_frame] = pos_x[s.start_frame] + s.observed_translation.0;
                pos_y[s.end_frame] = pos_y[s.start_frame] + s.observed_translation.1;
            }
        }

        let delta = config.huber_threshold_px.max(0.1);

        // Run IRLS iterations
        for _iter in 0..config.max_iterations {
            let mut u_x = vec![0.0_f32; num_vars];
            let mut u_y = vec![0.0_f32; num_vars];

            for coord in 0..2 {
                let mut h_mat = vec![vec![0.0_f32; num_vars]; num_vars];
                let mut g_vec = vec![0.0_f32; num_vars];

                // Tikhonov damping for numerical stability
                for (i, row) in h_mat.iter_mut().enumerate().take(num_vars) {
                    row[i] += 1e-6;
                }

                // Accumulate direct observed constraints
                for s in &self.spans {
                    if s.match_count == 0 {
                        continue;
                    }
                    let target = if coord == 0 {
                        s.observed_translation.0
                    } else {
                        s.observed_translation.1
                    };
                    let cur_pred = if coord == 0 {
                        pos_x[s.end_frame] - pos_x[s.start_frame]
                    } else {
                        pos_y[s.end_frame] - pos_y[s.start_frame]
                    };
                    let r = (cur_pred - target).abs();
                    let huber_w = if r <= delta { 1.0 } else { delta / r };
                    let w = (s.match_count as f32).sqrt() * huber_w;

                    let mut a_row = vec![0.0_f32; num_vars];
                    if s.end_frame > 0 {
                        a_row[s.end_frame - 1] += 1.0;
                    }
                    if s.start_frame > 0 {
                        a_row[s.start_frame - 1] -= 1.0;
                    }

                    for i in 0..num_vars {
                        if a_row[i].abs() > 1e-7 {
                            g_vec[i] += w * a_row[i] * target;
                            for j in 0..num_vars {
                                if a_row[j].abs() > 1e-7 {
                                    h_mat[i][j] += w * a_row[i] * a_row[j];
                                }
                            }
                        }
                    }
                }

                // Accumulate hierarchical chord closure consistency constraints
                if config.chord_consistency_weight > 0.0 {
                    for s in &self.spans {
                        if s.level == 0 {
                            continue;
                        }
                        let target = if coord == 0 {
                            s.composed_translation.0
                        } else {
                            s.composed_translation.1
                        };
                        let cur_pred = if coord == 0 {
                            pos_x[s.end_frame] - pos_x[s.start_frame]
                        } else {
                            pos_y[s.end_frame] - pos_y[s.start_frame]
                        };
                        let r = (cur_pred - target).abs();
                        let huber_w = if r <= delta { 1.0 } else { delta / r };
                        let w = config.chord_consistency_weight * huber_w;

                        let mut a_row = vec![0.0_f32; num_vars];
                        if s.end_frame > 0 {
                            a_row[s.end_frame - 1] += 1.0;
                        }
                        if s.start_frame > 0 {
                            a_row[s.start_frame - 1] -= 1.0;
                        }

                        for i in 0..num_vars {
                            if a_row[i].abs() > 1e-7 {
                                g_vec[i] += w * a_row[i] * target;
                                for j in 0..num_vars {
                                    if a_row[j].abs() > 1e-7 {
                                        h_mat[i][j] += w * a_row[i] * a_row[j];
                                    }
                                }
                            }
                        }
                    }
                }

                if let Some(sol) = solve_linear_system_dynamic(num_vars, &h_mat, &g_vec) {
                    if coord == 0 {
                        u_x = sol;
                    } else {
                        u_y = sol;
                    }
                }
            }

            // Update positions
            let mut max_change = 0.0_f32;
            for v in 1..n {
                let dx = u_x[v - 1] - pos_x[v];
                let dy = u_y[v - 1] - pos_y[v];
                max_change = max_change.max(dx.abs().max(dy.abs()));
                pos_x[v] = u_x[v - 1];
                pos_y[v] = u_y[v - 1];
            }

            if max_change < config.convergence_epsilon {
                break;
            }
        }

        // 3. Post-optimization diagnostics
        let camera_positions: Vec<(f32, f32)> =
            pos_x.iter().copied().zip(pos_y.iter().copied()).collect();

        let mut adjacent_translations = Vec::with_capacity(n - 1);
        for v in 0..n - 1 {
            adjacent_translations.push((pos_x[v + 1] - pos_x[v], pos_y[v + 1] - pos_y[v]));
        }

        // Interior camera center sags relative to the total (0, N-1) chord
        let chord_dy = pos_y[n - 1] - pos_y[0];
        let mut center_sags_px = Vec::with_capacity(n.saturating_sub(2));
        for v in 1..n - 1 {
            let frac = v as f32 / (n - 1) as f32;
            let expected_y = pos_y[0] + frac * chord_dy;
            center_sags_px.push(pos_y[v] - expected_y);
        }

        // Updated dyadic spans and per-level post-optimization RMSE
        let mut updated_spans = self.spans.clone();
        let mut post_sq_err = 0.0_f32;
        let mut post_count = 0usize;
        let mut level_sq = vec![0.0_f32; max_level + 1];
        let mut level_cnt = vec![0usize; max_level + 1];

        for s in &mut updated_spans {
            let opt_dx = pos_x[s.end_frame] - pos_x[s.start_frame];
            let opt_dy = pos_y[s.end_frame] - pos_y[s.start_frame];
            s.composed_translation = (opt_dx, opt_dy);

            if s.match_count > 0 {
                let rx = s.observed_translation.0 - opt_dx;
                let ry = s.observed_translation.1 - opt_dy;
                let res = rx.hypot(ry);
                s.chord_residual_px = res;

                post_sq_err += res * res;
                post_count += 1;

                if s.level <= max_level {
                    level_sq[s.level] += res * res;
                    level_cnt[s.level] += 1;
                }
            }
        }

        let rmse_after_px = if post_count > 0 {
            (post_sq_err / post_count as f32).sqrt()
        } else {
            0.0
        };

        let relative_improvement_pct = if rmse_before_px > 1e-4 {
            ((rmse_before_px - rmse_after_px) / rmse_before_px) * 100.0
        } else {
            0.0
        };

        let level_rmse_px: Vec<f32> = level_sq
            .iter()
            .zip(level_cnt.iter())
            .map(|(&sq, &cnt)| if cnt > 0 { (sq / cnt as f32).sqrt() } else { 0.0 })
            .collect();

        tracing::info!(
            num_cameras = n,
            rmse_before_px,
            rmse_after_px,
            relative_improvement_pct,
            "Hierarchical camera array extrinsic optimization completed"
        );

        HierarchicalOptimizationReport {
            num_cameras: n,
            rmse_before_px,
            rmse_after_px,
            relative_improvement_pct,
            camera_positions,
            adjacent_translations,
            center_sags_px,
            level_rmse_px,
            dyadic_spans: updated_spans,
        }
    }
}

/// Estimated 6-DoF chassis extrinsics and multi-view geometric alignment properties across sub-frames.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChassisExtrinsics {
    /// Relative translation from Frame 0 to Frame 1 `(along_baseline_dx, cross_baseline_dy)` in pixels.
    pub translation_01: (f32, f32),
    /// Relative translation from Frame 1 to Frame 2 `(along_baseline_dx, cross_baseline_dy)` in pixels.
    pub translation_12: (f32, f32),
    /// Relative translation from Frame 0 to Frame 2 `(along_baseline_dx, cross_baseline_dy)` in pixels.
    pub translation_02: (f32, f32),
    /// Physical vertical center sag in pixels (deviation of Center lens L1 from the L0-L2 chord).
    pub center_sag_px: f32,
    /// Measured empirical baseline ratio `|t_01| / |t_12|`.
    pub empirical_baseline_ratio: f32,
    /// Root-mean-square reprojection / disparity consistency error in pixels.
    pub rmse_consistency_px: f32,
    /// Number of verified triplet feature tracks used for estimation.
    pub inlier_count: usize,
    /// Hierarchical reduction tree optimization report for multi-lens arrays ($N \ge 3$).
    #[serde(default)]
    pub hierarchical_report: Option<HierarchicalOptimizationReport>,
}

impl Default for ChassisExtrinsics {
    fn default() -> Self {
        Self {
            translation_01: (0.0, 0.0),
            translation_12: (0.0, 0.0),
            translation_02: (0.0, 0.0),
            center_sag_px: 0.0,
            empirical_baseline_ratio: 1.0,
            rmse_consistency_px: 0.0,
            inlier_count: 0,
            hierarchical_report: None,
        }
    }
}

impl ChassisExtrinsics {
    /// Estimates multi-lens chassis extrinsics and geometric offsets from verified feature triplets.
    ///
    /// # Arguments
    /// * `triplets` - Slice of verified feature triplets.
    /// * `frames` - Slice of feature frames corresponding to sub-views (0, 1, 2).
    /// * `orientation` - Scan strip layout orientation.
    ///
    /// # Returns
    /// Estimated `ChassisExtrinsics` geometry summary.
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
    pub fn estimate_from_triplets(
        triplets: &[FeatureTriplet],
        frames: &[FeatureFrame],
        orientation: StripOrientation,
    ) -> Self {
        if triplets.is_empty() || frames.len() < 3 {
            return Self::default();
        }

        let mut d_base_01 = Vec::with_capacity(triplets.len());
        let mut d_cross_01 = Vec::with_capacity(triplets.len());
        let mut d_base_12 = Vec::with_capacity(triplets.len());
        let mut d_cross_12 = Vec::with_capacity(triplets.len());
        let mut d_base_02 = Vec::with_capacity(triplets.len());
        let mut d_cross_02 = Vec::with_capacity(triplets.len());
        let mut sum_sq_cascade = 0.0_f32;

        let mut m01_matches = Vec::with_capacity(triplets.len());
        let mut m12_matches = Vec::with_capacity(triplets.len());
        let mut m02_matches = Vec::with_capacity(triplets.len());

        for t in triplets {
            let p0 = frames[0].keypoints[t.index_0].point;
            let p1 = frames[1].keypoints[t.index_1].point;
            let p2 = frames[2].keypoints[t.index_2].point;

            let (p0_b, p0_c, p1_b, p1_c, p2_b, p2_c) = match orientation {
                StripOrientation::Horizontal => (p0.x, p0.y, p1.x, p1.y, p2.x, p2.y),
                StripOrientation::Vertical => (p0.y, p0.x, p1.y, p1.x, p2.y, p2.x),
            };

            d_base_01.push(p0_b - p1_b);
            d_cross_01.push(p1_c - p0_c);
            d_base_12.push(p1_b - p2_b);
            d_cross_12.push(p2_c - p1_c);
            d_base_02.push(p0_b - p2_b);
            d_cross_02.push(p2_c - p0_c);
            sum_sq_cascade += t.cascade_error * t.cascade_error;

            m01_matches.push(FeatureMatch::new(t.index_0, t.index_1, t.confidence));
            m12_matches.push(FeatureMatch::new(t.index_1, t.index_2, t.confidence));
            m02_matches.push(FeatureMatch::new(t.index_0, t.index_2, t.confidence));
        }

        let med_base_01 = compute_median(&mut d_base_01);
        let med_cross_01 = compute_median(&mut d_cross_01);
        let med_base_12 = compute_median(&mut d_base_12);
        let med_cross_12 = compute_median(&mut d_cross_12);
        let med_base_02 = compute_median(&mut d_base_02);
        let med_cross_02 = compute_median(&mut d_cross_02);

        // Center lens sag relative to the chord connecting L0 and L2
        let center_sag_px = med_cross_01 - 0.5 * med_cross_02;
        let empirical_baseline_ratio = if med_base_12.abs() > 1e-4 {
            med_base_01 / med_base_12
        } else {
            1.0
        };
        let rmse_consistency_px = (sum_sq_cascade / triplets.len() as f32).sqrt();

        // Build hierarchical reduction tree optimization report
        let match_sets = [
            PairwiseMatchSet::new((0, 1), m01_matches, MatchDirection::Mutual),
            PairwiseMatchSet::new((1, 2), m12_matches, MatchDirection::Mutual),
            PairwiseMatchSet::new((0, 2), m02_matches, MatchDirection::Mutual),
        ];
        let tree = HierarchicalReductionTree::build_from_match_sets(
            &frames[0..3],
            &match_sets,
            orientation,
        );
        let report = tree.optimize(&HierarchicalExtrinsicsConfig::default());

        Self {
            translation_01: (med_base_01, med_cross_01),
            translation_12: (med_base_12, med_cross_12),
            translation_02: (med_base_02, med_cross_02),
            center_sag_px,
            empirical_baseline_ratio,
            rmse_consistency_px,
            inlier_count: triplets.len(),
            hierarchical_report: Some(report),
        }
    }

    /// Estimates multi-lens chassis extrinsics and geometric offsets for general $N \ge 3$ lens arrays
    /// using hierarchical binary tree reduction.
    ///
    /// # Arguments
    /// * `frames` - Slice of feature frames ($N \ge 3$).
    /// * `match_sets` - Slice of pairwise match sets between camera views.
    /// * `orientation` - Scan strip layout orientation.
    /// * `config` - Optional optimization configuration.
    ///
    /// # Returns
    /// Estimated `ChassisExtrinsics` geometry summary and hierarchical optimization report.
    #[must_use]
    pub fn estimate_hierarchical(
        frames: &[FeatureFrame],
        match_sets: &[PairwiseMatchSet],
        orientation: StripOrientation,
        config: Option<&HierarchicalExtrinsicsConfig>,
    ) -> Self {
        let n = frames.len();
        if n < 3 || match_sets.is_empty() {
            return Self::default();
        }
        let cfg = config.copied().unwrap_or_default();
        let tree = HierarchicalReductionTree::build_from_match_sets(frames, match_sets, orientation);
        let report = tree.optimize(&cfg);

        let t01 = report.adjacent_translations.first().copied().unwrap_or((0.0, 0.0));
        let t12 = report.adjacent_translations.get(1).copied().unwrap_or((0.0, 0.0));
        let t02 = if report.camera_positions.len() >= 3 {
            report.camera_positions[2]
        } else {
            (t01.0 + t12.0, t01.1 + t12.1)
        };
        let center_sag_px = report.center_sags_px.first().copied().unwrap_or(0.0);
        let empirical_baseline_ratio = if t12.0.abs() > 1e-4 {
            t01.0 / t12.0
        } else {
            1.0
        };
        let total_inliers: usize = match_sets.iter().map(PairwiseMatchSet::len).sum();

        Self {
            translation_01: t01,
            translation_12: t12,
            translation_02: t02,
            center_sag_px,
            empirical_baseline_ratio,
            rmse_consistency_px: report.rmse_after_px,
            inlier_count: total_inliers,
            hierarchical_report: Some(report),
        }
    }
}

#[inline]
fn compute_median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        f32::midpoint(values[mid - 1], values[mid])
    } else {
        values[mid]
    }
}

/// Core interface for feature correspondence matching and triplet verification across sub-frames.
pub trait FeatureMatcher: Send + Sync {
    /// Computes pairwise matches between two sub-frames in directed order (A -> B).
    ///
    /// # Errors
    /// Returns [`AlignmentError`] if model inference or tensor layout extraction fails.
    fn match_pair(
        &self,
        frame_a: &FeatureFrame,
        frame_b: &FeatureFrame,
    ) -> AlignmentResult<PairwiseMatchSet>;

    /// Computes mutual bidirectional matches between two frames (A <-> B).
    ///
    /// # Errors
    /// Returns [`AlignmentError`] if model inference fails on either forward or reverse matching.
    fn match_pair_bidirectional(
        &self,
        frame_a: &FeatureFrame,
        frame_b: &FeatureFrame,
    ) -> AlignmentResult<PairwiseMatchSet> {
        let fwd = self.match_pair(frame_a, frame_b)?;
        let rev = self.match_pair(frame_b, frame_a)?;

        // Index reverse matches by (index_b, index_a)
        let mut rev_map = HashMap::with_capacity(rev.matches.len());
        for m in &rev.matches {
            rev_map.insert((m.index_a, m.index_b), m.confidence);
        }

        let mut mutual = Vec::new();
        for m in &fwd.matches {
            if let Some(&rev_conf) = rev_map.get(&(m.index_b, m.index_a)) {
                let conf = m.confidence.min(rev_conf);
                mutual.push(FeatureMatch::new(m.index_a, m.index_b, conf));
            }
        }

        Ok(PairwiseMatchSet::new(
            (frame_a.frame_index, frame_b.frame_index),
            mutual,
            MatchDirection::Mutual,
        ))
    }

    /// Computes pairwise correspondences for all triplet baselines: (0, 1), (1, 2), and (0, 2).
    ///
    /// # Errors
    /// Returns [`AlignmentError`] if frame slice count is less than 3 or pairwise matching fails.
    fn match_triplet_pairs(
        &self,
        frames: &[FeatureFrame],
        bidirectional: bool,
    ) -> AlignmentResult<[PairwiseMatchSet; 3]> {
        if frames.len() < 3 {
            return Err(AlignmentError::InvalidInput(
                "Expected at least 3 feature frames for triplet matching".to_string(),
            ));
        }

        let pairs = [(0, 1), (1, 2), (0, 2)];
        let m01 = if bidirectional {
            self.match_pair_bidirectional(&frames[pairs[0].0], &frames[pairs[0].1])?
        } else {
            self.match_pair(&frames[pairs[0].0], &frames[pairs[0].1])?
        };

        let m12 = if bidirectional {
            self.match_pair_bidirectional(&frames[pairs[1].0], &frames[pairs[1].1])?
        } else {
            self.match_pair(&frames[pairs[1].0], &frames[pairs[1].1])?
        };

        let m02 = if bidirectional {
            self.match_pair_bidirectional(&frames[pairs[2].0], &frames[pairs[2].1])?
        } else {
            self.match_pair(&frames[pairs[2].0], &frames[pairs[2].1])?
        };

        Ok([m01, m12, m02])
    }

    /// Extracts depth-consistent feature triplets across 3 views, filtering out any matches
    /// that violate the rigid cascaded transform boundary condition.
    ///
    /// # Errors
    /// Returns [`AlignmentError`] if frames count is less than 3 or pairwise matching fails.
    #[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
    fn extract_consistent_triplets(
        &self,
        frames: &[FeatureFrame],
        config: &TripletConsistencyConfig,
        tap: Option<&dyn AlignmentDiagnosticTap>,
    ) -> AlignmentResult<Vec<FeatureTriplet>> {
        if frames.len() < 3 {
            return Err(AlignmentError::InvalidInput(
                "Expected at least 3 feature frames for triplet extraction".to_string(),
            ));
        }

        let [m01, m12, m02] = self.match_triplet_pairs(frames, true)?;

        tracing::info!(
            m01_count = m01.len(),
            m12_count = m12.len(),
            m02_count = m02.len(),
            "Pairwise bidirectional match counts"
        );

        if let Some(t) = tap {
            t.on_matches_found(m01.pair, &m01.matches);
            t.on_matches_found(m12.pair, &m12.matches);
            t.on_matches_found(m02.pair, &m02.matches);
        }

        // Build adjacency map from 0 -> 1
        let mut map_01: HashMap<usize, (usize, f32)> = HashMap::with_capacity(m01.len());
        for m in &m01.matches {
            map_01.insert(m.index_a, (m.index_b, m.confidence));
        }

        // Build adjacency map from 1 -> 2
        let mut map_12: HashMap<usize, (usize, f32)> = HashMap::with_capacity(m12.len());
        for m in &m12.matches {
            map_12.insert(m.index_a, (m.index_b, m.confidence));
        }

        // Build lookup set for 0 -> 2 cycle closure
        let mut map_02: HashMap<(usize, usize), f32> = HashMap::with_capacity(m02.len());
        for m in &m02.matches {
            map_02.insert((m.index_a, m.index_b), m.confidence);
        }

        let mut verified_triplets = Vec::new();
        let mut cycle_candidates = 0usize;

        for (&i0, &(i1, conf_01)) in &map_01 {
            if let Some(&(i2, conf_12)) = map_12.get(&i1) {
                // Check cycle closure in 0-2 baseline
                if let Some(&conf_02) = map_02.get(&(i0, i2)) {
                    cycle_candidates += 1;
                    // Extract physical coordinates
                    let k0 = &frames[0].keypoints[i0];
                    let k1 = &frames[1].keypoints[i1];
                    let k2 = &frames[2].keypoints[i2];

                    if let Some((disp_01, disp_12, cascade_err)) = config
                        .verify_triplet_with_bounds(
                            k0.point,
                            k1.point,
                            k2.point,
                            Some(frames[0].image_size),
                        )
                    {
                        // Geometric mean of 3-way matching confidence
                        let conf = (conf_01 * conf_12 * conf_02).cbrt();
                        verified_triplets.push(FeatureTriplet {
                            index_0: i0,
                            index_1: i1,
                            index_2: i2,
                            confidence: conf,
                            disparity_01: disp_01,
                            disparity_12: disp_12,
                            cascade_error: cascade_err,
                        });
                    }
                }
            }
        }

        tracing::info!(
            cycle_candidates,
            verified_count = verified_triplets.len(),
            "Candidate triplets evaluated"
        );

        if tracing::enabled!(tracing::Level::DEBUG) && !verified_triplets.is_empty() {
            let n = verified_triplets.len() as f32;
            let sum_sq_cascade: f32 = verified_triplets
                .iter()
                .map(|t| t.cascade_error * t.cascade_error)
                .sum();
            let rmse_cascade_err = (sum_sq_cascade / n).sqrt();
            let max_cascade_err: f32 = verified_triplets
                .iter()
                .map(|t| t.cascade_error)
                .fold(0.0_f32, f32::max);
            let mean_disparity: f32 = verified_triplets
                .iter()
                .map(|t| f32::midpoint(t.disparity_01, t.disparity_12))
                .sum::<f32>()
                / n;
            let relative_extrinsic_loss_pct = if mean_disparity > 1e-4 {
                (rmse_cascade_err / mean_disparity) * 100.0
            } else {
                0.0
            };

            tracing::debug!(
                verified_triplets = verified_triplets.len(),
                rmse_cascade_error_px = rmse_cascade_err,
                max_cascade_error_px = max_cascade_err,
                mean_disparity_px = mean_disparity,
                relative_extrinsic_loss_pct,
                "Extrinsic parameter consistency & parallax-aware reprojection loss"
            );
        }

        // Sort triplets by confidence descending
        verified_triplets.sort_unstable_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(t) = tap {
            t.on_triplets_verified(&verified_triplets);
        }

        Ok(verified_triplets)
    }

    /// Computes pairwise match sets for all dyadic tree spans across $N \ge 2$ feature frames.
    ///
    /// # Errors
    /// Returns [`AlignmentError`] if frame slice count is less than 2 or pairwise matching fails.
    fn match_all_dyadic_pairs(
        &self,
        frames: &[FeatureFrame],
        bidirectional: bool,
    ) -> AlignmentResult<Vec<PairwiseMatchSet>> {
        let n = frames.len();
        if n < 2 {
            return Err(AlignmentError::InvalidInput(
                "Expected at least 2 feature frames for dyadic pair matching".to_string(),
            ));
        }

        let mut pairs = Vec::new();
        let mut stride = 1usize;
        while stride < n {
            for i in 0..n {
                let k = i + stride;
                if k < n && !pairs.contains(&(i, k)) {
                    pairs.push((i, k));
                }
            }
            stride *= 2;
        }
        if n > 2 && !pairs.contains(&(0, n - 1)) {
            pairs.push((0, n - 1));
        }

        let mut match_sets = Vec::with_capacity(pairs.len());
        for (i, k) in pairs {
            let ms = if bidirectional {
                self.match_pair_bidirectional(&frames[i], &frames[k])?
            } else {
                self.match_pair(&frames[i], &frames[k])?
            };
            match_sets.push(ms);
        }

        Ok(match_sets)
    }

    /// Extracts hierarchical extrinsics optimization report for $N \ge 3$ frames using dyadic tree reduction.
    ///
    /// # Errors
    /// Returns [`AlignmentError`] if matching fails or frame count is insufficient.
    fn extract_hierarchical_extrinsics(
        &self,
        frames: &[FeatureFrame],
        orientation: StripOrientation,
        config: &HierarchicalExtrinsicsConfig,
    ) -> AlignmentResult<HierarchicalOptimizationReport> {
        let match_sets = self.match_all_dyadic_pairs(frames, true)?;
        let tree =
            HierarchicalReductionTree::build_from_match_sets(frames, &match_sets, orientation);
        Ok(tree.optimize(config))
    }
}

/// Baseline descriptor matcher for `SuperPoint` features using nearest-neighbor dot products.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SuperPointDescriptorMatcher {
    /// Minimum cosine similarity threshold to consider a match valid.
    pub min_similarity: f32,
    /// Lowe's ratio test threshold (first nearest neighbor score vs second nearest neighbor score).
    pub ratio_threshold: f32,
}

impl Default for SuperPointDescriptorMatcher {
    fn default() -> Self {
        Self {
            min_similarity: 0.55,
            ratio_threshold: 0.88,
        }
    }
}

impl SuperPointDescriptorMatcher {
    /// Creates a new `SuperPointDescriptorMatcher` with specified thresholds.
    #[inline]
    #[must_use]
    pub const fn new(min_similarity: f32, ratio_threshold: f32) -> Self {
        Self {
            min_similarity,
            ratio_threshold,
        }
    }
}

impl FeatureMatcher for SuperPointDescriptorMatcher {
    fn match_pair(
        &self,
        frame_a: &FeatureFrame,
        frame_b: &FeatureFrame,
    ) -> AlignmentResult<PairwiseMatchSet> {
        let mut matches = Vec::new();

        let (mut valid_indices_b, mut desc_matrix_b) = (Vec::new(), Vec::new());
        for (i_b, kp_b) in frame_b.keypoints.iter().enumerate() {
            if let Some(ref desc_b) = kp_b.descriptor {
                valid_indices_b.push(i_b);
                desc_matrix_b.extend_from_slice(desc_b);
            }
        }

        if valid_indices_b.is_empty() {
            return Ok(PairwiseMatchSet::new(
                (frame_a.frame_index, frame_b.frame_index),
                matches,
                MatchDirection::Forward,
            ));
        }

        let num_b = valid_indices_b.len();
        let dim = 256;

        for (i_a, kp_a) in frame_a.keypoints.iter().enumerate() {
            let Some(ref desc_a) = kp_a.descriptor else {
                continue;
            };

            let mut best_sim = -1.0_f32;
            let mut second_best_sim = -1.0_f32;
            let mut best_idx = 0;

            for k in 0..num_b {
                let desc_b_slice = &desc_matrix_b[k * dim..(k + 1) * dim];
                let sim = compute_dot_product_256(desc_a, desc_b_slice);

                if sim > best_sim {
                    second_best_sim = best_sim;
                    best_sim = sim;
                    best_idx = valid_indices_b[k];
                } else if sim > second_best_sim {
                    second_best_sim = sim;
                }
            }

            // Lowe's ratio test check (distance ratio -> similarity ratio inverse)
            let passes_ratio = if second_best_sim > 0.0 {
                let dist_best = 1.0 - best_sim;
                let dist_second = 1.0 - second_best_sim;
                dist_best <= self.ratio_threshold * dist_second
            } else {
                true
            };

            if best_sim >= self.min_similarity && passes_ratio {
                let dist_ratio = if second_best_sim > 0.0 && (1.0 - second_best_sim).abs() > 1e-6 {
                    Some((1.0 - best_sim) / (1.0 - second_best_sim))
                } else {
                    None
                };
                let score = MatchScore::new(best_sim, best_sim, dist_ratio);
                matches.push(FeatureMatch::with_score(i_a, best_idx, score));
            }
        }

        Ok(PairwiseMatchSet::new(
            (frame_a.frame_index, frame_b.frame_index),
            matches,
            MatchDirection::Forward,
        ))
    }
}

/// Computes dot product of two 256-D slices vectorized with 8-lane SIMD unrolling.
#[inline]
#[must_use]
fn compute_dot_product_256(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), 256);
    debug_assert_eq!(b.len(), 256);

    let mut acc0 = 0.0_f32;
    let mut acc1 = 0.0_f32;
    let mut acc2 = 0.0_f32;
    let mut acc3 = 0.0_f32;

    for (chunk_a, chunk_b) in a.as_chunks::<16>().0.iter().zip(b.as_chunks::<16>().0) {
        acc0 = chunk_a[0].mul_add(
            chunk_b[0],
            chunk_a[1].mul_add(
                chunk_b[1],
                chunk_a[2].mul_add(chunk_b[2], chunk_a[3].mul_add(chunk_b[3], acc0)),
            ),
        );
        acc1 = chunk_a[4].mul_add(
            chunk_b[4],
            chunk_a[5].mul_add(
                chunk_b[5],
                chunk_a[6].mul_add(chunk_b[6], chunk_a[7].mul_add(chunk_b[7], acc1)),
            ),
        );
        acc2 = chunk_a[8].mul_add(
            chunk_b[8],
            chunk_a[9].mul_add(
                chunk_b[9],
                chunk_a[10].mul_add(chunk_b[10], chunk_a[11].mul_add(chunk_b[11], acc2)),
            ),
        );
        acc3 = chunk_a[12].mul_add(
            chunk_b[12],
            chunk_a[13].mul_add(
                chunk_b[13],
                chunk_a[14].mul_add(chunk_b[14], chunk_a[15].mul_add(chunk_b[15], acc3)),
            ),
        );
    }

    acc0 + acc1 + acc2 + acc3
}

/// Default no-op diagnostic tap.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoOpAlignmentDiagnosticTap;

impl AlignmentDiagnosticTap for NoOpAlignmentDiagnosticTap {}

/// Configuration parameters for `SuperPoint` feature extraction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuperPointConfig {
    /// Non-Maximum Suppression (NMS) radius in pixels (default: 4).
    pub nms_radius: usize,
    /// Minimum keypoint confidence score threshold (default: 0.005).
    pub keypoint_threshold: f32,
    /// Maximum number of keypoints to retain per sub-frame image (default: 4096).
    pub max_keypoints_per_image: usize,
    /// Distance from frame borders where keypoints are ignored (default: 4).
    pub remove_borders: usize,
    /// Target execution device (CPU or CUDA).
    pub device: BackendDevice,
    /// Whether to apply sub-pixel patch refinement on full-resolution ROI images (default: `true`).
    pub subpixel_refinement: bool,
    /// Half-window radius for sub-pixel patch refinement (default: 7, yielding a 15x15 pixel patch).
    pub subpixel_patch_radius: usize,
    /// Maximum iterations for sub-pixel refinement convergence (default: 5).
    pub subpixel_max_iterations: usize,
    /// Maximum allowed drift in pixels from the initial keypoint coordinate before falling back (default: 2.5).
    pub max_subpixel_drift_px: f32,
    /// Optional explicit path to `superpoint.onnx` model weights.
    pub model_path: Option<PathBuf>,
    /// Remote URL for downloading the model if missing from cache.
    pub model_url: String,
    /// Optional expected SHA-256 digest of the model binary.
    pub expected_sha256: Option<String>,
}

impl SuperPointConfig {
    /// Computes the total maximum keypoint budget scaled dynamically for a batch of `n_frames`.
    ///
    /// # Arguments
    /// * `n_frames` - The number of sub-frames in the processing batch.
    ///
    /// # Examples
    /// ```
    /// use reto_core::SuperPointConfig;
    ///
    /// let config = SuperPointConfig::default();
    /// assert_eq!(config.max_batch_keypoints(3), config.max_keypoints_per_image * 3);
    /// ```
    #[inline]
    #[must_use]
    pub const fn max_batch_keypoints(&self, n_frames: usize) -> usize {
        self.max_keypoints_per_image.saturating_mul(n_frames)
    }
}

impl Default for SuperPointConfig {
    fn default() -> Self {
        Self {
            nms_radius: 4,
            keypoint_threshold: 0.005,
            max_keypoints_per_image: 4096,
            remove_borders: 4,
            device: BackendDevice::default_available(),
            subpixel_refinement: true,
            subpixel_patch_radius: DEFAULT_SUBPIXEL_PATCH_RADIUS,
            subpixel_max_iterations: DEFAULT_SUBPIXEL_MAX_ITERATIONS,
            max_subpixel_drift_px: DEFAULT_SUBPIXEL_MAX_DRIFT_PX,
            model_path: None,
            model_url: DEFAULT_SUPERPOINT_MODEL_URL.to_string(),
            expected_sha256: Some(DEFAULT_SUPERPOINT_MODEL_SHA256.to_string()),
        }
    }
}

/// Core interface for 2D keypoint and feature detection on sub-frames.
pub trait PointDetector: Send + Sync {
    /// Detects keypoints and descriptors from a normalized single-channel `[0.0, 1.0]` luma plane.
    ///
    /// # Arguments
    /// * `luma` - Flat row-major floating point grayscale buffer.
    /// * `size` - Image dimensions in pixels.
    ///
    /// # Errors
    /// Returns [`AlignmentError`] on model inference or invalid buffer dimensions.
    fn detect_luma(&self, luma: &[f32], size: Size2D<u32>) -> AlignmentResult<Vec<KeyPoint>>;

    /// Detects keypoints on a sub-frame ROI directly borrowed from an image view.
    ///
    /// # Arguments
    /// * `image` - Source image view.
    /// * `roi` - Region of interest descriptor.
    /// * `tap` - Optional diagnostic tap.
    ///
    /// # Errors
    /// Returns [`AlignmentError`] if detection or image slicing fails.
    fn detect_roi<I: GenericImageView>(
        &self,
        image: &I,
        roi: &FrameRoi,
        tap: Option<&dyn AlignmentDiagnosticTap>,
    ) -> AlignmentResult<FeatureFrame>;

    /// Detects keypoints directly from a pre-scaled luma image (`ScaledLumaImage`).
    ///
    /// Avoids repeated RGB-to-Luma conversions by reusing the downscaled luma buffer.
    ///
    /// # Arguments
    /// * `luma` - Scaled luma image.
    /// * `roi` - Region of interest descriptor.
    /// * `tap` - Optional diagnostic tap.
    ///
    /// # Errors
    /// Returns [`AlignmentError`] if detection or sub-frame slicing fails.
    fn detect_luma_roi(
        &self,
        luma: &ScaledLumaImage,
        roi: &FrameRoi,
        tap: Option<&dyn AlignmentDiagnosticTap>,
    ) -> AlignmentResult<FeatureFrame>;

    /// Backward-compatible alias for [`PointDetector::detect_luma_roi`].
    ///
    /// # Errors
    /// Returns [`AlignmentError`] if detection or sub-frame slicing fails.
    fn detect_strip_roi(
        &self,
        luma: &ScaledLumaImage,
        roi: &FrameRoi,
        tap: Option<&dyn AlignmentDiagnosticTap>,
    ) -> AlignmentResult<FeatureFrame> {
        self.detect_luma_roi(luma, roi, tap)
    }

    /// Detects keypoints across all frames in an image layout.
    ///
    /// # Arguments
    /// * `luma` - Scaled luma image.
    /// * `rois` - Slice of detected sub-frame regions.
    /// * `tap` - Optional diagnostic tap.
    ///
    /// # Errors
    /// Returns [`AlignmentError`] if detection on any frame fails.
    fn detect_luma_all(
        &self,
        luma: &ScaledLumaImage,
        rois: &[FrameRoi],
        tap: Option<&dyn AlignmentDiagnosticTap>,
    ) -> AlignmentResult<Vec<FeatureFrame>> {
        rois.iter()
            .map(|roi| self.detect_luma_roi(luma, roi, tap))
            .collect()
    }

    /// Backward-compatible alias for [`PointDetector::detect_luma_all`].
    ///
    /// # Errors
    /// Returns [`AlignmentError`] if detection on any frame fails.
    fn detect_strip_all(
        &self,
        luma: &ScaledLumaImage,
        rois: &[FrameRoi],
        tap: Option<&dyn AlignmentDiagnosticTap>,
    ) -> AlignmentResult<Vec<FeatureFrame>> {
        self.detect_luma_all(luma, rois, tap)
    }
}

/// Backward-compatible alias for [`PointDetector`].
pub type FeatureExtractor = dyn PointDetector;

/// Execution parameters and geometric context for `SuperPoint` tensor decoding.
#[derive(Clone, Copy)]
pub(crate) struct SuperPointDecodeContext<'a> {
    /// Active orientation delegator for canvas coordinate mapping.
    pub delegator: &'a dyn OrientationDelegator,
    /// Dimensions (width, height) of the scaled sub-frame image.
    pub scaled_size: Size2D<u32>,
    /// Scale factor relative to the original unscaled sub-frame pixels.
    pub scale: f32,
    /// Distance from frame borders in pixels to ignore keypoints.
    pub border: f32,
    /// Minimum confidence threshold for keypoint detection.
    pub score_threshold: f32,
    /// Maximum number of keypoints to retain per sub-frame.
    pub max_keypoints: usize,
}

#[derive(Clone, Copy)]
struct KeypointSampleSpec {
    ox: f32,
    oy: f32,
    score: f32,
    base00: usize,
    base01: usize,
    base10: usize,
    base11: usize,
    w00: f32,
    w01: f32,
    w10: f32,
    w11: f32,
}

/// `SuperPoint` feature detector powered by the ONNX Runtime (`ort`) inference engine.
#[derive(Clone)]
pub struct SuperPointDetector {
    config: SuperPointConfig,
    model: ModelSlot,
}

impl std::fmt::Debug for SuperPointDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SuperPointDetector")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

/// Backward-compatible alias for [`SuperPointDetector`].
pub type SuperPointExtractor = SuperPointDetector;

impl SuperPointDetector {
    /// Creates a new `SuperPointDetector` with the given configuration.
    #[must_use]
    pub fn new(config: SuperPointConfig) -> Self {
        Self {
            config,
            model: Arc::new(Mutex::new(None)),
        }
    }

    /// Returns a reference to the active configuration.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &SuperPointConfig {
        &self.config
    }

    /// Ensures the neural network inference model is loaded and backend is ready.
    ///
    /// # Errors
    /// Returns [`AlignmentError::ModelLoad`] if downloading or compiling the model fails.
    pub fn ensure_backend_ready(&self) -> AlignmentResult<()> {
        let mut model_guard = self
            .model
            .lock()
            .map_err(|e| AlignmentError::Inference(format!("Model lock poisoned: {e}")))?;

        if model_guard.is_none() {
            let shared = get_or_load_superpoint_model(&self.config)?;
            *model_guard = Some(shared);
        }
        drop(model_guard);
        Ok(())
    }

    /// Acquires the shared model instance, lazily initializing if unpopulated.
    fn get_or_init_model(&self) -> Option<SharedModelHandle> {
        let mut guard = self.model.lock().ok()?;
        if let Some(ref model) = *guard {
            Some(Arc::clone(model))
        } else if let Ok(model) = get_or_load_superpoint_model(&self.config) {
            *guard = Some(Arc::clone(&model));
            drop(guard);
            Some(model)
        } else {
            None
        }
    }

    /// Performs Non-Maximum Suppression (NMS) on a 2D score heatmap.
    ///
    /// # Arguments
    /// * `scores` - Row-major score grid of size `width x height`.
    /// * `size` - Grid dimensions.
    /// * `radius` - Neighborhood suppression radius in pixels.
    /// * `threshold` - Minimum confidence threshold.
    /// * `border` - Distance from image boundaries where points are suppressed.
    #[must_use]
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        clippy::many_single_char_names,
        clippy::similar_names
    )]
    pub fn non_maximum_suppression(
        scores: &[f32],
        size: Size2D<u32>,
        radius: usize,
        threshold: f32,
        border: usize,
    ) -> Vec<KeypointScore> {
        let w = size.width as usize;
        let h = size.height as usize;
        let r = radius as isize;
        let b = border as isize;

        if w <= border * 2 || h <= border * 2 || scores.len() < w * h {
            return Vec::new();
        }

        let mut keypoints = Vec::new();

        for y in b..(h as isize - b) {
            let y_usize = y as usize;
            for x in b..(w as isize - b) {
                let x_usize = x as usize;
                let center_score = scores[y_usize * w + x_usize];
                if center_score < threshold {
                    continue;
                }

                let mut is_max = true;
                let min_ny = (y - r).max(0);
                let max_ny = (y + r).min(h as isize - 1);
                let min_nx = (x - r).max(0);
                let max_nx = (x + r).min(w as isize - 1);

                'window: for ny in min_ny..=max_ny {
                    let ny_idx = ny as usize * w;
                    for nx in min_nx..=max_nx {
                        if nx == x && ny == y {
                            continue;
                        }
                        if scores[ny_idx + nx as usize] >= center_score {
                            is_max = false;
                            break 'window;
                        }
                    }
                }

                if is_max {
                    keypoints.push((Point2D::new(x as f32, y as f32), center_score));
                }
            }
        }

        keypoints
    }

    /// Fallback corner response calculation for offline test environments.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::suboptimal_flops,
        clippy::suspicious_operation_groupings
    )]
    fn compute_luma_corner_response(luma: &[f32], size: Size2D<u32>) -> Vec<f32> {
        let w = size.width as usize;
        let h = size.height as usize;
        let mut response = vec![0.0_f32; w * h];

        if w < 5 || h < 5 || luma.len() < w * h {
            return response;
        }

        let mut dx2 = vec![0.0_f32; w * h];
        let mut dy2 = vec![0.0_f32; w * h];
        let mut dxy = vec![0.0_f32; w * h];

        for y in 1..(h - 1) {
            let y_stride = y * w;
            let prev_y = (y - 1) * w;
            let next_y = (y + 1) * w;
            for x in 1..(w - 1) {
                let dx = 0.5 * (luma[y_stride + x + 1] - luma[y_stride + x - 1]);
                let dy = 0.5 * (luma[next_y + x] - luma[prev_y + x]);
                let idx = y_stride + x;
                dx2[idx] = dx * dx;
                dy2[idx] = dy * dy;
                dxy[idx] = dx * dy;
            }
        }

        // Apply 3x3 box filtering on structure tensor components
        for y in 2..(h - 2) {
            let y_stride = y * w;
            for x in 2..(w - 2) {
                let mut s_dx2 = 0.0_f32;
                let mut s_dy2 = 0.0_f32;
                let mut s_dxy = 0.0_f32;

                for dy in -1..=1 {
                    let r_idx = ((y as isize + dy) as usize) * w;
                    for dx in -1..=1 {
                        let c_idx = r_idx + ((x as isize + dx) as usize);
                        s_dx2 += dx2[c_idx];
                        s_dy2 += dy2[c_idx];
                        s_dxy += dxy[c_idx];
                    }
                }

                // Harris corner score: det(M) - 0.04 * trace(M)^2
                let det = s_dx2 * s_dy2 - s_dxy * s_dxy;
                let trace = s_dx2 + s_dy2;
                let harris = det - 0.04 * trace * trace;
                if harris > 0.0 {
                    response[y_stride + x] = harris.min(1.0);
                }
            }
        }

        response
    }

    /// Decodes raw `SuperPoint` ONNX tensor outputs using batch candidate filtering,
    /// linear-time $\mathcal{O}(N)$ top-K selection, and contiguous SIMD descriptor interpolation.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::too_many_lines,
        clippy::suboptimal_flops,
        clippy::similar_names,
        clippy::needless_pass_by_value
    )]
    fn decode_superpoint_vectorized(
        kpts: TensorView<'_>,
        scs: TensorView<'_>,
        descs: TensorView<'_>,
        ctx: &SuperPointDecodeContext<'_>,
    ) -> AlignmentResult<Vec<KeyPoint>> {
        let (frame_kpts, frame_scs, frame_descs) = if kpts.ndim() == 3 {
            (
                kpts.index_axis(Axis(0), 0),
                if scs.ndim() == 2 {
                    scs.index_axis(Axis(0), 0)
                } else {
                    scs.view()
                },
                if descs.ndim() == 4 {
                    descs.index_axis(Axis(0), 0)
                } else {
                    descs.view()
                },
            )
        } else {
            (kpts.view(), scs.view(), descs.view())
        };

        if frame_kpts.ndim() < 2 || frame_descs.ndim() < 3 {
            return Err(AlignmentError::TensorLayout(
                "Invalid SuperPoint output tensor dimensions".to_string(),
            ));
        }

        let num_kpts = frame_kpts.shape()[0];
        if num_kpts == 0 {
            return Ok(Vec::new());
        }

        let min_rx = ctx.border;
        let min_ry = ctx.border;
        let max_rx = ctx.scaled_size.width as f32 - ctx.border;
        let max_ry = ctx.scaled_size.height as f32 - ctx.border;

        // 1. Batch candidate filtering: collect lightweight tuples (index, score, rx, ry)
        let mut candidates: Vec<(usize, f32, f32, f32)> =
            Vec::with_capacity(num_kpts.min(ctx.max_keypoints.saturating_mul(2)));

        for i in 0..num_kpts {
            let s = if frame_scs.ndim() >= 1 {
                frame_scs[[i]]
            } else {
                1.0
            };
            if s < ctx.score_threshold {
                continue;
            }

            let kx = frame_kpts[[i, 0]];
            let ky = frame_kpts[[i, 1]];
            let (rx, ry) = ctx.delegator.unmap_canvas_coords(kx, ky);

            if rx >= min_rx && ry >= min_ry && rx < max_rx && ry < max_ry {
                candidates.push((i, s, rx, ry));
            }
        }

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // 2. O(N) Linear-Time Top-K Selection
        let k = candidates.len().min(ctx.max_keypoints);
        if candidates.len() > k {
            candidates.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(k);
        }
        candidates
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 3. Batch Coordinate & Grid Sampling Calculation (Single contiguous allocation)
        let inv_scale = 1.0 / ctx.scale;
        let desc_channels = frame_descs.shape()[0];
        let desc_h = frame_descs.shape()[1];
        let desc_w = frame_descs.shape()[2];
        let hw = desc_h * desc_w;
        let max_gx = (desc_w.saturating_sub(1)) as f32;
        let max_gy = (desc_h.saturating_sub(1)) as f32;
        let max_gx_idx = desc_w.saturating_sub(1);
        let max_gy_idx = desc_h.saturating_sub(1);

        let mut sample_specs = Vec::with_capacity(k);

        for &(i, s, rx, ry) in &candidates {
            let kx = frame_kpts[[i, 0]];
            let ky = frame_kpts[[i, 1]];

            let ox = rx * inv_scale;
            let oy = ry * inv_scale;

            let gx = (kx * 0.125).clamp(0.0, max_gx);
            let gy = (ky * 0.125).clamp(0.0, max_gy);
            let g_x0 = gx.floor() as usize;
            let g_y0 = gy.floor() as usize;
            let g_x1 = g_x0.saturating_add(1).min(max_gx_idx);
            let g_y1 = g_y0.saturating_add(1).min(max_gy_idx);
            let wx = gx - g_x0 as f32;
            let wy = gy - g_y0 as f32;

            let base00 = g_y0 * desc_w + g_x0;
            let base01 = g_y0 * desc_w + g_x1;
            let base10 = g_y1 * desc_w + g_x0;
            let base11 = g_y1 * desc_w + g_x1;

            let w00 = (1.0 - wx) * (1.0 - wy);
            let w01 = wx * (1.0 - wy);
            let w10 = (1.0 - wx) * wy;
            let w11 = wx * wy;

            sample_specs.push(KeypointSampleSpec {
                ox,
                oy,
                score: s,
                base00,
                base01,
                base10,
                base11,
                w00,
                w01,
                w10,
                w11,
            });
        }

        // 4. SIMD Contiguous Descriptor Blending & Normalization
        let mut keypoints = Vec::with_capacity(k);
        let flat_descs = frame_descs.as_slice();

        for spec in sample_specs {
            let mut d_vec = Vec::with_capacity(desc_channels);
            let mut sum_sq = 0.0_f32;

            if let Some(descs_slice) = flat_descs {
                for c in 0..desc_channels {
                    let c_offset = c * hw;
                    let v00 = descs_slice[c_offset + spec.base00];
                    let v01 = descs_slice[c_offset + spec.base01];
                    let v10 = descs_slice[c_offset + spec.base10];
                    let v11 = descs_slice[c_offset + spec.base11];

                    let val = v00 * spec.w00 + v01 * spec.w01 + v10 * spec.w10 + v11 * spec.w11;
                    d_vec.push(val);
                    sum_sq += val * val;
                }
            } else {
                for c in 0..desc_channels {
                    let v00 = frame_descs[[c, spec.base00 / desc_w, spec.base00 % desc_w]];
                    let v01 = frame_descs[[c, spec.base01 / desc_w, spec.base01 % desc_w]];
                    let v10 = frame_descs[[c, spec.base10 / desc_w, spec.base10 % desc_w]];
                    let v11 = frame_descs[[c, spec.base11 / desc_w, spec.base11 % desc_w]];

                    let val = v00 * spec.w00 + v01 * spec.w01 + v10 * spec.w10 + v11 * spec.w11;
                    d_vec.push(val);
                    sum_sq += val * val;
                }
            }

            let inv_norm = 1.0 / sum_sq.sqrt().max(1e-7);
            for val in &mut d_vec {
                *val *= inv_norm;
            }

            keypoints.push(KeyPoint::new(
                Point2D::new(spec.ox, spec.oy),
                spec.score,
                Some(d_vec),
            ));
        }

        Ok(keypoints)
    }
}

/// Default patch radius in pixels for sub-pixel keypoint refinement (7 pixels -> 15x15 pixel patch).
pub const DEFAULT_SUBPIXEL_PATCH_RADIUS: usize = 7;

/// Default maximum iterations for sub-pixel keypoint refinement convergence (5 iterations).
pub const DEFAULT_SUBPIXEL_MAX_ITERATIONS: usize = 5;

/// Default convergence displacement epsilon in pixels for sub-pixel keypoint refinement (0.01 px).
pub const DEFAULT_SUBPIXEL_EPSILON_PX: f32 = 0.01;

/// Default maximum allowed sub-pixel drift in pixels from initial position before fallback (2.5 px).
pub const DEFAULT_SUBPIXEL_MAX_DRIFT_PX: f32 = 2.5;

/// Refines keypoint coordinates to sub-pixel accuracy using localized gradient-based photometric patch tracking (Förstner / Lucas-Kanade corner operator).
///
/// Delegates to [`refine_keypoints_subpixel_with_drift`] with [`DEFAULT_SUBPIXEL_MAX_DRIFT_PX`].
#[inline]
pub fn refine_keypoints_subpixel(
    image: &image::GrayImage,
    keypoints: &mut [KeyPoint],
    radius: usize,
    max_iterations: usize,
    epsilon: f32,
) {
    refine_keypoints_subpixel_with_drift(
        image,
        keypoints,
        radius,
        max_iterations,
        epsilon,
        DEFAULT_SUBPIXEL_MAX_DRIFT_PX,
    );
}

/// Refines keypoint coordinates to sub-pixel accuracy with a configurable maximum drift tolerance.
///
/// For each keypoint detected on coarse or downscaled grids, this samples a full-resolution patch
/// ($W \times W$ where $W = 2 \times \text{radius} + 1$) around the keypoint, evaluates spatial image gradients
/// $(I_x, I_y)$, and iteratively solves for the optimal sub-pixel displacement $\mathbf{\delta}$ that minimizes
/// photometric gradient orthogonal deviation.
///
/// # Arguments
/// * `image` - Grayscale image buffer of the region of interest at full resolution.
/// * `keypoints` - Mutable slice of detected keypoints to refine in-place.
/// * `radius` - Patch half-window radius in pixels (e.g. 7 for a 15x15 window).
/// * `max_iterations` - Maximum refinement iterations (e.g. 5).
/// * `epsilon` - Convergence displacement threshold in pixels (e.g. 0.01).
/// * `max_drift_px` - Maximum allowed drift in pixels from the initial coordinate (e.g. 2.5).
///
/// # Examples
/// ```
/// use reto_core::{refine_keypoints_subpixel_with_drift, KeyPoint, Point2D};
/// use image::GrayImage;
///
/// let mut img = GrayImage::new(32, 32);
/// for y in 0..16 {
///     for x in 0..16 {
///         img.put_pixel(x, y, image::Luma([255]));
///     }
/// }
///
/// let mut kps = vec![KeyPoint::new(Point2D::new(15.2, 15.3), 0.95, None)];
/// refine_keypoints_subpixel_with_drift(&img, &mut kps, 5, 5, 0.01, 2.0);
/// assert!(kps[0].point.x >= 14.0 && kps[0].point.x <= 16.5);
/// ```
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::suboptimal_flops,
    clippy::similar_names,
    clippy::suspicious_operation_groupings,
    clippy::too_many_lines
)]
pub fn refine_keypoints_subpixel_with_drift(
    image: &image::GrayImage,
    keypoints: &mut [KeyPoint],
    radius: usize,
    max_iterations: usize,
    epsilon: f32,
    max_drift_px: f32,
) {
    let (width, height) = image.dimensions();
    let r = radius as isize;
    let min_dim = (2 * radius + 3) as u32;

    if width < min_dim || height < min_dim || keypoints.is_empty() {
        return;
    }

    let raw = image.as_raw();
    let w = width as usize;
    let sigma = (radius as f32 / 2.0).max(1.0);
    let two_sigma_sq = 2.0 * sigma * sigma;
    let inv_two_sigma_sq = 1.0 / two_sigma_sq;
    let eps_sq = epsilon * epsilon;
    let max_step_sq = 1.5_f32 * 1.5_f32;
    let max_drift_sq = max_drift_px * max_drift_px;

    let min_bound = (radius + 1) as f32;
    let max_bound_x = (width.saturating_sub(radius as u32 + 2)) as f32;
    let max_bound_y = (height.saturating_sub(radius as u32 + 2)) as f32;

    // Parallel keypoint batch refinement via Rayon
    keypoints.par_iter_mut().for_each(|kp| {
        let x_init = kp.point.x;
        let y_init = kp.point.y;

        // Check if initial keypoint is within safe margins
        if x_init < min_bound || x_init > max_bound_x || y_init < min_bound || y_init > max_bound_y
        {
            kp.subpixel_status = SubpixelStatus::BoundarySkipped;
            return;
        }

        let mut x_curr = x_init;
        let mut y_curr = y_init;
        let mut iters_done = 0;
        let mut status = SubpixelStatus::NeuralOnly;

        for iter in 0..max_iterations {
            iters_done = iter + 1;
            let mut a = 0.0_f32; // sum w * Ix^2
            let mut b = 0.0_f32; // sum w * Ix * Iy
            let mut c = 0.0_f32; // sum w * Iy^2
            let mut vx = 0.0_f32; // sum w * (Ix^2 * delta_x + Ix * Iy * delta_y)
            let mut vy = 0.0_f32; // sum w * (Ix * Iy * delta_x + Iy^2 * delta_y)

            let cx_round = x_curr.round() as isize;
            let cy_round = y_curr.round() as isize;

            if cx_round - r - 1 < 0
                || cx_round + r + 1 >= width as isize
                || cy_round - r - 1 < 0
                || cy_round + r + 1 >= height as isize
            {
                status = SubpixelStatus::BoundarySkipped;
                break;
            }

            for dy in -r..=r {
                let py = (cy_round + dy) as usize;
                let delta_y = ((cy_round + dy) as f32) - y_curr;
                let weight_y = (-delta_y * delta_y * inv_two_sigma_sq).exp();

                let row = &raw[py * w..(py + 1) * w];
                let prev_row = &raw[(py - 1) * w..py * w];
                let next_row = &raw[(py + 1) * w..(py + 2) * w];

                for dx in -r..=r {
                    let px = (cx_round + dx) as usize;
                    let delta_x = ((cx_round + dx) as f32) - x_curr;
                    let weight_x = (-delta_x * delta_x * inv_two_sigma_sq).exp();
                    let weight = weight_y * weight_x;

                    // Branchless contiguous central gradient evaluation
                    let ix =
                        0.5 * (f32::from(row[px + 1]) - f32::from(row[px - 1])) * (1.0 / 255.0);
                    let iy =
                        0.5 * (f32::from(next_row[px]) - f32::from(prev_row[px])) * (1.0 / 255.0);

                    let ix2 = ix * ix;
                    let iy2 = iy * iy;
                    let ixy = ix * iy;

                    a += weight * ix2;
                    b += weight * ixy;
                    c += weight * iy2;

                    vx += weight * (ix2 * delta_x + ixy * delta_y);
                    vy += weight * (ixy * delta_x + iy2 * delta_y);
                }
            }

            let det = a * c - b * b;
            let tr = a + c;

            // Structure tensor conditioning check (ensure non-degenerate 2D corner)
            if det < 1e-7 || tr < 1e-5 {
                status = SubpixelStatus::PoorConditioning {
                    min_eigenvalue: 0.0,
                    cond_ratio: 0.0,
                };
                break;
            }

            let trace_sq_minus_4det = (tr * tr - 4.0 * det).max(0.0);
            let sqrt_term = trace_sq_minus_4det.sqrt();
            let lambda_min = 0.5 * (tr - sqrt_term);
            let lambda_max = 0.5 * (tr + sqrt_term);
            let cond_ratio = if lambda_max > 1e-6 {
                lambda_min / lambda_max
            } else {
                0.0
            };

            if lambda_min < 1e-5 || cond_ratio < 0.05 {
                status = SubpixelStatus::PoorConditioning {
                    min_eigenvalue: lambda_min,
                    cond_ratio,
                };
                break;
            }

            // Solve 2x2 linear system G * [step_x; step_y] = [vx; vy]
            let inv_det = 1.0 / det;
            let step_x = (c * vx - b * vy) * inv_det;
            let step_y = (-b * vx + a * vy) * inv_det;

            let step_norm_sq = step_x * step_x + step_y * step_y;
            if step_norm_sq > max_step_sq || step_x.is_nan() || step_y.is_nan() {
                break;
            }

            x_curr += step_x;
            y_curr += step_y;

            let drift_sq =
                (x_curr - x_init) * (x_curr - x_init) + (y_curr - y_init) * (y_curr - y_init);
            if drift_sq > max_drift_sq {
                // Revert to initial if displacement wandered too far
                let drift_px = drift_sq.sqrt();
                x_curr = x_init;
                y_curr = y_init;
                status = SubpixelStatus::DriftRejected { drift_px };
                break;
            }

            if step_norm_sq < eps_sq {
                status = SubpixelStatus::Refined {
                    delta: (x_curr - x_init, y_curr - y_init),
                    iterations: iters_done,
                };
                break;
            }
        }

        if status == SubpixelStatus::NeuralOnly {
            let dx = x_curr - x_init;
            let dy = y_curr - y_init;
            if dx * dx + dy * dy > 1e-6 {
                status = SubpixelStatus::Refined {
                    delta: (dx, dy),
                    iterations: iters_done,
                };
            }
        }

        kp.point = Point2D::new(x_curr, y_curr);
        kp.subpixel_status = status;
    });
}

/// Target maximum resolution for the longest dimension during keypoint feature extraction (720 pixels).
pub const POINT_DETECTION_MAX_LONGEST_EDGE: u32 = 720;

/// Standard model inference canvas height (720 pixels).
pub const MODEL_CANVAS_HEIGHT: u32 = 720;

/// Standard model inference canvas width (1280 pixels).
pub const MODEL_CANVAS_WIDTH: u32 = 1280;

impl PointDetector for SuperPointDetector {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::many_single_char_names,
        clippy::suboptimal_flops,
        clippy::too_many_lines,
        clippy::needless_range_loop
    )]
    fn detect_luma(&self, luma: &[f32], size: Size2D<u32>) -> AlignmentResult<Vec<KeyPoint>> {
        if size.width == 0 || size.height == 0 {
            return Err(AlignmentError::InvalidInput(
                "Image dimensions must be non-zero".to_string(),
            ));
        }

        let expected_len = (size.width * size.height) as usize;
        if luma.len() != expected_len {
            return Err(AlignmentError::TensorLayout(format!(
                "Luma slice length {} does not match dimensions {}x{} ({})",
                luma.len(),
                size.width,
                size.height,
                expected_len
            )));
        }

        // 1. Longest edge scaling: clamp longest side to POINT_DETECTION_MAX_LONGEST_EDGE (720)
        let longest_side = size.width.max(size.height);
        let (scale, scaled_w, scaled_h) = if longest_side > POINT_DETECTION_MAX_LONGEST_EDGE {
            let s = POINT_DETECTION_MAX_LONGEST_EDGE as f32 / longest_side as f32;
            let sw = (size.width as f32 * s).round() as u32;
            let sh = (size.height as f32 * s).round() as u32;
            (s, sw, sh)
        } else {
            (1.0, size.width, size.height)
        };

        // Convert slice to GrayImage for external imageops::resize (YAGNI standard API)
        let raw_bytes: Vec<u8> = luma
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        let gray_orig = image::GrayImage::from_raw(size.width, size.height, raw_bytes)
            .unwrap_or_else(|| image::GrayImage::new(size.width, size.height));

        let scaled_img = if scaled_w == size.width && scaled_h == size.height {
            gray_orig.clone()
        } else {
            image::imageops::resize(
                &gray_orig,
                scaled_w,
                scaled_h,
                image::imageops::FilterType::Triangle,
            )
        };

        let border = self.config.remove_borders as f32;
        let orientation = StripOrientation::from_size(size).unwrap_or(StripOrientation::Horizontal);
        let delegator = orientation.delegator();

        // 2. Attempt neural model inference via tract-onnx with graceful fallback if model is unavailable
        let model_shared = self.get_or_init_model();

        if let Some(shared_model) = model_shared {
            let mut canvas_luma =
                vec![0.0_f32; (MODEL_CANVAS_HEIGHT * MODEL_CANVAS_WIDTH) as usize];
            delegator.fill_inference_canvas(
                &mut canvas_luma,
                MODEL_CANVAS_WIDTH,
                &scaled_img,
                scaled_w,
                scaled_h,
            );

            let ctx = SuperPointDecodeContext {
                delegator,
                scaled_size: Size2D::new(scaled_w, scaled_h),
                scale,
                border,
                score_threshold: self.config.keypoint_threshold,
                max_keypoints: self.config.max_keypoints_per_image,
            };

            let plan_res = (|| -> AlignmentResult<Vec<KeyPoint>> {
                let tensor = Array4::from_shape_vec(
                    (
                        1,
                        1,
                        MODEL_CANVAS_HEIGHT as usize,
                        MODEL_CANVAS_WIDTH as usize,
                    ),
                    canvas_luma,
                )
                .map_err(|e| {
                    AlignmentError::TensorLayout(format!("Failed to build input tensor: {e}"))
                })?;

                shared_model.infer_and_decode(tensor, &ctx)
            })();

            match plan_res {
                Ok(mut kps) => {
                    if self.config.subpixel_refinement {
                        refine_keypoints_subpixel_with_drift(
                            &gray_orig,
                            &mut kps,
                            self.config.subpixel_patch_radius,
                            self.config.subpixel_max_iterations,
                            DEFAULT_SUBPIXEL_EPSILON_PX,
                            self.config.max_subpixel_drift_px,
                        );
                    }
                    return Ok(kps);
                }
                Err(e) => {
                    tracing::error!(error = %e, "ONNX SuperPoint model execution failed; falling back");
                }
            }
        }

        // Baseline fallback when neural model is offline
        let float_luma: Vec<f32> = scaled_img
            .as_raw()
            .iter()
            .map(|&b| f32::from(b) / 255.0)
            .collect();

        let corner_scores =
            Self::compute_luma_corner_response(&float_luma, Size2D::new(scaled_w, scaled_h));

        let nms_points = Self::non_maximum_suppression(
            &corner_scores,
            Size2D::new(scaled_w, scaled_h),
            self.config.nms_radius,
            self.config.keypoint_threshold,
            self.config.remove_borders,
        );

        let inv_scale = 1.0 / scale;
        let mut keypoints: Vec<KeyPoint> = nms_points
            .into_iter()
            .take(self.config.max_keypoints_per_image)
            .map(|(pt, score)| {
                KeyPoint::new(
                    Point2D::new(pt.x * inv_scale, pt.y * inv_scale),
                    score,
                    None,
                )
            })
            .collect();

        if self.config.subpixel_refinement {
            refine_keypoints_subpixel_with_drift(
                &gray_orig,
                &mut keypoints,
                self.config.subpixel_patch_radius,
                self.config.subpixel_max_iterations,
                DEFAULT_SUBPIXEL_EPSILON_PX,
                self.config.max_subpixel_drift_px,
            );
        }

        Ok(keypoints)
    }

    #[allow(
        clippy::cast_precision_loss,
        clippy::many_single_char_names,
        clippy::suboptimal_flops
    )]
    fn detect_roi<I: GenericImageView>(
        &self,
        image: &I,
        roi: &FrameRoi,
        tap: Option<&dyn AlignmentDiagnosticTap>,
    ) -> AlignmentResult<FeatureFrame> {
        let (w, h) = image.dimensions();
        let pixel_rect = roi.bounds.to_pixel_rect(Size2D::new(w, h));

        if pixel_rect.width == 0 || pixel_rect.height == 0 {
            return Err(AlignmentError::InvalidInput(
                "ROI pixel dimensions must be non-zero".to_string(),
            ));
        }

        let mut luma_buffer = Vec::with_capacity((pixel_rect.width * pixel_rect.height) as usize);

        for y in 0..pixel_rect.height {
            let src_y = pixel_rect.y + y;
            for x in 0..pixel_rect.width {
                let src_x = pixel_rect.x + x;
                let pixel = image.get_pixel(src_x, src_y);
                let channels = pixel.to_rgb();
                let r = channels[0].to_f32().unwrap_or(0.0);
                let g = channels[1].to_f32().unwrap_or(0.0);
                let b = channels[2].to_f32().unwrap_or(0.0);
                // Bt709 luma normalized to [0.0, 1.0]
                let y_val = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;
                luma_buffer.push(y_val);
            }
        }

        let keypoints = self.detect_luma(&luma_buffer, pixel_rect.size())?;
        let frame = FeatureFrame::new(roi.index, roi.bounds, pixel_rect.size(), keypoints);

        if let Some(t) = tap {
            t.on_features_extracted(roi.index, &frame);
        }

        Ok(frame)
    }

    #[allow(clippy::cast_precision_loss)]
    fn detect_luma_roi(
        &self,
        luma: &ScaledLumaImage,
        roi: &FrameRoi,
        tap: Option<&dyn AlignmentDiagnosticTap>,
    ) -> AlignmentResult<FeatureFrame> {
        let mut frames = self.detect_luma_all(luma, std::slice::from_ref(roi), tap)?;
        frames
            .pop()
            .ok_or_else(|| AlignmentError::Inference("Failed to extract feature frame".to_string()))
    }

    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::many_single_char_names,
        clippy::suboptimal_flops,
        clippy::too_many_lines
    )]
    #[tracing::instrument(skip(self, luma, rois, tap), level = "debug")]
    /// Detects keypoints and descriptors for multiple frame ROIs within a single luma image.
    ///
    /// Applies high-resolution sub-pixel patch refinement on the original full-resolution scan patches
    /// when `subpixel_refinement` is enabled to eliminate discretization errors from downscaled grid inference.
    fn detect_luma_all(
        &self,
        luma: &ScaledLumaImage,
        rois: &[FrameRoi],
        tap: Option<&dyn AlignmentDiagnosticTap>,
    ) -> AlignmentResult<Vec<FeatureFrame>> {
        struct ScaledSpec {
            roi: FrameRoi,
            orig_size: Size2D<u32>,
            scale: f32,
            scaled_w: u32,
            scaled_h: u32,
            scaled_img: Option<image::GrayImage>,
            orig_img: image::GrayImage,
        }

        if rois.is_empty() {
            return Ok(Vec::new());
        }

        let full_gray = luma.to_gray_image();

        let mut specs = Vec::with_capacity(rois.len());
        for roi in rois {
            let pixel_rect = roi.bounds.to_pixel_rect(luma.size());
            if pixel_rect.width == 0 || pixel_rect.height == 0 {
                return Err(AlignmentError::InvalidInput(
                    "ROI pixel dimensions must be non-zero".to_string(),
                ));
            }

            let crop = image::imageops::crop_imm(
                &full_gray,
                pixel_rect.x,
                pixel_rect.y,
                pixel_rect.width,
                pixel_rect.height,
            )
            .to_image();

            let longest_side = pixel_rect.width.max(pixel_rect.height);
            let (scale, scaled_w, scaled_h, scaled_img) = if longest_side
                > POINT_DETECTION_MAX_LONGEST_EDGE
            {
                let s = POINT_DETECTION_MAX_LONGEST_EDGE as f32 / longest_side as f32;
                let sw = (pixel_rect.width as f32 * s).round() as u32;
                let sh = (pixel_rect.height as f32 * s).round() as u32;
                let scaled =
                    image::imageops::resize(&crop, sw, sh, image::imageops::FilterType::Triangle);
                (s, sw, sh, Some(scaled))
            } else {
                (1.0, pixel_rect.width, pixel_rect.height, None)
            };

            specs.push(ScaledSpec {
                roi: roi.clone(),
                orig_size: pixel_rect.size(),
                scale,
                scaled_w,
                scaled_h,
                scaled_img,
                orig_img: crop,
            });
        }

        let n_frames = specs.len();
        let delegator = luma.orientation.delegator();

        let mut batched_results: Option<Vec<Vec<KeyPoint>>> = None;
        let model_shared = self.get_or_init_model();

        if let Some(shared_model) = model_shared {
            #[allow(clippy::significant_drop_tightening)]
            let plan_res: AlignmentResult<Vec<Vec<KeyPoint>>> = (|| {
                let mut all_keypoints = Vec::with_capacity(n_frames);

                for spec in &specs {
                    let mut canvas_luma =
                        vec![0.0_f32; (MODEL_CANVAS_HEIGHT * MODEL_CANVAS_WIDTH) as usize];
                    let active_img = spec.scaled_img.as_ref().unwrap_or(&spec.orig_img);
                    delegator.fill_inference_canvas(
                        &mut canvas_luma,
                        MODEL_CANVAS_WIDTH,
                        active_img,
                        spec.scaled_w,
                        spec.scaled_h,
                    );

                    let tensor = Array4::from_shape_vec(
                        (
                            1,
                            1,
                            MODEL_CANVAS_HEIGHT as usize,
                            MODEL_CANVAS_WIDTH as usize,
                        ),
                        canvas_luma,
                    )
                    .map_err(|e| {
                        AlignmentError::TensorLayout(format!("Failed to build input tensor: {e}"))
                    })?;

                    let ctx = SuperPointDecodeContext {
                        delegator,
                        scaled_size: Size2D::new(spec.scaled_w, spec.scaled_h),
                        scale: spec.scale,
                        border: self.config.remove_borders as f32,
                        score_threshold: self.config.keypoint_threshold,
                        max_keypoints: self.config.max_keypoints_per_image,
                    };

                    let keypoints = shared_model.infer_and_decode(tensor, &ctx)?;
                    all_keypoints.push(keypoints);
                }

                if all_keypoints.len() == n_frames {
                    Ok(all_keypoints)
                } else {
                    Err(AlignmentError::Inference(
                        "Incomplete frame inference results".to_string(),
                    ))
                }
            })();

            match plan_res {
                Ok(all_kps) => {
                    batched_results = Some(all_kps);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "ONNX SuperPoint execution failed; falling back");
                }
            }
        }

        let mut results = Vec::with_capacity(specs.len());
        for (b_idx, spec) in specs.into_iter().enumerate() {
            let mut keypoints = if let Some(ref all_kps) = batched_results {
                all_kps.get(b_idx).cloned().unwrap_or_default()
            } else {
                let active_img = spec.scaled_img.as_ref().unwrap_or(&spec.orig_img);
                let float_luma: Vec<f32> = active_img
                    .as_raw()
                    .iter()
                    .map(|&b| f32::from(b) / 255.0)
                    .collect();
                let scaled_size = Size2D::new(spec.scaled_w, spec.scaled_h);
                let scores = Self::compute_luma_corner_response(&float_luma, scaled_size);
                let mut raw_kps = Self::non_maximum_suppression(
                    &scores,
                    scaled_size,
                    self.config.nms_radius,
                    self.config.keypoint_threshold,
                    self.config.remove_borders,
                );
                raw_kps.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                raw_kps.truncate(self.config.max_keypoints_per_image);

                raw_kps
                    .into_iter()
                    .map(|(pt, score)| {
                        let ox = pt.x / spec.scale;
                        let oy = pt.y / spec.scale;
                        KeyPoint::new(Point2D::new(ox, oy), score, None)
                    })
                    .collect()
            };

            if self.config.subpixel_refinement {
                refine_keypoints_subpixel_with_drift(
                    &spec.orig_img,
                    &mut keypoints,
                    self.config.subpixel_patch_radius,
                    self.config.subpixel_max_iterations,
                    DEFAULT_SUBPIXEL_EPSILON_PX,
                    self.config.max_subpixel_drift_px,
                );
            }

            let frame =
                FeatureFrame::new(spec.roi.index, spec.roi.bounds, spec.orig_size, keypoints);

            if let Some(t) = tap {
                t.on_features_extracted(spec.roi.index, &frame);
            }

            results.push(frame);
        }

        Ok(results)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::geom::NormalizedRect;
    use crate::luma::{Bt709LumaConverter, ScaledGrayscaleStrip, PROJECTION_MAX_DIMENSION};
    use image::{Rgba, RgbaImage};

    #[test]
    fn test_superpoint_nms() {
        let mut scores = vec![0.0_f32; 100]; // 10x10
        scores[5 * 10 + 5] = 0.9;
        scores[5 * 10 + 6] = 0.8; // suppressed by 0.9 within radius 2

        let kps =
            SuperPointDetector::non_maximum_suppression(&scores, Size2D::new(10, 10), 2, 0.5, 1);
        assert_eq!(kps.len(), 1);
        assert_eq!(kps[0].0.x, 5.0);
        assert_eq!(kps[0].0.y, 5.0);
        assert_eq!(kps[0].1, 0.9);
    }

    #[test]
    fn test_detect_luma() {
        let config = SuperPointConfig {
            nms_radius: 2,
            keypoint_threshold: 0.0,
            max_keypoints_per_image: 50,
            remove_borders: 1,
            device: BackendDevice::Cpu,
            model_path: Some(PathBuf::from("/nonexistent/model.onnx")),
            ..SuperPointConfig::default()
        };
        let detector = SuperPointDetector::new(config);

        // 200x200 image with high-contrast checkerboard corners
        let mut luma = vec![0.1_f32; 40000];
        for y in 40..120 {
            for x in 40..120 {
                luma[y * 200 + x] = 0.95;
            }
        }

        let kps = detector.detect_luma(&luma, Size2D::new(200, 200)).unwrap();
        assert!(!kps.is_empty());
    }

    #[test]
    fn test_detect_strip_roi() {
        let mut img = RgbaImage::from_pixel(300, 100, Rgba([30, 30, 30, 255]));
        // Draw bright box in frame 0
        for y in 20..40 {
            for x in 20..40 {
                img.put_pixel(x, y, Rgba([240, 240, 240, 255]));
            }
        }

        let converter = Bt709LumaConverter::new();
        let strip =
            ScaledGrayscaleStrip::from_image(&img, &converter, PROJECTION_MAX_DIMENSION).unwrap();

        let roi = FrameRoi {
            index: 0,
            bounds: NormalizedRect::new(0.0, 0.0, 0.333, 1.0).unwrap(),
            confidence: 0.99,
        };

        let config = SuperPointConfig {
            keypoint_threshold: 0.0001,
            model_path: Some(PathBuf::from("/nonexistent/model.onnx")),
            ..SuperPointConfig::default()
        };
        let detector = SuperPointDetector::new(config);
        let frame = detector.detect_strip_roi(&strip, &roi, None).unwrap();
        assert_eq!(frame.frame_index, 0);
        assert!(!frame.is_empty());
    }

    #[test]
    fn test_superpoint_shared_model_lifecycle() {
        let model_path = get_default_model_cache_path();
        if !model_path.exists() {
            return;
        }

        let config = SuperPointConfig {
            model_path: Some(model_path),
            ..SuperPointConfig::default()
        };

        let detector_a = SuperPointDetector::new(config.clone());
        let detector_b = SuperPointDetector::new(config);

        // Before initialization, detectors have not loaded the shared model
        assert!(detector_a.model.lock().unwrap().is_none());
        assert!(detector_b.model.lock().unwrap().is_none());

        // Initialize detector A
        detector_a.ensure_backend_ready().unwrap();
        assert!(detector_a.model.lock().unwrap().is_some());

        // Initialize detector B
        detector_b.ensure_backend_ready().unwrap();
        assert!(detector_b.model.lock().unwrap().is_some());

        let model_a = detector_a.get_or_init_model().unwrap();
        let model_b = detector_b.get_or_init_model().unwrap();

        // Both detectors share the exact same Arc instance from the cache
        assert!(Arc::ptr_eq(&model_a, &model_b));
    }

    #[test]
    fn test_superpoint_concurrent_single_flight() {
        let model_path = get_default_model_cache_path();
        if !model_path.exists() {
            return;
        }

        let config = SuperPointConfig {
            model_path: Some(model_path),
            ..SuperPointConfig::default()
        };

        // Simulate 8 threads requesting the same model concurrently
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let mut handles = Vec::new();

        for _ in 0..8 {
            let c = barrier.clone();
            let cfg = config.clone();

            handles.push(std::thread::spawn(move || {
                c.wait();
                get_or_load_superpoint_model(&cfg).unwrap()
            }));
        }

        let mut results = Vec::new();
        for h in handles {
            results.push(h.join().unwrap());
        }

        let first = &results[0];
        for r in &results {
            assert!(Arc::ptr_eq(r, first));
        }
    }

    #[test]
    fn test_compute_file_sha256_and_cache_hit() {
        use std::io::Write;
        let temp_dir = std::env::temp_dir().join(format!("reto_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&temp_dir);
        let test_file = temp_dir.join("test_model.bin");

        let content = b"SuperPoint ONNX Model Test Payload";
        let mut f = std::fs::File::create(&test_file).unwrap();
        f.write_all(content).unwrap();
        drop(f);

        let hash = compute_file_sha256(&test_file).unwrap();
        assert!(!hash.is_empty());

        // When hash matches, ensure_model_cached succeeds without attempting download
        let res = ensure_model_cached(&test_file, "http://invalid-url.local", Some(&hash));
        assert!(res.is_ok());

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_triplet_consistency_boundary_condition() {
        let config = TripletConsistencyConfig::default();

        // 1. Valid rigid inlier:
        // pt0 = (100.0, 50.0), pt1 = (80.0, 50.0), pt2 = (60.0, 50.0)
        // disp_01 = 20.0, disp_12 = 20.0, disp_02 = 40.0
        // cascade_error = |40.0 - (20.0 + 20.0)| = 0.0
        let p0 = Point2D::new(100.0, 50.0);
        let p1 = Point2D::new(80.0, 50.0);
        let p2 = Point2D::new(60.0, 50.0);
        let res = config.verify_triplet(p0, p1, p2);
        assert!(res.is_some());
        let (d01, d12, err) = res.unwrap();
        assert_eq!(d01, 20.0);
        assert_eq!(d12, 20.0);
        assert_eq!(err, 0.0);

        // 2. Outlier due to vertical disparity jitter exceeding tolerance:
        let p1_vertical_jitter = Point2D::new(80.0, 80.0);
        assert!(config.verify_triplet(p0, p1_vertical_jitter, p2).is_none());

        // 3. Outlier due to opposing disparity directions (forward vs reverse):
        let p1_opposing_disp = Point2D::new(120.0, 50.0);
        assert!(config.verify_triplet(p0, p1_opposing_disp, p2).is_none());

        // 4. Outlier due to disparity ratio deviation exceeding tolerance:
        let p2_ratio_outlier = Point2D::new(75.0, 50.0); // disp_01 = 20, disp_12 = 5 -> ratio = 4.0
        assert!(config.verify_triplet(p0, p1, p2_ratio_outlier).is_none());
    }

    #[test]
    fn test_chassis_extrinsics_estimation() {
        let dummy_rect = NormalizedRect::new(0.0, 0.0, 0.33, 1.0).unwrap();
        let dummy_size = Size2D::new(100, 100);

        let f0 = FeatureFrame::new(
            0,
            dummy_rect,
            dummy_size,
            vec![
                KeyPoint::new(Point2D::new(100.0, 50.0), 0.9, None),
                KeyPoint::new(Point2D::new(200.0, 80.0), 0.9, None),
            ],
        );
        let f1 = FeatureFrame::new(
            1,
            dummy_rect,
            dummy_size,
            vec![
                KeyPoint::new(Point2D::new(80.0, 52.0), 0.9, None),
                KeyPoint::new(Point2D::new(180.0, 82.0), 0.9, None),
            ],
        );
        let f2 = FeatureFrame::new(
            2,
            dummy_rect,
            dummy_size,
            vec![
                KeyPoint::new(Point2D::new(60.0, 50.0), 0.9, None),
                KeyPoint::new(Point2D::new(160.0, 80.0), 0.9, None),
            ],
        );

        let triplets = vec![
            FeatureTriplet {
                index_0: 0,
                index_1: 0,
                index_2: 0,
                confidence: 0.9,
                disparity_01: 20.0,
                disparity_12: 20.0,
                cascade_error: 0.0,
            },
            FeatureTriplet {
                index_0: 1,
                index_1: 1,
                index_2: 1,
                confidence: 0.9,
                disparity_01: 20.0,
                disparity_12: 20.0,
                cascade_error: 0.0,
            },
        ];

        let frames = [f0, f1, f2];
        let extrinsics = ChassisExtrinsics::estimate_from_triplets(
            &triplets,
            &frames,
            StripOrientation::Horizontal,
        );

        assert_eq!(extrinsics.inlier_count, 2);
        assert_eq!(extrinsics.translation_01.0, 20.0);
        assert_eq!(extrinsics.translation_01.1, 2.0); // center frame dropped by 2px relative to frame 0
        assert_eq!(extrinsics.translation_12.0, 20.0);
        assert_eq!(extrinsics.translation_12.1, -2.0); // frame 2 moved up by 2px relative to frame 1
        assert_eq!(extrinsics.translation_02.0, 40.0);
        assert_eq!(extrinsics.translation_02.1, 0.0); // frame 0 and frame 2 collinear on Y
        assert_eq!(extrinsics.center_sag_px, 2.0); // center sag is 2.0px
        assert!((extrinsics.empirical_baseline_ratio - 1.0).abs() < 1e-4);
    }

    struct MockFeatureMatcher {
        p01: Vec<FeatureMatch>,
        p10: Vec<FeatureMatch>,
        p12: Vec<FeatureMatch>,
        p21: Vec<FeatureMatch>,
        p02: Vec<FeatureMatch>,
        p20: Vec<FeatureMatch>,
    }

    impl FeatureMatcher for MockFeatureMatcher {
        fn match_pair(
            &self,
            frame_a: &FeatureFrame,
            frame_b: &FeatureFrame,
        ) -> AlignmentResult<PairwiseMatchSet> {
            let matches = match (frame_a.frame_index, frame_b.frame_index) {
                (0, 1) => self.p01.clone(),
                (1, 0) => self.p10.clone(),
                (1, 2) => self.p12.clone(),
                (2, 1) => self.p21.clone(),
                (0, 2) => self.p02.clone(),
                (2, 0) => self.p20.clone(),
                _ => Vec::new(),
            };
            Ok(PairwiseMatchSet::new(
                (frame_a.frame_index, frame_b.frame_index),
                matches,
                MatchDirection::Forward,
            ))
        }
    }

    #[test]
    fn test_feature_matcher_extract_consistent_triplets() {
        let dummy_rect = NormalizedRect::new(0.0, 0.0, 0.33, 1.0).unwrap();
        let dummy_size = Size2D::new(100, 100);

        // Frame 0: inlier point at (100.0, 50.0), outlier point at (150.0, 50.0)
        let f0 = FeatureFrame::new(
            0,
            dummy_rect,
            dummy_size,
            vec![
                KeyPoint::new(Point2D::new(100.0, 50.0), 0.9, None),
                KeyPoint::new(Point2D::new(150.0, 50.0), 0.9, None),
            ],
        );
        // Frame 1: inlier point at (80.0, 50.0), outlier point at (120.0, 50.0)
        let f1 = FeatureFrame::new(
            1,
            dummy_rect,
            dummy_size,
            vec![
                KeyPoint::new(Point2D::new(80.0, 50.0), 0.9, None),
                KeyPoint::new(Point2D::new(120.0, 50.0), 0.9, None),
            ],
        );
        // Frame 2: inlier point at (60.0, 50.0), outlier point at (130.0, 50.0)
        let f2 = FeatureFrame::new(
            2,
            dummy_rect,
            dummy_size,
            vec![
                KeyPoint::new(Point2D::new(60.0, 50.0), 0.9, None),
                KeyPoint::new(Point2D::new(130.0, 50.0), 0.9, None),
            ],
        );

        let matcher = MockFeatureMatcher {
            // Pair 0-1
            p01: vec![
                FeatureMatch::new(0, 0, 0.95), // inlier candidate
                FeatureMatch::new(1, 1, 0.80), // outlier candidate (disp_01 = 30)
            ],
            p10: vec![FeatureMatch::new(0, 0, 0.95), FeatureMatch::new(1, 1, 0.80)],
            // Pair 1-2
            p12: vec![
                FeatureMatch::new(0, 0, 0.90), // inlier candidate
                FeatureMatch::new(1, 1, 0.80), // outlier candidate (disp_12 = 5) -> disp_01/disp_12 = 6!
            ],
            p21: vec![FeatureMatch::new(0, 0, 0.90), FeatureMatch::new(1, 1, 0.80)],
            // Pair 0-2
            p02: vec![
                FeatureMatch::new(0, 0, 0.92), // inlier candidate
                FeatureMatch::new(1, 1, 0.80), // outlier candidate
            ],
            p20: vec![FeatureMatch::new(0, 0, 0.92), FeatureMatch::new(1, 1, 0.80)],
        };

        let frames = [f0, f1, f2];
        let triplets = matcher
            .extract_consistent_triplets(&frames, &TripletConsistencyConfig::default(), None)
            .unwrap();

        // Exactly one triplet survives the boundary condition verification!
        assert_eq!(triplets.len(), 1);
        assert_eq!(triplets[0].index_0, 0);
        assert_eq!(triplets[0].index_1, 0);
        assert_eq!(triplets[0].index_2, 0);
        assert_eq!(triplets[0].disparity_01, 20.0);
        assert_eq!(triplets[0].disparity_12, 20.0);
        assert_eq!(triplets[0].cascade_error, 0.0);
    }

    #[test]
    fn test_subpixel_refinement_checkerboard_corner() {
        use image::GrayImage;

        // 64x64 image with a sharp high-contrast corner at (32.0, 32.0)
        let mut img = GrayImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                if (x < 32 && y < 32) || (x >= 32 && y >= 32) {
                    img.put_pixel(x, y, image::Luma([240]));
                } else {
                    img.put_pixel(x, y, image::Luma([15]));
                }
            }
        }

        // Perturbed initial detection at (32.4, 31.7)
        let mut kps = vec![KeyPoint::new(Point2D::new(32.4, 31.7), 0.99, None)];

        refine_keypoints_subpixel(&img, &mut kps, 7, 5, 0.005);

        // Keypoint should converge precisely to true continuous corner interface (31.5, 31.5)
        assert!(
            (kps[0].point.x - 31.5).abs() < 0.05,
            "Refined X {} is not within 0.05 of true corner 31.5",
            kps[0].point.x
        );
        assert!(
            (kps[0].point.y - 31.5).abs() < 0.05,
            "Refined Y {} is not within 0.05 of true corner 31.5",
            kps[0].point.y
        );
        assert!(matches!(
            kps[0].subpixel_status,
            SubpixelStatus::Refined { .. }
        ));
    }

    #[test]
    fn test_subpixel_refinement_flat_and_boundary_safety() {
        use image::GrayImage;

        let flat_img = GrayImage::from_pixel(64, 64, image::Luma([128]));
        let mut kps = vec![
            KeyPoint::new(Point2D::new(32.0, 32.0), 0.9, None),
            KeyPoint::new(Point2D::new(1.0, 1.0), 0.9, None), // Near border
            KeyPoint::new(Point2D::new(63.0, 63.0), 0.9, None), // Near border
        ];

        refine_keypoints_subpixel(&flat_img, &mut kps, 7, 5, 0.01);

        // Flat region should report PoorConditioning and border points should report BoundarySkipped
        assert!((kps[0].point.x - 32.0).abs() < f32::EPSILON);
        assert!((kps[0].point.y - 32.0).abs() < f32::EPSILON);
        assert!(matches!(
            kps[0].subpixel_status,
            SubpixelStatus::PoorConditioning { .. }
        ));

        assert!((kps[1].point.x - 1.0).abs() < f32::EPSILON);
        assert!((kps[1].point.y - 1.0).abs() < f32::EPSILON);
        assert_eq!(kps[1].subpixel_status, SubpixelStatus::BoundarySkipped);

        assert!((kps[2].point.x - 63.0).abs() < f32::EPSILON);
        assert!((kps[2].point.y - 63.0).abs() < f32::EPSILON);
        assert_eq!(kps[2].subpixel_status, SubpixelStatus::BoundarySkipped);
    }

    #[test]
    fn test_subpixel_refinement_drift_rejection() {
        use image::GrayImage;

        let mut img = GrayImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                if (x < 32 && y < 32) || (x >= 32 && y >= 32) {
                    img.put_pixel(x, y, image::Luma([240]));
                } else {
                    img.put_pixel(x, y, image::Luma([15]));
                }
            }
        }

        // Perturbed initial point placed at (32.3, 32.2) which is ~1.06px away from corner (31.5, 31.5)
        let mut kps = vec![KeyPoint::new(Point2D::new(32.3, 32.2), 0.99, None)];

        // Run with strict max drift = 0.4px (smaller than required 1.06px displacement)
        refine_keypoints_subpixel_with_drift(&img, &mut kps, 7, 5, 0.005, 0.4);

        // Should be rejected due to drift exceeding 0.4px and reverted to initial (32.3, 32.2)
        assert!((kps[0].point.x - 32.3).abs() < f32::EPSILON);
        assert!((kps[0].point.y - 32.2).abs() < f32::EPSILON);
        assert!(matches!(
            kps[0].subpixel_status,
            SubpixelStatus::DriftRejected { .. }
        ));
    }

    #[test]
    fn test_subpixel_refinement_config_toggle() {
        let config_disabled = SuperPointConfig {
            keypoint_threshold: 0.0001,
            subpixel_refinement: false,
            model_path: Some(PathBuf::from("/nonexistent/model.onnx")),
            ..SuperPointConfig::default()
        };
        let config_enabled = SuperPointConfig {
            keypoint_threshold: 0.0001,
            subpixel_refinement: true,
            model_path: Some(PathBuf::from("/nonexistent/model.onnx")),
            ..SuperPointConfig::default()
        };

        let mut luma = vec![0.1_f32; 10000]; // 100x100
        for y in 30..70 {
            for x in 30..70 {
                luma[y * 100 + x] = 0.9;
            }
        }

        let detector_off = SuperPointDetector::new(config_disabled);
        let detector_on = SuperPointDetector::new(config_enabled);

        let kps_off = detector_off
            .detect_luma(&luma, Size2D::new(100, 100))
            .unwrap();
        let kps_on = detector_on
            .detect_luma(&luma, Size2D::new(100, 100))
            .unwrap();

        assert!(!kps_off.is_empty());
        assert!(!kps_on.is_empty());
    }

    #[test]
    fn test_hierarchical_reduction_tree_n3() {
        let dummy_rect = NormalizedRect::new(0.0, 0.0, 0.33, 1.0).unwrap();
        let dummy_size = Size2D::new(100, 100);

        let f0 = FeatureFrame::new(
            0,
            dummy_rect,
            dummy_size,
            vec![
                KeyPoint::new(Point2D::new(100.0, 50.0), 0.9, None),
                KeyPoint::new(Point2D::new(200.0, 80.0), 0.9, None),
            ],
        );
        let f1 = FeatureFrame::new(
            1,
            dummy_rect,
            dummy_size,
            vec![
                KeyPoint::new(Point2D::new(80.0, 52.0), 0.9, None),
                KeyPoint::new(Point2D::new(180.0, 82.0), 0.9, None),
            ],
        );
        let f2 = FeatureFrame::new(
            2,
            dummy_rect,
            dummy_size,
            vec![
                KeyPoint::new(Point2D::new(60.0, 50.0), 0.9, None),
                KeyPoint::new(Point2D::new(160.0, 80.0), 0.9, None),
            ],
        );

        let frames = [f0, f1, f2];
        let m01 = PairwiseMatchSet::new(
            (0, 1),
            vec![
                FeatureMatch::new(0, 0, 0.9),
                FeatureMatch::new(1, 1, 0.9),
            ],
            MatchDirection::Mutual,
        );
        let m12 = PairwiseMatchSet::new(
            (1, 2),
            vec![
                FeatureMatch::new(0, 0, 0.9),
                FeatureMatch::new(1, 1, 0.9),
            ],
            MatchDirection::Mutual,
        );
        let m02 = PairwiseMatchSet::new(
            (0, 2),
            vec![
                FeatureMatch::new(0, 0, 0.9),
                FeatureMatch::new(1, 1, 0.9),
            ],
            MatchDirection::Mutual,
        );

        let match_sets = [m01, m12, m02];
        let tree = HierarchicalReductionTree::build_from_match_sets(
            &frames,
            &match_sets,
            StripOrientation::Horizontal,
        );

        assert_eq!(tree.num_frames, 3);
        assert!(!tree.spans.is_empty());

        let report = tree.optimize(&HierarchicalExtrinsicsConfig::default());
        assert_eq!(report.num_cameras, 3);
        assert_eq!(report.camera_positions.len(), 3);
        assert_eq!(report.camera_positions[0], (0.0, 0.0));
        assert!((report.camera_positions[1].0 - 20.0).abs() < 1e-3);
        assert!((report.camera_positions[1].1 - 2.0).abs() < 1e-3);
        assert!((report.camera_positions[2].0 - 40.0).abs() < 1e-3);
        assert!((report.camera_positions[2].1 - 0.0).abs() < 1e-3);
        assert_eq!(report.center_sags_px.len(), 1);
        assert!((report.center_sags_px[0] - 2.0).abs() < 1e-3);
        assert!(report.rmse_after_px < 0.01);
    }

    #[test]
    fn test_hierarchical_reduction_tree_n4() {
        let dummy_rect = NormalizedRect::new(0.0, 0.0, 0.25, 1.0).unwrap();
        let dummy_size = Size2D::new(100, 100);

        let f0 = FeatureFrame::new(
            0,
            dummy_rect,
            dummy_size,
            vec![KeyPoint::new(Point2D::new(100.0, 50.0), 0.9, None)],
        );
        let f1 = FeatureFrame::new(
            1,
            dummy_rect,
            dummy_size,
            vec![KeyPoint::new(Point2D::new(80.0, 51.0), 0.9, None)],
        );
        let f2 = FeatureFrame::new(
            2,
            dummy_rect,
            dummy_size,
            vec![KeyPoint::new(Point2D::new(60.0, 52.0), 0.9, None)],
        );
        let f3 = FeatureFrame::new(
            3,
            dummy_rect,
            dummy_size,
            vec![KeyPoint::new(Point2D::new(40.0, 50.0), 0.9, None)],
        );

        let frames = [f0, f1, f2, f3];
        let m01 = PairwiseMatchSet::new(
            (0, 1),
            vec![FeatureMatch::new(0, 0, 0.9)],
            MatchDirection::Mutual,
        );
        let m12 = PairwiseMatchSet::new(
            (1, 2),
            vec![FeatureMatch::new(0, 0, 0.9)],
            MatchDirection::Mutual,
        );
        let m23 = PairwiseMatchSet::new(
            (2, 3),
            vec![FeatureMatch::new(0, 0, 0.9)],
            MatchDirection::Mutual,
        );
        let m02 = PairwiseMatchSet::new(
            (0, 2),
            vec![FeatureMatch::new(0, 0, 0.9)],
            MatchDirection::Mutual,
        );
        let m13 = PairwiseMatchSet::new(
            (1, 3),
            vec![FeatureMatch::new(0, 0, 0.9)],
            MatchDirection::Mutual,
        );
        let m03 = PairwiseMatchSet::new(
            (0, 3),
            vec![FeatureMatch::new(0, 0, 0.9)],
            MatchDirection::Mutual,
        );

        let match_sets = [m01, m12, m23, m02, m13, m03];
        let tree = HierarchicalReductionTree::build_from_match_sets(
            &frames,
            &match_sets,
            StripOrientation::Horizontal,
        );

        assert_eq!(tree.num_frames, 4);
        let report = tree.optimize(&HierarchicalExtrinsicsConfig::default());
        assert_eq!(report.num_cameras, 4);
        assert_eq!(report.adjacent_translations.len(), 3);
        assert_eq!(report.center_sags_px.len(), 2);
        assert!(report.rmse_after_px < 0.05);
    }
}
