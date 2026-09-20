#![forbid(unsafe_code)]
#![allow(clippy::redundant_pub_crate, clippy::doc_markdown)]
//! Core library for Reto-Split: 3D film camera image splitting, alignment, and parallax processing.

// Encapsulated internal modules (thin public API facade)
pub(crate) mod detector;
pub(crate) mod error;
pub(crate) mod face;
pub(crate) mod feature;
pub(crate) mod geom;
pub(crate) mod gif;
pub(crate) mod luma;
pub(crate) mod partition;
pub(crate) mod pipeline;
pub(crate) mod stats;
pub(crate) mod visualizer;

// Public Facade Exports
pub use detector::{
    CompositeDiagnosticTap, EvenSplitDetector, NoOpDiagnosticTap, PillarStatsDetector,
    RoiDetectionConfig, RoiDetector, RoiDiagnosticTap, SaveDenoisedLumaDiagnosticTap,
    SaveLumaDiagnosticTap,
};
pub use error::{AlignmentError, Error, FaceError, Result, RoiError};
pub use face::{
    clear_retinaface_model_cache, generate_retinaface_anchors, get_default_retinaface_cache_path,
    select_dominant_face, FaceDetection, FaceDetector, FaceDiagnosticTap, FacialLandmarks,
    NoOpFaceDiagnosticTap, RetinaFaceConfig, RetinaFaceDetector, DEFAULT_CENTER_SALIENT_SIGMA,
    DEFAULT_FACE_CONFIDENCE_THRESHOLD, DEFAULT_FACE_NMS_THRESHOLD, DEFAULT_RETINAFACE_MODEL_SHA256,
    DEFAULT_RETINAFACE_MODEL_URL, FALLBACK_RETINAFACE_MODEL_URL, RETINAFACE_CANVAS_HEIGHT,
    RETINAFACE_CANVAS_WIDTH,
};
pub use feature::{
    clear_superpoint_model_cache, compute_file_sha256, ensure_model_cached,
    get_default_model_cache_path, get_default_ort_dylib_path, init_ort_environment_if_needed,
    refine_keypoints_subpixel, refine_keypoints_subpixel_with_drift, AlignmentDiagnosticTap,
    AlignmentResult, BackendDevice, BranchStatus, ChassisExtrinsics, DyadicSpan,
    ExtrinsicsTolerance, FallbackStatus, FeatureExtractor, FeatureFrame, FeatureMatch,
    FeatureMatcher, FeatureTriplet, FramePair, HierarchicalExtrinsicsConfig,
    HierarchicalOptimizationReport, HierarchicalReductionTree, KeyPoint, KeypointScore,
    MatchDirection, MatchScore, NoOpAlignmentDiagnosticTap, PairwiseMatchSet, PointDetector,
    RmseMetric, RotationMatrix3x3, SubpixelStatus, SuperPointConfig, SuperPointDescriptorMatcher,
    SuperPointDetector, SuperPointExtractor, TranslationVector3, TripletConsistencyConfig,
    DEFAULT_SUBPIXEL_EPSILON_PX, DEFAULT_SUBPIXEL_MAX_DRIFT_PX, DEFAULT_SUBPIXEL_MAX_ITERATIONS,
    DEFAULT_SUBPIXEL_PATCH_RADIUS, DEFAULT_SUPERPOINT_MODEL_SHA256, DEFAULT_SUPERPOINT_MODEL_URL,
    FALLBACK_SUPERPOINT_MODEL_URL, MODEL_CANVAS_HEIGHT, MODEL_CANVAS_WIDTH,
    POINT_DETECTION_MAX_LONGEST_EDGE,
};
pub use geom::{
    FrameRoi, FrameRoiSet, GlobalCoord, HorizontalDelegator, LocalCoord, NormalizedRect,
    OrientationDelegator, PixelRect, Point2D, Size2D, StripOrientation, VerticalDelegator,
    HORIZONTAL_DELEGATOR, VERTICAL_DELEGATOR,
};
pub use gif::{
    DisparityBinRecord, DisparityHistogramData, NeuQuantColorMap, WiggleAligner, WiggleGifBuilder,
    WiggleGifConfig, DEFAULT_CLUSTER_TOLERANCE_PX, DEFAULT_DISPARITY_BIN_SIZE_PX,
    DEFAULT_FOREGROUND_MIN_PROPORTION, DEFAULT_FRAME_DELAY_MS, DEFAULT_INV_DISPARITY_BIN_SIZE,
    DEFAULT_NEUQUANT_SAMPLE_FAC, DEFAULT_PALETTE_COLORS, MIN_DISPARITY_FOR_DEPTH_PX,
};
pub use luma::{
    median9, Bt709LumaConverter, LumaConverter, ScaledGrayscaleStrip, ScaledLumaImage,
    SimpleGrayConverter, BT709_RGB_WEIGHTS, DEFAULT_CROSS_PERCENTILE, DEFAULT_INVERSE_GAMMA,
    DEFAULT_PROFILE_PERCENTILE, PROJECTION_MAX_DIMENSION,
};
pub use partition::{
    EvenSplitPartitioner, GutterPartitioner, OptimalGridPartitioner, PartitionResult,
    PartitionValidator, PrioritizedPartitionEngine, ThresholdPartitioner,
};
pub use pipeline::{
    is_supported_image, process_item, process_single_image, run_batch, BatchProcessingRequest,
    ImageItemContext, ItemProcessingOutcome, ProcessSummary, ProgressEvent, ProgressObserver,
    SUPPORTED_EXTENSIONS,
};
pub use stats::{
    matched_filter_1d, AxisPixelStats, AxisStatisticsProfile, GutterSpan, OptimalGridResult,
    ThresholdPartitionResult, DEFAULT_HIGH_PERCENTILE, DEFAULT_LOW_PERCENTILE,
};
pub use visualizer::{
    FrameFaceRecord, RoiVisualizer, SaveFacesDiagnosticTap, SaveFeaturesDiagnosticTap,
    SaveMatchesDiagnosticTap,
};
