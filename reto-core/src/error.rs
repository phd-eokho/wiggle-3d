//! Error handling and Result types for reto-core.

use thiserror::Error;

/// Specific error conditions during Region of Interest (`RoI`) extraction and validation.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum RoiError {
    /// Invalid normalized coordinates.
    #[error("Invalid normalized bounding box coordinates: x={x}, y={y}, w={width}, h={height}")]
    InvalidBounds {
        /// X coordinate in unit space.
        x: f32,
        /// Y coordinate in unit space.
        y: f32,
        /// Width in unit space.
        width: f32,
        /// Height in unit space.
        height: f32,
    },

    /// Square image rejected.
    #[error("Square images ({width}x{height}) are not supported: film strips must have aspect ratio != 1.0")]
    SquareImageNotSupported {
        /// Image width.
        width: u32,
        /// Image height.
        height: u32,
    },

    /// Requested frame index is out of range.
    #[error("Frame index {requested} is out of range (total frames detected: {available})")]
    FrameIndexOutOfRange {
        /// The requested frame index.
        requested: usize,
        /// Total number of available frames.
        available: usize,
    },

    /// Detected frame count mismatch.
    #[error("Detection failed: expected {expected} frames, but found {found}")]
    MismatchedFrameCount {
        /// Expected count of frames.
        expected: usize,
        /// Actually detected count of frames.
        found: usize,
    },

    /// Film strip gutters could not be located.
    #[error("Insufficient contrast or film gutters not found")]
    GuttersNotFound,

    /// Image is too small for processing.
    #[error("Image dimensions ({width}x{height}) are too small for reliable RoI detection")]
    ImageTooSmall {
        /// Image width.
        width: u32,
        /// Image height.
        height: u32,
    },

    /// Zero frames configured.
    #[error("Expected frame count must be greater than zero, got {0}")]
    ZeroExpectedFrames(usize),
}

/// Errors occurring during feature extraction, matching, and parallax alignment.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum AlignmentError {
    /// Deep learning model inference failure.
    #[error("Inference failure: {0}")]
    Inference(String),

    /// Model loading or initialization error.
    #[error("Model load error: {0}")]
    ModelLoad(String),

    /// Tensor shape, layout, or type mismatch.
    #[error("Tensor layout mismatch: {0}")]
    TensorLayout(String),

    /// Requested execution provider or backend device is unavailable.
    #[error("Backend unavailable: {0}")]
    BackendUnavailable(String),

    /// Insufficient keypoints extracted for robust correspondence.
    #[error("Insufficient keypoints: detected {found}, minimum required is {required}")]
    InsufficientKeypoints {
        /// Number of keypoints found.
        found: usize,
        /// Minimum number required.
        required: usize,
    },

    /// Invalid input parameters or image buffer dimensions.
    #[error("Invalid alignment input: {0}")]
    InvalidInput(String),

    /// Underlying `RoI` extraction or geometric error.
    #[error("RoI error during alignment: {0}")]
    Roi(#[from] RoiError),
}

/// Errors occurring during face detection and facial focal plane selection.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum FaceError {
    /// Deep learning model inference failure.
    #[error("Inference failure: {0}")]
    Inference(String),

    /// Model loading or initialization error.
    #[error("Model load error: {0}")]
    ModelLoad(String),

    /// Tensor shape, layout, or type mismatch.
    #[error("Tensor layout mismatch: {0}")]
    TensorLayout(String),

    /// Invalid input parameters or image buffer dimensions.
    #[error("Invalid face detection input: {0}")]
    InvalidInput(String),

    /// Region of Interest or geometry error during face detection.
    #[error("RoI error during face detection: {0}")]
    Roi(#[from] RoiError),
}

/// Common error type for reto-core.
#[derive(Debug, Error)]
pub enum Error {
    /// IO error wrapper.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// `RoI` detection / geometry error wrapper.
    #[error("RoI error: {0}")]
    Roi(#[from] RoiError),

    /// Feature alignment and parallax error wrapper.
    #[error("Alignment error: {0}")]
    Alignment(#[from] AlignmentError),

    /// Face detection and focal plane error wrapper.
    #[error("Face error: {0}")]
    Face(#[from] FaceError),

    /// Image processing error wrapper.
    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),

    /// Video encoding and container error wrapper.
    #[error("Video error: {0}")]
    Video(#[from] crate::video::VideoError),

    /// Placeholder for other errors.
    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// Type alias for `Result` with `reto_core::Error`.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn test_error_from_io() {
        let io_error = io::Error::new(io::ErrorKind::NotFound, "test error");
        let error: Error = io_error.into();
        assert!(matches!(error, Error::Io(_)));
        assert!(error.to_string().contains("IO error: test error"));
    }

    #[test]
    fn test_error_from_roi() {
        let roi_err = RoiError::SquareImageNotSupported {
            width: 100,
            height: 100,
        };
        let error: Error = roi_err.into();
        assert!(matches!(error, Error::Roi(_)));
        assert!(error.to_string().contains("Square images"));
    }
}
