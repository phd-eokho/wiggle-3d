//! SOLID Gutter partitioning strategies and validation interface for multi-frame film strips.
//!
//! # Architecture & SOLID Design
//! - **Single Responsibility Principle (SRP)**: Each [`GutterPartitioner`] implementation encapsulates
//!   a single mathematical or topological partitioning technique.
//! - **Open-Closed Principle (OCP)**: New partitioning strategies can be added by implementing [`GutterPartitioner`]
//!   without modifying the detector orchestrator or existing partitioners.
//! - **Liskov Substitution Principle (LSP)**: All partitioners return a uniform [`PartitionResult`].
//! - **Interface Segregation Principle (ISP)**: Focused traits [`GutterPartitioner`] and validation in [`PartitionValidator`].
//! - **Dependency Inversion Principle (DIP)**: [`PrioritizedPartitionEngine`] depends on the [`GutterPartitioner`]
//!   abstraction, evaluating candidates in strict priority order with physical validation gating.

use crate::stats::{AxisStatisticsProfile, GutterSpan};
use serde::{Deserialize, Serialize};

/// Unified result from any [`GutterPartitioner`] strategy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartitionResult {
    /// Identifier of the strategy that produced this partition.
    pub strategy_name: &'static str,
    /// Measured confidence score in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Individual active frame pixel spans `(start_pixel, length_pixels)`.
    pub frame_spans: Vec<(u32, u32)>,
    /// Center coordinates of all gutter gaps along the major axis.
    pub gutter_centers: Vec<u32>,
    /// Detailed physical gutter boundary spans.
    pub gutters: Vec<GutterSpan>,
}

/// Interface for film strip gutter partitioning algorithms (Open-Closed Principle).
pub trait GutterPartitioner: Send + Sync {
    /// Returns the human-readable identifier of this strategy.
    fn strategy_name(&self) -> &'static str;

    /// Attempts to partition the 1D axis profile into $N$ frames and $N-1$ gutters.
    ///
    /// Returns `Some(PartitionResult)` if mathematical/topological separation succeeds,
    /// or `None` if the signal is incompatible.
    fn partition(
        &self,
        profile: &AxisStatisticsProfile,
        expected_frames: usize,
    ) -> Option<PartitionResult>;
}

/// Evaluator that validates whether a candidate [`PartitionResult`] satisfies physical camera chassis constraints.
#[derive(Debug, Clone, Copy)]
pub struct PartitionValidator {
    /// Maximum allowed deviation of frame width from nominal pitch (e.g. 0.30 = ±30%).
    pub max_nominal_pitch_deviation: f32,
    /// Maximum allowed variation ratio between adjacent frames (e.g. 0.15 = 15%).
    pub max_adjacent_frame_variation: f32,
}

impl Default for PartitionValidator {
    fn default() -> Self {
        Self {
            max_nominal_pitch_deviation: 0.30,
            max_adjacent_frame_variation: 0.15,
        }
    }
}

impl PartitionValidator {
    /// Creates a new `PartitionValidator` with default tolerances.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_nominal_pitch_deviation: 0.30,
            max_adjacent_frame_variation: 0.15,
        }
    }

    /// Evaluates if a partition result satisfies physical chassis requirements.
    ///
    /// Checks:
    /// 1. Exactly $N$ frame spans and $N-1$ gutters.
    /// 2. Each frame length is within $[(1 - \text{dev}) \cdot \frac{L}{N}, (1 + \text{dev}) \cdot \frac{L}{N}]$.
    /// 3. Adjacent frame lengths differ by no more than $\text{variation} \cdot \text{pitch}$.
    /// 4. Spans are strictly sequential without negative gaps or out-of-bounds indices.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn is_reasonable(
        &self,
        result: &PartitionResult,
        expected_frames: usize,
        major_len: usize,
    ) -> bool {
        if result.frame_spans.len() != expected_frames || expected_frames == 0 || major_len == 0 {
            return false;
        }

        let nominal_pitch = major_len as f32 / expected_frames as f32;
        let min_w = (nominal_pitch * (1.0 - self.max_nominal_pitch_deviation)).round() as u32;
        let max_w = (nominal_pitch * (1.0 + self.max_nominal_pitch_deviation)).round() as u32;

        let mut prev_len: Option<u32> = None;
        let mut prev_end: u32 = 0;

        for &(start, len) in &result.frame_spans {
            // Width bounds relative to nominal pitch
            if len < min_w || len > max_w {
                return false;
            }
            // Sequential ordering
            if start < prev_end && prev_len.is_some() {
                return false;
            }
            // Major axis boundary
            if start + len > major_len as u32 {
                return false;
            }

            // Consistency across adjacent frames
            if let Some(plen) = prev_len {
                let diff = (i64::from(len) - i64::from(plen)).unsigned_abs() as f32;
                let max_allowed_diff = (plen as f32 * self.max_adjacent_frame_variation).max(18.0);
                if diff > max_allowed_diff {
                    return false;
                }
            }

            prev_len = Some(len);
            prev_end = start + len;
        }

        true
    }
}

// ============================================================================
// Concrete Strategy 1: Topological Threshold Partitioner (Priority 1)
// ============================================================================

/// Priority 1: Topological $(2N - 1)$ binary threshold partitioner with stable plateau selection.
#[derive(Debug, Clone, Default)]
pub struct ThresholdPartitioner;

impl ThresholdPartitioner {
    /// Creates a new `ThresholdPartitioner`.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl GutterPartitioner for ThresholdPartitioner {
    fn strategy_name(&self) -> &'static str {
        "TopologicalThreshold"
    }

    fn partition(
        &self,
        profile: &AxisStatisticsProfile,
        expected_frames: usize,
    ) -> Option<PartitionResult> {
        let part = profile.find_threshold_partition(expected_frames)?;
        let gutter_centers: Vec<u32> = part.gutters.iter().map(|g| g.center).collect();

        Some(PartitionResult {
            strategy_name: self.strategy_name(),
            confidence: part.confidence,
            frame_spans: part.frame_spans,
            gutter_centers,
            gutters: part.gutters,
        })
    }
}

// ============================================================================
// Concrete Strategy 2: Bounded Partition Valley Search (Priority 2)
// ============================================================================

/// Priority 2: Bounded partition-window valley search with nominal anchor fallback.
#[derive(Debug, Clone, Default)]
pub struct OptimalGridPartitioner;

impl OptimalGridPartitioner {
    /// Creates a new `OptimalGridPartitioner`.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl GutterPartitioner for OptimalGridPartitioner {
    fn strategy_name(&self) -> &'static str {
        "BoundedValleyGrid"
    }

    fn partition(
        &self,
        profile: &AxisStatisticsProfile,
        expected_frames: usize,
    ) -> Option<PartitionResult> {
        let grid = profile.find_optimal_grid(expected_frames)?;

        Some(PartitionResult {
            strategy_name: self.strategy_name(),
            confidence: 0.50,
            frame_spans: grid.frame_spans,
            gutter_centers: grid.gutter_centers,
            gutters: grid.gutters,
        })
    }
}

// ============================================================================
// Concrete Strategy 3: Deterministic Even Division (Priority 3 Fallback)
// ============================================================================

/// Priority 3: Deterministic even division into $N$ equal chunks of width $\frac{L}{N}$.
#[derive(Debug, Clone, Default)]
pub struct EvenSplitPartitioner;

impl EvenSplitPartitioner {
    /// Creates a new `EvenSplitPartitioner`.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl GutterPartitioner for EvenSplitPartitioner {
    fn strategy_name(&self) -> &'static str {
        "EvenSplitFallback"
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    fn partition(
        &self,
        profile: &AxisStatisticsProfile,
        expected_frames: usize,
    ) -> Option<PartitionResult> {
        if expected_frames == 0 || profile.is_empty() {
            return None;
        }

        let l = profile.len() as u32;
        let n = expected_frames as u32;
        let step = (l / n).max(1);

        let mut frame_spans = Vec::with_capacity(expected_frames);
        let mut gutter_centers = Vec::with_capacity(expected_frames.saturating_sub(1));

        for i in 0..n {
            let start = i * step;
            let len = if i == n - 1 { l - start } else { step };
            frame_spans.push((start, len));

            if i > 0 {
                gutter_centers.push(start);
            }
        }

        Some(PartitionResult {
            strategy_name: self.strategy_name(),
            confidence: 0.10,
            frame_spans,
            gutter_centers,
            gutters: Vec::new(),
        })
    }
}

// ============================================================================
// Prioritized Partition Engine (SOLID Orchestrator)
// ============================================================================

/// Orchestrator that evaluates partitioning strategies in strict priority order
/// and terminates at the first strategy that satisfies physical validation.
pub struct PrioritizedPartitionEngine {
    partitioners: Vec<Box<dyn GutterPartitioner>>,
    validator: PartitionValidator,
}

impl Default for PrioritizedPartitionEngine {
    fn default() -> Self {
        Self::standard()
    }
}

impl PrioritizedPartitionEngine {
    /// Creates a standard prioritized engine configured with:
    /// 1. Priority 1: [`ThresholdPartitioner`]
    /// 2. Priority 2: [`OptimalGridPartitioner`]
    /// 3. Priority 3: [`EvenSplitPartitioner`]
    #[must_use]
    pub fn standard() -> Self {
        Self {
            partitioners: vec![
                Box::new(ThresholdPartitioner::new()),
                Box::new(OptimalGridPartitioner::new()),
                Box::new(EvenSplitPartitioner::new()),
            ],
            validator: PartitionValidator::new(),
        }
    }

    /// Evaluates the priority chain against the axis profile and returns the first validated partition.
    #[must_use]
    #[tracing::instrument(skip(self, profile), level = "debug")]
    pub fn execute(
        &self,
        profile: &AxisStatisticsProfile,
        expected_frames: usize,
    ) -> PartitionResult {
        let major_len = profile.len();

        for (priority, partitioner) in self.partitioners.iter().enumerate() {
            let name = partitioner.strategy_name();
            if let Some(candidate) = partitioner.partition(profile, expected_frames) {
                if self
                    .validator
                    .is_reasonable(&candidate, expected_frames, major_len)
                {
                    tracing::debug!(
                        strategy = name,
                        priority = priority + 1,
                        confidence = format_args!("{:.2}", candidate.confidence),
                        frame_count = candidate.frame_spans.len(),
                        "Partitioning strategy accepted"
                    );
                    return candidate;
                }
                tracing::debug!(
                    strategy = name,
                    priority = priority + 1,
                    "Partition result rejected by physical validator"
                );
            }
        }

        // Guaranteed fallback via EvenSplitPartitioner
        EvenSplitPartitioner::new()
            .partition(profile, expected_frames)
            .unwrap_or_else(|| PartitionResult {
                strategy_name: "EvenSplitFallback",
                confidence: 0.05,
                frame_spans: Vec::new(),
                gutter_centers: Vec::new(),
                gutters: Vec::new(),
            })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::color::SimpleGrayConverter;
    use crate::luma::{ScaledGrayscaleStrip, PROJECTION_MAX_DIMENSION};
    use image::{Rgba, RgbaImage};

    #[test]
    fn test_prioritized_partition_engine_priority1() {
        // Create 300x100 strip with 3 frames and 2 gutters at x=100 and x=200
        let mut img = RgbaImage::from_pixel(300, 100, Rgba([0, 0, 0, 255]));
        for y in 0..100 {
            for x in 0..300 {
                if (96..=104).contains(&x) || (196..=204).contains(&x) {
                    img.put_pixel(x, y, Rgba([20, 20, 20, 255]));
                } else {
                    let val = if y % 2 == 0 { 240 } else { 40 };
                    img.put_pixel(x, y, Rgba([val, val, val, 255]));
                }
            }
        }

        let conv = SimpleGrayConverter::new();
        let strip =
            ScaledGrayscaleStrip::from_image(&img, &conv, PROJECTION_MAX_DIMENSION).unwrap();
        let profile = AxisStatisticsProfile::compute(&strip);

        let engine = PrioritizedPartitionEngine::standard();
        let result = engine.execute(&profile, 3);

        assert_eq!(result.strategy_name, "TopologicalThreshold");
        assert_eq!(result.frame_spans.len(), 3);
        assert!(result.confidence > 0.2);
    }
}
