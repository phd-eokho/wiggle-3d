//! Wiggle 3D animation generation, sub-frame alignment, and GIF assembly.
//!
//! Provides [`WiggleAligner`] to translate sub-frames relative to anchor frame median keypoints,
//! and [`WiggleGifBuilder`] to quantize and dither multi-frame pixel buffers using a unified
//! 256-color [`color_quant::NeuQuant`] palette into a smooth ping-pong GIF loop ($0 \to 1 \to 2 \to 1$).

use crate::error::{Error, Result};
use crate::feature::{FeatureFrame, FeatureMatch, FeatureTriplet, FramePair};
use color_quant::NeuQuant;
use image::codecs::gif::{GifEncoder, Repeat};
use image::imageops::colorops::ColorMap;
use image::{Delay, DynamicImage, Frame, Rgba, RgbaImage};
use std::io::Write;

/// Default frame delay in milliseconds (100ms = 10 fps).
pub const DEFAULT_FRAME_DELAY_MS: u32 = 100;

/// Default NeuQuant sample factor (1..=30, where 1 is highest quality and 10 is balanced).
pub const DEFAULT_NEUQUANT_SAMPLE_FAC: i32 = 10;

/// Default palette color count (256 colors maximum for GIF89a).
pub const DEFAULT_PALETTE_COLORS: usize = 256;

/// Configuration for Wiggle 3D GIF generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WiggleGifConfig {
    /// Inter-frame delay in milliseconds.
    pub delay_ms: u32,
    /// Sample factor for NeuQuant color quantization (1 = best, 30 = fastest).
    pub sample_factor: i32,
    /// Whether to apply Floyd-Steinberg error-diffusion dithering.
    pub dither: bool,
    /// Optional adaptive non-uniform delays `[delay_01_ms, delay_12_ms]` derived from physical extrinsics.
    pub adaptive_delays_ms: Option<[u32; 2]>,
}

impl Default for WiggleGifConfig {
    fn default() -> Self {
        Self {
            delay_ms: DEFAULT_FRAME_DELAY_MS,
            sample_factor: DEFAULT_NEUQUANT_SAMPLE_FAC,
            dither: true,
            adaptive_delays_ms: None,
        }
    }
}

impl WiggleGifConfig {
    /// Creates a new configuration with the specified delay in milliseconds.
    #[must_use]
    pub const fn new(delay_ms: u32) -> Self {
        Self {
            delay_ms,
            sample_factor: DEFAULT_NEUQUANT_SAMPLE_FAC,
            dither: true,
            adaptive_delays_ms: None,
        }
    }

    /// Sets the NeuQuant sample factor.
    #[must_use]
    pub const fn with_sample_factor(mut self, sample_factor: i32) -> Self {
        self.sample_factor = sample_factor;
        self
    }

    /// Enables or disables Floyd-Steinberg dithering.
    #[must_use]
    pub const fn with_dither(mut self, dither: bool) -> Self {
        self.dither = dither;
        self
    }

    /// Sets adaptive non-uniform frame delays `[delay_01_ms, delay_12_ms]`.
    #[must_use]
    pub const fn with_adaptive_delays(mut self, delays_ms: [u32; 2]) -> Self {
        self.adaptive_delays_ms = Some(delays_ms);
        self
    }
}

/// Default inverse disparity histogram bin size (1/px).
///
/// A step of 0.005 in 1/d provides fine resolution across typical stereo parallax ranges
/// (e.g. 5px to 50px maps to 0.02 to 0.20 in 1/d, spanning ~36 bins).
pub const DEFAULT_INV_DISPARITY_BIN_SIZE: f32 = 0.005;

/// Backward-compatible alias for default disparity bin size in pixels.
pub const DEFAULT_DISPARITY_BIN_SIZE_PX: f32 = DEFAULT_INV_DISPARITY_BIN_SIZE;

/// Default cluster grouping tolerance in 1/px units (0.0 means individual bins are not grouped across gaps).
pub const DEFAULT_CLUSTER_TOLERANCE_PX: f32 = 0.0;

/// Minimum disparity magnitude in pixels to be included in inverse disparity binning.
///
/// Disparities below this threshold ($d \le 0.5\,\text{px}$) correspond to optical infinity ($Z \to \infty$)
/// or non-parallax points, preventing infinite / undefined inverse depth values ($1/0$).
pub const MIN_DISPARITY_FOR_DEPTH_PX: f32 = 0.5;

/// Default minimum proportion of points required for a cluster to qualify as a valid foreground target surface (1% = 0.01).
pub const DEFAULT_FOREGROUND_MIN_PROPORTION: f32 = 0.01;

/// Record representing a single inverse disparity / relative depth histogram bin for debugging and visualization.
#[derive(Debug, Clone, PartialEq)]
pub struct DisparityBinRecord {
    /// Integer bin index in inverse disparity space.
    pub bin_idx: i32,
    /// Starting inverse disparity (1/px) of this bin.
    pub bin_start_px: f32,
    /// Ending inverse disparity (1/px) of this bin.
    pub bin_end_px: f32,
    /// Center inverse disparity (1/px) of this bin.
    pub bin_center_px: f32,
    /// Number of feature correspondences in this bin.
    pub count: usize,
    /// Normalized fraction (0.0 to 1.0) of total feature correspondences in this bin.
    pub fraction: f32,
    /// Percentage (0.0% to 100.0%) of total feature correspondences in this bin.
    pub percentage: f32,
    /// ID of the cluster/depth surface to which this bin was assigned, if any.
    pub cluster_id: Option<usize>,
    /// Whether this bin is part of the selected depth surface cluster.
    pub is_selected_surface: bool,
}

/// Detailed inverse disparity histogram and depth surface clustering debug data.
#[derive(Debug, Clone, PartialEq)]
pub struct DisparityHistogramData {
    /// Inverse disparity bin size (1/px).
    pub bin_size: f32,
    /// Cluster tolerance in inverse disparity units (1/px).
    pub cluster_tolerance: f32,
    /// Total number of correspondence samples evaluated (excluding optical infinity / zero disparity).
    pub total_samples: usize,
    /// Selected depth surface cluster index.
    pub selected_cluster_id: Option<usize>,
    /// Number of points in the selected cluster.
    pub selected_cluster_points: usize,
    /// Computed alignment shifts `[(dx0, dy0), (dx1, dy1), (dx2, dy2)]`.
    pub shifts: [(f32, f32); 3],
    /// All populated and unpopulated/contiguous histogram bins.
    pub bins: Vec<DisparityBinRecord>,
}

impl DisparityHistogramData {
    /// Serializes this histogram debug data to CSV format.
    ///
    /// # Arguments
    /// * `writer` - Target writer.
    ///
    /// # Errors
    /// Returns [`std::io::Error`] if writing fails.
    pub fn write_csv<W: Write>(&self, mut writer: W) -> std::io::Result<()> {
        writeln!(
            writer,
            "# metric=inv_disparity,bin_size={:.4},cluster_tolerance={:.4},total_samples={},selected_cluster={:?},selected_points={}",
            self.bin_size,
            self.cluster_tolerance,
            self.total_samples,
            self.selected_cluster_id,
            self.selected_cluster_points
        )?;
        writeln!(
            writer,
            "# shift_0=({:.3},{:.3}),shift_1=({:.3},{:.3}),shift_2=({:.3},{:.3})",
            self.shifts[0].0,
            self.shifts[0].1,
            self.shifts[1].0,
            self.shifts[1].1,
            self.shifts[2].0,
            self.shifts[2].1
        )?;
        writeln!(
            writer,
            "bin_idx,bin_start_px,bin_end_px,bin_center_px,count,fraction,percentage,cluster_id,is_selected_surface"
        )?;
        for b in &self.bins {
            let cluster_str = b.cluster_id.map_or_else(String::new, |id| id.to_string());
            writeln!(
                writer,
                "{},{:.4},{:.4},{:.4},{},{:.4},{:.2},{},{}",
                b.bin_idx,
                b.bin_start_px,
                b.bin_end_px,
                b.bin_center_px,
                b.count,
                b.fraction,
                b.percentage,
                cluster_str,
                i32::from(b.is_selected_surface)
            )?;
        }
        Ok(())
    }
}

/// Disparity histogram binning and depth surface clustering result.
#[derive(Debug)]
struct HistogramClusteringResult {
    bin_map: std::collections::BTreeMap<i32, Vec<usize>>,
    populated_bins: Vec<i32>,
    clusters: Vec<Vec<i32>>,
    target_cluster_idx: usize,
}

impl HistogramClusteringResult {
    #[allow(
        clippy::too_many_lines,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn compute<T, F>(
        items: &[T],
        get_inv_disp: F,
        effective_bin_size: f32,
        effective_tolerance: f32,
    ) -> Option<Self>
    where
        F: Fn(&T) -> f32,
    {
        if items.is_empty() {
            return None;
        }

        let mut bin_map: std::collections::BTreeMap<i32, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (idx, item) in items.iter().enumerate() {
            let bin_idx = (get_inv_disp(item) / effective_bin_size).floor() as i32;
            bin_map.entry(bin_idx).or_default().push(idx);
        }

        let populated_bins: Vec<i32> = bin_map.keys().copied().collect();
        if populated_bins.is_empty() {
            return None;
        }

        let clusters = if effective_tolerance <= 0.0 {
            populated_bins.iter().map(|&b| vec![b]).collect()
        } else {
            let max_bin_gap = (effective_tolerance / effective_bin_size).ceil().max(1.0) as i32;
            let mut clusters: Vec<Vec<i32>> = Vec::new();
            let mut current_cluster: Vec<i32> = Vec::new();

            for &b in &populated_bins {
                if let Some(&last_b) = current_cluster.last() {
                    if (b - last_b).abs() <= max_bin_gap {
                        current_cluster.push(b);
                    } else {
                        clusters.push(std::mem::replace(&mut current_cluster, vec![b]));
                    }
                } else {
                    current_cluster.push(b);
                }
            }
            if !current_cluster.is_empty() {
                clusters.push(current_cluster);
            }
            clusters
        };

        let total_pts = items.len();
        let min_points_threshold = ((total_pts as f32) * DEFAULT_FOREGROUND_MIN_PROPORTION)
            .ceil()
            .max(1.0) as usize;

        let first_foreground_bin = populated_bins
            .iter()
            .copied()
            .find(|b| bin_map.get(b).map_or(0, Vec::len) >= min_points_threshold);

        let target_bin = first_foreground_bin.map(|b_0| {
            let mut best_bin = b_0;
            let mut best_count = bin_map.get(&b_0).map_or(0, Vec::len);
            let mut curr_bin = b_0;

            loop {
                let next_bin = curr_bin + 1;
                let next_count = bin_map.get(&next_bin).map_or(0, Vec::len);
                if next_count == 0 {
                    break;
                }
                if next_count >= best_count {
                    best_bin = next_bin;
                    best_count = next_count;
                    curr_bin = next_bin;
                } else {
                    break;
                }
            }
            best_bin
        });

        let target_cluster_idx = if clusters.len() == 1 {
            0
        } else if let Some(bin) = target_bin {
            clusters
                .iter()
                .enumerate()
                .find(|(_, c)| c.contains(&bin))
                .map_or_else(
                    || {
                        clusters
                            .iter()
                            .enumerate()
                            .max_by_key(|(_, c)| {
                                c.iter()
                                    .filter_map(|b| bin_map.get(b))
                                    .map(Vec::len)
                                    .sum::<usize>()
                            })
                            .map_or(0, |(i, _)| i)
                    },
                    |(i, _)| i,
                )
        } else {
            clusters
                .iter()
                .enumerate()
                .max_by_key(|(_, c)| {
                    c.iter()
                        .filter_map(|b| bin_map.get(b))
                        .map(Vec::len)
                        .sum::<usize>()
                })
                .map_or(0, |(i, _)| i)
        };

        Some(Self {
            bin_map,
            populated_bins,
            clusters,
            target_cluster_idx,
        })
    }
}

/// Aligns sub-frames to an anchor frame based on depth surface correspondence shifts.
pub struct WiggleAligner;

impl WiggleAligner {
    /// Computes translation shifts $(\delta x_i, \delta y_i)$ relative to anchor frame 1 (center)
    /// by grouping verified triplets into depth surface clusters via disparity histogram binning.
    ///
    /// Rather than a simple median across all keypoints (which is heavily biased toward
    /// background objects where keypoints are disproportionately dense), this algorithm:
    /// 1. Computes the Euclidean disparity magnitude $D_k = \sqrt{\Delta x_k^2 + \Delta y_k^2}$ for each triplet.
    /// 2. Bins the disparities into a histogram to discover populated depth levels.
    /// 3. Groups adjacent populated bins within `cluster_tolerance` into discrete depth surfaces.
    /// 4. Identifies the primary salient foreground depth surface cluster (avoiding background bias).
    /// 5. Averages the translation shifts $(\Delta x_i, \Delta y_i)$ exclusively among triplets
    ///    belonging to this selected depth surface cluster.
    ///
    /// # Arguments
    /// * `features` - Extracted feature frames across views.
    /// * `triplets` - Verified feature triplets across frames 0, 1, and 2.
    /// * `bin_size` - Disparity histogram bin width in pixels (e.g. 1.0).
    /// * `cluster_tolerance` - Maximum gap between bins to belong to the same depth surface.
    ///
    /// Computes translation shifts $(\delta x_i, \delta y_i)$ relative to anchor frame 1 (center)
    /// with dominant portrait face priority (Tier 1), falling back to disparity histogram clustering (Tier 2).
    ///
    /// # Arguments
    /// * `features` - Extracted feature frames across views.
    /// * `triplets` - Verified feature triplets across frames 0, 1, and 2.
    /// * `dominant_face_bbox` - Optional normalized bounding box of the dominant face on anchor frame 1.
    /// * `bin_size` - Disparity histogram bin width in pixels.
    /// * `cluster_tolerance` - Maximum gap between bins to belong to the same depth surface.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::imprecise_flops,
        clippy::suboptimal_flops,
        clippy::similar_names
    )]
    pub fn compute_depth_surface_shifts_from_triplets_with_face_priority(
        features: &[FeatureFrame],
        triplets: &[FeatureTriplet],
        dominant_face_bbox: Option<crate::geom::NormalizedRect>,
        bin_size: f32,
        cluster_tolerance: f32,
    ) -> [(f32, f32); 3] {
        if let Some(face_bbox) = dominant_face_bbox {
            if features.len() >= 3 && !triplets.is_empty() {
                let frame1_w = features[1].image_size.width as f32;
                let frame1_h = features[1].image_size.height as f32;

                if frame1_w > 0.0 && frame1_h > 0.0 {
                    let mut accum = [0.0_f32; 4]; // [sum_dx_01, sum_dy_01, sum_dx_21, sum_dy_21]
                    let mut face_displacements: Option<Vec<[f32; 4]>> =
                        if tracing::enabled!(tracing::Level::DEBUG) {
                            Some(Vec::new())
                        } else {
                            None
                        };
                    let mut count = 0usize;

                    for t in triplets {
                        if t.index_0 < features[0].keypoints.len()
                            && t.index_1 < features[1].keypoints.len()
                            && t.index_2 < features[2].keypoints.len()
                        {
                            let p1 = features[1].keypoints[t.index_1].point;
                            let norm_x = p1.x / frame1_w;
                            let norm_y = p1.y / frame1_h;

                            if face_bbox.contains_point(norm_x, norm_y) {
                                let p0 = features[0].keypoints[t.index_0].point;
                                let p2 = features[2].keypoints[t.index_2].point;
                                let d = [p1.x - p0.x, p1.y - p0.y, p1.x - p2.x, p1.y - p2.y];
                                accum[0] += d[0];
                                accum[1] += d[1];
                                accum[2] += d[2];
                                accum[3] += d[3];
                                count += 1;
                                if let Some(ref mut disps) = face_displacements {
                                    disps.push(d);
                                }
                            }
                        }
                    }

                    if count > 0 {
                        let inv_count = 1.0 / (count as f32);
                        let avg_dx_01 = accum[0] * inv_count;
                        let avg_dy_01 = accum[1] * inv_count;
                        let avg_dx_21 = accum[2] * inv_count;
                        let avg_dy_21 = accum[3] * inv_count;

                        tracing::info!(
                            triplet_points = count,
                            shift_0 = ?(avg_dx_01, avg_dy_01),
                            shift_2 = ?(avg_dx_21, avg_dy_21),
                            "Locked parallax focal plane onto dominant portrait face (Tier 1 Face Priority)"
                        );

                        if let Some(disps) = face_displacements {
                            let mut var_accum = [0.0_f32; 4];
                            for d in &disps {
                                let diff = [
                                    d[0] - avg_dx_01,
                                    d[1] - avg_dy_01,
                                    d[2] - avg_dx_21,
                                    d[3] - avg_dy_21,
                                ];
                                var_accum[0] += diff[0] * diff[0];
                                var_accum[1] += diff[1] * diff[1];
                                var_accum[2] += diff[2] * diff[2];
                                var_accum[3] += diff[3] * diff[3];
                            }
                            let std_shift_0_px = ((var_accum[0] + var_accum[1]) * inv_count).sqrt();
                            let std_shift_2_px = ((var_accum[2] + var_accum[3]) * inv_count).sqrt();

                            tracing::debug!(
                                triplet_points = count,
                                std_shift_0_px,
                                std_shift_2_px,
                                "Portrait face alignment focal plane dispersion residuals"
                            );
                        }

                        return [(avg_dx_01, avg_dy_01), (0.0, 0.0), (avg_dx_21, avg_dy_21)];
                    }
                }
            }
        }

        Self::compute_depth_surface_shifts_from_triplets(
            features,
            triplets,
            bin_size,
            cluster_tolerance,
        )
    }

    /// Computes translation shifts $(\delta x_i, \delta y_i)$ relative to anchor frame 1 (center)
    /// by grouping verified triplets into depth surface clusters via disparity histogram binning.
    ///
    /// Rather than a simple median across all keypoints (which is heavily biased toward
    /// background objects where keypoints are disproportionately dense), this algorithm:
    /// 1. Computes the Euclidean disparity magnitude $D_k = \sqrt{\Delta x_k^2 + \Delta y_k^2}$ for each triplet.
    /// 2. Bins the disparities into a histogram to discover populated depth levels.
    /// 3. Groups adjacent populated bins within `cluster_tolerance` into discrete depth surfaces.
    /// 4. Identifies the primary salient foreground depth surface cluster (avoiding background bias).
    /// 5. Averages the translation shifts $(\Delta x_i, \Delta y_i)$ exclusively among triplets
    ///    belonging to this selected depth surface cluster.
    ///
    /// # Arguments
    /// * `features` - Extracted feature frames across views.
    /// * `triplets` - Verified feature triplets across frames 0, 1, and 2.
    /// * `bin_size` - Disparity histogram bin width in pixels (e.g. 1.0).
    /// * `cluster_tolerance` - Maximum gap between bins to belong to the same depth surface.
    ///
    /// # Returns
    /// Array of `[(dx0, dy0), (dx1, dy1), (dx2, dy2)]` shifts where anchor frame 1 has `(0, 0)`.
    #[must_use]
    pub fn compute_depth_surface_shifts_from_triplets(
        features: &[FeatureFrame],
        triplets: &[FeatureTriplet],
        bin_size: f32,
        cluster_tolerance: f32,
    ) -> [(f32, f32); 3] {
        Self::compute_depth_surface_shifts_from_triplets_with_debug(
            features,
            triplets,
            bin_size,
            cluster_tolerance,
        )
        .0
    }
    /// Computes translation shifts $(\delta x_i, \delta y_i)$ relative to anchor frame 1 (center)
    /// along with detailed [`DisparityHistogramData`] for visual debugging and distribution inspection.
    ///
    /// # Arguments
    /// * `features` - Extracted feature frames across views.
    /// * `triplets` - Verified feature triplets across frames 0, 1, and 2.
    /// * `bin_size` - Disparity histogram bin width in pixels (e.g. 1.0).
    /// * `cluster_tolerance` - Maximum gap between bins to belong to the same depth surface.
    ///
    /// # Returns
    /// Tuple of `([(dx0, dy0), (dx1, dy1), (dx2, dy2)], DisparityHistogramData)`.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::too_many_lines,
        clippy::imprecise_flops,
        clippy::suboptimal_flops,
        clippy::redundant_closure_for_method_calls,
        clippy::single_match_else,
        clippy::map_unwrap_or,
        clippy::items_after_statements,
        clippy::similar_names
    )]
    pub fn compute_depth_surface_shifts_from_triplets_with_debug(
        features: &[FeatureFrame],
        triplets: &[FeatureTriplet],
        bin_size: f32,
        cluster_tolerance: f32,
    ) -> ([(f32, f32); 3], DisparityHistogramData) {
        let effective_bin_size = if bin_size > 1e-4 {
            bin_size
        } else {
            DEFAULT_DISPARITY_BIN_SIZE_PX
        };
        let effective_tolerance = if cluster_tolerance > 1e-4 {
            cluster_tolerance
        } else {
            DEFAULT_CLUSTER_TOLERANCE_PX
        };

        let empty_debug = DisparityHistogramData {
            bin_size: effective_bin_size,
            cluster_tolerance: effective_tolerance,
            total_samples: 0,
            selected_cluster_id: None,
            selected_cluster_points: 0,
            shifts: [(0.0, 0.0); 3],
            bins: Vec::new(),
        };

        if features.len() < 3 || triplets.is_empty() {
            return ([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)], empty_debug);
        }

        // Collect per-triplet displacement data, filtering out zero/near-zero disparity (optical infinity / d <= MIN_DISPARITY_FOR_DEPTH_PX)
        struct TripletDisplacement {
            inv_disp: f32,
            dx_01: f32,
            dy_01: f32,
            dx_21: f32,
            dy_21: f32,
        }

        let mut data = Vec::with_capacity(triplets.len());

        for t in triplets {
            if t.index_0 < features[0].keypoints.len()
                && t.index_1 < features[1].keypoints.len()
                && t.index_2 < features[2].keypoints.len()
            {
                let p0 = features[0].keypoints[t.index_0].point;
                let p1 = features[1].keypoints[t.index_1].point;
                let p2 = features[2].keypoints[t.index_2].point;

                let dx_01 = p1.x - p0.x;
                let dy_01 = p1.y - p0.y;
                let dx_21 = p1.x - p2.x;
                let dy_21 = p1.y - p2.y;

                // Combined average disparity magnitude across baselines 0-1 and 2-1
                let mag_01 = (dx_01 * dx_01 + dy_01 * dy_01).sqrt();
                let mag_21 = (dx_21 * dx_21 + dy_21 * dy_21).sqrt();
                let disp_mag = f32::midpoint(mag_01, mag_21);

                // Exclude 0 or near-zero disparity values to avoid infinite / undefined depth values (1/d)
                if disp_mag > MIN_DISPARITY_FOR_DEPTH_PX {
                    let inv_disp = 1.0 / disp_mag;
                    data.push(TripletDisplacement {
                        inv_disp,
                        dx_01,
                        dy_01,
                        dx_21,
                        dy_21,
                    });
                }
            }
        }

        let Some(clustering) = HistogramClusteringResult::compute(
            &data,
            |item| item.inv_disp,
            effective_bin_size,
            effective_tolerance,
        ) else {
            return ([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)], empty_debug);
        };

        let bin_map = clustering.bin_map;
        let populated_bins = clustering.populated_bins;
        let clusters = clustering.clusters;
        let target_cluster_idx = clustering.target_cluster_idx;
        let target_cluster = &clusters[target_cluster_idx];

        // 5. Gather all triplet indices in the selected depth surface cluster
        let mut cluster_triplet_indices = Vec::new();
        for &b in target_cluster {
            if let Some(indices) = bin_map.get(&b) {
                cluster_triplet_indices.extend_from_slice(indices);
            }
        }

        if cluster_triplet_indices.is_empty() {
            return ([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)], empty_debug);
        }

        // 6. Compute average translation shifts for the depth surface set
        let count = cluster_triplet_indices.len() as f32;
        let mut accum = [0.0_f32; 4]; // [sum_01_x, sum_01_y, sum_21_x, sum_21_y]

        for &idx in &cluster_triplet_indices {
            let item = &data[idx];
            accum[0] += item.dx_01;
            accum[1] += item.dy_01;
            accum[2] += item.dx_21;
            accum[3] += item.dy_21;
        }

        let inv_count = 1.0 / count;
        let shift_0 = (accum[0] * inv_count, accum[1] * inv_count);
        let shift_1 = (0.0, 0.0);
        let shift_2 = (accum[2] * inv_count, accum[3] * inv_count);
        let shifts = [shift_0, shift_1, shift_2];

        // 7. Construct complete DisparityHistogramData records
        let min_bin = *populated_bins.first().unwrap_or(&0);
        let max_bin = *populated_bins.last().unwrap_or(&0);

        let mut bin_to_cluster: std::collections::HashMap<i32, usize> =
            std::collections::HashMap::new();
        for (c_idx, c) in clusters.iter().enumerate() {
            for &b in c {
                bin_to_cluster.insert(b, c_idx);
            }
        }

        let total_samples = data.len();
        let mut bin_records = Vec::new();
        for b in min_bin..=max_bin {
            let cnt = bin_map.get(&b).map_or(0, |v| v.len());
            let fraction = if total_samples > 0 {
                cnt as f32 / total_samples as f32
            } else {
                0.0
            };
            let percentage = fraction * 100.0;
            let cluster_id = bin_to_cluster.get(&b).copied();
            let is_selected_surface = cluster_id == Some(target_cluster_idx);
            let bin_start_px = b as f32 * effective_bin_size;
            let bin_end_px = (b + 1) as f32 * effective_bin_size;
            let bin_center_px = bin_start_px + 0.5 * effective_bin_size;

            bin_records.push(DisparityBinRecord {
                bin_idx: b,
                bin_start_px,
                bin_end_px,
                bin_center_px,
                count: cnt,
                fraction,
                percentage,
                cluster_id,
                is_selected_surface,
            });
        }

        let debug_data = DisparityHistogramData {
            bin_size: effective_bin_size,
            cluster_tolerance: effective_tolerance,
            total_samples: data.len(),
            selected_cluster_id: Some(target_cluster_idx),
            selected_cluster_points: cluster_triplet_indices.len(),
            shifts,
            bins: bin_records,
        };

        tracing::info!(
            surface_points = cluster_triplet_indices.len(),
            total_triplets = data.len(),
            shift_0 = ?shift_0,
            shift_2 = ?shift_2,
            "Calculated depth surface cluster baseline alignment shifts"
        );

        if tracing::enabled!(tracing::Level::DEBUG) && !cluster_triplet_indices.is_empty() {
            let mut var_accum = [0.0_f32; 4];
            for &idx in &cluster_triplet_indices {
                let d = &data[idx];
                let diff = [
                    d.dx_01 - shift_0.0,
                    d.dy_01 - shift_0.1,
                    d.dx_21 - shift_2.0,
                    d.dy_21 - shift_2.1,
                ];
                var_accum[0] += diff[0] * diff[0];
                var_accum[1] += diff[1] * diff[1];
                var_accum[2] += diff[2] * diff[2];
                var_accum[3] += diff[3] * diff[3];
            }
            let std_shift_0_px = ((var_accum[0] + var_accum[1]) * inv_count).sqrt();
            let std_shift_2_px = ((var_accum[2] + var_accum[3]) * inv_count).sqrt();
            tracing::debug!(
                surface_points = cluster_triplet_indices.len(),
                std_shift_0_px,
                std_shift_2_px,
                "Depth surface alignment focal plane dispersion residuals"
            );
        }

        (shifts, debug_data)
    }

    /// Computes translation shifts from pairwise matches with dominant portrait face priority (Tier 1),
    /// falling back to disparity histogram depth clustering (Tier 2).
    ///
    /// # Arguments
    /// * `features` - Extracted feature frames.
    /// * `pairwise_matches` - Pairwise match lists keyed by frame pair `(a, b)`.
    /// * `dominant_face_bbox` - Optional normalized bounding box of the dominant face on anchor frame 1.
    /// * `bin_size` - Disparity histogram bin width in pixels.
    /// * `cluster_tolerance` - Maximum gap between bins.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::imprecise_flops,
        clippy::suboptimal_flops,
        clippy::similar_names
    )]
    pub fn compute_depth_surface_shifts_from_pairs_with_face_priority(
        features: &[FeatureFrame],
        pairwise_matches: &[(FramePair, Vec<FeatureMatch>)],
        dominant_face_bbox: Option<crate::geom::NormalizedRect>,
        bin_size: f32,
        cluster_tolerance: f32,
    ) -> [(f32, f32); 3] {
        if let Some(face_bbox) = dominant_face_bbox {
            if features.len() >= 3 {
                let frame1_w = features[1].image_size.width as f32;
                let frame1_h = features[1].image_size.height as f32;

                if frame1_w > 0.0 && frame1_h > 0.0 {
                    let mut shift_0 = None;
                    let mut shift_2 = None;

                    for &((src, r#ref), ref matches) in pairwise_matches {
                        if r#ref == 1 && (src == 0 || src == 2) {
                            let mut dx_list = Vec::new();
                            let mut dy_list = Vec::new();

                            for m in matches {
                                if m.index_a < features[src].keypoints.len()
                                    && m.index_b < features[1].keypoints.len()
                                {
                                    let p_ref = features[1].keypoints[m.index_b].point;
                                    let norm_x = p_ref.x / frame1_w;
                                    let norm_y = p_ref.y / frame1_h;

                                    if face_bbox.contains_point(norm_x, norm_y) {
                                        let p_src = features[src].keypoints[m.index_a].point;
                                        dx_list.push(p_ref.x - p_src.x);
                                        dy_list.push(p_ref.y - p_src.y);
                                    }
                                }
                            }

                            if !dx_list.is_empty() {
                                let count = dx_list.len() as f32;
                                let avg_dx: f32 = dx_list.iter().sum::<f32>() / count;
                                let avg_dy: f32 = dy_list.iter().sum::<f32>() / count;
                                if src == 0 {
                                    shift_0 = Some((avg_dx, avg_dy));
                                } else {
                                    shift_2 = Some((avg_dx, avg_dy));
                                }
                            }
                        }
                    }

                    if let (Some(s0), Some(s2)) = (shift_0, shift_2) {
                        tracing::info!(
                            shift_0 = ?s0,
                            shift_2 = ?s2,
                            "Locked pairwise parallax focal plane onto dominant portrait face (Tier 1 Face Priority)"
                        );
                        return [s0, (0.0, 0.0), s2];
                    }
                }
            }
        }

        Self::compute_depth_surface_shifts_from_pairs(
            features,
            pairwise_matches,
            bin_size,
            cluster_tolerance,
        )
    }

    /// Computes translation shifts from pairwise match lists if triplets are unavailable
    /// using depth surface clustering.
    ///
    /// # Arguments
    /// * `features` - Extracted feature frames.
    /// * `pairwise_matches` - Pairwise match lists keyed by frame pair `(a, b)`.
    /// * `bin_size` - Disparity histogram bin width in pixels.
    /// * `cluster_tolerance` - Maximum gap between bins to belong to the same depth surface.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::too_many_lines,
        clippy::imprecise_flops,
        clippy::suboptimal_flops,
        clippy::redundant_closure_for_method_calls,
        clippy::single_match_else,
        clippy::map_unwrap_or,
        clippy::items_after_statements
    )]
    pub fn compute_depth_surface_shifts_from_pairs(
        features: &[FeatureFrame],
        pairwise_matches: &[(FramePair, Vec<FeatureMatch>)],
        bin_size: f32,
        cluster_tolerance: f32,
    ) -> [(f32, f32); 3] {
        if features.len() < 3 {
            return [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
        }

        let effective_bin_size = if bin_size > 1e-4 {
            bin_size
        } else {
            DEFAULT_DISPARITY_BIN_SIZE_PX
        };
        let effective_tolerance = if cluster_tolerance > 1e-4 {
            cluster_tolerance
        } else {
            DEFAULT_CLUSTER_TOLERANCE_PX
        };

        let calculate_pair_surface_shift = |pair_matches: &[FeatureMatch],
                                            f_src: &FeatureFrame,
                                            f_ref: &FeatureFrame|
         -> (f32, f32) {
            struct MatchDisp {
                dx: f32,
                dy: f32,
                inv_disp: f32,
            }
            let mut list = Vec::with_capacity(pair_matches.len());
            for m in pair_matches {
                if m.index_a < f_src.keypoints.len() && m.index_b < f_ref.keypoints.len() {
                    let p_src = f_src.keypoints[m.index_a].point;
                    let p_ref = f_ref.keypoints[m.index_b].point;
                    let dx = p_ref.x - p_src.x;
                    let dy = p_ref.y - p_src.y;
                    let mag = (dx * dx + dy * dy).sqrt();
                    if mag > MIN_DISPARITY_FOR_DEPTH_PX {
                        let inv_disp = 1.0 / mag;
                        list.push(MatchDisp { dx, dy, inv_disp });
                    }
                }
            }

            let Some(clustering) = HistogramClusteringResult::compute(
                &list,
                |item| item.inv_disp,
                effective_bin_size,
                effective_tolerance,
            ) else {
                return (0.0, 0.0);
            };

            let bin_map = clustering.bin_map;
            let target_cluster = &clustering.clusters[clustering.target_cluster_idx];

            let mut cluster_indices = Vec::new();
            for &b in target_cluster {
                if let Some(indices) = bin_map.get(&b) {
                    cluster_indices.extend_from_slice(indices);
                }
            }

            if cluster_indices.is_empty() {
                return (0.0, 0.0);
            }

            let count = cluster_indices.len() as f32;
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            for &idx in &cluster_indices {
                sum_x += list[idx].dx;
                sum_y += list[idx].dy;
            }

            (sum_x / count, sum_y / count)
        };

        let mut shift_0 = (0.0, 0.0);
        let mut shift_2 = (0.0, 0.0);

        for ((idx_a, idx_b), matches) in pairwise_matches {
            if *idx_a == 0 && *idx_b == 1 && !matches.is_empty() {
                shift_0 = calculate_pair_surface_shift(matches, &features[0], &features[1]);
            } else if *idx_a == 1 && *idx_b == 2 && !matches.is_empty() {
                shift_2 = calculate_pair_surface_shift(matches, &features[2], &features[1]);
            }
        }

        [shift_0, (0.0, 0.0), shift_2]
    }

    /// Aligns extracted sub-frame images by applying calculated translation shifts and cropping
    /// to their shared intersection rectangle.
    ///
    /// # Arguments
    /// * `frames` - Slice of sub-frame images (must contain at least 2 images, typically 3).
    /// * `shifts` - Slice of `(dx, dy)` shift tuples corresponding to each frame.
    ///
    /// # Errors
    /// Returns [`Error`] if frames list is empty or dimensions do not permit intersection.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        clippy::cast_possible_wrap
    )]
    pub fn align_and_crop(
        frames: &[DynamicImage],
        shifts: &[(f32, f32)],
    ) -> Result<Vec<RgbaImage>> {
        if frames.is_empty() {
            return Err(Error::Unknown("No frames provided for alignment".into()));
        }

        let n = frames.len();
        let rgba_frames: Vec<RgbaImage> = frames.iter().map(DynamicImage::to_rgba8).collect();

        // Round shifts to nearest integer pixels
        let int_shifts: Vec<(i32, i32)> = shifts
            .iter()
            .take(n)
            .map(|&(dx, dy)| (dx.round() as i32, dy.round() as i32))
            .collect();

        // Determine bounding boxes in anchor frame 1 coordinates
        // Anchor frame coordinate space: frame i is at position (-shift_x, -shift_y)
        let mut min_x = 0i32;
        let mut min_y = 0i32;
        let mut max_x = i32::MAX;
        let mut max_y = i32::MAX;

        for (i, frame) in rgba_frames.iter().enumerate() {
            let (w, h) = frame.dimensions();
            let (dx, dy) = int_shifts.get(i).copied().unwrap_or((0, 0));

            // In frame i space: pixel (px, py) corresponds to (px + dx, py + dy) in anchor space
            // Frame i covers anchor space [dx, dx + w) x [dy, dy + h)
            min_x = min_x.max(dx);
            min_y = min_y.max(dy);
            max_x = max_x.min(dx + w as i32);
            max_y = max_y.min(dy + h as i32);
        }

        let crop_w = (max_x - min_x).max(0) as u32;
        let crop_h = (max_y - min_y).max(0) as u32;

        if crop_w == 0 || crop_h == 0 {
            return Err(Error::Unknown(
                "Intersection bounding box is empty after alignment shifts".into(),
            ));
        }

        tracing::info!(
            crop_width = crop_w,
            crop_height = crop_h,
            min_x,
            min_y,
            "Cropping aligned frames to mutual intersection canvas"
        );

        let mut aligned_crops = Vec::with_capacity(n);
        for (i, frame) in rgba_frames.iter().enumerate() {
            let (dx, dy) = int_shifts.get(i).copied().unwrap_or((0, 0));
            // In frame i's local coordinates, the intersection starts at (min_x - dx, min_y - dy)
            let local_src_x = (min_x - dx).max(0) as u32;
            let local_src_y = (min_y - dy).max(0) as u32;

            let crop = image::imageops::crop_imm(frame, local_src_x, local_src_y, crop_w, crop_h)
                .to_image();
            aligned_crops.push(crop);
        }

        Ok(aligned_crops)
    }
}

/// Unified `ColorMap` wrapper implementing [`image::imageops::colorops::ColorMap`] for [`NeuQuant`].
pub struct NeuQuantColorMap {
    quant: NeuQuant,
}

impl NeuQuantColorMap {
    /// Creates a new `NeuQuantColorMap` from trained `NeuQuant`.
    #[must_use]
    pub const fn new(quant: NeuQuant) -> Self {
        Self { quant }
    }
}

impl ColorMap for NeuQuantColorMap {
    type Color = Rgba<u8>;

    #[inline]
    fn index_of(&self, color: &Self::Color) -> usize {
        self.quant.index_of(&color.0)
    }

    #[inline]
    fn map_color(&self, color: &mut Self::Color) {
        self.quant.map_pixel(&mut color.0);
    }

    #[inline]
    fn lookup(&self, index: usize) -> Option<Self::Color> {
        self.quant.lookup(index).map(Rgba)
    }

    #[inline]
    fn has_lookup(&self) -> bool {
        true
    }
}

/// Wiggle 3D GIF builder with global multi-frame palette quantization and dithering.
///
/// # TODO (Modern Video Container & Codec Support)
/// GIF format is constrained to an 8-bit indexed palette (256 colors) and large file sizes.
/// Add native video export targets: MP4 (H.264 / H.265 / AV1) and WebM (VP9 / AV1) with full 24-bit
/// RGB/RGBA true color depth and significantly reduced file sizes, as well as Animated PNG (APNG)
/// for lossless 24-bit animated web presentation without color quantization artifacts.
pub struct WiggleGifBuilder;

impl WiggleGifBuilder {
    /// Trains a global unified 256-color palette across all provided frames.
    ///
    /// Concatenates pixel reservoirs from all frames to ensure temporal color stability
    /// and prevent frame-to-frame palette flickering during playback.
    ///
    /// # Arguments
    /// * `frames` - Slices of RGBA image frames.
    /// * `sample_factor` - `NeuQuant` sample factor (1..=30).
    #[must_use]
    pub fn build_unified_palette(frames: &[RgbaImage], sample_factor: i32) -> NeuQuant {
        let total_pixels: usize = frames.iter().map(|f| f.len()).sum();
        let mut reservoir = Vec::with_capacity(total_pixels);

        for frame in frames {
            reservoir.extend_from_slice(frame.as_raw());
        }

        NeuQuant::new(sample_factor, DEFAULT_PALETTE_COLORS, &reservoir)
    }

    /// Assembles an animated Wiggle GIF from aligned frames and writes it into the target stream.
    ///
    /// Constructs the canonical ping-pong loop sequence: $0 \to 1 \to 2 \to 1 \to \text{loop}$.
    ///
    /// # Arguments
    /// * `frames` - Slices of aligned RGBA sub-frames (must contain at least 2 frames).
    /// * `config` - Wiggle GIF rendering configuration.
    /// * `writer` - Target output stream implementing [`std::io::Write`].
    ///
    /// # Errors
    /// Returns [`Error`] if frames list is empty, image dimensions differ, or encoding fails.
    pub fn build_wiggle_gif<W: Write>(
        frames: &[RgbaImage],
        config: &WiggleGifConfig,
        writer: &mut W,
    ) -> Result<()> {
        if frames.is_empty() {
            return Err(Error::Unknown("No frames provided for Wiggle GIF".into()));
        }

        let (width, height) = frames[0].dimensions();
        for frame in frames {
            if frame.dimensions() != (width, height) {
                return Err(Error::Unknown(
                    "All frames must have identical dimensions for Wiggle GIF assembly".into(),
                ));
            }
        }

        // 1. Quantize frames with unified color palette
        let colormap =
            NeuQuantColorMap::new(Self::build_unified_palette(frames, config.sample_factor));

        // 2. Prepare ping-pong animation sequence indices
        // For 3 frames [0, 1, 2], sequence is [0, 1, 2, 1]
        let loop_indices: Vec<usize> = if frames.len() == 3 {
            vec![0, 1, 2, 1]
        } else if frames.len() == 2 {
            vec![0, 1]
        } else {
            (0..frames.len()).collect()
        };

        let default_delay = Delay::from_numer_denom_ms(config.delay_ms, 1);
        let mut gif_frames = Vec::with_capacity(loop_indices.len());

        for (step_idx, &frame_idx) in loop_indices.iter().enumerate() {
            let mut buf = frames[frame_idx].clone();
            if config.dither {
                image::imageops::colorops::dither(&mut buf, &colormap);
            } else {
                for pixel in buf.pixels_mut() {
                    colormap.map_color(pixel);
                }
            }

            let frame_delay = if let Some([d01, d12]) = config.adaptive_delays_ms {
                // Ping-pong step: 0 (0->1: d01), 1 (1->2: d12), 2 (2->1: d12), 3 (1->0: d01)
                let delay_ms = match step_idx {
                    0 | 3 => d01,
                    1 | 2 => d12,
                    _ => config.delay_ms,
                };
                Delay::from_numer_denom_ms(delay_ms, 1)
            } else {
                default_delay
            };

            let frame = Frame::from_parts(buf, 0, 0, frame_delay);
            gif_frames.push(frame);
        }

        // 3. Encode frames into animated GIF
        let mut encoder = GifEncoder::new(writer);
        encoder
            .set_repeat(Repeat::Infinite)
            .map_err(|e| Error::Unknown(format!("Failed to set GIF repeat: {e}")))?;

        encoder
            .encode_frames(gif_frames)
            .map_err(|e| Error::Unknown(format!("Failed to encode GIF frames: {e}")))?;

        tracing::info!(
            frames_encoded = loop_indices.len(),
            delay_ms = config.delay_ms,
            "Successfully assembled and encoded Wiggle GIF"
        );

        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::feature::KeyPoint;
    use crate::geom::{NormalizedRect, Point2D, Size2D};

    #[test]
    fn test_compute_depth_surface_shifts_from_triplets() {
        let f0 = FeatureFrame::new(
            0,
            NormalizedRect::new(0.0, 0.0, 0.3, 1.0).unwrap(),
            Size2D::new(100, 100),
            vec![
                KeyPoint::new(Point2D::new(10.0, 20.0), 1.0, None),
                KeyPoint::new(Point2D::new(30.0, 40.0), 1.0, None),
            ],
        );
        let f1 = FeatureFrame::new(
            1,
            NormalizedRect::new(0.3, 0.0, 0.3, 1.0).unwrap(),
            Size2D::new(100, 100),
            vec![
                KeyPoint::new(Point2D::new(15.0, 22.0), 1.0, None),
                KeyPoint::new(Point2D::new(35.0, 42.0), 1.0, None),
            ],
        );
        let f2 = FeatureFrame::new(
            2,
            NormalizedRect::new(0.6, 0.0, 0.3, 1.0).unwrap(),
            Size2D::new(100, 100),
            vec![
                KeyPoint::new(Point2D::new(20.0, 24.0), 1.0, None),
                KeyPoint::new(Point2D::new(40.0, 44.0), 1.0, None),
            ],
        );

        let triplets = vec![
            FeatureTriplet {
                index_0: 0,
                index_1: 0,
                index_2: 0,
                confidence: 1.0,
                disparity_01: 5.0,
                disparity_12: 5.0,
                cascade_error: 0.0,
            },
            FeatureTriplet {
                index_0: 1,
                index_1: 1,
                index_2: 1,
                confidence: 1.0,
                disparity_01: 5.0,
                disparity_12: 5.0,
                cascade_error: 0.0,
            },
        ];

        let frames = [f0, f1, f2];
        let shifts = WiggleAligner::compute_depth_surface_shifts_from_triplets(
            &frames,
            &triplets,
            DEFAULT_DISPARITY_BIN_SIZE_PX,
            DEFAULT_CLUSTER_TOLERANCE_PX,
        );
        // p1 - p0 = (15-10, 22-20) = (5.0, 2.0)
        assert_eq!(shifts[0], (5.0, 2.0));
        assert_eq!(shifts[1], (0.0, 0.0));
        // p1 - p2 = (15-20, 22-24) = (-5.0, -2.0)
        assert_eq!(shifts[2], (-5.0, -2.0));
        let (shifts_dbg, debug_hist) =
            WiggleAligner::compute_depth_surface_shifts_from_triplets_with_debug(
                &frames,
                &triplets,
                DEFAULT_DISPARITY_BIN_SIZE_PX,
                DEFAULT_CLUSTER_TOLERANCE_PX,
            );
        assert_eq!(shifts_dbg[0], (5.0, 2.0));
        assert_eq!(shifts_dbg[1], (0.0, 0.0));
        assert_eq!(shifts_dbg[2], (-5.0, -2.0));
        assert_eq!(debug_hist.total_samples, 2);
        assert_eq!(debug_hist.selected_cluster_points, 2);
        assert!(!debug_hist.bins.is_empty());

        let mut csv_out = Vec::new();
        debug_hist.write_csv(&mut csv_out).unwrap();
        let csv_str = String::from_utf8(csv_out).unwrap();
        assert!(csv_str.contains("bin_idx,bin_start_px,bin_end_px,bin_center_px,count,fraction,percentage,cluster_id,is_selected_surface"));
        assert!(csv_str.contains("# metric=inv_disparity,bin_size="));
    }

    #[test]
    fn test_align_and_crop_and_build_gif() {
        let img0 =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(100, 100, Rgba([255, 0, 0, 255])));
        let img1 =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(100, 100, Rgba([0, 255, 0, 255])));
        let img2 =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(100, 100, Rgba([0, 0, 255, 255])));

        let shifts = [(2.0, 0.0), (0.0, 0.0), (-2.0, 0.0)];
        let aligned = WiggleAligner::align_and_crop(&[img0, img1, img2], &shifts).unwrap();
        assert_eq!(aligned.len(), 3);
        assert_eq!(aligned[0].dimensions(), (96, 100));

        let mut out = Vec::new();
        let config = WiggleGifConfig::default();
        WiggleGifBuilder::build_wiggle_gif(&aligned, &config, &mut out).unwrap();
        assert!(!out.is_empty());
        // GIF magic header
        assert_eq!(&out[0..6], b"GIF89a");
    }

    #[test]
    fn test_compute_depth_surface_shifts_with_face_priority() {
        let f0 = FeatureFrame::new(
            0,
            NormalizedRect::new(0.0, 0.0, 0.33, 1.0).unwrap(),
            Size2D::new(100, 100),
            vec![
                KeyPoint::new(Point2D::new(10.0, 20.0), 0.9, None), // background
                KeyPoint::new(Point2D::new(50.0, 50.0), 0.9, None), // face keypoint
            ],
        );
        let f1 = FeatureFrame::new(
            1,
            NormalizedRect::new(0.33, 0.0, 0.33, 1.0).unwrap(),
            Size2D::new(100, 100),
            vec![
                KeyPoint::new(Point2D::new(15.0, 22.0), 0.9, None), // background
                KeyPoint::new(Point2D::new(55.0, 52.0), 0.9, None), // face keypoint
            ],
        );
        let f2 = FeatureFrame::new(
            2,
            NormalizedRect::new(0.66, 0.0, 0.33, 1.0).unwrap(),
            Size2D::new(100, 100),
            vec![
                KeyPoint::new(Point2D::new(20.0, 24.0), 0.9, None), // background
                KeyPoint::new(Point2D::new(60.0, 54.0), 0.9, None), // face keypoint
            ],
        );

        let triplets = vec![
            FeatureTriplet {
                index_0: 0,
                index_1: 0,
                index_2: 0,
                confidence: 0.9,
                disparity_01: 5.0,
                disparity_12: 5.0,
                cascade_error: 0.0,
            },
            FeatureTriplet {
                index_0: 1,
                index_1: 1,
                index_2: 1,
                confidence: 0.95,
                disparity_01: 5.0,
                disparity_12: 5.0,
                cascade_error: 0.0,
            },
        ];

        let frames = [f0, f1, f2];
        // Dominant face bounding box centered around (0.5, 0.5) covering keypoint 1 at (55.0, 52.0)
        let face_bbox = NormalizedRect::new(0.4, 0.4, 0.3, 0.3).unwrap();

        let shifts = WiggleAligner::compute_depth_surface_shifts_from_triplets_with_face_priority(
            &frames,
            &triplets,
            Some(face_bbox),
            DEFAULT_DISPARITY_BIN_SIZE_PX,
            DEFAULT_CLUSTER_TOLERANCE_PX,
        );

        // Should lock shifts exclusively onto face keypoint 1:
        // p1 - p0 = (55 - 50, 52 - 50) = (5.0, 2.0)
        assert_eq!(shifts[0], (5.0, 2.0));
        assert_eq!(shifts[1], (0.0, 0.0));
        assert_eq!(shifts[2], (-5.0, -2.0));
    }

    #[test]
    fn test_build_wiggle_gif_with_adaptive_delays() {
        let f0 = RgbaImage::new(50, 50);
        let f1 = RgbaImage::new(50, 50);
        let f2 = RgbaImage::new(50, 50);
        let frames = [f0, f1, f2];

        let config = WiggleGifConfig::new(100).with_adaptive_delays([80, 120]);

        let mut output = Vec::new();
        let res = WiggleGifBuilder::build_wiggle_gif(&frames, &config, &mut output);
        assert!(res.is_ok());
        assert!(!output.is_empty());
    }
}
