//! Per-pixel statistical extraction along film strip stacking axes.

use crate::geom::StripOrientation;
use crate::luma::ScaledLumaImage;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Default cross-axis lower percentile for dynamic range contrast profile (5th percentile = 0.05).
pub const DEFAULT_LOW_PERCENTILE: f32 = 0.05;

/// Default cross-axis upper percentile for dynamic range contrast profile (98th percentile = 0.98).
pub const DEFAULT_HIGH_PERCENTILE: f32 = 0.98;

/// Statistical summary of cross-axis luma samples for a single pixel position along the stacking axis.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AxisPixelStats {
    /// Minimum luma intensity in `[0, 255]`.
    pub min: u8,
    /// Maximum luma intensity in `[0, 255]`.
    pub max: u8,
    /// Arithmetic mean luma value in `[0.0, 255.0]`.
    pub avg: f32,
    /// Population standard deviation in `[0.0, 128.0]`.
    pub std: f32,
    /// 5th percentile luma value in `[0, 255]`.
    pub p5: u8,
    /// 98th percentile luma value in `[0, 255]`.
    pub p98: u8,
}

impl AxisPixelStats {
    /// Computes statistical metrics from a slice of 8-bit luma samples.
    ///
    /// # Arguments
    /// * `samples` - Non-empty slice of 8-bit luma intensities.
    ///
    /// # Panics
    /// Panics if `samples` is empty.
    ///
    /// # Examples
    /// ```
    /// use reto_core::AxisPixelStats;
    ///
    /// let samples = [10_u8, 20, 30, 40, 50];
    /// let stats = AxisPixelStats::from_samples(&samples);
    /// assert_eq!(stats.min, 10);
    /// assert_eq!(stats.max, 50);
    /// assert_eq!(stats.avg, 30.0);
    /// ```
    #[must_use]
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_samples(samples: &[u8]) -> Self {
        Self::from_iter(samples.iter().copied(), samples.len() as u32)
    }

    /// Constructs statistics from an arbitrary iterator of byte samples.
    ///
    /// Computes summary statistics in a single pass without heap allocating sample vectors.
    ///
    /// # Panics
    /// Panics if `total_count` is zero.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::suboptimal_flops
    )]
    pub fn from_iter<I: Iterator<Item = u8>>(iter: I, total_count: u32) -> Self {
        assert!(total_count > 0, "Samples count cannot be zero");

        let n = total_count as f32;
        let mut min_val = u8::MAX;
        let mut max_val = u8::MIN;
        let mut sum = 0_u64;
        let mut sum_sq = 0_u64;
        let mut hist = [0_u32; 256];

        for val in iter {
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
            let v_u64 = u64::from(val);
            sum += v_u64;
            sum_sq += v_u64 * v_u64;
            hist[val as usize] += 1;
        }

        let avg = (sum as f32) / n;
        let variance = ((sum_sq as f32) / n) - (avg * avg);
        let std = if variance > 0.0 { variance.sqrt() } else { 0.0 };

        // 5th percentile rank and 98th percentile rank
        let p5_target = ((n * DEFAULT_LOW_PERCENTILE).round() as u32).min(total_count - 1);
        let p98_target = ((n * DEFAULT_HIGH_PERCENTILE).round() as u32).min(total_count - 1);

        let mut cum = 0_u32;
        let mut p5 = min_val;
        let mut p98 = max_val;
        let mut found_p5 = false;

        for (val, &count) in hist.iter().enumerate() {
            cum += count;
            if !found_p5 && cum >= p5_target {
                p5 = val as u8;
                found_p5 = true;
            }
            if cum >= p98_target {
                p98 = val as u8;
                break;
            }
        }

        Self {
            min: min_val,
            max: max_val,
            avg,
            std,
            p5,
            p98,
        }
    }
}

/// Aggregated strip profiles evaluated across scanline cross-sections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisStatisticsProfile {
    /// Ordered sequence of cross-sectional statistics computed for each position on the primary axis.
    pub stats: Vec<AxisPixelStats>,
    /// Number of cross-axis sample pixels integrated into each statistics slice.
    pub sample_count: u32,
    /// Orientation of the analyzed film strip.
    pub orientation: StripOrientation,
}

impl AxisStatisticsProfile {
    /// Computes the cross-sectional statistical profile of a scaled luma image along its primary stacking axis.
    ///
    /// # Arguments
    /// * `luma` - The scaled grayscale image to profile.
    ///
    /// # Examples
    /// ```
    /// use reto_core::{AxisStatisticsProfile, ScaledLumaImage, Bt709LumaConverter, PROJECTION_MAX_DIMENSION};
    /// use image::{Rgba, RgbaImage};
    ///
    /// let img = RgbaImage::from_pixel(300, 100, Rgba([120, 120, 120, 255]));
    /// let luma = ScaledLumaImage::from_image(&img, &Bt709LumaConverter::new(), PROJECTION_MAX_DIMENSION).unwrap();
    /// let profile = AxisStatisticsProfile::compute(&luma);
    /// assert_eq!(profile.len(), 300);
    /// assert_eq!(profile.stats[0].min, 120);
    /// ```
    #[must_use]
    pub fn compute(luma: &ScaledLumaImage) -> Self {
        let major_len = luma.major_len;
        let minor_len = luma.minor_len;

        let stats: Vec<AxisPixelStats> = (0..major_len)
            .into_par_iter()
            .map(|major| {
                let iter = (0..minor_len).map(|cross| luma.get(major, cross));
                AxisPixelStats::from_iter(iter, minor_len)
            })
            .collect();

        Self {
            stats,
            sample_count: minor_len,
            orientation: luma.orientation,
        }
    }

    /// Number of pixel positions along the major stacking axis.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.stats.len()
    }

    /// Checks if the statistics profile is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.stats.is_empty()
    }

    /// Extracts the series of minimum luma values along the stacking axis.
    #[must_use]
    pub fn min_series(&self) -> Vec<u8> {
        self.stats.iter().map(|s| s.min).collect()
    }

    /// Extracts the series of maximum luma values along the stacking axis.
    #[must_use]
    pub fn max_series(&self) -> Vec<u8> {
        self.stats.iter().map(|s| s.max).collect()
    }

    /// Extracts the series of average luma values along the stacking axis.
    #[must_use]
    pub fn avg_series(&self) -> Vec<f32> {
        self.stats.iter().map(|s| s.avg).collect()
    }

    /// Extracts the series of standard deviation values along the stacking axis.
    #[must_use]
    pub fn std_series(&self) -> Vec<f32> {
        self.stats.iter().map(|s| s.std).collect()
    }

    /// Extracts the series of 5th percentile luma values along the stacking axis.
    #[must_use]
    pub fn p5_series(&self) -> Vec<u8> {
        self.stats.iter().map(|s| s.p5).collect()
    }

    /// Extracts the series of 98th percentile luma values along the stacking axis.
    #[must_use]
    pub fn p98_series(&self) -> Vec<u8> {
        self.stats.iter().map(|s| s.p98).collect()
    }

    /// Extracts the series of dynamic range contrast spreads (`p98 - p5`) along the stacking axis.
    #[must_use]
    pub fn p98_minus_p5_series(&self) -> Vec<u8> {
        self.stats
            .iter()
            .map(|s| s.p98.saturating_sub(s.p5))
            .collect()
    }
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn find_gutter_spans(
    diffs: &[u8],
    l: usize,
    n: usize,
    nominal_pitch: f32,
) -> (Vec<u32>, Vec<GutterSpan>, f32) {
    let search_radius = ((nominal_pitch * 0.12).round() as usize).max(8);
    let max_gutter_half = ((nominal_pitch * 0.04).round() as usize).max(4);

    let mut gutters = Vec::with_capacity(n - 1);
    let mut gutter_centers = Vec::with_capacity(n - 1);
    let mut total_score = 0.0_f32;

    for k in 1..n {
        let nominal_center = ((k as f32) * nominal_pitch).round() as usize;
        let s = nominal_center.saturating_sub(search_radius);
        let e = (nominal_center + search_radius + 1).min(l);

        let mut min_idx = nominal_center;
        let mut min_val = u8::MAX;
        for (idx, &val) in diffs[s..e].iter().enumerate() {
            if val < min_val {
                min_val = val;
                min_idx = s + idx;
            }
        }

        let edge_thresh = (f32::from(min_val) + 12.0).min(32.0);

        let mut g_start = min_idx;
        while g_start > nominal_center.saturating_sub(max_gutter_half)
            && g_start > 0
            && f32::from(diffs[g_start - 1]) <= edge_thresh
        {
            g_start -= 1;
        }

        let mut g_end = min_idx;
        while g_end + 1 < (nominal_center + max_gutter_half).min(l)
            && f32::from(diffs[g_end + 1]) <= edge_thresh
        {
            g_end += 1;
        }

        if g_end <= g_start {
            let default_half = ((nominal_pitch * 0.01).round() as usize).max(2);
            g_start = min_idx.saturating_sub(default_half);
            g_end = (min_idx + default_half).min(l - 1);
        }

        gutter_centers.push(min_idx as u32);
        gutters.push(GutterSpan {
            center: min_idx as u32,
            start: g_start as u32,
            end: g_end as u32,
            width: (g_end - g_start + 1) as u32,
        });
        total_score += f32::from(min_val);
    }

    (gutter_centers, gutters, total_score)
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn find_scan_margins(
    stats: &[AxisPixelStats],
    l: usize,
    nominal_pitch: f32,
    avg_gutter_min: f32,
) -> (usize, usize) {
    let margin_thresh = (avg_gutter_min + 6.0).min(24.0);
    let margin_max = ((nominal_pitch * 0.03).round() as usize).max(2);

    let mut margin_start = 0_usize;
    while margin_start < margin_max
        && f32::from(stats[margin_start].p98.saturating_sub(stats[margin_start].p5)) <= margin_thresh
    {
        margin_start += 1;
    }

    let mut margin_end = l.saturating_sub(1);
    while margin_end > l.saturating_sub(margin_max)
        && f32::from(stats[margin_end].p98.saturating_sub(stats[margin_end].p5)) <= margin_thresh
    {
        margin_end -= 1;
    }

    (margin_start, margin_end)
}

#[allow(clippy::cast_possible_truncation)]
fn compute_active_frame_spans(
    n: usize,
    margin_start: usize,
    margin_end: usize,
    gutters: &[GutterSpan],
) -> Vec<(u32, u32)> {
    let mut frame_spans = Vec::with_capacity(n);
    for i in 0..n {
        let f_start = if i == 0 {
            margin_start as u32
        } else {
            gutters[i - 1].end + 1
        };

        let f_end = if i == n - 1 {
            margin_end as u32
        } else {
            gutters[i].start.saturating_sub(1)
        };

        let f_len = (f_end.saturating_sub(f_start) + 1).max(10);
        frame_spans.push((f_start, f_len));
    }
    frame_spans
}

impl AxisStatisticsProfile {
    /// Estimates optimal equi-spaced film sub-frame divider grid using median-trough alignment.
    ///
    /// # Arguments
    /// * `expected_frames` - Number of expected film frames $N$ (e.g. 3 for RETO3D Classic, 4 for N4).
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn find_optimal_grid(&self, expected_frames: usize) -> Option<OptimalGridResult> {
        if expected_frames < 2 || self.stats.len() < expected_frames * 20 {
            return None;
        }

        let l = self.stats.len();
        let n = expected_frames;
        let nominal_pitch = l as f32 / n as f32;

        let diffs = self.p98_minus_p5_series();
        let (gutter_centers, gutters, total_score) =
            find_gutter_spans(&diffs, l, n, nominal_pitch);

        let avg_gutter_min = total_score / (n - 1) as f32;
        let (margin_start, margin_end) =
            find_scan_margins(&self.stats, l, nominal_pitch, avg_gutter_min);

        let frame_spans = compute_active_frame_spans(n, margin_start, margin_end, &gutters);
        let quality_score = total_score / (n - 1) as f32;

        Some(OptimalGridResult {
            start_offset: margin_start as u32,
            frame_pitch: nominal_pitch.round() as u32,
            gutter_centers,
            gutters,
            frame_spans,
            score: quality_score,
        })
    }
}

type ThresholdRun = (bool, usize, usize);
type ThresholdResult = (u8, Vec<ThresholdRun>);

#[inline]
fn run_length_threshold(diffs: &[u8], t: u8) -> Vec<ThresholdRun> {
    if diffs.is_empty() {
        return Vec::new();
    }
    let mut runs = Vec::new();
    let mut curr_val = diffs[0] >= t;
    let mut curr_start = 0_usize;
    let mut curr_len = 1_usize;

    for (i, &val) in diffs.iter().enumerate().skip(1) {
        let is_image = val >= t;
        if is_image == curr_val {
            curr_len += 1;
        } else {
            runs.push((curr_val, curr_start, curr_len));
            curr_val = is_image;
            curr_start = i;
            curr_len = 1;
        }
    }
    runs.push((curr_val, curr_start, curr_len));
    runs
}

fn validate_threshold_runs(
    runs: &[ThresholdRun],
    n: usize,
    min_frame_w: usize,
    max_frame_w: usize,
    min_gutter_w: usize,
    max_gutter_w: usize,
) -> bool {
    if runs.len() != 2 * n - 1 {
        return false;
    }
    for (i, &(is_img, _s, length)) in runs.iter().enumerate() {
        if i % 2 == 0 {
            if !is_img || length < min_frame_w || length > max_frame_w {
                return false;
            }
        } else if is_img || length < min_gutter_w || length > max_gutter_w {
            return false;
        }
    }
    true
}

fn select_plateau_runs(
    valid_thresholds: &[ThresholdResult],
) -> Option<ThresholdResult> {
    let mut plateau_counts: std::collections::HashMap<Vec<usize>, usize> =
        std::collections::HashMap::new();
    for (_, runs) in valid_thresholds {
        let signature: Vec<usize> = runs.iter().map(|r| r.2).collect();
        *plateau_counts.entry(signature).or_insert(0) += 1;
    }

    let stable_signatures: Vec<Vec<usize>> = plateau_counts
        .iter()
        .filter(|&(_, &count)| count >= 3)
        .map(|(sig, _)| sig.clone())
        .collect();

    if stable_signatures.is_empty() {
        let (most_frequent_sig, _) =
            plateau_counts.into_iter().max_by_key(|&(_, count)| count)?;
        valid_thresholds.iter().find(|&(_, runs)| {
            let sig: Vec<usize> = runs.iter().map(|r| r.2).collect();
            sig == most_frequent_sig
        }).cloned()
    } else {
        valid_thresholds.iter().find(|&(_, runs)| {
            let sig: Vec<usize> = runs.iter().map(|r| r.2).collect();
            stable_signatures.contains(&sig)
        }).cloned()
    }
}

impl AxisStatisticsProfile {
    /// Finds the maximum contrast threshold that decomposes the profile into exactly $2N-1$
    /// consecutive alternating regions (Image, Gutter, Image, ..., Image).
    ///
    /// Evaluates candidate binary thresholds and returns the topological segmentation
    /// along with a confidence metric representing threshold stability margin.
    ///
    /// # Arguments
    /// * `expected_frames` - Number of expected film frames $N$ (e.g. 3 for RETO3D Classic).
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn find_threshold_partition(
        &self,
        expected_frames: usize,
    ) -> Option<ThresholdPartitionResult> {
        if expected_frames < 2 || self.stats.len() < expected_frames * 20 {
            return None;
        }

        let l = self.stats.len();
        let n = expected_frames;
        let nominal_pitch = l as f32 / n as f32;

        let min_frame_w = (nominal_pitch * 0.65).round() as usize;
        let max_frame_w = (nominal_pitch * 1.35).round() as usize;
        let min_gutter_w = ((nominal_pitch * 0.008).round() as usize).max(4);
        let max_gutter_w = ((nominal_pitch * 0.10).round() as usize).max(8);

        let diffs: Vec<u8> = self.p98_minus_p5_series();

        let mut sample_diffs = diffs.clone();
        let min_d = *sample_diffs.iter().min().unwrap_or(&0);
        let mid_idx = l / 2;
        let (_, &mut median_val, _) = sample_diffs.select_nth_unstable(mid_idx);
        let median_d = f32::from(median_val);
        let p80_idx = (l * 4) / 5;
        let (_, &mut max_d, _) = sample_diffs.select_nth_unstable(p80_idx);

        let mut valid_thresholds = Vec::new();
        for t in (min_d + 1)..=max_d {
            let runs = run_length_threshold(&diffs, t);
            if validate_threshold_runs(&runs, n, min_frame_w, max_frame_w, min_gutter_w, max_gutter_w) {
                valid_thresholds.push((t, runs));
            }
        }

        if valid_thresholds.is_empty() {
            return None;
        }

        let t_min = valid_thresholds.first()?.0;
        let t_max = valid_thresholds.last()?.0;
        let t_span = f32::from(t_max.saturating_sub(t_min) + 1);

        let (chosen_t, chosen_runs) = select_plateau_runs(&valid_thresholds)?;
        let dynamic_range = (median_d - f32::from(min_d)).max(1.0);
        let confidence = (t_span / dynamic_range * 2.0).clamp(0.0, 1.0);

        let mut gutters = Vec::with_capacity(n - 1);
        let mut frame_spans = Vec::with_capacity(n);

        for (i, &(_is_img, start, length)) in chosen_runs.iter().enumerate() {
            if i % 2 == 0 {
                frame_spans.push((start as u32, length as u32));
            } else {
                let center = (start + length / 2) as u32;
                gutters.push(GutterSpan {
                    center,
                    start: start as u32,
                    end: (start + length - 1) as u32,
                    width: length as u32,
                });
            }
        }

        Some(ThresholdPartitionResult {
            threshold: chosen_t,
            threshold_range: (t_min, t_max),
            confidence,
            gutters,
            frame_spans,
        })
    }
}

/// Result of searching for a binary contrast threshold that cleanly separates
/// the profile into $2N-1$ alternating (Frame, Gutter, ..., Frame) consecutive topological regions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThresholdPartitionResult {
    /// Optimal separation threshold value used.
    pub threshold: u8,
    /// Range of all valid threshold values $[T_{\min}, T_{\max}]$ that yield exactly $2N-1$ regions.
    pub threshold_range: (u8, u8),
    /// Measured confidence score in $[0.0, 1.0]$ based on threshold stability span.
    pub confidence: f32,
    /// Physical gutter boundary spans.
    pub gutters: Vec<GutterSpan>,
    /// Individual active frame pixel spans `(start_pixel, length_pixels)`.
    pub frame_spans: Vec<(u32, u32)>,
}

/// Physical boundary span of an individual gutter gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GutterSpan {
    /// Center pixel of the gutter.
    pub center: u32,
    /// Left / top boundary of the gutter gap.
    pub start: u32,
    /// Right / bottom boundary of the gutter gap.
    pub end: u32,
    /// Total width of the gutter in pixels.
    pub width: u32,
}

/// Result of 1D energy minimization regular grid search for $N-1$ frame dividers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimalGridResult {
    /// Start coordinate (left/top margin) of the frame sequence in pixels.
    pub start_offset: u32,
    /// Regular frame pitch (frame span) in pixels.
    pub frame_pitch: u32,
    /// Centers of the $N-1$ physical gutter dividers in pixel coordinates.
    pub gutter_centers: Vec<u32>,
    /// Exact physical gutter boundary spans.
    pub gutters: Vec<GutterSpan>,
    /// Individual active frame pixel spans `(start_pixel, length_pixels)`.
    pub frame_spans: Vec<(u32, u32)>,
    /// Quality score of the optimal grid match.
    pub score: f32,
}

/// Applies a 1D inverted-notch matched filter to highlight gutter gaps of expected width.
///
/// The filter response computes the difference between surrounding frame margins and the central trough:
///
/// $$\text{Response}(x) = \text{Mean}(\text{Flanks}) - \text{Mean}(\text{Trough})$$
///
/// # Arguments
/// * `profile` - 1D scalar profile values (e.g. $\Delta P(x)$ or luma values).
/// * `trough_radius` - Half-width of the expected central gutter trough.
/// * `margin_radius` - Width of the surrounding comparison margins on each side.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
pub fn matched_filter_1d(profile: &[f32], trough_radius: usize, margin_radius: usize) -> Vec<f32> {
    let n = profile.len();
    if n == 0 {
        return Vec::new();
    }

    // Build prefix sum for O(1) interval queries
    let mut prefix = vec![0.0_f64; n + 1];
    for i in 0..n {
        prefix[i + 1] = prefix[i] + f64::from(profile[i]);
    }
    let range_sum = |start: usize, end: usize| -> f64 {
        if start >= end || start >= n {
            0.0
        } else {
            let end_clamped = end.min(n);
            prefix[end_clamped] - prefix[start]
        }
    };

    let total_radius = trough_radius + margin_radius;
    let mut response = vec![0.0_f32; n];

    for (i, out_slot) in response.iter_mut().enumerate() {
        let left_m_start = i.saturating_sub(total_radius);
        let left_m_end = i.saturating_sub(trough_radius);
        let right_m_start = (i + trough_radius + 1).min(n);
        let right_m_end = (i + total_radius + 1).min(n);

        let trough_start = i.saturating_sub(trough_radius);
        let trough_end = (i + trough_radius + 1).min(n);

        let left_m_count = left_m_end.saturating_sub(left_m_start);
        let right_m_count = right_m_end.saturating_sub(right_m_start);
        let trough_count = trough_end.saturating_sub(trough_start);

        let margin_sum =
            range_sum(left_m_start, left_m_end) + range_sum(right_m_start, right_m_end);
        let margin_count = left_m_count + right_m_count;

        let trough_sum = range_sum(trough_start, trough_end);

        if margin_count > 0 && trough_count > 0 {
            let margin_mean = margin_sum / (margin_count as f64);
            let trough_mean = trough_sum / (trough_count as f64);
            *out_slot = (margin_mean - trough_mean) as f32;
        }
    }

    response
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;
    use crate::color::{Bt709LumaConverter, SimpleGrayConverter};
    use crate::luma::{ScaledLumaImage, PROJECTION_MAX_DIMENSION};
    use image::{Rgba, RgbaImage};

    #[test]
    fn test_axis_pixel_stats_distribution() {
        let mut samples = Vec::new();
        for i in 0..100 {
            samples.push(i as u8);
        }

        let stats = AxisPixelStats::from_samples(&samples);
        assert_eq!(stats.min, 0);
        assert_eq!(stats.max, 99);
        assert!((stats.avg - 49.5).abs() < 1e-4);
        assert!(stats.std > 0.0);
        assert_eq!(stats.p5, 4);
        assert_eq!(stats.p98, 97);
    }

    #[test]
    fn test_axis_statistics_profile_horizontal() {
        let mut img = RgbaImage::from_pixel(300, 100, Rgba([50, 50, 50, 255]));
        // Add a brighter column at x=150
        for y in 0..100 {
            img.put_pixel(150, y, Rgba([200, 200, 200, 255]));
        }

        let conv = Bt709LumaConverter::new();
        let strip = ScaledLumaImage::from_image(&img, &conv, PROJECTION_MAX_DIMENSION).unwrap();
        let profile = AxisStatisticsProfile::compute(&strip);

        assert_eq!(profile.len(), 300);
        assert_eq!(profile.orientation, StripOrientation::Horizontal);
        assert_eq!(profile.sample_count, 100);

        assert_eq!(profile.stats[0].min, 50);
        assert_eq!(profile.stats[0].max, 50);
        assert_eq!(profile.stats[150].min, 200);
        assert_eq!(profile.stats[150].max, 200);

        let mins = profile.min_series();
        assert_eq!(mins.len(), 300);
        assert_eq!(mins[150], 200);
    }

    #[test]
    fn test_axis_statistics_profile_vertical() {
        let mut img = RgbaImage::from_pixel(100, 300, Rgba([60, 60, 60, 255]));
        // In vertical strip, row at y=100 is major index 100
        for x in 0..100 {
            img.put_pixel(x, 100, Rgba([180, 180, 180, 255]));
        }

        let conv = SimpleGrayConverter::new();
        let strip = ScaledLumaImage::from_image(&img, &conv, PROJECTION_MAX_DIMENSION).unwrap();
        let profile = AxisStatisticsProfile::compute(&strip);

        assert_eq!(profile.len(), 300);
        assert_eq!(profile.orientation, StripOrientation::Vertical);
        assert_eq!(profile.sample_count, 100);

        assert_eq!(profile.stats[0].min, 60);
        assert_eq!(profile.stats[100].min, 180);
        assert_eq!(profile.stats[100].max, 180);
    }

    #[test]
    fn test_matched_filter_1d() {
        // Flat signal: matched filter response should be ~0
        let flat = vec![100.0_f32; 100];
        let resp_flat = matched_filter_1d(&flat, 5, 10);
        for &r in &resp_flat[15..85] {
            assert!(r.abs() < 1e-4);
        }

        // Notch at index 50
        let mut notch = vec![180.0_f32; 100];
        notch[47..=53].fill(20.0);
        let resp_notch = matched_filter_1d(&notch, 3, 10);
        // Peak should be at or near index 50
        let max_idx = resp_notch
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 50);
        assert!(resp_notch[50] > 100.0);
    }

    #[test]
    fn test_find_optimal_grid_n3() {
        // Create 300x100 strip with 3 high-contrast frames and 2 low-contrast gutters at x=100 and x=200
        let mut img = RgbaImage::from_pixel(300, 100, Rgba([0, 0, 0, 255]));
        for y in 0..100 {
            for x in 0..300 {
                if (95..=105).contains(&x) || (195..=205).contains(&x) {
                    img.put_pixel(x, y, Rgba([20, 20, 20, 255]));
                } else {
                    let val = if y % 2 == 0 { 240 } else { 40 };
                    img.put_pixel(x, y, Rgba([val, val, val, 255]));
                }
            }
        }

        let conv = SimpleGrayConverter::new();
        let strip = ScaledLumaImage::from_image(&img, &conv, PROJECTION_MAX_DIMENSION).unwrap();
        let profile = AxisStatisticsProfile::compute(&strip);

        let grid = profile
            .find_optimal_grid(3)
            .expect("Grid search finds optimal 3 frames");
        assert_eq!(grid.gutter_centers.len(), 2);
        // Gutter 1 center should be ~100
        assert!((i64::from(grid.gutter_centers[0]) - 100).abs() <= 5);
        // Gutter 2 center should be ~200
        assert!((i64::from(grid.gutter_centers[1]) - 200).abs() <= 5);
    }

    #[test]
    fn test_find_threshold_partition_n3() {
        // Create 300x100 strip with 3 high-contrast frames and 2 low-contrast gutters at x=100 and x=200
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
        let strip = ScaledLumaImage::from_image(&img, &conv, PROJECTION_MAX_DIMENSION).unwrap();
        let profile = AxisStatisticsProfile::compute(&strip);

        let partition = profile
            .find_threshold_partition(3)
            .expect("Threshold partition finds 3 frames");
        assert_eq!(partition.frame_spans.len(), 3);
        assert_eq!(partition.gutters.len(), 2);
        assert!(partition.confidence > 0.3);
    }
}
