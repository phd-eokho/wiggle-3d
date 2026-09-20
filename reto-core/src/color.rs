//! Color space definitions, high-definition ITU-R BT.709 colorimetry transforms, and YUV420p conversion.
//!
//! Provides unified matrix transformations, Luma converters, and planar video frame representations.

#![allow(
    clippy::cast_lossless,
    clippy::suboptimal_flops,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::needless_range_loop
)]

use image::RgbaImage;
use ndarray::{arr1, Array1, ArrayView2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Standard ITU-R BT.709 transformation matrix for converting RGB to YUV color space:
///
/// $$\begin{bmatrix} Y \\ U \\ V \end{bmatrix} = \begin{bmatrix} 0.2126 & 0.7152 & 0.0722 \\ -0.1146 & -0.3854 & 0.5000 \\ 0.5000 & -0.4542 & -0.0458 \end{bmatrix} \begin{bmatrix} R \\ G \\ B \end{bmatrix} + \begin{bmatrix} 0 \\ 128 \\ 128 \end{bmatrix}$$
pub const BT709_YUV_MATRIX: [[f32; 3]; 3] = [
    [0.2126, 0.7152, 0.0722],   // Y (Luma / Grayscale intensity)
    [-0.1146, -0.3854, 0.5000], // U (Chroma Cb)
    [0.5000, -0.4542, -0.0458], // V (Chroma Cr)
];

/// ITU-R BT.709 linear weights for RGB components `[R=0.2126, G=0.7152, B=0.0722]` derived from [`BT709_YUV_MATRIX`].
pub const BT709_RGB_WEIGHTS: [f32; 3] = BT709_YUV_MATRIX[0];

/// Interface contract for converting RGB color components to scalar luma / grayscale intensity.
pub trait LumaConverter: Send + Sync {
    /// 3-element channel linear weight vector `[W_r, W_g, W_b]` for matrix / dot product computations.
    fn weights(&self) -> [f32; 3];

    /// Converts 8-bit RGB color channels to a scalar 8-bit luma intensity in `[0, 255]` via dot product.
    ///
    /// # Arguments
    /// * `r` - Red channel `[0, 255]`.
    /// * `g` - Green channel `[0, 255]`.
    /// * `b` - Blue channel `[0, 255]`.
    #[inline]
    fn rgb_to_luma(&self, r: u8, g: u8, b: u8) -> u8 {
        let w = self.weights();
        (w[0] * f32::from(r) + w[1] * f32::from(g) + w[2] * f32::from(b))
            .round()
            .clamp(0.0, 255.0) as u8
    }

    /// Vectorized conversion of an `[N, 3]` RGB float tensor to a 1D `[N]` luma tensor using matrix-vector multiplication ($Y = X \cdot W$).
    #[must_use]
    fn convert_rgb_tensor(&self, rgb_matrix: &ArrayView2<f32>) -> Array1<f32> {
        let weights = arr1(&self.weights());
        rgb_matrix.dot(&weights)
    }

    /// Vectorized conversion of packed 24-bit RGB pixel buffers into a scalar 8-bit luma slice.
    ///
    /// # Arguments
    /// * `rgb` - Interleaved 8-bit RGB bytes (length must be multiple of 3).
    /// * `out_luma` - Output buffer destination.
    fn convert_rgb_slice(&self, rgb: &[u8], out_luma: &mut [u8]) {
        let count = (rgb.len() / 3).min(out_luma.len());
        if count == 0 {
            return;
        }
        let [wr, wg, wb] = self.weights();
        for (out, chunk) in out_luma[..count].iter_mut().zip(rgb.as_chunks::<3>().0) {
            let r = f32::from(chunk[0]);
            let g = f32::from(chunk[1]);
            let b = f32::from(chunk[2]);
            *out = (wr * r + wg * g + wb * b).round().clamp(0.0, 255.0) as u8;
        }
    }

    /// Vectorized conversion of packed 32-bit RGBA pixel buffers into a scalar 8-bit luma slice.
    ///
    /// # Arguments
    /// * `rgba` - Interleaved 8-bit RGBA bytes (length must be multiple of 4).
    /// * `out_luma` - Output buffer destination.
    fn convert_rgba_slice(&self, rgba: &[u8], out_luma: &mut [u8]) {
        let count = (rgba.len() / 4).min(out_luma.len());
        if count == 0 {
            return;
        }
        let [wr, wg, wb] = self.weights();
        for (out, chunk) in out_luma[..count].iter_mut().zip(rgba.as_chunks::<4>().0) {
            let r = f32::from(chunk[0]);
            let g = f32::from(chunk[1]);
            let b = f32::from(chunk[2]);
            *out = (wr * r + wg * g + wb * b).round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Simple arithmetic mean RGB to Grayscale converter: $Y = \frac{R + G + B}{3}$.
///
/// # Examples
/// ```
/// use reto_core::{LumaConverter, SimpleGrayConverter};
///
/// let converter = SimpleGrayConverter::new();
/// assert_eq!(converter.rgb_to_luma(30, 60, 90), 60);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimpleGrayConverter;

impl SimpleGrayConverter {
    /// Creates a new `SimpleGrayConverter`.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl LumaConverter for SimpleGrayConverter {
    #[inline]
    fn weights(&self) -> [f32; 3] {
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    }
}

/// ITU-R BT.709 high-definition luma converter: $Y = 0.2126 R + 0.7152 G + 0.0722 B$.
///
/// # Examples
/// ```
/// use reto_core::{Bt709LumaConverter, LumaConverter};
///
/// let converter = Bt709LumaConverter::new();
/// let luma = converter.rgb_to_luma(255, 255, 255);
/// assert_eq!(luma, 255);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bt709LumaConverter;

impl Bt709LumaConverter {
    /// Creates a new `Bt709LumaConverter`.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl LumaConverter for Bt709LumaConverter {
    #[inline]
    fn weights(&self) -> [f32; 3] {
        BT709_RGB_WEIGHTS
    }
}

/// Planar YUV420 (I420) video frame buffer.
///
/// Contains distinct Y (luminance), U (chrominance Cb), and V (chrominance Cr)
/// planes with standard 4:2:0 chroma subsampling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Yuv420PlanarFrame {
    /// Width of the video frame in pixels (always even).
    pub width: u32,
    /// Height of the video frame in pixels (always even).
    pub height: u32,
    /// Luminance (Y) plane byte buffer with dimensions `[height * y_stride]`.
    pub y_plane: Vec<u8>,
    /// Chrominance Cb (U) plane byte buffer with dimensions `[(height / 2) * uv_stride]`.
    pub u_plane: Vec<u8>,
    /// Chrominance Cr (V) plane byte buffer with dimensions `[(height / 2) * uv_stride]`.
    pub v_plane: Vec<u8>,
    /// Row stride for the Y plane in bytes (equal to `width`).
    pub y_stride: usize,
    /// Row stride for the U and V planes in bytes (equal to `width / 2`).
    pub uv_stride: usize,
}

impl Yuv420PlanarFrame {
    /// Creates a new empty `Yuv420PlanarFrame` with allocated planar buffers.
    ///
    /// # Arguments
    /// * `width` - Target width (must be even, or will be rounded up to the nearest even number).
    /// * `height` - Target height (must be even, or will be rounded up to the nearest even number).
    #[must_use]
    #[inline]
    pub fn new(width: u32, height: u32) -> Self {
        let w = (width + 1) & !1;
        let h = (height + 1) & !1;
        let y_stride = w as usize;
        let uv_stride = (w / 2) as usize;
        let y_size = y_stride * (h as usize);
        let uv_size = uv_stride * ((h / 2) as usize);

        Self {
            width: w,
            height: h,
            y_plane: vec![0u8; y_size],
            u_plane: vec![128u8; uv_size],
            v_plane: vec![128u8; uv_size],
            y_stride,
            uv_stride,
        }
    }

    /// Borrows the contiguous luminance (Y) plane slice.
    #[must_use]
    #[inline]
    pub fn y_slice(&self) -> &[u8] {
        &self.y_plane
    }

    /// Borrows the contiguous chrominance Cb (U) plane slice.
    #[must_use]
    #[inline]
    pub fn u_slice(&self) -> &[u8] {
        &self.u_plane
    }

    /// Borrows the contiguous chrominance Cr (V) plane slice.
    #[must_use]
    #[inline]
    pub fn v_slice(&self) -> &[u8] {
        &self.v_plane
    }

    /// Converts the Y (luma) plane directly to an 8-bit [`image::GrayImage`].
    ///
    /// Useful for feeding keypoint detectors (SuperPoint) and edge feature extractors without recomputing luma.
    #[must_use]
    pub fn to_gray_image(&self) -> image::GrayImage {
        image::GrayImage::from_raw(self.width, self.height, self.y_plane.clone())
            .unwrap_or_else(|| image::GrayImage::new(self.width, self.height))
    }
}

/// Fast parallel RGBA to YUV420p converter using BT.709 matrix multiplication.
pub struct RgbaToYuv420Converter;

impl RgbaToYuv420Converter {
    /// Converts an [`image::RgbaImage`] to a [`Yuv420PlanarFrame`] using vectorized BT.709 matrix multiplication.
    ///
    /// Iterates over contiguous raw byte slices via Rayon for maximum L1 cache efficiency and SIMD auto-vectorization.
    ///
    /// # Arguments
    /// * `rgba` - Input RGBA image.
    ///
    /// # Examples
    /// ```
    /// use image::RgbaImage;
    /// use reto_core::{RgbaToYuv420Converter, Yuv420PlanarFrame};
    ///
    /// let img = RgbaImage::new(64, 64);
    /// let yuv = RgbaToYuv420Converter::convert(&img);
    /// assert_eq!(yuv.width, 64);
    /// assert_eq!(yuv.height, 64);
    /// ```
    #[must_use]
    #[allow(clippy::similar_names)]
    pub fn convert(rgba: &RgbaImage) -> Yuv420PlanarFrame {
        let (orig_w, orig_h) = rgba.dimensions();
        let target_w = (orig_w + 1) & !1;
        let target_h = (orig_h + 1) & !1;
        let mut frame = Yuv420PlanarFrame::new(target_w, target_h);

        let y_stride = frame.y_stride;
        let uv_stride = frame.uv_stride;
        let half_w = (target_w / 2) as usize;
        let raw_rgba = rgba.as_raw();
        let raw_stride = (orig_w * 4) as usize;

        let [y_weights, u_weights, v_weights] = BT709_YUV_MATRIX;

        // 1. Vectorized Y plane extraction over contiguous row slices
        frame
            .y_plane
            .par_chunks_exact_mut(y_stride)
            .enumerate()
            .for_each(|(y_idx, y_row)| {
                let src_y = (y_idx as u32).min(orig_h.saturating_sub(1)) as usize;
                let row_start = src_y * raw_stride;
                let row_bytes = &raw_rgba[row_start..row_start + raw_stride];

                let chunks = row_bytes.as_chunks::<4>().0;
                let valid_count = chunks.len().min(y_row.len());

                for (out_y, chunk) in y_row[..valid_count].iter_mut().zip(chunks) {
                    let r = f32::from(chunk[0]);
                    let g = f32::from(chunk[1]);
                    let b = f32::from(chunk[2]);
                    *out_y = (y_weights[0] * r + y_weights[1] * g + y_weights[2] * b)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                }

                // If target width exceeds original (odd width padding), duplicate edge luma
                if valid_count < y_row.len() && valid_count > 0 {
                    let last_val = y_row[valid_count - 1];
                    for out_y in &mut y_row[valid_count..] {
                        *out_y = last_val;
                    }
                }
            });

        // 2. Vectorized 2x2 box subsampling for U and V chroma planes
        frame
            .u_plane
            .par_chunks_exact_mut(uv_stride)
            .zip(frame.v_plane.par_chunks_exact_mut(uv_stride))
            .enumerate()
            .for_each(|(uv_y, (u_row, v_row))| {
                let y0 = uv_y * 2;
                let y1 = (y0 + 1).min(orig_h.saturating_sub(1) as usize);
                let y0_clamped = y0.min(orig_h.saturating_sub(1) as usize);
                let row0_start = y0_clamped * raw_stride;
                let row1_start = y1 * raw_stride;

                let row0_chunks = raw_rgba[row0_start..row0_start + raw_stride].as_chunks::<4>().0;
                let row1_chunks = raw_rgba[row1_start..row1_start + raw_stride].as_chunks::<4>().0;
                let max_x = row0_chunks.len().saturating_sub(1);

                for (uv_x, (u_out, v_out)) in u_row.iter_mut().zip(v_row.iter_mut()).take(half_w).enumerate() {
                    let x0 = uv_x * 2;
                    let x1 = (x0 + 1).min(max_x);
                    let x0_clamped = x0.min(max_x);

                    let p00 = row0_chunks[x0_clamped];
                    let p01 = row0_chunks[x1];
                    let p10 = row1_chunks[x0_clamped];
                    let p11 = row1_chunks[x1];

                    let avg_r = (f32::from(p00[0]) + f32::from(p01[0]) + f32::from(p10[0]) + f32::from(p11[0])) * 0.25;
                    let avg_g = (f32::from(p00[1]) + f32::from(p01[1]) + f32::from(p10[1]) + f32::from(p11[1])) * 0.25;
                    let avg_b = (f32::from(p00[2]) + f32::from(p01[2]) + f32::from(p10[2]) + f32::from(p11[2])) * 0.25;

                    *u_out = (u_weights[0] * avg_r + u_weights[1] * avg_g + u_weights[2] * avg_b + 128.0)
                        .round()
                        .clamp(0.0, 255.0) as u8;

                    *v_out = (v_weights[0] * avg_r + v_weights[1] * avg_g + v_weights[2] * avg_b + 128.0)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                }

                if half_w < u_row.len() && half_w > 0 {
                    let last_u = u_row[half_w - 1];
                    let last_v = v_row[half_w - 1];
                    for u in &mut u_row[half_w..] {
                        *u = last_u;
                    }
                    for v in &mut v_row[half_w..] {
                        *v = last_v;
                    }
                }
            });

        frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;
    use ndarray::arr2;

    #[test]
    fn test_bt709_luma_converter() {
        let converter = Bt709LumaConverter::new();
        assert_eq!(converter.rgb_to_luma(255, 255, 255), 255);
        assert_eq!(converter.rgb_to_luma(0, 0, 0), 0);
        let pure_green = converter.rgb_to_luma(0, 255, 0);
        assert_eq!(pure_green, (0.7152 * 255.0_f32).round() as u8);
    }

    #[test]
    fn test_simple_gray_converter() {
        let converter = SimpleGrayConverter::new();
        assert_eq!(converter.rgb_to_luma(30, 60, 90), 60);
    }

    #[test]
    fn test_bt709_tensor_dot() {
        let converter = Bt709LumaConverter::new();
        let matrix = arr2(&[[255.0, 255.0, 255.0], [0.0, 0.0, 0.0]]);
        let luma = converter.convert_rgb_tensor(&matrix.view());
        assert_eq!(luma.len(), 2);
        assert!((luma[0] - 255.0).abs() < 1e-4);
        assert!((luma[1] - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_yuv420_conversion_dimensions() {
        let mut img = RgbaImage::new(101, 75);
        for pixel in img.pixels_mut() {
            *pixel = Rgba([100, 150, 200, 255]);
        }

        let frame = RgbaToYuv420Converter::convert(&img);
        assert_eq!(frame.width, 102);
        assert_eq!(frame.height, 76);
        assert_eq!(frame.y_plane.len(), 102 * 76);
        assert_eq!(frame.u_plane.len(), 51 * 38);
        assert_eq!(frame.v_plane.len(), 51 * 38);
    }

    #[test]
    fn test_pure_color_conversion_bt709() {
        let mut img = RgbaImage::new(4, 4);
        for pixel in img.pixels_mut() {
            *pixel = Rgba([255, 255, 255, 255]);
        }

        let frame = RgbaToYuv420Converter::convert(&img);
        for &y in frame.y_slice() {
            assert_eq!(y, 255);
        }
        for &u in frame.u_slice() {
            assert_eq!(u, 128);
        }
        for &v in frame.v_slice() {
            assert_eq!(v, 128);
        }
    }
}
