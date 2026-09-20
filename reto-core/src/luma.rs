//! Grayscale and Luma conversion interfaces, downscaling, and orientation-delegated image representations.

use crate::error::RoiError;
use crate::geom::StripOrientation;
use image::{GenericImageView, Pixel};
use num_traits::ToPrimitive;
use rayon::prelude::*;

/// Target minor (shortest) dimension when downscaling for fast projection analysis (e.g. 1080px).
pub const PROJECTION_MAX_DIMENSION: u32 = 1080;

/// Default inverse gamma power exponent for shadow contrast expansion ($\gamma \ge 1.0$).
pub const DEFAULT_INVERSE_GAMMA: f32 = 1.12;

/// Default slice percentile rank along cross-axis (5th percentile = 0.05).
pub const DEFAULT_CROSS_PERCENTILE: f32 = 0.05;

/// Default baseline percentile rank across major axis slice profiles (1st percentile = 0.01).
pub const DEFAULT_PROFILE_PERCENTILE: f32 = 0.01;

use crate::color::LumaConverter;

/// Scaled single-channel 8-bit luma representation of an image oriented along its major stacking axis.
///
/// Uses [`crate::geom::OrientationDelegator`] for all coordinate mapping and axis operations,
/// guaranteeing complete orientation-agnostic behavior for both horizontal and vertical layouts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScaledLumaImage {
    /// 1D contiguous pixel buffer of size `major_len * minor_len`.
    pub buffer: Vec<u8>,
    /// Number of pixels along major stacking dimension.
    pub major_len: u32,
    /// Number of pixels along minor cross dimension.
    pub minor_len: u32,
    /// Total width of the scaled 2D image in pixels.
    pub width: u32,
    /// Total height of the scaled 2D image in pixels.
    pub height: u32,
    /// Orientation of the image layout.
    pub orientation: StripOrientation,
    /// Applied scaling factor relative to the original image dimensions.
    pub scale_factor_bits: u32,
}

/// Backward-compatible alias for [`ScaledLumaImage`].
pub type ScaledGrayscaleStrip = ScaledLumaImage;

impl ScaledLumaImage {
    /// Converts and optionally downscales an input image view into a scaled luma representation.
    ///
    /// The major axis is clamped to `max_dimension` (e.g. [`PROJECTION_MAX_DIMENSION`]), preserving aspect ratio.
    /// All pixel lookups follow the delegation pattern via [`crate::geom::OrientationDelegator::to_xy`].
    /// Rows are processed in parallel using chunked contiguous slices for vectorization.
    ///
    /// # Arguments
    /// * `image` - Source image view to convert.
    /// * `converter` - Luma conversion implementation (e.g. [`Bt709LumaConverter`] or [`SimpleGrayConverter`]).
    /// * `max_dimension` - Maximum allowed dimension along the major stacking axis.
    ///
    /// # Errors
    /// Returns [`RoiError::ImageTooSmall`] if image dimensions are zero or invalid.
    /// Returns [`RoiError::SquareImageNotSupported`] if the input image is square.
    ///
    /// # Examples
    /// ```
    /// use reto_core::{Bt709LumaConverter, ScaledLumaImage, PROJECTION_MAX_DIMENSION};
    /// use image::{Rgba, RgbaImage};
    ///
    /// let img = RgbaImage::from_pixel(300, 100, Rgba([200, 200, 200, 255]));
    /// let converter = Bt709LumaConverter::new();
    /// let luma = ScaledLumaImage::from_image(&img, &converter, PROJECTION_MAX_DIMENSION).unwrap();
    /// assert_eq!(luma.major_len, 300);
    /// assert_eq!(luma.minor_len, 100);
    /// ```
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::many_single_char_names
    )]
    #[tracing::instrument(skip(image, converter), level = "debug")]
    pub fn from_image<I: GenericImageView + Sync, C: LumaConverter>(
        image: &I,
        converter: &C,
        max_dimension: u32,
    ) -> Result<Self, RoiError> {
        let (orig_w, orig_h) = image.dimensions();
        let orig_size = crate::geom::Size2D::new(orig_w, orig_h);
        let orientation = StripOrientation::from_size(orig_size)?;
        let delegator = orientation.delegator();

        let orig_major = delegator.major_dimension(orig_size);
        let orig_minor = delegator.minor_dimension(orig_size);

        let scale_factor = if orig_minor > max_dimension {
            max_dimension as f32 / orig_minor as f32
        } else {
            1.0_f32
        };

        let major_len = ((orig_major as f32 * scale_factor).round() as u32).max(1);
        let minor_len = ((orig_minor as f32 * scale_factor).round() as u32).max(1);
        let (width, height) = delegator.to_xy(major_len, minor_len);

        let mut buffer = vec![0_u8; (width * height) as usize];

        // Process standard image 2D scanlines (y * width + x) in parallel across Rayon worker threads
        buffer
            .par_chunks_mut(width as usize)
            .enumerate()
            .for_each(|(y_idx, row_slice)| {
                let y = y_idx as u32;
                let src_y = ((y as f32 / scale_factor).round() as u32).min(orig_h - 1);

                for (x_idx, out_pixel) in row_slice.iter_mut().enumerate() {
                    let x = x_idx as u32;
                    let src_x = ((x as f32 / scale_factor).round() as u32).min(orig_w - 1);

                    let pixel = image.get_pixel(src_x, src_y);
                    let channels = pixel.channels();
                    let r = channels[0].to_u8().unwrap_or(0);
                    let g = channels.get(1).and_then(ToPrimitive::to_u8).unwrap_or(r);
                    let b = channels.get(2).and_then(ToPrimitive::to_u8).unwrap_or(r);

                    *out_pixel = converter.rgb_to_luma(r, g, b);
                }
            });

        Ok(Self {
            buffer,
            major_len,
            minor_len,
            width,
            height,
            orientation,
            scale_factor_bits: scale_factor.to_bits(),
        })
    }

    /// Scaling factor applied relative to the source image.
    #[must_use]
    pub const fn scale_factor(&self) -> f32 {
        f32::from_bits(self.scale_factor_bits)
    }

    /// Gets the scalar luma value at stacking coordinate `(major, cross)`.
    ///
    /// # Arguments
    /// * `major` - Offset along major stacking axis `[0, major_len)`.
    /// * `cross` - Offset along minor cross axis `[0, minor_len)`.
    #[inline]
    #[must_use]
    pub fn get(&self, major: u32, cross: u32) -> u8 {
        let (x, y) = self.orientation.delegator().to_xy(major, cross);
        let idx = (y * self.width + x) as usize;
        self.buffer[idx]
    }

    /// Gets the scalar luma value at 2D image coordinates `(x, y)`.
    ///
    /// # Arguments
    /// * `x` - Horizontal pixel coordinate `[0, width)`.
    /// * `y` - Vertical pixel coordinate `[0, height)`.
    #[inline]
    #[must_use]
    pub fn get_xy(&self, x: u32, y: u32) -> u8 {
        let idx = (y * self.width + x) as usize;
        self.buffer[idx]
    }

    /// Converts the scaled grayscale buffer into an owned `image::GrayImage`.
    ///
    /// # Examples
    /// ```
    /// use reto_core::{Bt709LumaConverter, ScaledGrayscaleStrip, PROJECTION_MAX_DIMENSION};
    /// use image::{Rgba, RgbaImage};
    ///
    /// let img = RgbaImage::from_pixel(300, 100, Rgba([128, 128, 128, 255]));
    /// let strip = ScaledGrayscaleStrip::from_image(&img, &Bt709LumaConverter::new(), PROJECTION_MAX_DIMENSION).unwrap();
    /// let gray = strip.to_gray_image();
    /// assert_eq!(gray.dimensions(), (300, 100));
    /// ```
    #[must_use]
    pub fn to_gray_image(&self) -> image::GrayImage {
        image::GrayImage::from_raw(self.width, self.height, self.buffer.clone())
            .unwrap_or_else(|| image::GrayImage::new(self.width, self.height))
    }

    /// Applies a 2D 3x3 branchless sorting network median filter in-place across all image scanlines in parallel.
    ///
    /// Removes isolated single-pixel and speckle noise while preserving 100% sharp horizontal and vertical edges.
    ///
    /// # Examples
    /// ```
    /// use reto_core::{Bt709LumaConverter, ScaledGrayscaleStrip, PROJECTION_MAX_DIMENSION};
    /// use image::{Rgba, RgbaImage};
    ///
    /// let mut img = RgbaImage::from_pixel(200, 100, Rgba([100, 100, 100, 255]));
    /// img.put_pixel(50, 50, Rgba([255, 255, 255, 255])); // Isolated speckle
    /// let mut strip = ScaledGrayscaleStrip::from_image(&img, &Bt709LumaConverter::new(), PROJECTION_MAX_DIMENSION).unwrap();
    /// strip.median_filter_3x3();
    /// assert_eq!(strip.get_xy(50, 50), 100); // Speckle eliminated in-place
    /// ```
    #[allow(clippy::cast_possible_truncation)]
    pub fn median_filter_3x3(&mut self) {
        let width = self.width;
        let height = self.height;

        if width < 3 || height < 3 {
            return;
        }

        let source_buffer = self.buffer.clone();

        // Process interior scanlines [1..height-1] in parallel across Rayon worker threads
        self.buffer
            .par_chunks_mut(width as usize)
            .enumerate()
            .for_each(|(y_idx, row_slice)| {
                let y = y_idx as u32;
                if y == 0 || y == height - 1 {
                    return;
                }

                let y_stride = y as usize * width as usize;
                let prev_stride = (y as usize - 1) * width as usize;
                let next_stride = (y as usize + 1) * width as usize;

                for x in 1..width - 1 {
                    let x_u = x as usize;
                    let p0 = source_buffer[prev_stride + x_u - 1];
                    let p1 = source_buffer[prev_stride + x_u];
                    let p2 = source_buffer[prev_stride + x_u + 1];
                    let p3 = source_buffer[y_stride + x_u - 1];
                    let p4 = source_buffer[y_stride + x_u];
                    let p5 = source_buffer[y_stride + x_u + 1];
                    let p6 = source_buffer[next_stride + x_u - 1];
                    let p7 = source_buffer[next_stride + x_u];
                    let p8 = source_buffer[next_stride + x_u + 1];

                    row_slice[x_u] = median9([p0, p1, p2, p3, p4, p5, p6, p7, p8]);
                }
            });
    }

    /// Returns the buffer dimensions as a structured `Size2D<u32>`.
    #[inline]
    #[must_use]
    pub const fn size(&self) -> crate::geom::Size2D<u32> {
        crate::geom::Size2D::new(self.width, self.height)
    }

    /// Extracts the percentile intensity along the perpendicular slice at `major_idx`
    /// using orientation-delegated stride indexing over pre-computed dimensions.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn slice_percentile(&self, major_idx: u32, percentile: f32) -> u8 {
        let minor_len = self.minor_len;
        if minor_len == 0 {
            return 0;
        }

        let (start, stride) = self
            .orientation
            .delegator()
            .slice_stride(major_idx, self.size());

        let target_rank = ((minor_len as f32 * percentile.clamp(0.0, 1.0)).round() as u32)
            .min(minor_len.saturating_sub(1));
        let mut hist = [0_u32; 256];
        let mut curr_idx = start;

        for _ in 0..minor_len {
            hist[self.buffer[curr_idx] as usize] += 1;
            curr_idx += stride;
        }

        let mut cumulative = 0_u32;
        for (val, &count) in hist.iter().enumerate() {
            cumulative += count;
            if cumulative >= target_rank {
                return val as u8;
            }
        }
        0
    }

    /// Computes the $`P_1`$ percentile across all major-axis $`P_5(x)`$ slices ($`P_1`$ of $`P_5`$'s),
    /// providing a noise-resistant, edge-immune baseline floor for film gutters.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn compute_axis_p1_of_p5(&self, cross_percentile: f32, profile_percentile: f32) -> u8 {
        let major_len = self.major_len;
        if major_len == 0 || self.minor_len == 0 {
            return 0;
        }

        let p5_slices: Vec<u8> = (0..major_len)
            .into_par_iter()
            .map(|m| self.slice_percentile(m, cross_percentile))
            .collect();

        let mut hist = [0_u32; 256];
        for &v in &p5_slices {
            hist[v as usize] += 1;
        }

        let target_rank = ((major_len as f32 * profile_percentile.clamp(0.0, 1.0)).round() as u32)
            .min(major_len.saturating_sub(1));
        let mut cum = 0_u32;
        for (val, &count) in hist.iter().enumerate() {
            cum += count;
            if cum >= target_rank {
                return val as u8;
            }
        }
        0
    }

    /// Performs baseline subtraction and inverse gamma expansion in-place using an explicit optical baseline floor.
    ///
    /// Subtracts the $`P_1`$ of $`P_5`$'s baseline ($B_{\min}$), so that the lowest gutter
    /// baseline shifts towards 0 without flattening intra-gutter variance, and applies inverse gamma scaling:
    ///
    /// $$\text{stretched}(v) = \text{clamp}\left(255.0 \times \left(\frac{\max(0, v - B_{\min})}{255.0 - B_{\min}}\right)^\gamma, 0, 255\right)$$
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn inverse_gamma_stretch_with_baseline(&mut self, gamma: f32, baseline: u8) {
        if self.buffer.is_empty() {
            return;
        }

        let safe_gamma = if gamma > 0.1 { gamma } else { 1.0 };
        let denom = (255.0_f32 - f32::from(baseline)).max(1.0);

        let mut lut = [0_u8; 256];
        for (i, slot) in lut.iter_mut().enumerate() {
            let v = i as u8;
            if v <= baseline {
                *slot = 0;
            } else {
                let norm = (f32::from(v - baseline) / denom).clamp(0.0, 1.0);
                let stretched = (norm.powf(safe_gamma) * 255.0).round();
                *slot = (stretched.clamp(0.0, 255.0)) as u8;
            }
        }

        for p in &mut self.buffer {
            *p = lut[*p as usize];
        }
    }

    /// Performs baseline subtraction and inverse gamma expansion in-place using the $`P_1`$ of $`P_5`$'s baseline.
    pub fn inverse_gamma_stretch(&mut self, gamma: f32) {
        let baseline =
            self.compute_axis_p1_of_p5(DEFAULT_CROSS_PERCENTILE, DEFAULT_PROFILE_PERCENTILE);
        self.inverse_gamma_stretch_with_baseline(gamma, baseline);
    }
}

#[inline]
const fn sort_pair(p: &mut [u8; 9], a: usize, b: usize) {
    if p[a] > p[b] {
        let temp = p[a];
        p[a] = p[b];
        p[b] = temp;
    }
}

/// Branchless 19-comparison sorting network computing the exact median of 9 values.
#[inline]
#[must_use]
pub const fn median9(mut p: [u8; 9]) -> u8 {
    sort_pair(&mut p, 1, 2);
    sort_pair(&mut p, 4, 5);
    sort_pair(&mut p, 7, 8);
    sort_pair(&mut p, 0, 1);
    sort_pair(&mut p, 3, 4);
    sort_pair(&mut p, 6, 7);
    sort_pair(&mut p, 1, 2);
    sort_pair(&mut p, 4, 5);
    sort_pair(&mut p, 7, 8);
    sort_pair(&mut p, 0, 3);
    sort_pair(&mut p, 5, 8);
    sort_pair(&mut p, 4, 7);
    sort_pair(&mut p, 3, 6);
    sort_pair(&mut p, 1, 4);
    sort_pair(&mut p, 2, 5);
    sort_pair(&mut p, 4, 7);
    sort_pair(&mut p, 4, 2);
    sort_pair(&mut p, 6, 4);
    sort_pair(&mut p, 4, 2);
    p[4]
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::color::{Bt709LumaConverter, SimpleGrayConverter};
    use image::{Rgba, RgbaImage};

    #[test]
    fn test_simple_gray_converter() {
        let conv = SimpleGrayConverter::new();
        assert_eq!(conv.rgb_to_luma(0, 0, 0), 0);
        assert_eq!(conv.rgb_to_luma(255, 255, 255), 255);
        assert_eq!(conv.rgb_to_luma(100, 150, 200), 150);

        let rgb = [100, 150, 200, 30, 60, 90];
        let mut out = [0_u8; 2];
        conv.convert_rgb_slice(&rgb, &mut out);
        assert_eq!(out[0], 150);
        assert_eq!(out[1], 60);

        let rgba = [100, 150, 200, 255, 30, 60, 90, 255];
        let mut out_rgba = [0_u8; 2];
        conv.convert_rgba_slice(&rgba, &mut out_rgba);
        assert_eq!(out_rgba[0], 150);
        assert_eq!(out_rgba[1], 60);
    }

    #[test]
    fn test_bt709_luma_converter() {
        let conv = Bt709LumaConverter::new();
        assert_eq!(conv.rgb_to_luma(0, 0, 0), 0);
        assert_eq!(conv.rgb_to_luma(255, 255, 255), 255);
        // Green should have highest weight (~71.5%)
        let green_luma = conv.rgb_to_luma(0, 255, 0);
        let red_luma = conv.rgb_to_luma(255, 0, 0);
        let blue_luma = conv.rgb_to_luma(0, 0, 255);
        assert!(green_luma > red_luma);
        assert!(red_luma > blue_luma);

        let rgb = [0, 255, 0, 255, 0, 0];
        let mut out = [0_u8; 2];
        conv.convert_rgb_slice(&rgb, &mut out);
        assert_eq!(out[0], green_luma);
        assert_eq!(out[1], red_luma);
    }

    #[test]
    fn test_scaled_grayscale_strip_horizontal_delegation() {
        let img = RgbaImage::from_pixel(6000, 2000, Rgba([120, 120, 120, 255]));
        let conv = Bt709LumaConverter::new();
        let strip =
            ScaledGrayscaleStrip::from_image(&img, &conv, 1080).expect("Conversion succeeds");

        assert_eq!(strip.orientation, StripOrientation::Horizontal);
        assert_eq!(strip.major_len, 3240);
        assert_eq!(strip.minor_len, 1080);
        assert_eq!(strip.width, 3240);
        assert_eq!(strip.height, 1080);
        assert_eq!(strip.get(100, 50), 120);

        let gray = strip.to_gray_image();
        assert_eq!(gray.dimensions(), (3240, 1080));
        assert_eq!(gray.get_pixel(100, 50)[0], 120);
    }

    #[test]
    fn test_scaled_grayscale_strip_vertical_delegation() {
        let mut img = RgbaImage::from_pixel(2000, 6000, Rgba([80, 80, 80, 255]));
        // Put a unique marker at (x=200, y=750)
        img.put_pixel(200, 750, Rgba([200, 200, 200, 255]));

        let conv = SimpleGrayConverter::new();
        let strip =
            ScaledGrayscaleStrip::from_image(&img, &conv, 1080).expect("Conversion succeeds");

        assert_eq!(strip.orientation, StripOrientation::Vertical);
        assert_eq!(strip.major_len, 3240);
        assert_eq!(strip.minor_len, 1080);
        assert_eq!(strip.width, 1080);
        assert_eq!(strip.height, 3240);

        let gray = strip.to_gray_image();
        assert_eq!(gray.dimensions(), (1080, 3240));
    }

    #[test]
    fn test_median9() {
        let p = [10, 50, 30, 90, 20, 80, 70, 40, 60];
        assert_eq!(median9(p), 50);

        let p_ties = [5, 5, 5, 5, 5, 10, 20, 30, 40];
        assert_eq!(median9(p_ties), 5);
    }

    #[test]
    fn test_scaled_grayscale_strip_median_filtering() {
        let mut img = RgbaImage::from_pixel(200, 100, Rgba([80, 80, 80, 255]));
        // Put isolated 1-pixel bright speckle at (50, 50) and dark speckle at (20, 20)
        img.put_pixel(50, 50, Rgba([255, 255, 255, 255]));
        img.put_pixel(20, 20, Rgba([0, 0, 0, 255]));

        let conv = SimpleGrayConverter::new();
        let strip =
            ScaledGrayscaleStrip::from_image(&img, &conv, PROJECTION_MAX_DIMENSION).unwrap();
        assert_eq!(strip.get_xy(50, 50), 255);
        assert_eq!(strip.get_xy(20, 20), 0);

        let mut denoised = strip;
        denoised.median_filter_3x3();
        assert_eq!(denoised.get_xy(50, 50), 80);
        assert_eq!(denoised.get_xy(20, 20), 80);
    }

    #[test]
    fn test_scaled_grayscale_strip_inverse_gamma_stretch() {
        // Create 100x50 strip with base intensity 50
        let mut img = RgbaImage::from_pixel(100, 50, Rgba([50, 50, 50, 255]));
        // Make 200 pixels have intensity 20 (>= 4% of pixels) so P5 is around 20..50
        for x in 0..10 {
            for y in 0..20 {
                img.put_pixel(x, y, Rgba([20, 20, 20, 255]));
            }
        }
        // Single dead pixel with intensity 0 (should not break baseline P5)
        img.put_pixel(0, 0, Rgba([0, 0, 0, 255]));
        img.put_pixel(90, 40, Rgba([220, 220, 220, 255]));

        let conv = SimpleGrayConverter::new();
        let mut strip =
            ScaledGrayscaleStrip::from_image(&img, &conv, PROJECTION_MAX_DIMENSION).unwrap();

        strip.inverse_gamma_stretch(1.5);
        // Single 0 pixel and baseline pixels (20) should map to 0
        assert_eq!(strip.get_xy(0, 0), 0);
        assert_eq!(strip.get_xy(5, 5), 0);
        // Max pixel (220) should stretch higher
        assert!(strip.get_xy(90, 40) > 150);
    }
}
