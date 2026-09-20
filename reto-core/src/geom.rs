//! Geometric coordinate types, orientation definitions, and zero-copy frame `RoI` views.

use crate::error::RoiError;
use image::{GenericImageView, SubImage};
use ndarray::{ArrayView2, ArrayViewMut2, Axis};
use serde::{Deserialize, Serialize};

/// 2D dimension container representing width and height.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Size2D<T = u32> {
    /// Width dimension.
    pub width: T,
    /// Height dimension.
    pub height: T,
}

impl<T> Size2D<T> {
    /// Creates a new `Size2D` with specified width and height.
    #[inline]
    #[must_use]
    pub const fn new(width: T, height: T) -> Self {
        Self { width, height }
    }
}

/// Concrete 2D coordinate point in a specific reference space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
pub struct Point2D<T = f32> {
    /// Horizontal X coordinate.
    pub x: T,
    /// Vertical Y coordinate.
    pub y: T,
}

impl<T> Point2D<T> {
    /// Creates a new `Point2D` with specified coordinates.
    #[inline]
    #[must_use]
    pub const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

impl Point2D<f32> {
    /// Converts a local coordinate point (within an ROI) to the global scan space.
    ///
    /// # Arguments
    /// * `roi_bounds` - The bounding rectangle of the sub-frame in normalized unit space.
    /// * `strip_size` - Full scan strip dimensions (width, height).
    ///
    /// # Examples
    /// ```
    /// use reto_core::{NormalizedRect, Point2D, Size2D};
    ///
    /// let roi = NormalizedRect::new(0.5, 0.0, 0.5, 1.0).unwrap();
    /// let strip_size = Size2D::new(200, 100);
    /// let local_pt = Point2D::new(10.0, 20.0);
    /// let global_pt = local_pt.to_global(roi, strip_size);
    /// assert_eq!(global_pt.x, 110.0);
    /// assert_eq!(global_pt.y, 20.0);
    /// ```
    #[inline]
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_global(self, roi_bounds: NormalizedRect, strip_size: Size2D<u32>) -> Self {
        let px = roi_bounds.to_pixel_rect(strip_size);
        Self {
            x: self.x + px.x as f32,
            y: self.y + px.y as f32,
        }
    }

    /// Converts a global scan strip coordinate point into an ROI's local sub-frame space.
    ///
    /// # Arguments
    /// * `roi_bounds` - The bounding rectangle of the sub-frame.
    /// * `strip_size` - Full scan strip dimensions (width, height).
    ///
    /// # Returns
    /// Returns `Some(Point2D)` if the point lies inside the sub-frame bounds, else `None`.
    ///
    /// # Examples
    /// ```
    /// use reto_core::{NormalizedRect, Point2D, Size2D};
    ///
    /// let roi = NormalizedRect::new(0.5, 0.0, 0.5, 1.0).unwrap();
    /// let strip_size = Size2D::new(200, 100);
    /// let global_pt = Point2D::new(110.0, 20.0);
    /// let local_pt = global_pt.to_local(roi, strip_size).unwrap();
    /// assert_eq!(local_pt.x, 10.0);
    /// assert_eq!(local_pt.y, 20.0);
    /// ```
    #[inline]
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_local(self, roi_bounds: NormalizedRect, strip_size: Size2D<u32>) -> Option<Self> {
        let px = roi_bounds.to_pixel_rect(strip_size);
        let lx = self.x - px.x as f32;
        let ly = self.y - px.y as f32;
        if lx >= 0.0 && ly >= 0.0 && lx < px.width as f32 && ly < px.height as f32 {
            Some(Self { x: lx, y: ly })
        } else {
            None
        }
    }
}

/// A coordinate point anchored in the frame's local bounding box space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LocalCoord {
    /// Relative pixel offset within the sub-frame `[0, width) x [0, height)`.
    pub pixel: Point2D<f32>,
    /// Relative unit coordinates `[0.0, 1.0]^2` within the sub-frame.
    pub normalized: Point2D<f32>,
}

impl LocalCoord {
    /// Constructs a new `LocalCoord` from sub-frame pixel and normalized coordinates.
    #[inline]
    #[must_use]
    pub const fn new(pixel: Point2D<f32>, normalized: Point2D<f32>) -> Self {
        Self { pixel, normalized }
    }

    /// Converts this local coordinate into full strip global coordinate space.
    ///
    /// # Arguments
    /// * `roi_bounds` - The bounding rectangle of the sub-frame in normalized unit space.
    /// * `strip_size` - Full scan strip dimensions.
    ///
    /// # Examples
    /// ```
    /// use reto_core::{LocalCoord, NormalizedRect, Point2D, Size2D};
    ///
    /// let roi = NormalizedRect::new(0.5, 0.0, 0.5, 1.0).unwrap();
    /// let strip_size = Size2D::new(200, 100);
    /// let local = LocalCoord::new(Point2D::new(10.0, 20.0), Point2D::new(0.1, 0.2));
    /// let global = local.to_global(roi, strip_size);
    /// assert_eq!(global.pixel.x, 110.0);
    /// assert_eq!(global.pixel.y, 20.0);
    /// assert_eq!(global.normalized.x, 0.55);
    /// assert_eq!(global.normalized.y, 0.2);
    /// ```
    #[inline]
    #[must_use]
    #[allow(clippy::suboptimal_flops)]
    pub fn to_global(self, roi_bounds: NormalizedRect, strip_size: Size2D<u32>) -> GlobalCoord {
        GlobalCoord {
            pixel: self.pixel.to_global(roi_bounds, strip_size),
            normalized: Point2D::new(
                self.normalized.x.mul_add(roi_bounds.width, roi_bounds.x),
                self.normalized.y.mul_add(roi_bounds.height, roi_bounds.y),
            ),
        }
    }
}

/// A coordinate point anchored in the full strip scan space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GlobalCoord {
    /// Pixel coordinates `[0, strip_width) x [0, strip_height)`.
    pub pixel: Point2D<f32>,
    /// Unit coordinates `[0.0, 1.0]^2` across the entire scan strip.
    pub normalized: Point2D<f32>,
}

impl GlobalCoord {
    /// Constructs a new `GlobalCoord` from scan strip pixel and normalized coordinates.
    #[inline]
    #[must_use]
    pub const fn new(pixel: Point2D<f32>, normalized: Point2D<f32>) -> Self {
        Self { pixel, normalized }
    }

    /// Converts this global coordinate into an ROI's local sub-frame coordinate space.
    ///
    /// # Arguments
    /// * `roi_bounds` - The bounding rectangle of the sub-frame.
    /// * `strip_size` - Full scan strip dimensions.
    ///
    /// # Returns
    /// Returns `Some(LocalCoord)` if the coordinate falls inside the ROI bounds, else `None`.
    ///
    /// # Examples
    /// ```
    /// use reto_core::{GlobalCoord, NormalizedRect, Point2D, Size2D};
    ///
    /// let roi = NormalizedRect::new(0.5, 0.0, 0.5, 1.0).unwrap();
    /// let strip_size = Size2D::new(200, 100);
    /// let global = GlobalCoord::new(Point2D::new(110.0, 20.0), Point2D::new(0.55, 0.2));
    /// let local = global.to_local(roi, strip_size).unwrap();
    /// assert_eq!(local.pixel.x, 10.0);
    /// assert_eq!(local.pixel.y, 20.0);
    /// assert!((local.normalized.x - 0.1).abs() < 1e-6);
    /// assert!((local.normalized.y - 0.2).abs() < 1e-6);
    /// ```
    #[inline]
    #[must_use]
    pub fn to_local(
        self,
        roi_bounds: NormalizedRect,
        strip_size: Size2D<u32>,
    ) -> Option<LocalCoord> {
        let pixel = self.pixel.to_local(roi_bounds, strip_size)?;
        let norm_x = (self.normalized.x - roi_bounds.x) / roi_bounds.width;
        let norm_y = (self.normalized.y - roi_bounds.y) / roi_bounds.height;
        if norm_x >= 0.0 && norm_y >= 0.0 && norm_x <= 1.0001 && norm_y <= 1.0001 {
            Some(LocalCoord {
                pixel,
                normalized: Point2D::new(norm_x.clamp(0.0, 1.0), norm_y.clamp(0.0, 1.0)),
            })
        } else {
            None
        }
    }
}

/// Film strip scan orientation based on dimensional aspect ratio.
///
/// Invariant: Input images always have aspect ratio != 1.0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StripOrientation {
    /// Width > Height: A set of portrait frames stacked horizontally side-by-side.
    /// This is the primary format in practical RETO3D film strip scans.
    Horizontal,
    /// Height > Width: A set of landscape frames stacked vertically top-to-bottom.
    Vertical,
}

impl StripOrientation {
    /// Determines orientation from raw pixel dimensions.
    ///
    /// # Arguments
    /// * `width` - Source image width in pixels.
    /// * `height` - Source image height in pixels.
    ///
    /// # Errors
    /// Returns [`RoiError::ImageTooSmall`] if width or height is zero.
    /// Returns [`RoiError::SquareImageNotSupported`] if `width == height`.
    /// Determines the film orientation from explicit dimensions.
    ///
    /// # Errors
    /// Returns [`RoiError`] if dimensions are zero or the image is square.
    ///
    /// # Examples
    /// ```
    /// use reto_core::StripOrientation;
    ///
    /// let orientation = StripOrientation::from_dimensions(300, 100).unwrap();
    /// assert_eq!(orientation, StripOrientation::Horizontal);
    /// ```
    pub const fn from_dimensions(width: u32, height: u32) -> Result<Self, RoiError> {
        Self::from_size(Size2D::new(width, height))
    }

    /// Determines the film orientation from structured 2D dimensions.
    ///
    /// # Errors
    /// Returns [`RoiError`] if dimensions are zero or the image is square.
    pub const fn from_size(size: Size2D<u32>) -> Result<Self, RoiError> {
        let width = size.width;
        let height = size.height;
        if width == 0 || height == 0 {
            return Err(RoiError::ImageTooSmall { width, height });
        }
        if width > height {
            Ok(Self::Horizontal)
        } else if height > width {
            Ok(Self::Vertical)
        } else {
            Err(RoiError::SquareImageNotSupported { width, height })
        }
    }

    /// Selects the pre-defined delegator for this orientation.
    ///
    /// # Examples
    /// ```
    /// use reto_core::{Size2D, StripOrientation};
    ///
    /// let delegator = StripOrientation::Horizontal.delegator();
    /// assert_eq!(delegator.major_dimension(Size2D::new(300, 100)), 300);
    /// ```
    #[inline]
    #[must_use]
    pub fn delegator(self) -> &'static dyn OrientationDelegator {
        match self {
            Self::Horizontal => &HORIZONTAL_DELEGATOR,
            Self::Vertical => &VERTICAL_DELEGATOR,
        }
    }
}

/// Pre-defined delegator contract for orientation-specific axis operations.
///
/// Eliminates scattered control-flow branching throughout detection and alignment pipelines.
pub trait OrientationDelegator: Send + Sync {
    /// Length of the major stacking dimension (width for horizontal, height for vertical).
    fn major_dimension(&self, size: Size2D<u32>) -> u32;

    /// Length of the minor frame dimension (height for horizontal, width for vertical).
    fn minor_dimension(&self, size: Size2D<u32>) -> u32;

    /// Constructs a `NormalizedRect` from stacking axis bounds and cross axis bounds.
    fn build_rect(
        &self,
        stack_min: f32,
        stack_len: f32,
        cross_min: f32,
        cross_len: f32,
    ) -> NormalizedRect;

    /// Maps 1D stacking coordinates `(major, cross)` to 2D image pixel coordinates `(x, y)`.
    fn to_xy(&self, major: u32, cross: u32) -> (u32, u32);

    /// Maps 2D image pixel coordinates `(x, y)` to 1D stacking coordinates `(major, cross)`.
    fn to_stack_coords(&self, x: u32, y: u32) -> (u32, u32);

    /// Returns the `(start_index, stride)` in the flat 2D buffer for the perpendicular slice at `major_idx`.
    fn slice_stride(&self, major_idx: u32, size: Size2D<u32>) -> (usize, usize);

    /// Fills the neural network inference canvas with normalized luma values according to orientation layout.
    fn fill_inference_canvas(
        &self,
        canvas: &mut [f32],
        canvas_width: u32,
        scaled_img: &image::GrayImage,
        scaled_w: u32,
        scaled_h: u32,
    );

    /// Maps raw keypoint coordinates `(kx, ky)` extracted from the neural inference canvas back to scaled frame coordinates `(rx, ry)`.
    fn unmap_canvas_coords(&self, kx: f32, ky: f32) -> (f32, f32);
}

/// Delegator operations for horizontally-stacked portrait frames (width > height).
#[derive(Debug, Clone, Copy)]
pub struct HorizontalDelegator;

/// Delegator operations for vertically-stacked landscape frames (height > width).
#[derive(Debug, Clone, Copy)]
pub struct VerticalDelegator;

/// Static singleton delegator for horizontal strips.
pub static HORIZONTAL_DELEGATOR: HorizontalDelegator = HorizontalDelegator;

/// Static singleton delegator for vertical strips.
pub static VERTICAL_DELEGATOR: VerticalDelegator = VerticalDelegator;

impl OrientationDelegator for HorizontalDelegator {
    #[inline]
    fn major_dimension(&self, size: Size2D<u32>) -> u32 {
        size.width
    }

    #[inline]
    fn minor_dimension(&self, size: Size2D<u32>) -> u32 {
        size.height
    }

    #[inline]
    fn build_rect(
        &self,
        stack_min: f32,
        stack_len: f32,
        cross_min: f32,
        cross_len: f32,
    ) -> NormalizedRect {
        NormalizedRect {
            x: stack_min,
            y: cross_min,
            width: stack_len,
            height: cross_len,
        }
    }

    #[inline]
    fn to_xy(&self, major: u32, cross: u32) -> (u32, u32) {
        (major, cross)
    }

    #[inline]
    fn to_stack_coords(&self, x: u32, y: u32) -> (u32, u32) {
        (x, y)
    }

    #[inline]
    fn slice_stride(&self, major_idx: u32, size: Size2D<u32>) -> (usize, usize) {
        (major_idx as usize, size.width as usize)
    }

    #[inline]
    #[allow(clippy::cast_precision_loss)]
    fn fill_inference_canvas(
        &self,
        canvas: &mut [f32],
        canvas_width: u32,
        scaled_img: &image::GrayImage,
        scaled_w: u32,
        scaled_h: u32,
    ) {
        let Ok(src_view) = ArrayView2::<'_, u8>::from_shape(
            (scaled_h as usize, scaled_w as usize),
            scaled_img.as_raw(),
        ) else {
            return;
        };

        // Eager vectorized tensor normalization and transpose: [scaled_h, scaled_w] -> [scaled_w, scaled_h]
        let transposed = src_view.mapv(|p| f32::from(p) / 255.0).reversed_axes();

        let effective_w = (scaled_h as usize).min(canvas_width as usize);
        let canvas_height = canvas.len() / (canvas_width as usize);
        if let Ok(mut canvas_view) =
            ArrayViewMut2::<'_, f32>::from_shape((canvas_height, canvas_width as usize), canvas)
        {
            let row_limit = (scaled_w as usize).min(canvas_height);
            for (mut dst_row, src_row) in canvas_view
                .axis_iter_mut(Axis(0))
                .zip(transposed.axis_iter(Axis(0)))
                .take(row_limit)
            {
                if let Some(dst) = dst_row.as_slice_mut() {
                    for (d, &s) in dst[..effective_w]
                        .iter_mut()
                        .zip(src_row.iter().take(effective_w))
                    {
                        *d = s;
                    }
                }
            }
        }
    }

    #[inline]
    fn unmap_canvas_coords(&self, kx: f32, ky: f32) -> (f32, f32) {
        (ky, kx)
    }
}

impl OrientationDelegator for VerticalDelegator {
    #[inline]
    fn major_dimension(&self, size: Size2D<u32>) -> u32 {
        size.height
    }

    #[inline]
    fn minor_dimension(&self, size: Size2D<u32>) -> u32 {
        size.width
    }

    #[inline]
    fn build_rect(
        &self,
        stack_min: f32,
        stack_len: f32,
        cross_min: f32,
        cross_len: f32,
    ) -> NormalizedRect {
        NormalizedRect {
            x: cross_min,
            y: stack_min,
            width: cross_len,
            height: stack_len,
        }
    }

    #[inline]
    fn to_xy(&self, major: u32, cross: u32) -> (u32, u32) {
        (cross, major)
    }

    #[inline]
    fn to_stack_coords(&self, x: u32, y: u32) -> (u32, u32) {
        (y, x)
    }

    #[inline]
    fn slice_stride(&self, major_idx: u32, size: Size2D<u32>) -> (usize, usize) {
        ((major_idx * size.width) as usize, 1)
    }

    #[inline]
    #[allow(clippy::cast_precision_loss)]
    fn fill_inference_canvas(
        &self,
        canvas: &mut [f32],
        canvas_width: u32,
        scaled_img: &image::GrayImage,
        scaled_w: u32,
        scaled_h: u32,
    ) {
        let Ok(src_view) = ArrayView2::<'_, u8>::from_shape(
            (scaled_h as usize, scaled_w as usize),
            scaled_img.as_raw(),
        ) else {
            return;
        };

        // Eager vectorized tensor normalization [scaled_h, scaled_w]
        let normalized = src_view.mapv(|p| f32::from(p) / 255.0);

        let effective_w = (scaled_w as usize).min(canvas_width as usize);
        let canvas_height = canvas.len() / (canvas_width as usize);
        if let Ok(mut canvas_view) =
            ArrayViewMut2::<'_, f32>::from_shape((canvas_height, canvas_width as usize), canvas)
        {
            let row_limit = (scaled_h as usize).min(canvas_height);
            for (mut dst_row, src_row) in canvas_view
                .axis_iter_mut(Axis(0))
                .zip(normalized.axis_iter(Axis(0)))
                .take(row_limit)
            {
                if let Some(dst) = dst_row.as_slice_mut() {
                    for (d, &s) in dst[..effective_w]
                        .iter_mut()
                        .zip(src_row.iter().take(effective_w))
                    {
                        *d = s;
                    }
                }
            }
        }
    }

    #[inline]
    fn unmap_canvas_coords(&self, kx: f32, ky: f32) -> (f32, f32) {
        (kx, ky)
    }
}

/// Normalized 2D bounding box representing relative unit sub-regions in `[0.0, 1.0]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NormalizedRect {
    /// Normalized horizontal start coordinate in `[0.0, 1.0]`.
    pub x: f32,
    /// Normalized vertical start coordinate in `[0.0, 1.0]`.
    pub y: f32,
    /// Normalized width in `[0.0, 1.0]`.
    pub width: f32,
    /// Normalized height in `[0.0, 1.0]`.
    pub height: f32,
}

impl NormalizedRect {
    /// Constructs and validates a new `NormalizedRect`.
    ///
    /// # Errors
    /// Returns [`RoiError`] if values are negative, not finite, or if width/height is zero.
    ///
    /// # Examples
    /// ```
    /// use reto_core::NormalizedRect;
    ///
    /// let rect = NormalizedRect::new(0.0, 0.0, 0.5, 1.0).unwrap();
    /// assert_eq!(rect.width, 0.5);
    /// ```
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Result<Self, RoiError> {
        if !x.is_finite()
            || !y.is_finite()
            || !width.is_finite()
            || !height.is_finite()
            || x < 0.0_f32
            || y < 0.0_f32
            || width <= 0.0_f32
            || height <= 0.0_f32
            || (x + width) > 1.0001_f32
            || (y + height) > 1.0001_f32
        {
            return Err(RoiError::InvalidBounds {
                x,
                y,
                width,
                height,
            });
        }
        Ok(Self {
            x: x.clamp(0.0_f32, 1.0_f32),
            y: y.clamp(0.0_f32, 1.0_f32),
            width: width.min(1.0_f32 - x),
            height: height.min(1.0_f32 - y),
        })
    }

    /// Converts normalized bounds to concrete pixel coordinates for an image of dimensions `size`.
    ///
    /// # Arguments
    /// * `size` - Total image dimensions in pixels.
    ///
    /// # Examples
    /// ```
    /// use reto_core::{NormalizedRect, Size2D};
    ///
    /// let rect = NormalizedRect::new(0.0, 0.0, 0.5, 1.0).unwrap();
    /// let px = rect.to_pixel_rect(Size2D::new(200, 100));
    /// assert_eq!(px.width, 100);
    /// assert_eq!(px.height, 100);
    /// ```
    #[inline]
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub fn to_pixel_rect(self, size: Size2D<u32>) -> PixelRect {
        let x = (self.x * size.width as f32).round() as u32;
        let y = (self.y * size.height as f32).round() as u32;
        let width =
            ((self.width * size.width as f32).round() as u32).min(size.width.saturating_sub(x));
        let height =
            ((self.height * size.height as f32).round() as u32).min(size.height.saturating_sub(y));
        PixelRect {
            x,
            y,
            width,
            height,
        }
    }

    /// Checks whether a point in normalized coordinates $(px, py) \in [0.0, 1.0]^2$ is inside this bounding box.
    ///
    /// # Arguments
    /// * `px` - Normalized horizontal coordinate $x$.
    /// * `py` - Normalized vertical coordinate $y$.
    ///
    /// # Examples
    /// ```
    /// use reto_core::NormalizedRect;
    ///
    /// let rect = NormalizedRect::new(0.2, 0.2, 0.4, 0.4).unwrap();
    /// assert!(rect.contains_point(0.3, 0.3));
    /// assert!(!rect.contains_point(0.1, 0.3));
    /// ```
    #[inline]
    #[must_use]
    pub fn contains_point(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= (self.x + self.width) && py >= self.y && py <= (self.y + self.height)
    }
}

/// Concrete integer pixel rectangle bounding box.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PixelRect {
    /// Horizontal pixel offset from left.
    pub x: u32,
    /// Vertical pixel offset from top.
    pub y: u32,
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
}

impl PixelRect {
    /// Returns the rectangle dimensions as a `Size2D<u32>`.
    #[inline]
    #[must_use]
    pub const fn size(&self) -> Size2D<u32> {
        Size2D::new(self.width, self.height)
    }
}

/// Information about a single identified film frame `RoI`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameRoi {
    /// 0-indexed position within the strip (e.g. 0 = left, 1 = middle, 2 = right).
    pub index: usize,
    /// Bounding rectangle in normalized unit space.
    pub bounds: NormalizedRect,
    /// Confidence score `[0.0, 1.0]` of boundary detection.
    pub confidence: f32,
}

/// Ordered collection of detected film frames representing a single exposure strip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameRoiSet {
    /// Source image width at detection time.
    pub source_width: u32,
    /// Source image height at detection time.
    pub source_height: u32,
    /// Strip orientation (Horizontal for portrait frames, Vertical for landscape frames).
    pub orientation: StripOrientation,
    /// Detected frame regions in spatial stacking order.
    pub frames: Vec<FrameRoi>,
}

impl FrameRoiSet {
    /// Creates a new `FrameRoiSet`.
    ///
    /// # Arguments
    /// * `source_width` - Detected image pixel width.
    /// * `source_height` - Detected image pixel height.
    /// * `orientation` - Scan orientation.
    /// * `frames` - Ordered list of detected frame regions.
    #[must_use]
    pub const fn new(
        source_width: u32,
        source_height: u32,
        orientation: StripOrientation,
        frames: Vec<FrameRoi>,
    ) -> Self {
        Self {
            source_width,
            source_height,
            orientation,
            frames,
        }
    }

    /// Returns the number of detected frames.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.frames.len()
    }

    /// Checks if no frames were detected.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Provides a zero-copy borrowed sub-image view of the requested frame.
    ///
    /// # Safety / Cost
    /// Performs zero pixel buffer copies, returning an `image::SubImage`
    /// that borrows directly from the source image reference.
    ///
    /// # Arguments
    /// * `image` - Source image buffer reference to slice.
    /// * `frame_idx` - 0-indexed frame number to view.
    ///
    /// # Errors
    /// Returns [`RoiError::FrameIndexOutOfRange`] if `frame_idx >= self.len()`.
    ///
    /// # Examples
    /// ```
    /// use reto_core::{EvenSplitDetector, RoiDetectionConfig, RoiDetector};
    /// use image::{GenericImageView, Rgba, RgbaImage};
    ///
    /// let img = RgbaImage::from_pixel(300, 100, Rgba([255, 255, 255, 255]));
    /// let detector = EvenSplitDetector::new();
    /// let rois = detector.detect(&img, &RoiDetectionConfig::default(), None).unwrap();
    /// let sub = rois.sub_image(&img, 0).unwrap();
    /// assert_eq!(sub.dimensions(), (100, 100));
    /// ```
    pub fn sub_image<'a, I: GenericImageView>(
        &self,
        image: &'a I,
        frame_idx: usize,
    ) -> Result<SubImage<&'a I>, RoiError> {
        let frame = self
            .frames
            .get(frame_idx)
            .ok_or(RoiError::FrameIndexOutOfRange {
                requested: frame_idx,
                available: self.frames.len(),
            })?;

        let (w, h) = image.dimensions();
        let pixel = frame.bounds.to_pixel_rect(Size2D::new(w, h));
        Ok(SubImage::new(
            image,
            pixel.x,
            pixel.y,
            pixel.width,
            pixel.height,
        ))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp, clippy::suboptimal_flops)]
mod tests {
    use super::*;

    #[test]
    fn test_point2d_transforms() {
        let roi = NormalizedRect::new(0.25, 0.1, 0.5, 0.8).unwrap();
        let strip_size = Size2D::new(1000, 500);

        let local_pt = Point2D::new(50.0, 100.0);
        let global_pt = local_pt.to_global(roi, strip_size);

        // roi.x * 1000 = 250, roi.y * 500 = 50
        assert_eq!(global_pt.x, 300.0);
        assert_eq!(global_pt.y, 150.0);

        let roundtrip = global_pt.to_local(roi, strip_size).unwrap();
        assert_eq!(roundtrip.x, local_pt.x);
        assert_eq!(roundtrip.y, local_pt.y);

        // Point outside the roi
        let outside = Point2D::new(10.0, 10.0);
        assert!(outside.to_local(roi, strip_size).is_none());
    }

    #[test]
    fn test_coord_transforms() {
        let roi = NormalizedRect::new(0.25, 0.1, 0.5, 0.8).unwrap();
        let strip_size = Size2D::new(1000, 500);

        let local = LocalCoord::new(Point2D::new(50.0, 100.0), Point2D::new(0.1, 0.25));
        let global = local.to_global(roi, strip_size);

        assert_eq!(global.pixel.x, 300.0);
        assert_eq!(global.pixel.y, 150.0);
        assert!((global.normalized.x - (0.25 + 0.1 * 0.5)).abs() < 1e-6);
        assert!((global.normalized.y - (0.1 + 0.25 * 0.8)).abs() < 1e-6);

        let local_back = global.to_local(roi, strip_size).unwrap();
        assert_eq!(local_back.pixel.x, local.pixel.x);
        assert_eq!(local_back.pixel.y, local.pixel.y);
        assert!((local_back.normalized.x - local.normalized.x).abs() < 1e-6);
        assert!((local_back.normalized.y - local.normalized.y).abs() < 1e-6);
    }
}

// ---------------------------------------------------------------------------
// Linear Algebra Constants & Spline Transformation Matrices
// ---------------------------------------------------------------------------

/// Uniform cubic B-spline basis transformation matrix:
/// $M_{\text{bspline}} = \frac{1}{6} \begin{bmatrix} -1 & 3 & -3 & 1 \\ 3 & -6 & 3 & 0 \\ -3 & 0 & 3 & 0 \\ 1 & 4 & 1 & 0 \end{bmatrix}$.
pub const BSPLINE_BASIS_MATRIX: [[f32; 4]; 4] = [
    [-1.0 / 6.0, 3.0 / 6.0, -3.0 / 6.0, 1.0 / 6.0],
    [3.0 / 6.0, -6.0 / 6.0, 3.0 / 6.0, 0.0],
    [-3.0 / 6.0, 0.0, 3.0 / 6.0, 0.0],
    [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0, 0.0],
];

/// Uniform Catmull-Rom spline transformation matrix:
/// $M_{\text{catmull\_rom}} = \frac{1}{2} \begin{bmatrix} -1 & 3 & -3 & 1 \\ 2 & -5 & 4 & -1 \\ -1 & 0 & 1 & 0 \\ 0 & 2 & 0 & 0 \end{bmatrix}$.
pub const CATMULL_ROM_BASIS_MATRIX: [[f32; 4]; 4] = [
    [-0.5, 1.5, -1.5, 0.5],
    [1.0, -2.5, 2.0, -0.5],
    [-0.5, 0.0, 0.5, 0.0],
    [0.0, 1.0, 0.0, 0.0],
];

/// Computes the monomial time vector $\mathbf{T}(t) = [t^3, t^2, t, 1]$ clamped to $t \in [0, 1]$.
#[inline]
#[must_use]
pub fn monomial_time_vec4(t: f32) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);
    let t2 = t * t;
    [t2 * t, t2, t, 1.0]
}

/// Multiplies a $1 \times 4$ row vector by a $4 \times 4$ constant matrix via FMA dot products: $\mathbf{v} \cdot \mathbf{M}$.
#[inline]
#[must_use]
pub fn vec4_mat4_mul(v: [f32; 4], m: &[[f32; 4]; 4]) -> [f32; 4] {
    let mut res = [0.0_f32; 4];
    for (col, out) in res.iter_mut().enumerate() {
        *out = v[0].mul_add(
            m[0][col],
            v[1].mul_add(m[1][col], v[2].mul_add(m[2][col], v[3] * m[3][col])),
        );
    }
    res
}

/// Multiplies a $3 \times 3$ matrix by a $3 \times 1$ column vector via FMA: $\mathbf{M} \cdot \mathbf{v}$.
#[inline]
#[must_use]
pub fn mat3_vec3_mul(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    let mut res = [0.0_f32; 3];
    for (row, out) in res.iter_mut().enumerate() {
        *out = m[row][0].mul_add(v[0], m[row][1].mul_add(v[1], m[row][2] * v[2]));
    }
    res
}

/// Multiplies two $3 \times 3$ matrices via FMA: $\mathbf{A} \cdot \mathbf{B}$.
#[inline]
#[must_use]
pub fn mat3_mat3_mul(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut res = [[0.0_f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            res[r][c] = a[r][0].mul_add(b[0][c], a[r][1].mul_add(b[1][c], a[r][2] * b[2][c]));
        }
    }
    res
}

// ---------------------------------------------------------------------------
// Spline Trajectory Interpolation Functions
// ---------------------------------------------------------------------------

/// Evaluates uniform cubic B-spline basis $[b_0(t), b_1(t), b_2(t), b_3(t)] = \mathbf{T}(t) \cdot \mathbf{M}_{\text{bspline}}$.
///
/// # Examples
/// ```
/// use reto_core::cubic_bspline_basis;
///
/// let b = cubic_bspline_basis(0.0);
/// assert!((b.iter().sum::<f32>() - 1.0).abs() < 1e-5);
/// ```
#[inline]
#[must_use]
pub fn cubic_bspline_basis(t: f32) -> [f32; 4] {
    vec4_mat4_mul(monomial_time_vec4(t), &BSPLINE_BASIS_MATRIX)
}

/// Evaluates uniform Catmull-Rom blending weights $[w_0(t), w_1(t), w_2(t), w_3(t)] = \mathbf{T}(t) \cdot \mathbf{M}_{\text{catmull\_rom}}$.
#[inline]
#[must_use]
pub fn catmull_rom_basis(t: f32) -> [f32; 4] {
    vec4_mat4_mul(monomial_time_vec4(t), &CATMULL_ROM_BASIS_MATRIX)
}

/// Interpolates a 1D scalar across 4 B-spline control points via FMA dot product.
#[inline]
#[must_use]
pub fn interpolate_bspline_1d(p: &[f32; 4], t: f32) -> f32 {
    let b = cubic_bspline_basis(t);
    b[0].mul_add(p[0], b[1].mul_add(p[1], b[2].mul_add(p[2], b[3] * p[3])))
}

/// Interpolates a 2D point across 4 B-spline control points via vectorized FMA dot product.
#[inline]
#[must_use]
pub fn interpolate_bspline_2d(p: &[[f32; 2]; 4], t: f32) -> [f32; 2] {
    let b = cubic_bspline_basis(t);
    let mut out = [0.0_f32; 2];
    for (i, v) in out.iter_mut().enumerate() {
        *v = b[0].mul_add(
            p[0][i],
            b[1].mul_add(p[1][i], b[2].mul_add(p[2][i], b[3] * p[3][i])),
        );
    }
    out
}

/// Interpolates a 3D point across 4 B-spline control points via vectorized FMA dot product.
#[inline]
#[must_use]
pub fn interpolate_bspline_3d(p: &[[f32; 3]; 4], t: f32) -> [f32; 3] {
    let b = cubic_bspline_basis(t);
    let mut out = [0.0_f32; 3];
    for (i, v) in out.iter_mut().enumerate() {
        *v = b[0].mul_add(
            p[0][i],
            b[1].mul_add(p[1][i], b[2].mul_add(p[2][i], b[3] * p[3][i])),
        );
    }
    out
}

/// Interpolates a 3D position using Catmull-Rom spline passing exactly through $p_1$ and $p_2$.
#[inline]
#[must_use]
pub fn interpolate_catmull_rom_3d(
    p0: [f32; 3],
    p1: [f32; 3],
    p2: [f32; 3],
    p3: [f32; 3],
    t: f32,
) -> [f32; 3] {
    let w = catmull_rom_basis(t);
    let mut out = [0.0_f32; 3];
    for (i, v) in out.iter_mut().enumerate() {
        *v = w[0].mul_add(
            p0[i],
            w[1].mul_add(p1[i], w[2].mul_add(p2[i], w[3] * p3[i])),
        );
    }
    out
}

/// Unit quaternion representing 3D spatial rotation in $\mathrm{SO}(3)$.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Quaternion {
    /// Scalar real component $w$.
    pub w: f32,
    /// Imaginary vector component $x$.
    pub x: f32,
    /// Imaginary vector component $y$.
    pub y: f32,
    /// Imaginary vector component $z$.
    pub z: f32,
}

impl Default for Quaternion {
    fn default() -> Self {
        Self::identity()
    }
}

impl Quaternion {
    /// Constructs a new quaternion $[w, x, y, z]$.
    #[inline]
    #[must_use]
    pub const fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }

    /// Constructs the identity rotation quaternion $[1, 0, 0, 0]$.
    #[inline]
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Normalizes the quaternion to unit length using FMA sum of squares.
    #[inline]
    #[must_use]
    pub fn normalize(self) -> Self {
        let norm_sq = self.w.mul_add(
            self.w,
            self.x
                .mul_add(self.x, self.y.mul_add(self.y, self.z * self.z)),
        );
        if norm_sq > 1e-12 {
            let inv = 1.0 / norm_sq.sqrt();
            Self {
                w: self.w * inv,
                x: self.x * inv,
                y: self.y * inv,
                z: self.z * inv,
            }
        } else {
            Self::identity()
        }
    }

    /// Converts a $3 \times 3$ rotation matrix into a unit quaternion via Shepperd's branch-stable method.
    #[must_use]
    pub fn from_rotation_matrix(r: &[[f32; 3]; 3]) -> Self {
        let trace = r[0][0] + r[1][1] + r[2][2];
        if trace > 0.0 {
            let s = 0.5 / (trace + 1.0).sqrt();
            Self {
                w: 0.25 / s,
                x: (r[2][1] - r[1][2]) * s,
                y: (r[0][2] - r[2][0]) * s,
                z: (r[1][0] - r[0][1]) * s,
            }
            .normalize()
        } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
            let s = 2.0 * (1.0 + r[0][0] - r[1][1] - r[2][2]).max(1e-6).sqrt();
            Self {
                w: (r[2][1] - r[1][2]) / s,
                x: 0.25 * s,
                y: (r[0][1] + r[1][0]) / s,
                z: (r[0][2] + r[2][0]) / s,
            }
            .normalize()
        } else if r[1][1] > r[2][2] {
            let s = 2.0 * (1.0 + r[1][1] - r[0][0] - r[2][2]).max(1e-6).sqrt();
            Self {
                w: (r[0][2] - r[2][0]) / s,
                x: (r[0][1] + r[1][0]) / s,
                y: 0.25 * s,
                z: (r[1][2] + r[2][1]) / s,
            }
            .normalize()
        } else {
            let s = 2.0 * (1.0 + r[2][2] - r[0][0] - r[1][1]).max(1e-6).sqrt();
            Self {
                w: (r[1][0] - r[0][1]) / s,
                x: (r[0][2] + r[2][0]) / s,
                y: (r[1][2] + r[2][1]) / s,
                z: 0.25 * s,
            }
            .normalize()
        }
    }

    /// Converts the unit quaternion into an orthogonal $3 \times 3$ rotation matrix.
    #[must_use]
    #[allow(clippy::suboptimal_flops)]
    pub fn to_rotation_matrix(&self) -> [[f32; 3]; 3] {
        let q = self.normalize();
        let xx = q.x * q.x;
        let yy = q.y * q.y;
        let zz = q.z * q.z;
        let xy = q.x * q.y;
        let xz = q.x * q.z;
        let yz = q.y * q.z;
        let wx = q.w * q.x;
        let wy = q.w * q.y;
        let wz = q.w * q.z;

        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    }

    /// Performs Spherical Linear Interpolation ($\mathrm{SLERP}$) between two unit quaternions.
    ///
    /// Selects shortest path on 4D hypersphere and falls back to LERP when $\theta \to 0$.
    #[must_use]
    pub fn slerp(q0: Self, mut q1: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let mut dot =
            q0.w.mul_add(q1.w, q0.x.mul_add(q1.x, q0.y.mul_add(q1.y, q0.z * q1.z)));

        // Take the shortest path on the 4D sphere
        if dot < 0.0 {
            q1 = Self::new(-q1.w, -q1.x, -q1.y, -q1.z);
            dot = -dot;
        }

        if dot > 0.9995 {
            // Linear interpolation fallback for nearly identical rotations
            Self {
                w: t.mul_add(q1.w - q0.w, q0.w),
                x: t.mul_add(q1.x - q0.x, q0.x),
                y: t.mul_add(q1.y - q0.y, q0.y),
                z: t.mul_add(q1.z - q0.z, q0.z),
            }
            .normalize()
        } else {
            let theta_0 = dot.clamp(-1.0, 1.0).acos();
            let theta = theta_0 * t;
            let sin_theta = theta.sin();
            let sin_theta_0 = theta_0.sin();

            let s0 = (theta_0 - theta).cos() - dot * sin_theta / sin_theta_0;
            let s1 = sin_theta / sin_theta_0;

            Self {
                w: s0.mul_add(q0.w, s1 * q1.w),
                x: s0.mul_add(q0.x, s1 * q1.x),
                y: s0.mul_add(q0.y, s1 * q1.y),
                z: s0.mul_add(q0.z, s1 * q1.z),
            }
            .normalize()
        }
    }

    /// Computes the $\mathrm{SO}(3)$ Lie algebra logarithm vector $\boldsymbol{\omega} \in \mathfrak{so}(3)$.
    ///
    /// Vector magnitude $\|\boldsymbol{\omega}\|$ equals rotation angle in radians.
    #[must_use]
    pub fn log_axis_angle(&self) -> [f32; 3] {
        let q = self.normalize();
        let vec_norm_sq = q.x.mul_add(q.x, q.y.mul_add(q.y, q.z * q.z));
        if vec_norm_sq < 1e-12 {
            [0.0, 0.0, 0.0]
        } else {
            let vec_norm = vec_norm_sq.sqrt();
            let angle = 2.0 * vec_norm.atan2(q.w);
            let factor = angle / vec_norm;
            [q.x * factor, q.y * factor, q.z * factor]
        }
    }
}

/// Computes composite $\mathrm{SE}(3)$ distance metric balancing 3D translation (px) and rotation (rad).
#[must_use]
pub fn compute_se3_distance(
    t1: [f32; 3],
    r1: &[[f32; 3]; 3],
    t2: [f32; 3],
    r2: &[[f32; 3]; 3],
    alpha: f32,
) -> f32 {
    let dx = t2[0] - t1[0];
    let dy = t2[1] - t1[1];
    let dz = t2[2] - t1[2];
    let trans_dist_sq = dx.mul_add(dx, dy.mul_add(dy, dz * dz));

    let q1 = Quaternion::from_rotation_matrix(r1);
    let q2 = Quaternion::from_rotation_matrix(r2);
    // Relative rotation: q_rel = q2 * q1^-1
    let q1_inv = Quaternion::new(q1.w, -q1.x, -q1.y, -q1.z);
    let q_rel = Quaternion::new(
        q2.w.mul_add(
            q1_inv.w,
            (-q2.x).mul_add(q1_inv.x, (-q2.y).mul_add(q1_inv.y, -q2.z * q1_inv.z)),
        ),
        q2.w.mul_add(
            q1_inv.x,
            q2.x.mul_add(q1_inv.w, q2.y.mul_add(q1_inv.z, -q2.z * q1_inv.y)),
        ),
        q2.w.mul_add(
            q1_inv.y,
            (-q2.x).mul_add(q1_inv.z, q2.y.mul_add(q1_inv.w, q2.z * q1_inv.x)),
        ),
        q2.w.mul_add(
            q1_inv.z,
            q2.x.mul_add(q1_inv.y, (-q2.y).mul_add(q1_inv.x, q2.z * q1_inv.w)),
        ),
    );
    let omega = q_rel.log_axis_angle();
    let rot_dist_sq = omega[0].mul_add(omega[0], omega[1].mul_add(omega[1], omega[2] * omega[2]));

    alpha.mul_add(rot_dist_sq, trans_dist_sq).sqrt()
}

/// Computes non-uniform frame delays ($\text{ms}$) proportional to inter-frame physical motion distances.
///
/// Automatically balances rounding residuals onto the dominant segment to preserve `total_period_ms`.
///
/// # Examples
/// ```
/// use reto_core::compute_non_uniform_frame_delays;
///
/// let delays = compute_non_uniform_frame_delays(&[10.0, 20.0], 300, 20);
/// assert_eq!(delays.len(), 2);
/// assert_eq!(delays[0] + delays[1], 300);
/// assert!(delays[1] > delays[0]);
/// ```
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn compute_non_uniform_frame_delays(
    distances: &[f32],
    total_period_ms: u32,
    min_delay_ms: u32,
) -> Vec<u32> {
    if distances.is_empty() {
        return Vec::new();
    }
    let total_dist: f32 = distances.iter().sum();
    if total_dist <= 1e-4 {
        let count = u32::try_from(distances.len()).unwrap_or(1);
        let even = (total_period_ms / count).max(min_delay_ms);
        return vec![even; distances.len()];
    }

    let mut delays: Vec<u32> = distances
        .iter()
        .map(|&d| {
            let frac = (d / total_dist).max(0.0);
            ((total_period_ms as f32 * frac).round() as u32).max(min_delay_ms)
        })
        .collect();

    // Adjust residual rounding difference onto the largest segment
    let sum_delays: u32 = delays.iter().sum();
    if sum_delays != total_period_ms && !delays.is_empty() {
        let (max_idx, _) = delays
            .iter()
            .enumerate()
            .max_by_key(|&(_, val)| *val)
            .unwrap_or((0, &0));
        if sum_delays < total_period_ms {
            delays[max_idx] += total_period_ms - sum_delays;
        } else {
            let diff = sum_delays - total_period_ms;
            delays[max_idx] = delays[max_idx].saturating_sub(diff).max(min_delay_ms);
        }
    }

    delays
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::suboptimal_flops,
    clippy::cast_precision_loss
)]
mod spline_timing_tests {
    use super::*;

    #[test]
    fn test_bspline_partition_of_unity() {
        for step in 0..=10 {
            let t = step as f32 / 10.0;
            let b = cubic_bspline_basis(t);
            let sum: f32 = b.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "B-spline must sum to 1 at t={t}");
            for &w in &b {
                assert!(w >= 0.0, "Basis weight must be non-negative at t={t}");
            }
        }
    }

    #[test]
    fn test_bspline_interpolation_2d_and_3d() {
        let p2d = [[0.0, 0.0], [10.0, 5.0], [20.0, 10.0], [30.0, 15.0]];
        let mid2d = interpolate_bspline_2d(&p2d, 0.5);
        assert!(mid2d[0] > 0.0 && mid2d[0] < 30.0);
        assert!(mid2d[1] > 0.0 && mid2d[1] < 15.0);

        let p3d = [
            [0.0, 0.0, 0.0],
            [10.0, 5.0, 1.0],
            [20.0, 10.0, 2.0],
            [30.0, 15.0, 3.0],
        ];
        let mid3d = interpolate_bspline_3d(&p3d, 0.5);
        assert!(mid3d[0] > 0.0 && mid3d[0] < 30.0);
        assert!(mid3d[2] > 0.0 && mid3d[2] < 3.0);
    }

    #[test]
    fn test_catmull_rom_endpoints() {
        let p0 = [0.0, 0.0, 0.0];
        let p1 = [10.0, 5.0, 1.0];
        let p2 = [20.0, 10.0, 2.0];
        let p3 = [30.0, 15.0, 3.0];

        let start = interpolate_catmull_rom_3d(p0, p1, p2, p3, 0.0);
        assert!((start[0] - p1[0]).abs() < 1e-5);
        assert!((start[1] - p1[1]).abs() < 1e-5);
        assert!((start[2] - p1[2]).abs() < 1e-5);

        let end = interpolate_catmull_rom_3d(p0, p1, p2, p3, 1.0);
        assert!((end[0] - p2[0]).abs() < 1e-5);
        assert!((end[1] - p2[1]).abs() < 1e-5);
        assert!((end[2] - p2[2]).abs() < 1e-5);
    }

    #[test]
    fn test_quaternion_slerp_and_matrix_roundtrip() {
        let q_id = Quaternion::identity();
        let r_id = q_id.to_rotation_matrix();
        assert!((r_id[0][0] - 1.0).abs() < 1e-5);
        assert!((r_id[1][1] - 1.0).abs() < 1e-5);
        assert!((r_id[2][2] - 1.0).abs() < 1e-5);

        // 90 degree rotation around Z
        let q_z90 = Quaternion::new(
            (std::f32::consts::FRAC_PI_4).cos(),
            0.0,
            0.0,
            (std::f32::consts::FRAC_PI_4).sin(),
        );
        let slerp_mid = Quaternion::slerp(q_id, q_z90, 0.5);
        let omega = slerp_mid.log_axis_angle();
        let angle = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
        assert!((angle - std::f32::consts::FRAC_PI_4).abs() < 1e-4);
    }

    #[test]
    fn test_compute_non_uniform_frame_delays() {
        let distances = [10.0, 30.0];
        let total_ms = 400;
        let delays = compute_non_uniform_frame_delays(&distances, total_ms, 20);
        assert_eq!(delays.len(), 2);
        assert_eq!(delays.iter().sum::<u32>(), total_ms);
        assert_eq!(delays[0], 100);
        assert_eq!(delays[1], 300);
    }

    #[test]
    fn test_constant_matrix_operations() {
        let eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let v = [2.0, 3.0, 4.0];
        assert_eq!(mat3_vec3_mul(&eye, v), v);
        assert_eq!(mat3_mat3_mul(&eye, &eye), eye);

        let cr_mid = catmull_rom_basis(0.5);
        let sum_cr: f32 = cr_mid.iter().sum();
        assert!((sum_cr - 1.0).abs() < 1e-5);
    }
}
