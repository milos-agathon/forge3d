use crate::gis::error::{GisError, GisResult};
use crate::gis::types::{AffineTransform, RasterBounds, RasterInfo};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelWindow {
    pub col_off: i64,
    pub row_off: i64,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RasterWindow {
    pub window: PixelWindow,
    pub clipped_bounds: RasterBounds,
    pub output_transform: AffineTransform,
    pub output_shape: (u32, u32),
}

pub fn transform_from_tuple(values: (f64, f64, f64, f64, f64, f64)) -> GisResult<AffineTransform> {
    AffineTransform::new([values.0, values.1, values.2, values.3, values.4, values.5])
}

pub fn require_transform(info: &RasterInfo) -> GisResult<AffineTransform> {
    transform_from_tuple(info.transform.ok_or_else(|| {
        GisError::MissingTransform("missing_transform: raster has no affine transform".to_string())
    })?)
}

pub fn transform_from_origin(
    west: f64,
    north: f64,
    xsize: f64,
    ysize: f64,
) -> GisResult<AffineTransform> {
    if [west, north, xsize, ysize]
        .iter()
        .any(|value| !value.is_finite())
        || xsize <= 0.0
        || ysize <= 0.0
    {
        return Err(GisError::InvalidTransform(
            "origin transform requires finite origin and positive pixel sizes".to_string(),
        ));
    }
    AffineTransform::new([xsize, 0.0, west, 0.0, -ysize, north])
}

pub fn transform_from_bounds(
    bounds: RasterBounds,
    width: u32,
    height: u32,
) -> GisResult<AffineTransform> {
    if width == 0 || height == 0 {
        return Err(GisError::InvalidShape(
            "width and height must be positive".to_string(),
        ));
    }
    validate_bounds_tuple(bounds.tuple(), false)?;
    AffineTransform::new([
        (bounds.right - bounds.left) / width as f64,
        0.0,
        bounds.left,
        0.0,
        -((bounds.top - bounds.bottom) / height as f64),
        bounds.top,
    ])
}

pub fn validate_bounds_tuple(
    values: (f64, f64, f64, f64),
    allow_antimeridian: bool,
) -> GisResult<RasterBounds> {
    if [values.0, values.1, values.2, values.3]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(GisError::InvalidBounds(
            "bounds values must be finite".to_string(),
        ));
    }
    if values.1 >= values.3 {
        return Err(GisError::InvalidBounds(
            "bounds order must be (left, bottom, right, top)".to_string(),
        ));
    }
    if values.0 >= values.2 && !allow_antimeridian {
        return Err(GisError::InvalidBounds(
            "bounds order must be (left, bottom, right, top)".to_string(),
        ));
    }
    Ok(RasterBounds {
        left: values.0,
        bottom: values.1,
        right: values.2,
        top: values.3,
    })
}

pub fn raster_bounds(info: &RasterInfo) -> GisResult<RasterBounds> {
    let transform = require_transform(info)?;
    let bounds = transform.bounds(info.width, info.height);
    validate_bounds_tuple(bounds.tuple(), false)?;
    Ok(bounds)
}

pub fn raster_transform(info: &RasterInfo) -> GisResult<AffineTransform> {
    require_transform(info)
}

pub fn raster_resolution(info: &RasterInfo) -> GisResult<(f64, f64)> {
    Ok(require_transform(info)?.resolution())
}

pub fn array_bounds(
    height: u32,
    width: u32,
    transform: AffineTransform,
) -> GisResult<RasterBounds> {
    if height == 0 || width == 0 {
        return Err(GisError::InvalidShape(
            "height and width must be positive".to_string(),
        ));
    }
    let bounds = transform.bounds(width, height);
    validate_bounds_tuple(bounds.tuple(), false)
}

pub fn window_from_bounds(
    info: &RasterInfo,
    bounds: RasterBounds,
    boundless: bool,
) -> GisResult<RasterWindow> {
    validate_bounds_tuple(bounds.tuple(), false)?;
    let transform = require_transform(info)?;
    let (col0, row0) = inverse_apply(transform, bounds.left, bounds.top)?;
    let (col1, row1) = inverse_apply(transform, bounds.right, bounds.top)?;
    let (col2, row2) = inverse_apply(transform, bounds.left, bounds.bottom)?;
    let (col3, row3) = inverse_apply(transform, bounds.right, bounds.bottom)?;
    let min_col = col0.min(col1).min(col2).min(col3).floor() as i64;
    let max_col = col0.max(col1).max(col2).max(col3).ceil() as i64;
    let min_row = row0.min(row1).min(row2).min(row3).floor() as i64;
    let max_row = row0.max(row1).max(row2).max(row3).ceil() as i64;

    if max_col <= min_col || max_row <= min_row {
        return Err(GisError::InvalidBounds(
            "bounds produce an empty raster window".to_string(),
        ));
    }

    let (col_off, row_off, end_col, end_row) = if boundless {
        (min_col, min_row, max_col, max_row)
    } else {
        let col_off = min_col.clamp(0, info.width as i64);
        let row_off = min_row.clamp(0, info.height as i64);
        let end_col = max_col.clamp(0, info.width as i64);
        let end_row = max_row.clamp(0, info.height as i64);
        if end_col <= col_off || end_row <= row_off {
            return Err(GisError::InvalidBounds(
                "bounds do not intersect raster extent".to_string(),
            ));
        }
        (col_off, row_off, end_col, end_row)
    };

    let width = u32::try_from(end_col - col_off)
        .map_err(|_| GisError::InvalidBounds("window width exceeds supported range".to_string()))?;
    let height = u32::try_from(end_row - row_off).map_err(|_| {
        GisError::InvalidBounds("window height exceeds supported range".to_string())
    })?;
    let output_transform = translated_transform(transform, col_off, row_off)?;
    let clipped_bounds = output_transform.bounds(width, height);

    Ok(RasterWindow {
        window: PixelWindow {
            col_off,
            row_off,
            width,
            height,
        },
        clipped_bounds,
        output_transform,
        output_shape: (height, width),
    })
}

pub fn window_transform(info: &RasterInfo, window: PixelWindow) -> GisResult<AffineTransform> {
    let transform = require_transform(info)?;
    translated_transform(transform, window.col_off, window.row_off)
}

pub fn xy(transform: AffineTransform, row: i64, col: i64) -> GisResult<(f64, f64)> {
    let row = row as f64 + 0.5;
    let col = col as f64 + 0.5;
    let (x, y) = transform.apply(col, row);
    if !x.is_finite() || !y.is_finite() {
        return Err(GisError::InvalidTransform(
            "pixel coordinate produced non-finite world coordinate".to_string(),
        ));
    }
    Ok((x, y))
}

pub fn rowcol(transform: AffineTransform, x: f64, y: f64) -> GisResult<(i64, i64)> {
    let (col, row) = inverse_apply(transform, x, y)?;
    Ok((row.floor() as i64, col.floor() as i64))
}

pub fn translated_transform(
    transform: AffineTransform,
    col_off: i64,
    row_off: i64,
) -> GisResult<AffineTransform> {
    let col = col_off as f64;
    let row = row_off as f64;
    AffineTransform::new([
        transform.a,
        transform.b,
        transform
            .a
            .mul_add(col, transform.b.mul_add(row, transform.c)),
        transform.d,
        transform.e,
        transform
            .d
            .mul_add(col, transform.e.mul_add(row, transform.f)),
    ])
}

/// Reject a source transform whose linear part is singular (det ~ 0) up front.
///
/// A singular transform cannot be inverted, so a resample/align that mapped
/// destination pixels back through it would silently fill EVERY pixel with
/// nodata and report success. Callers that resample without a per-pixel
/// raise/nodata policy (e.g. `align_raster_to`) preflight this so the failure
/// is an explicit `InvalidTransform`, never a silent all-nodata output.
pub fn require_invertible(transform: AffineTransform) -> GisResult<()> {
    let det = transform.a * transform.e - transform.b * transform.d;
    if !det.is_finite() || det.abs() <= f64::EPSILON {
        return Err(GisError::InvalidTransform(
            "invalid_transform: source transform is singular (non-invertible); \
             its linear part has a zero determinant"
                .to_string(),
        ));
    }
    Ok(())
}

pub fn inverse_apply(transform: AffineTransform, x: f64, y: f64) -> GisResult<(f64, f64)> {
    let det = transform.a * transform.e - transform.b * transform.d;
    if !det.is_finite() || det.abs() <= f64::EPSILON {
        return Err(GisError::InvalidTransform(
            "transform is not invertible".to_string(),
        ));
    }
    let x = x - transform.c;
    let y = y - transform.f;
    Ok((
        (transform.e * x - transform.b * y) / det,
        (-transform.d * x + transform.a * y) / det,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_from_bounds_uses_gis_bounds_order() {
        let mut info = RasterInfo::new("memory".into(), 5, 4, 1);
        info.transform = Some((10.0, 0.0, 100.0, 0.0, -10.0, 200.0));

        let bounds = validate_bounds_tuple((110.0, 170.0, 140.0, 190.0), false).unwrap();
        let window = window_from_bounds(&info, bounds, false).unwrap();

        assert_eq!(window.window.col_off, 1);
        assert_eq!(window.window.row_off, 1);
        assert_eq!(window.window.width, 3);
        assert_eq!(window.window.height, 2);
        assert_eq!(
            window.output_transform.tuple(),
            (10.0, 0.0, 110.0, 0.0, -10.0, 190.0)
        );
    }
}
