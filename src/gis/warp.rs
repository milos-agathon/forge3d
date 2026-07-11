use std::path::Path;

use crate::gis::affine::{raster_bounds, require_transform, transform_from_tuple};
use crate::gis::crs::{authority_map, crs_equal, require_crs, transform_point};
use crate::gis::error::{GisError, GisResult};
use crate::gis::raster_info::{read_raster_data, LoadedRaster};
use crate::gis::raster_write::{write_raster, CrsSpec, RasterArray, WriteRasterOptions};
use crate::gis::types::{
    AffineTransform, RasterDType, RasterInfo, RasterWarning, WARNING_ASSIGNMENT_NOT_REPROJECTION,
    WARNING_MISSING_CRS, WARNING_MISSING_TRANSFORM, WARNING_NOT_GEOREFERENCED,
    WARNING_PER_BAND_NODATA_MISMATCH, WARNING_ROTATED_OR_SHEARED,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResamplingMethod {
    Nearest,
    Bilinear,
}

impl ResamplingMethod {
    pub fn parse(value: Option<&str>) -> GisResult<Self> {
        match value.unwrap_or("").trim().to_ascii_lowercase().as_str() {
            "" => Err(GisError::ResamplingRequired(
                "resampling_required: resampling method is required".to_string(),
            )),
            "nearest" => Ok(Self::Nearest),
            "bilinear" => Ok(Self::Bilinear),
            other => Err(GisError::UnsupportedResamplingMethod(format!(
                "unsupported_resampling_method: unsupported resampling method {other:?}; supported methods are nearest and bilinear"
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Nearest => "nearest",
            Self::Bilinear => "bilinear",
        }
    }
}

/// What `reproject_raster` does when a pixel's coordinate transform fails
/// (MENSURA item 5). The default is to RAISE — an unsupported transform must
/// never silently become nodata with a success status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransformErrorPolicy {
    #[default]
    Raise,
    Nodata,
}

impl TransformErrorPolicy {
    pub fn parse(value: Option<&str>) -> GisResult<Self> {
        match value
            .unwrap_or("raise")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "raise" => Ok(Self::Raise),
            "nodata" => Ok(Self::Nodata),
            other => Err(GisError::InvalidArgument(format!(
                "unsupported_option: on_transform_error must be 'raise' or 'nodata', got {other:?}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ResampleTarget {
    Shape { height: u32, width: u32 },
    Resolution { x: f64, y: f64 },
}

#[derive(Debug, Clone)]
pub struct RasterOperationResult {
    pub array: RasterArray,
    pub info: RasterInfo,
    pub resampling: String,
    pub diagnostics: Vec<RasterWarning>,
}

pub fn assign_crs(path: impl AsRef<Path>, crs: CrsSpec, overwrite: bool) -> GisResult<RasterInfo> {
    let path = path.as_ref();
    let loaded = read_raster(path)?;
    if require_crs(&loaded.info).is_ok() && !overwrite {
        return Err(GisError::CrsAlreadyExists(
            "crs_exists: raster already has CRS metadata; pass overwrite=True to replace it"
                .to_string(),
        ));
    }
    let options = WriteRasterOptions {
        crs: Some(crs),
        transform: loaded
            .info
            .transform
            .map(transform_from_tuple)
            .transpose()?,
        nodata: loaded.info.nodata_per_band.clone(),
        driver: "GTiff".to_string(),
        overwrite: true,
        creation_options: Default::default(),
        creation_options_explicit: false,
        like_info: None,
    };
    let mut info = write_raster(path, loaded.array, options)?;
    info.warnings.push(RasterWarning::new(
        WARNING_ASSIGNMENT_NOT_REPROJECTION,
        "CRS assignment updates metadata only; coordinates and pixel values were not reprojected",
        Some("crs"),
    ));
    Ok(info)
}

pub fn resample_raster(
    path: impl AsRef<Path>,
    target: ResampleTarget,
    method: ResamplingMethod,
) -> GisResult<RasterOperationResult> {
    let loaded = read_raster(path)?;
    let (height, width, transform) = match target {
        ResampleTarget::Shape { height, width } => {
            validate_shape(height, width)?;
            let transform = loaded
                .info
                .transform
                .map(|tuple| {
                    scaled_transform(tuple, loaded.info.width, loaded.info.height, width, height)
                })
                .transpose()?;
            (height, width, transform)
        }
        ResampleTarget::Resolution { x, y } => {
            if !x.is_finite() || !y.is_finite() || x <= 0.0 || y <= 0.0 {
                return Err(GisError::InvalidArgument(
                    "resolution target must contain positive finite magnitudes".to_string(),
                ));
            }
            let transform = require_transform(&loaded.info)?;
            if transform.is_rotated_or_sheared() {
                return Err(GisError::InvalidTransform(
                    "rotated_or_sheared_transform: resolution resampling requires a north-up grid"
                        .to_string(),
                ));
            }
            let bounds = raster_bounds(&loaded.info)?;
            let width = ((bounds.right - bounds.left) / x).ceil().max(1.0) as u32;
            let height = ((bounds.top - bounds.bottom) / y).ceil().max(1.0) as u32;
            let transform = transform_from_bounds_resolution(bounds, width, height, transform)?;
            (height, width, Some(transform))
        }
    };

    let array = resize_array(
        &loaded.array,
        width,
        height,
        method,
        &loaded.info.nodata_per_band,
    );
    let info = operation_info(&loaded.info, width, height, transform, None, &array)?;
    Ok(RasterOperationResult {
        array,
        info,
        resampling: method.name().to_string(),
        diagnostics: Vec::new(),
    })
}

pub fn resample_array(
    array: RasterArray,
    target: ResampleTarget,
    method: ResamplingMethod,
) -> GisResult<RasterOperationResult> {
    let source = RasterInfo::new(
        "".into(),
        array.width as u32,
        array.height as u32,
        array.bands as u16,
    );
    let (height, width) = match target {
        ResampleTarget::Shape { height, width } => {
            validate_shape(height, width)?;
            (height, width)
        }
        ResampleTarget::Resolution { .. } => {
            return Err(GisError::MissingTransform(
                "missing_transform: array sources need an affine transform for resolution targets"
                    .to_string(),
            ));
        }
    };
    let nodata = vec![None; array.bands];
    let array = resize_array(&array, width, height, method, &nodata);
    let info = operation_info(&source, width, height, None, None, &array)?;
    Ok(RasterOperationResult {
        array,
        info,
        resampling: method.name().to_string(),
        diagnostics: Vec::new(),
    })
}

pub fn align_raster_to(
    path: impl AsRef<Path>,
    target: &RasterInfo,
    method: ResamplingMethod,
) -> GisResult<RasterOperationResult> {
    let loaded = read_raster(path)?;
    let source_crs = require_crs(&loaded.info)?;
    let target_crs = require_crs(target)?;
    if !crs_equal(&source_crs, &target_crs) {
        return Err(GisError::CrsMismatch(
            "crs_mismatch: align_raster_to does not reproject rasters".to_string(),
        ));
    }
    let source_transform = require_transform(&loaded.info)?;
    let target_transform = require_transform(target)?;
    let diagnostics = alignment_diagnostics(&loaded.info, target);
    let array = sample_to_grid(
        &loaded.array,
        source_transform,
        target_transform,
        target.width,
        target.height,
        method,
        &loaded.info.nodata_per_band,
    );
    let info = operation_info(
        &loaded.info,
        target.width,
        target.height,
        Some(target_transform),
        Some(target_crs),
        &array,
    )?;
    Ok(RasterOperationResult {
        array,
        info,
        resampling: method.name().to_string(),
        diagnostics,
    })
}

pub fn assert_grid_compatible(
    source: &RasterInfo,
    target: &RasterInfo,
    compare_nodata: bool,
) -> Vec<RasterWarning> {
    let mut diagnostics = Vec::new();
    match (
        CrsSpec::from_raster_info(source),
        CrsSpec::from_raster_info(target),
    ) {
        (Some(source_crs), Some(target_crs)) if !crs_equal(&source_crs, &target_crs) => {
            diagnostics.push(RasterWarning::new(
                "crs_mismatch",
                "source and target CRS differ",
                Some("crs"),
            ));
        }
        (Some(_), None) | (None, Some(_)) => diagnostics.push(RasterWarning::new(
            "crs_mismatch",
            "one raster has CRS metadata and the other does not",
            Some("crs"),
        )),
        _ => {}
    }
    diagnostics.extend(
        alignment_diagnostics(source, target)
            .into_iter()
            .filter(|warning| compare_nodata || warning.code != "nodata_mismatch"),
    );
    diagnostics
}

pub fn reproject_raster(
    path: impl AsRef<Path>,
    dst_crs: CrsSpec,
    method: ResamplingMethod,
    on_transform_error: TransformErrorPolicy,
) -> GisResult<RasterOperationResult> {
    let loaded = read_raster(path)?;
    let src_crs = require_crs(&loaded.info)?;
    let source_transform = require_transform(&loaded.info)?;
    let source_bounds = raster_bounds(&loaded.info)?;
    let dst_bounds = match crate::gis::crs::transform_bounds(source_bounds, &src_crs, &dst_crs) {
        Ok(bounds) => bounds,
        // Bounds calculation is part of transform execution. For an
        // unsupported pair, keep the source grid only long enough to apply
        // the caller's per-pixel error policy below; malformed CRS/arguments
        // still fail immediately before reaching this point.
        Err(GisError::BackendUnavailable(_)) => source_bounds,
        Err(error) => return Err(error),
    };
    let dst_transform = AffineTransform::new([
        (dst_bounds.right - dst_bounds.left) / loaded.info.width as f64,
        0.0,
        dst_bounds.left,
        0.0,
        -((dst_bounds.top - dst_bounds.bottom) / loaded.info.height as f64),
        dst_bounds.top,
    ])?;
    let source_values = raster_to_f64(&loaded.array);
    let mut out = vec![0.0; loaded.array.bands * loaded.array.height * loaded.array.width];
    // MENSURA item 5: transform failures are counted, never silently
    // absorbed. Out-of-extent samples remain legitimate nodata.
    let mut failure_count: usize = 0;
    let mut first_pixel: Option<(usize, usize)> = None;
    for band in 0..loaded.array.bands {
        let fill = loaded
            .info
            .nodata_per_band
            .get(band)
            .copied()
            .flatten()
            .unwrap_or(0.0);
        for row in 0..loaded.array.height {
            for col in 0..loaded.array.width {
                let (x, y) = dst_transform.apply(col as f64 + 0.5, row as f64 + 0.5);
                let value = match transform_point(x, y, &dst_crs, &src_crs) {
                    Ok((sx, sy)) => crate::gis::affine::inverse_apply(source_transform, sx, sy)
                        .ok()
                        .and_then(|(src_col, src_row)| {
                            sample_band(
                                &source_values,
                                &loaded.array,
                                band,
                                src_col - 0.5,
                                src_row - 0.5,
                                method,
                                loaded.info.nodata_per_band.get(band).copied().flatten(),
                            )
                        }),
                    Err(_) => {
                        failure_count += 1;
                        if first_pixel.is_none() {
                            first_pixel = Some((row, col));
                        }
                        None
                    }
                };
                out[band * loaded.array.height * loaded.array.width
                    + row * loaded.array.width
                    + col] = value.unwrap_or(fill);
            }
        }
    }
    let mut diagnostics = Vec::new();
    if failure_count > 0 {
        let (row, col) = first_pixel.expect("failure recorded");
        match on_transform_error {
            TransformErrorPolicy::Raise => {
                return Err(GisError::TransformFailed(format!(
                    "transform_failed: {failure_count} pixel(s) failed to transform \
                     (first at row {row}, col {col}); pass on_transform_error='nodata' \
                     to fill them with nodata instead"
                )));
            }
            TransformErrorPolicy::Nodata => {
                diagnostics.push(RasterWarning::new(
                    "transform_failures_filled_nodata",
                    format!(
                        "{failure_count} pixel(s) failed to transform (first at row {row}, \
                         col {col}) and were filled with nodata"
                    ),
                    Some("array"),
                ));
            }
        }
    }
    let array = f64_to_raster_data(
        loaded.array.dtype(),
        out,
        loaded.array.bands,
        loaded.array.height,
        loaded.array.width,
    )?;
    let info = operation_info(
        &loaded.info,
        loaded.info.width,
        loaded.info.height,
        Some(dst_transform),
        Some(dst_crs),
        &array,
    )?;
    Ok(RasterOperationResult {
        array,
        info,
        resampling: method.name().to_string(),
        diagnostics,
    })
}

fn validate_shape(height: u32, width: u32) -> GisResult<()> {
    if height == 0 || width == 0 {
        Err(GisError::InvalidShape(
            "target shape must be positive".to_string(),
        ))
    } else {
        Ok(())
    }
}

pub(crate) fn read_raster(path: impl AsRef<Path>) -> GisResult<LoadedRaster> {
    read_raster_data(path)
}

fn resize_array(
    array: &RasterArray,
    width: u32,
    height: u32,
    method: ResamplingMethod,
    nodata: &[Option<f64>],
) -> RasterArray {
    let src = raster_to_f64(array);
    let width_usize = width as usize;
    let height_usize = height as usize;
    let mut out = vec![0.0; array.bands * width_usize * height_usize];
    let sx = array.width as f64 / width as f64;
    let sy = array.height as f64 / height as f64;
    for band in 0..array.bands {
        let fill = nodata.get(band).copied().flatten().unwrap_or(0.0);
        for row in 0..height_usize {
            for col in 0..width_usize {
                let src_col = (col as f64 + 0.5) * sx - 0.5;
                let src_row = (row as f64 + 0.5) * sy - 0.5;
                let value = sample_band(
                    &src,
                    array,
                    band,
                    src_col,
                    src_row,
                    method,
                    nodata.get(band).copied().flatten(),
                )
                .unwrap_or(fill);
                out[band * width_usize * height_usize + row * width_usize + col] = value;
            }
        }
    }
    f64_to_raster_data(array.dtype(), out, array.bands, height_usize, width_usize)
        .expect("resampled array shape is valid")
}

fn sample_to_grid(
    array: &RasterArray,
    source_transform: AffineTransform,
    target_transform: AffineTransform,
    width: u32,
    height: u32,
    method: ResamplingMethod,
    nodata: &[Option<f64>],
) -> RasterArray {
    let src = raster_to_f64(array);
    let width_usize = width as usize;
    let height_usize = height as usize;
    let mut out = vec![0.0; array.bands * width_usize * height_usize];
    for band in 0..array.bands {
        let fill = nodata.get(band).copied().flatten().unwrap_or(0.0);
        for row in 0..height_usize {
            for col in 0..width_usize {
                let (x, y) = target_transform.apply(col as f64 + 0.5, row as f64 + 0.5);
                let value = crate::gis::affine::inverse_apply(source_transform, x, y)
                    .ok()
                    .and_then(|(src_col, src_row)| {
                        sample_band(
                            &src,
                            array,
                            band,
                            src_col - 0.5,
                            src_row - 0.5,
                            method,
                            nodata.get(band).copied().flatten(),
                        )
                    })
                    .unwrap_or(fill);
                out[band * width_usize * height_usize + row * width_usize + col] = value;
            }
        }
    }
    f64_to_raster_data(array.dtype(), out, array.bands, height_usize, width_usize)
        .expect("aligned array shape is valid")
}

fn sample_band(
    values: &[f64],
    array: &RasterArray,
    band: usize,
    col: f64,
    row: f64,
    method: ResamplingMethod,
    nodata: Option<f64>,
) -> Option<f64> {
    if !col.is_finite() || !row.is_finite() {
        return None;
    }
    match method {
        ResamplingMethod::Nearest => {
            let col = col.round() as isize;
            let row = row.round() as isize;
            get_valid_sample(values, array, band, col, row, nodata)
        }
        ResamplingMethod::Bilinear => {
            let c0 = col.floor();
            let r0 = row.floor();
            let dc = col - c0;
            let dr = row - r0;
            let c0 = c0 as isize;
            let r0 = r0 as isize;
            let samples = [
                (c0, r0, (1.0 - dc) * (1.0 - dr)),
                (c0 + 1, r0, dc * (1.0 - dr)),
                (c0, r0 + 1, (1.0 - dc) * dr),
                (c0 + 1, r0 + 1, dc * dr),
            ];
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;
            for (sample_col, sample_row, weight) in samples {
                if weight <= 0.0 {
                    continue;
                }
                if let Some(value) =
                    get_valid_sample(values, array, band, sample_col, sample_row, nodata)
                {
                    weighted_sum += value * weight;
                    weight_sum += weight;
                }
            }
            (weight_sum > 0.0).then_some(weighted_sum / weight_sum)
        }
    }
}

fn get_valid_sample(
    values: &[f64],
    array: &RasterArray,
    band: usize,
    col: isize,
    row: isize,
    nodata: Option<f64>,
) -> Option<f64> {
    let value = get_sample(values, array, band, col, row)?;
    if nodata.is_some_and(|nodata| nodata_eq(Some(value), Some(nodata))) {
        None
    } else {
        Some(value)
    }
}

fn get_sample(
    values: &[f64],
    array: &RasterArray,
    band: usize,
    col: isize,
    row: isize,
) -> Option<f64> {
    if col < 0 || row < 0 || col >= array.width as isize || row >= array.height as isize {
        return None;
    }
    Some(values[band * array.height * array.width + row as usize * array.width + col as usize])
}

pub fn raster_to_f64(array: &RasterArray) -> Vec<f64> {
    crate::gis::raster_info::raster_to_f64(array)
}

pub(crate) fn f64_to_raster_data(
    dtype: RasterDType,
    values: Vec<f64>,
    bands: usize,
    height: usize,
    width: usize,
) -> GisResult<RasterArray> {
    crate::gis::raster_info::f64_to_raster_data(dtype, values, bands, height, width)
}

fn scaled_transform(
    tuple: (f64, f64, f64, f64, f64, f64),
    old_width: u32,
    old_height: u32,
    new_width: u32,
    new_height: u32,
) -> GisResult<AffineTransform> {
    let transform = transform_from_tuple(tuple)?;
    let sx = old_width as f64 / new_width as f64;
    let sy = old_height as f64 / new_height as f64;
    AffineTransform::new([
        transform.a * sx,
        transform.b * sy,
        transform.c,
        transform.d * sx,
        transform.e * sy,
        transform.f,
    ])
}

fn transform_from_bounds_resolution(
    bounds: crate::gis::types::RasterBounds,
    width: u32,
    height: u32,
    source: AffineTransform,
) -> GisResult<AffineTransform> {
    let xres = (bounds.right - bounds.left) / width as f64;
    let yres = (bounds.top - bounds.bottom) / height as f64;
    let a = if source.a < 0.0 { -xres } else { xres };
    let e = if source.e >= 0.0 { yres } else { -yres };
    let c = if a >= 0.0 { bounds.left } else { bounds.right };
    let f = if e >= 0.0 { bounds.bottom } else { bounds.top };
    AffineTransform::new([a, 0.0, c, 0.0, e, f])
}

pub(crate) fn operation_info(
    source: &RasterInfo,
    width: u32,
    height: u32,
    transform: Option<AffineTransform>,
    crs: Option<CrsSpec>,
    array: &RasterArray,
) -> GisResult<RasterInfo> {
    let mut info = RasterInfo::new(source.path.clone().into(), width, height, source.band_count);
    info.dtype_per_band = vec![array.dtype().name().to_string(); source.band_count as usize];
    if let Some(crs) = crs {
        info.crs_wkt = crs.wkt.clone();
        info.crs_authority = authority_map(&crs);
    } else {
        info.crs_wkt = source.crs_wkt.clone();
        info.crs_authority = source.crs_authority.clone();
    }
    if let Some(transform) = transform {
        if transform.is_rotated_or_sheared() {
            info.warnings.push(RasterWarning::new(
                WARNING_ROTATED_OR_SHEARED,
                "transform contains rotation or shear terms",
                Some("transform"),
            ));
        }
        info.transform = Some(transform.tuple());
        info.bounds = Some(transform.bounds(width, height).tuple());
        info.resolution = Some(transform.resolution());
    }
    info.nodata_per_band = source.nodata_per_band.clone();
    if nodata_values_differ(&info.nodata_per_band) {
        info.warnings.push(RasterWarning::new(
            WARNING_PER_BAND_NODATA_MISMATCH,
            "bands advertise different nodata values",
            Some("nodata_per_band"),
        ));
    }
    info.block_size = source.block_size.clone();
    info.tiling = source.tiling.clone();
    info.compression = source.compression.clone();
    info.is_georeferenced =
        (info.crs_wkt.is_some() || info.crs_authority.is_some()) && info.transform.is_some();
    if info.crs_wkt.is_none() && info.crs_authority.is_none() {
        push_warning_if_absent(
            &mut info.warnings,
            WARNING_MISSING_CRS,
            "raster has no CRS metadata",
            Some("crs"),
        );
    }
    if info.transform.is_none() {
        push_warning_if_absent(
            &mut info.warnings,
            WARNING_MISSING_TRANSFORM,
            "raster has no affine transform",
            Some("transform"),
        );
    }
    if !info.is_georeferenced {
        push_warning_if_absent(
            &mut info.warnings,
            WARNING_NOT_GEOREFERENCED,
            "raster is not fully georeferenced",
            None,
        );
    }
    Ok(info)
}

fn push_warning_if_absent(
    warnings: &mut Vec<RasterWarning>,
    code: &'static str,
    message: &'static str,
    field: Option<&'static str>,
) {
    if warnings.iter().all(|warning| warning.code != code) {
        warnings.push(RasterWarning::new(code, message, field));
    }
}

fn alignment_diagnostics(source: &RasterInfo, target: &RasterInfo) -> Vec<RasterWarning> {
    let mut diagnostics = Vec::new();
    if source.width != target.width || source.height != target.height {
        diagnostics.push(RasterWarning::new(
            "shape_mismatch",
            "source and target raster shapes differ",
            Some("shape"),
        ));
    }
    if source.transform != target.transform {
        diagnostics.push(RasterWarning::new(
            "transform_mismatch",
            "source and target affine transforms differ",
            Some("transform"),
        ));
    }
    if source.resolution != target.resolution {
        diagnostics.push(RasterWarning::new(
            "resolution_mismatch",
            "source and target resolutions differ",
            Some("resolution"),
        ));
    }
    if source.bounds != target.bounds {
        diagnostics.push(RasterWarning::new(
            "bounds_mismatch",
            "source and target bounds differ",
            Some("bounds"),
        ));
    }
    if !nodata_vec_eq(&source.nodata_per_band, &target.nodata_per_band) {
        diagnostics.push(RasterWarning::new(
            "nodata_mismatch",
            "source and target nodata metadata differ",
            Some("nodata_per_band"),
        ));
    }
    diagnostics
}

fn nodata_values_differ(values: &[Option<f64>]) -> bool {
    values.first().is_some_and(|first| {
        values
            .iter()
            .skip(1)
            .any(|value| !nodata_eq(*first, *value))
    })
}

fn nodata_vec_eq(left: &[Option<f64>], right: &[Option<f64>]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| nodata_eq(*left, *right))
}

fn nodata_eq(left: Option<f64>, right: Option<f64>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left == right || (left.is_nan() && right.is_nan()),
        (None, None) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_parse_has_stable_error_codes() {
        let required = ResamplingMethod::parse(None).unwrap_err();
        let unsupported = ResamplingMethod::parse(Some("cubic")).unwrap_err();

        assert_eq!(required.code(), "resampling_required");
        assert_eq!(unsupported.code(), "unsupported_resampling_method");
    }
}
