use serde_json::Value;

use crate::gis::affine::{self, PixelWindow};
use crate::gis::crs;
use crate::gis::error::{GisError, GisResult};
use crate::gis::raster_info;
use crate::gis::raster_write::{CrsSpec, RasterArray};
use crate::gis::types::{AffineTransform, RasterDType, RasterInfo, RasterWarning};
use crate::gis::vector;

const MISSING_CRS: &str = "missing_crs";
const CRS_MISMATCH: &str = "crs_mismatch";
const MASK_POLARITY_EXPLICIT: &str = "mask_polarity_explicit";
const UNSUPPORTED_DTYPE: &str = "unsupported_dtype";
const UNSUPPORTED_GEOMETRY_TYPE: &str = "unsupported_geometry_type";
const INVALID_GEOMETRY: &str = "invalid_geometry";
const INVALID_ARGUMENT: &str = "invalid_argument";
const EMPTY_GEOMETRY: &str = "empty_geometry";
const EMPTY_FEATURE_SET: &str = "empty_feature_set";
const EMPTY_RASTER: &str = "empty_raster";
const SHAPE_MISMATCH: &str = "shape_mismatch";
const INVALID_NODATA: &str = "invalid_nodata";

#[derive(Debug, Clone)]
pub struct RasterizeOptions {
    pub value: f64,
    pub attribute: Option<String>,
    pub dtype: RasterDType,
    pub fill: f64,
    pub all_touched: bool,
}

#[derive(Debug, Clone)]
pub struct RasterizeResult {
    pub values: Vec<f64>,
    pub info: RasterInfo,
    pub target_shape: (usize, usize),
    pub target_transform: AffineTransform,
    pub target_bounds: (f64, f64, f64, f64),
    pub dtype: RasterDType,
    pub fill: f64,
    pub burned_pixels: usize,
    pub all_touched: bool,
    pub warnings: Vec<RasterWarning>,
}

#[derive(Debug, Clone)]
pub struct GeometryMaskOptions {
    pub invert: bool,
    pub all_touched: bool,
    pub mask_polarity: MaskPolarity,
}

#[derive(Debug, Clone)]
pub struct GeometryMaskResult {
    pub mask: Vec<bool>,
    pub info: RasterInfo,
    pub mask_polarity: MaskPolarity,
    pub true_count: usize,
    pub false_count: usize,
    pub warnings: Vec<RasterWarning>,
}

#[derive(Debug, Clone)]
pub struct MaskRasterOptions {
    pub mask_polarity: RasterMaskPolarity,
    pub crop: bool,
    pub fill: Option<f64>,
    pub nodata: Option<Vec<Option<f64>>>,
}

#[derive(Debug, Clone)]
pub struct MaskRasterResult {
    pub array: RasterArray,
    pub mask: Vec<bool>,
    pub info: RasterInfo,
    pub mask_polarity: RasterMaskPolarity,
    pub fill: Option<f64>,
    pub valid_count: usize,
    pub crop_window: Option<PixelWindow>,
    pub nodata_per_band: Vec<Option<f64>>,
    pub warnings: Vec<RasterWarning>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskPolarity {
    TrueInside,
    TrueOutside,
}

impl MaskPolarity {
    pub fn parse(value: &str) -> GisResult<Self> {
        match value {
            "true_inside" => Ok(Self::TrueInside),
            "true_outside" => Ok(Self::TrueOutside),
            _ => Err(GisError::InvalidArgument(format!(
                "{MASK_POLARITY_EXPLICIT}: mask_polarity must be 'true_inside' or 'true_outside'"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::TrueInside => "true_inside",
            Self::TrueOutside => "true_outside",
        }
    }

    fn flipped(self) -> Self {
        match self {
            Self::TrueInside => Self::TrueOutside,
            Self::TrueOutside => Self::TrueInside,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RasterMaskPolarity {
    TrueValid,
    TrueInside,
    TrueOutside,
}

impl RasterMaskPolarity {
    pub fn parse(value: Option<&str>) -> GisResult<Self> {
        match value {
            Some("true_valid") => Ok(Self::TrueValid),
            Some("true_inside") => Ok(Self::TrueInside),
            Some("true_outside") => Ok(Self::TrueOutside),
            _ => Err(GisError::InvalidArgument(format!(
                "{MASK_POLARITY_EXPLICIT}: mask_polarity must be explicit: 'true_valid', 'true_inside', or 'true_outside'"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::TrueValid => "true_valid",
            Self::TrueInside => "true_inside",
            Self::TrueOutside => "true_outside",
        }
    }
}

#[derive(Clone)]
struct TargetGrid {
    info: RasterInfo,
    transform: AffineTransform,
    bounds: (f64, f64, f64, f64),
    crs: CrsSpec,
}

#[derive(Clone)]
struct VectorSource {
    features: Vec<Value>,
    crs: CrsSpec,
}

#[derive(Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

type Ring = Vec<Point>;
type Polygon = Vec<Ring>;

pub fn parse_raster_dtype(value: &str) -> GisResult<RasterDType> {
    match value.to_ascii_lowercase().as_str() {
        "uint8" => Ok(RasterDType::UInt8),
        "int16" => Ok(RasterDType::Int16),
        "uint16" => Ok(RasterDType::UInt16),
        "int32" => Ok(RasterDType::Int32),
        "uint32" => Ok(RasterDType::UInt32),
        "float32" => Ok(RasterDType::Float32),
        "float64" => Ok(RasterDType::Float64),
        _ => Err(GisError::UnsupportedDType(format!(
            "{UNSUPPORTED_DTYPE}: unsupported raster dtype {value:?}"
        ))),
    }
}

pub fn rasterize_vectors(
    source: &Value,
    target_info: &RasterInfo,
    options: RasterizeOptions,
) -> GisResult<RasterizeResult> {
    validate_value_for_dtype(options.dtype, options.fill, "fill")?;
    validate_value_for_dtype(options.dtype, options.value, "value")?;
    let grid = target_grid(target_info)?;
    let source = resolve_vector_source(source)?;
    require_same_crs(&source.crs, &grid.crs)?;
    let (values, burned_pixels) = rasterize_source(&source, &grid, &options)?;
    let info = output_info(&grid.info, options.dtype, grid.info.nodata_per_band.clone())?;
    Ok(RasterizeResult {
        values,
        info,
        target_shape: (grid.info.height as usize, grid.info.width as usize),
        target_transform: grid.transform,
        target_bounds: grid.bounds,
        dtype: options.dtype,
        fill: options.fill,
        burned_pixels,
        all_touched: options.all_touched,
        warnings: Vec::new(),
    })
}

pub fn geometry_mask(
    source: &Value,
    target_info: &RasterInfo,
    options: GeometryMaskOptions,
) -> GisResult<GeometryMaskResult> {
    let grid = target_grid(target_info)?;
    let source = resolve_vector_source(source)?;
    require_same_crs(&source.crs, &grid.crs)?;
    let inside = rasterize_bool(&source, &grid, options.all_touched)?;
    let mut polarity = options.mask_polarity;
    let mut mask = match options.mask_polarity {
        MaskPolarity::TrueInside => inside,
        MaskPolarity::TrueOutside => inside.into_iter().map(|value| !value).collect(),
    };
    if options.invert {
        mask.iter_mut().for_each(|value| *value = !*value);
        polarity = polarity.flipped();
    }
    let true_count = mask.iter().filter(|&&value| value).count();
    let false_count = mask.len() - true_count;
    Ok(GeometryMaskResult {
        mask,
        info: output_info(&grid.info, RasterDType::UInt8, vec![None])?,
        mask_polarity: polarity,
        true_count,
        false_count,
        warnings: vec![RasterWarning::new(
            MASK_POLARITY_EXPLICIT,
            "mask polarity is explicit",
            Some("mask_polarity"),
        )],
    })
}

pub fn mask_raster(
    source_array: RasterArray,
    source_info: RasterInfo,
    mask: Vec<bool>,
    mask_shape: &[usize],
    options: MaskRasterOptions,
) -> GisResult<MaskRasterResult> {
    let dtype = source_array.dtype();
    let mut nodata = match options.nodata {
        Some(values) => values,
        None => normalize_nodata_len(source_info.nodata_per_band.clone(), source_array.bands)?,
    };
    for value in nodata.iter().flatten() {
        validate_nodata_for_dtype(dtype, *value)?;
    }
    if let Some(fill) = options.fill {
        validate_value_for_dtype(dtype, fill, "fill")?;
    }

    let valid_mask = expand_mask(mask, mask_shape, &source_array)?;
    let valid_count = valid_mask.iter().filter(|&&value| value).count();
    let mut values = raster_info::raster_to_f64(&source_array);
    apply_mask_values(
        &mut values,
        &valid_mask,
        &source_array,
        options.fill,
        &nodata,
    );

    let mut info = source_info_for_output(&source_info, &source_array, nodata.clone());
    let mut crop_window = None;
    let mut warnings = Vec::new();

    let (values, valid_mask, height, width) = if options.crop {
        match retained_window(&valid_mask, &source_array) {
            Some(window) => {
                crop_window = Some(window);
                update_info_for_window(&mut info, window)?;
                (
                    crop_values(&values, &source_array, window),
                    crop_values_bool(&valid_mask, &source_array, window),
                    window.height as usize,
                    window.width as usize,
                )
            }
            None => {
                crop_window = Some(PixelWindow {
                    col_off: 0,
                    row_off: 0,
                    width: 0,
                    height: 0,
                });
                info.width = 0;
                info.height = 0;
                info.bounds = None;
                warnings.push(RasterWarning::new(
                    EMPTY_RASTER,
                    "empty_raster: mask retains no pixels",
                    Some("mask"),
                ));
                (Vec::new(), Vec::new(), 0, 0)
            }
        }
    } else {
        (values, valid_mask, source_array.height, source_array.width)
    };

    let array = if height == 0 || width == 0 {
        source_array.clone()
    } else {
        raster_info::f64_to_raster_data(dtype, values, source_array.bands, height, width)?
    };
    if nodata.len() != source_array.bands {
        nodata = normalize_nodata_len(nodata, source_array.bands)?;
    }
    Ok(MaskRasterResult {
        array,
        mask: valid_mask,
        info,
        mask_polarity: options.mask_polarity,
        fill: options.fill,
        valid_count,
        crop_window,
        nodata_per_band: nodata,
        warnings,
    })
}

fn target_grid(info: &RasterInfo) -> GisResult<TargetGrid> {
    if info.width == 0 || info.height == 0 {
        return Err(GisError::InvalidArgument(format!(
            "{INVALID_ARGUMENT}: target width and height must be positive"
        )));
    }
    let transform = affine::require_transform(info)?;
    let crs = crs::require_crs(info).map_err(|_| {
        GisError::MissingCrs(format!("{MISSING_CRS}: target raster has no CRS metadata"))
    })?;
    let bounds = affine::raster_bounds(info)?.tuple();
    Ok(TargetGrid {
        info: info.clone(),
        transform,
        bounds,
        crs,
    })
}

fn resolve_vector_source(source: &Value) -> GisResult<VectorSource> {
    let features = vector::normalize_features(source).map_err(invalid_arg_to_geometry)?;
    if features.is_empty() {
        return Err(GisError::InvalidGeometry(format!(
            "{EMPTY_FEATURE_SET}: vector source has zero features"
        )));
    }
    let info_crs = source
        .get("info")
        .map(vector::crs_spec_from_info_json)
        .transpose()?
        .flatten();
    let crs = vector::compatible_metadata_crs(info_crs, vector::geojson_crs(source)?, "source")?
        .ok_or_else(|| {
            GisError::MissingCrs(format!("{MISSING_CRS}: vector source has no CRS metadata"))
        })?;
    Ok(VectorSource { features, crs })
}

fn invalid_arg_to_geometry(error: GisError) -> GisError {
    match error {
        GisError::InvalidArgument(message) => GisError::InvalidGeometry(message),
        other => other,
    }
}

fn require_same_crs(left: &CrsSpec, right: &CrsSpec) -> GisResult<()> {
    if crs::crs_equal(left, right) {
        Ok(())
    } else {
        Err(GisError::CrsMismatch(format!(
            "{CRS_MISMATCH}: source CRS {} does not match target CRS {}",
            crs::canonical_label(left)?,
            crs::canonical_label(right)?
        )))
    }
}

fn rasterize_source(
    source: &VectorSource,
    grid: &TargetGrid,
    options: &RasterizeOptions,
) -> GisResult<(Vec<f64>, usize)> {
    let width = grid.info.width as usize;
    let height = grid.info.height as usize;
    let mut values = vec![options.fill; width * height];
    let mut burned = vec![false; width * height];
    for feature in &source.features {
        let geometry = vector::feature_geometry(feature)?;
        let burn = burn_value(feature, options)?;
        let polygons = parse_polygonal_geometry(geometry)?;
        if polygons.is_empty() {
            return Err(GisError::InvalidGeometry(format!(
                "{EMPTY_GEOMETRY}: geometry is empty"
            )));
        }
        for row in 0..height {
            for col in 0..width {
                if polygons.iter().any(|polygon| {
                    pixel_hits_polygon(polygon, grid.transform, row, col, options.all_touched)
                }) {
                    let index = row * width + col;
                    values[index] = burn;
                    burned[index] = true;
                }
            }
        }
    }
    Ok((values, burned.iter().filter(|&&value| value).count()))
}

fn rasterize_bool(
    source: &VectorSource,
    grid: &TargetGrid,
    all_touched: bool,
) -> GisResult<Vec<bool>> {
    let options = RasterizeOptions {
        value: 1.0,
        attribute: None,
        dtype: RasterDType::UInt8,
        fill: 0.0,
        all_touched,
    };
    rasterize_source(source, grid, &options).map(|(values, _)| {
        values
            .into_iter()
            .map(|value| (value - 1.0).abs() <= f64::EPSILON)
            .collect()
    })
}

fn burn_value(feature: &Value, options: &RasterizeOptions) -> GisResult<f64> {
    let Some(attribute) = options.attribute.as_deref() else {
        return Ok(options.value);
    };
    let value = feature
        .get("properties")
        .and_then(Value::as_object)
        .and_then(|properties| properties.get(attribute))
        .and_then(Value::as_f64)
        .ok_or_else(|| {
            GisError::InvalidArgument(format!(
                "{INVALID_ARGUMENT}: attribute {attribute:?} must exist and be numeric"
            ))
        })?;
    validate_value_for_dtype(options.dtype, value, attribute)?;
    Ok(value)
}

fn parse_polygonal_geometry(value: &Value) -> GisResult<Vec<Polygon>> {
    let object = value.as_object().ok_or_else(|| {
        GisError::InvalidGeometry(format!("{INVALID_GEOMETRY}: geometry must be an object"))
    })?;
    match object.get("type").and_then(Value::as_str) {
        Some("Polygon") => Ok(vec![parse_polygon_coordinates(
            object.get("coordinates").ok_or_else(|| {
                GisError::InvalidGeometry(format!(
                    "{INVALID_GEOMETRY}: Polygon requires coordinates"
                ))
            })?,
        )?]),
        Some("MultiPolygon") => {
            let polygons = object
                .get("coordinates")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    GisError::InvalidGeometry(format!(
                        "{INVALID_GEOMETRY}: MultiPolygon coordinates must be an array"
                    ))
                })?;
            if polygons.is_empty() {
                return Err(GisError::InvalidGeometry(format!(
                    "{EMPTY_GEOMETRY}: geometry is empty"
                )));
            }
            polygons
                .iter()
                .map(parse_polygon_coordinates)
                .collect::<GisResult<Vec<_>>>()
        }
        Some(other) => Err(GisError::InvalidGeometry(format!(
            "{UNSUPPORTED_GEOMETRY_TYPE}: rasterize_vectors supports Polygon and MultiPolygon, got {other}"
        ))),
        None => Err(GisError::InvalidGeometry(format!(
            "{INVALID_GEOMETRY}: geometry requires a type string"
        ))),
    }
}

fn parse_polygon_coordinates(value: &Value) -> GisResult<Polygon> {
    let rings = value.as_array().ok_or_else(|| {
        GisError::InvalidGeometry(format!(
            "{INVALID_GEOMETRY}: Polygon coordinates must be an array"
        ))
    })?;
    if rings.is_empty() {
        return Err(GisError::InvalidGeometry(format!(
            "{EMPTY_GEOMETRY}: geometry is empty"
        )));
    }
    rings.iter().map(parse_ring).collect::<GisResult<Vec<_>>>()
}

fn parse_ring(value: &Value) -> GisResult<Ring> {
    let points = value.as_array().ok_or_else(|| {
        GisError::InvalidGeometry(format!("{INVALID_GEOMETRY}: polygon ring must be an array"))
    })?;
    if points.len() < 4 {
        return Err(GisError::InvalidGeometry(format!(
            "{INVALID_GEOMETRY}: polygon ring must contain at least four positions"
        )));
    }
    points.iter().map(parse_point).collect()
}

fn parse_point(value: &Value) -> GisResult<Point> {
    let items = value.as_array().ok_or_else(|| {
        GisError::InvalidGeometry(format!("{INVALID_GEOMETRY}: position must be an array"))
    })?;
    if items.len() < 2 {
        return Err(GisError::InvalidGeometry(format!(
            "{INVALID_GEOMETRY}: position requires x and y"
        )));
    }
    let x = items[0].as_f64().ok_or_else(|| {
        GisError::InvalidGeometry(format!("{INVALID_GEOMETRY}: x coordinate must be numeric"))
    })?;
    let y = items[1].as_f64().ok_or_else(|| {
        GisError::InvalidGeometry(format!("{INVALID_GEOMETRY}: y coordinate must be numeric"))
    })?;
    if !x.is_finite() || !y.is_finite() {
        return Err(GisError::InvalidGeometry(format!(
            "{INVALID_GEOMETRY}: coordinates must be finite"
        )));
    }
    Ok(Point { x, y })
}

fn pixel_hits_polygon(
    polygon: &Polygon,
    transform: AffineTransform,
    row: usize,
    col: usize,
    all_touched: bool,
) -> bool {
    let center = affine::xy(transform, row as i64, col as i64)
        .map(|(x, y)| Point { x, y })
        .ok();
    if center.is_some_and(|point| point_in_polygon(point, polygon)) {
        return true;
    }
    if !all_touched {
        return false;
    }
    let corners = cell_corners(transform, row, col);
    if corners
        .iter()
        .any(|&corner| point_in_polygon(corner, polygon))
    {
        return true;
    }
    if polygon
        .iter()
        .flat_map(|ring| ring.iter())
        .any(|&point| point_in_cell(transform, point, row, col))
    {
        return true;
    }
    for ring in polygon {
        for segment in ring.windows(2) {
            if cell_edges(corners)
                .iter()
                .any(|edge| segments_intersect(segment[0], segment[1], edge.0, edge.1))
            {
                return true;
            }
        }
    }
    false
}

fn cell_corners(transform: AffineTransform, row: usize, col: usize) -> [Point; 4] {
    let row = row as f64;
    let col = col as f64;
    [
        point_from_tuple(transform.apply(col, row)),
        point_from_tuple(transform.apply(col + 1.0, row)),
        point_from_tuple(transform.apply(col + 1.0, row + 1.0)),
        point_from_tuple(transform.apply(col, row + 1.0)),
    ]
}

fn point_from_tuple(value: (f64, f64)) -> Point {
    Point {
        x: value.0,
        y: value.1,
    }
}

fn cell_edges(corners: [Point; 4]) -> [(Point, Point); 4] {
    [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    ]
}

fn point_in_cell(transform: AffineTransform, point: Point, row: usize, col: usize) -> bool {
    affine::inverse_apply(transform, point.x, point.y).is_ok_and(|(pixel_col, pixel_row)| {
        let col = col as f64;
        let row = row as f64;
        pixel_col >= col && pixel_col <= col + 1.0 && pixel_row >= row && pixel_row <= row + 1.0
    })
}

fn point_in_polygon(point: Point, polygon: &Polygon) -> bool {
    let Some(shell) = polygon.first() else {
        return false;
    };
    point_in_ring(point, shell)
        && !polygon
            .iter()
            .skip(1)
            .any(|ring| point_in_ring(point, ring))
}

fn point_in_ring(point: Point, ring: &Ring) -> bool {
    if ring
        .windows(2)
        .any(|segment| point_on_segment(point, segment[0], segment[1]))
    {
        return true;
    }
    let mut inside = false;
    for segment in ring.windows(2) {
        let a = segment[0];
        let b = segment[1];
        if (a.y > point.y) != (b.y > point.y) {
            let x = (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x;
            if point.x < x {
                inside = !inside;
            }
        }
    }
    inside
}

fn point_on_segment(point: Point, a: Point, b: Point) -> bool {
    let cross = (point.y - a.y) * (b.x - a.x) - (point.x - a.x) * (b.y - a.y);
    if cross.abs() > 1.0e-10 {
        return false;
    }
    let dot = (point.x - a.x) * (b.x - a.x) + (point.y - a.y) * (b.y - a.y);
    if dot < 0.0 {
        return false;
    }
    let len_sq = (b.x - a.x).powi(2) + (b.y - a.y).powi(2);
    dot <= len_sq
}

fn segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool {
    let o1 = orientation(a, b, c);
    let o2 = orientation(a, b, d);
    let o3 = orientation(c, d, a);
    let o4 = orientation(c, d, b);
    if o1.abs() <= 1.0e-10 && point_on_segment(c, a, b) {
        return true;
    }
    if o2.abs() <= 1.0e-10 && point_on_segment(d, a, b) {
        return true;
    }
    if o3.abs() <= 1.0e-10 && point_on_segment(a, c, d) {
        return true;
    }
    if o4.abs() <= 1.0e-10 && point_on_segment(b, c, d) {
        return true;
    }
    (o1 > 0.0) != (o2 > 0.0) && (o3 > 0.0) != (o4 > 0.0)
}

fn orientation(a: Point, b: Point, c: Point) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn validate_value_for_dtype(dtype: RasterDType, value: f64, field: &str) -> GisResult<()> {
    if dtype.nodata_fits(value) {
        Ok(())
    } else {
        Err(GisError::InvalidArgument(format!(
            "{INVALID_ARGUMENT}: {field} value {value} does not fit {}",
            dtype.name()
        )))
    }
}

fn validate_nodata_for_dtype(dtype: RasterDType, value: f64) -> GisResult<()> {
    if dtype.nodata_fits(value) {
        Ok(())
    } else {
        Err(GisError::InvalidNodata(format!(
            "{INVALID_NODATA}: nodata value {value} does not fit {}",
            dtype.name()
        )))
    }
}

fn normalize_nodata_len(
    mut values: Vec<Option<f64>>,
    band_count: usize,
) -> GisResult<Vec<Option<f64>>> {
    if values.is_empty() {
        values = vec![None; band_count];
    }
    if values.len() == 1 && band_count > 1 {
        return Ok(vec![values[0]; band_count]);
    }
    if values.len() != band_count {
        return Err(GisError::InvalidNodata(format!(
            "{INVALID_NODATA}: nodata length {} does not match band count {band_count}",
            values.len()
        )));
    }
    Ok(values)
}

fn expand_mask(mask: Vec<bool>, shape: &[usize], array: &RasterArray) -> GisResult<Vec<bool>> {
    let pixels = array.height * array.width;
    match shape {
        [height, width] if *height == array.height && *width == array.width => {
            let mut out = Vec::with_capacity(array.bands * pixels);
            for _ in 0..array.bands {
                out.extend(mask.iter().copied());
            }
            Ok(out)
        }
        [bands, height, width]
            if *bands == 1 && *height == array.height && *width == array.width =>
        {
            let mut out = Vec::with_capacity(array.bands * pixels);
            for _ in 0..array.bands {
                out.extend(mask.iter().copied());
            }
            Ok(out)
        }
        [bands, height, width]
            if *bands == array.bands && *height == array.height && *width == array.width =>
        {
            Ok(mask)
        }
        _ => Err(GisError::ShapeMismatch(format!(
            "{SHAPE_MISMATCH}: mask shape {:?} is not compatible with raster shape ({}, {}, {})",
            shape, array.bands, array.height, array.width
        ))),
    }
}

fn apply_mask_values(
    values: &mut [f64],
    valid_mask: &[bool],
    array: &RasterArray,
    fill: Option<f64>,
    nodata: &[Option<f64>],
) {
    let pixels = array.height * array.width;
    for band in 0..array.bands {
        let fallback = fill.or_else(|| nodata.get(band).copied().flatten());
        let Some(fill_value) = fallback else {
            continue;
        };
        for pixel in 0..pixels {
            let index = band * pixels + pixel;
            if !valid_mask[index] {
                values[index] = fill_value;
            }
        }
    }
}

fn retained_window(mask: &[bool], array: &RasterArray) -> Option<PixelWindow> {
    let pixels = array.height * array.width;
    let mut min_row = array.height;
    let mut min_col = array.width;
    let mut max_row = 0usize;
    let mut max_col = 0usize;
    let mut found = false;
    for band in 0..array.bands {
        for row in 0..array.height {
            for col in 0..array.width {
                if mask[band * pixels + row * array.width + col] {
                    found = true;
                    min_row = min_row.min(row);
                    min_col = min_col.min(col);
                    max_row = max_row.max(row);
                    max_col = max_col.max(col);
                }
            }
        }
    }
    found.then(|| PixelWindow {
        col_off: min_col as i64,
        row_off: min_row as i64,
        width: (max_col - min_col + 1) as u32,
        height: (max_row - min_row + 1) as u32,
    })
}

fn crop_values(values: &[f64], array: &RasterArray, window: PixelWindow) -> Vec<f64> {
    let mut out = Vec::with_capacity(array.bands * window.height as usize * window.width as usize);
    let pixels = array.height * array.width;
    for band in 0..array.bands {
        for row in window.row_off as usize..(window.row_off as usize + window.height as usize) {
            let start = band * pixels + row * array.width + window.col_off as usize;
            out.extend_from_slice(&values[start..start + window.width as usize]);
        }
    }
    out
}

fn crop_values_bool(values: &[bool], array: &RasterArray, window: PixelWindow) -> Vec<bool> {
    let mut out = Vec::with_capacity(array.bands * window.height as usize * window.width as usize);
    let pixels = array.height * array.width;
    for band in 0..array.bands {
        for row in window.row_off as usize..(window.row_off as usize + window.height as usize) {
            let start = band * pixels + row * array.width + window.col_off as usize;
            out.extend_from_slice(&values[start..start + window.width as usize]);
        }
    }
    out
}

fn source_info_for_output(
    source_info: &RasterInfo,
    array: &RasterArray,
    nodata: Vec<Option<f64>>,
) -> RasterInfo {
    let mut info = source_info.clone();
    info.width = array.width as u32;
    info.height = array.height as u32;
    info.band_count = array.bands as u16;
    info.dtype_per_band = vec![array.dtype().name().to_string(); array.bands];
    info.nodata_per_band = nodata;
    info
}

fn output_info(
    source: &RasterInfo,
    dtype: RasterDType,
    nodata: Vec<Option<f64>>,
) -> GisResult<RasterInfo> {
    let mut info = source.clone();
    info.band_count = 1;
    info.dtype_per_band = vec![dtype.name().to_string()];
    info.nodata_per_band = normalize_nodata_len(nodata, 1)?;
    info.bounds = Some(affine::raster_bounds(&info)?.tuple());
    info.resolution = Some(affine::raster_resolution(&info)?);
    info.is_georeferenced =
        info.transform.is_some() && (info.crs_wkt.is_some() || info.crs_authority.is_some());
    Ok(info)
}

fn update_info_for_window(info: &mut RasterInfo, window: PixelWindow) -> GisResult<()> {
    info.width = window.width;
    info.height = window.height;
    if info.transform.is_some() {
        let transform = affine::window_transform(info, window)?;
        info.transform = Some(transform.tuple());
        info.bounds = Some(transform.bounds(window.width, window.height).tuple());
        info.resolution = Some(transform.resolution());
    }
    Ok(())
}

#[cfg(feature = "extension-module")]
mod py {
    use super::*;

    use std::collections::HashMap;

    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::prelude::*;
    use pyo3::types::{PyAny, PyDict, PyDictMethods, PyList, PyTuple};
    use pyo3::IntoPy;

    use crate::gis::py_json::{py_to_json_strict, warnings_to_py};

    #[pyfunction(
        name = "rasterize_vectors",
        signature = (vectors, target_info, *, value = 1.0, attribute = None, dtype = "uint8", fill = 0.0, all_touched = false)
    )]
    pub fn rasterize_vectors_py(
        py: Python<'_>,
        vectors: &Bound<'_, PyAny>,
        target_info: &Bound<'_, PyAny>,
        value: f64,
        attribute: Option<String>,
        dtype: &str,
        fill: f64,
        all_touched: bool,
    ) -> PyResult<PyObject> {
        let source = vector_source_to_json(vectors)?;
        let target_info = raster_info_from_py(target_info)?;
        let result = super::rasterize_vectors(
            &source,
            &target_info,
            RasterizeOptions {
                value,
                attribute,
                dtype: parse_raster_dtype(dtype)?,
                fill,
                all_touched,
            },
        )?;
        rasterize_result_to_py(py, &result)
    }

    #[pyfunction(
        name = "geometry_mask",
        signature = (geometries, target_info, *, invert = false, all_touched = false, mask_polarity = "true_inside")
    )]
    pub fn geometry_mask_py(
        py: Python<'_>,
        geometries: &Bound<'_, PyAny>,
        target_info: &Bound<'_, PyAny>,
        invert: bool,
        all_touched: bool,
        mask_polarity: &str,
    ) -> PyResult<PyObject> {
        let source = vector_source_to_json(geometries)?;
        let target_info = raster_info_from_py(target_info)?;
        let result = super::geometry_mask(
            &source,
            &target_info,
            GeometryMaskOptions {
                invert,
                all_touched,
                mask_polarity: MaskPolarity::parse(mask_polarity)?,
            },
        )?;
        geometry_mask_result_to_py(py, &result)
    }

    #[pyfunction(
        name = "mask_raster",
        signature = (source, mask, *, mask_polarity = None, crop = false, fill = None, nodata = None)
    )]
    pub fn mask_raster_py(
        py: Python<'_>,
        source: &Bound<'_, PyAny>,
        mask: &Bound<'_, PyAny>,
        mask_polarity: Option<String>,
        crop: bool,
        fill: Option<&Bound<'_, PyAny>>,
        nodata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let (array, info) = raster_source_from_py(source)?;
        let (mask, mask_shape) = super::super::extract_typed_array::<bool>(mask)?;
        let fill = optional_f64(fill, "fill")?;
        let nodata = optional_nodata(nodata, array.bands, array.dtype())?;
        let result = super::mask_raster(
            array,
            info,
            mask,
            &mask_shape,
            MaskRasterOptions {
                mask_polarity: RasterMaskPolarity::parse(mask_polarity.as_deref())?,
                crop,
                fill,
                nodata,
            },
        )?;
        mask_raster_result_to_py(py, &result)
    }

    fn vector_source_to_json(value: &Bound<'_, PyAny>) -> PyResult<Value> {
        if let Ok(path) = value.extract::<String>() {
            let result = vector::read_vector(
                path,
                vector::VectorReadOptions {
                    layer: None,
                    columns: None,
                    bbox: None,
                    limit: None,
                },
            )?;
            return vector_result_to_json(result);
        }
        py_to_json_strict(value)
    }

    fn vector_result_to_json(result: vector::VectorReadResult) -> PyResult<Value> {
        let info = serde_json::to_value(vector_info_map(&result.info)).map_err(|err| {
            GisError::InvalidArgument(format!(
                "{INVALID_ARGUMENT}: failed to serialize info: {err}"
            ))
        })?;
        Ok(serde_json::json!({
            "type": "FeatureCollection",
            "features": result.features,
            "info": info,
            "warnings": warnings_json(&result.warnings),
        }))
    }

    fn vector_info_map(info: &vector::VectorInfo) -> HashMap<&'static str, Value> {
        let mut out = HashMap::new();
        out.insert("path", Value::String(info.path.clone()));
        out.insert("driver", Value::String(info.driver.clone()));
        out.insert(
            "layer_name",
            info.layer_name
                .clone()
                .map(Value::String)
                .unwrap_or(Value::Null),
        );
        out.insert("layer_count", Value::from(info.layer_count));
        out.insert("geometry_type", Value::String(info.geometry_type.clone()));
        out.insert("feature_count", Value::from(info.feature_count));
        out.insert(
            "crs_wkt",
            info.crs_wkt
                .clone()
                .map(Value::String)
                .unwrap_or(Value::Null),
        );
        out.insert(
            "crs_authority",
            serde_json::to_value(&info.crs_authority).unwrap_or(Value::Null),
        );
        out.insert(
            "bounds",
            serde_json::to_value(info.bounds).unwrap_or(Value::Null),
        );
        out.insert("is_georeferenced", Value::Bool(info.is_georeferenced));
        out
    }

    fn warnings_json(warnings: &[RasterWarning]) -> Value {
        Value::Array(
            warnings
                .iter()
                .map(|warning| {
                    serde_json::json!({
                        "code": warning.code,
                        "message": warning.message,
                        "field": warning.field,
                    })
                })
                .collect(),
        )
    }

    fn raster_info_from_py(value: &Bound<'_, PyAny>) -> PyResult<RasterInfo> {
        if let Ok(info) = value.extract::<PyRef<'_, RasterInfo>>() {
            return Ok(info.clone());
        }
        let dict = value.downcast::<PyDict>().map_err(|_| {
            GisError::InvalidArgument(format!(
                "{INVALID_ARGUMENT}: raster info must be RasterInfo or dict"
            ))
        })?;
        let width = required_u32(dict, "width")?;
        let height = required_u32(dict, "height")?;
        let band_count = dict
            .get_item("band_count")?
            .map(|value| value.extract::<u16>())
            .transpose()?
            .unwrap_or(1);
        let transform = optional_tuple6(dict.get_item("transform")?)?;
        let dtype_per_band = dict
            .get_item("dtype_per_band")?
            .map(|value| value.extract::<Vec<String>>())
            .transpose()?
            .unwrap_or_else(|| vec!["uint8".to_string(); band_count as usize]);
        let nodata_per_band = dict
            .get_item("nodata_per_band")?
            .map(|value| value.extract::<Vec<Option<f64>>>())
            .transpose()?
            .unwrap_or_else(|| vec![None; band_count as usize]);
        let mut info = RasterInfo::new(
            dict.get_item("path")?
                .and_then(|value| value.extract::<String>().ok())
                .unwrap_or_default()
                .into(),
            width,
            height,
            band_count,
        );
        info.driver = dict
            .get_item("driver")?
            .and_then(|value| value.extract::<String>().ok())
            .unwrap_or_else(|| "memory".to_string());
        info.dtype_per_band = dtype_per_band;
        info.crs_wkt = optional_string(dict.get_item("crs_wkt")?)?;
        info.crs_authority = dict
            .get_item("crs_authority")?
            .map(|value| {
                if value.is_none() {
                    Ok(None)
                } else {
                    value.extract::<HashMap<String, String>>().map(Some)
                }
            })
            .transpose()?
            .flatten();
        info.transform = transform;
        info.bounds = optional_tuple4(dict.get_item("bounds")?)?;
        if info.bounds.is_none() && info.transform.is_some() && width > 0 && height > 0 {
            info.bounds = Some(affine::raster_bounds(&info)?.tuple());
        }
        info.resolution = optional_tuple2(dict.get_item("resolution")?)?;
        if info.resolution.is_none() && info.transform.is_some() {
            info.resolution = Some(affine::raster_resolution(&info)?);
        }
        info.nodata_per_band = nodata_per_band;
        info.is_georeferenced =
            info.transform.is_some() && (info.crs_wkt.is_some() || info.crs_authority.is_some());
        Ok(info)
    }

    fn raster_source_from_py(value: &Bound<'_, PyAny>) -> PyResult<(RasterArray, RasterInfo)> {
        if let Ok(path) = value.extract::<String>() {
            let result = raster_info::read_raster(path, None, None, false)?;
            return Ok((result.array, result.info));
        }
        if let Ok(dict) = value.downcast::<PyDict>() {
            if let Some(array_value) = dict.get_item("array")? {
                let array = super::super::extract_raster_array(&array_value)?;
                let info = if let Some(info_value) = dict.get_item("info")? {
                    raster_info_from_py(&info_value)?
                } else {
                    synthetic_info(&array)
                };
                validate_info_shape(&info, &array)?;
                return Ok((array, info));
            }
        }
        let array = super::super::extract_raster_array(value)?;
        let info = synthetic_info(&array);
        Ok((array, info))
    }

    fn synthetic_info(array: &RasterArray) -> RasterInfo {
        let mut info = RasterInfo::new(
            "".into(),
            array.width as u32,
            array.height as u32,
            array.bands as u16,
        );
        info.driver = "memory".to_string();
        info.dtype_per_band = vec![array.dtype().name().to_string(); array.bands];
        info.nodata_per_band = vec![None; array.bands];
        info
    }

    fn validate_info_shape(info: &RasterInfo, array: &RasterArray) -> PyResult<()> {
        if info.width as usize != array.width || info.height as usize != array.height {
            return Err(GisError::ShapeMismatch(format!(
                "{SHAPE_MISMATCH}: source info shape {}x{} does not match array shape {}x{}",
                info.width, info.height, array.width, array.height
            ))
            .into());
        }
        Ok(())
    }

    fn required_u32(dict: &Bound<'_, PyDict>, key: &'static str) -> PyResult<u32> {
        dict.get_item(key)?
            .ok_or_else(|| {
                GisError::InvalidArgument(format!("{INVALID_ARGUMENT}: raster info missing {key}"))
            })?
            .extract::<u32>()
            .map_err(Into::into)
    }

    fn optional_string(value: Option<Bound<'_, PyAny>>) -> PyResult<Option<String>> {
        value
            .map(|value| {
                if value.is_none() {
                    Ok(None)
                } else {
                    value.extract::<String>().map(Some)
                }
            })
            .transpose()
            .map(Option::flatten)
    }

    fn optional_tuple6(
        value: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Option<(f64, f64, f64, f64, f64, f64)>> {
        value
            .map(|value| {
                if value.is_none() {
                    Ok(None)
                } else {
                    value.extract::<(f64, f64, f64, f64, f64, f64)>().map(Some)
                }
            })
            .transpose()
            .map(Option::flatten)
    }

    fn optional_tuple4(value: Option<Bound<'_, PyAny>>) -> PyResult<Option<(f64, f64, f64, f64)>> {
        value
            .map(|value| {
                if value.is_none() {
                    Ok(None)
                } else {
                    value.extract::<(f64, f64, f64, f64)>().map(Some)
                }
            })
            .transpose()
            .map(Option::flatten)
    }

    fn optional_tuple2(value: Option<Bound<'_, PyAny>>) -> PyResult<Option<(f64, f64)>> {
        value
            .map(|value| {
                if value.is_none() {
                    Ok(None)
                } else {
                    value.extract::<(f64, f64)>().map(Some)
                }
            })
            .transpose()
            .map(Option::flatten)
    }

    fn optional_f64(value: Option<&Bound<'_, PyAny>>, field: &str) -> PyResult<Option<f64>> {
        let Some(value) = value else {
            return Ok(None);
        };
        if value.is_none() {
            return Ok(None);
        }
        value.extract::<f64>().map(Some).map_err(|_| {
            GisError::InvalidArgument(format!("{INVALID_ARGUMENT}: {field} must be numeric")).into()
        })
    }

    fn optional_nodata(
        value: Option<&Bound<'_, PyAny>>,
        bands: usize,
        dtype: RasterDType,
    ) -> PyResult<Option<Vec<Option<f64>>>> {
        let Some(value) = value else {
            return Ok(None);
        };
        if value.is_none() {
            return Ok(None);
        }
        let values = if let Ok(number) = value.extract::<f64>() {
            vec![Some(number); bands]
        } else if let Ok(list) = value.downcast::<PyList>() {
            nodata_from_iter(list.iter(), bands)?
        } else if let Ok(tuple) = value.downcast::<PyTuple>() {
            nodata_from_iter(tuple.iter(), bands)?
        } else {
            return Err(GisError::InvalidNodata(format!(
                "{INVALID_NODATA}: nodata must be a scalar or per-band list"
            ))
            .into());
        };
        for item in values.iter().flatten() {
            validate_nodata_for_dtype(dtype, *item)?;
        }
        Ok(Some(values))
    }

    fn nodata_from_iter<'py>(
        items: impl Iterator<Item = Bound<'py, PyAny>>,
        bands: usize,
    ) -> PyResult<Vec<Option<f64>>> {
        let values = items
            .map(|item| {
                if item.is_none() {
                    Ok(None)
                } else {
                    item.extract::<f64>().map(Some).map_err(|_| {
                        GisError::InvalidNodata(format!(
                            "{INVALID_NODATA}: nodata list values must be numeric or None"
                        ))
                        .into()
                    })
                }
            })
            .collect::<PyResult<Vec<_>>>()?;
        if values.len() != bands {
            return Err(GisError::InvalidNodata(format!(
                "{INVALID_NODATA}: nodata length {} does not match band count {bands}",
                values.len()
            ))
            .into());
        }
        Ok(values)
    }

    fn rasterize_result_to_py(py: Python<'_>, result: &RasterizeResult) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item(
            "array",
            values_to_py_2d(
                py,
                result.dtype,
                result.values.clone(),
                result.target_shape.0,
                result.target_shape.1,
            )?,
        )?;
        dict.set_item(
            "info",
            super::super::raster_info_to_py_dict(py, &result.info)?,
        )?;
        dict.set_item("target_shape", result.target_shape)?;
        dict.set_item("target_transform", result.target_transform.tuple())?;
        dict.set_item("target_bounds", result.target_bounds)?;
        dict.set_item("dtype", result.dtype.name())?;
        dict.set_item("fill", result.fill)?;
        dict.set_item("burned_pixels", result.burned_pixels)?;
        dict.set_item("all_touched", result.all_touched)?;
        dict.set_item("warnings", warnings_to_py(py, &result.warnings)?)?;
        Ok(dict.into_py(py))
    }

    fn geometry_mask_result_to_py(
        py: Python<'_>,
        result: &GeometryMaskResult,
    ) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        let height = result.info.height as usize;
        let width = result.info.width as usize;
        dict.set_item(
            "mask",
            PyArray1::from_vec_bound(py, result.mask.clone())
                .reshape([height, width])?
                .into_py(py),
        )?;
        dict.set_item(
            "info",
            super::super::raster_info_to_py_dict(py, &result.info)?,
        )?;
        dict.set_item("mask_polarity", result.mask_polarity.as_str())?;
        dict.set_item("true_count", result.true_count)?;
        dict.set_item("false_count", result.false_count)?;
        dict.set_item("crop_window", py.None())?;
        dict.set_item("warnings", warnings_to_py(py, &result.warnings)?)?;
        Ok(dict.into_py(py))
    }

    fn mask_raster_result_to_py(py: Python<'_>, result: &MaskRasterResult) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        let empty_crop = result
            .crop_window
            .is_some_and(|window| window.width == 0 || window.height == 0);
        if empty_crop {
            dict.set_item(
                "array",
                empty_array_to_py(py, result.array.dtype(), result.array.bands)?,
            )?;
            dict.set_item(
                "mask",
                PyArray1::from_vec_bound(py, Vec::<bool>::new())
                    .reshape([result.array.bands, 0, 0])?
                    .into_py(py),
            )?;
        } else {
            dict.set_item(
                "array",
                super::super::raster_array_to_py(py, &result.array)?,
            )?;
            dict.set_item(
                "mask",
                super::super::bool_array_to_py_shape(
                    py,
                    result.mask.clone(),
                    (result.array.bands, result.array.height, result.array.width),
                )?,
            )?;
        }
        dict.set_item(
            "info",
            super::super::raster_info_to_py_dict(py, &result.info)?,
        )?;
        dict.set_item("mask_polarity", result.mask_polarity.as_str())?;
        dict.set_item("fill", result.fill)?;
        dict.set_item("nodata", result.nodata_per_band.clone())?;
        dict.set_item("valid_count", result.valid_count)?;
        let true_count = result.mask.iter().filter(|&&value| value).count();
        dict.set_item("true_count", true_count)?;
        dict.set_item("false_count", result.mask.len() - true_count)?;
        dict.set_item(
            "crop_window",
            result
                .crop_window
                .map(|window| (window.col_off, window.row_off, window.width, window.height)),
        )?;
        dict.set_item("nodata_per_band", result.nodata_per_band.clone())?;
        dict.set_item("warnings", warnings_to_py(py, &result.warnings)?)?;
        Ok(dict.into_py(py))
    }

    fn empty_array_to_py(py: Python<'_>, dtype: RasterDType, bands: usize) -> PyResult<PyObject> {
        match dtype {
            RasterDType::UInt8 => Ok(PyArray1::from_vec_bound(py, Vec::<u8>::new())
                .reshape([bands, 0, 0])?
                .into_py(py)),
            RasterDType::Int16 => Ok(PyArray1::from_vec_bound(py, Vec::<i16>::new())
                .reshape([bands, 0, 0])?
                .into_py(py)),
            RasterDType::UInt16 => Ok(PyArray1::from_vec_bound(py, Vec::<u16>::new())
                .reshape([bands, 0, 0])?
                .into_py(py)),
            RasterDType::Int32 => Ok(PyArray1::from_vec_bound(py, Vec::<i32>::new())
                .reshape([bands, 0, 0])?
                .into_py(py)),
            RasterDType::UInt32 => Ok(PyArray1::from_vec_bound(py, Vec::<u32>::new())
                .reshape([bands, 0, 0])?
                .into_py(py)),
            RasterDType::Float32 => Ok(PyArray1::from_vec_bound(py, Vec::<f32>::new())
                .reshape([bands, 0, 0])?
                .into_py(py)),
            RasterDType::Float64 => Ok(PyArray1::from_vec_bound(py, Vec::<f64>::new())
                .reshape([bands, 0, 0])?
                .into_py(py)),
        }
    }

    fn values_to_py_2d(
        py: Python<'_>,
        dtype: RasterDType,
        values: Vec<f64>,
        height: usize,
        width: usize,
    ) -> PyResult<PyObject> {
        let array = raster_info::f64_to_raster_data(dtype, values, 1, height, width)?;
        match array.data {
            crate::gis::raster_write::RasterData::U8(data) => {
                Ok(PyArray1::from_vec_bound(py, data)
                    .reshape([height, width])?
                    .into_py(py))
            }
            crate::gis::raster_write::RasterData::I16(data) => {
                Ok(PyArray1::from_vec_bound(py, data)
                    .reshape([height, width])?
                    .into_py(py))
            }
            crate::gis::raster_write::RasterData::U16(data) => {
                Ok(PyArray1::from_vec_bound(py, data)
                    .reshape([height, width])?
                    .into_py(py))
            }
            crate::gis::raster_write::RasterData::I32(data) => {
                Ok(PyArray1::from_vec_bound(py, data)
                    .reshape([height, width])?
                    .into_py(py))
            }
            crate::gis::raster_write::RasterData::U32(data) => {
                Ok(PyArray1::from_vec_bound(py, data)
                    .reshape([height, width])?
                    .into_py(py))
            }
            crate::gis::raster_write::RasterData::F32(data) => {
                Ok(PyArray1::from_vec_bound(py, data)
                    .reshape([height, width])?
                    .into_py(py))
            }
            crate::gis::raster_write::RasterData::F64(data) => {
                Ok(PyArray1::from_vec_bound(py, data)
                    .reshape([height, width])?
                    .into_py(py))
            }
        }
    }
}

#[cfg(feature = "extension-module")]
pub use py::{geometry_mask_py, mask_raster_py, rasterize_vectors_py};
