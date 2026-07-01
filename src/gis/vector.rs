use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Map, Number, Value};

use crate::gis::affine::validate_bounds_tuple;
use crate::gis::crs;
use crate::gis::error::{GisError, GisResult};
use crate::gis::raster_write::CrsSpec;
use crate::gis::types::{RasterBounds, RasterWarning, WARNING_MISSING_CRS};

pub const WARNING_EMPTY_FEATURE_SET: &str = "empty_feature_set";
pub const WARNING_EMPTY_GEOMETRY: &str = "empty_geometry";
pub const WARNING_INVALID_GEOMETRY: &str = "invalid_geometry";

#[derive(Debug, Clone, PartialEq)]
pub struct SchemaField {
    pub name: String,
    pub field_type: String,
    pub nullable: Option<bool>,
    pub width: Option<u32>,
    pub precision: Option<u32>,
}

#[cfg_attr(
    feature = "extension-module",
    pyo3::pyclass(module = "forge3d._forge3d", name = "VectorInfo")
)]
#[derive(Debug, Clone, PartialEq)]
pub struct VectorInfo {
    pub path: String,
    pub driver: String,
    pub layer_name: Option<String>,
    pub layer_count: u32,
    pub geometry_type: String,
    pub feature_count: u64,
    pub schema: Vec<SchemaField>,
    pub crs_wkt: Option<String>,
    pub crs_authority: Option<HashMap<String, String>>,
    pub bounds: Option<(f64, f64, f64, f64)>,
    pub is_georeferenced: bool,
    pub warnings: Vec<RasterWarning>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorReadOptions {
    pub layer: Option<String>,
    pub columns: Option<Vec<String>>,
    pub bbox: Option<(f64, f64, f64, f64)>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorReadResult {
    pub features: Vec<Value>,
    pub info: VectorInfo,
    pub warnings: Vec<RasterWarning>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VectorReprojectInput {
    Path(PathBuf),
    Features {
        features: Vec<Value>,
        info: Option<VectorInfo>,
        geojson_crs: Option<CrsSpec>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorReprojectResult {
    pub features: Vec<Value>,
    pub info: VectorInfo,
    pub src_crs: CrsSpec,
    pub dst_crs: CrsSpec,
    pub bounds: Option<(f64, f64, f64, f64)>,
    pub feature_count: u64,
    pub warnings: Vec<RasterWarning>,
}

pub fn read_vector(
    path: impl AsRef<Path>,
    options: VectorReadOptions,
) -> GisResult<VectorReadResult> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(GisError::NotFound(path.to_path_buf()));
    }
    validate_vector_path(path)?;

    let text = fs::read_to_string(path)?;
    let root: Value = serde_json::from_str(&text).map_err(|err| {
        GisError::InvalidArgument(format!("invalid_geometry: invalid GeoJSON document: {err}"))
    })?;
    let layer_name = geojson_layer_name(path, &root);
    if let Some(layer) = options.layer.as_deref() {
        if Some(layer) != layer_name.as_deref() {
            return Err(GisError::InvalidArgument(format!(
                "missing_layer: GeoJSON source has layer {:?}, not {layer:?}",
                layer_name
            )));
        }
    }

    let mut warnings = Vec::new();
    let mut features = normalize_features(&root)?;
    let crs = geojson_crs(&root)?;

    if let Some(bbox) = options.bbox {
        let bbox = validate_bounds_tuple(bbox, false)?;
        features.retain(|feature| {
            feature_bounds(feature).is_some_and(|bounds| bounds_intersects(bounds, bbox))
        });
    }
    if let Some(limit) = options.limit {
        features.truncate(limit);
    }
    if let Some(columns) = options.columns.as_ref() {
        features = filter_columns(features, columns);
    }

    let geometry_type = geometry_type_for_features(&features);
    let bounds = bounds_for_features(&features).map(RasterBounds::tuple);
    if features.is_empty() {
        warnings.push(RasterWarning::new(
            WARNING_EMPTY_FEATURE_SET,
            "vector source or filtered result has zero features",
            Some("features"),
        ));
    }
    let (crs_wkt, crs_authority) = match crs {
        Some(spec) => (spec.wkt.clone(), crs::authority_map(&spec)),
        None => {
            warnings.push(RasterWarning::new(
                WARNING_MISSING_CRS,
                "vector source has no CRS metadata",
                Some("crs_wkt"),
            ));
            (None, None)
        }
    };

    let is_georeferenced = bounds.is_some() && (crs_wkt.is_some() || crs_authority.is_some());
    let info = VectorInfo {
        path: path.to_string_lossy().to_string(),
        driver: "GeoJSON".to_string(),
        layer_name,
        layer_count: 1,
        geometry_type,
        feature_count: features.len() as u64,
        schema: infer_schema(&features),
        crs_wkt,
        crs_authority,
        bounds,
        is_georeferenced,
        warnings: warnings.clone(),
    };
    Ok(VectorReadResult {
        features,
        info,
        warnings,
    })
}

pub fn read_vector_info(path: impl AsRef<Path>, layer: Option<String>) -> GisResult<VectorInfo> {
    Ok(read_vector(
        path,
        VectorReadOptions {
            layer,
            columns: None,
            bbox: None,
            limit: None,
        },
    )?
    .info)
}

pub fn reproject_vector(
    input: VectorReprojectInput,
    dst_crs: CrsSpec,
    src_crs: Option<CrsSpec>,
) -> GisResult<VectorReprojectResult> {
    let (features, info, geojson_crs) = match input {
        VectorReprojectInput::Path(path) => {
            let result = read_vector(
                &path,
                VectorReadOptions {
                    layer: None,
                    columns: None,
                    bbox: None,
                    limit: None,
                },
            )?;
            let geojson_crs = crs_spec_from_vector_info(&result.info)?;
            (result.features, Some(result.info), geojson_crs)
        }
        VectorReprojectInput::Features {
            features,
            info,
            geojson_crs,
        } => {
            let features = features
                .iter()
                .map(normalize_reproject_feature)
                .collect::<GisResult<Vec<_>>>()?;
            (features, info, geojson_crs)
        }
    };
    let metadata_crs = vector_metadata_crs(info.as_ref(), geojson_crs)?;
    let src_crs = resolve_vector_source_crs(src_crs, metadata_crs)?;

    let mut warnings = Vec::new();
    if features.is_empty() {
        warnings.push(RasterWarning::new(
            WARNING_EMPTY_FEATURE_SET,
            "vector source has zero features",
            Some("features"),
        ));
    }

    let mut out_features = Vec::with_capacity(features.len());
    for feature in &features {
        out_features.push(reproject_feature(
            feature,
            &src_crs,
            &dst_crs,
            &mut warnings,
        )?);
    }

    let bounds = bounds_for_features(&out_features).map(RasterBounds::tuple);
    let feature_count = out_features.len() as u64;
    let out_info =
        build_reprojected_vector_info(info.as_ref(), &out_features, &dst_crs, warnings.clone());

    Ok(VectorReprojectResult {
        features: out_features,
        info: out_info,
        src_crs,
        dst_crs,
        bounds,
        feature_count,
        warnings,
    })
}

pub fn dissolve_vector(source: &Value, by: Option<Vec<String>>) -> GisResult<Value> {
    let _ = (source, by);
    crate::gis::geometry::topology::require_topology_backend("dissolve_vector")?;
    unreachable!("topology backend is not wired in C4.0")
}

pub fn clip_vector(
    source: &Value,
    clip_geometry: &Value,
    clip_crs: Option<Value>,
) -> GisResult<Value> {
    let _ = (source, clip_geometry, clip_crs);
    crate::gis::geometry::topology::require_topology_backend("clip_vector")?;
    unreachable!("topology backend is not wired in C4.0")
}

pub fn intersect_vectors(left: &Value, right: &Value, suffixes: (&str, &str)) -> GisResult<Value> {
    let _ = (left, right, suffixes);
    crate::gis::geometry::topology::require_topology_backend("intersect_vectors")?;
    unreachable!("topology backend is not wired in C4.0")
}

pub fn load_boundary(
    path: impl AsRef<Path>,
    layer: Option<String>,
    where_clause: Option<String>,
) -> GisResult<Value> {
    let _ = (path.as_ref(), layer);
    if where_clause.is_some() {
        return Err(GisError::InvalidArgument(
            "unsupported_option: load_boundary where filtering is not implemented".to_string(),
        ));
    }
    crate::gis::geometry::topology::require_topology_backend("load_boundary")?;
    unreachable!("topology backend is not wired in C4.0")
}

#[cfg(feature = "extension-module")]
fn normalize_reproject_features(root: &Value) -> GisResult<Vec<Value>> {
    match root.get("type").and_then(Value::as_str) {
        Some("FeatureCollection") => normalize_feature_array(root),
        None if root.get("features").is_some() => normalize_feature_array(root),
        _ => normalize_features(root),
    }
}

#[cfg(feature = "extension-module")]
fn normalize_feature_array(root: &Value) -> GisResult<Vec<Value>> {
    let features = root
        .get("features")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            GisError::InvalidGeometry(
                "invalid_geometry: FeatureCollection requires a features array".to_string(),
            )
        })?;
    features.iter().map(normalize_reproject_feature).collect()
}

fn normalize_reproject_feature(value: &Value) -> GisResult<Value> {
    normalize_feature(value).map_err(invalid_argument_to_invalid_geometry)
}

fn invalid_argument_to_invalid_geometry(err: GisError) -> GisError {
    match err {
        GisError::InvalidArgument(message) if message.contains(WARNING_INVALID_GEOMETRY) => {
            GisError::InvalidGeometry(message)
        }
        other => other,
    }
}

fn vector_metadata_crs(
    info: Option<&VectorInfo>,
    geojson_crs: Option<CrsSpec>,
) -> GisResult<Option<CrsSpec>> {
    let info_crs = info.map(crs_spec_from_vector_info).transpose()?.flatten();
    match (info_crs, geojson_crs) {
        (Some(info_crs), Some(geojson_crs)) => {
            if crs::crs_equal(&info_crs, &geojson_crs) {
                Ok(Some(info_crs))
            } else {
                Err(GisError::CrsMismatch(format!(
                    "CrsMismatch: input info CRS {} conflicts with GeoJSON CRS {}",
                    crs::canonical_label(&info_crs)?,
                    crs::canonical_label(&geojson_crs)?
                )))
            }
        }
        (Some(info_crs), None) => Ok(Some(info_crs)),
        (None, Some(geojson_crs)) => Ok(Some(geojson_crs)),
        (None, None) => Ok(None),
    }
}

fn resolve_vector_source_crs(
    explicit: Option<CrsSpec>,
    metadata: Option<CrsSpec>,
) -> GisResult<CrsSpec> {
    match (explicit, metadata) {
        (Some(explicit), Some(metadata)) => {
            if crs::crs_equal(&explicit, &metadata) {
                Ok(explicit)
            } else {
                Err(GisError::CrsMismatch(format!(
                    "CrsMismatch: explicit src_crs {} conflicts with input CRS metadata {}",
                    crs::canonical_label(&explicit)?,
                    crs::canonical_label(&metadata)?
                )))
            }
        }
        (Some(explicit), None) => Ok(explicit),
        (None, Some(metadata)) => Ok(metadata),
        (None, None) => Err(GisError::MissingCrs(
            "missing_crs: vector source has no CRS metadata; provide src_crs".to_string(),
        )),
    }
}

fn crs_spec_from_vector_info(info: &VectorInfo) -> GisResult<Option<CrsSpec>> {
    if let Some(authority) = info.crs_authority.as_ref() {
        let name = authority.get("name").cloned().ok_or_else(|| {
            GisError::InvalidCrs("invalid_crs: vector CRS authority is missing name".to_string())
        })?;
        let code = authority.get("code").cloned().ok_or_else(|| {
            GisError::InvalidCrs("invalid_crs: vector CRS authority is missing code".to_string())
        })?;
        return CrsSpec::from_parts(Some(name), Some(code), None).map(Some);
    }
    info.crs_wkt
        .clone()
        .map(|wkt| CrsSpec::from_parts(None, None, Some(wkt)))
        .transpose()
}

fn reproject_feature(
    feature: &Value,
    src: &CrsSpec,
    dst: &CrsSpec,
    warnings: &mut Vec<RasterWarning>,
) -> GisResult<Value> {
    let object = feature.as_object().ok_or_else(|| {
        GisError::InvalidGeometry("invalid_geometry: feature must be an object".to_string())
    })?;
    if object.get("type").and_then(Value::as_str) != Some("Feature") {
        return Err(GisError::InvalidGeometry(
            "invalid_geometry: features must be Feature objects".to_string(),
        ));
    }

    let mut out = object.clone();
    let geometry = object.get("geometry").unwrap_or(&Value::Null);
    out.insert(
        "geometry".to_string(),
        reproject_geometry(geometry, src, dst, warnings)?,
    );
    let transformed = Value::Object(out.clone());
    update_bbox_member(&mut out, feature_bounds(&transformed))?;
    Ok(Value::Object(out))
}

fn reproject_geometry(
    geometry: &Value,
    src: &CrsSpec,
    dst: &CrsSpec,
    warnings: &mut Vec<RasterWarning>,
) -> GisResult<Value> {
    if geometry.is_null() {
        warning_once(
            warnings,
            WARNING_EMPTY_GEOMETRY,
            "vector feature has null or empty geometry",
            Some("geometry"),
        );
        return Ok(Value::Null);
    }
    let object = geometry.as_object().ok_or_else(|| {
        GisError::InvalidGeometry("invalid_geometry: geometry must be an object".to_string())
    })?;
    let kind = object.get("type").and_then(Value::as_str).ok_or_else(|| {
        GisError::InvalidGeometry("invalid_geometry: geometry requires a type string".to_string())
    })?;

    if kind == "GeometryCollection" {
        let geometries = object
            .get("geometries")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                GisError::InvalidGeometry(
                    "invalid_geometry: GeometryCollection requires a geometries array".to_string(),
                )
            })?;
        if geometries.is_empty() {
            warning_once(
                warnings,
                WARNING_EMPTY_GEOMETRY,
                "vector feature has null or empty geometry",
                Some("geometry"),
            );
        }
        let mut out = object.clone();
        let transformed = geometries
            .iter()
            .map(|geometry| reproject_geometry(geometry, src, dst, warnings))
            .collect::<GisResult<Vec<_>>>()?;
        out.insert("geometries".to_string(), Value::Array(transformed));
        let transformed_geometry = Value::Object(out.clone());
        update_bbox_member(&mut out, geometry_bounds(&transformed_geometry))?;
        return Ok(Value::Object(out));
    }

    if !matches!(
        kind,
        "Point" | "LineString" | "Polygon" | "MultiPoint" | "MultiLineString" | "MultiPolygon"
    ) {
        return Err(GisError::InvalidGeometry(format!(
            "invalid_geometry: unsupported geometry type {kind:?}"
        )));
    }

    let coordinates = object.get("coordinates").ok_or_else(|| {
        GisError::InvalidGeometry("invalid_geometry: geometry requires coordinates".to_string())
    })?;
    let mut out = object.clone();
    if coordinates_are_empty(coordinates) {
        warning_once(
            warnings,
            WARNING_EMPTY_GEOMETRY,
            "vector feature has null or empty geometry",
            Some("geometry"),
        );
        out.insert("coordinates".to_string(), coordinates.clone());
    } else {
        out.insert(
            "coordinates".to_string(),
            reproject_coordinates_for_geometry_type(kind, coordinates, src, dst)?,
        );
    }
    let transformed_geometry = Value::Object(out.clone());
    update_bbox_member(&mut out, geometry_bounds(&transformed_geometry))?;
    Ok(Value::Object(out))
}

fn reproject_coordinates_for_geometry_type(
    kind: &str,
    coordinates: &Value,
    src: &CrsSpec,
    dst: &CrsSpec,
) -> GisResult<Value> {
    match kind {
        "Point" => reproject_position(coordinates, src, dst),
        "LineString" | "MultiPoint" => reproject_position_nested(coordinates, 1, src, dst),
        "Polygon" | "MultiLineString" => reproject_position_nested(coordinates, 2, src, dst),
        "MultiPolygon" => reproject_position_nested(coordinates, 3, src, dst),
        _ => Err(GisError::InvalidGeometry(format!(
            "invalid_geometry: unsupported geometry type {kind:?}"
        ))),
    }
}

fn reproject_position_nested(
    value: &Value,
    depth: usize,
    src: &CrsSpec,
    dst: &CrsSpec,
) -> GisResult<Value> {
    if depth == 0 {
        return reproject_position(value, src, dst);
    }
    let items = value.as_array().ok_or_else(|| {
        GisError::InvalidGeometry(
            "invalid_geometry: coordinate nesting does not match geometry type".to_string(),
        )
    })?;
    if items.is_empty() {
        return Err(GisError::InvalidGeometry(
            "invalid_geometry: nested coordinate arrays must not be empty".to_string(),
        ));
    }
    items
        .iter()
        .map(|item| reproject_position_nested(item, depth - 1, src, dst))
        .collect::<GisResult<Vec<_>>>()
        .map(Value::Array)
}

fn reproject_position(position: &Value, src: &CrsSpec, dst: &CrsSpec) -> GisResult<Value> {
    let items = position.as_array().ok_or_else(|| {
        GisError::InvalidGeometry("invalid_geometry: position must be an array".to_string())
    })?;
    if items.len() < 2 {
        return Err(GisError::InvalidGeometry(
            "invalid_geometry: position requires at least x and y".to_string(),
        ));
    }
    let x = finite_coordinate(&items[0], "x")?;
    let y = finite_coordinate(&items[1], "y")?;
    let (x, y) = crs::transform_point(x, y, src, dst)?;
    let mut out = items.clone();
    out[0] = finite_json_number(x)?;
    out[1] = finite_json_number(y)?;
    Ok(Value::Array(out))
}

fn finite_coordinate(value: &Value, axis: &str) -> GisResult<f64> {
    let number = value.as_f64().ok_or_else(|| {
        GisError::InvalidGeometry(format!(
            "invalid_geometry: coordinate {axis} must be numeric"
        ))
    })?;
    if !number.is_finite() {
        return Err(GisError::InvalidGeometry(format!(
            "invalid_geometry: coordinate {axis} must be finite"
        )));
    }
    Ok(number)
}

fn finite_json_number(value: f64) -> GisResult<Value> {
    Number::from_f64(value).map(Value::Number).ok_or_else(|| {
        GisError::InvalidGeometry(
            "invalid_geometry: transformed coordinate is not finite".to_string(),
        )
    })
}

fn update_bbox_member(
    object: &mut Map<String, Value>,
    bounds: Option<RasterBounds>,
) -> GisResult<()> {
    if !object.contains_key("bbox") {
        return Ok(());
    }
    match bounds {
        Some(bounds) => {
            object.insert("bbox".to_string(), bbox_value(bounds)?);
        }
        None => {
            object.remove("bbox");
        }
    }
    Ok(())
}

fn bbox_value(bounds: RasterBounds) -> GisResult<Value> {
    Ok(Value::Array(vec![
        finite_json_number(bounds.left)?,
        finite_json_number(bounds.bottom)?,
        finite_json_number(bounds.right)?,
        finite_json_number(bounds.top)?,
    ]))
}

fn build_reprojected_vector_info(
    base: Option<&VectorInfo>,
    features: &[Value],
    dst: &CrsSpec,
    warnings: Vec<RasterWarning>,
) -> VectorInfo {
    let bounds = bounds_for_features(features).map(RasterBounds::tuple);
    let crs_wkt = dst.wkt.clone();
    let crs_authority = crs::authority_map(dst);
    VectorInfo {
        path: base.map(|info| info.path.clone()).unwrap_or_default(),
        driver: base
            .map(|info| info.driver.clone())
            .unwrap_or_else(|| "GeoJSON".to_string()),
        layer_name: base.and_then(|info| info.layer_name.clone()),
        layer_count: base.map(|info| info.layer_count).unwrap_or(1),
        geometry_type: geometry_type_for_features(features),
        feature_count: features.len() as u64,
        schema: infer_schema(features),
        crs_wkt,
        crs_authority,
        bounds,
        is_georeferenced: bounds.is_some(),
        warnings,
    }
}

fn warning_once(
    warnings: &mut Vec<RasterWarning>,
    code: &'static str,
    message: &str,
    field: Option<&'static str>,
) {
    if warnings.iter().any(|warning| warning.code == code) {
        return;
    }
    warnings.push(RasterWarning::new(code, message, field));
}

fn validate_vector_path(path: &Path) -> GisResult<()> {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("geojson" | "json") => Ok(()),
        Some("gpkg" | "shp" | "fgb" | "flatgeobuf") => Err(GisError::BackendUnavailable(
            "backend_unavailable: gdal-vector feature required for this vector driver".to_string(),
        )),
        _ => Err(GisError::UnsupportedDriver(format!(
            "unsupported_driver: C1 vector IO supports local GeoJSON only: {}",
            path.display()
        ))),
    }
}

fn geojson_layer_name(path: &Path, root: &Value) -> Option<String> {
    root.get("name")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| path.file_stem()?.to_str().map(str::to_string))
}

fn geojson_crs(root: &Value) -> GisResult<Option<CrsSpec>> {
    let Some(crs_value) = root.get("crs") else {
        return Ok(None);
    };
    let Some(properties) = crs_value.get("properties").and_then(Value::as_object) else {
        return Ok(None);
    };
    let Some(name) = properties.get("name").and_then(Value::as_str) else {
        return Ok(None);
    };
    if let Some(spec) = crs_name_to_spec(name)? {
        return Ok(Some(spec));
    }
    Ok(None)
}

fn crs_name_to_spec(name: &str) -> GisResult<Option<CrsSpec>> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    if trimmed.to_ascii_uppercase().starts_with("EPSG:") {
        return parse_vector_crs_spec(trimmed);
    }
    for marker in ["EPSG::", "EPSG/0/", "EPSG/"] {
        if let Some((_, code)) = trimmed.rsplit_once(marker) {
            if code.chars().all(|ch| ch.is_ascii_digit()) {
                return parse_vector_crs_spec(&format!("EPSG:{code}"));
            }
        }
    }
    Err(GisError::InvalidCrs(format!(
        "invalid_crs: unsupported GeoJSON CRS name {trimmed:?}"
    )))
}

fn parse_vector_crs_spec(value: &str) -> GisResult<Option<CrsSpec>> {
    CrsSpec::from_string(value.to_string())
        .map(Some)
        .map_err(|err| GisError::InvalidCrs(format!("invalid_crs: {}", err.message())))
}

fn normalize_features(root: &Value) -> GisResult<Vec<Value>> {
    match root.get("type").and_then(Value::as_str) {
        Some("FeatureCollection") => {
            let features = root
                .get("features")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    GisError::InvalidArgument(
                        "invalid_geometry: FeatureCollection requires a features array".to_string(),
                    )
                })?;
            features.iter().map(normalize_feature).collect()
        }
        Some("Feature") => normalize_feature(root).map(|feature| vec![feature]),
        Some(_) => geometry_to_feature(root).map(|feature| vec![feature]),
        None => Err(GisError::InvalidArgument(
            "invalid_geometry: GeoJSON object requires a type string".to_string(),
        )),
    }
}

fn normalize_feature(value: &Value) -> GisResult<Value> {
    let object = value.as_object().ok_or_else(|| {
        GisError::InvalidArgument("invalid_geometry: feature must be an object".to_string())
    })?;
    if object.get("type").and_then(Value::as_str) != Some("Feature") {
        return Err(GisError::InvalidArgument(
            "invalid_geometry: features array entries must be Feature objects".to_string(),
        ));
    }
    let mut out = object.clone();
    if !out.get("properties").is_some_and(Value::is_object) {
        out.insert("properties".to_string(), Value::Object(Map::new()));
    }
    if !out.contains_key("geometry") {
        out.insert("geometry".to_string(), Value::Null);
    }
    Ok(Value::Object(out))
}

fn geometry_to_feature(value: &Value) -> GisResult<Value> {
    if !value.is_object() {
        return Err(GisError::InvalidArgument(
            "invalid_geometry: geometry must be an object".to_string(),
        ));
    }
    let mut feature = Map::new();
    feature.insert("type".to_string(), Value::String("Feature".to_string()));
    feature.insert("geometry".to_string(), value.clone());
    feature.insert("properties".to_string(), Value::Object(Map::new()));
    Ok(Value::Object(feature))
}

fn filter_columns(features: Vec<Value>, columns: &[String]) -> Vec<Value> {
    let wanted = columns.iter().collect::<BTreeSet<_>>();
    features
        .into_iter()
        .map(|feature| {
            let mut object = match feature.as_object() {
                Some(object) => object.clone(),
                None => return feature,
            };
            let properties = object
                .get("properties")
                .and_then(Value::as_object)
                .map(|properties| {
                    properties
                        .iter()
                        .filter(|(key, _)| wanted.contains(key))
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect::<Map<String, Value>>()
                })
                .unwrap_or_default();
            object.insert("properties".to_string(), Value::Object(properties));
            Value::Object(object)
        })
        .collect()
}

fn infer_schema(features: &[Value]) -> Vec<SchemaField> {
    #[derive(Default)]
    struct State {
        present: usize,
        nullable: bool,
        types: BTreeSet<&'static str>,
    }

    let mut states = BTreeMap::<String, State>::new();
    for feature in features {
        let Some(properties) = feature.get("properties").and_then(Value::as_object) else {
            continue;
        };
        for (name, value) in properties {
            let state = states.entry(name.clone()).or_default();
            state.present += 1;
            if value.is_null() {
                state.nullable = true;
            } else {
                state.types.insert(json_type_name(value));
            }
        }
    }

    states
        .into_iter()
        .map(|(name, state)| SchemaField {
            name,
            field_type: public_field_type(&state.types).to_string(),
            nullable: Some(state.nullable || state.present < features.len()),
            width: None,
            precision: None,
        })
        .collect()
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Bool(_) => "boolean",
        Value::Number(number) if number.is_i64() || number.is_u64() => "integer",
        Value::Number(_) => "float",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
        Value::Null => "null",
    }
}

fn public_field_type(types: &BTreeSet<&'static str>) -> &'static str {
    if types.is_empty() {
        "null"
    } else if types.len() == 1 {
        types.iter().next().copied().unwrap_or("unknown")
    } else if types.len() == 2 && types.contains("integer") && types.contains("float") {
        "float"
    } else {
        "mixed"
    }
}

fn geometry_type_for_features(features: &[Value]) -> String {
    if features.is_empty() {
        return "Empty".to_string();
    }
    let types = features
        .iter()
        .map(feature_geometry_type)
        .collect::<BTreeSet<_>>();
    if types.len() == 1 {
        types
            .into_iter()
            .next()
            .unwrap_or_else(|| "Unknown".to_string())
    } else {
        "Mixed".to_string()
    }
}

fn feature_geometry_type(feature: &Value) -> String {
    feature
        .get("geometry")
        .map(geometry_type_from_value)
        .unwrap_or_else(|| "Empty".to_string())
}

fn geometry_type_from_value(value: &Value) -> String {
    if value.is_null() {
        return "Empty".to_string();
    }
    let Some(kind) = value.get("type").and_then(Value::as_str) else {
        return "Unknown".to_string();
    };
    match kind {
        "Point" | "LineString" | "Polygon" | "MultiPoint" | "MultiLineString" | "MultiPolygon" => {
            if value.get("coordinates").is_none_or(coordinates_are_empty) {
                "Empty".to_string()
            } else {
                kind.to_string()
            }
        }
        "GeometryCollection" => {
            if value
                .get("geometries")
                .and_then(Value::as_array)
                .is_none_or(Vec::is_empty)
            {
                "Empty".to_string()
            } else {
                "GeometryCollection".to_string()
            }
        }
        _ => "Unknown".to_string(),
    }
}

fn coordinates_are_empty(value: &Value) -> bool {
    value.as_array().is_none_or(Vec::is_empty)
}

fn bounds_for_features(features: &[Value]) -> Option<RasterBounds> {
    let mut out: Option<RasterBounds> = None;
    for bounds in features.iter().filter_map(feature_bounds) {
        out = Some(match out {
            Some(current) => RasterBounds {
                left: current.left.min(bounds.left),
                bottom: current.bottom.min(bounds.bottom),
                right: current.right.max(bounds.right),
                top: current.top.max(bounds.top),
            },
            None => bounds,
        });
    }
    out
}

fn feature_bounds(feature: &Value) -> Option<RasterBounds> {
    geometry_bounds(feature.get("geometry")?)
}

fn geometry_bounds(geometry: &Value) -> Option<RasterBounds> {
    if geometry.is_null() {
        return None;
    }
    if geometry.get("type").and_then(Value::as_str) == Some("GeometryCollection") {
        let mut out: Option<RasterBounds> = None;
        for geometry in geometry.get("geometries")?.as_array()? {
            if let Some(bounds) = geometry_bounds(geometry) {
                out = Some(match out {
                    Some(current) => RasterBounds {
                        left: current.left.min(bounds.left),
                        bottom: current.bottom.min(bounds.bottom),
                        right: current.right.max(bounds.right),
                        top: current.top.max(bounds.top),
                    },
                    None => bounds,
                });
            }
        }
        return out;
    }
    let mut out = None;
    collect_coordinate_bounds(geometry.get("coordinates")?, &mut out);
    out
}

fn collect_coordinate_bounds(value: &Value, out: &mut Option<RasterBounds>) {
    let Some(items) = value.as_array() else {
        return;
    };
    if items.len() >= 2 {
        if let (Some(x), Some(y)) = (items[0].as_f64(), items[1].as_f64()) {
            if x.is_finite() && y.is_finite() {
                let point = RasterBounds {
                    left: x,
                    bottom: y,
                    right: x,
                    top: y,
                };
                *out = Some(match out {
                    Some(current) => RasterBounds {
                        left: current.left.min(point.left),
                        bottom: current.bottom.min(point.bottom),
                        right: current.right.max(point.right),
                        top: current.top.max(point.top),
                    },
                    None => point,
                });
            }
            return;
        }
    }
    for item in items {
        collect_coordinate_bounds(item, out);
    }
}

fn bounds_intersects(left: RasterBounds, right: RasterBounds) -> bool {
    left.left <= right.right
        && left.right >= right.left
        && left.bottom <= right.top
        && left.top >= right.bottom
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn epsg(code: &str) -> CrsSpec {
        CrsSpec::from_string(format!("EPSG:{code}")).expect("valid test CRS")
    }

    fn coordinate(value: &Value, index: usize) -> f64 {
        value
            .as_array()
            .and_then(|items| items.get(index))
            .and_then(Value::as_f64)
            .expect("numeric coordinate")
    }

    #[test]
    fn reproject_position_preserves_extra_ordinates() {
        let src = epsg("4326");
        let dst = epsg("3857");

        let out = reproject_position(&json!([1.0, 1.0, 42.0, "kept"]), &src, &dst)
            .expect("position reprojects");

        assert!((coordinate(&out, 0) - 111_319.490_793_273_57).abs() < 1e-6);
        assert!((coordinate(&out, 1) - 111_325.142_866_385_1).abs() < 1e-6);
        assert_eq!(out[2], json!(42.0));
        assert_eq!(out[3], json!("kept"));
    }

    #[test]
    fn resolve_vector_source_crs_errors_when_missing() {
        let err = resolve_vector_source_crs(None, None).expect_err("missing CRS must fail");

        assert!(matches!(err, GisError::MissingCrs(_)));
        assert!(err.to_string().contains("missing_crs"));
    }

    #[test]
    fn resolve_vector_source_crs_rejects_conflict() {
        let err = resolve_vector_source_crs(Some(epsg("3857")), Some(epsg("4326")))
            .expect_err("conflicting source CRS must fail");

        assert!(matches!(err, GisError::CrsMismatch(_)));
    }

    #[test]
    fn reproject_geometry_collection_recurses() {
        let src = epsg("4326");
        let dst = epsg("3857");
        let geometry = json!({
            "type": "GeometryCollection",
            "geometries": [
                {"type": "Point", "coordinates": [1.0, 1.0]},
                {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0]]}
            ]
        });
        let mut warnings = Vec::new();

        let out = reproject_geometry(&geometry, &src, &dst, &mut warnings)
            .expect("collection reprojects");

        assert!(
            (out["geometries"][0]["coordinates"][0].as_f64().unwrap() - 111_319.490_793_273_57)
                .abs()
                < 1e-6
        );
        assert!(
            (out["geometries"][1]["coordinates"][1][1].as_f64().unwrap() - 111_325.142_866_385_1)
                .abs()
                < 1e-6
        );
        assert!(warnings.is_empty());
    }

    #[test]
    fn same_crs_reprojection_is_copy_with_destination_metadata() {
        let crs = epsg("4326");
        let feature = json!({
            "type": "Feature",
            "id": "point-a",
            "properties": {"name": "alpha"},
            "geometry": {"type": "Point", "coordinates": [1.0, 2.0]}
        });

        let result = reproject_vector(
            VectorReprojectInput::Features {
                features: vec![feature],
                info: None,
                geojson_crs: Some(crs.clone()),
            },
            crs,
            None,
        )
        .expect("same-CRS reprojection succeeds");

        assert_eq!(
            result.features[0]["geometry"]["coordinates"],
            json!([1.0, 2.0])
        );
        assert_eq!(result.info.crs_authority, crs::authority_map(&epsg("4326")));
        assert_eq!(result.feature_count, 1);
    }
}

#[cfg(feature = "extension-module")]
mod py {
    use super::*;
    use crate::gis::py_json::{json_to_py, py_to_json, py_to_json_strict, warnings_to_py};
    use pyo3::prelude::*;
    use pyo3::types::{PyAny, PyDict, PyDictMethods, PyList, PyTuple};
    use pyo3::IntoPy;

    #[pyo3::pymethods]
    impl VectorInfo {
        #[getter]
        fn path(&self) -> String {
            self.path.clone()
        }

        #[getter]
        fn driver(&self) -> String {
            self.driver.clone()
        }

        #[getter]
        fn layer_name(&self) -> Option<String> {
            self.layer_name.clone()
        }

        #[getter]
        fn layer_count(&self) -> u32 {
            self.layer_count
        }

        #[getter]
        fn geometry_type(&self) -> String {
            self.geometry_type.clone()
        }

        #[getter]
        fn feature_count(&self) -> u64 {
            self.feature_count
        }

        #[getter]
        fn schema<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
            schema_to_py(py, &self.schema)
        }

        #[getter]
        fn crs_wkt(&self) -> Option<String> {
            self.crs_wkt.clone()
        }

        #[getter]
        fn crs_authority(&self) -> Option<HashMap<String, String>> {
            self.crs_authority.clone()
        }

        #[getter]
        fn bounds(&self) -> Option<(f64, f64, f64, f64)> {
            self.bounds
        }

        #[getter]
        fn is_georeferenced(&self) -> bool {
            self.is_georeferenced
        }

        #[getter]
        fn warnings<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
            warnings_to_py(py, &self.warnings)
        }

        fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
            vector_info_to_py_dict(py, self)
        }

        fn __repr__(&self) -> String {
            format!(
                "VectorInfo(path={:?}, driver={:?}, layer_name={:?}, feature_count={})",
                self.path, self.driver, self.layer_name, self.feature_count
            )
        }
    }

    #[pyfunction(
        name = "read_vector",
        signature = (path, *, layer = None, columns = None, bbox = None, limit = None)
    )]
    pub fn read_vector_py(
        py: Python<'_>,
        path: String,
        layer: Option<String>,
        columns: Option<Vec<String>>,
        bbox: Option<(f64, f64, f64, f64)>,
        limit: Option<usize>,
    ) -> PyResult<PyObject> {
        let result = read_vector(
            path,
            VectorReadOptions {
                layer,
                columns,
                bbox,
                limit,
            },
        )?;
        vector_read_result_to_py(py, &result)
    }

    #[pyfunction(name = "geometry_type", signature = (source, *, layer = None))]
    pub fn geometry_type_py(source: &Bound<'_, PyAny>, layer: Option<String>) -> PyResult<String> {
        if let Some(info) = vector_info_from_source(source, layer.clone())? {
            return Ok(info.geometry_type);
        }
        let value = py_to_json(source)?;
        Ok(geometry_type_for_source_json(&value))
    }

    #[pyfunction(name = "vector_schema", signature = (source, *, layer = None))]
    pub fn vector_schema_py(
        py: Python<'_>,
        source: &Bound<'_, PyAny>,
        layer: Option<String>,
    ) -> PyResult<PyObject> {
        if let Some(info) = vector_info_from_source(source, layer.clone())? {
            return schema_to_py(py, &info.schema);
        }
        let value = py_to_json(source)?;
        schema_to_py(py, &schema_for_source_json(&value))
    }

    #[pyfunction(name = "feature_count", signature = (source, *, layer = None))]
    pub fn feature_count_py(source: &Bound<'_, PyAny>, layer: Option<String>) -> PyResult<u64> {
        if let Some(info) = vector_info_from_source(source, layer.clone())? {
            return Ok(info.feature_count);
        }
        let value = py_to_json(source)?;
        Ok(feature_count_for_source_json(&value))
    }

    #[pyfunction(name = "vector_crs", signature = (source, *, layer = None))]
    pub fn vector_crs_py(
        py: Python<'_>,
        source: &Bound<'_, PyAny>,
        layer: Option<String>,
    ) -> PyResult<PyObject> {
        let info = vector_info_from_source(source, layer)?.unwrap_or_else(|| VectorInfo {
            path: String::new(),
            driver: "GeoJSON".to_string(),
            layer_name: None,
            layer_count: 1,
            geometry_type: "Unknown".to_string(),
            feature_count: 0,
            schema: Vec::new(),
            crs_wkt: None,
            crs_authority: None,
            bounds: None,
            is_georeferenced: false,
            warnings: vec![RasterWarning::new(
                WARNING_MISSING_CRS,
                "vector source has no CRS metadata",
                Some("crs_wkt"),
            )],
        });
        vector_crs_to_py(py, &info)
    }

    #[pyfunction(name = "vector_bounds", signature = (source, *, layer = None))]
    pub fn vector_bounds_py(
        source: &Bound<'_, PyAny>,
        layer: Option<String>,
    ) -> PyResult<(f64, f64, f64, f64)> {
        if let Some(info) = vector_info_from_source(source, layer.clone())? {
            return info.bounds.ok_or_else(|| {
                GisError::InvalidArgument(
                    "empty_feature_set: vector source has no feature bounds".to_string(),
                )
                .into()
            });
        }
        let value = py_to_json(source)?;
        bounds_for_source_json(&value)
            .map(RasterBounds::tuple)
            .ok_or_else(|| {
                GisError::InvalidArgument(
                    "empty_feature_set: vector source has no feature bounds".to_string(),
                )
                .into()
            })
    }

    #[pyfunction(name = "reproject_vector", signature = (input, dst_crs, src_crs = None))]
    pub fn reproject_vector_py(
        py: Python<'_>,
        input: &Bound<'_, PyAny>,
        dst_crs: &Bound<'_, PyAny>,
        src_crs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let dst_crs = crate::gis::extract_required_crs(Some(dst_crs))?;
        let src_crs = crate::gis::extract_crs(src_crs)?;
        let input = reproject_input_from_py(input)?;
        let result = reproject_vector(input, dst_crs, src_crs)?;
        vector_reproject_result_to_py(py, &result)
    }

    #[pyfunction(name = "dissolve_vector", signature = (source, *, by = None))]
    pub fn dissolve_vector_py(
        py: Python<'_>,
        source: &Bound<'_, PyAny>,
        by: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let source = py_to_json_strict(source)?;
        let by = by_from_py(by)?;
        let result = dissolve_vector(&source, by)?;
        json_to_py(py, &result)
    }

    #[pyfunction(name = "clip_vector", signature = (source, clip_geometry, *, clip_crs = None))]
    pub fn clip_vector_py(
        py: Python<'_>,
        source: &Bound<'_, PyAny>,
        clip_geometry: &Bound<'_, PyAny>,
        clip_crs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let source = py_to_json_strict(source)?;
        let clip_geometry = py_to_json_strict(clip_geometry)?;
        let clip_crs = clip_crs.map(py_to_json_strict).transpose()?;
        let result = clip_vector(&source, &clip_geometry, clip_crs)?;
        json_to_py(py, &result)
    }

    #[pyfunction(name = "intersect_vectors", signature = (left, right, *, suffixes = None))]
    pub fn intersect_vectors_py(
        py: Python<'_>,
        left: &Bound<'_, PyAny>,
        right: &Bound<'_, PyAny>,
        suffixes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let left = py_to_json_strict(left)?;
        let right = py_to_json_strict(right)?;
        let suffixes = suffixes_from_py(suffixes)?;
        let result = intersect_vectors(&left, &right, (&suffixes.0, &suffixes.1))?;
        json_to_py(py, &result)
    }

    #[pyfunction(name = "load_boundary", signature = (path, *, layer = None, r#where = None))]
    pub fn load_boundary_py(
        py: Python<'_>,
        path: String,
        layer: Option<String>,
        r#where: Option<String>,
    ) -> PyResult<PyObject> {
        let result = load_boundary(path, layer, r#where)?;
        json_to_py(py, &result)
    }

    fn vector_read_result_to_py(py: Python<'_>, result: &VectorReadResult) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("type", "FeatureCollection")?;
        dict.set_item(
            "features",
            json_to_py(py, &Value::Array(result.features.clone()))?,
        )?;
        dict.set_item("info", vector_info_to_py_dict(py, &result.info)?)?;
        dict.set_item("vector_info", Py::new(py, result.info.clone())?)?;
        dict.set_item("warnings", warnings_to_py(py, &result.warnings)?)?;
        Ok(dict.into_py(py))
    }

    fn vector_reproject_result_to_py(
        py: Python<'_>,
        result: &VectorReprojectResult,
    ) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("type", "FeatureCollection")?;
        dict.set_item(
            "features",
            json_to_py(py, &Value::Array(result.features.clone()))?,
        )?;
        dict.set_item("info", vector_info_to_py_dict(py, &result.info)?)?;
        dict.set_item("vector_info", Py::new(py, result.info.clone())?)?;
        dict.set_item("src_crs", crs_spec_to_py(py, &result.src_crs, "crs")?)?;
        dict.set_item("dst_crs", crs_spec_to_py(py, &result.dst_crs, "crs")?)?;
        dict.set_item("bounds", result.bounds)?;
        dict.set_item("feature_count", result.feature_count)?;
        dict.set_item("warnings", warnings_to_py(py, &result.warnings)?)?;
        Ok(dict.into_py(py))
    }

    fn by_from_py(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<String>>> {
        let Some(value) = value else {
            return Ok(None);
        };
        if value.is_none() {
            return Ok(None);
        }
        if let Ok(field) = value.extract::<String>() {
            return Ok(Some(vec![field]));
        }
        let items = if let Ok(list) = value.downcast::<PyList>() {
            strings_from_iter(list.iter(), "by")?
        } else if let Ok(tuple) = value.downcast::<PyTuple>() {
            strings_from_iter(tuple.iter(), "by")?
        } else {
            return Err(GisError::InvalidArgument(
                "invalid_argument: by must be None, a string, or a sequence of strings".to_string(),
            )
            .into());
        };
        if items.is_empty() {
            return Err(GisError::InvalidArgument(
                "invalid_argument: by sequence must not be empty".to_string(),
            )
            .into());
        }
        Ok(Some(items))
    }

    fn suffixes_from_py(value: Option<&Bound<'_, PyAny>>) -> PyResult<(String, String)> {
        let Some(value) = value else {
            return Ok(("_left".to_string(), "_right".to_string()));
        };
        if value.is_none() {
            return Ok(("_left".to_string(), "_right".to_string()));
        }
        let items = if let Ok(list) = value.downcast::<PyList>() {
            strings_from_iter(list.iter(), "suffixes")
        } else if let Ok(tuple) = value.downcast::<PyTuple>() {
            strings_from_iter(tuple.iter(), "suffixes")
        } else {
            Err(GisError::InvalidArgument(
                "invalid_argument: suffixes must be a pair of strings".to_string(),
            )
            .into())
        }?;
        if items.len() != 2 {
            return Err(GisError::InvalidArgument(
                "invalid_argument: suffixes must contain exactly two strings".to_string(),
            )
            .into());
        }
        Ok((items[0].clone(), items[1].clone()))
    }

    fn strings_from_iter<'py>(
        items: impl Iterator<Item = Bound<'py, PyAny>>,
        field: &str,
    ) -> PyResult<Vec<String>> {
        let mut out = Vec::new();
        for item in items {
            out.push(item.extract::<String>().map_err(|_| {
                GisError::InvalidArgument(format!(
                    "invalid_argument: {field} must contain only strings"
                ))
            })?);
        }
        Ok(out)
    }

    fn crs_spec_to_py(py: Python<'_>, spec: &CrsSpec, source_kind: &str) -> PyResult<PyObject> {
        let inspection = crs::inspect_crs_spec(spec, source_kind);
        let dict = PyDict::new_bound(py);
        dict.set_item("source_kind", inspection.source_kind)?;
        dict.set_item("missing", inspection.missing)?;
        dict.set_item("wkt", inspection.wkt)?;
        dict.set_item("authority", inspection.authority)?;
        dict.set_item("axis_order", inspection.axis_order)?;
        dict.set_item("axis_order_policy", inspection.axis_order_policy)?;
        dict.set_item("warnings", warnings_to_py(py, &inspection.warnings)?)?;
        Ok(dict.into_py(py))
    }

    fn vector_info_to_py_dict(py: Python<'_>, info: &VectorInfo) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("path", info.path.clone())?;
        dict.set_item("driver", info.driver.clone())?;
        dict.set_item("layer_name", info.layer_name.clone())?;
        dict.set_item("layer_count", info.layer_count)?;
        dict.set_item("geometry_type", info.geometry_type.clone())?;
        dict.set_item("feature_count", info.feature_count)?;
        dict.set_item("schema", schema_to_py(py, &info.schema)?)?;
        dict.set_item("crs_wkt", info.crs_wkt.clone())?;
        dict.set_item("crs_authority", info.crs_authority.clone())?;
        dict.set_item("bounds", info.bounds)?;
        dict.set_item("is_georeferenced", info.is_georeferenced)?;
        dict.set_item("warnings", warnings_to_py(py, &info.warnings)?)?;
        Ok(dict.into_py(py))
    }

    fn reproject_input_from_py(source: &Bound<'_, PyAny>) -> PyResult<VectorReprojectInput> {
        if source.extract::<PyRef<'_, VectorInfo>>().is_ok() {
            return Err(GisError::InvalidArgument(
                "feature payload is required; VectorInfo alone cannot be reprojected".to_string(),
            )
            .into());
        }
        if let Ok(path) = source.extract::<String>() {
            return Ok(VectorReprojectInput::Path(PathBuf::from(path)));
        }
        let dict = source.downcast::<PyDict>().map_err(|_| {
            GisError::InvalidArgument(
                "input must be a vector path, read_vector result, or GeoJSON-like dict".to_string(),
            )
        })?;
        if is_info_dict(dict)? && !dict.contains("features")? {
            return Err(GisError::InvalidArgument(
                "feature payload is required; VectorInfo metadata alone cannot be reprojected"
                    .to_string(),
            )
            .into());
        }
        let info = reproject_info_from_dict(dict)?;
        let root = reproject_source_json(source)?;
        let geojson_crs = geojson_crs(&root)?;
        let features = normalize_reproject_features(&root)?;
        Ok(VectorReprojectInput::Features {
            features,
            info,
            geojson_crs,
        })
    }

    fn reproject_info_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<Option<VectorInfo>> {
        if let Some(vector_info) = dict.get_item("vector_info")? {
            if vector_info.is_none() {
                return Ok(None);
            }
            return vector_info_from_info_value(&vector_info).map(Some);
        }
        if let Some(info) = dict.get_item("info")? {
            if info.is_none() {
                return Ok(None);
            }
            return vector_info_from_info_value(&info).map(Some);
        }
        Ok(None)
    }

    fn reproject_source_json(source: &Bound<'_, PyAny>) -> PyResult<Value> {
        if let Ok(dict) = source.downcast::<PyDict>() {
            let type_name = dict
                .get_item("type")?
                .map(|value| value.extract::<String>())
                .transpose()
                .ok()
                .flatten();
            if type_name.as_deref() == Some("FeatureCollection")
                || (type_name.is_none() && dict.contains("features")?)
            {
                let mut out = Map::new();
                if let Some(type_value) = dict.get_item("type")? {
                    out.insert("type".to_string(), py_to_json_strict(&type_value)?);
                }
                if let Some(features) = dict.get_item("features")? {
                    out.insert("features".to_string(), py_to_json_strict(&features)?);
                }
                if let Some(crs) = dict.get_item("crs")? {
                    out.insert("crs".to_string(), py_to_json_strict(&crs)?);
                }
                if let Some(bbox) = dict.get_item("bbox")? {
                    out.insert("bbox".to_string(), py_to_json_strict(&bbox)?);
                }
                return Ok(Value::Object(out));
            }
        }
        py_to_json_strict(source)
    }

    fn vector_crs_to_py(py: Python<'_>, info: &VectorInfo) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("source_kind", "vector")?;
        dict.set_item(
            "missing",
            info.crs_wkt.is_none() && info.crs_authority.is_none(),
        )?;
        dict.set_item("wkt", info.crs_wkt.clone())?;
        dict.set_item("authority", info.crs_authority.clone())?;
        dict.set_item(
            "axis_order",
            info.crs_authority.as_ref().map(|_| "xy".to_string()),
        )?;
        dict.set_item("axis_order_policy", "traditional_gis_xy")?;
        dict.set_item(
            "warnings",
            warnings_to_py(
                py,
                &info
                    .warnings
                    .iter()
                    .filter(|warning| warning.code == WARNING_MISSING_CRS)
                    .cloned()
                    .collect::<Vec<_>>(),
            )?,
        )?;
        Ok(dict.into_py(py))
    }

    fn vector_info_from_source(
        source: &Bound<'_, PyAny>,
        layer: Option<String>,
    ) -> PyResult<Option<VectorInfo>> {
        if let Ok(info) = source.extract::<PyRef<'_, VectorInfo>>() {
            return Ok(Some(info.clone()));
        }
        if let Ok(path) = source.extract::<String>() {
            return read_vector_info(path, layer).map(Some).map_err(Into::into);
        }
        let Ok(dict) = source.downcast::<PyDict>() else {
            return Ok(None);
        };
        if let Some(vector_info) = dict.get_item("vector_info")? {
            if let Ok(info) = vector_info.extract::<PyRef<'_, VectorInfo>>() {
                return Ok(Some(info.clone()));
            }
        }
        let info_value = if is_info_dict(dict)? {
            Some(source.clone())
        } else {
            dict.get_item("info")?
        };
        match info_value {
            Some(value) => vector_info_from_info_value(&value).map(Some),
            None => Ok(None),
        }
    }

    fn is_info_dict(dict: &Bound<'_, PyDict>) -> PyResult<bool> {
        Ok(dict.contains("geometry_type")? && dict.contains("feature_count")?)
    }

    fn vector_info_from_info_value(value: &Bound<'_, PyAny>) -> PyResult<VectorInfo> {
        if let Ok(info) = value.extract::<PyRef<'_, VectorInfo>>() {
            return Ok(info.clone());
        }
        let dict = value.downcast::<PyDict>().map_err(|_| {
            GisError::InvalidArgument("source info must be a VectorInfo or dict".to_string())
        })?;
        let schema = dict
            .get_item("schema")?
            .map(|value| schema_from_py(&value))
            .transpose()?
            .unwrap_or_default();
        let warnings = dict
            .get_item("warnings")?
            .map(|value| warnings_from_py(&value))
            .transpose()?
            .unwrap_or_default();
        Ok(VectorInfo {
            path: dict_string(dict, "path")?.unwrap_or_default(),
            driver: dict_string(dict, "driver")?.unwrap_or_else(|| "GeoJSON".to_string()),
            layer_name: dict_string(dict, "layer_name")?,
            layer_count: dict
                .get_item("layer_count")?
                .map(|value| value.extract::<u32>())
                .transpose()?
                .unwrap_or(1),
            geometry_type: dict_string(dict, "geometry_type")?
                .unwrap_or_else(|| "Unknown".to_string()),
            feature_count: dict
                .get_item("feature_count")?
                .map(|value| value.extract::<u64>())
                .transpose()?
                .unwrap_or(0),
            schema,
            crs_wkt: dict_string(dict, "crs_wkt")?,
            crs_authority: dict
                .get_item("crs_authority")?
                .map(|value| value.extract::<HashMap<String, String>>())
                .transpose()?,
            bounds: dict
                .get_item("bounds")?
                .map(|value| {
                    if value.is_none() {
                        Ok(None)
                    } else {
                        value.extract::<(f64, f64, f64, f64)>().map(Some)
                    }
                })
                .transpose()?
                .flatten(),
            is_georeferenced: dict
                .get_item("is_georeferenced")?
                .map(|value| value.extract::<bool>())
                .transpose()?
                .unwrap_or(false),
            warnings,
        })
    }

    fn dict_string(dict: &Bound<'_, PyDict>, key: &'static str) -> PyResult<Option<String>> {
        dict.get_item(key)?
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

    fn geometry_type_for_source_json(value: &Value) -> String {
        match value.get("type").and_then(Value::as_str) {
            Some("FeatureCollection") => value
                .get("features")
                .and_then(Value::as_array)
                .map(|features| geometry_type_for_features(features))
                .unwrap_or_else(|| "Unknown".to_string()),
            Some("Feature") => feature_geometry_type(value),
            Some(_) => geometry_type_from_value(value),
            None => "Unknown".to_string(),
        }
    }

    fn schema_for_source_json(value: &Value) -> Vec<SchemaField> {
        match value.get("type").and_then(Value::as_str) {
            Some("FeatureCollection") => value
                .get("features")
                .and_then(Value::as_array)
                .map(|features| infer_schema(features))
                .unwrap_or_default(),
            Some("Feature") => infer_schema(std::slice::from_ref(value)),
            _ => Vec::new(),
        }
    }

    fn feature_count_for_source_json(value: &Value) -> u64 {
        match value.get("type").and_then(Value::as_str) {
            Some("FeatureCollection") => value
                .get("features")
                .and_then(Value::as_array)
                .map(|features| features.len() as u64)
                .unwrap_or(0),
            Some("Feature") | Some(_) => 1,
            None => 0,
        }
    }

    fn bounds_for_source_json(value: &Value) -> Option<RasterBounds> {
        match value.get("type").and_then(Value::as_str) {
            Some("FeatureCollection") => value
                .get("features")
                .and_then(Value::as_array)
                .and_then(|features| bounds_for_features(features)),
            Some("Feature") => feature_bounds(value),
            Some(_) => geometry_bounds(value),
            None => None,
        }
    }

    fn schema_to_py(py: Python<'_>, schema: &[SchemaField]) -> PyResult<PyObject> {
        let mut items = Vec::with_capacity(schema.len());
        for field in schema {
            let dict = PyDict::new_bound(py);
            dict.set_item("name", field.name.clone())?;
            dict.set_item("type", field.field_type.clone())?;
            dict.set_item("nullable", field.nullable)?;
            dict.set_item("width", field.width)?;
            dict.set_item("precision", field.precision)?;
            items.push(dict.into_py(py));
        }
        Ok(PyList::new_bound(py, items).into_py(py))
    }

    fn schema_from_py(value: &Bound<'_, PyAny>) -> PyResult<Vec<SchemaField>> {
        let list = value.downcast::<PyList>().map_err(|_| {
            GisError::InvalidArgument("schema must be a list of field dicts".to_string())
        })?;
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            let dict = item.downcast::<PyDict>().map_err(|_| {
                GisError::InvalidArgument("schema entries must be dicts".to_string())
            })?;
            out.push(SchemaField {
                name: required_string(dict, "name")?,
                field_type: required_string(dict, "type")?,
                nullable: dict
                    .get_item("nullable")?
                    .map(|value| {
                        if value.is_none() {
                            Ok(None)
                        } else {
                            value.extract::<bool>().map(Some)
                        }
                    })
                    .transpose()?
                    .flatten(),
                width: dict
                    .get_item("width")?
                    .map(|value| {
                        if value.is_none() {
                            Ok(None)
                        } else {
                            value.extract::<u32>().map(Some)
                        }
                    })
                    .transpose()?
                    .flatten(),
                precision: dict
                    .get_item("precision")?
                    .map(|value| {
                        if value.is_none() {
                            Ok(None)
                        } else {
                            value.extract::<u32>().map(Some)
                        }
                    })
                    .transpose()?
                    .flatten(),
            });
        }
        Ok(out)
    }

    fn warnings_from_py(value: &Bound<'_, PyAny>) -> PyResult<Vec<RasterWarning>> {
        let list = value.downcast::<PyList>().map_err(|_| {
            GisError::InvalidArgument("warnings must be a list of warning dicts".to_string())
        })?;
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            let dict = item.downcast::<PyDict>().map_err(|_| {
                GisError::InvalidArgument("warning entries must be dicts".to_string())
            })?;
            out.push(RasterWarning {
                code: required_string(dict, "code")?,
                message: required_string(dict, "message")?,
                field: dict_string(dict, "field")?,
            });
        }
        Ok(out)
    }

    fn required_string(dict: &Bound<'_, PyDict>, key: &'static str) -> PyResult<String> {
        dict.get_item(key)?
            .ok_or_else(|| GisError::InvalidArgument(format!("{key} is required")))?
            .extract::<String>()
            .map_err(Into::into)
    }
}

#[cfg(feature = "extension-module")]
pub use py::{
    clip_vector_py, dissolve_vector_py, feature_count_py, geometry_type_py, intersect_vectors_py,
    load_boundary_py, read_vector_py, reproject_vector_py, vector_bounds_py, vector_crs_py,
    vector_schema_py,
};
