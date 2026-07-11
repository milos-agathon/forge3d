use serde_json::{json, Map, Value};

use crate::gis::error::{GisError, GisResult};
use crate::gis::types::RasterWarning;

mod centroid;
mod line_ops;
mod math;
mod measure;
mod model;
mod parse;
#[cfg(feature = "extension-module")]
mod py;
pub(crate) mod topology;
mod validate;

use centroid::centroid_for_geometries;
use line_ops::{
    interpolate_lines, lines_for_interpolation, normalized_target_distance,
    representative_for_geometry, total_line_length, validate_distance,
};
use math::{looks_geographic, unwrap_dateline, wrap_geometry_lons};
pub use measure::MeasureMode;
use measure::{measure_geometries, validate_metric_names};
use model::{
    finite_value, operation_value, point_value, Coord, Geometry, NormalizedInput, EMPTY_INPUT,
    EMPTY_OUTPUT, EPSILON, GEOMETRY_TYPE_CHANGED, INVALID_ARGUMENT, UNSUPPORTED_GEOMETRY_TYPE,
    UNSUPPORTED_OPTION,
};
use parse::normalize_input;

/// MENSURA dateline handling for the topology path: unwrap geographic
/// antimeridian-crossing inputs so planar topology runs on continuous
/// longitudes. Returns the (possibly unwrapped) geometries and whether the
/// operation's output longitudes must be wrapped back.
fn declared_wgs84(crs: Option<&Value>) -> Option<bool> {
    let authority = crs?.get("authority")?;
    let code = authority.get("code")?;
    Some(code.as_str().is_some_and(|code| code == "4326") || code.as_u64() == Some(4326))
}

fn geographic_for_input(input: &NormalizedInput) -> bool {
    declared_wgs84(input.crs.as_ref()).unwrap_or_else(|| looks_geographic(&input.geometries))
}

fn dateline_normalized(geometries: &[Geometry], geographic: bool) -> (Vec<Geometry>, bool) {
    let mut out = geometries.to_vec();
    let unwrapped = geographic && unwrap_dateline(&mut out);
    (out, unwrapped)
}
use topology::{
    buffer_topology, intersection_polygonal, require_topology_backend, simplify_topology,
    union_polygonal,
};
use validate::{validate_geometry_value, validate_input_or_error};

pub(crate) struct PolygonalClipMask {
    geometry: Geometry,
}

pub fn validate_geometry(source: &Value) -> GisResult<Value> {
    Ok(validate_geometry_value(source))
}

pub fn repair_geometry(source: &Value, method: &str) -> GisResult<Value> {
    if method != "make_valid" {
        return Err(GisError::InvalidArgument(format!(
            "{UNSUPPORTED_OPTION}: unsupported repair method {method:?}"
        )));
    }
    let input = normalize_input(source, false)?;
    if input.input_geometry_type == "FeatureCollection" {
        return Err(GisError::InvalidGeometry(format!(
            "{UNSUPPORTED_GEOMETRY_TYPE}: repair_geometry does not accept FeatureCollection"
        )));
    }
    require_topology_backend("make_valid")?;
    unreachable!("topology backend is not wired in C4.0")
}

/// Resolve the measurement mode for a CRS. Geographic coordinates are only
/// measured geodesically (metres / m²) and only for WGS84; any other
/// geographic CRS raises rather than silently returning square degrees.
pub fn measure_mode_for_crs(spec: &crate::gis::raster_write::CrsSpec) -> GisResult<MeasureMode> {
    match crate::gis::crs::epsg_code(spec) {
        Some(4326) => Ok(MeasureMode::GeodesicWgs84),
        // The EPSG 4000-4999 block (and 4979) is the geographic-CRS family:
        // planar math there would report degrees as metres.
        Some(code) if (4000..5000).contains(&code) || code == 4979 => {
            Err(GisError::InvalidCrs(format!(
                "invalid_crs: geometry_measure supports geodesic measurement only on EPSG:4326; \
                 geographic CRS EPSG:{code} would yield degree-based lengths/areas"
            )))
        }
        Some(_) => Ok(MeasureMode::Planar),
        None => Err(GisError::InvalidCrs(
            "invalid_crs: geometry_measure requires an EPSG-identified CRS to determine \
             measurement units"
                .to_string(),
        )),
    }
}

pub fn geometry_measure(source: &Value, metrics: &[String], mode: MeasureMode) -> GisResult<Value> {
    let metric_flags = validate_metric_names(metrics)?;
    let input = normalize_input(source, true)?;
    validate_input_or_error(&input)?;
    let stats = measure_geometries(&input.geometries, mode)?;

    let operation = operation_value(
        "geometry_measure",
        &input.input_geometry_type,
        None,
        input.input_count,
        1,
        false,
        input.crs,
        Vec::new(),
    );
    let mut out = Map::new();
    out.insert(
        "area".to_string(),
        if metric_flags.0 && stats.has_area {
            finite_value(stats.area)?
        } else {
            Value::Null
        },
    );
    out.insert(
        "length".to_string(),
        if metric_flags.1 && stats.has_length {
            finite_value(stats.length)?
        } else {
            Value::Null
        },
    );
    out.insert(
        "units".to_string(),
        Value::String(
            match mode {
                MeasureMode::Planar => "source_crs_planar",
                MeasureMode::GeodesicWgs84 => "metres_geodesic_wgs84",
            }
            .to_string(),
        ),
    );
    out.insert("operation".to_string(), operation);
    Ok(Value::Object(out))
}

pub fn geometry_centroid(source: &Value) -> GisResult<Value> {
    let input = normalize_input(source, true)?;
    validate_input_or_error(&input)?;
    let centroid = centroid_for_geometries(&input.geometries, geographic_for_input(&input))?;
    let operation = operation_value(
        "geometry_centroid",
        &input.input_geometry_type,
        Some("Point"),
        input.input_count,
        1,
        false,
        input.crs,
        Vec::new(),
    );
    Ok(json!({
        "geometry": point_value(centroid)?,
        "operation": operation,
    }))
}

pub fn representative_point(source: &Value) -> GisResult<Value> {
    let input = normalize_input(source, false)?;
    if input.input_geometry_type == "FeatureCollection" {
        return Err(GisError::InvalidGeometry(format!(
            "{UNSUPPORTED_GEOMETRY_TYPE}: representative_point does not accept FeatureCollection"
        )));
    }
    let geometry = input
        .geometries
        .first()
        .ok_or_else(model::empty_geometry_error)?;
    let geographic = geographic_for_input(&input);
    let (geometries, wrapped) = dateline_normalized(core::slice::from_ref(geometry), geographic);
    let mut point = representative_for_geometry(&geometries[0])?;
    if wrapped {
        point.x = math::wrap_lon(point.x);
    }
    let operation = operation_value(
        "representative_point",
        &input.input_geometry_type,
        Some("Point"),
        input.input_count,
        1,
        false,
        input.crs,
        Vec::new(),
    );
    Ok(json!({
        "geometry": point_value(point)?,
        "operation": operation,
    }))
}

pub fn interpolate_line(source: &Value, distance: f64, normalized: bool) -> GisResult<Value> {
    validate_distance(distance)?;
    let input = normalize_input(source, false)?;
    if input.input_geometry_type == "FeatureCollection" {
        return Err(GisError::InvalidGeometry(format!(
            "{UNSUPPORTED_GEOMETRY_TYPE}: interpolate_line does not accept FeatureCollection"
        )));
    }
    let geometry = input
        .geometries
        .first()
        .ok_or_else(model::empty_geometry_error)?;
    validate::validate_geometry_or_error(geometry)?;
    let geographic = geographic_for_input(&input);
    let (geometries, wrapped) = dateline_normalized(core::slice::from_ref(geometry), geographic);
    let lines = lines_for_interpolation(&geometries[0])?;
    let total = total_line_length(&lines);
    if total <= EPSILON {
        return Err(GisError::InvalidGeometry(format!(
            "{}: line length must be positive",
            model::INVALID_GEOMETRY
        )));
    }
    let target = normalized_target_distance(distance, normalized, total)?;
    let mut point = interpolate_lines(&lines, target)?;
    if wrapped {
        point.x = math::wrap_lon(point.x);
    }
    let operation = operation_value(
        "interpolate_line",
        &input.input_geometry_type,
        Some("Point"),
        input.input_count,
        1,
        false,
        input.crs,
        Vec::new(),
    );
    Ok(json!({
        "geometry": point_value(point)?,
        "distance": distance,
        "normalized": normalized,
        "operation": operation,
    }))
}

pub fn union_geometries(source: &Value) -> GisResult<Value> {
    let input = normalize_union_input(source)?;
    if input.input_count == 0 {
        let warnings = vec![RasterWarning::new(
            EMPTY_INPUT,
            "union_geometries received no geometries",
            Some("geometries"),
        )];
        return Ok(json!({
            "geometry": Value::Null,
            "operation": operation_value(
                "union_geometries",
                &input.input_geometry_type,
                None,
                0,
                0,
                false,
                input.crs,
                warnings,
            ),
        }));
    }
    validate_input_or_error(&input)?;
    let (geometries, wrapped) =
        dateline_normalized(&input.geometries, geographic_for_input(&input));
    let mut geometry = union_polygonal(&geometries)?;
    if wrapped {
        wrap_geometry_lons(&mut geometry);
    }
    let output_geometry_type = geometry.geometry_type();
    let type_changed = output_geometry_type != input.input_geometry_type;
    let warnings = if type_changed {
        vec![RasterWarning::new(
            GEOMETRY_TYPE_CHANGED,
            format!(
                "union_geometries output type changed from {} to {output_geometry_type}",
                input.input_geometry_type
            ),
            Some("geometry"),
        )]
    } else {
        Vec::new()
    };
    Ok(json!({
        "geometry": geometry_value(&geometry)?,
        "operation": operation_value(
            "union_geometries",
            &input.input_geometry_type,
            Some(output_geometry_type),
            input.input_count,
            1,
            input.input_count != 1 || type_changed,
            input.crs,
            warnings,
        ),
    }))
}

pub fn buffer_geometry(source: &Value, distance: f64, quad_segs: i64) -> GisResult<Value> {
    if !distance.is_finite() {
        return Err(GisError::InvalidArgument(format!(
            "{INVALID_ARGUMENT}: buffer distance must be finite"
        )));
    }
    if quad_segs < 1 {
        return Err(GisError::InvalidArgument(format!(
            "{INVALID_ARGUMENT}: quad_segs must be at least 1"
        )));
    }
    let input = normalize_input(source, false)?;
    validate_input_or_error(&input)?;
    require_topology_backend("buffer_geometry")?;
    let geometry = input
        .geometries
        .first()
        .ok_or_else(model::empty_geometry_error)?;

    if distance == 0.0 && matches!(geometry, Geometry::Polygon(_) | Geometry::MultiPolygon(_)) {
        return buffer_geometry_output(&input, geometry.clone(), false);
    }

    let (geometries, wrapped) = dateline_normalized(
        core::slice::from_ref(geometry),
        geographic_for_input(&input),
    );
    let mut output = buffer_topology(&geometries[0], distance, quad_segs as usize)?;
    if wrapped {
        wrap_geometry_lons(&mut output);
    }
    buffer_geometry_output(&input, output, true)
}

pub fn simplify_geometry(
    source: &Value,
    tolerance: f64,
    preserve_topology: bool,
) -> GisResult<Value> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(GisError::InvalidArgument(format!(
            "{INVALID_ARGUMENT}: simplify tolerance must be finite and non-negative"
        )));
    }
    let input = normalize_input(source, false)?;
    validate_input_or_error(&input)?;
    require_topology_backend("simplify_geometry")?;
    let geometry = input
        .geometries
        .first()
        .ok_or_else(model::empty_geometry_error)?;
    let (geometries, wrapped) = dateline_normalized(
        core::slice::from_ref(geometry),
        geographic_for_input(&input),
    );
    let mut output = simplify_topology(&geometries[0], tolerance, preserve_topology)?;
    if wrapped {
        wrap_geometry_lons(&mut output);
    }
    simplify_geometry_output(&input, output, &input.crs)
}

pub(crate) fn prepare_polygonal_clip_mask(source: &Value) -> GisResult<PolygonalClipMask> {
    let input = normalize_input(source, true)?;
    validate_input_or_error(&input)?;
    for geometry in &input.geometries {
        require_polygonal_geometry(geometry, "clip_vector")?;
    }
    let (geometries, wrapped) =
        dateline_normalized(&input.geometries, geographic_for_input(&input));
    let mut geometry = if geometries.len() == 1 {
        geometries[0].clone()
    } else {
        union_polygonal(&geometries)?
    };
    if wrapped {
        wrap_geometry_lons(&mut geometry);
    }
    Ok(PolygonalClipMask { geometry })
}

pub(crate) fn clip_polygonal_geometry_value(
    source: &Value,
    mask: &PolygonalClipMask,
) -> GisResult<Option<Value>> {
    intersect_polygonal_geometry_values_for_operation(
        source,
        &geometry_value(&mask.geometry)?,
        "clip_vector",
    )
}

pub(crate) fn validate_polygonal_geometry_value(source: &Value, operation: &str) -> GisResult<()> {
    let input = normalize_input(source, false)?;
    validate_input_or_error(&input)?;
    let geometry = input
        .geometries
        .first()
        .ok_or_else(model::empty_geometry_error)?;
    require_polygonal_geometry(geometry, operation)
}

pub(crate) fn union_polygonal_geometry_values(
    sources: &[Value],
    operation: &str,
) -> GisResult<Option<Value>> {
    let mut geometries = Vec::with_capacity(sources.len());
    let mut declared_geographic = true;
    let mut has_declared_crs = false;
    for source in sources {
        let input = normalize_input(source, false)?;
        validate_input_or_error(&input)?;
        if let Some(is_wgs84) = declared_wgs84(input.crs.as_ref()) {
            has_declared_crs = true;
            declared_geographic &= is_wgs84;
        }
        let geometry = input
            .geometries
            .first()
            .ok_or_else(model::empty_geometry_error)?;
        require_polygonal_geometry(geometry, operation)?;
        geometries.push(geometry.clone());
    }
    let geographic = if has_declared_crs {
        declared_geographic
    } else {
        looks_geographic(&geometries)
    };
    let (geometries, wrapped) = dateline_normalized(&geometries, geographic);
    let mut output = union_polygonal(&geometries)?;
    if wrapped {
        wrap_geometry_lons(&mut output);
    }
    if output.is_empty() {
        Ok(None)
    } else {
        geometry_value(&output).map(Some)
    }
}

pub(crate) fn intersect_polygonal_geometry_values(
    left: &Value,
    right: &Value,
) -> GisResult<Option<Value>> {
    intersect_polygonal_geometry_values_for_operation(left, right, "intersect_vectors")
}

fn intersect_polygonal_geometry_values_for_operation(
    left: &Value,
    right: &Value,
    operation: &str,
) -> GisResult<Option<Value>> {
    let left_input = normalize_input(left, false)?;
    validate_input_or_error(&left_input)?;
    let left_geometry = left_input
        .geometries
        .first()
        .ok_or_else(model::empty_geometry_error)?;
    require_polygonal_geometry(left_geometry, operation)?;

    let right_input = normalize_input(right, false)?;
    validate_input_or_error(&right_input)?;
    let right_geometry = right_input
        .geometries
        .first()
        .ok_or_else(model::empty_geometry_error)?;
    require_polygonal_geometry(right_geometry, operation)?;

    // Unwrap the pair, then align both operands onto the same 360° sheet
    // (independent unwrapping can leave one around +180 and the other -180).
    let both = [left_geometry.clone(), right_geometry.clone()];
    let geographic = match (
        declared_wgs84(left_input.crs.as_ref()),
        declared_wgs84(right_input.crs.as_ref()),
    ) {
        (Some(left), Some(right)) => left && right,
        _ => looks_geographic(&both),
    };
    let (mut both, wrapped) = dateline_normalized(&both, geographic);
    if wrapped {
        if let (Some(l0), Some(r0)) = (math::first_lon(&both[0]), math::first_lon(&both[1])) {
            if r0 - l0 > 180.0 {
                math::shift_geometry_lons(&mut both[1], -360.0);
            } else if r0 - l0 < -180.0 {
                math::shift_geometry_lons(&mut both[1], 360.0);
            }
        }
    }
    let mut output = intersection_polygonal(&both[0], &both[1], operation)?;
    if wrapped {
        wrap_geometry_lons(&mut output);
    }
    if output.is_empty() {
        Ok(None)
    } else {
        geometry_value(&output).map(Some)
    }
}

fn buffer_geometry_output(
    input: &NormalizedInput,
    geometry: Geometry,
    changed: bool,
) -> GisResult<Value> {
    if geometry.is_empty() {
        let warnings = vec![RasterWarning::new(
            EMPTY_OUTPUT,
            "buffer_geometry produced an empty geometry",
            Some("geometry"),
        )];
        return Ok(json!({
            "geometry": Value::Null,
            "operation": operation_value(
                "buffer_geometry",
                &input.input_geometry_type,
                None,
                input.input_count,
                0,
                true,
                input.crs.clone(),
                warnings,
            ),
        }));
    }

    let output_geometry_type = geometry.geometry_type();
    let type_changed = output_geometry_type != input.input_geometry_type;
    let warnings = if type_changed {
        vec![RasterWarning::new(
            GEOMETRY_TYPE_CHANGED,
            format!(
                "buffer_geometry output type changed from {} to {output_geometry_type}",
                input.input_geometry_type
            ),
            Some("geometry"),
        )]
    } else {
        Vec::new()
    };

    Ok(json!({
        "geometry": geometry_value(&geometry)?,
        "operation": operation_value(
            "buffer_geometry",
            &input.input_geometry_type,
            Some(output_geometry_type),
            input.input_count,
            1,
            changed || type_changed,
            input.crs.clone(),
            warnings,
        ),
    }))
}

fn simplify_geometry_output(
    input: &NormalizedInput,
    geometry: Geometry,
    crs: &Option<Value>,
) -> GisResult<Value> {
    if geometry.is_empty() {
        let warnings = vec![RasterWarning::new(
            EMPTY_OUTPUT,
            "simplify_geometry produced an empty geometry",
            Some("geometry"),
        )];
        return Ok(json!({
            "geometry": Value::Null,
            "operation": operation_value(
                "simplify_geometry",
                &input.input_geometry_type,
                None,
                input.input_count,
                0,
                true,
                crs.clone(),
                warnings,
            ),
        }));
    }

    let output_geometry_type = geometry.geometry_type();
    let type_changed = output_geometry_type != input.input_geometry_type;
    let warnings = if type_changed {
        vec![RasterWarning::new(
            GEOMETRY_TYPE_CHANGED,
            format!(
                "simplify_geometry output type changed from {} to {output_geometry_type}",
                input.input_geometry_type
            ),
            Some("geometry"),
        )]
    } else {
        Vec::new()
    };

    Ok(json!({
        "geometry": geometry_value(&geometry)?,
        "operation": operation_value(
            "simplify_geometry",
            &input.input_geometry_type,
            Some(output_geometry_type),
            input.input_count,
            1,
            geometry != input.geometries[0] || type_changed,
            crs.clone(),
            warnings,
        ),
    }))
}

fn require_polygonal_geometry(geometry: &Geometry, operation: &str) -> GisResult<()> {
    match geometry {
        Geometry::Polygon(_) | Geometry::MultiPolygon(_) => Ok(()),
        other => Err(GisError::InvalidGeometry(format!(
            "{UNSUPPORTED_GEOMETRY_TYPE}: {operation} supports Polygon and MultiPolygon, got {}",
            other.geometry_type()
        ))),
    }
}

fn normalize_union_input(source: &Value) -> GisResult<NormalizedInput> {
    if let Some(items) = source.as_array() {
        let mut geometries = Vec::with_capacity(items.len());
        for item in items {
            let input = normalize_input(item, false)?;
            geometries.extend(input.geometries);
        }
        return Ok(NormalizedInput {
            input_geometry_type: common_geometry_type(&geometries),
            input_count: items.len(),
            geometries,
            crs: None,
        });
    }
    if source.get("type").and_then(Value::as_str) == Some("FeatureCollection") {
        let features = source
            .get("features")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                GisError::InvalidArgument(format!(
                    "{INVALID_ARGUMENT}: FeatureCollection requires a features array"
                ))
            })?;
        let mut input = normalize_input(source, true)?;
        input.input_geometry_type = common_geometry_type(&input.geometries);
        input.input_count = features.len();
        return Ok(input);
    }
    Err(GisError::InvalidArgument(format!(
        "{INVALID_ARGUMENT}: union_geometries requires a sequence of geometries"
    )))
}

fn common_geometry_type(geometries: &[Geometry]) -> String {
    let Some(first) = geometries.first() else {
        return "Empty".to_string();
    };
    let first_type = first.geometry_type();
    if geometries
        .iter()
        .all(|geometry| geometry.geometry_type() == first_type)
    {
        first_type.to_string()
    } else {
        "Mixed".to_string()
    }
}

fn geometry_value(geometry: &Geometry) -> GisResult<Value> {
    match geometry {
        Geometry::LineString(points) => Ok(json!({
            "type": "LineString",
            "coordinates": line_value(points)?,
        })),
        Geometry::Polygon(rings) => Ok(json!({
            "type": "Polygon",
            "coordinates": rings_value(rings)?,
        })),
        Geometry::MultiLineString(lines) => {
            let mut out = Vec::with_capacity(lines.len());
            for line in lines {
                out.push(line_value(line)?);
            }
            Ok(json!({
                "type": "MultiLineString",
                "coordinates": out,
            }))
        }
        Geometry::MultiPolygon(polygons) => {
            let mut out = Vec::with_capacity(polygons.len());
            for rings in polygons {
                out.push(rings_value(rings)?);
            }
            Ok(json!({
                "type": "MultiPolygon",
                "coordinates": out,
            }))
        }
        Geometry::Empty => Ok(Value::Null),
        other => Err(GisError::InvalidGeometry(format!(
            "{UNSUPPORTED_GEOMETRY_TYPE}: cannot serialize {} as polygonal output",
            other.geometry_type()
        ))),
    }
}

fn line_value(points: &[Coord]) -> GisResult<Vec<Value>> {
    points
        .iter()
        .map(|coord| Ok(json!([finite_value(coord.x)?, finite_value(coord.y)?])))
        .collect()
}

fn rings_value(rings: &[Vec<Coord>]) -> GisResult<Vec<Vec<Value>>> {
    rings.iter().map(|ring| line_value(ring)).collect()
}

#[cfg(feature = "extension-module")]
pub use py::{
    buffer_geometry_py, geometry_centroid_py, geometry_measure_py, interpolate_line_py,
    measure_geometries_py, repair_geometry_py, representative_point_py, simplify_geometry_py,
    union_geometries_py, validate_geometry_py,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polygon_area_length_and_centroid_are_planar() {
        let geometry = json!({
            "type": "Polygon",
            "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]
        });

        let measure = geometry_measure(
            &geometry,
            &["area".to_string(), "length".to_string()],
            MeasureMode::Planar,
        )
        .expect("measurement succeeds");
        let centroid = geometry_centroid(&geometry).expect("centroid succeeds");

        assert_eq!(measure["area"], json!(1.0));
        assert_eq!(measure["length"], json!(4.0));
        assert_eq!(centroid["geometry"]["coordinates"], json!([0.5, 0.5]));
    }

    #[test]
    fn geodesic_measure_returns_metres_not_degrees() {
        // 1°×1° quad at the equator: ~1.2308e10 m² and ~443 km perimeter.
        let geometry = json!({
            "type": "Polygon",
            "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]
        });
        let measure = geometry_measure(
            &geometry,
            &["area".to_string(), "length".to_string()],
            MeasureMode::GeodesicWgs84,
        )
        .expect("measurement succeeds");
        let area = measure["area"].as_f64().unwrap();
        let length = measure["length"].as_f64().unwrap();
        assert!((area - 1.2308e10).abs() < 2e8, "area = {area}");
        assert!((length - 443_770.0).abs() < 2_000.0, "length = {length}");
        assert_eq!(measure["units"], json!("metres_geodesic_wgs84"));
    }

    #[test]
    fn dateline_polygon_measures_and_centroids_locally() {
        // Regression (MENSURA item 4): a polygon spanning 179° → -179° must
        // behave as a 2°-wide patch, not the 358°-wide complement.
        let geometry = json!({
            "type": "Polygon",
            "coordinates": [[[179.0, 0.0], [-179.0, 0.0], [-179.0, 1.0], [179.0, 1.0], [179.0, 0.0]]]
        });
        let measure =
            geometry_measure(&geometry, &["area".to_string()], MeasureMode::GeodesicWgs84)
                .expect("measurement succeeds");
        let area = measure["area"].as_f64().unwrap();
        assert!((area - 2.0 * 1.2308e10).abs() < 4e8, "area = {area}");

        let centroid = geometry_centroid(&geometry).expect("centroid succeeds");
        let lon = centroid["geometry"]["coordinates"][0].as_f64().unwrap();
        let lat = centroid["geometry"]["coordinates"][1].as_f64().unwrap();
        assert!(
            lon.abs() > 179.0,
            "centroid lon must sit at the dateline, got {lon}"
        );
        assert!((lat - 0.5).abs() < 1e-9, "centroid lat = {lat}");
    }

    #[test]
    fn measure_mode_rejects_non_wgs84_geographic_crs() {
        let wgs84 = crate::gis::raster_write::CrsSpec::from_string("EPSG:4326".to_string())
            .expect("crs parses");
        assert_eq!(
            measure_mode_for_crs(&wgs84).unwrap(),
            MeasureMode::GeodesicWgs84
        );
        let utm = crate::gis::raster_write::CrsSpec::from_string("EPSG:32633".to_string())
            .expect("crs parses");
        assert_eq!(measure_mode_for_crs(&utm).unwrap(), MeasureMode::Planar);
        let nad83 = crate::gis::raster_write::CrsSpec::from_string("EPSG:4269".to_string())
            .expect("crs parses");
        assert!(measure_mode_for_crs(&nad83).is_err());
    }

    #[test]
    fn bowtie_polygon_is_invalid() {
        let geometry = json!({
            "type": "Polygon",
            "coordinates": [[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]]
        });

        let result = validate_geometry(&geometry).expect("validation returns a result");

        assert_eq!(result["valid"], json!(false));
        assert!(result["reason"]
            .as_str()
            .unwrap()
            .contains(model::INVALID_GEOMETRY));
    }

    #[test]
    fn multiline_interpolation_is_cumulative() {
        let geometry = json!({
            "type": "MultiLineString",
            "coordinates": [
                [[0.0, 0.0], [2.0, 0.0]],
                [[2.0, 0.0], [2.0, 2.0]]
            ]
        });

        let result = interpolate_line(&geometry, 3.0, false).expect("interpolation succeeds");

        assert_eq!(result["geometry"]["coordinates"], json!([2.0, 1.0]));
    }
}
