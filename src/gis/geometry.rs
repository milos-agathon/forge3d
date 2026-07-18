use serde_json::{json, Map, Value};

use crate::gis::error::{GisError, GisResult};
use crate::gis::types::RasterWarning;

mod antimeridian;
mod centroid;
mod crs_resolve;
mod line_ops;
mod math;
mod measure;
mod model;
mod parse;
#[cfg(feature = "extension-module")]
mod py;
pub(crate) mod topology;
mod topology_buffer;
mod topology_simplify;
mod validate;

use centroid::centroid_for_geometries;
pub(crate) use crs_resolve::geographic_from_spec;
pub use crs_resolve::measure_mode_for_crs;
use crs_resolve::{dateline_normalized, embedded_crs_spec, resolve_geographic};
use line_ops::{
    interpolate_lines, lines_for_interpolation, normalized_target_distance,
    representative_for_geometry, total_line_length, validate_distance,
};

use crate::gis::raster_write::CrsSpec;
pub use measure::MeasureMode;
use measure::{measure_geometries, validate_metric_names};
use model::{
    finite_value, operation_value, point_value, Coord, Geometry, NormalizedInput, EMPTY_INPUT,
    EMPTY_OUTPUT, EPSILON, GEOMETRY_TYPE_CHANGED, INVALID_ARGUMENT, UNSUPPORTED_GEOMETRY_TYPE,
    UNSUPPORTED_OPTION,
};
use parse::normalize_input;

use topology::{
    buffer_topology, difference_polygonal, intersection_polygonal, polygonal_validity,
    require_topology_backend, simplify_topology, symmetric_difference_polygonal, union_polygonal,
};
use validate::{validate_geometry_value, validate_input_or_error};

pub(crate) struct PolygonalClipMask {
    geometry: Geometry,
}

/// Canonical ±180 contract: EVERY geographic output is split at the
/// antimeridian with longitudes wrapped into [-180, 180] — including outputs
/// from same-sheet inputs (e.g. 175..185) that never needed unwrapping.
/// Planar outputs pass through untouched.
fn canonical_output(geometry: Geometry, geographic: bool) -> Geometry {
    if geographic {
        antimeridian::split_at_antimeridian(&geometry)
    } else {
        geometry
    }
}

/// Point-output companion of `canonical_output`: geographic longitudes are
/// wrapped into (-180, 180].
fn canonical_point(mut point: Coord, geographic: bool) -> Coord {
    if geographic {
        point.x = math::wrap_lon(point.x);
    }
    point
}

pub fn validate_geometry(source: &Value) -> GisResult<Value> {
    Ok(validate_geometry_value(source))
}

pub fn is_valid(source: &Value) -> GisResult<Value> {
    let basic = validate_geometry_value(source);
    if !basic.get("valid").and_then(Value::as_bool).unwrap_or(false) {
        let reason = basic
            .get("reason")
            .and_then(Value::as_str)
            .unwrap_or("invalid geometry")
            .to_string();
        return Ok(json!({
            "valid": false,
            "reason": reason,
            "reasons": [reason],
        }));
    }
    let input = normalize_input(source, false)?;
    let geometry = input
        .geometries
        .first()
        .ok_or_else(model::empty_geometry_error)?;
    let report = polygonal_validity(geometry)?;
    Ok(json!({
        "valid": report.valid,
        "reason": report.reasons.first().cloned(),
        "reasons": report.reasons,
    }))
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

pub fn geometry_centroid(source: &Value, crs: Option<CrsSpec>) -> GisResult<Value> {
    let input = normalize_input(source, true)?;
    validate_input_or_error(&input)?;
    let geographic = resolve_geographic(&input, crs.as_ref())?;
    let centroid = centroid_for_geometries(&input.geometries, geographic)?;
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

pub fn representative_point(source: &Value, crs: Option<CrsSpec>) -> GisResult<Value> {
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
    let geographic = resolve_geographic(&input, crs.as_ref())?;
    let (geometries, _) = dateline_normalized(core::slice::from_ref(geometry), geographic);
    let point = canonical_point(
        representative_for_geometry(&geometries[0], geographic)?,
        geographic,
    );
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

pub fn interpolate_line(
    source: &Value,
    distance: f64,
    normalized: bool,
    crs: Option<CrsSpec>,
) -> GisResult<Value> {
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
    let geographic = resolve_geographic(&input, crs.as_ref())?;
    let (geometries, _) = dateline_normalized(core::slice::from_ref(geometry), geographic);
    let lines = lines_for_interpolation(&geometries[0])?;
    // MENSURA M-04: with a geographic CRS, `total` and `distance` are geodesic
    // metres; with a projected CRS they are planar CRS units. Degrees are
    // never a distance unit.
    let total = total_line_length(&lines, geographic);
    if total <= EPSILON {
        return Err(GisError::InvalidGeometry(format!(
            "{}: line length must be positive",
            model::INVALID_GEOMETRY
        )));
    }
    let target = normalized_target_distance(distance, normalized, total)?;
    let point = canonical_point(interpolate_lines(&lines, target, geographic)?, geographic);
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

pub fn union_geometries(source: &Value, crs: Option<CrsSpec>) -> GisResult<Value> {
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
    let geographic = resolve_geographic(&input, crs.as_ref())?;
    let (geometries, _) = dateline_normalized(&input.geometries, geographic);
    let geometry = canonical_output(union_polygonal(&geometries)?, geographic);
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

fn boolean_geometries(
    source: &Value,
    crs: Option<CrsSpec>,
    name: &str,
    combine: fn(&Geometry, &Geometry, &str) -> GisResult<Geometry>,
) -> GisResult<Value> {
    let input = normalize_union_input(source)?;
    if input.input_count == 0 {
        return Ok(json!({
            "geometry": Value::Null,
            "operation": operation_value(
                name,
                &input.input_geometry_type,
                None,
                0,
                0,
                false,
                input.crs,
                vec![RasterWarning::new(EMPTY_INPUT, format!("{name} received no geometries"), Some("geometries"))],
            ),
        }));
    }
    validate_input_or_error(&input)?;
    let geographic = resolve_geographic(&input, crs.as_ref())?;
    let (geometries, _) = dateline_normalized(&input.geometries, geographic);
    let mut output = union_polygonal(&geometries[..1])?;
    for geometry in &geometries[1..] {
        output = combine(&output, geometry, name)?;
    }
    let geometry = canonical_output(output, geographic);
    let output_geometry_type = geometry.geometry_type();
    let type_changed = output_geometry_type != input.input_geometry_type;
    let warnings = if type_changed {
        vec![RasterWarning::new(
            GEOMETRY_TYPE_CHANGED,
            format!(
                "{name} output type changed from {} to {output_geometry_type}",
                input.input_geometry_type
            ),
            Some("geometry"),
        )]
    } else {
        Vec::new()
    };
    let output_count = usize::from(!geometry.is_empty());
    Ok(json!({
        "geometry": geometry_value(&geometry)?,
        "operation": operation_value(
            name,
            &input.input_geometry_type,
            Some(output_geometry_type),
            input.input_count,
            output_count,
            input.input_count != 1 || type_changed,
            input.crs,
            warnings,
        ),
    }))
}

pub fn intersection_geometries(source: &Value, crs: Option<CrsSpec>) -> GisResult<Value> {
    boolean_geometries(source, crs, "intersection", intersection_polygonal)
}

pub fn difference_geometries(source: &Value, crs: Option<CrsSpec>) -> GisResult<Value> {
    boolean_geometries(source, crs, "difference", difference_polygonal)
}

pub fn symmetric_difference_geometries(source: &Value, crs: Option<CrsSpec>) -> GisResult<Value> {
    boolean_geometries(
        source,
        crs,
        "symmetric_difference",
        symmetric_difference_polygonal,
    )
}

pub fn buffer_geometry(
    source: &Value,
    distance: f64,
    quad_segs: i64,
    crs: Option<CrsSpec>,
) -> GisResult<Value> {
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

    let geographic = resolve_geographic(&input, crs.as_ref())?;

    // Zero-distance polygonal buffering is an identity for the shape, but it
    // still honours the full CRS contract: the CRS must resolve (missing_crs
    // otherwise) and a geographic dateline crossing is split like any other
    // topology output.
    if distance == 0.0 && matches!(geometry, Geometry::Polygon(_) | Geometry::MultiPolygon(_)) {
        let (mut geometries, _) = dateline_normalized(core::slice::from_ref(geometry), geographic);
        let output = canonical_output(geometries.remove(0), geographic);
        let changed = &output != geometry;
        return buffer_geometry_output(&input, output, changed);
    }

    let (geometries, _) = dateline_normalized(core::slice::from_ref(geometry), geographic);
    let output = canonical_output(
        buffer_topology(&geometries[0], distance, quad_segs as usize)?,
        geographic,
    );
    buffer_geometry_output(&input, output, true)
}

pub fn simplify_geometry(
    source: &Value,
    tolerance: f64,
    preserve_topology: bool,
    crs: Option<CrsSpec>,
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
    let geographic = resolve_geographic(&input, crs.as_ref())?;
    let (geometries, _) = dateline_normalized(core::slice::from_ref(geometry), geographic);
    let output = canonical_output(
        simplify_topology(&geometries[0], tolerance, preserve_topology)?,
        geographic,
    );
    simplify_geometry_output(&input, output, &input.crs)
}

pub(crate) fn prepare_polygonal_clip_mask(
    source: &Value,
    geographic: bool,
) -> GisResult<PolygonalClipMask> {
    let input = normalize_input(source, true)?;
    validate_input_or_error(&input)?;
    for geometry in &input.geometries {
        require_polygonal_geometry(geometry, "clip_vector")?;
    }
    // Build the clip mask WITHOUT antimeridian splitting here. The per-feature
    // intersection (`intersect_polygonal_geometry_values_for_operation`) unwraps
    // the (source, mask) PAIR onto one continuous longitude sheet and splits the
    // completed result exactly once at the output boundary. Pre-splitting the
    // mask stranded its two dateline pieces on opposite 360° sheets, so the
    // intersection dropped whichever piece missed the source's sheet — the west
    // half of a dateline-crossing clip was silently lost (MENSURA M-04).
    let geometry = if input.geometries.len() == 1 {
        // Single clip polygon: hand it to the intersection in its authored
        // (wrapped) coordinates so the pair-unwrap can re-detect the crossing.
        input.geometries[0].clone()
    } else {
        // Multiple clip polygons must be unioned on a continuous sheet; wrap the
        // completed union back to authored coordinates so the intersection still
        // sees (and splits, exactly once) any antimeridian crossing.
        let (geometries, _) = dateline_normalized(&input.geometries, geographic);
        let mut unioned = union_polygonal(&geometries)?;
        if geographic {
            math::wrap_geometry_lons(&mut unioned);
        }
        unioned
    };
    Ok(PolygonalClipMask { geometry })
}

pub(crate) fn clip_polygonal_geometry_value(
    source: &Value,
    mask: &PolygonalClipMask,
    geographic: bool,
) -> GisResult<Option<Value>> {
    intersect_polygonal_geometry_values_for_operation(
        source,
        &geometry_value(&mask.geometry)?,
        "clip_vector",
        geographic,
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
    geographic: bool,
) -> GisResult<Option<Value>> {
    let mut geometries = Vec::with_capacity(sources.len());
    for source in sources {
        let input = normalize_input(source, false)?;
        validate_input_or_error(&input)?;
        let geometry = input
            .geometries
            .first()
            .ok_or_else(model::empty_geometry_error)?;
        require_polygonal_geometry(geometry, operation)?;
        geometries.push(geometry.clone());
    }
    let (geometries, _) = dateline_normalized(&geometries, geographic);
    let output = canonical_output(union_polygonal(&geometries)?, geographic);
    if output.is_empty() {
        Ok(None)
    } else {
        geometry_value(&output).map(Some)
    }
}

pub(crate) fn intersect_polygonal_geometry_values(
    left: &Value,
    right: &Value,
    geographic: bool,
) -> GisResult<Option<Value>> {
    intersect_polygonal_geometry_values_for_operation(left, right, "intersect_vectors", geographic)
}

fn intersect_polygonal_geometry_values_for_operation(
    left: &Value,
    right: &Value,
    operation: &str,
    geographic: bool,
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

    // Unwrap the pair, then align every polygonal PART of both operands onto
    // the left operand's 360° sheet. Per-part alignment (not a whole-operand
    // shift) is required: a MultiPolygon that was previously split at the
    // antimeridian carries parts on opposite sheets, and operands authored on
    // different sheets (175..185 vs -185..-175) must still intersect.
    let both = [left_geometry.clone(), right_geometry.clone()];
    let (mut both, _) = dateline_normalized(&both, geographic);
    if geographic {
        if let Some(reference) = math::first_part_mean_lon(&both[0]) {
            math::align_parts_to_sheet(&mut both[0], reference);
            math::align_parts_to_sheet(&mut both[1], reference);
        }
    }
    // Canonicalize unconditionally — two operands on the SAME non-canonical
    // sheet (both 175..185) need neither unwrap nor sheet alignment, yet
    // their result still crosses ±180.
    let output = canonical_output(
        intersection_polygonal(&both[0], &both[1], operation)?,
        geographic,
    );
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
        // Carry the items' embedded CRS metadata into `resolve_geographic`.
        // Mixed-CRS input is ill-defined, so every declared CRS must agree —
        // two items declaring different CRSs raise `crs_mismatch` rather than
        // silently unioning under the first declaration. An explicit `crs=`
        // arg (if supplied) must still be compatible with the shared one.
        let mut crs: Option<Value> = None;
        let mut crs_spec: Option<CrsSpec> = None;
        for item in items {
            let input = normalize_input(item, false)?;
            if let Some(value) = &input.crs {
                let spec = embedded_crs_spec(value)?;
                crs_spec = crate::gis::vector::compatible_metadata_crs(
                    crs_spec.take(),
                    spec,
                    "union_geometries",
                )?;
                if crs.is_none() {
                    crs = Some(value.clone());
                }
            }
            geometries.extend(input.geometries);
        }
        return Ok(NormalizedInput {
            input_geometry_type: common_geometry_type(&geometries),
            input_count: items.len(),
            geometries,
            crs,
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
        // `normalize_input` enforces the same mixed-CRS contract on the
        // FeatureCollection path: collection-level and per-feature CRS
        // metadata must all agree or it raises `crs_mismatch`.
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
    buffer_geometry_py, difference_geometries_py, geometry_centroid_py, geometry_measure_py,
    interpolate_line_py, intersection_geometries_py, is_valid_py, measure_geometries_py,
    repair_geometry_py, representative_point_py, simplify_geometry_py,
    symmetric_difference_geometries_py, union_geometries_py, union_py, validate_geometry_py,
};

#[cfg(test)]
mod tests;
