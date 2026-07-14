//! CRS resolution for geometry-only operations (MENSURA M-04): every
//! operation decides geographic (WGS84 lon/lat, antimeridian handling on)
//! versus planar handling from an EXPLICIT CRS — an explicit `crs=` argument
//! and/or the input's embedded CRS metadata — never from coordinate ranges.

use serde_json::Value;

use crate::gis::error::{GisError, GisResult};
use crate::gis::raster_write::CrsSpec;

use super::math::unwrap_dateline;
use super::measure::MeasureMode;
use super::model::{Geometry, NormalizedInput};

/// Decide geographic vs planar handling from an explicit CRS. EPSG:4326 =>
/// geographic; a known projected CRS (EPSG-coded or a custom projected-method
/// definition) => planar; a non-WGS84 geographic CRS, a geocentric CRS, an
/// unclassified code, or an unidentifiable CRS is a stable error (see
/// `crs::epsg_crs_kind`).
pub(crate) fn geographic_from_spec(spec: &CrsSpec) -> GisResult<bool> {
    use crate::gis::crs::EpsgCrsKind;
    // A custom projected-method CRS ({"method": "aea", ...} =>
    // CrsSpec::from_projection) is planar by construction: every supported
    // ProjectionDefinition maps lon/lat onto a Cartesian easting/northing
    // plane. It carries no EPSG authority, so it must be classified before
    // the EPSG-code requirement below.
    if spec.projection.is_some() {
        return Ok(false);
    }
    let Some(code) = crate::gis::crs::epsg_code(spec) else {
        return Err(GisError::InvalidCrs(
            "invalid_crs: geometry operation requires an EPSG-identified CRS to determine \
             geographic-versus-planar handling"
                .to_string(),
        ));
    };
    match crate::gis::crs::epsg_crs_kind(code) {
        EpsgCrsKind::GeographicWgs84 => Ok(true),
        EpsgCrsKind::Projected => Ok(false),
        // Treating a geographic CRS as planar would report degrees as metres,
        // and only WGS84 is supported for geodesic/dateline handling.
        EpsgCrsKind::Geographic => Err(GisError::InvalidCrs(format!(
            "invalid_crs: geographic CRS EPSG:{code} is not supported for geometry \
             operations; only EPSG:4326 is handled geographically"
        ))),
        EpsgCrsKind::Geocentric => Err(GisError::InvalidCrs(format!(
            "invalid_crs: geocentric CRS EPSG:{code} has 3D Cartesian axes and is not \
             supported for 2D geometry operations"
        ))),
        EpsgCrsKind::Unclassified => Err(GisError::InvalidCrs(format!(
            "invalid_crs: EPSG:{code} is not in the built-in geographic/projected \
             classification table; geometry operations support EPSG:4326 and the curated \
             projected CRSs"
        ))),
    }
}

/// Resolve the measurement mode for a CRS. Geographic coordinates are only
/// measured geodesically (metres / m²) and only for WGS84; any other
/// geographic CRS raises rather than silently returning square degrees.
pub fn measure_mode_for_crs(spec: &CrsSpec) -> GisResult<MeasureMode> {
    use crate::gis::crs::EpsgCrsKind;
    // Custom projected-method CRSs are planar by construction (see
    // `geographic_from_spec`) and are measured in their Cartesian CRS units.
    if spec.projection.is_some() {
        return Ok(MeasureMode::Planar);
    }
    let Some(code) = crate::gis::crs::epsg_code(spec) else {
        return Err(GisError::InvalidCrs(
            "invalid_crs: geometry_measure requires an EPSG-identified CRS to determine \
             measurement units"
                .to_string(),
        ));
    };
    match crate::gis::crs::epsg_crs_kind(code) {
        EpsgCrsKind::GeographicWgs84 => Ok(MeasureMode::GeodesicWgs84),
        EpsgCrsKind::Projected => Ok(MeasureMode::Planar),
        // Planar math on a degree-axis CRS would report degrees as metres.
        EpsgCrsKind::Geographic => Err(GisError::InvalidCrs(format!(
            "invalid_crs: geometry_measure supports geodesic measurement only on EPSG:4326; \
             geographic CRS EPSG:{code} would yield degree-based lengths/areas"
        ))),
        EpsgCrsKind::Geocentric => Err(GisError::InvalidCrs(format!(
            "invalid_crs: geocentric CRS EPSG:{code} has 3D Cartesian axes and is not \
             supported for 2D geometry measurement"
        ))),
        EpsgCrsKind::Unclassified => Err(GisError::InvalidCrs(format!(
            "invalid_crs: EPSG:{code} is not in the built-in geographic/projected \
             classification table; geometry_measure supports EPSG:4326 and the curated \
             projected CRSs"
        ))),
    }
}

/// Parse the embedded (`info`-derived) CRS metadata carried on a geometry input
/// (`{source_kind, wkt, authority}`) into a `CrsSpec`.
pub(super) fn embedded_crs_spec(crs: &Value) -> GisResult<Option<CrsSpec>> {
    crate::gis::vector::crs_spec_from_authority_wkt(crs.get("authority"), crs.get("wkt"))
}

/// Merge a FeatureCollection's collection-level CRS metadata with each
/// feature's embedded metadata. Mixed-CRS input is ill-defined, so every
/// declared CRS must agree: a FeatureCollection mixing EPSG:4326 and
/// EPSG:3857 features raises `crs_mismatch` instead of silently operating
/// under the collection-level (or first) declaration.
pub(super) fn merged_feature_collection_crs(
    collection_crs: Option<Value>,
    feature_crs: impl Iterator<Item = Value>,
) -> GisResult<Option<Value>> {
    let mut value = collection_crs;
    let mut spec = value.as_ref().map(embedded_crs_spec).transpose()?.flatten();
    for candidate in feature_crs {
        let candidate_spec = embedded_crs_spec(&candidate)?;
        spec = crate::gis::vector::compatible_metadata_crs(
            spec.take(),
            candidate_spec,
            "FeatureCollection",
        )?;
        if value.is_none() {
            value = Some(candidate);
        }
    }
    Ok(value)
}

/// Resolve the geographic-vs-planar treatment for a geometry-only operation from
/// an explicit `crs=` argument and/or the input's embedded CRS metadata. The two
/// must be compatible when both are present; if NEITHER is present the operation
/// raises a stable `missing_crs` error. Coordinate ranges are never consulted.
pub(super) fn resolve_geographic(
    input: &NormalizedInput,
    explicit: Option<&CrsSpec>,
) -> GisResult<bool> {
    let embedded = input
        .crs
        .as_ref()
        .map(embedded_crs_spec)
        .transpose()?
        .flatten();
    let spec =
        crate::gis::vector::compatible_metadata_crs(explicit.cloned(), embedded, "geometry")?
            .ok_or_else(|| {
                GisError::MissingCrs(
                    "missing_crs: geometry operation requires an explicit crs=... or embedded CRS \
                 metadata on the input; coordinate ranges are not used to infer a CRS"
                        .to_string(),
                )
            })?;
    geographic_from_spec(&spec)
}

/// Unwrap geographic geometries onto one continuous longitude sheet before a
/// topology operation; planar inputs pass through untouched.
pub(super) fn dateline_normalized(
    geometries: &[Geometry],
    geographic: bool,
) -> (Vec<Geometry>, bool) {
    let mut out = geometries.to_vec();
    let unwrapped = geographic && unwrap_dateline(&mut out);
    (out, unwrapped)
}
