use std::collections::HashMap;

use crate::gis::affine::validate_bounds_tuple;
use crate::gis::error::{GisError, GisResult};
use crate::gis::raster_write::CrsSpec;
use crate::gis::types::{RasterBounds, RasterInfo, RasterWarning, WARNING_MISSING_CRS};

const WEB_MERCATOR_MAX_LAT: f64 = 85.051_128_78;

#[derive(Debug, Clone)]
pub struct CrsInspection {
    pub source_kind: String,
    pub missing: bool,
    pub wkt: Option<String>,
    pub authority: Option<HashMap<String, String>>,
    pub axis_order: Option<String>,
    pub axis_order_policy: String,
    pub warnings: Vec<RasterWarning>,
}

pub fn inspect_info_crs(info: &RasterInfo, source_kind: &str) -> GisResult<CrsInspection> {
    let warnings = info
        .warnings
        .iter()
        .filter(|warning| warning.code == WARNING_MISSING_CRS)
        .cloned()
        .collect::<Vec<_>>();
    Ok(CrsInspection {
        source_kind: source_kind.to_string(),
        missing: info.crs_wkt.is_none() && info.crs_authority.is_none(),
        wkt: info.crs_wkt.clone(),
        authority: info.crs_authority.clone(),
        axis_order: info.crs_authority.as_ref().map(|_| "xy".to_string()),
        axis_order_policy: "traditional_gis_xy".to_string(),
        warnings,
    })
}

pub fn inspect_crs_spec(spec: &CrsSpec, source_kind: &str) -> CrsInspection {
    CrsInspection {
        source_kind: source_kind.to_string(),
        missing: false,
        wkt: spec.wkt.clone(),
        authority: authority_map(spec),
        axis_order: Some("xy".to_string()),
        axis_order_policy: "traditional_gis_xy".to_string(),
        warnings: Vec::new(),
    }
}

pub fn require_crs(info: &RasterInfo) -> GisResult<CrsSpec> {
    CrsSpec::from_raster_info(info)
        .ok_or_else(|| GisError::MissingCrs("missing_crs: raster has no CRS metadata".to_string()))
}

pub fn crs_equal(left: &CrsSpec, right: &CrsSpec) -> bool {
    left.equivalent_to(right)
}

pub fn authority_map(spec: &CrsSpec) -> Option<HashMap<String, String>> {
    let (name, code) = spec.authority.as_ref()?;
    let mut authority = HashMap::new();
    authority.insert("name".to_string(), name.to_ascii_uppercase());
    authority.insert("code".to_string(), code.to_string());
    Some(authority)
}

pub fn epsg_code(spec: &CrsSpec) -> Option<u32> {
    if let Some((name, code)) = &spec.authority {
        if name.eq_ignore_ascii_case("EPSG") {
            return code.parse().ok();
        }
    }
    if spec
        .wkt
        .as_deref()
        .is_some_and(|wkt| wkt.to_ascii_uppercase().contains("WGS 84"))
    {
        return Some(4326);
    }
    None
}

pub fn canonical_label(spec: &CrsSpec) -> GisResult<String> {
    if let Some(projection) = spec.projection {
        return Ok(format!("forge3d:{projection:?}"));
    }
    if let Some((name, code)) = &spec.authority {
        return Ok(format!("{}:{code}", name.to_ascii_uppercase()));
    }
    if let Some(wkt) = &spec.wkt {
        return Ok(wkt.clone());
    }
    Err(GisError::MissingCrs(
        "missing_crs: CRS definition is empty".to_string(),
    ))
}

pub fn transform_bounds(
    bounds: RasterBounds,
    src: &CrsSpec,
    dst: &CrsSpec,
) -> GisResult<RasterBounds> {
    // densify = 1 samples exactly the four corners (legacy behaviour).
    transform_bounds_densified(bounds, src, dst, 1)
}

/// Transform bounds by densifying each edge into `densify` segments before
/// reprojecting — required for correctness under any projection with curved
/// parallels/meridians, where the extremum of an edge lies between corners.
pub fn transform_bounds_densified(
    bounds: RasterBounds,
    src: &CrsSpec,
    dst: &CrsSpec,
    densify: u32,
) -> GisResult<RasterBounds> {
    if crs_equal(src, dst) {
        return Ok(bounds);
    }
    if densify == 0 {
        return Err(GisError::InvalidArgument(
            "invalid_argument: transform_bounds densify must be at least 1".to_string(),
        ));
    }
    let n = densify as usize;
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut fold = |x: f64, y: f64| -> GisResult<()> {
        let (tx, ty) = transform_point(x, y, src, dst)?;
        min_x = min_x.min(tx);
        max_x = max_x.max(tx);
        min_y = min_y.min(ty);
        max_y = max_y.max(ty);
        Ok(())
    };
    for i in 0..=n {
        let t = i as f64 / n as f64;
        let x = bounds.left + t * (bounds.right - bounds.left);
        let y = bounds.bottom + t * (bounds.top - bounds.bottom);
        fold(x, bounds.bottom)?;
        fold(x, bounds.top)?;
        fold(bounds.left, y)?;
        fold(bounds.right, y)?;
    }
    validate_bounds_tuple((min_x, min_y, max_x, max_y), false)
}

/// Check whether a CRS pair is dispatchable by the built-in engine without
/// evaluating any coordinate.
pub fn transform_pair_supported(src: &CrsSpec, dst: &CrsSpec) -> GisResult<()> {
    use crate::geo::projections::epsg_projection;
    if crs_equal(src, dst) {
        return Ok(());
    }
    if src.projection.is_some() || dst.projection.is_some() {
        return Ok(());
    }
    let is_geocentric_pair = matches!(
        (epsg_code(src), epsg_code(dst)),
        (Some(4326 | 4979), Some(4978)) | (Some(4978), Some(4326 | 4979))
    );
    if is_geocentric_pair {
        return Ok(());
    }
    let ok = |code: u32| code == 4326 || epsg_projection(code).is_some();
    match (epsg_code(src), epsg_code(dst)) {
        (Some(s), Some(d)) if ok(s) && ok(d) => Ok(()),
        _ => Err(GisError::BackendUnavailable(format!(
            "BackendUnavailable: CRS transform {} to {} requires an unavailable PROJ backend",
            canonical_label(src)?,
            canonical_label(dst)?
        ))),
    }
}

/// Transform a WGS84 geographic 3D coordinate to/from WGS84 geocentric
/// coordinates (EPSG method 9602). Planar methods deliberately remain on
/// [`transform_point`], whose two-value contract cannot carry ellipsoidal
/// height.
pub fn transform_point3(
    x: f64,
    y: f64,
    z: f64,
    src: &CrsSpec,
    dst: &CrsSpec,
) -> GisResult<(f64, f64, f64)> {
    use crate::geo::projections::geocentric::{wgs84_ecef_to_geodetic, wgs84_geodetic_to_ecef};
    use glam::DVec3;

    if !x.is_finite() || !y.is_finite() || !z.is_finite() {
        return Err(GisError::TransformFailed(
            "coordinate values must be finite".to_string(),
        ));
    }
    match (epsg_code(src), epsg_code(dst)) {
        (Some(4326 | 4979), Some(4978)) => {
            let point = wgs84_geodetic_to_ecef(x, y, z)
                .map_err(|error| GisError::TransformFailed(error.to_string()))?;
            Ok((point.x, point.y, point.z))
        }
        (Some(4978), Some(4326 | 4979)) => wgs84_ecef_to_geodetic(DVec3::new(x, y, z))
            .map_err(|error| GisError::TransformFailed(error.to_string())),
        _ => Err(GisError::BackendUnavailable(format!(
            "BackendUnavailable: CRS transform {} to {} has no built-in 3D path",
            canonical_label(src)?,
            canonical_label(dst)?
        ))),
    }
}

/// Dispatch a coordinate through the built-in pure-Rust projection engine
/// (src/geo/projections). Supported EPSG codes: 4326, 3857, and the WGS84 UTM
/// zones 326zz/327zz. Anything else is an explicit `BackendUnavailable` — the
/// engine never silently passes coordinates through.
pub fn transform_point(x: f64, y: f64, src: &CrsSpec, dst: &CrsSpec) -> GisResult<(f64, f64)> {
    use crate::geo::projections::{epsg_forward, epsg_inverse, epsg_projection, ProjError};

    if !x.is_finite() || !y.is_finite() {
        return Err(GisError::TransformFailed(
            "coordinate values must be finite".to_string(),
        ));
    }
    if crs_equal(src, dst) {
        return Ok((x, y));
    }
    let unsupported = |src: &CrsSpec, dst: &CrsSpec| -> GisResult<(f64, f64)> {
        Err(GisError::BackendUnavailable(format!(
            "BackendUnavailable: CRS transform {} to {} requires an unavailable PROJ backend",
            canonical_label(src)?,
            canonical_label(dst)?
        )))
    };
    let map_err = |err: ProjError| -> GisError {
        match err {
            ProjError::Domain(msg) => GisError::InvalidBounds(msg),
            other => GisError::TransformFailed(other.to_string()),
        }
    };
    let (src_code, dst_code) = (epsg_code(src), epsg_code(dst));
    // Route through geographic WGS84: src → 4326 → dst.
    let (lon, lat) = if let Some(projection) = src.projection {
        projection.inverse(x, y).map_err(map_err)?
    } else if src_code == Some(4326) {
        (x, y)
    } else if let Some(code) = src_code.filter(|code| epsg_projection(*code).is_some()) {
        epsg_inverse(code, x, y)
            .expect("code checked supported")
            .map_err(map_err)?
    } else {
        return unsupported(src, dst);
    };
    if let Some(projection) = dst.projection {
        projection.forward(lon, lat).map_err(map_err)
    } else if dst_code == Some(4326) {
        Ok((lon, lat))
    } else if let Some(code) = dst_code.filter(|code| epsg_projection(*code).is_some()) {
        epsg_forward(code, lon, lat)
            .expect("code checked supported")
            .map_err(map_err)
    } else {
        unsupported(src, dst)
    }
}

pub fn web_mercator_bounds(bounds: RasterBounds, src: &CrsSpec) -> GisResult<RasterBounds> {
    let src_epsg = epsg_code(src);
    validate_bounds_tuple(bounds.tuple(), src_epsg == Some(4326))?;
    if src_epsg == Some(4326) {
        if bounds.left > bounds.right {
            return Err(GisError::InvalidBounds(
                "antimeridian_bounds_unsupported: split antimeridian bounds before transforming"
                    .to_string(),
            ));
        }
        if bounds.bottom < -WEB_MERCATOR_MAX_LAT || bounds.top > WEB_MERCATOR_MAX_LAT {
            return Err(GisError::InvalidBounds(format!(
                "invalid_latitude_range: Web Mercator supports latitudes within +/-{WEB_MERCATOR_MAX_LAT}"
            )));
        }
    }
    let dst = CrsSpec::from_string("EPSG:3857".to_string())?;
    transform_bounds(bounds, src, &dst)
}

pub fn parse_crs_string(value: String) -> GisResult<CrsSpec> {
    if let Some(code) = parse_epsg_code(&value) {
        return CrsSpec::from_string(format!("EPSG:{code}"));
    }
    CrsSpec::from_string(value)
}

fn parse_epsg_code(crs: &str) -> Option<u32> {
    let trimmed = crs.trim();
    let (authority, code) = trimmed.split_once(':')?;
    if authority.eq_ignore_ascii_case("EPSG") {
        code.parse::<u32>().ok()
    } else {
        None
    }
}

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use pyo3::types::PyAny;

#[cfg_attr(
    feature = "extension-module",
    pyo3::pyclass(module = "forge3d._forge3d", name = "CrsTransform")
)]
#[derive(Debug, Clone)]
pub struct CrsTransform {
    pub src: CrsSpec,
    pub dst: CrsSpec,
    pub axis_order_policy: String,
}

impl CrsTransform {
    pub fn new(src: CrsSpec, dst: CrsSpec, always_xy: bool) -> GisResult<Self> {
        // Check dispatchability once so unsupported pairs fail at creation,
        // not first use. (A numeric probe point would false-negative on
        // domain limits, e.g. UTM-south coordinates outside Web Mercator's
        // latitude range.)
        transform_pair_supported(&src, &dst)?;
        Ok(Self {
            src,
            dst,
            axis_order_policy: if always_xy { "always_xy" } else { "native" }.to_string(),
        })
    }
}

#[cfg(feature = "extension-module")]
#[pyo3::pymethods]
impl CrsTransform {
    #[staticmethod]
    #[pyo3(signature = (src_crs, dst_crs, *, always_xy = true))]
    fn from_crs(
        src_crs: &Bound<'_, PyAny>,
        dst_crs: &Bound<'_, PyAny>,
        always_xy: bool,
    ) -> PyResult<Self> {
        Self::new(
            crate::gis::extract_required_crs(Some(src_crs))?,
            crate::gis::extract_required_crs(Some(dst_crs))?,
            always_xy,
        )
        .map_err(Into::into)
    }

    #[getter]
    fn src_crs(&self) -> PyResult<String> {
        canonical_label(&self.src).map_err(Into::into)
    }

    #[getter]
    fn dst_crs(&self) -> PyResult<String> {
        canonical_label(&self.dst).map_err(Into::into)
    }

    #[getter]
    fn src_authority(&self) -> Option<HashMap<String, String>> {
        authority_map(&self.src)
    }

    #[getter]
    fn dst_authority(&self) -> Option<HashMap<String, String>> {
        authority_map(&self.dst)
    }

    #[getter]
    fn axis_order_policy(&self) -> String {
        self.axis_order_policy.clone()
    }

    fn transform_point(&self, x: f64, y: f64) -> PyResult<(f64, f64)> {
        transform_point(x, y, &self.src, &self.dst).map_err(Into::into)
    }

    fn transform_point3(&self, x: f64, y: f64, z: f64) -> PyResult<(f64, f64, f64)> {
        transform_point3(x, y, z, &self.src, &self.dst).map_err(Into::into)
    }

    fn transform_bounds(&self, bounds: (f64, f64, f64, f64)) -> PyResult<(f64, f64, f64, f64)> {
        let bounds = validate_bounds_tuple(bounds, false)?;
        transform_bounds(bounds, &self.src, &self.dst)
            .map(|bounds| bounds.tuple())
            .map_err(Into::into)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "CrsTransform(src_crs={:?}, dst_crs={:?}, axis_order_policy={:?})",
            canonical_label(&self.src)?,
            canonical_label(&self.dst)?,
            self.axis_order_policy
        ))
    }
}
