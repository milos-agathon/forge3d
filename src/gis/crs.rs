use std::collections::HashMap;
use std::f64::consts::PI;

use crate::gis::affine::validate_bounds_tuple;
use crate::gis::error::{GisError, GisResult};
use crate::gis::raster_write::CrsSpec;
use crate::gis::types::{RasterBounds, RasterInfo, RasterWarning, WARNING_MISSING_CRS};

const WEB_MERCATOR_RADIUS: f64 = 6_378_137.0;
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
    if crs_equal(src, dst) {
        return Ok(bounds);
    }
    let corners = [
        transform_point(bounds.left, bounds.bottom, src, dst)?,
        transform_point(bounds.left, bounds.top, src, dst)?,
        transform_point(bounds.right, bounds.bottom, src, dst)?,
        transform_point(bounds.right, bounds.top, src, dst)?,
    ];
    let left = corners
        .iter()
        .map(|point| point.0)
        .fold(f64::INFINITY, f64::min);
    let right = corners
        .iter()
        .map(|point| point.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let bottom = corners
        .iter()
        .map(|point| point.1)
        .fold(f64::INFINITY, f64::min);
    let top = corners
        .iter()
        .map(|point| point.1)
        .fold(f64::NEG_INFINITY, f64::max);
    validate_bounds_tuple((left, bottom, right, top), false)
}

pub fn transform_point(x: f64, y: f64, src: &CrsSpec, dst: &CrsSpec) -> GisResult<(f64, f64)> {
    if !x.is_finite() || !y.is_finite() {
        return Err(GisError::TransformFailed(
            "coordinate values must be finite".to_string(),
        ));
    }
    if crs_equal(src, dst) {
        return Ok((x, y));
    }
    match (epsg_code(src), epsg_code(dst)) {
        (Some(4326), Some(3857)) => lonlat_to_web_mercator(x, y),
        (Some(3857), Some(4326)) => web_mercator_to_lonlat(x, y),
        _ => Err(GisError::BackendUnavailable(format!(
            "BackendUnavailable: CRS transform {} to {} requires an unavailable PROJ backend",
            canonical_label(src)?,
            canonical_label(dst)?
        ))),
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

fn lonlat_to_web_mercator(lon: f64, lat: f64) -> GisResult<(f64, f64)> {
    if !(-WEB_MERCATOR_MAX_LAT..=WEB_MERCATOR_MAX_LAT).contains(&lat) {
        return Err(GisError::InvalidBounds(format!(
            "invalid_latitude_range: Web Mercator supports latitudes within +/-{WEB_MERCATOR_MAX_LAT}"
        )));
    }
    let x = WEB_MERCATOR_RADIUS * lon.to_radians();
    let y = WEB_MERCATOR_RADIUS * ((PI / 4.0 + lat.to_radians() / 2.0).tan()).ln();
    Ok((x, y))
}

fn web_mercator_to_lonlat(x: f64, y: f64) -> GisResult<(f64, f64)> {
    let lon = (x / WEB_MERCATOR_RADIUS).to_degrees();
    let lat = (2.0 * (y / WEB_MERCATOR_RADIUS).exp().atan() - PI / 2.0).to_degrees();
    Ok((lon, lat))
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
        // Probe once so unsupported pairs fail at creation, not first use.
        transform_point(0.0, 0.0, &src, &dst)?;
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
