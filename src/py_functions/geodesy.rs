// src/py_functions/geodesy.rs
// MENSURA Python surface: EGM96 geoid undulation, Karney geodesics, and the
// full-f64 geodetic ⇄ geocentric (ECEF) conversion.
// RELEVANT FILES: src/geo/geoid.rs, src/geo/geodesic.rs, src/geo/projections/geocentric.rs

use super::super::*;

#[cfg(feature = "extension-module")]
use numpy::{PyArray2, PyReadonlyArray2};

/// Return the registered datum constants for Earth, Moon, or Mars.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (name))]
pub(crate) fn body_info(py: Python<'_>, name: &str) -> PyResult<PyObject> {
    let body = crate::geo::body::body(name).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new_bound(py);
    dict.set_item("name", body.name)?;
    dict.set_item("semi_major_m", body.ellipsoid.a)?;
    dict.set_item("semi_minor_m", body.ellipsoid.b())?;
    dict.set_item("flattening", body.ellipsoid.f)?;
    dict.set_item("prime_meridian_w0_deg", body.prime_meridian_w0)?;
    dict.set_item("rotation_rate_deg_per_day", body.rotation_rate)?;
    dict.set_item(
        "gravity_surface",
        body.gravity_surface.map(|surface| surface.name()),
    )?;
    Ok(dict.into_py(py))
}

/// EGM96 geoid undulation N(lat, lon) in metres (degree/order 120 synthesis,
/// NGA F477 convention, WGS84 ellipsoid).
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (lat, lon))]
pub(crate) fn geoid_undulation(lat: f64, lon: f64) -> PyResult<f64> {
    if !(-90.0..=90.0).contains(&lat) || !lon.is_finite() {
        return Err(PyValueError::new_err(format!(
            "invalid_argument: latitude must be in [-90, 90] and longitude finite, got ({lat}, {lon})"
        )));
    }
    Ok(crate::geo::geoid::undulation_deg(lat, lon))
}

/// GMM3 Mars areoid undulation above its reference ellipsoid, metres.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (lat, lon))]
pub(crate) fn areoid_undulation(lat: f64, lon: f64) -> PyResult<f64> {
    if !(-90.0..=90.0).contains(&lat) || !lon.is_finite() {
        return Err(PyValueError::new_err(format!(
            "invalid_argument: latitude must be in [-90, 90] and longitude finite, got ({lat}, {lon})"
        )));
    }
    Ok(crate::geo::geoid::areoid_undulation_deg(lat, lon))
}

/// Convert an orthometric (EGM96) height to an ellipsoidal height:
/// h = H + N(lat, lon). Returns metres.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (h_orthometric, lat, lon))]
pub(crate) fn orthometric_to_ellipsoidal(h_orthometric: f64, lat: f64, lon: f64) -> PyResult<f64> {
    if !(-90.0..=90.0).contains(&lat) || !lon.is_finite() || !h_orthometric.is_finite() {
        return Err(PyValueError::new_err(
            "invalid_argument: height must be finite, latitude in [-90, 90], longitude finite"
                .to_string(),
        ));
    }
    use crate::geo::units::{Angle, Height};
    Ok(crate::geo::geoid::orthometric_to_ellipsoidal(
        Height::new(h_orthometric),
        Angle::new(lat),
        Angle::new(lon),
    )
    .metres())
}

/// Convert an ellipsoidal height to an orthometric (EGM96) height:
/// H = h − N(lat, lon). Returns metres.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (h_ellipsoidal, lat, lon))]
pub(crate) fn ellipsoidal_to_orthometric(h_ellipsoidal: f64, lat: f64, lon: f64) -> PyResult<f64> {
    if !(-90.0..=90.0).contains(&lat) || !lon.is_finite() || !h_ellipsoidal.is_finite() {
        return Err(PyValueError::new_err(
            "invalid_argument: height must be finite, latitude in [-90, 90], longitude finite"
                .to_string(),
        ));
    }
    use crate::geo::units::{Angle, Height};
    Ok(crate::geo::geoid::ellipsoidal_to_orthometric(
        Height::new(h_ellipsoidal),
        Angle::new(lat),
        Angle::new(lon),
    )
    .metres())
}

/// Karney inverse geodesic on a registered planetary ellipsoid: distance and
/// azimuths between two points. Returns
/// {"s12": m, "azi1": deg, "azi2": deg, "a12": deg}.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (lat1, lon1, lat2, lon2, *, body = "earth"))]
pub(crate) fn geodesic_inverse(
    py: Python<'_>,
    lat1: f64,
    lon1: f64,
    lat2: f64,
    lon2: f64,
    body: &str,
) -> PyResult<PyObject> {
    for (name, lat) in [("lat1", lat1), ("lat2", lat2)] {
        if !(-90.0..=90.0).contains(&lat) {
            return Err(PyValueError::new_err(format!(
                "invalid_argument: {name} must be in [-90, 90], got {lat}"
            )));
        }
    }
    if !lon1.is_finite() || !lon2.is_finite() {
        return Err(PyValueError::new_err(
            "invalid_argument: longitudes must be finite".to_string(),
        ));
    }
    let body = crate::geo::body::body(body).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let g = crate::geo::geodesic::Geodesic::new(&body.ellipsoid);
    let r = g.inverse(lat1, lon1, lat2, lon2);
    let dict = PyDict::new_bound(py);
    dict.set_item("s12", r.s12)?;
    dict.set_item("azi1", r.azi1)?;
    dict.set_item("azi2", r.azi2)?;
    dict.set_item("a12", r.a12)?;
    Ok(dict.into_py(py))
}

/// Karney direct geodesic on a registered planetary ellipsoid: destination
/// from start point, azimuth, and distance. Returns
/// {"lat2": deg, "lon2": deg, "azi2": deg, "a12": deg}.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (lat1, lon1, azi1, s12, *, body = "earth"))]
pub(crate) fn geodesic_direct(
    py: Python<'_>,
    lat1: f64,
    lon1: f64,
    azi1: f64,
    s12: f64,
    body: &str,
) -> PyResult<PyObject> {
    if !(-90.0..=90.0).contains(&lat1) || !lon1.is_finite() || !azi1.is_finite() || !s12.is_finite()
    {
        return Err(PyValueError::new_err(
            "invalid_argument: lat1 must be in [-90, 90]; lon1/azi1/s12 must be finite".to_string(),
        ));
    }
    let body = crate::geo::body::body(body).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let g = crate::geo::geodesic::Geodesic::new(&body.ellipsoid);
    let r = g.direct(lat1, lon1, azi1, s12);
    let dict = PyDict::new_bound(py);
    dict.set_item("lat2", r.lat2)?;
    dict.set_item("lon2", r.lon2)?;
    dict.set_item("azi2", r.azi2)?;
    dict.set_item("a12", r.a12)?;
    Ok(dict.into_py(py))
}

/// WGS84 geodetic (lon, lat in degrees, ELLIPSOIDAL height in metres) →
/// geocentric ECEF metres, full f64 (EPSG method 9602).
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (lon, lat, h = 0.0))]
pub(crate) fn wgs84_to_ecef(lon: f64, lat: f64, h: f64) -> PyResult<(f64, f64, f64)> {
    let v = crate::geo::projections::geocentric::wgs84_geodetic_to_ecef(lon, lat, h)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((v.x, v.y, v.z))
}

/// Geocentric ECEF metres → WGS84 geodetic (lon, lat degrees, ellipsoidal
/// height metres), full f64.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (x, y, z))]
pub(crate) fn ecef_to_wgs84(x: f64, y: f64, z: f64) -> PyResult<(f64, f64, f64)> {
    crate::geo::projections::geocentric::wgs84_ecef_to_geodetic(glam::DVec3::new(x, y, z))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Convert a DEM of orthometric (EGM96) heights to ellipsoidal heights by
/// adding N(lat, lon) per pixel. `bounds` is (left, bottom, right, top) in
/// EPSG:4326 degrees; pixel centres are sampled. Returns float64.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (dem, bounds))]
pub(crate) fn dem_orthometric_to_ellipsoidal<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    bounds: (f64, f64, f64, f64),
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (left, bottom, right, top) = bounds;
    if !left.is_finite()
        || !right.is_finite()
        || !(right > left && top > bottom)
        || !(-90.0..=90.0).contains(&bottom)
        || !(-90.0..=90.0).contains(&top)
    {
        return Err(PyValueError::new_err(format!(
            "invalid_bounds: expected (left, bottom, right, top) EPSG:4326 degrees, got {bounds:?}"
        )));
    }
    let arr = dem.as_array();
    let (rows, cols) = (arr.nrows(), arr.ncols());
    let mut out = ndarray::Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        let lat = top - (r as f64 + 0.5) * (top - bottom) / rows as f64;
        for c in 0..cols {
            let lon = left + (c as f64 + 0.5) * (right - left) / cols as f64;
            out[(r, c)] = arr[(r, c)] + crate::geo::geoid::undulation_deg(lat, lon);
        }
    }
    Ok(PyArray2::from_owned_array_bound(py, out))
}
