// src/py_functions/geodesy.rs
// MENSURA Python surface: EGM96 geoid undulation, Karney geodesics, and the
// full-f64 geodetic ⇄ geocentric (ECEF) conversion.
// RELEVANT FILES: src/geo/geoid.rs, src/geo/geodesic.rs, src/geo/projections/geocentric.rs

use super::super::*;

#[cfg(feature = "extension-module")]
use numpy::{PyArray2, PyReadonlyArray2};

/// NREL SPA topocentric solar vector.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (
    utc,
    lat,
    lon,
    elev_m = 0.0,
    *,
    tz_offset_hours = 0.0,
    delta_t_seconds = 69.0,
    pressure_mbar = 1013.25,
    temperature_c = 15.0
))]
pub(crate) fn solar_position(
    py: Python<'_>,
    utc: &Bound<'_, PyAny>,
    lat: f64,
    lon: f64,
    elev_m: f64,
    tz_offset_hours: f64,
    delta_t_seconds: f64,
    pressure_mbar: f64,
    temperature_c: f64,
) -> PyResult<PyObject> {
    let (year, month, day, hour, minute, second) = utc
        .extract::<(i32, u32, u32, u32, u32, f64)>()
        .map_err(|_| {
            PyValueError::new_err("utc must be (year, month, day, hour, minute, second)")
        })?;
    let vector = crate::geo::solar::solar_position(&crate::geo::solar::SolarTime {
        year,
        month,
        day,
        hour,
        minute,
        second,
        tz_offset_hours,
        delta_t_seconds,
        latitude_deg: lat,
        longitude_deg: lon,
        elevation_m: elev_m,
        pressure_mbar,
        temperature_c,
    })
    .map_err(PyValueError::new_err)?;
    let dict = PyDict::new_bound(py);
    dict.set_item("zenith_deg", vector.zenith_deg)?;
    dict.set_item("azimuth_deg", vector.azimuth_deg)?;
    dict.set_item("apparent_elevation_deg", vector.apparent_elevation_deg)?;
    dict.set_item("true_elevation_deg", vector.true_elevation_deg)?;
    dict.set_item("distance_au", vector.distance_au)?;
    dict.set_item("equation_of_time_min", vector.equation_of_time_min)?;
    Ok(dict.into_py(py))
}

/// Curvature- and refraction-aware GPU viewshed over a north-up EPSG:4326 DEM.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (
    dem,
    observer,
    bounds,
    height_system,
    *,
    observer_height = 1.7,
    target_height = 0.0,
    max_distance = None,
    earth_model = "ellipsoid",
    sphere_radius_m = 6_371_008.8,
    refraction_model = "bennett",
    refraction_k = 0.13,
    pressure_mbar = 1013.25,
    temperature_c = 15.0
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn terrain_viewshed<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f32>,
    observer: (f64, f64),
    bounds: (f64, f64, f64, f64),
    height_system: &str,
    observer_height: f64,
    target_height: f64,
    max_distance: Option<f64>,
    earth_model: &str,
    sphere_radius_m: f64,
    refraction_model: &str,
    refraction_k: f64,
    pressure_mbar: f64,
    temperature_c: f64,
) -> PyResult<PyObject> {
    let (observer_lat, observer_lon) = observer;
    let (left, bottom, right, top) = bounds;
    let right_unwrapped = if right <= left { right + 360.0 } else { right };
    let longitude_span = right_unwrapped - left;
    let mut observer_lon_unwrapped = observer_lon;
    if observer_lon_unwrapped < left {
        observer_lon_unwrapped += 360.0;
    }
    if !(-90.0..=90.0).contains(&observer_lat)
        || !(-180.0..=180.0).contains(&observer_lon)
        || !(-180.0..=180.0).contains(&left)
        || !(-180.0..=180.0).contains(&right)
        || !(longitude_span > 0.0 && top > bottom)
        || longitude_span >= 180.0
        || top - bottom >= 180.0
        || !(-90.0..=90.0).contains(&bottom)
        || !(-90.0..=90.0).contains(&top)
        || observer_lon_unwrapped < left
        || observer_lon_unwrapped > right_unwrapped
        || observer_lat < bottom
        || observer_lat > top
    {
        return Err(PyValueError::new_err(
            "observer must be finite and inside local EPSG:4326 bounds spanning less than 180 degrees",
        ));
    }
    let array = dem.as_array();
    let (height, width) = (array.nrows(), array.ncols());
    if width < 2 || height < 2 || width > u32::MAX as usize || height > u32::MAX as usize {
        return Err(PyValueError::new_err(
            "dem must be a two-dimensional array at least 2x2",
        ));
    }
    let geodesic = crate::geo::geodesic::Geodesic::wgs84();
    let longitude_step = longitude_span / width as f64;
    let latitude_step = (top - bottom) / height as f64;
    let normalized_observer_lon = (observer_lon + 180.0).rem_euclid(360.0) - 180.0;
    let earth =
        crate::geo::refraction::EarthModel::from_name(earth_model, observer_lat, sphere_radius_m)
            .map_err(PyValueError::new_err)?;
    let mut positions_m = Vec::with_capacity(width * height);
    let mut farthest_m = 0.0_f64;
    for row in 0..height {
        let latitude = top - (row as f64 + 0.5) * latitude_step;
        for column in 0..width {
            let longitude_unwrapped = left + (column as f64 + 0.5) * longitude_step;
            let longitude = (longitude_unwrapped + 180.0).rem_euclid(360.0) - 180.0;
            let (distance_m, azimuth) = match earth {
                crate::geo::refraction::EarthModel::Sphere { radius_m } => {
                    let latitude_1 = observer_lat.to_radians();
                    let latitude_2 = latitude.to_radians();
                    let longitude_delta = (longitude - normalized_observer_lon).to_radians();
                    let central_angle = (latitude_1.sin() * latitude_2.sin()
                        + latitude_1.cos() * latitude_2.cos() * longitude_delta.cos())
                    .clamp(-1.0, 1.0)
                    .acos();
                    let azimuth = (longitude_delta.sin() * latitude_2.cos()).atan2(
                        latitude_1.cos() * latitude_2.sin()
                            - latitude_1.sin() * latitude_2.cos() * longitude_delta.cos(),
                    );
                    (radius_m * central_angle, azimuth)
                }
                _ => {
                    let inverse = geodesic.inverse(
                        observer_lat,
                        normalized_observer_lon,
                        latitude,
                        longitude,
                    );
                    (inverse.s12, inverse.azi1.to_radians())
                }
            };
            positions_m.push([
                (distance_m * azimuth.sin()) as f32,
                (distance_m * azimuth.cos()) as f32,
            ]);
            farthest_m = farthest_m.max(distance_m);
        }
    }
    let max_distance_m = max_distance.unwrap_or(farthest_m);
    let refraction = crate::geo::refraction::RefractionModel::from_name(
        refraction_model,
        pressure_mbar,
        temperature_c,
        refraction_k,
    )
    .map_err(PyValueError::new_err)?;
    let options = crate::terrain::analysis::viewshed::ViewshedOptions {
        width: width as u32,
        height: height as u32,
        observer_x: ((observer_lon_unwrapped - left) / longitude_step - 0.5) as f32,
        observer_y: ((top - observer_lat) / latitude_step - 0.5) as f32,
        observer_height_m: observer_height as f32,
        target_height_m: target_height as f32,
        max_distance_m: max_distance_m as f32,
        observer_latitude_rad: observer_lat.to_radians() as f32,
        observer_longitude_rad: normalized_observer_lon.to_radians() as f32,
        left_unwrapped_deg: left as f32,
        top_deg: top as f32,
        longitude_step_deg: longitude_step as f32,
        latitude_step_deg: latitude_step as f32,
        geodesic_sphere_radius_m: match earth {
            crate::geo::refraction::EarthModel::Sphere { radius_m } => radius_m as f32,
            _ => 0.0,
        },
        earth_model: earth,
        refraction_model: refraction,
    };
    let heights: Vec<f32> = match height_system {
        "ellipsoidal" => array.iter().copied().collect(),
        "orthometric_egm96" => {
            let mut converted = Vec::with_capacity(width * height);
            for row in 0..height {
                let latitude = top - (row as f64 + 0.5) * latitude_step;
                for column in 0..width {
                    let longitude_unwrapped =
                        left + (column as f64 + 0.5) * longitude_step;
                    let longitude = (longitude_unwrapped + 180.0).rem_euclid(360.0) - 180.0;
                    converted.push(
                        array[(row, column)]
                            + crate::geo::geoid::undulation_deg(latitude, longitude) as f32,
                    );
                }
            }
            converted
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported height_system {height_system:?}; expected 'ellipsoidal' or 'orthometric_egm96'"
            )))
        }
    };
    let output =
        crate::terrain::analysis::viewshed::compute_viewshed(&heights, &positions_m, &options)
            .map_err(|error| PyRuntimeError::new_err(format!("viewshed failed: {error}")))?;
    let shape = (height, width);
    let visibility = ndarray::Array2::from_shape_vec(
        shape,
        output.visibility.into_iter().map(u8::from).collect(),
    )
    .map_err(|error| PyRuntimeError::new_err(format!("viewshed shape failed: {error}")))?;
    let curvature = ndarray::Array2::from_shape_vec(shape, output.curvature_drop_m)
        .map_err(|error| PyRuntimeError::new_err(format!("viewshed shape failed: {error}")))?;
    let refraction = ndarray::Array2::from_shape_vec(shape, output.refraction_gain_m)
        .map_err(|error| PyRuntimeError::new_err(format!("viewshed shape failed: {error}")))?;
    let horizon = ndarray::Array2::from_shape_vec(shape, output.horizon_distance_m)
        .map_err(|error| PyRuntimeError::new_err(format!("viewshed shape failed: {error}")))?;
    let dict = PyDict::new_bound(py);
    dict.set_item(
        "visibility",
        PyArray2::from_owned_array_bound(py, visibility),
    )?;
    dict.set_item(
        "curvature_drop_m",
        PyArray2::from_owned_array_bound(py, curvature),
    )?;
    dict.set_item(
        "refraction_gain_m",
        PyArray2::from_owned_array_bound(py, refraction),
    )?;
    dict.set_item(
        "horizon_distance_m",
        PyArray2::from_owned_array_bound(py, horizon),
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

/// Karney inverse geodesic on WGS84: distance and azimuths between two
/// points. Returns {"s12": m, "azi1": deg, "azi2": deg, "a12": deg}.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (lat1, lon1, lat2, lon2))]
pub(crate) fn geodesic_inverse(
    py: Python<'_>,
    lat1: f64,
    lon1: f64,
    lat2: f64,
    lon2: f64,
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
    let g = crate::geo::geodesic::Geodesic::wgs84();
    let r = g.inverse(lat1, lon1, lat2, lon2);
    let dict = PyDict::new_bound(py);
    dict.set_item("s12", r.s12)?;
    dict.set_item("azi1", r.azi1)?;
    dict.set_item("azi2", r.azi2)?;
    dict.set_item("a12", r.a12)?;
    Ok(dict.into_py(py))
}

/// Karney direct geodesic on WGS84: destination from start point, azimuth,
/// and distance. Returns {"lat2": deg, "lon2": deg, "azi2": deg, "a12": deg}.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (lat1, lon1, azi1, s12))]
pub(crate) fn geodesic_direct(
    py: Python<'_>,
    lat1: f64,
    lon1: f64,
    azi1: f64,
    s12: f64,
) -> PyResult<PyObject> {
    if !(-90.0..=90.0).contains(&lat1) || !lon1.is_finite() || !azi1.is_finite() || !s12.is_finite()
    {
        return Err(PyValueError::new_err(
            "invalid_argument: lat1 must be in [-90, 90]; lon1/azi1/s12 must be finite".to_string(),
        ));
    }
    let g = crate::geo::geodesic::Geodesic::wgs84();
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
