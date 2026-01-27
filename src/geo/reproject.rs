// src/geo/reproject.rs
// CRS reprojection using the PROJ library (feature-gated)
// RELEVANT FILES: src/geo/mod.rs, python/forge3d/crs.py

use thiserror::Error;

/// Error type for geographic operations
#[derive(Error, Debug)]
pub enum GeoError {
    #[error("Projection error: {0}")]
    Projection(String),

    #[error("Invalid CRS: {0}")]
    InvalidCrs(String),

    #[error("PROJ feature not enabled")]
    ProjNotAvailable,
}

/// Reproject an array of 2D coordinates from one CRS to another.
///
/// # Arguments
/// * `coords` - Array of [lon, lat] or [x, y] coordinate pairs
/// * `from_crs` - Source CRS as EPSG code (e.g., "EPSG:4326") or PROJ string
/// * `to_crs` - Target CRS as EPSG code or PROJ string
///
/// # Returns
/// * `Ok(Vec<[f64; 2]>)` - Reprojected coordinates
/// * `Err(GeoError)` - If reprojection fails
///
/// # Example
/// ```ignore
/// let wgs84_coords = vec![[138.73, 35.36], [138.74, 35.37]];
/// let utm_coords = reproject_coords(&wgs84_coords, "EPSG:4326", "EPSG:32654")?;
/// ```
#[cfg(feature = "proj")]
pub fn reproject_coords(
    coords: &[[f64; 2]],
    from_crs: &str,
    to_crs: &str,
) -> Result<Vec<[f64; 2]>, GeoError> {
    use proj::Proj;

    // Create the transformation
    let transformer = Proj::new_known_crs(from_crs, to_crs, None)
        .map_err(|e| GeoError::Projection(format!("Failed to create transform: {}", e)))?;

    // Transform each coordinate
    let mut result = Vec::with_capacity(coords.len());
    for coord in coords {
        let (x, y) = transformer
            .convert((coord[0], coord[1]))
            .map_err(|e| GeoError::Projection(format!("Transform failed at ({}, {}): {}", coord[0], coord[1], e)))?;
        result.push([x, y]);
    }

    Ok(result)
}

/// Reproject a single coordinate pair.
#[cfg(feature = "proj")]
pub fn reproject_point(
    x: f64,
    y: f64,
    from_crs: &str,
    to_crs: &str,
) -> Result<(f64, f64), GeoError> {
    use proj::Proj;

    let transformer = Proj::new_known_crs(from_crs, to_crs, None)
        .map_err(|e| GeoError::Projection(format!("Failed to create transform: {}", e)))?;

    transformer
        .convert((x, y))
        .map_err(|e| GeoError::Projection(format!("Transform failed: {}", e)))
}

/// Parse a CRS string and validate it.
/// Accepts EPSG codes (e.g., "EPSG:4326") or PROJ strings.
#[cfg(feature = "proj")]
pub fn validate_crs(crs: &str) -> Result<(), GeoError> {
    use proj::Proj;

    // Try to create a CRS object to validate
    // We use WGS84 as a dummy target just to check if the source CRS is valid
    Proj::new_known_crs(crs, "EPSG:4326", None)
        .map(|_| ())
        .map_err(|e| GeoError::InvalidCrs(format!("{}: {}", crs, e)))
}

/// Get the EPSG code from a CRS string if it's in EPSG format.
pub fn parse_epsg_code(crs: &str) -> Option<u32> {
    let crs_upper = crs.to_uppercase();
    if crs_upper.starts_with("EPSG:") {
        crs_upper[5..].parse::<u32>().ok()
    } else {
        None
    }
}

/// Check if two CRS strings refer to the same coordinate system.
#[cfg(feature = "proj")]
pub fn crs_equal(crs1: &str, crs2: &str) -> bool {
    // Quick check for identical strings
    if crs1 == crs2 {
        return true;
    }

    // Check EPSG codes
    let code1 = parse_epsg_code(crs1);
    let code2 = parse_epsg_code(crs2);
    if let (Some(c1), Some(c2)) = (code1, code2) {
        return c1 == c2;
    }

    // More complex comparison would require PROJ authority lookup
    // For now, fall back to string comparison
    false
}

#[cfg(not(feature = "proj"))]
pub fn crs_equal(crs1: &str, crs2: &str) -> bool {
    crs1 == crs2
}

/// Stub for when proj feature is disabled
#[cfg(not(feature = "proj"))]
pub fn reproject_coords(
    _coords: &[[f64; 2]],
    _from_crs: &str,
    _to_crs: &str,
) -> Result<Vec<[f64; 2]>, GeoError> {
    Err(GeoError::ProjNotAvailable)
}

#[cfg(not(feature = "proj"))]
pub fn reproject_point(
    _x: f64,
    _y: f64,
    _from_crs: &str,
    _to_crs: &str,
) -> Result<(f64, f64), GeoError> {
    Err(GeoError::ProjNotAvailable)
}

#[cfg(not(feature = "proj"))]
pub fn validate_crs(_crs: &str) -> Result<(), GeoError> {
    Err(GeoError::ProjNotAvailable)
}

// -----------------------------------------------------------------------------
// Python bindings
// -----------------------------------------------------------------------------

/// Check if PROJ feature is available (for Python).
#[cfg(feature = "extension-module")]
#[pyo3::pyfunction]
#[pyo3(name = "proj_available")]
pub fn proj_available_py() -> bool {
    cfg!(feature = "proj")
}

/// Reproject coordinates from one CRS to another (Python binding).
#[cfg(feature = "extension-module")]
#[pyo3::pyfunction]
#[pyo3(name = "reproject_coords")]
pub fn reproject_coords_py(
    coords: Vec<[f64; 2]>,
    from_crs: &str,
    to_crs: &str,
) -> pyo3::PyResult<Vec<[f64; 2]>> {
    reproject_coords(&coords, from_crs, to_crs)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_epsg_code() {
        assert_eq!(parse_epsg_code("EPSG:4326"), Some(4326));
        assert_eq!(parse_epsg_code("epsg:32654"), Some(32654));
        assert_eq!(parse_epsg_code("WGS84"), None);
        assert_eq!(parse_epsg_code("EPSG:invalid"), None);
    }

    #[cfg(feature = "proj")]
    #[test]
    fn test_reproject_wgs84_to_utm() {
        // Mt. Fuji approximate location
        let wgs84 = vec![[138.7274, 35.3606]];
        let utm = reproject_coords(&wgs84, "EPSG:4326", "EPSG:32654").unwrap();

        // UTM zone 54N coordinates should be roughly in the range
        assert!(utm[0][0] > 300_000.0 && utm[0][0] < 500_000.0);
        assert!(utm[0][1] > 3_900_000.0 && utm[0][1] < 4_000_000.0);
    }

    #[cfg(feature = "proj")]
    #[test]
    fn test_roundtrip() {
        let original = vec![[138.7274, 35.3606]];
        let utm = reproject_coords(&original, "EPSG:4326", "EPSG:32654").unwrap();
        let back = reproject_coords(&utm, "EPSG:32654", "EPSG:4326").unwrap();

        // Should round-trip within tolerance
        let diff_lon = (back[0][0] - original[0][0]).abs();
        let diff_lat = (back[0][1] - original[0][1]).abs();
        assert!(diff_lon < 1e-6, "Longitude diff {} > 1e-6", diff_lon);
        assert!(diff_lat < 1e-6, "Latitude diff {} > 1e-6", diff_lat);
    }
}
