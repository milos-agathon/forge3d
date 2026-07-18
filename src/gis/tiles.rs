#![cfg_attr(not(feature = "extension-module"), allow(dead_code))]

use std::f64::consts::PI;

use crate::gis::crs;
use crate::gis::error::{GisError, GisResult};
use crate::gis::raster_write::CrsSpec;
use crate::gis::types::{RasterBounds, RasterWarning};

const WEB_MERCATOR_RADIUS: f64 = 6_378_137.0;
const WEB_MERCATOR_MAX_LAT: f64 = 85.051_128_78;

#[derive(Debug, Clone)]
pub struct SlippyTile {
    pub z: u8,
    pub x: u32,
    pub y: u32,
    pub bounds_wgs84: RasterBounds,
    pub bounds_web_mercator: RasterBounds,
}

#[derive(Debug, Clone)]
pub struct SlippyTileIndex {
    pub zoom: u8,
    pub crs: String,
    pub bounds_wgs84: RasterBounds,
    pub antimeridian_split: bool,
    pub tiles: Vec<SlippyTile>,
    pub warnings: Vec<RasterWarning>,
}

pub(crate) fn slippy_tiles(
    bounds: RasterBounds,
    zoom: i64,
    crs_spec: &CrsSpec,
) -> GisResult<SlippyTileIndex> {
    if !(0..=24).contains(&zoom) {
        return Err(GisError::InvalidArgument(
            "invalid_argument: zoom must be an integer in 0..24".to_string(),
        ));
    }
    let zoom = zoom as u8;
    let mut warnings = Vec::new();
    let mut bounds_wgs84 = to_wgs84(bounds, crs_spec)?;
    validate_latitude(bounds_wgs84)?;

    if bounds_wgs84.bottom < -WEB_MERCATOR_MAX_LAT || bounds_wgs84.top > WEB_MERCATOR_MAX_LAT {
        bounds_wgs84.bottom = bounds_wgs84.bottom.max(-WEB_MERCATOR_MAX_LAT);
        bounds_wgs84.top = bounds_wgs84.top.min(WEB_MERCATOR_MAX_LAT);
        warnings.push(RasterWarning::new(
            "invalid_bounds",
            "latitude was clamped to the Web Mercator valid range",
            Some("bounds"),
        ));
        crate::core::degradation::record_degradation(
            "input_clamped",
            "web_mercator_latitude",
            "latitude bounds were clamped to the Web Mercator valid range",
        );
    }
    let antimeridian_split = bounds_wgs84.left > bounds_wgs84.right;
    let intervals = if antimeridian_split {
        vec![(bounds_wgs84.left, 180.0), (-180.0, bounds_wgs84.right)]
    } else {
        if bounds_wgs84.left >= bounds_wgs84.right || bounds_wgs84.bottom >= bounds_wgs84.top {
            return Err(GisError::InvalidBounds(
                "invalid_bounds: bounds must be ordered as (left, bottom, right, top)".to_string(),
            ));
        }
        vec![(bounds_wgs84.left, bounds_wgs84.right)]
    };

    let n = 1u32 << zoom;
    let y_min = lat_to_tile_y(bounds_wgs84.top, zoom);
    let y_max = lat_to_tile_y(bounds_wgs84.bottom, zoom);
    let mut tiles = Vec::new();
    for (left, right) in intervals {
        let x_min = lon_to_tile_x(left, zoom);
        let x_max = lon_to_tile_x(right, zoom);
        for x in x_min..=x_max {
            for y in y_min..=y_max {
                let x = x % n;
                let y = y.min(n - 1);
                let bounds_wgs84 = tile_bounds_wgs84(x, y, zoom);
                let bounds_web_mercator = tile_bounds_web_mercator(bounds_wgs84);
                if !tiles
                    .iter()
                    .any(|tile: &SlippyTile| tile.x == x && tile.y == y)
                {
                    tiles.push(SlippyTile {
                        z: zoom,
                        x,
                        y,
                        bounds_wgs84,
                        bounds_web_mercator,
                    });
                }
            }
        }
    }
    tiles.sort_by_key(|tile| (tile.z, tile.x, tile.y));
    Ok(SlippyTileIndex {
        zoom,
        crs: crs::canonical_label(crs_spec)?,
        bounds_wgs84,
        antimeridian_split,
        tiles,
        warnings,
    })
}

fn to_wgs84(bounds: RasterBounds, crs_spec: &CrsSpec) -> GisResult<RasterBounds> {
    match crs::epsg_code(crs_spec) {
        Some(4326) => Ok(bounds),
        Some(3857) => {
            let dst = CrsSpec::from_string("EPSG:4326".to_string())?;
            crs::transform_bounds(bounds, crs_spec, &dst)
        }
        _ => Err(GisError::BackendUnavailable(
            "backend_unavailable: proj feature required to transform tile bounds CRS".to_string(),
        )),
    }
}

fn validate_latitude(bounds: RasterBounds) -> GisResult<()> {
    if !bounds.left.is_finite()
        || !bounds.right.is_finite()
        || !bounds.bottom.is_finite()
        || !bounds.top.is_finite()
        || bounds.bottom < -90.0
        || bounds.top > 90.0
        || bounds.bottom >= bounds.top
    {
        return Err(GisError::InvalidBounds(
            "invalid_bounds: latitude must be finite and within [-90, 90]".to_string(),
        ));
    }
    Ok(())
}

fn lon_to_tile_x(lon: f64, zoom: u8) -> u32 {
    let n = (1u32 << zoom) as f64;
    (((lon + 180.0) / 360.0 * n).floor() as i64).clamp(0, n as i64 - 1) as u32
}

fn lat_to_tile_y(lat: f64, zoom: u8) -> u32 {
    let n = (1u32 << zoom) as f64;
    let lat_rad = lat.to_radians();
    let y = (1.0 - ((lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / PI)) / 2.0 * n;
    (y.floor() as i64).clamp(0, n as i64 - 1) as u32
}

pub(crate) fn tile_bounds_wgs84(x: u32, y: u32, z: u8) -> RasterBounds {
    let n = (1u32 << z) as f64;
    let lon_left = x as f64 / n * 360.0 - 180.0;
    let lon_right = (x + 1) as f64 / n * 360.0 - 180.0;
    let lat_top = tile_y_to_lat(y as f64, n);
    let lat_bottom = tile_y_to_lat((y + 1) as f64, n);
    RasterBounds {
        left: lon_left,
        bottom: lat_bottom,
        right: lon_right,
        top: lat_top,
    }
}

fn tile_y_to_lat(y: f64, n: f64) -> f64 {
    (PI - 2.0 * PI * y / n).sinh().atan().to_degrees()
}

fn tile_bounds_web_mercator(bounds: RasterBounds) -> RasterBounds {
    let (left, bottom) = lonlat_to_web(bounds.left, bounds.bottom);
    let (right, top) = lonlat_to_web(bounds.right, bounds.top);
    RasterBounds {
        left,
        bottom,
        right,
        top,
    }
}

fn lonlat_to_web(lon: f64, lat: f64) -> (f64, f64) {
    let lat = lat.clamp(-WEB_MERCATOR_MAX_LAT, WEB_MERCATOR_MAX_LAT);
    (
        WEB_MERCATOR_RADIUS * lon.to_radians(),
        WEB_MERCATOR_RADIUS * ((PI / 4.0 + lat.to_radians() / 2.0).tan()).ln(),
    )
}

#[cfg(feature = "extension-module")]
pub use py::slippy_tile_index_py;

#[cfg(feature = "extension-module")]
mod py {
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyDictMethods, PyList};
    use pyo3::IntoPy;

    use crate::gis::affine::validate_bounds_tuple;
    use crate::gis::error::GisError;
    use crate::gis::py_json::warnings_to_py;
    use crate::gis::raster_write::CrsSpec;
    use crate::gis::tiles::{slippy_tiles, SlippyTile};

    #[pyfunction(name = "slippy_tile_index", signature = (bounds, zoom, crs = "EPSG:4326"))]
    pub fn slippy_tile_index_py(
        py: Python<'_>,
        bounds: (f64, f64, f64, f64),
        zoom: i64,
        crs: &str,
    ) -> PyResult<PyObject> {
        let crs_spec = CrsSpec::from_string(crs.to_string())?;
        let bounds =
            validate_bounds_tuple(bounds, crate::gis::crs::epsg_code(&crs_spec) == Some(4326))
                .or_else(|err| {
                    if bounds.0 > bounds.2 && crate::gis::crs::epsg_code(&crs_spec) == Some(4326) {
                        Ok(crate::gis::types::RasterBounds {
                            left: bounds.0,
                            bottom: bounds.1,
                            right: bounds.2,
                            top: bounds.3,
                        })
                    } else {
                        Err(err)
                    }
                })?;
        if bounds.left == bounds.right {
            return Err(GisError::InvalidBounds(
                "invalid_bounds: longitude interval must have non-zero width".to_string(),
            )
            .into());
        }
        let result = slippy_tiles(bounds, zoom, &crs_spec)?;
        let dict = PyDict::new_bound(py);
        dict.set_item("zoom", result.zoom)?;
        dict.set_item("crs", result.crs)?;
        dict.set_item("bounds_wgs84", result.bounds_wgs84.tuple())?;
        dict.set_item("antimeridian_split", result.antimeridian_split)?;
        dict.set_item("tile_count", result.tiles.len())?;
        dict.set_item("tiles", tiles_to_py(py, &result.tiles)?)?;
        dict.set_item("warnings", warnings_to_py(py, &result.warnings)?)?;
        Ok(dict.into_py(py))
    }

    fn tiles_to_py(py: Python<'_>, tiles: &[SlippyTile]) -> PyResult<PyObject> {
        let items = tiles
            .iter()
            .map(|tile| {
                let dict = PyDict::new_bound(py);
                dict.set_item("z", tile.z)?;
                dict.set_item("x", tile.x)?;
                dict.set_item("y", tile.y)?;
                dict.set_item("bounds_wgs84", tile.bounds_wgs84.tuple())?;
                dict.set_item("bounds_web_mercator", tile.bounds_web_mercator.tuple())?;
                Ok(dict.into_py(py))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new_bound(py, items).into_py(py))
    }
}
