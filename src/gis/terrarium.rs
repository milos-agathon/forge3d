use std::path::Path;

use crate::gis::error::{GisError, GisResult};
use crate::gis::raster_write::{RasterArray, RasterData};

/// Height-system tag carried by every Terrarium DEM ingestion result
/// (MENSURA): Terrarium tiles encode orthometric-style heights (AWS/Mapzen
/// terrain derives from EGM96-referenced sources), NOT ellipsoidal ones.
/// Converting to ellipsoidal requires `forge3d.crs.orthometric_to_ellipsoidal`.
pub const TERRARIUM_HEIGHT_SYSTEM: &str = "orthometric_egm96";

pub fn decode_terrarium_rgb(data: &[u8], height: usize, width: usize) -> GisResult<RasterArray> {
    let expected = height
        .checked_mul(width)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| {
            GisError::InvalidShape("shape_mismatch: Terrarium tile is too large".to_string())
        })?;
    if data.len() != expected {
        return Err(GisError::InvalidShape(
            "shape_mismatch: Terrarium input must be shaped (height, width, 3)".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(height * width);
    for pixel in data.chunks_exact(3) {
        let height_m =
            pixel[0] as f32 * 256.0 + pixel[1] as f32 + pixel[2] as f32 / 256.0 - 32768.0;
        out.push(height_m);
    }
    RasterArray::new(RasterData::F32(out), &[1, height, width])
}

pub fn decode_terrarium_png(path: &Path) -> GisResult<RasterArray> {
    let bytes = std::fs::read(path)?;
    decode_terrarium_png_bytes(&bytes)
}

pub fn decode_terrarium_png_bytes(bytes: &[u8]) -> GisResult<RasterArray> {
    let image = image::load_from_memory(&bytes)
        .map_err(|err| GisError::InvalidRaster(format!("malformed_payload: invalid PNG: {err}")))?
        .to_rgb8();
    let (width, height) = image.dimensions();
    decode_terrarium_rgb(image.as_raw(), height as usize, width as usize)
}

#[cfg(feature = "extension-module")]
pub use py::{build_terrarium_dem_py, decode_terrarium_dem_py};

#[cfg(feature = "extension-module")]
mod py {
    use std::path::PathBuf;

    use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
    use pyo3::prelude::*;
    use pyo3::types::{PyAny, PyDict, PyDictMethods, PyList};
    use pyo3::IntoPy;

    use crate::gis::affine::validate_bounds_tuple;
    use crate::gis::error::GisError;
    use crate::gis::py_json::warnings_to_py;
    use crate::gis::raster_info::raster_to_f64;
    use crate::gis::raster_write::CrsSpec;
    use crate::gis::terrarium::{decode_terrarium_png, decode_terrarium_rgb};
    use crate::gis::tiles::slippy_tiles;
    use crate::gis::types::{RasterBounds, RasterInfo, RasterWarning};

    #[pyfunction(name = "decode_terrarium_dem")]
    pub fn decode_terrarium_dem_py(
        py: Python<'_>,
        rgb_array_or_path: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let array = if let Ok(path) = rgb_array_or_path.extract::<String>() {
            decode_terrarium_png(PathBuf::from(path).as_path())?
        } else {
            let array: PyReadonlyArrayDyn<'_, u8> = rgb_array_or_path.extract().map_err(|_| {
                GisError::UnsupportedDType(
                    "unsupported_dtype: Terrarium input must be uint8 ndarray or PNG path"
                        .to_string(),
                )
            })?;
            let shape = array.shape();
            if shape.len() != 3 || shape[2] != 3 {
                return Err(GisError::InvalidShape(
                    "shape_mismatch: Terrarium input must be shaped (height, width, 3)".to_string(),
                )
                .into());
            }
            let data = array.as_array().iter().copied().collect::<Vec<_>>();
            decode_terrarium_rgb(&data, shape[0], shape[1])?
        };
        terrarium_result_to_py(py, &array, Vec::new())
    }

    #[pyfunction(name = "build_terrarium_dem", signature = (bounds, zoom, cache = None, *, url_template = None, timeout = None))]
    pub fn build_terrarium_dem_py(
        py: Python<'_>,
        bounds: (f64, f64, f64, f64),
        zoom: i64,
        cache: Option<&Bound<'_, PyAny>>,
        url_template: Option<String>,
        timeout: Option<f64>,
    ) -> PyResult<PyObject> {
        let (cache_dir, cached_template) = cache_policy(cache)?;
        let url_template = url_template.or(cached_template);
        if cache_dir.is_none() && url_template.is_none() {
            return Err(GisError::InvalidRaster(
                "cache_miss: build_terrarium_dem requires cache_dir with explicit cached tiles or url_template".to_string(),
            )
            .into());
        }
        if let Some(template) = url_template.as_deref() {
            for placeholder in ["{z}", "{x}", "{y}"] {
                if !template.contains(placeholder) {
                    return Err(GisError::InvalidArgument(format!(
                        "invalid_argument: Terrarium url_template is missing {placeholder}"
                    ))
                    .into());
                }
            }
        }
        let bounds = validate_bounds_tuple(bounds, true)?;
        let crs = CrsSpec::from_string("EPSG:4326".to_string())?;
        let index = slippy_tiles(bounds, zoom, &crs)?;
        let mut decoded = Vec::new();
        let mut manifest = Vec::new();
        let mut warnings = Vec::new();
        for tile in &index.tiles {
            let path = cache_dir.as_ref().map(|cache_dir| {
                cache_dir
                    .join(tile.z.to_string())
                    .join(tile.x.to_string())
                    .join(format!("{}.png", tile.y))
            });
            if path.as_ref().is_some_and(|path| path.exists()) {
                let path = path.as_ref().expect("checked above");
                let array = decode_terrarium_png(&path)?;
                manifest.push((tile.z, tile.x, tile.y, path.clone(), "hit".to_string()));
                decoded.push((tile.x, tile.y, array));
            } else if let Some(template) = url_template.as_deref() {
                let url = template
                    .replace("{z}", &tile.z.to_string())
                    .replace("{x}", &tile.x.to_string())
                    .replace("{y}", &tile.y.to_string());
                let fetched = py.allow_threads(|| {
                    crate::gis::remote::fetch_remote_geodata_payload(&url, None, timeout)
                });
                match fetched {
                    Ok((bytes, _remote)) => {
                        let array = super::decode_terrarium_png_bytes(&bytes)?;
                        if let Some(path) = path.as_ref() {
                            if let Some(parent) = path.parent() {
                                std::fs::create_dir_all(parent)?;
                            }
                            let temp = path.with_extension("png.tmp");
                            std::fs::write(&temp, &bytes)?;
                            std::fs::rename(&temp, path)?;
                        }
                        manifest.push((
                            tile.z,
                            tile.x,
                            tile.y,
                            path.unwrap_or_else(|| PathBuf::from(url)),
                            "fetched".to_string(),
                        ));
                        decoded.push((tile.x, tile.y, array));
                    }
                    Err(error) => {
                        warnings.push(RasterWarning::new(
                            "missing_tile",
                            format!("missing_tile: failed to fetch {url}: {error}"),
                            Some("tiles"),
                        ));
                        manifest.push((
                            tile.z,
                            tile.x,
                            tile.y,
                            path.unwrap_or_else(|| PathBuf::from(url)),
                            "missing_tile".to_string(),
                        ));
                    }
                }
            } else {
                manifest.push((
                    tile.z,
                    tile.x,
                    tile.y,
                    path.expect("cache_dir exists when no template"),
                    "missing_tile".to_string(),
                ));
            }
        }
        if decoded.is_empty() {
            return Err(GisError::InvalidRaster(
                "missing_tile: no cached Terrarium tiles were available".to_string(),
            )
            .into());
        }
        if decoded.len() < index.tiles.len() {
            warnings.push(RasterWarning::new(
                "partial_mosaic",
                "partial_mosaic: at least one Terrarium tile was missing",
                Some("tiles"),
            ));
        }
        let array = mosaic(decoded)?;
        let dict = PyDict::new_bound(py);
        dict.set_item("array", super::super::raster_array_to_py(py, &array)?)?;
        dict.set_item(
            "info",
            super::super::raster_info_to_py_dict(py, &terrarium_info(&array, bounds))?,
        )?;
        dict.set_item(
            "height_system",
            crate::gis::terrarium::TERRARIUM_HEIGHT_SYSTEM,
        )?;
        dict.set_item("tile_count", index.tiles.len())?;
        dict.set_item("manifest", manifest_to_py(py, manifest)?)?;
        dict.set_item("warnings", warnings_to_py(py, &warnings)?)?;
        Ok(dict.into_py(py))
    }

    fn cache_policy(
        cache: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Option<PathBuf>, Option<String>)> {
        let Some(cache) = cache else {
            return Ok((None, None));
        };
        if cache.is_none() {
            return Ok((None, None));
        }
        if let Ok(path) = cache.extract::<String>() {
            return Ok((Some(PathBuf::from(path)), None));
        }
        let dict = cache.downcast::<PyDict>().map_err(|_| {
            GisError::InvalidArgument(
                "invalid_argument: cache must be None, path, or dict with cache_dir".to_string(),
            )
        })?;
        let cache_dir = dict
            .get_item("cache_dir")?
            .map(|value| value.extract::<String>().map(PathBuf::from))
            .transpose()?;
        let url_template = dict
            .get_item("url_template")?
            .map(|value| value.extract::<String>())
            .transpose()?;
        Ok((cache_dir, url_template))
    }

    fn mosaic(
        mut decoded: Vec<(u32, u32, crate::gis::raster_write::RasterArray)>,
    ) -> PyResult<crate::gis::raster_write::RasterArray> {
        decoded.sort_by_key(|(x, y, _)| (*y, *x));
        let min_x = decoded.iter().map(|(x, _, _)| *x).min().unwrap();
        let min_y = decoded.iter().map(|(_, y, _)| *y).min().unwrap();
        let max_x = decoded.iter().map(|(x, _, _)| *x).max().unwrap();
        let max_y = decoded.iter().map(|(_, y, _)| *y).max().unwrap();
        let tile_width = decoded[0].2.width;
        let tile_height = decoded[0].2.height;
        let width = (max_x - min_x + 1) as usize * tile_width;
        let height = (max_y - min_y + 1) as usize * tile_height;
        let mut out = vec![f32::NAN; width * height];
        for (x, y, array) in decoded {
            let values = raster_to_f64(&array);
            let x0 = (x - min_x) as usize * tile_width;
            let y0 = (y - min_y) as usize * tile_height;
            for row in 0..tile_height {
                for col in 0..tile_width {
                    out[(y0 + row) * width + x0 + col] = values[row * tile_width + col] as f32;
                }
            }
        }
        crate::gis::raster_write::RasterArray::new(
            crate::gis::raster_write::RasterData::F32(out),
            &[1, height, width],
        )
        .map_err(Into::into)
    }

    fn terrarium_result_to_py(
        py: Python<'_>,
        array: &crate::gis::raster_write::RasterArray,
        warnings: Vec<RasterWarning>,
    ) -> PyResult<PyObject> {
        let values = raster_to_f64(array);
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let dict = PyDict::new_bound(py);
        dict.set_item("array", super::super::raster_array_to_py(py, array)?)?;
        dict.set_item(
            "info",
            super::super::raster_info_to_py_dict(
                py,
                &terrarium_info(
                    array,
                    RasterBounds {
                        left: 0.0,
                        bottom: 0.0,
                        right: array.width as f64,
                        top: array.height as f64,
                    },
                ),
            )?,
        )?;
        dict.set_item("encoding", "terrarium")?;
        dict.set_item(
            "height_system",
            crate::gis::terrarium::TERRARIUM_HEIGHT_SYSTEM,
        )?;
        dict.set_item("nodata", py.None())?;
        dict.set_item("valid_count", array.width * array.height)?;
        dict.set_item("min", min)?;
        dict.set_item("max", max)?;
        dict.set_item("warnings", warnings_to_py(py, &warnings)?)?;
        Ok(dict.into_py(py))
    }

    fn terrarium_info(
        array: &crate::gis::raster_write::RasterArray,
        bounds: RasterBounds,
    ) -> RasterInfo {
        let mut info = RasterInfo::new("".into(), array.width as u32, array.height as u32, 1);
        info.driver = "memory".to_string();
        info.dtype_per_band = vec!["float32".to_string()];
        info.crs_authority = Some(
            [
                ("name".to_string(), "EPSG".to_string()),
                ("code".to_string(), "4326".to_string()),
            ]
            .into_iter()
            .collect(),
        );
        info.transform = Some((
            (bounds.right - bounds.left) / array.width as f64,
            0.0,
            bounds.left,
            0.0,
            -((bounds.top - bounds.bottom) / array.height as f64),
            bounds.top,
        ));
        info.bounds = Some(bounds.tuple());
        info.resolution = Some((
            (bounds.right - bounds.left).abs() / array.width as f64,
            (bounds.top - bounds.bottom).abs() / array.height as f64,
        ));
        info.nodata_per_band = vec![None];
        info.is_georeferenced = true;
        // Terrarium tiles encode orthometric (EGM96-ish) elevations; persist it
        // on the RasterInfo, not only the sidecar dict key.
        info.height_system = crate::gis::terrarium::TERRARIUM_HEIGHT_SYSTEM.to_string();
        info
    }

    fn manifest_to_py(
        py: Python<'_>,
        rows: Vec<(u8, u32, u32, PathBuf, String)>,
    ) -> PyResult<PyObject> {
        let items = rows
            .into_iter()
            .map(|(z, x, y, path, status)| {
                let dict = PyDict::new_bound(py);
                dict.set_item("z", z)?;
                dict.set_item("x", x)?;
                dict.set_item("y", y)?;
                dict.set_item("path", path.to_string_lossy().to_string())?;
                dict.set_item("status", status)?;
                Ok(dict.into_py(py))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new_bound(py, items).into_py(py))
    }
}
