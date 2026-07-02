#[cfg(feature = "extension-module")]
pub use py::{
    estimate_local_utm_py, extract_building_heights_py, load_building_footprints_py,
    load_context_vectors_py, prepare_dem_py, prepare_landcover_raster_py, prepare_osm_scene_py,
    prepare_population_raster_py, prepare_terrain_derivatives_py, read_cog_py,
    read_gridded_dataset_py, subset_grid_py,
};

#[cfg(feature = "extension-module")]
mod py {
    use std::collections::{BTreeMap, HashMap};
    use std::path::{Path, PathBuf};

    use pyo3::prelude::*;
    use pyo3::types::{PyAny, PyDict, PyDictMethods, PyList, PyTuple};
    use pyo3::IntoPy;
    use serde_json::{json, Map, Value};

    use crate::gis::affine::{validate_bounds_tuple, window_from_bounds, PixelWindow};
    use crate::gis::error::GisError;
    use crate::gis::osm::{parse_osm_features_value, query_osm_features_value};
    use crate::gis::py_json::{json_to_py, py_to_json, warnings_to_py};
    use crate::gis::raster_info::{self, copy_window, raster_to_f64, valid_mask};
    use crate::gis::raster_write::{RasterArray, RasterData};
    use crate::gis::types::{AffineTransform, RasterBounds, RasterInfo, RasterWarning};
    use crate::gis::vector::{read_vector, VectorReadOptions};
    use crate::gis::warp;

    struct RasterSource {
        array: RasterArray,
        info: RasterInfo,
    }

    #[pyfunction(name = "read_cog", signature = (path_or_url, window = None, overview = None))]
    pub fn read_cog_py(
        py: Python<'_>,
        path_or_url: String,
        window: Option<(i64, i64, u32, u32)>,
        overview: Option<i64>,
    ) -> PyResult<PyObject> {
        if path_or_url.starts_with("http://") || path_or_url.starts_with("https://") {
            return Err(GisError::BackendUnavailable(
                "backend_unavailable: cog_streaming feature required for remote COG reads"
                    .to_string(),
            )
            .into());
        }
        if overview.unwrap_or(0) != 0 {
            return Err(GisError::InvalidArgument(
                "metadata_unavailable: local TIFF reader has no overview selection support"
                    .to_string(),
            )
            .into());
        }
        let result = raster_info::read_raster(
            &path_or_url,
            None,
            window.map(|(col_off, row_off, width, height)| PixelWindow {
                col_off,
                row_off,
                width,
                height,
            }),
            false,
        )?;
        let dict = PyDict::new_bound(py);
        dict.set_item(
            "array",
            super::super::raster_array_to_py(py, &result.array)?,
        )?;
        dict.set_item(
            "info",
            super::super::raster_info_to_py_dict(py, &result.info)?,
        )?;
        dict.set_item(
            "window",
            result
                .window
                .map(|w| (w.col_off, w.row_off, w.width, w.height)),
        )?;
        dict.set_item("overview", py.None())?;
        dict.set_item(
            "is_cog_like",
            result.info.tiling.as_deref() == Some("tiled"),
        )?;
        let tile_info = PyDict::new_bound(py);
        tile_info.set_item("tiling", result.info.tiling.clone())?;
        tile_info.set_item("block_size", result.info.block_size.clone())?;
        tile_info.set_item("compression", result.info.compression.clone())?;
        dict.set_item("tile_info", tile_info)?;
        dict.set_item("warnings", warnings_to_py(py, &result.warnings)?)?;
        Ok(dict.into_py(py))
    }

    #[pyfunction(name = "load_context_vectors", signature = (path_or_features, layers = None))]
    pub fn load_context_vectors_py(
        py: Python<'_>,
        path_or_features: &Bound<'_, PyAny>,
        layers: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let requested = optional_string_list(layers)?;
        let layer_map = context_layers(path_or_features)?;
        let wanted = requested.unwrap_or_else(|| layer_map.keys().cloned().collect());
        let out = PyDict::new_bound(py);
        let mut total = 0usize;
        for name in wanted {
            let Some(value) = layer_map.get(&name) else {
                return Err(GisError::InvalidArgument(format!(
                    "missing_layer: requested context layer {name:?} was not found"
                ))
                .into());
            };
            let summary = layer_summary(py, value)?;
            total += summary.0;
            out.set_item(name, summary.1)?;
        }
        let dict = PyDict::new_bound(py);
        dict.set_item("layers", out)?;
        dict.set_item(
            "operation",
            operation_py(
                py,
                "load_context_vectors",
                layer_map.len(),
                total,
                false,
                Vec::new(),
            )?,
        )?;
        dict.set_item("warnings", Vec::<String>::new())?;
        Ok(dict.into_py(py))
    }

    #[pyfunction(name = "load_building_footprints", signature = (path_or_features, dst_crs = None))]
    pub fn load_building_footprints_py(
        py: Python<'_>,
        path_or_features: &Bound<'_, PyAny>,
        dst_crs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        if dst_crs.is_some() {
            return Err(GisError::BackendUnavailable(
                "backend_unavailable: proj feature required for building footprint reprojection"
                    .to_string(),
            )
            .into());
        }
        let value = input_geojson_or_cityjson(path_or_features)?;
        let features = if value.get("type").and_then(Value::as_str) == Some("CityJSON") {
            cityjson_building_features(&value)?
        } else {
            crate::gis::vector::normalize_features(&value)?
                .into_iter()
                .filter(|feature| feature.get("geometry").is_some_and(is_polygonal))
                .collect::<Vec<_>>()
        };
        if features.is_empty() {
            return Err(GisError::InvalidArgument(
                "empty_feature_set: no building footprints were found".to_string(),
            )
            .into());
        }
        let collection = feature_collection(features);
        let bounds = bounds_for_features(collection.get("features").unwrap().as_array().unwrap());
        let dict = PyDict::new_bound(py);
        dict.set_item("type", "FeatureCollection")?;
        dict.set_item(
            "features",
            json_to_py(py, collection.get("features").unwrap())?,
        )?;
        dict.set_item("feature_count", feature_count(&collection))?;
        dict.set_item("bounds", bounds.map(RasterBounds::tuple))?;
        dict.set_item(
            "crs",
            json_to_py(py, &json!({"name": "EPSG", "code": "4326"}))?,
        )?;
        dict.set_item(
            "operation",
            operation_py(
                py,
                "load_building_footprints",
                1,
                feature_count(&collection),
                false,
                Vec::new(),
            )?,
        )?;
        dict.set_item("warnings", Vec::<String>::new())?;
        Ok(dict.into_py(py))
    }

    #[pyfunction(name = "extract_building_heights", signature = (features, defaults = None))]
    pub fn extract_building_heights_py(
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        defaults: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let defaults = height_defaults(defaults)?;
        let value = py_to_json(features)?;
        let features = crate::gis::vector::normalize_features(&value)?;
        let mut heights = Vec::with_capacity(features.len());
        let mut attrs = Vec::with_capacity(features.len());
        let mut fallback_count = 0usize;
        let mut warnings = Vec::new();
        for feature in features {
            let props = feature
                .get("properties")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            match height_from_properties(&props, defaults) {
                Ok((height, attr, fallback)) => {
                    heights.push(height);
                    attrs.push(attr);
                    if fallback {
                        fallback_count += 1;
                    }
                }
                Err(message) => {
                    warnings.push(RasterWarning::new(
                        "invalid_height",
                        message,
                        Some("height"),
                    ));
                    warnings.push(RasterWarning::new(
                        "height_fallback",
                        "height_fallback: using default building height",
                        Some("height"),
                    ));
                    heights.push(defaults.height_m);
                    attrs.push("default".to_string());
                    fallback_count += 1;
                }
            }
        }
        let dict = PyDict::new_bound(py);
        dict.set_item("heights_m", heights)?;
        dict.set_item("attributes", attrs)?;
        dict.set_item("fallback_count", fallback_count)?;
        dict.set_item("warnings", warnings_to_py(py, &warnings)?)?;
        Ok(dict.into_py(py))
    }

    #[pyfunction(name = "prepare_dem", signature = (source, target_info = None, nodata = None))]
    pub fn prepare_dem_py(
        py: Python<'_>,
        source: &Bound<'_, PyAny>,
        target_info: Option<&Bound<'_, PyAny>>,
        nodata: Option<f64>,
    ) -> PyResult<PyObject> {
        let mut source = raster_source_from_py(source)?;
        if let Some(target) = target_info {
            let target = raster_info_from_py(target)?;
            if source.array.width != target.width as usize
                || source.array.height != target.height as usize
            {
                source.array = resize_nearest_f32(
                    &source.array,
                    target.width as usize,
                    target.height as usize,
                )?;
            }
            source.info.width = target.width;
            source.info.height = target.height;
            source.info.transform = target.transform;
            source.info.bounds = target.bounds;
            source.info.resolution = target.resolution;
            source.info.crs_wkt = target.crs_wkt;
            source.info.crs_authority = target.crs_authority;
        }
        let array = to_f32_array(&source.array)?;
        let nodata_per_band =
            vec![nodata.or_else(|| source.info.nodata_per_band.first().copied().flatten())];
        let mask = valid_mask(&array, &nodata_per_band, None);
        let valid_count = mask.iter().filter(|&&valid| valid).count();
        if valid_count == 0 {
            return Err(GisError::InvalidRaster(
                "empty_raster: DEM source has no valid pixels".to_string(),
            )
            .into());
        }
        let mut info = source.info.clone();
        info.dtype_per_band = vec!["float32".to_string()];
        info.nodata_per_band = nodata_per_band.clone();
        let dict = PyDict::new_bound(py);
        dict.set_item("array", super::super::raster_array_to_py(py, &array)?)?;
        dict.set_item("info", super::super::raster_info_to_py_dict(py, &info)?)?;
        dict.set_item("mask", super::super::bool_array_to_py(py, mask, &array)?)?;
        dict.set_item("mask_polarity", "true_valid")?;
        dict.set_item("valid_count", valid_count)?;
        dict.set_item("nodata_summary", nodata_per_band)?;
        dict.set_item("scale", json_to_py(py, &json!({"units": "meters"}))?)?;
        dict.set_item(
            "operation",
            operation_py(py, "prepare_dem", 1, 1, target_info.is_some(), Vec::new())?,
        )?;
        dict.set_item("warnings", warnings_to_py(py, &info.warnings)?)?;
        Ok(dict.into_py(py))
    }

    #[pyfunction(name = "prepare_terrain_derivatives", signature = (dem, derivatives = None))]
    pub fn prepare_terrain_derivatives_py(
        py: Python<'_>,
        dem: &Bound<'_, PyAny>,
        derivatives: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let source = raster_source_from_py(dem)?;
        let transform = source.info.transform.ok_or_else(|| {
            GisError::MissingTransform(
                "missing_transform: terrain derivatives require an affine transform".to_string(),
            )
        })?;
        let transform = AffineTransform::new([
            transform.0,
            transform.1,
            transform.2,
            transform.3,
            transform.4,
            transform.5,
        ])?;
        let requested = optional_string_list(derivatives)?
            .unwrap_or_else(|| vec!["slope".to_string(), "hillshade".to_string()]);
        let values = raster_to_f64(&source.array);
        let out = PyDict::new_bound(py);
        for derivative in requested {
            match derivative.as_str() {
                "slope" => {
                    let array = slope_array(&values, &source.array, transform)?;
                    let item = PyDict::new_bound(py);
                    item.set_item("array", super::super::raster_array_to_py(py, &array)?)?;
                    item.set_item("units", "degrees")?;
                    out.set_item("slope", item)?;
                }
                "hillshade" => {
                    let array = hillshade_array(&values, &source.array, transform)?;
                    let item = PyDict::new_bound(py);
                    item.set_item("array", super::super::raster_array_to_py(py, &array)?)?;
                    item.set_item("units", "uint8")?;
                    out.set_item("hillshade", item)?;
                }
                other => {
                    return Err(GisError::InvalidArgument(format!(
                        "unsupported_option: unsupported terrain derivative {other:?}"
                    ))
                    .into())
                }
            }
        }
        let dict = PyDict::new_bound(py);
        let derivative_count = out.len();
        dict.set_item("derivatives", out)?;
        dict.set_item(
            "info",
            super::super::raster_info_to_py_dict(py, &source.info)?,
        )?;
        dict.set_item(
            "operation",
            operation_py(
                py,
                "prepare_terrain_derivatives",
                1,
                derivative_count,
                false,
                Vec::new(),
            )?,
        )?;
        dict.set_item("warnings", warnings_to_py(py, &source.info.warnings)?)?;
        Ok(dict.into_py(py))
    }

    #[pyfunction(name = "read_gridded_dataset", signature = (path, variable = None))]
    pub fn read_gridded_dataset_py(
        py: Python<'_>,
        path: String,
        variable: Option<String>,
    ) -> PyResult<PyObject> {
        let ext = Path::new(&path)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if matches!(ext.as_str(), "nc" | "netcdf" | "h5" | "hdf5") {
            return Err(GisError::BackendUnavailable(
                "backend_unavailable: netcdf backend required for read_gridded_dataset".to_string(),
            )
            .into());
        }
        let result = raster_info::read_raster(&path, None, None, false)?;
        let variables = (1..=result.array.bands)
            .map(|band| format!("band_{band}"))
            .collect::<Vec<_>>();
        if variable.is_none() && variables.len() > 1 {
            return Err(GisError::InvalidArgument(
                "ambiguous_variable: variable must be selected for multiband raster-like grids"
                    .to_string(),
            )
            .into());
        }
        let dict = PyDict::new_bound(py);
        dict.set_item(
            "array",
            super::super::raster_array_to_py(py, &result.array)?,
        )?;
        dict.set_item(
            "info",
            super::super::raster_info_to_py_dict(py, &result.info)?,
        )?;
        dict.set_item("variables", variables)?;
        dict.set_item("variable", variable.unwrap_or_else(|| "band_1".to_string()))?;
        dict.set_item(
            "dimensions",
            json_to_py(
                py,
                &json!({"band": result.array.bands, "y": result.array.height, "x": result.array.width}),
            )?,
        )?;
        dict.set_item("warnings", warnings_to_py(py, &result.warnings)?)?;
        Ok(dict.into_py(py))
    }

    #[pyfunction(name = "subset_grid", signature = (source, bounds_or_coords, variable = None))]
    pub fn subset_grid_py(
        py: Python<'_>,
        source: &Bound<'_, PyAny>,
        bounds_or_coords: (f64, f64, f64, f64),
        variable: Option<String>,
    ) -> PyResult<PyObject> {
        let path = source.extract::<String>().map_err(|_| {
            GisError::InvalidArgument(
                "unsupported_layout: subset_grid first pass supports raster-like path sources"
                    .to_string(),
            )
        })?;
        let loaded = raster_info::read_raster_data(&path)?;
        if loaded.info.crs_wkt.is_none() && loaded.info.crs_authority.is_none() {
            return Err(GisError::MissingCrs(
                "missing_crs: spatial grid subset requires CRS metadata".to_string(),
            )
            .into());
        }
        let bounds = validate_bounds_tuple(bounds_or_coords, false)?;
        let window = window_from_bounds(&loaded.info, bounds, false)?.window;
        let array = copy_window(&loaded.array, &loaded.info.nodata_per_band, window)?;
        let transform = crate::gis::affine::window_transform(&loaded.info, window)?;
        let info = warp::operation_info(
            &loaded.info,
            window.width,
            window.height,
            Some(transform),
            None,
            &array,
        )?;
        let dict = PyDict::new_bound(py);
        dict.set_item("array", super::super::raster_array_to_py(py, &array)?)?;
        dict.set_item("info", super::super::raster_info_to_py_dict(py, &info)?)?;
        dict.set_item("bounds", bounds.tuple())?;
        dict.set_item("variable", variable.unwrap_or_else(|| "band_1".to_string()))?;
        dict.set_item("warnings", warnings_to_py(py, &info.warnings)?)?;
        Ok(dict.into_py(py))
    }

    #[pyfunction(name = "prepare_landcover_raster", signature = (source, target_info, classes = None))]
    pub fn prepare_landcover_raster_py(
        py: Python<'_>,
        source: &Bound<'_, PyAny>,
        target_info: &Bound<'_, PyAny>,
        classes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let mut source = raster_source_from_py(source)?;
        let target = raster_info_from_py(target_info)?;
        let mut warnings = Vec::new();
        if source.array.width != target.width as usize
            || source.array.height != target.height as usize
        {
            source.array =
                resize_nearest_f32(&source.array, target.width as usize, target.height as usize)?;
            warnings.push(RasterWarning::new(
                "categorical_resampling_warning",
                "categorical_resampling_warning: landcover was aligned with nearest-neighbor",
                Some("resampling"),
            ));
        }
        let class_labels = class_labels(classes)?;
        let values = raster_to_f64(&source.array);
        let mut counts = BTreeMap::<i64, usize>::new();
        let mut unknown = false;
        for value in values {
            let id = value.round() as i64;
            *counts.entry(id).or_default() += 1;
            if !class_labels.is_empty() && !class_labels.contains_key(&id) {
                unknown = true;
            }
        }
        if unknown {
            warnings.push(RasterWarning::new(
                "unknown_class",
                "unknown_class: landcover contains values outside classes",
                Some("classes"),
            ));
        }
        let mut info = target;
        info.dtype_per_band = vec![source.array.dtype().name().to_string()];
        let dict = PyDict::new_bound(py);
        dict.set_item(
            "array",
            super::super::raster_array_to_py(py, &source.array)?,
        )?;
        dict.set_item("info", super::super::raster_info_to_py_dict(py, &info)?)?;
        dict.set_item("class_counts", counts)?;
        dict.set_item(
            "class_table",
            class_table_py(py, &class_labels, &dict.get_item("class_counts")?)?,
        )?;
        dict.set_item(
            "operation",
            operation_py(
                py,
                "prepare_landcover_raster",
                1,
                1,
                !warnings.is_empty(),
                warnings.clone(),
            )?,
        )?;
        dict.set_item("warnings", warnings_to_py(py, &warnings)?)?;
        Ok(dict.into_py(py))
    }

    #[pyfunction(name = "prepare_population_raster", signature = (source, target_info = None, normalization = None))]
    pub fn prepare_population_raster_py(
        py: Python<'_>,
        source: &Bound<'_, PyAny>,
        target_info: Option<&Bound<'_, PyAny>>,
        normalization: Option<&str>,
    ) -> PyResult<PyObject> {
        let mut source = raster_source_from_py(source)?;
        let mut values = raster_to_f64(&source.array);
        if values.iter().any(|value| *value < 0.0) {
            return Err(GisError::InvalidArgument(
                "invalid_argument: population values must be non-negative".to_string(),
            )
            .into());
        }
        if let Some(target) = target_info {
            let target = raster_info_from_py(target)?;
            if source.array.width != target.width as usize
                || source.array.height != target.height as usize
            {
                source.array = resize_nearest_f32(
                    &source.array,
                    target.width as usize,
                    target.height as usize,
                )?;
                values = raster_to_f64(&source.array);
            }
            source.info = target;
        }
        let (array, method) = match normalization {
            None => (to_f32_array(&source.array)?, None),
            Some("minmax") => {
                let min = values.iter().copied().fold(f64::INFINITY, f64::min);
                let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let span = max - min;
                let out = values
                    .into_iter()
                    .map(|value| {
                        if span == 0.0 {
                            0.0
                        } else {
                            ((value - min) / span) as f32
                        }
                    })
                    .collect::<Vec<_>>();
                (
                    RasterArray::new(
                        RasterData::F32(out),
                        &[source.array.bands, source.array.height, source.array.width],
                    )?,
                    Some("minmax"),
                )
            }
            Some(other) => {
                return Err(GisError::InvalidArgument(format!(
                    "unsupported_option: unsupported population normalization {other:?}"
                ))
                .into())
            }
        };
        let mut info = source.info;
        info.dtype_per_band = vec!["float32".to_string(); array.bands];
        let dict = PyDict::new_bound(py);
        dict.set_item("array", super::super::raster_array_to_py(py, &array)?)?;
        dict.set_item("info", super::super::raster_info_to_py_dict(py, &info)?)?;
        dict.set_item("normalization", method)?;
        dict.set_item(
            "operation",
            operation_py(
                py,
                "prepare_population_raster",
                1,
                1,
                target_info.is_some(),
                Vec::new(),
            )?,
        )?;
        dict.set_item("warnings", warnings_to_py(py, &info.warnings)?)?;
        Ok(dict.into_py(py))
    }

    #[pyfunction(name = "prepare_osm_scene", signature = (aoi, tags = None, cache = None))]
    pub fn prepare_osm_scene_py(
        py: Python<'_>,
        aoi: (f64, f64, f64, f64),
        tags: Option<&Bound<'_, PyAny>>,
        cache: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let aoi = validate_bounds_tuple(aoi, true)?;
        let tags = tags.map(py_to_json).transpose()?.unwrap_or_else(|| {
            json!({"highway": true, "building": true, "natural": "water", "waterway": true, "landuse": true})
        });
        let cache = cache.map(py_to_json).transpose()?;
        let query = query_osm_features_value(aoi, &tags, cache.as_ref())?;
        let osm_json = query.get("osm_json").ok_or_else(|| {
            GisError::InvalidArgument(
                "malformed_payload: query_osm_features returned no osm_json".to_string(),
            )
        })?;
        let parsed = parse_osm_features_value(osm_json, None)?;
        let features = parsed
            .get("features")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let roads = filter_features(&features, |props| props.contains_key("highway"));
        let buildings = filter_features(&features, |props| props.contains_key("building"));
        let water = filter_features(&features, |props| {
            props.get("natural").and_then(Value::as_str) == Some("water")
                || props.contains_key("waterway")
        });
        let context = filter_features(&features, |props| props.contains_key("landuse"));
        let layers = PyDict::new_bound(py);
        layers.set_item("roads", layer_summary(py, &feature_collection(roads))?.1)?;
        layers.set_item(
            "buildings",
            layer_summary(py, &feature_collection(buildings.clone()))?.1,
        )?;
        layers.set_item("water", layer_summary(py, &feature_collection(water))?.1)?;
        layers.set_item(
            "context",
            layer_summary(py, &feature_collection(context))?.1,
        )?;
        let building_collection = feature_collection(buildings);
        let building_heights = heights_value(py, &building_collection)?;
        let dict = PyDict::new_bound(py);
        dict.set_item("layers", layers)?;
        dict.set_item("building_heights", building_heights)?;
        dict.set_item("osm", json_to_py(py, &parsed)?)?;
        dict.set_item(
            "remote",
            json_to_py(py, query.get("remote").unwrap_or(&Value::Null))?,
        )?;
        dict.set_item(
            "operation",
            operation_py(
                py,
                "prepare_osm_scene",
                features.len(),
                features.len(),
                false,
                Vec::new(),
            )?,
        )?;
        dict.set_item(
            "warnings",
            json_to_py(
                py,
                parsed.get("warnings").unwrap_or(&Value::Array(Vec::new())),
            )?,
        )?;
        Ok(dict.into_py(py))
    }

    #[pyfunction(name = "estimate_local_utm")]
    pub fn estimate_local_utm_py(
        py: Python<'_>,
        bounds_or_geometry: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let bounds = if let Ok(bounds) = bounds_or_geometry.extract::<(f64, f64, f64, f64)>() {
            RasterBounds {
                left: bounds.0,
                bottom: bounds.1,
                right: bounds.2,
                top: bounds.3,
            }
        } else {
            let value = py_to_json(bounds_or_geometry)?;
            bounds_for_features(&crate::gis::vector::normalize_features(&value)?).ok_or_else(
                || GisError::InvalidBounds("invalid_bounds: geometry has no bounds".to_string()),
            )?
        };
        if bounds.top > 84.0 || bounds.bottom < -80.0 {
            return Err(GisError::InvalidBounds(
                "invalid_bounds: UTM is not valid for polar bounds".to_string(),
            )
            .into());
        }
        let antimeridian = bounds.left > bounds.right;
        if !antimeridian && (bounds.left >= bounds.right || bounds.bottom >= bounds.top) {
            return Err(GisError::InvalidBounds(
                "invalid_bounds: bounds must be ordered as (left, bottom, right, top)".to_string(),
            )
            .into());
        }
        let lon_center = if antimeridian {
            let right = bounds.right + 360.0;
            let mut center = (bounds.left + right) / 2.0;
            if center > 180.0 {
                center -= 360.0;
            }
            center
        } else {
            (bounds.left + bounds.right) / 2.0
        };
        let lat_center = (bounds.bottom + bounds.top) / 2.0;
        let zone = (((lon_center + 180.0) / 6.0).floor() as i64 + 1).clamp(1, 60);
        let epsg = if lat_center >= 0.0 {
            32600 + zone
        } else {
            32700 + zone
        };
        let dict = PyDict::new_bound(py);
        dict.set_item("epsg", epsg)?;
        dict.set_item("crs", format!("EPSG:{epsg}"))?;
        dict.set_item("zone", zone)?;
        dict.set_item(
            "hemisphere",
            if lat_center >= 0.0 { "north" } else { "south" },
        )?;
        dict.set_item("center", (lon_center, lat_center))?;
        dict.set_item("confidence", if antimeridian { "low" } else { "high" })?;
        dict.set_item("antimeridian", antimeridian)?;
        dict.set_item("warnings", Vec::<String>::new())?;
        Ok(dict.into_py(py))
    }

    fn raster_source_from_py(value: &Bound<'_, PyAny>) -> PyResult<RasterSource> {
        if let Ok(path) = value.extract::<String>() {
            let result = raster_info::read_raster(&path, None, None, false)?;
            return Ok(RasterSource {
                array: result.array,
                info: result.info,
            });
        }
        if let Ok(info) = value.extract::<PyRef<'_, RasterInfo>>() {
            if info.path.is_empty() {
                return Err(GisError::InvalidArgument(
                    "invalid_argument: RasterInfo source must include a path".to_string(),
                )
                .into());
            }
            let result = raster_info::read_raster(&info.path, None, None, false)?;
            return Ok(RasterSource {
                array: result.array,
                info: result.info,
            });
        }
        if let Ok(dict) = value.downcast::<PyDict>() {
            let Some(array_value) = dict.get_item("array")? else {
                return Err(GisError::InvalidArgument(
                    "invalid_argument: source dict must include array".to_string(),
                )
                .into());
            };
            let array = super::super::extract_raster_array(&array_value)?;
            let info = if let Some(info_value) = dict.get_item("info")? {
                raster_info_from_py(&info_value)?
            } else {
                synthetic_info(&array)
            };
            return Ok(RasterSource { array, info });
        }
        let array = super::super::extract_raster_array(value)?;
        let info = synthetic_info(&array);
        Ok(RasterSource { array, info })
    }

    fn raster_info_from_py(value: &Bound<'_, PyAny>) -> PyResult<RasterInfo> {
        if let Ok(info) = value.extract::<PyRef<'_, RasterInfo>>() {
            return Ok(info.clone());
        }
        let dict = value.downcast::<PyDict>().map_err(|_| {
            GisError::InvalidArgument(
                "invalid_argument: raster info must be RasterInfo or dict".to_string(),
            )
        })?;
        let width = required_u32(dict, "width")?;
        let height = required_u32(dict, "height")?;
        let band_count = optional_extract::<u16>(dict.get_item("band_count")?)?.unwrap_or(1);
        let mut info = RasterInfo::new(
            optional_extract::<String>(dict.get_item("path")?)?
                .unwrap_or_default()
                .into(),
            width,
            height,
            band_count,
        );
        info.driver = optional_extract::<String>(dict.get_item("driver")?)?
            .unwrap_or_else(|| "memory".to_string());
        info.dtype_per_band = dict
            .get_item("dtype_per_band")?
            .map(|value| value.extract::<Vec<String>>())
            .transpose()?
            .unwrap_or_else(|| vec!["float32".to_string(); band_count as usize]);
        info.crs_wkt = optional_extract::<String>(dict.get_item("crs_wkt")?)?;
        info.crs_authority = dict
            .get_item("crs_authority")?
            .map(|value| {
                if value.is_none() {
                    Ok(None)
                } else {
                    value.extract::<HashMap<String, String>>().map(Some)
                }
            })
            .transpose()?
            .flatten();
        info.transform =
            optional_extract::<(f64, f64, f64, f64, f64, f64)>(dict.get_item("transform")?)?;
        info.bounds = optional_extract::<(f64, f64, f64, f64)>(dict.get_item("bounds")?)?;
        info.resolution = optional_extract::<(f64, f64)>(dict.get_item("resolution")?)?;
        info.nodata_per_band = dict
            .get_item("nodata_per_band")?
            .map(|value| value.extract::<Vec<Option<f64>>>())
            .transpose()?
            .unwrap_or_else(|| vec![None; band_count as usize]);
        info.is_georeferenced =
            info.transform.is_some() && (info.crs_wkt.is_some() || info.crs_authority.is_some());
        Ok(info)
    }

    fn synthetic_info(array: &RasterArray) -> RasterInfo {
        let mut info = RasterInfo::new(
            "".into(),
            array.width as u32,
            array.height as u32,
            array.bands as u16,
        );
        info.driver = "memory".to_string();
        info.dtype_per_band = vec![array.dtype().name().to_string(); array.bands];
        info.nodata_per_band = vec![None; array.bands];
        info
    }

    fn required_u32(dict: &Bound<'_, PyDict>, key: &'static str) -> PyResult<u32> {
        dict.get_item(key)?
            .ok_or_else(|| {
                GisError::InvalidArgument(format!("invalid_argument: raster info missing {key}"))
            })?
            .extract()
            .map_err(Into::into)
    }

    fn optional_extract<T>(value: Option<Bound<'_, PyAny>>) -> PyResult<Option<T>>
    where
        T: for<'py> FromPyObject<'py>,
    {
        value
            .map(|value| {
                if value.is_none() {
                    Ok(None)
                } else {
                    value.extract::<T>().map(Some)
                }
            })
            .transpose()
            .map(Option::flatten)
    }

    fn to_f32_array(array: &RasterArray) -> PyResult<RasterArray> {
        let values = raster_to_f64(array)
            .into_iter()
            .map(|value| value as f32)
            .collect();
        RasterArray::new(
            RasterData::F32(values),
            &[array.bands, array.height, array.width],
        )
        .map_err(Into::into)
    }

    fn resize_nearest_f32(
        array: &RasterArray,
        width: usize,
        height: usize,
    ) -> PyResult<RasterArray> {
        let src = raster_to_f64(array);
        let mut out = vec![0.0f32; array.bands * width * height];
        for band in 0..array.bands {
            for row in 0..height {
                let src_row = ((row as f64 + 0.5) * array.height as f64 / height as f64)
                    .floor()
                    .min((array.height - 1) as f64) as usize;
                for col in 0..width {
                    let src_col = ((col as f64 + 0.5) * array.width as f64 / width as f64)
                        .floor()
                        .min((array.width - 1) as f64) as usize;
                    out[band * width * height + row * width + col] = src
                        [band * array.width * array.height + src_row * array.width + src_col]
                        as f32;
                }
            }
        }
        RasterArray::new(RasterData::F32(out), &[array.bands, height, width]).map_err(Into::into)
    }

    fn slope_array(
        values: &[f64],
        array: &RasterArray,
        transform: AffineTransform,
    ) -> PyResult<RasterArray> {
        let (xres, yres) = transform.resolution();
        let mut out = vec![0.0f32; array.height * array.width];
        for row in 0..array.height {
            for col in 0..array.width {
                let left = sample(values, array, col.saturating_sub(1), row);
                let right = sample(values, array, (col + 1).min(array.width - 1), row);
                let up = sample(values, array, col, row.saturating_sub(1));
                let down = sample(values, array, col, (row + 1).min(array.height - 1));
                let dzdx = (right - left) / (2.0 * xres);
                let dzdy = (down - up) / (2.0 * yres);
                out[row * array.width + col] = dzdx.hypot(dzdy).atan().to_degrees() as f32;
            }
        }
        RasterArray::new(RasterData::F32(out), &[1, array.height, array.width]).map_err(Into::into)
    }

    fn hillshade_array(
        values: &[f64],
        array: &RasterArray,
        transform: AffineTransform,
    ) -> PyResult<RasterArray> {
        let slope = raster_to_f64(&slope_array(values, array, transform)?);
        let azimuth = 315f64.to_radians();
        let altitude = 45f64.to_radians();
        let mut out = Vec::with_capacity(slope.len());
        for slope_deg in slope {
            let slope_rad = slope_deg.to_radians();
            let shade = (altitude.sin() * slope_rad.cos()
                + altitude.cos() * slope_rad.sin() * azimuth.cos())
            .max(0.0)
                * 255.0;
            out.push(shade.round().clamp(0.0, 255.0) as u8);
        }
        RasterArray::new(RasterData::U8(out), &[1, array.height, array.width]).map_err(Into::into)
    }

    fn sample(values: &[f64], array: &RasterArray, col: usize, row: usize) -> f64 {
        values[row * array.width + col]
    }

    fn context_layers(source: &Bound<'_, PyAny>) -> PyResult<BTreeMap<String, Value>> {
        if let Ok(path) = source.extract::<String>() {
            let ext = Path::new(&path)
                .extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if ext == "gpkg" {
                return Err(GisError::BackendUnavailable(
                    "backend_unavailable: gdal-vector feature required for GPKG".to_string(),
                )
                .into());
            }
            let result = read_vector(
                path,
                VectorReadOptions {
                    layer: None,
                    columns: None,
                    bbox: None,
                    limit: None,
                },
            )?;
            let mut map = BTreeMap::new();
            map.insert(
                result
                    .info
                    .layer_name
                    .unwrap_or_else(|| "default".to_string()),
                feature_collection(result.features),
            );
            return Ok(map);
        }
        let value = py_to_json(source)?;
        if value.get("type").and_then(Value::as_str).is_some() {
            let mut map = BTreeMap::new();
            map.insert("default".to_string(), value);
            return Ok(map);
        }
        let object = value.as_object().ok_or_else(|| {
            GisError::InvalidArgument(
                "invalid_argument: context vectors must be a path, GeoJSON, or dict of layers"
                    .to_string(),
            )
        })?;
        Ok(object
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect())
    }

    fn input_geojson_or_cityjson(source: &Bound<'_, PyAny>) -> PyResult<Value> {
        if let Ok(path) = source.extract::<String>() {
            let path_buf = PathBuf::from(&path);
            let ext = path_buf
                .extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if ext == "gpkg" {
                return Err(GisError::BackendUnavailable(
                    "backend_unavailable: gdal-vector feature required for GPKG".to_string(),
                )
                .into());
            }
            let text = std::fs::read_to_string(path_buf)?;
            return serde_json::from_str(&text).map_err(|err| {
                GisError::InvalidArgument(format!("malformed_payload: invalid vector JSON: {err}"))
                    .into()
            });
        }
        py_to_json(source)
    }

    fn layer_summary(py: Python<'_>, value: &Value) -> PyResult<(usize, PyObject)> {
        let features = crate::gis::vector::normalize_features(value)?;
        let bounds = bounds_for_features(&features);
        let dict = PyDict::new_bound(py);
        dict.set_item("type", "FeatureCollection")?;
        dict.set_item("features", json_to_py(py, &Value::Array(features.clone()))?)?;
        dict.set_item("feature_count", features.len())?;
        dict.set_item("bounds", bounds.map(RasterBounds::tuple))?;
        dict.set_item(
            "crs",
            json_to_py(py, &json!({"name": "EPSG", "code": "4326"}))?,
        )?;
        Ok((features.len(), dict.into_py(py)))
    }

    fn cityjson_building_features(value: &Value) -> PyResult<Vec<Value>> {
        let vertices = value
            .get("vertices")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                GisError::InvalidArgument(
                    "unsupported_layout: CityJSON missing vertices".to_string(),
                )
            })?;
        let objects = value
            .get("CityObjects")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                GisError::InvalidArgument(
                    "unsupported_layout: CityJSON missing CityObjects".to_string(),
                )
            })?;
        let mut features = Vec::new();
        for (id, object) in objects {
            let kind = object.get("type").and_then(Value::as_str).unwrap_or("");
            if kind != "Building" && kind != "BuildingPart" {
                continue;
            }
            let boundaries = object
                .get("geometry")
                .and_then(Value::as_array)
                .and_then(|items| items.first())
                .and_then(|geom| geom.get("boundaries"))
                .ok_or_else(|| {
                    GisError::InvalidArgument(
                        "unsupported_layout: CityJSON building geometry missing boundaries"
                            .to_string(),
                    )
                })?;
            let ring_indices = first_index_ring(boundaries).ok_or_else(|| {
                GisError::InvalidArgument(
                    "unsupported_layout: CityJSON footprint ring was not found".to_string(),
                )
            })?;
            let mut ring = Vec::new();
            for idx in ring_indices {
                let Some(vertex) = vertices.get(idx).and_then(Value::as_array) else {
                    continue;
                };
                ring.push(json!([
                    vertex.first().and_then(Value::as_f64).unwrap_or(0.0),
                    vertex.get(1).and_then(Value::as_f64).unwrap_or(0.0)
                ]));
            }
            if ring.first() != ring.last() {
                if let Some(first) = ring.first().cloned() {
                    ring.push(first);
                }
            }
            let mut props = object
                .get("attributes")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            props.insert("id".to_string(), Value::String(id.clone()));
            features.push(json!({
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }));
        }
        Ok(features)
    }

    fn first_index_ring(value: &Value) -> Option<Vec<usize>> {
        if let Some(array) = value.as_array() {
            if array.iter().all(|item| item.as_u64().is_some()) {
                return Some(
                    array
                        .iter()
                        .filter_map(|item| item.as_u64().map(|v| v as usize))
                        .collect(),
                );
            }
            for item in array {
                if let Some(ring) = first_index_ring(item) {
                    return Some(ring);
                }
            }
        }
        None
    }

    fn is_polygonal(geometry: &Value) -> bool {
        matches!(
            geometry.get("type").and_then(Value::as_str),
            Some("Polygon" | "MultiPolygon")
        )
    }

    #[derive(Clone, Copy)]
    struct HeightDefaults {
        height_m: f64,
        level_height_m: f64,
    }

    fn height_defaults(defaults: Option<&Bound<'_, PyAny>>) -> PyResult<HeightDefaults> {
        let mut out = HeightDefaults {
            height_m: 10.0,
            level_height_m: 3.0,
        };
        if let Some(defaults) = defaults {
            if !defaults.is_none() {
                let dict = defaults.downcast::<PyDict>().map_err(|_| {
                    GisError::InvalidArgument(
                        "invalid_argument: defaults must be a dict".to_string(),
                    )
                })?;
                if let Some(value) = dict.get_item("height_m")? {
                    out.height_m = value.extract()?;
                }
                if let Some(value) = dict.get_item("level_height_m")? {
                    out.level_height_m = value.extract()?;
                }
            }
        }
        Ok(out)
    }

    fn height_from_properties(
        props: &Map<String, Value>,
        defaults: HeightDefaults,
    ) -> Result<(f64, String, bool), String> {
        for key in ["height", "building:height", "render_height", "roof:height"] {
            if let Some(value) = props.get(key) {
                let height = parse_height(value)?;
                if height > 0.0 {
                    return Ok((height, key.to_string(), false));
                }
                return Err("invalid_height: height must be positive".to_string());
            }
        }
        for key in ["building:levels", "levels", "building:part:levels"] {
            if let Some(value) = props.get(key).and_then(numberish) {
                if value > 0.0 {
                    return Ok((value * defaults.level_height_m, key.to_string(), false));
                }
                return Err("invalid_height: levels must be positive".to_string());
            }
        }
        Ok((defaults.height_m, "default".to_string(), true))
    }

    fn parse_height(value: &Value) -> Result<f64, String> {
        if let Some(value) = numberish(value) {
            return Ok(value);
        }
        let text = value
            .as_str()
            .ok_or_else(|| "invalid_height: height must be numeric".to_string())?
            .trim()
            .to_ascii_lowercase();
        let mut parts = text.split_whitespace();
        let number = parts
            .next()
            .ok_or_else(|| "invalid_height: height must include a value".to_string())?
            .parse::<f64>()
            .map_err(|_| "invalid_height: height value is invalid".to_string())?;
        let unit = parts.next().unwrap_or("m");
        match unit {
            "m" | "meter" | "meters" => Ok(number),
            "ft" | "feet" => Ok(number * 0.3048),
            _ => Err("invalid_height: unsupported height unit".to_string()),
        }
    }

    fn numberish(value: &Value) -> Option<f64> {
        value
            .as_f64()
            .or_else(|| value.as_str()?.trim().parse().ok())
    }

    fn feature_collection(features: Vec<Value>) -> Value {
        json!({
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
            "features": features,
        })
    }

    fn feature_count(value: &Value) -> usize {
        value
            .get("features")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0)
    }

    fn bounds_for_features(features: &[Value]) -> Option<RasterBounds> {
        let mut bounds: Option<RasterBounds> = None;
        for feature in features {
            visit_coords(feature.get("geometry")?, &mut |x, y| {
                bounds = Some(match bounds {
                    Some(mut b) => {
                        b.left = b.left.min(x);
                        b.right = b.right.max(x);
                        b.bottom = b.bottom.min(y);
                        b.top = b.top.max(y);
                        b
                    }
                    None => RasterBounds {
                        left: x,
                        bottom: y,
                        right: x,
                        top: y,
                    },
                });
            });
        }
        bounds
    }

    fn visit_coords(geometry: &Value, f: &mut impl FnMut(f64, f64)) {
        match geometry.get("type").and_then(Value::as_str) {
            Some("Point") => visit_coord(geometry.get("coordinates"), f),
            Some("LineString") => visit_coord_array(geometry.get("coordinates"), f),
            Some("Polygon") => {
                if let Some(rings) = geometry.get("coordinates").and_then(Value::as_array) {
                    for ring in rings {
                        visit_coord_array(Some(ring), f);
                    }
                }
            }
            _ => {}
        }
    }

    fn visit_coord_array(value: Option<&Value>, f: &mut impl FnMut(f64, f64)) {
        if let Some(coords) = value.and_then(Value::as_array) {
            for coord in coords {
                visit_coord(Some(coord), f);
            }
        }
    }

    fn visit_coord(value: Option<&Value>, f: &mut impl FnMut(f64, f64)) {
        let Some(coord) = value.and_then(Value::as_array) else {
            return;
        };
        if let (Some(x), Some(y)) = (
            coord.first().and_then(Value::as_f64),
            coord.get(1).and_then(Value::as_f64),
        ) {
            f(x, y);
        }
    }

    fn optional_string_list(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<String>>> {
        let Some(value) = value else {
            return Ok(None);
        };
        if value.is_none() {
            return Ok(None);
        }
        if let Ok(text) = value.extract::<String>() {
            return Ok(Some(vec![text]));
        }
        if let Ok(list) = value.downcast::<PyList>() {
            return list
                .iter()
                .map(|item| item.extract::<String>())
                .collect::<PyResult<Vec<_>>>()
                .map(Some);
        }
        if let Ok(tuple) = value.downcast::<PyTuple>() {
            return tuple
                .iter()
                .map(|item| item.extract::<String>())
                .collect::<PyResult<Vec<_>>>()
                .map(Some);
        }
        Err(GisError::InvalidArgument(
            "invalid_argument: value must be a string or sequence of strings".to_string(),
        )
        .into())
    }

    fn operation_py(
        py: Python<'_>,
        name: &'static str,
        input_count: usize,
        output_count: usize,
        changed: bool,
        warnings: Vec<RasterWarning>,
    ) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("name", name)?;
        dict.set_item("source_kind", "memory")?;
        dict.set_item("input_count", input_count)?;
        dict.set_item("output_count", output_count)?;
        dict.set_item("input_crs", py.None())?;
        dict.set_item("output_crs", py.None())?;
        dict.set_item("target_grid", py.None())?;
        dict.set_item("changed", changed)?;
        dict.set_item("warnings", warnings_to_py(py, &warnings)?)?;
        Ok(dict.into_py(py))
    }

    fn class_labels(classes: Option<&Bound<'_, PyAny>>) -> PyResult<BTreeMap<i64, String>> {
        let Some(classes) = classes else {
            return Ok(BTreeMap::new());
        };
        if classes.is_none() {
            return Ok(BTreeMap::new());
        }
        let dict = classes.downcast::<PyDict>().map_err(|_| {
            GisError::InvalidArgument("invalid_argument: classes must be a dict".to_string())
        })?;
        let mut out = BTreeMap::new();
        for (key, value) in dict.iter() {
            let id = if let Ok(id) = key.extract::<i64>() {
                id
            } else {
                key.extract::<String>()?.parse().map_err(|_| {
                    GisError::InvalidArgument(
                        "invalid_argument: class ids must be integers".to_string(),
                    )
                })?
            };
            out.insert(id, value.extract::<String>()?);
        }
        Ok(out)
    }

    fn class_table_py(
        py: Python<'_>,
        labels: &BTreeMap<i64, String>,
        _counts: &Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let items = labels
            .iter()
            .map(|(class_id, label)| {
                let dict = PyDict::new_bound(py);
                dict.set_item("class_id", *class_id)?;
                dict.set_item("label", label)?;
                Ok(dict.into_py(py))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new_bound(py, items).into_py(py))
    }

    fn filter_features(
        features: &[Value],
        pred: impl Fn(&Map<String, Value>) -> bool,
    ) -> Vec<Value> {
        features
            .iter()
            .filter(|feature| {
                feature
                    .get("properties")
                    .and_then(Value::as_object)
                    .map(&pred)
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    }

    fn heights_value(py: Python<'_>, collection: &Value) -> PyResult<PyObject> {
        let features = crate::gis::vector::normalize_features(collection)?;
        let mut heights = Vec::new();
        let mut attrs = Vec::new();
        for feature in features {
            let props = feature
                .get("properties")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let (height, attr, _) = height_from_properties(
                &props,
                HeightDefaults {
                    height_m: 10.0,
                    level_height_m: 3.0,
                },
            )
            .unwrap_or((10.0, "default".to_string(), true));
            heights.push(height);
            attrs.push(attr);
        }
        let dict = PyDict::new_bound(py);
        dict.set_item("heights_m", heights)?;
        dict.set_item("attributes", attrs)?;
        Ok(dict.into_py(py))
    }
}
