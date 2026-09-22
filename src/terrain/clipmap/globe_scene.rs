//! Native ORBIS scene orchestration exposed to Python.

use crate::terrain::cog::py_bindings::PyCogDataset;
use crate::terrain::planetary_tiles::global_tile_lonlat_bounds;
use crate::terrain::renderer::TerrainRenderer;
use numpy::{PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

mod metrics;
mod source;

use metrics::{GlobeMetrics, StreamingEvidence, StreamingProgressSnapshot};
use source::{
    coverage_error, default_render_params, extract_source_path, load_covered_overview,
    normalize_source, source_error, tile_extent_m, OverviewSeed, OverviewSeedCache,
};

const DEFAULT_START_ALTITUDE_M: f64 = 408_000.0;
const MAX_SUPPORTED_ALTITUDE_M: f64 = DEFAULT_START_ALTITUDE_M;
const DEFAULT_WAYPOINT_COUNT: usize = 25;
const GPU_VISIBLE_BUDGET: u64 = 64 * 1024 * 1024;
/// Globe clipmap rings. The target-LOD leaf footprint covers the centre and
/// the first four rings; the outer rings reach across the source extent and
/// sample the coarse context pages.
const GLOBE_RING_COUNT: u32 = 7;
const GLOBE_CAMERA_MODE: &str = "clipmap:7:32:32:10:0.3:zup";
const MICRO_STEP_M: f64 = 0.002;
const ORBIS_PROBE_BASE_ALTITUDE_M: f64 = 6_000.0;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Waypoint {
    lon: f64,
    lat: f64,
    altitude: f64,
    heading: f64,
    pitch: f64,
}

#[derive(Clone, Copy, Debug)]
struct CameraPose {
    radius: f32,
    phi_deg: f32,
    theta_deg: f32,
    target: [f32; 3],
}

/// A native renderer, COG streamer, and camera-relative globe frame managed as
/// one deterministic descent scene. Every rendered waypoint is constrained to
/// the fully covered overview tile seeded by the constructor target.
#[pyclass(module = "forge3d._forge3d", name = "GlobeScene")]
pub struct GlobeScene {
    source: String,
    target: Waypoint,
    target_name: String,
    renderer: TerrainRenderer,
    dataset: PyCogDataset,
    material_set: Py<crate::render::material_set::MaterialSet>,
    env_maps: Py<crate::lighting::ibl_wrapper::IBL>,
    params: Py<crate::terrain::render_params::TerrainRenderParams>,
    overview_seeds: OverviewSeedCache,
    legacy_ground_seeds: OverviewSeedCache,
    ground_legacy_active: bool,
    active_overview_tile: crate::terrain::tiling::TileId,
    source_lod: u32,
    source_dimensions: (u32, u32),
    source_sample_spacing_m: f32,
    terrain_extent_m: f32,
    source_bounds: (f64, f64, f64, f64),
    current: Option<Waypoint>,
    last_frame: Option<Py<crate::Frame>>,
    rendered_waypoints: usize,
    descent_complete: bool,
    metrics: Option<GlobeMetrics>,
    streaming_evidence: StreamingEvidence,
}

#[pymethods]
impl GlobeScene {
    #[new]
    #[pyo3(signature = (cog_source, target_lon, target_lat, target_name, material_set=None, env_maps=None, params=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        cog_source: &Bound<'_, PyAny>,
        target_lon: f64,
        target_lat: f64,
        target_name: &str,
        material_set: Option<Py<crate::render::material_set::MaterialSet>>,
        env_maps: Option<Py<crate::lighting::ibl_wrapper::IBL>>,
        params: Option<Py<crate::terrain::render_params::TerrainRenderParams>>,
    ) -> PyResult<Self> {
        validate_target(target_lon, target_lat, target_name)?;
        let cog_source = extract_source_path(py, cog_source)?;
        if cog_source.trim().is_empty() {
            return Err(PyValueError::new_err("cog_source must not be empty"));
        }
        let source_url = normalize_source(&cog_source).map_err(|error| {
            source_error(&cog_source, target_name, target_lon, target_lat, error)
        })?;
        let dataset = PyCogDataset::new(&source_url, 64, None, Some(64)).map_err(|error| {
            source_error(
                &cog_source,
                target_name,
                target_lon,
                target_lat,
                error.to_string(),
            )
        })?;
        let bounds = dataset.bounds();
        if target_lon < bounds.0
            || target_lon > bounds.2
            || target_lat < bounds.1
            || target_lat > bounds.3
        {
            return Err(coverage_error(
                &cog_source,
                target_name,
                target_lon,
                target_lat,
                bounds,
            ));
        }
        let (initial_seed, overview_lod) =
            load_covered_overview(&dataset, target_lon, target_lat).map_err(|error| {
                source_error(&cog_source, target_name, target_lon, target_lat, error)
            })?;
        let tile = initial_seed.tile;
        let tile_bounds = initial_seed.bounds;
        let terrain_extent_m = tile_extent_m(tile_bounds, target_lat);
        let source_lod = overview_lod.max(source::MIN_DETAIL_LOD);
        let detail_tile = source::target_tile(target_lon, target_lat, source_lod);
        let detail_bounds = global_tile_lonlat_bounds(detail_tile).map_err(PyValueError::new_err)?;
        let source_sample_spacing_m = source::tile_sample_spacing_m(
            detail_bounds,
            target_lat,
            source::STREAM_TILE_RESOLUTION,
        );
        let source_dimensions = dataset
            .reader()
            .header()
            .full_resolution()
            .map(|full| (full.width, full.height))
            .ok_or_else(|| {
                source_error(
                    &cog_source,
                    target_name,
                    target_lon,
                    target_lat,
                    "COG contains no full-resolution image",
                )
            })?;
        let initial_legacy_seed = if initial_seed.size == source::MIN_OVERVIEW_SIZE {
            initial_seed.clone()
        } else {
            source::read_covered_seed(&dataset, tile, source::MIN_OVERVIEW_SIZE)
                .map_err(|error| {
                    source_error(&cog_source, target_name, target_lon, target_lat, error)
                })?
                .ok_or_else(|| {
                    source_error(
                        &cog_source,
                        target_name,
                        target_lon,
                        target_lat,
                        "legacy ground overview tile is not fully covered",
                    )
                })?
        };

        // Source and target are fully validated before GPU initialization, so
        // diagnostic failures never get masked by hosted-adapter availability.
        let session = crate::core::session::Session::new(false, None).map_err(|error| {
            source_error(
                &cog_source,
                target_name,
                target_lon,
                target_lat,
                format!("GPU adapter/session unavailable: {error}"),
            )
        })?;
        let mut renderer = TerrainRenderer::new(&session).map_err(|error| {
            source_error(
                &cog_source,
                target_name,
                target_lon,
                target_lat,
                format!("terrain renderer unavailable: {error}"),
            )
        })?;
        renderer.enable_height_streaming_cog_globe(
            &dataset,
            terrain_extent_m,
            GLOBE_RING_COUNT,
            32,
            source_lod,
            source::STREAM_TILE_RESOLUTION,
            64,
            8,
            true,
            Some(GPU_VISIBLE_BUDGET),
            Some(tile_bounds),
        )
        .map_err(|error| {
            source_error(
                &cog_source,
                target_name,
                target_lon,
                target_lat,
                format!("globe streaming unavailable: {error}"),
            )
        })?;
        renderer
            .activate_height_streaming_overview(
                tile_bounds,
                initial_seed.size,
                initial_seed.size,
                &initial_seed.heights,
            )
            .map_err(|error| {
                source_error(
                    &cog_source,
                    target_name,
                    target_lon,
                    target_lat,
                    format!("initial overview activation failed: {error:#}"),
                )
            })?;
        renderer
            .set_height_streaming_context_tiles(source::context_tiles(
                bounds,
                target_lon,
                target_lat,
                source_lod,
            ))
            .map_err(|error| {
                source_error(
                    &cog_source,
                    target_name,
                    target_lon,
                    target_lat,
                    format!("context tile setup failed: {error:#}"),
                )
            })?;

        let material_set = match material_set {
            Some(value) => value,
            None => Py::new(
                py,
                crate::render::material_set::MaterialSet::terrain_default(6.0, 1.0, 4.0)?,
            )?,
        };
        let env_maps = match env_maps {
            Some(value) => value,
            None => Py::new(py, crate::lighting::ibl_wrapper::IBL::neutral_orbis())?,
        };
        let params = match params {
            Some(value) => value,
            None => Py::new(py, default_render_params(py, terrain_extent_m)?)?,
        };

        Ok(Self {
            source: cog_source,
            target: Waypoint {
                lon: target_lon,
                lat: target_lat,
                altitude: 0.0,
                heading: 0.0,
                pitch: 0.0,
            },
            target_name: target_name.trim().to_string(),
            renderer,
            dataset,
            material_set,
            env_maps,
            params,
            overview_seeds: OverviewSeedCache::new(initial_seed),
            legacy_ground_seeds: OverviewSeedCache::new(initial_legacy_seed),
            ground_legacy_active: false,
            active_overview_tile: tile,
            source_lod,
            source_dimensions,
            source_sample_spacing_m,
            terrain_extent_m,
            source_bounds: bounds,
            current: None,
            last_frame: None,
            rendered_waypoints: 0,
            descent_complete: false,
            metrics: None,
            streaming_evidence: StreamingEvidence::default(),
        })
    }

    #[staticmethod]
    fn default_descent_altitudes() -> Vec<f64> {
        default_descent_altitudes()
    }

    #[pyo3(signature = (lon, lat, altitude, heading=0.0, pitch=0.0))]
    fn fly_to(
        &mut self,
        py: Python<'_>,
        lon: f64,
        lat: f64,
        altitude: f64,
        heading: f64,
        pitch: f64,
    ) -> PyResult<Py<crate::Frame>> {
        let waypoint = validate_waypoint(lon, lat, altitude, heading, pitch).map_err(|error| {
            self.waypoint_validation_error(
                Waypoint {
                    lon,
                    lat,
                    altitude,
                    heading,
                    pitch,
                },
                "validation",
                error,
            )
        })?;
        self.validate_waypoint_coverage(waypoint).map_err(|error| {
            self.waypoint_validation_error(waypoint, "coverage validation", error)
        })?;
        let tile = self.prepare_overview_seed(waypoint)?;
        self.render_waypoint_with_overview(py, waypoint, tile)
    }

    #[pyo3(signature = (waypoints=None))]
    fn scripted_descent(
        &mut self,
        py: Python<'_>,
        waypoints: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<GlobeMetrics> {
        let path = match waypoints {
            Some(values) => {
                let mut path = Vec::new();
                let items = values.iter().map_err(|error| {
                    PyValueError::new_err(format!(
                        "ORBIS waypoint parsing failed for COG source {:?}, target {:?}: {error}",
                        self.source, self.target_name,
                    ))
                })?;
                for item in items {
                    let item = item.map_err(|error| {
                        PyValueError::new_err(format!(
                            "ORBIS waypoint parsing failed for COG source {:?}, target {:?}: {error}",
                            self.source, self.target_name,
                        ))
                    })?;
                    path.push(parse_waypoint(&item).map_err(|error| {
                        self.waypoint_parse_error(&item, error)
                    })?);
                }
                if path.is_empty() {
                    return Err(PyValueError::new_err("waypoints must not be empty"));
                }
                path
            }
            None => default_descent_altitudes()
                .into_iter()
                .map(|altitude| Waypoint {
                    altitude,
                    ..self.target
                })
                .collect(),
        };
        // Validate and synchronously preload every distinct CPU overview seed
        // before descent state, metrics, or GPU work is mutated. The frame
        // loop below performs only cached activation plus bounded GPU work.
        for waypoint in &path {
            self.validate_waypoint_coverage(*waypoint).map_err(|error| {
                self.waypoint_validation_error(*waypoint, "coverage validation", error)
            })?;
        }
        let prepared_path = path
            .iter()
            .copied()
            .map(|waypoint| self.prepare_overview_seed(waypoint).map(|tile| (waypoint, tile)))
            .collect::<PyResult<Vec<_>>>()?;
        let owner = self.renderer.allocation_owner();
        let (adapter, software_fallback) = self
            .renderer
            .adapter_evidence()
            .map_err(|error| PyRuntimeError::new_err(format!("ORBIS adapter evidence failed: {error:#}")))?;
        validate_selected_acceptance_adapter(&adapter, software_fallback)?;
        self.streaming_evidence = StreamingEvidence::default();
        let baseline_stats = self.renderer.height_streaming_stats(py)?;
        let baseline = self.streaming_snapshot(py, &baseline_stats)?;
        self.streaming_evidence.seed(baseline);
        self.metrics = None;
        self.descent_complete = false;
        self.renderer.begin_orbis_descent();
        crate::core::resource_tracker::begin_owner_ledger_capture(&owner);
        let frame_result = (|| {
            for (waypoint, tile) in prepared_path {
                self.render_waypoint_with_overview(py, waypoint, tile)?;
            }
            self.run_orbis_physical_probe(py)?;
            let physical = self
                .renderer
                .finish_orbis_physical_capture()
                .map_err(|error| PyRuntimeError::new_err(format!("ORBIS physical metric readback failed: {error:#}")))?;
            Ok::<_, PyErr>(physical)
        })();
        let allocation = crate::core::resource_tracker::finish_owner_ledger_capture(&owner);
        self.renderer.end_orbis_descent();
        let physical = match frame_result {
            Ok(value) => value,
            Err(error) => {
                self.renderer.abort_orbis_physical_capture();
                return Err(error);
            }
        };
        let metrics = GlobeMetrics::measured(
            physical,
            allocation.peak_total_bytes,
            &self.streaming_evidence,
            adapter,
            software_fallback,
        )
        .map_err(|error| PyRuntimeError::new_err(format!("ORBIS metrics incomplete: {error}")))?;
        self.descent_complete = true;
        self.metrics = Some(metrics.clone());
        Ok(metrics)
    }

    fn snapshot(&self, py: Python<'_>) -> PyResult<Py<crate::Frame>> {
        self.last_frame
            .as_ref()
            .map(|frame| frame.clone_ref(py))
            .ok_or_else(|| PyRuntimeError::new_err("snapshot unavailable before a rendered frame"))
    }

    fn metrics(&self) -> PyResult<GlobeMetrics> {
        if !self.descent_complete {
            return Err(PyRuntimeError::new_err(
                "metrics unavailable before scripted descent is complete",
            ));
        }
        self.metrics.clone().ok_or_else(|| {
            PyRuntimeError::new_err("metrics unavailable: physical GPU evidence is incomplete")
        })
    }

    #[getter]
    fn source(&self) -> &str {
        &self.source
    }

    #[getter]
    fn target_name(&self) -> &str {
        &self.target_name
    }

    #[getter]
    fn current_position(&self) -> Option<(f64, f64, f64)> {
        self.current.map(|p| (p.lon, p.lat, p.altitude))
    }

    #[getter]
    fn rendered_waypoint_count(&self) -> usize {
        self.rendered_waypoints
    }

    #[getter]
    fn source_bounds(&self) -> (f64, f64, f64, f64) {
        self.source_bounds
    }

    #[getter]
    fn source_dimensions(&self) -> (u32, u32) {
        self.source_dimensions
    }

    fn streaming_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = self.renderer.height_streaming_stats(py)?;
        let dict = stats
            .bind(py)
            .downcast::<pyo3::types::PyDict>()
            .map_err(|_| {
                PyRuntimeError::new_err("ORBIS streaming stats were not a native dict")
            })?;
        dict.set_item("source_sample_spacing_m", self.source_sample_spacing_m)?;
        Ok(stats)
    }

    fn __repr__(&self) -> String {
        format!(
            "GlobeScene(target={:?}, lon={:.6}, lat={:.6}, rendered_waypoints={})",
            self.target_name, self.target.lon, self.target.lat, self.rendered_waypoints
        )
    }
}

fn validate_selected_acceptance_adapter(
    adapter: &wgpu::AdapterInfo,
    software_fallback: bool,
) -> PyResult<()> {
    let selected = std::env::var("FORGE3D_RUN_ORBIS_GPU").as_deref() == Ok("1");
    if !selected {
        return Ok(());
    }
    let valid = adapter.vendor == 0x10de
        && adapter.name.to_ascii_lowercase().contains("nvidia")
        && adapter.backend == wgpu::Backend::Vulkan
        && adapter.device_type == wgpu::DeviceType::DiscreteGpu
        && !software_fallback;
    if valid {
        Ok(())
    } else {
        Err(PyRuntimeError::new_err(format!(
            "FORGE3D_RUN_ORBIS_GPU=1 requires a physical NVIDIA Vulkan discrete adapter with software_fallback=false; got name={:?} vendor={:#06x} backend={:?} type={:?} software_fallback={software_fallback}",
            adapter.name, adapter.vendor, adapter.backend, adapter.device_type,
        )))
    }
}

impl GlobeScene {
    fn streaming_snapshot(
        &self,
        py: Python<'_>,
        stats: &PyObject,
    ) -> PyResult<StreamingProgressSnapshot> {
        let dict = stats.bind(py).downcast::<pyo3::types::PyDict>().map_err(|_| {
            PyRuntimeError::new_err("ORBIS streaming stats were not a native dict")
        })?;
        let get_u64 = |name: &str| -> PyResult<u64> {
            dict.get_item(name)?
                .ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "ORBIS streaming stat {name:?} missing"
                    ))
                })?
                .extract::<u64>()
        };
        Ok(StreamingProgressSnapshot {
            bounded_step: get_u64("bounded_steps")?,
            tiles_requested: get_u64("tiles_requested")?,
            loader_completed: get_u64("loader_completed")?,
            tiles_uploaded: get_u64("tiles_uploaded")?,
            pending_ring_tiles: get_u64("pending_ring_tiles")?,
            loader_pending: get_u64("loader_pending")?,
            resident_fine_tiles: get_u64("resident_fine_tiles")?,
            resident_ancestor_tiles: get_u64("resident_ancestor_tiles")?,
            loaded_ring_tiles: get_u64("loaded_ring_tiles")?,
            page_table_updates: get_u64("page_table_updates")?,
            coarse_prefilled: get_u64("coarse_prefilled")?,
            ancestor_fallbacks: get_u64("ancestor_fallbacks")?,
            required_leaf_tiles: get_u64("required_leaf_tiles")?,
        })
    }

    fn record_streaming_evidence(&mut self, py: Python<'_>, stats: &PyObject) -> PyResult<()> {
        let dict = stats.bind(py).downcast::<pyo3::types::PyDict>().map_err(|_| {
            PyRuntimeError::new_err("ORBIS streaming stats were not a native dict")
        })?;
        let get_u64 = |name: &str| -> PyResult<u64> {
            dict.get_item(name)?
                .ok_or_else(|| PyRuntimeError::new_err(format!("ORBIS streaming stat {name:?} missing")))?
                .extract::<u64>()
        };
        let snapshot = self.streaming_snapshot(py, stats)?;
        let pending = snapshot.pending_ring_tiles + snapshot.loader_pending;
        let effective_target_lod = get_u64("effective_target_lod")?;
        if effective_target_lod != u64::from(self.source_lod) {
            return Err(PyRuntimeError::new_err(format!(
                "ORBIS streaming target LOD changed from validated source LOD {} to {effective_target_lod}",
                self.source_lod
            )));
        }
        let coarse_valid = get_u64("coarse_prefilled")? > 0 && get_u64("ancestor_fallbacks")? > 0;
        self.streaming_evidence
            .observe(snapshot)
            .map_err(PyRuntimeError::new_err)?;
        self.streaming_evidence.pending_frames += u64::from(pending > 0);
        self.streaming_evidence.coarse_fallback_frames += u64::from(coarse_valid);
        Ok(())
    }

    fn waypoint_context(&self, waypoint: Waypoint, stage: &str, error: impl std::fmt::Display) -> String {
        waypoint_context_message(&self.source, &self.target_name, waypoint, stage, error)
    }

    fn waypoint_validation_error(
        &self,
        waypoint: Waypoint,
        stage: &str,
        error: impl std::fmt::Display,
    ) -> PyErr {
        PyValueError::new_err(self.waypoint_context(waypoint, stage, error))
    }

    fn waypoint_parse_error(&self, item: &Bound<'_, PyAny>, error: PyErr) -> PyErr {
        if let Ok((lon, lat, altitude, heading, pitch)) =
            item.extract::<(f64, f64, f64, f64, f64)>()
        {
            return PyValueError::new_err(self.waypoint_context(
                Waypoint {
                    lon,
                    lat,
                    altitude,
                    heading,
                    pitch,
                },
                "validation",
                error,
            ));
        }
        if let Ok((lon, lat, altitude)) = item.extract::<(f64, f64, f64)>() {
            return PyValueError::new_err(self.waypoint_context(
                Waypoint {
                    lon,
                    lat,
                    altitude,
                    heading: 0.0,
                    pitch: 0.0,
                },
                "validation",
                error,
            ));
        }
        PyValueError::new_err(format!(
            "ORBIS waypoint parsing failed for COG source {:?}, target {:?}: {error}",
            self.source, self.target_name,
        ))
    }

    fn validate_waypoint_coverage(&self, waypoint: Waypoint) -> PyResult<()> {
        let bounds = self.source_bounds;
        if waypoint.lon < bounds.0
            || waypoint.lon > bounds.2
            || waypoint.lat < bounds.1
            || waypoint.lat > bounds.3
        {
            return Err(coverage_error(
                &self.source,
                &self.target_name,
                waypoint.lon,
                waypoint.lat,
                bounds,
            ));
        }
        Ok(())
    }

    fn prepare_overview_seed(
        &mut self,
        waypoint: Waypoint,
    ) -> PyResult<crate::terrain::tiling::TileId> {
        let tile = if let Some(tile) = self
            .overview_seeds
            .find_covering(waypoint.lon, waypoint.lat)
        {
            tile
        } else {
            let dataset = &self.dataset;
            let (seed, _overview_lod) = load_covered_overview(
                dataset,
                waypoint.lon,
                waypoint.lat,
            )
            .map_err(|error| {
                PyValueError::new_err(self.waypoint_context(
                    waypoint,
                    "overview seed preparation",
                    error,
                ))
            })?;
            let tile = seed.tile;
            self.overview_seeds
                .prepare_with(tile, || Ok(seed))
                .map_err(|error| {
                    PyValueError::new_err(self.waypoint_context(
                        waypoint,
                        "overview seed preparation",
                        error,
                    ))
                })?;
            tile
        };
        let dataset = &self.dataset;
        self.legacy_ground_seeds
            .prepare_with(tile, || {
                source::read_covered_seed(dataset, tile, source::MIN_OVERVIEW_SIZE).and_then(
                    |seed| {
                        seed.ok_or_else(|| {
                            "legacy ground overview tile is not fully covered".to_string()
                        })
                    },
                )
            })
            .map_err(|error| {
                PyValueError::new_err(self.waypoint_context(
                    waypoint,
                    "overview seed preparation",
                    error,
                ))
            })?;
        Ok(tile)
    }

    fn render_waypoint_with_overview(
        &mut self,
        py: Python<'_>,
        waypoint: Waypoint,
        tile: crate::terrain::tiling::TileId,
    ) -> PyResult<Py<crate::Frame>> {
        let legacy = waypoint.altitude == 0.0;
        if tile == self.active_overview_tile && self.ground_legacy_active == legacy {
            return self.render_waypoint(py, waypoint);
        }
        let previous_tile = self.active_overview_tile;
        let previous_legacy = self.ground_legacy_active;
        let (bounds, size, heights) = if legacy {
            let seed = self.legacy_ground_seeds.get(tile).ok_or_else(|| {
                PyRuntimeError::new_err(self.waypoint_context(
                    waypoint,
                    "overview activation",
                    "legacy ground overview seed is missing from the cache",
                ))
            })?;
            (seed.bounds, seed.size, seed.heights.clone())
        } else {
            let seed = self.overview_seeds.get(tile).ok_or_else(|| {
                PyRuntimeError::new_err(self.waypoint_context(
                    waypoint,
                    "overview activation",
                    "prepared overview seed is missing from the cache",
                ))
            })?;
            (seed.bounds, seed.size, seed.heights.clone())
        };
        let activation = self
            .renderer
            .begin_height_streaming_overview_activation(bounds, size, size, &heights)
            .map_err(|error| {
                PyRuntimeError::new_err(self.waypoint_context(
                    waypoint,
                    "overview activation",
                    format!("{error:#}"),
                ))
            })?;
        self.active_overview_tile = tile;
        self.ground_legacy_active = legacy;
        match self.render_waypoint(py, waypoint) {
            Ok(frame) => {
                self.renderer
                    .commit_height_streaming_overview_activation(activation);
                Ok(frame)
            }
            Err(render_error) => {
                self.renderer
                    .rollback_height_streaming_overview_activation(activation);
                self.active_overview_tile = previous_tile;
                self.ground_legacy_active = previous_legacy;
                Err(render_error)
            }
        }
    }

    fn active_overview(&self) -> PyResult<&OverviewSeed> {
        self.overview_seeds
            .get(self.active_overview_tile)
            .ok_or_else(|| PyRuntimeError::new_err("active ORBIS overview seed is missing"))
    }

    fn waypoint_height_source(&self) -> PyResult<&OverviewSeed> {
        let active_tile = self.active_overview()?.tile;
        self.legacy_ground_seeds.get(active_tile).ok_or_else(|| {
            PyRuntimeError::new_err(
                "legacy ground overview seed is missing from the cache".to_string(),
            )
        })
    }

    fn render_waypoint(
        &mut self,
        py: Python<'_>,
        waypoint: Waypoint,
    ) -> PyResult<Py<crate::Frame>> {
        let seed = super::globe::GlobeFrame::globe(
            super::globe::GlobeFrame::WGS84_MEAN_RADIUS_M,
            glam::DVec3::X * super::globe::GlobeFrame::WGS84_MEAN_RADIUS_M,
        )
        .map_err(|error| {
            PyValueError::new_err(self.waypoint_context(waypoint, "ECEF validation", error))
        })?;
        let mut params = self.params.borrow(py).clone();
        params.camera_mode = GLOBE_CAMERA_MODE.to_string();
        params.terrain_span = self.terrain_extent_m;
        let target_sample = self
            .waypoint_height_source()?
            .sample_height(waypoint.lon, waypoint.lat);
        let target_height = target_sample
            .map_err(|error| {
                PyValueError::new_err(self.waypoint_context(
                    waypoint,
                    "ground reference sampling",
                    error,
                ))
            })?;
        let height_range = params.decoded().clamp.height_range;
        let rendered_target_height = crate::terrain::renderer::visibility_buffer::apply_height_curve(
            target_height,
            height_range,
            &params,
        );
        let anchor_altitude = camera_anchor_altitude(
            waypoint.altitude,
            rendered_target_height,
            height_range,
            params.z_scale,
        )
        .map_err(|error| {
            PyValueError::new_err(self.waypoint_context(
                waypoint,
                "ground reference validation",
                error,
            ))
        })?;
        let target_ecef = seed
            .lonlat_alt_to_ecef(
                waypoint.lon,
                waypoint.lat,
                anchor_altitude - waypoint.altitude,
            )
            .map_err(|error| {
                PyValueError::new_err(self.waypoint_context(
                    waypoint,
                    "ECEF validation",
                    error,
                ))
            })?;
        let camera_ecef = oriented_camera_ecef(target_ecef, waypoint).map_err(|error| {
            PyValueError::new_err(self.waypoint_context(waypoint, "ECEF validation", error))
        })?;
        let camera_frame = super::globe::GlobeFrame::globe(
            super::globe::GlobeFrame::WGS84_MEAN_RADIUS_M,
            camera_ecef,
        )
        .map_err(|error| {
            PyValueError::new_err(self.waypoint_context(waypoint, "ECEF validation", error))
        })?;
        let pose = camera_pose(camera_frame, target_ecef, waypoint).map_err(|error| {
            PyValueError::new_err(self.waypoint_context(waypoint, "camera validation", error))
        })?;
        params.cam_phi_deg = pose.phi_deg;
        params.cam_theta_deg = pose.theta_deg;
        params.cam_radius = pose.radius;
        params.cam_target = pose.target;
        params.clip = (
            ((waypoint.altitude * 0.00025) as f32).max(0.05),
            (pose.radius + self.terrain_extent_m * 4.0).max(10_000.0),
        );
        let (eye, view, projection) =
            crate::terrain::renderer::TerrainScene::build_camera_matrices(&params);
        if !eye.is_finite() || !view.is_finite() || !projection.is_finite() {
            return Err(PyValueError::new_err(self.waypoint_context(
                waypoint,
                "camera validation",
                format!("camera eye/matrices must be finite; eye={eye:?}"),
            )));
        }

        // Exactly one bounded poll/upload step per rendered waypoint. This is
        // intentionally not a convergence loop.
        let stream_result = self.renderer.stream_height_tiles_globe_focus(
            py,
            camera_ecef,
            target_ecef,
            64,
        );
        let stream_stats = contextualize_waypoint_runtime(
            &self.source,
            &self.target_name,
            waypoint,
            "streaming",
            stream_result,
        )?;
        let rows = {
            let source = self.waypoint_height_source()?;
            source
                .heights
                .chunks_exact(source.size as usize)
                .map(|row| row.to_vec())
                .collect::<Vec<_>>()
        };
        let heights = PyArray2::from_vec2_bound(py, &rows)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        self.renderer
            .set_height_detail_blend_override(Some(if waypoint.altitude == 0.0 {
                0.0
            } else {
                1.0
            }));
        let render_result = self.renderer.render_terrain_pbr_pom(
            py,
            &self.material_set.borrow(py),
            &self.env_maps.borrow(py),
            &params,
            heights.readonly(),
            None,
            None,
            self.rendered_waypoints as f32,
            None,
            None,
        );
        self.renderer.set_height_detail_blend_override(None);
        let frame = contextualize_waypoint_runtime(
            &self.source,
            &self.target_name,
            waypoint,
            "render",
            render_result,
        )?;
        // Count progress only after both the one bounded stream step and the
        // corresponding render submission have completed successfully.
        if self.renderer.orbis_descent_active() {
            self.record_streaming_evidence(py, &stream_stats)?;
        }
        self.current = Some(waypoint);
        self.rendered_waypoints += 1;
        self.last_frame = Some(frame.clone_ref(py));
        Ok(frame)
    }

    fn run_orbis_physical_probe(&mut self, py: Python<'_>) -> PyResult<()> {
        let waypoint = self.current.unwrap_or(self.target);
        let radius = crate::terrain::clipmap::globe::GlobeFrame::WGS84_MEAN_RADIUS_M;
        let seed = super::globe::GlobeFrame::globe(radius, glam::DVec3::X * radius)
            .map_err(|error| {
                PyRuntimeError::new_err(format!("ORBIS probe base frame failed: {error}"))
            })?;
        let surface = seed
            .lonlat_alt_to_ecef(waypoint.lon, waypoint.lat, 0.0)
            .map_err(|error| {
                PyRuntimeError::new_err(format!("ORBIS probe base coordinate failed: {error}"))
            })?;
        let probe_base = safe_orbis_probe_base(surface).map_err(PyRuntimeError::new_err)?;
        let base_stream = self.renderer.stream_height_tiles_globe(
            py,
            (probe_base.x, probe_base.y, probe_base.z),
            0,
        );
        contextualize_waypoint_runtime(
            &self.source,
            &self.target_name,
            waypoint,
            "physical probe base streaming",
            base_stream,
        )?;

        let (anchor, threshold) = self.renderer.orbis_reanchor_state().map_err(|error| {
            PyRuntimeError::new_err(format!("ORBIS reanchor probe state failed: {error:#}"))
        })?;
        if anchor != probe_base {
            return Err(PyRuntimeError::new_err(
                "ORBIS globe streamer did not reanchor at the requested elevated probe base",
            ));
        }
        let [camera_0, camera_1] =
            orbis_probe_cameras(anchor, threshold).map_err(PyRuntimeError::new_err)?;
        let anchor_altitude = anchor.length() - radius;
        for (frame_index, camera) in [camera_0, camera_1].into_iter().enumerate() {
            let stream = self.renderer.stream_height_tiles_globe(
                py,
                (camera.x, camera.y, camera.z),
                0,
            );
            contextualize_waypoint_runtime(
                &self.source,
                &self.target_name,
                waypoint,
                "physical probe streaming",
                stream,
            )?;
            let camera_altitude = camera.length() - radius;
            let actual_anchor_altitude = if frame_index == 0 {
                anchor_altitude
            } else {
                camera_altitude
            };
            let mut params = self.params.borrow(py).clone();
            params.camera_mode = GLOBE_CAMERA_MODE.to_string();
            params.terrain_span = self.terrain_extent_m;
            params.cam_phi_deg = 0.0;
            params.cam_theta_deg = 0.0;
            params.cam_radius = camera_altitude as f32;
            params.cam_target = [0.0, 0.0, -(actual_anchor_altitude as f32)];
            params.clip = (
                0.1,
                (camera_altitude as f32 + self.terrain_extent_m * 4.0).max(10_000.0),
            );
            self.renderer
                .begin_orbis_physical_capture(camera)
                .map_err(|error| PyRuntimeError::new_err(format!(
                    "ORBIS physical frame {frame_index} setup failed: {error:#}"
                )))?;
            let overview = self.active_overview()?;
            let rows = overview
                .heights
                .chunks_exact(overview.size as usize)
                .map(|row| row.to_vec())
                .collect::<Vec<_>>();
            let heights = PyArray2::from_vec2_bound(py, &rows)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            let rendered = self.renderer.render_terrain_pbr_pom(
                py,
                &self.material_set.borrow(py),
                &self.env_maps.borrow(py),
                &params,
                heights.readonly(),
                None,
                None,
                self.rendered_waypoints as f32 + frame_index as f32,
                None,
                None,
            );
            contextualize_waypoint_runtime(
                &self.source,
                &self.target_name,
                waypoint,
                "physical probe render",
                rendered,
            )?;
        }
        Ok(())
    }
}

fn safe_orbis_probe_base(surface_ecef: glam::DVec3) -> Result<glam::DVec3, String> {
    let radius = crate::terrain::clipmap::globe::GlobeFrame::WGS84_MEAN_RADIUS_M;
    let surface_radius = surface_ecef.length();
    if !surface_ecef.is_finite() || !surface_radius.is_finite() || surface_radius <= 0.0 {
        return Err("ORBIS probe surface coordinate must be finite and non-zero".to_string());
    }
    Ok(surface_ecef / surface_radius * (radius + ORBIS_PROBE_BASE_ALTITUDE_M))
}

fn orbis_probe_cameras(
    anchor: glam::DVec3,
    threshold: f64,
) -> Result<[glam::DVec3; 2], String> {
    if !anchor.is_finite() || anchor.length_squared() == 0.0 {
        return Err("ORBIS elevated probe anchor must be finite and non-zero".to_string());
    }
    if !threshold.is_finite() || threshold <= MICRO_STEP_M {
        return Err("ORBIS clipmap reanchor threshold is too small for the 2 mm probe".to_string());
    }
    let up = anchor.normalize();
    let camera_0 = anchor + up * (threshold - MICRO_STEP_M * 0.5);
    let camera_1 = anchor + up * (threshold + MICRO_STEP_M * 0.5);
    if ((camera_1 - camera_0).length() - MICRO_STEP_M).abs() > 1.0e-9 {
        return Err("ORBIS physical probe cameras are not separated by exactly 2 mm".to_string());
    }
    Ok([camera_0, camera_1])
}

fn waypoint_context_message(
    source: &str,
    target_name: &str,
    waypoint: Waypoint,
    stage: &str,
    error: impl std::fmt::Display,
) -> String {
    format!(
        "ORBIS {stage} failed for COG source {source:?}, target {target_name:?}, waypoint ({}, {}, {} m): {error}",
        waypoint.lon, waypoint.lat, waypoint.altitude,
    )
}

fn contextualize_waypoint_runtime<T>(
    source: &str,
    target_name: &str,
    waypoint: Waypoint,
    stage: &str,
    result: PyResult<T>,
) -> PyResult<T> {
    result.map_err(|error| {
        PyRuntimeError::new_err(waypoint_context_message(
            source,
            target_name,
            waypoint,
            stage,
            error,
        ))
    })
}

fn validate_target(lon: f64, lat: f64, name: &str) -> PyResult<()> {
    if !lon.is_finite() || !lat.is_finite() {
        return Err(PyValueError::new_err(
            "target longitude and latitude must be finite",
        ));
    }
    if !(-180.0..=180.0).contains(&lon) {
        return Err(PyValueError::new_err(
            "target longitude must be in [-180, 180]",
        ));
    }
    if !(-90.0..=90.0).contains(&lat) {
        return Err(PyValueError::new_err(
            "target latitude must be in [-90, 90]",
        ));
    }
    if name.trim().is_empty() {
        return Err(PyValueError::new_err("target_name must not be empty"));
    }
    Ok(())
}

fn parse_waypoint(value: &Bound<'_, PyAny>) -> PyResult<Waypoint> {
    if let Ok((lon, lat, altitude, heading, pitch)) =
        value.extract::<(f64, f64, f64, f64, f64)>()
    {
        return validate_waypoint(lon, lat, altitude, heading, pitch);
    }
    if let Ok((lon, lat, altitude)) = value.extract::<(f64, f64, f64)>() {
        return validate_waypoint(lon, lat, altitude, 0.0, 0.0);
    }
    Err(PyValueError::new_err(
        "waypoints must contain (lon, lat, altitude) or (lon, lat, altitude, heading, pitch) tuples",
    ))
}

fn validate_waypoint(
    lon: f64,
    lat: f64,
    altitude: f64,
    heading: f64,
    pitch: f64,
) -> PyResult<Waypoint> {
    validate_target(lon, lat, "waypoint")?;
    if !altitude.is_finite() || !(0.0..=MAX_SUPPORTED_ALTITUDE_M).contains(&altitude) {
        return Err(PyValueError::new_err(
            "waypoint altitude must be finite and in [0, 408000] metres",
        ));
    }
    if !heading.is_finite() {
        return Err(PyValueError::new_err("waypoint heading must be finite"));
    }
    if !pitch.is_finite() || !(0.0..90.0).contains(&pitch) {
        return Err(PyValueError::new_err(
            "waypoint pitch must be finite and in [0, 90) degrees",
        ));
    }
    Ok(Waypoint {
        lon,
        lat,
        altitude,
        heading: heading.rem_euclid(360.0),
        pitch,
    })
}

fn oriented_camera_ecef(
    target_ecef: glam::DVec3,
    waypoint: Waypoint,
) -> Result<glam::DVec3, String> {
    let up = target_ecef
        .try_normalize()
        .ok_or_else(|| "camera target ECEF must be non-zero".to_string())?;
    let lon = waypoint.lon.to_radians();
    let lat = waypoint.lat.to_radians();
    let east = glam::DVec3::new(-lon.sin(), lon.cos(), 0.0);
    let north = glam::DVec3::new(
        -lat.sin() * lon.cos(),
        -lat.sin() * lon.sin(),
        lat.cos(),
    );
    let heading = waypoint.heading.to_radians();
    let forward = east * heading.sin() + north * heading.cos();
    let horizontal = waypoint.altitude * waypoint.pitch.to_radians().tan();
    let camera = target_ecef + up * waypoint.altitude - forward * horizontal;
    if camera.is_finite() {
        Ok(camera)
    } else {
        Err("oriented camera ECEF is non-finite".to_string())
    }
}

fn camera_pose(
    frame: super::globe::GlobeFrame,
    target_ecef: glam::DVec3,
    waypoint: Waypoint,
) -> Result<CameraPose, String> {
    if waypoint.altitude == 0.0 {
        // At ground the exactly representable target and offset cancel
        // bit-for-bit. The vertical view uses the renderer's +Y fallback up
        // vector, so the look-at basis remains nondegenerate.
        return Ok(CameraPose {
            radius: 1.0,
            phi_deg: 0.0,
            theta_deg: 0.0,
            target: [0.0, 0.0, -1.0],
        });
    }
    let target = frame
        .camera_relative(target_ecef)
        .map_err(|error| format!("camera target rebase failed: {error}"))?
        .position;
    let offset = -target;
    let radius = offset.length();
    if !radius.is_finite() || radius <= 0.0 {
        return Err("oriented camera radius must be finite and positive".to_string());
    }
    Ok(CameraPose {
        radius,
        phi_deg: offset.y.atan2(offset.x).to_degrees(),
        theta_deg: (offset.z / radius).clamp(-1.0, 1.0).acos().to_degrees(),
        target: target.to_array(),
    })
}

fn camera_anchor_altitude(
    requested_altitude: f64,
    rendered_target_height: f32,
    height_range: (f32, f32),
    z_scale: f32,
) -> Result<f64, String> {
    if !requested_altitude.is_finite()
        || !rendered_target_height.is_finite()
        || !height_range.0.is_finite()
        || !height_range.1.is_finite()
        || height_range.0 > height_range.1
        || !z_scale.is_finite()
        || z_scale <= 0.0
    {
        return Err("ground-relative camera inputs must be finite and ordered".to_string());
    }
    let altitude = requested_altitude + f64::from(rendered_target_height * z_scale);
    if !altitude.is_finite() {
        return Err("ground-relative camera altitude is not finite".to_string());
    }
    Ok(altitude)
}

fn default_descent_altitudes() -> Vec<f64> {
    let log_start = (DEFAULT_START_ALTITUDE_M + 1.0).ln();
    (0..DEFAULT_WAYPOINT_COUNT)
        .map(|index| {
            if index == 0 {
                DEFAULT_START_ALTITUDE_M
            } else if index + 1 == DEFAULT_WAYPOINT_COUNT {
                0.0
            } else {
                let t = index as f64 / (DEFAULT_WAYPOINT_COUNT - 1) as f64;
                (log_start * (1.0 - t)).exp() - 1.0
            }
        })
        .collect()
}

pub fn register_globe_scene_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GlobeScene>()?;
    m.add_class::<GlobeMetrics>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_waypoints_are_exact_logarithmic_iss_to_ground() {
        let altitudes = default_descent_altitudes();
        assert_eq!(altitudes.len(), DEFAULT_WAYPOINT_COUNT);
        assert_eq!(altitudes[0], DEFAULT_START_ALTITUDE_M);
        assert_eq!(altitudes.last().copied(), Some(0.0));
        for pair in altitudes.windows(2) {
            assert!(pair[0] > pair[1]);
        }
        let ratios = altitudes
            .windows(2)
            .map(|pair| (pair[0] + 1.0) / (pair[1] + 1.0))
            .collect::<Vec<_>>();
        assert!(ratios.iter().all(|ratio| (ratio - ratios[0]).abs() < 1.0e-10));
    }

    #[test]
    fn waypoint_altitude_is_measured_above_the_rendered_target_surface() {
        assert_eq!(
            camera_anchor_altitude(1_000.0, 4_380.0, (0.0, 5_000.0), 1.0).unwrap(),
            5_380.0
        );
        assert_eq!(
            camera_anchor_altitude(0.0, 2_500.0, (0.0, 5_000.0), 1.0).unwrap(),
            2_500.0
        );
        assert!(camera_anchor_altitude(0.0, f32::NAN, (0.0, 5_000.0), 1.0).is_err());
    }

    #[test]
    fn target_tile_contains_cardinal_and_rainier_points() {
        source::assert_test_points_in_tiles();
    }

    #[test]
    fn ground_camera_eye_is_exactly_local_anchor_with_nondegenerate_view() {
        let waypoint = Waypoint {
            lon: -121.7603,
            lat: 46.8523,
            altitude: 0.0,
            heading: 0.0,
            pitch: 0.0,
        };
        let radius = crate::terrain::clipmap::globe::GlobeFrame::WGS84_MEAN_RADIUS_M;
        let frame =
            crate::terrain::clipmap::globe::GlobeFrame::globe(radius, glam::DVec3::X * radius)
                .unwrap();
        let pose = camera_pose(frame, glam::DVec3::X * radius, waypoint).unwrap();
        let phi = pose.phi_deg.to_radians();
        let theta = pose.theta_deg.to_radians();
        let offset = glam::Vec3::new(
            pose.radius * theta.sin() * phi.cos(),
            pose.radius * theta.sin() * phi.sin(),
            pose.radius * theta.cos(),
        );
        let target = glam::Vec3::from_array(pose.target);
        let eye = target + offset;
        assert_eq!(eye, glam::Vec3::ZERO, "ground eye must equal anchor bit-exactly");
        assert!((target - eye).length() > 0.5);
        assert!(glam::Mat4::look_at_rh(eye, target, glam::Vec3::Y).is_finite());
    }

    #[test]
    fn physical_probe_uses_safe_elevated_base_and_exact_threshold_crossing() {
        let radius = crate::terrain::clipmap::globe::GlobeFrame::WGS84_MEAN_RADIUS_M;
        let surface = glam::DVec3::new(1.0, 2.0, 3.0).normalize() * radius;
        let base = safe_orbis_probe_base(surface).unwrap();
        assert!((base.length() - radius - ORBIS_PROBE_BASE_ALTITUDE_M).abs() < 1.0e-9);
        assert!((base.normalize() - surface.normalize()).length() < 1.0e-15);

        let threshold = 32.0;
        let [camera_0, camera_1] = orbis_probe_cameras(base, threshold).unwrap();
        assert!(((camera_1 - camera_0).length() - MICRO_STEP_M).abs() < 1.0e-9);
        assert!((camera_0 - base).length() < threshold);
        assert!((camera_1 - base).length() > threshold);

        let operand_scale = camera_1.length() - radius;
        let quantization_bound = f64::from(f32::EPSILON) * operand_scale + 1.0e-6;
        assert!(quantization_bound < MICRO_STEP_M * 0.5);
    }

    #[test]
    fn forced_stream_and_render_failures_keep_full_waypoint_context() {
        pyo3::prepare_freethreaded_python();
        let waypoint = Waypoint {
            lon: -121.7603,
            lat: 46.8523,
            altitude: 1_000.0,
            heading: 0.0,
            pitch: 0.0,
        };
        for stage in ["streaming", "render"] {
            let result: PyResult<()> = contextualize_waypoint_runtime(
                "rainier.tif",
                "Mount Rainier",
                waypoint,
                stage,
                Err(PyRuntimeError::new_err("forced failure")),
            );
            let message = result.unwrap_err().to_string();
            for required in [
                stage,
                "rainier.tif",
                "Mount Rainier",
                "-121.7603",
                "46.8523",
                "1000",
                "forced failure",
            ] {
                assert!(message.contains(required), "missing {required:?}: {message}");
            }
        }
    }

}
