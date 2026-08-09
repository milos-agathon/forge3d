//! Native ORBIS scene orchestration exposed to Python.

use crate::terrain::cog::py_bindings::PyCogDataset;
use crate::terrain::planetary_tiles::global_tile_lonlat_bounds;
use crate::terrain::renderer::TerrainRenderer;
use numpy::{PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

mod metrics;
mod source;

use metrics::GlobeMetrics;
use source::{
    coverage_error, default_render_params, extract_source_path, load_covered_overview,
    normalize_source, overview_coverage_error, source_error, tile_extent_m,
};

const DEFAULT_START_ALTITUDE_M: f64 = 408_000.0;
const MAX_SUPPORTED_ALTITUDE_M: f64 = DEFAULT_START_ALTITUDE_M;
const DEFAULT_WAYPOINT_COUNT: usize = 25;
const OVERVIEW_SIZE: u32 = source::OVERVIEW_SIZE;
const GPU_VISIBLE_BUDGET: u64 = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Waypoint {
    lon: f64,
    lat: f64,
    altitude: f64,
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
    _dataset: PyCogDataset,
    material_set: Py<crate::render::material_set::MaterialSet>,
    env_maps: Py<crate::lighting::ibl_wrapper::IBL>,
    params: Py<crate::terrain::render_params::TerrainRenderParams>,
    overview: Vec<f32>,
    overview_size: u32,
    terrain_extent_m: f32,
    source_bounds: (f64, f64, f64, f64),
    overview_bounds: (f64, f64, f64, f64),
    current: Option<Waypoint>,
    last_frame: Option<Py<crate::Frame>>,
    rendered_waypoints: usize,
    descent_complete: bool,
    metrics: GlobeMetrics,
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
        let (overview, _source_lod, tile) =
            load_covered_overview(&dataset, target_lon, target_lat).map_err(|error| {
                source_error(&cog_source, target_name, target_lon, target_lat, error)
            })?;
        let tile_bounds = global_tile_lonlat_bounds(tile).map_err(PyValueError::new_err)?;
        let terrain_extent_m = tile_extent_m(tile_bounds, target_lat);

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
        renderer.enable_height_streaming_cog(
            &dataset,
            terrain_extent_m,
            4,
            32,
            6,
            64,
            8,
            2,
            false,
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
            },
            target_name: target_name.trim().to_string(),
            renderer,
            _dataset: dataset,
            material_set,
            env_maps,
            params,
            overview,
            overview_size: OVERVIEW_SIZE,
            terrain_extent_m,
            source_bounds: bounds,
            overview_bounds: tile_bounds,
            current: None,
            last_frame: None,
            rendered_waypoints: 0,
            descent_complete: false,
            metrics: GlobeMetrics::unmeasured(),
        })
    }

    #[staticmethod]
    fn default_descent_altitudes() -> Vec<f64> {
        default_descent_altitudes()
    }

    fn fly_to(
        &mut self,
        py: Python<'_>,
        lon: f64,
        lat: f64,
        altitude: f64,
    ) -> PyResult<Py<crate::Frame>> {
        let waypoint = validate_waypoint(lon, lat, altitude).map_err(|error| {
            self.waypoint_validation_error(lon, lat, altitude, "validation", error)
        })?;
        self.validate_waypoint_coverage(waypoint).map_err(|error| {
            self.waypoint_validation_error(lon, lat, altitude, "coverage validation", error)
        })?;
        self.validate_waypoint_overview(waypoint).map_err(|error| {
            self.waypoint_validation_error(lon, lat, altitude, "overview validation", error)
        })?;
        self.render_waypoint(py, waypoint)
    }

    #[pyo3(signature = (waypoints=None))]
    fn scripted_descent(
        &mut self,
        py: Python<'_>,
        waypoints: Option<Vec<(f64, f64, f64)>>,
    ) -> PyResult<()> {
        let path = match waypoints {
            Some(values) => {
                if values.is_empty() {
                    return Err(PyValueError::new_err("waypoints must not be empty"));
                }
                let path = values
                    .into_iter()
                    .map(|(lon, lat, altitude)| {
                        validate_waypoint(lon, lat, altitude).map_err(|error| {
                            self.waypoint_validation_error(
                                lon,
                                lat,
                                altitude,
                                "validation",
                                error,
                            )
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
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
        // Validate the entire default or custom path before issuing streaming
        // or GPU work, so one bad late waypoint cannot leave a half-applied
        // descent and every frame uses the seeded overview honestly.
        for waypoint in &path {
            self.validate_waypoint_coverage(*waypoint).map_err(|error| {
                self.waypoint_validation_error(
                    waypoint.lon,
                    waypoint.lat,
                    waypoint.altitude,
                    "coverage validation",
                    error,
                )
            })?;
            self.validate_waypoint_overview(*waypoint).map_err(|error| {
                self.waypoint_validation_error(
                    waypoint.lon,
                    waypoint.lat,
                    waypoint.altitude,
                    "overview validation",
                    error,
                )
            })?;
        }
        for waypoint in path {
            self.render_waypoint(py, waypoint)?;
        }
        self.descent_complete = true;
        Ok(())
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
        let _ = &self.metrics;
        Err(PyRuntimeError::new_err(
            "metrics unmeasured: Task 8 physical GPU probes have not completed",
        ))
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

    fn __repr__(&self) -> String {
        format!(
            "GlobeScene(target={:?}, lon={:.6}, lat={:.6}, rendered_waypoints={})",
            self.target_name, self.target.lon, self.target.lat, self.rendered_waypoints
        )
    }
}

impl GlobeScene {
    fn waypoint_context(&self, waypoint: Waypoint, stage: &str, error: impl std::fmt::Display) -> String {
        waypoint_context_message(&self.source, &self.target_name, waypoint, stage, error)
    }

    fn waypoint_validation_error(
        &self,
        lon: f64,
        lat: f64,
        altitude: f64,
        stage: &str,
        error: impl std::fmt::Display,
    ) -> PyErr {
        PyValueError::new_err(self.waypoint_context(
            Waypoint { lon, lat, altitude },
            stage,
            error,
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

    fn validate_waypoint_overview(&self, waypoint: Waypoint) -> PyResult<()> {
        let bounds = self.overview_bounds;
        if waypoint.lon < bounds.0
            || waypoint.lon > bounds.2
            || waypoint.lat < bounds.1
            || waypoint.lat > bounds.3
        {
            return Err(overview_coverage_error(
                &self.source,
                &self.target_name,
                waypoint.lon,
                waypoint.lat,
                bounds,
            ));
        }
        Ok(())
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
        let camera_ecef = seed
            .lonlat_alt_to_ecef(waypoint.lon, waypoint.lat, waypoint.altitude)
            .filter(|ecef| ecef.is_finite())
            .ok_or_else(|| {
                PyValueError::new_err(self.waypoint_context(
                    waypoint,
                    "ECEF validation",
                    "waypoint cannot be represented as finite ECEF",
                ))
            })?;

        let mut params = self.params.borrow(py).clone();
        params.camera_mode = "clipmap:4:32:32:10:0.3:zup".to_string();
        params.terrain_span = self.terrain_extent_m;
        let pose = camera_pose(waypoint);
        params.cam_phi_deg = pose.phi_deg;
        params.cam_theta_deg = pose.theta_deg;
        params.cam_radius = pose.radius;
        params.cam_target = pose.target;
        params.clip = (
            0.1,
            (waypoint.altitude as f32 + self.terrain_extent_m * 4.0).max(10_000.0),
        );
        let (eye, view, projection) =
            crate::terrain::renderer::TerrainScene::build_camera_matrices(&params);
        if !eye.is_finite()
            || !view.is_finite()
            || !projection.is_finite()
            || eye != glam::Vec3::ZERO
        {
            return Err(PyValueError::new_err(self.waypoint_context(
                waypoint,
                "camera validation",
                format!("camera eye/matrices must be finite and eye must equal local anchor; eye={eye:?}"),
            )));
        }

        // Exactly one bounded poll/upload step per rendered waypoint. This is
        // intentionally not a convergence loop.
        let stream_result = self.renderer.stream_height_tiles_globe(
            py,
            (camera_ecef.x, camera_ecef.y, camera_ecef.z),
            8,
        );
        contextualize_waypoint_runtime(
            &self.source,
            &self.target_name,
            waypoint,
            "streaming",
            stream_result,
        )?;

        let rows = self
            .overview
            .chunks_exact(self.overview_size as usize)
            .map(|row| row.to_vec())
            .collect::<Vec<_>>();
        let heights = PyArray2::from_vec2_bound(py, &rows)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
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
        let frame = contextualize_waypoint_runtime(
            &self.source,
            &self.target_name,
            waypoint,
            "render",
            render_result,
        )?;
        self.current = Some(waypoint);
        self.rendered_waypoints += 1;
        self.last_frame = Some(frame.clone_ref(py));
        Ok(frame)
    }
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

fn validate_waypoint(lon: f64, lat: f64, altitude: f64) -> PyResult<Waypoint> {
    validate_target(lon, lat, "waypoint")?;
    if !altitude.is_finite() || !(0.0..=MAX_SUPPORTED_ALTITUDE_M).contains(&altitude) {
        return Err(PyValueError::new_err(
            "waypoint altitude must be finite and in [0, 408000] metres",
        ));
    }
    Ok(Waypoint { lon, lat, altitude })
}

fn camera_pose(waypoint: Waypoint) -> CameraPose {
    if waypoint.altitude > 0.0 {
        CameraPose {
            radius: waypoint.altitude as f32,
            phi_deg: 0.0,
            theta_deg: 0.0,
            target: [0.0, 0.0, -(waypoint.altitude as f32)],
        }
    } else {
        // At ground the exactly representable target and offset cancel
        // bit-for-bit. The vertical view uses the renderer's +Y fallback up
        // vector, so the look-at basis remains nondegenerate.
        CameraPose {
            radius: 1.0,
            phi_deg: 0.0,
            theta_deg: 0.0,
            target: [0.0, 0.0, -1.0],
        }
    }
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
    fn target_tile_contains_cardinal_and_rainier_points() {
        source::assert_test_points_in_tiles();
    }

    #[test]
    fn ground_camera_eye_is_exactly_local_anchor_with_nondegenerate_view() {
        let waypoint = Waypoint { lon: -121.7603, lat: 46.8523, altitude: 0.0 };
        let pose = camera_pose(waypoint);
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
    fn forced_stream_and_render_failures_keep_full_waypoint_context() {
        pyo3::prepare_freethreaded_python();
        let waypoint = Waypoint {
            lon: -121.7603,
            lat: 46.8523,
            altitude: 1_000.0,
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
