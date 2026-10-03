// src/py_functions/splat.rs
// SPLAT-FUSED Python seam: Gaussian splat loading and the fused
// splat + LiDAR + terrain ReSTIR render (HybridPathTracer::render_fused),
// plus the hard-proxy path-traced reference the acceptance tests score
// against. The typed public surface lives in python/forge3d/splat.py.
// RELEVANT FILES: src/splat/mod.rs, src/splat/fusion.rs,
//                 src/path_tracing/hybrid_compute/render_fused.rs,
//                 python/forge3d/splat.py

use super::super::*;
use std::sync::Arc;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};

use crate::path_tracing::fused_reference::{
    render_fused_reference, FusedReferenceDesc, ReceiverClass,
};
use crate::path_tracing::hybrid_compute::{
    FusedRenderDesc, FusedRenderOutput, FusedTerrainDesc, HybridPathTracer,
};
use crate::splat::bvh::Aabb;
use crate::splat::fusion::{
    FusedScene, FusionParams, MediaParams, PagingPolicy, PagingStats, ShadingParams,
};
use crate::splat::kernel::{ray_gaussian, splat_transmittance};
use crate::splat::stream::{
    write_splat_page_store, write_synthetic_point_field, CopcPageSource, PageSource,
    PageStoreFile, PageStoreSummary, PointCloudFrame, SplatCloudSource, SyntheticFieldDesc,
};
use crate::splat::{inverse_covariance, GaussianSplatCloud, ShRest};

/// Anisotropic 3D Gaussian splat cloud (structure of arrays) with the packed
/// inverse covariance of every splat precomputed.
#[pyclass(module = "forge3d._forge3d", name = "GaussianSplatCloud")]
pub struct PyGaussianSplatCloud {
    pub(crate) inner: Arc<GaussianSplatCloud>,
}

fn rows<const N: usize>(
    array: &PyReadonlyArray2<'_, f32>,
    name: &str,
) -> PyResult<Vec<[f32; N]>> {
    let view = array.as_array();
    if view.shape()[1] != N {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (N, {N}), got {:?}",
            view.shape()
        )));
    }
    Ok(view
        .rows()
        .into_iter()
        .map(|row| std::array::from_fn(|k| row[k]))
        .collect())
}

fn array2<'py, const N: usize>(
    py: Python<'py>,
    values: &[[f32; N]],
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    let flat: Vec<f32> = values.iter().flatten().copied().collect();
    PyArray1::from_vec_bound(py, flat).reshape([values.len(), N])
}

#[pymethods]
impl PyGaussianSplatCloud {
    /// Build a cloud from attribute arrays: positions (N, 3), per-axis sigmas
    /// (N, 3), rotation quaternions (N, 4) as (w, x, y, z), opacities (N,) in
    /// [0, 1], DC SH coefficients (N, 3) and optional higher SH bands
    /// (N, K, 3) with K = 3, 8 or 15.
    #[staticmethod]
    #[pyo3(signature = (positions, scales, rotations, opacities, sh0, sh_rest = None))]
    fn from_arrays(
        positions: PyReadonlyArray2<'_, f32>,
        scales: PyReadonlyArray2<'_, f32>,
        rotations: PyReadonlyArray2<'_, f32>,
        opacities: PyReadonlyArray1<'_, f32>,
        sh0: PyReadonlyArray2<'_, f32>,
        sh_rest: Option<PyReadonlyArray3<'_, f32>>,
    ) -> PyResult<Self> {
        let rest = match sh_rest {
            None => None,
            Some(array) => {
                let view = array.as_array();
                let degree = match (view.shape()[1], view.shape()[2]) {
                    (3, 3) => 1,
                    (8, 3) => 2,
                    (15, 3) => 3,
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "sh_rest must have shape (N, 3|8|15, 3), got {:?}",
                            view.shape()
                        )))
                    }
                };
                let flat: Vec<f32> = view.iter().copied().collect();
                Some(ShRest {
                    degree,
                    coeffs: flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
                })
            }
        };
        let cloud = GaussianSplatCloud::from_parts(
            rows::<3>(&positions, "positions")?,
            rows::<3>(&scales, "scales")?,
            rows::<4>(&rotations, "rotations")?,
            opacities.as_array().iter().copied().collect(),
            rows::<3>(&sh0, "sh0")?,
            rest,
        )?;
        Ok(Self {
            inner: Arc::new(cloud),
        })
    }

    /// Number of splats.
    #[getter]
    fn count(&self) -> usize {
        self.inner.len()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Stored spherical-harmonic degree (0 = DC only).
    #[getter]
    fn sh_degree(&self) -> u32 {
        self.inner.sh_degree()
    }

    /// Host bytes the cloud keeps resident (registered with the memory tracker).
    #[getter]
    fn byte_size(&self) -> usize {
        self.inner.byte_size()
    }

    /// ((min_x, min_y, min_z), (max_x, max_y, max_z)) of the 3-sigma proxies.
    #[getter]
    fn bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        self.inner.bounds()
    }

    #[getter]
    fn positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        array2(py, &self.inner.positions)
    }

    /// Per-axis standard deviations (already exponentiated).
    #[getter]
    fn scales<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        array2(py, &self.inner.scales)
    }

    /// Unit quaternions (w, x, y, z).
    #[getter]
    fn rotations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        array2(py, &self.inner.rotations)
    }

    #[getter]
    fn opacities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice_bound(py, &self.inner.opacities)
    }

    /// DC spherical-harmonic coefficients.
    #[getter]
    fn sh0<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        array2(py, &self.inner.sh0)
    }

    /// Packed upper-triangular inverse covariance (xx, xy, xz, yy, yz, zz).
    #[getter]
    fn inverse_covariance<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        array2(py, self.inner.inv_cov())
    }

    /// Linear colour of splat `index` seen along unit `direction`.
    fn color(&self, index: usize, direction: [f32; 3]) -> PyResult<[f32; 3]> {
        if index >= self.inner.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "splat index {index} out of range for {} splats",
                self.inner.len()
            )));
        }
        Ok(self.inner.color(index, direction))
    }

    /// Write the cloud as a binary 3DGS `.ply`.
    fn save(&self, path: &str) -> PyResult<()> {
        Ok(crate::splat::save_gaussian_splats(path, &self.inner)?)
    }

    /// Write the cloud as an out-of-core page store; returns its summary.
    #[pyo3(signature = (path, page_capacity = 4096))]
    fn write_page_store(
        &self,
        py: Python<'_>,
        path: &str,
        page_capacity: u32,
    ) -> PyResult<Py<PyDict>> {
        summary_dict(py, write_splat_page_store(path, &self.inner, page_capacity)?)
    }

    fn __repr__(&self) -> String {
        format!(
            "GaussianSplatCloud(count={}, sh_degree={})",
            self.inner.len(),
            self.inner.sh_degree()
        )
    }
}

fn summary_dict(py: Python<'_>, summary: PageStoreSummary) -> PyResult<Py<PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("page_count", summary.page_count)?;
    d.set_item("logical_primitives", summary.logical_primitives)?;
    d.set_item("payload_bytes", summary.payload_bytes)?;
    d.set_item("file_bytes", summary.file_bytes)?;
    Ok(d.into())
}

/// Load a 3D Gaussian Splatting `.ply` (`x y z`, `scale_0..2`, `rot_0..3`,
/// `opacity`, `f_dc_0..2`, `f_rest_*`).
#[pyfunction]
pub(crate) fn load_gaussian_splats(path: &str) -> PyResult<PyGaussianSplatCloud> {
    Ok(PyGaussianSplatCloud {
        inner: Arc::new(crate::splat::load_gaussian_splats(path)?),
    })
}

/// Build an out-of-core splat page store from a `.ply` that never has to fit
/// in memory (external bucketed sort bounded by `memory_budget_bytes`).
#[pyfunction]
#[pyo3(signature = (ply_path, out_path, page_capacity = 4096, memory_budget_bytes = 268_435_456))]
pub(crate) fn build_splat_page_store(
    py: Python<'_>,
    ply_path: &str,
    out_path: &str,
    page_capacity: u32,
    memory_budget_bytes: u64,
) -> PyResult<Py<PyDict>> {
    let summary = py.allow_threads(|| {
        crate::splat::stream::build_splat_page_store_from_ply(
            ply_path,
            out_path,
            page_capacity,
            memory_budget_bytes,
        )
    })?;
    summary_dict(py, summary)
}

/// Write a tiled point field whose index aliases a few on-disk payload pages
/// across a large grid (a billion-primitive index for out-of-core tests).
#[pyfunction]
#[pyo3(signature = (
    path, *, page_capacity = 4096, template_pages = 16, grid = (496, 496), cell_size = 16.0,
    origin = (0.0, 0.0, 0.0), thickness = 1.5, hole = (0.0, 0.0, 0.0, 0.0), seed = 0
))]
#[pyo3(name = "write_synthetic_point_field")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_synthetic_point_field_py(
    py: Python<'_>,
    path: &str,
    page_capacity: u32,
    template_pages: u32,
    grid: (u32, u32),
    cell_size: f32,
    origin: (f32, f32, f32),
    thickness: f32,
    hole: (f32, f32, f32, f32),
    seed: u32,
) -> PyResult<Py<PyDict>> {
    let desc = SyntheticFieldDesc {
        page_capacity,
        template_pages,
        grid,
        cell_size,
        origin: [origin.0, origin.1, origin.2],
        thickness,
        hole: [hole.0, hole.1, hole.2, hole.3],
        seed,
    };
    let summary = py.allow_threads(|| write_synthetic_point_field(path, &desc))?;
    summary_dict(py, summary)
}

/// The analytic ray / anisotropic-Gaussian intersection (CPU mirror of
/// gaussian_intersect.wgsl) for one splat.
#[pyfunction]
#[pyo3(signature = (
    origin, direction, center, scale, rotation, opacity, *, tmin = 0.0, tmax = f32::INFINITY,
    kappa = 4.0
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn splat_ray_gaussian(
    py: Python<'_>,
    origin: [f32; 3],
    direction: [f32; 3],
    center: [f32; 3],
    scale: [f32; 3],
    rotation: [f32; 4],
    opacity: f32,
    tmin: f32,
    tmax: f32,
    kappa: f32,
) -> PyResult<Py<PyDict>> {
    let norm = direction.iter().map(|v| v * v).sum::<f32>().sqrt();
    let qnorm = rotation.iter().map(|v| v * v).sum::<f32>().sqrt();
    if !(norm.is_finite() && norm > 0.0) || !(qnorm.is_finite() && qnorm > 0.0) {
        return Err(PyValueError::new_err(
            "direction and rotation must be finite and non-zero",
        ));
    }
    if scale.iter().any(|s| !(s.is_finite() && *s > 0.0)) {
        return Err(PyValueError::new_err("scale must be finite and > 0"));
    }
    let hit = ray_gaussian(
        origin,
        direction.map(|v| v / norm),
        tmin,
        tmax,
        center,
        inverse_covariance(scale, rotation.map(|v| v / qnorm)),
        opacity,
    );
    let d = PyDict::new_bound(py);
    d.set_item("t_star", hit.t_star)?;
    d.set_item("g_star", hit.g_star)?;
    d.set_item("a", hit.a)?;
    d.set_item("response", hit.response)?;
    d.set_item("t_hit", hit.t_hit)?;
    d.set_item("transmittance", splat_transmittance(hit.response, kappa))?;
    Ok(d.into())
}

fn brdf_index(name: &str) -> PyResult<u32> {
    Ok(match name {
        "lambert" => 0,
        "phong" => 1,
        "blinn_phong" => 2,
        "oren_nayar" => 3,
        "cook_torrance_ggx" | "ggx" => 4,
        "cook_torrance_beckmann" | "beckmann" => 5,
        "disney" | "disney_principled" => 6,
        "ashikhmin_shirley" => 7,
        "ward" => 8,
        "toon" => 9,
        "minnaert" => 10,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown brdf {other:?}; expected lambert, phong, blinn_phong, oren_nayar, \
                 cook_torrance_ggx, cook_torrance_beckmann, disney, ashikhmin_shirley, ward, \
                 toon or minnaert"
            )))
        }
    })
}

/// Inputs shared by the fused render and its reference.
struct SceneInputs {
    scene: FusedScene,
    terrain: Option<FusedTerrainDesc>,
}

#[allow(clippy::too_many_arguments)]
fn build_scene(
    splats: Option<PyRef<'_, PyGaussianSplatCloud>>,
    splat_page_size: Option<u32>,
    splat_stores: Vec<String>,
    pointclouds: Vec<(String, (f64, f64, f64), bool)>,
    heights: Option<PyReadonlyArray2<'_, f32>>,
    spacing: (f32, f32),
    exaggeration: f32,
    terrain_albedo: (f32, f32, f32),
    params: FusionParams,
) -> PyResult<SceneInputs> {
    params.validate()?;
    let mut sources: Vec<Arc<dyn PageSource>> = Vec::new();
    if let Some(cloud) = splats {
        let page_size = splat_page_size.unwrap_or(params.page_capacity);
        if page_size == 0 || page_size > params.page_capacity {
            return Err(PyValueError::new_err(format!(
                "splat_page_size must be in 1..={} (the page capacity), got {page_size}",
                params.page_capacity
            )));
        }
        if !cloud.inner.is_empty() {
            sources.push(Arc::new(SplatCloudSource::new(
                cloud.inner.clone(),
                page_size as usize,
            )));
        }
    }
    for path in splat_stores {
        let store = PageStoreFile::open(&path)?;
        if store.kind() != crate::splat::stream::PageKind::Splat {
            return Err(PyValueError::new_err(format!(
                "{path} is a point page store, not a splat page store; pass it as a pointcloud"
            )));
        }
        sources.push(Arc::new(store));
    }
    for (path, origin, z_up) in pointclouds {
        // A forge3d page store announces itself by magic; anything else is
        // opened as COPC.
        let mut magic = [0u8; 8];
        let is_store = std::fs::File::open(&path)
            .and_then(|mut file| std::io::Read::read_exact(&mut file, &mut magic))
            .map(|()| &magic == b"F3DPGST1")
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{path}: {e}")))?;
        if is_store {
            let store = PageStoreFile::open(&path)?;
            if store.kind() != crate::splat::stream::PageKind::Points {
                return Err(PyValueError::new_err(format!(
                    "{path} is a splat page store, not a point page store; pass it as splats"
                )));
            }
            sources.push(Arc::new(store));
        } else {
            sources.push(Arc::new(CopcPageSource::open(
                &path,
                PointCloudFrame {
                    origin: [origin.0, origin.1, origin.2],
                    z_up,
                },
                params.page_capacity,
            )?));
        }
    }
    let terrain = match heights {
        None => None,
        Some(array) => {
            let view = array.as_array();
            let (rows, cols) = (view.shape()[0], view.shape()[1]);
            Some(FusedTerrainDesc {
                heights: view.iter().copied().collect(),
                width: cols as u32,
                height: rows as u32,
                spacing,
                exaggeration,
                albedo: [terrain_albedo.0, terrain_albedo.1, terrain_albedo.2],
            })
        }
    };
    if sources.is_empty() && terrain.is_none() {
        return Err(PyValueError::new_err(
            "fused render has no inputs: pass at least one of splats, pointcloud or terrain",
        ));
    }
    Ok(SceneInputs {
        scene: FusedScene::new(sources, params)?,
        terrain,
    })
}

fn paging_dict(py: Python<'_>, paging: &PagingStats) -> PyResult<Py<PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("miss_events", paging.miss_events)?;
    d.set_item("services", paging.services)?;
    d.set_item("loads", paging.residency.loads)?;
    d.set_item("evictions", paging.residency.evictions)?;
    d.set_item("resident_pages", paging.residency.resident_pages)?;
    d.set_item("peak_resident_pages", paging.residency.peak_resident_pages)?;
    d.set_item("deferred_pages", paging.deferred_pages)?;
    d.set_item("pool_bytes", paging.pool_bytes)?;
    d.set_item("index_bytes", paging.index_bytes)?;
    Ok(d.into())
}

fn output_dict(py: Python<'_>, out: FusedRenderOutput) -> PyResult<Py<PyDict>> {
    let (h, w) = (out.height as usize, out.width as usize);
    let d = PyDict::new_bound(py);
    d.set_item(
        "rgba",
        PyArray1::from_vec_bound(py, out.rgba).reshape([h, w, 4])?,
    )?;
    for (name, values) in [
        ("radiance", out.radiance),
        ("albedo", out.albedo),
        ("normal", out.normal),
        ("direct", out.direct),
        ("position", out.position),
    ] {
        d.set_item(
            name,
            PyArray1::from_vec_bound(py, values).reshape([h, w, 3])?,
        )?;
    }
    d.set_item(
        "transmittance",
        PyArray1::from_vec_bound(py, out.transmittance).reshape([h, w, 4])?,
    )?;
    for (name, values) in [
        ("depth", out.depth),
        ("sun_cosine", out.sun_cosine),
        ("self_bias", out.self_bias),
        ("reservoir_visibility", out.reservoir_visibility),
    ] {
        d.set_item(name, PyArray1::from_vec_bound(py, values).reshape([h, w])?)?;
    }
    d.set_item(
        "hit_kind",
        PyArray1::from_vec_bound(py, out.hit_kind).reshape([h, w])?,
    )?;
    d.set_item("frames", out.frames)?;
    d.set_item("restarts", out.restarts)?;
    d.set_item("stale_frames", out.stale_frames)?;
    d.set_item("variance", out.variance)?;
    d.set_item("logical_primitives", out.logical_primitives)?;
    d.set_item("page_count", out.page_count)?;
    d.set_item("tlas_node_count", out.tlas_node_count)?;
    d.set_item("reservoir_valid_count", out.reservoir_valid_count)?;
    d.set_item("paging", paging_dict(py, &out.paging)?)?;
    d.set_item("peak_host_visible_bytes", out.peak_host_visible_bytes)?;
    d.set_item("peak_device_local_bytes", out.peak_device_local_bytes)?;
    d.set_item("peak_total_bytes", out.peak_total_bytes)?;
    d.set_item(
        "tracker_peak_host_visible_bytes",
        out.tracker_peak_host_visible_bytes,
    )?;
    d.set_item("tracker_limit_bytes", out.tracker_limit_bytes)?;
    Ok(d.into())
}

/// Render Gaussian splats, LiDAR/COPC points and a terrain heightfield with
/// the fused ReSTIR integrator. Low-level seam; see `forge3d.splat.render_fused`.
#[pyfunction]
#[pyo3(signature = (
    *,
    cam_origin,
    cam_look_at,
    width,
    height,
    splats = None,
    splat_stores = Vec::new(),
    pointclouds = Vec::new(),
    heights = None,
    spacing = (1.0, 1.0),
    exaggeration = 1.0,
    terrain_albedo = (0.5, 0.5, 0.5),
    cam_up = (0.0, 1.0, 0.0),
    fov_y_deg = 45.0,
    spp = 2,
    frames = 32,
    seed = 0,
    exposure = 1.0,
    sun_azimuth_deg = 135.0,
    sun_elevation_deg = 45.0,
    sun_intensity = 2.5,
    sun_color = (1.0, 0.97, 0.92),
    sky_turbidity = 2.5,
    sky_ground_albedo = 0.2,
    sky_intensity = 0.35,
    kappa = 4.0,
    lidar_radius = 0.2,
    lidar_opacity = 1.0,
    ibl_occlusion_distance = 12.0,
    sun_angular_radius_deg = 0.2665,
    splat_self_bias_sigmas = 2.0,
    lidar_self_bias_radii = 1.0,
    brdf = "lambert",
    roughness = 0.6,
    metallic = 0.0,
    fog_density = 0.0,
    fog_height_falloff = 0.0,
    page_capacity = 4096,
    splat_page_size = None,
    splat_slots = 96,
    point_slots = 96,
    policy = "exact",
    loader_threads = 2,
    certificate = None,
    cache = None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_fused(
    py: Python<'_>,
    cam_origin: (f32, f32, f32),
    cam_look_at: (f32, f32, f32),
    width: u32,
    height: u32,
    splats: Option<PyRef<'_, PyGaussianSplatCloud>>,
    splat_stores: Vec<String>,
    pointclouds: Vec<(String, (f64, f64, f64), bool)>,
    heights: Option<PyReadonlyArray2<'_, f32>>,
    spacing: (f32, f32),
    exaggeration: f32,
    terrain_albedo: (f32, f32, f32),
    cam_up: (f32, f32, f32),
    fov_y_deg: f32,
    spp: u32,
    frames: u32,
    seed: u32,
    exposure: f32,
    sun_azimuth_deg: f32,
    sun_elevation_deg: f32,
    sun_intensity: f32,
    sun_color: (f32, f32, f32),
    sky_turbidity: f32,
    sky_ground_albedo: f32,
    sky_intensity: f32,
    kappa: f32,
    lidar_radius: f32,
    lidar_opacity: f32,
    ibl_occlusion_distance: f32,
    sun_angular_radius_deg: f32,
    splat_self_bias_sigmas: f32,
    lidar_self_bias_radii: f32,
    brdf: &str,
    roughness: f32,
    metallic: f32,
    fog_density: f32,
    fog_height_falloff: f32,
    page_capacity: u32,
    splat_page_size: Option<u32>,
    splat_slots: u32,
    point_slots: u32,
    policy: &str,
    loader_threads: usize,
    certificate: Option<Bound<'_, PyAny>>,
    cache: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyDict>> {
    // Accepted for the ANAMNESIS render contract; the fused integrator has no
    // render-graph cache (its paged scene is streamed per view).
    let _ = cache;
    let policy_name = policy;
    let policy = match policy {
        "exact" => PagingPolicy::Exact,
        "progressive" => PagingPolicy::Progressive,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown paging policy {other:?}; expected 'exact' or 'progressive'"
            )))
        }
    };
    let params = FusionParams {
        kappa,
        lidar_radius,
        lidar_opacity,
        transmittance_epsilon: FusionParams::default().transmittance_epsilon,
        ibl_occlusion_distance,
        sun_angular_radius_deg,
        restir_defensive: FusionParams::default().restir_defensive,
        splat_self_bias_sigmas,
        lidar_self_bias_radii,
        shading: ShadingParams {
            brdf: brdf_index(brdf)?,
            metallic,
            roughness,
            ..ShadingParams::default()
        },
        media: MediaParams {
            density: fog_density,
            height_falloff: fog_height_falloff,
            phase_g: 0.0,
        },
        page_capacity,
        splat_slots,
        point_slots,
        policy,
        loader_threads,
    };
    let inputs = build_scene(
        splats,
        splat_page_size,
        splat_stores,
        pointclouds,
        heights,
        spacing,
        exaggeration,
        terrain_albedo,
        params,
    )?;
    let certificate_capture = crate::core::certificate::begin_render_capture("render_fused");
    // Fallible first GPU touch before the long-running section.
    crate::core::gpu::try_ctx()?;
    crate::core::certificate::record_model(
        "splat_fusion.unified_occlusion",
        "shadow_transmittance = T_splat * T_lidar * T_terrain; analytic ray/Gaussian response \
         T = exp(-kappa * alpha * exp(-g*/2)); stochastic alpha acceptance on scattering rays",
    );
    for (key, value) in [
        ("fused.page_count", inputs.scene.page_count().to_string()),
        ("fused.logical_primitives", inputs.scene.logical_primitive_count().to_string()),
        ("fused.paging_policy", policy_name.to_string()),
        ("fused.kappa", kappa.to_string()),
        ("fused.lidar_radius", lidar_radius.to_string()),
        ("fused.lidar_opacity", lidar_opacity.to_string()),
        ("fused.seed", seed.to_string()),
        ("fused.spp", spp.to_string()),
        ("fused.frames", frames.to_string()),
        ("fused.brdf", brdf.to_string()),
    ] {
        crate::core::certificate::record_input(key, value);
    }
    let out = py.allow_threads(|| -> Result<FusedRenderOutput, crate::core::error::RenderError> {
        let tracer = HybridPathTracer::new_fused()?;
        tracer.render_fused(&FusedRenderDesc {
            scene: &inputs.scene,
            terrain: inputs.terrain.clone(),
            cam_origin: [cam_origin.0, cam_origin.1, cam_origin.2],
            cam_look_at: [cam_look_at.0, cam_look_at.1, cam_look_at.2],
            cam_up: [cam_up.0, cam_up.1, cam_up.2],
            fov_y_deg,
            exposure,
            sun_azimuth_deg,
            sun_elevation_deg,
            sun_intensity,
            sun_color: [sun_color.0, sun_color.1, sun_color.2],
            sky_turbidity,
            sky_ground_albedo,
            sky_intensity,
            width,
            height,
            seed,
            spp,
            frames,
        })
    })?;
    let dict = output_dict(py, out)?;
    // The hybrid_pt.* passes are recorded inside the fused tracer.
    certificate_capture.finish();
    crate::core::certificate::emit_certificate_for_kwarg(py, certificate.as_ref())?;
    Ok(dict)
}

/// Render the hard-proxy path-traced occlusion reference of a fused scene
/// with the AEQUITAS wavefront tracer and classify every pixel.
///
/// Returns `radiance`, `albedo`, `normal` (H, W, 3), `depth` (H, W),
/// `receiver_class` (H, W) uint8 using the fused hit kinds (0 miss, 3
/// terrain, 4 splat, 5 LiDAR) and `shadow` (H, W) int8 (-1 unclassified,
/// 0 lit, 1 shadowed).
#[pyfunction]
#[pyo3(signature = (
    *,
    cam_origin,
    cam_look_at,
    width,
    height,
    region,
    splats = None,
    splat_stores = Vec::new(),
    pointclouds = Vec::new(),
    heights = None,
    spacing = (1.0, 1.0),
    exaggeration = 1.0,
    cam_up = (0.0, 1.0, 0.0),
    fov_y_deg = 45.0,
    frames = 64,
    seed = 0,
    sun_azimuth_deg = 135.0,
    sun_elevation_deg = 45.0,
    sun_intensity = 2.5,
    sun_color = (1.0, 0.97, 0.92),
    kappa = 4.0,
    lidar_radius = 0.2,
    lidar_opacity = 1.0,
    page_capacity = 4096,
    splat_page_size = None,
    min_sun_cosine = 0.15,
    certificate = None,
    cache = None,
))]
#[pyo3(name = "render_fused_reference")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_fused_reference_py(
    py: Python<'_>,
    cam_origin: (f32, f32, f32),
    cam_look_at: (f32, f32, f32),
    width: u32,
    height: u32,
    region: ((f32, f32, f32), (f32, f32, f32)),
    splats: Option<PyRef<'_, PyGaussianSplatCloud>>,
    splat_stores: Vec<String>,
    pointclouds: Vec<(String, (f64, f64, f64), bool)>,
    heights: Option<PyReadonlyArray2<'_, f32>>,
    spacing: (f32, f32),
    exaggeration: f32,
    cam_up: (f32, f32, f32),
    fov_y_deg: f32,
    frames: u32,
    seed: u32,
    sun_azimuth_deg: f32,
    sun_elevation_deg: f32,
    sun_intensity: f32,
    sun_color: (f32, f32, f32),
    kappa: f32,
    lidar_radius: f32,
    lidar_opacity: f32,
    page_capacity: u32,
    splat_page_size: Option<u32>,
    min_sun_cosine: f32,
    certificate: Option<Bound<'_, PyAny>>,
    cache: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyDict>> {
    // Accepted for the ANAMNESIS render contract; the reference trace has no
    // render-graph cache.
    let _ = cache;
    let params = FusionParams {
        kappa,
        lidar_radius,
        lidar_opacity,
        page_capacity,
        ..FusionParams::default()
    };
    let inputs = build_scene(
        splats,
        splat_page_size,
        splat_stores,
        pointclouds,
        heights,
        spacing,
        exaggeration,
        (0.5, 0.5, 0.5),
        params,
    )?;
    let certificate_capture =
        crate::core::certificate::begin_render_capture("render_fused_reference");
    let ctx = crate::core::gpu::try_ctx()?;
    crate::core::certificate::record_model(
        "splat_fusion.hard_proxy_reference",
        "splats as half-transmittance iso-ellipsoid meshes, LiDAR sphelets as icosahedra of the \
         half-coverage radius, terrain as a triangle mesh; AEQUITAS wavefront path tracer",
    );
    for (key, value) in [
        (
            "fused_reference.logical_primitives",
            inputs.scene.logical_primitive_count().to_string(),
        ),
        ("fused_reference.kappa", kappa.to_string()),
        ("fused_reference.lidar_radius", lidar_radius.to_string()),
        ("fused_reference.seed", seed.to_string()),
        ("fused_reference.frames", frames.to_string()),
    ] {
        crate::core::certificate::record_input(key, value);
    }
    let (lo, hi) = region;
    let out = py.allow_threads(|| {
        render_fused_reference(
            &ctx.device,
            &ctx.queue,
            &FusedReferenceDesc {
                scene: &inputs.scene,
                terrain: inputs.terrain.as_ref(),
                region: Aabb::new([lo.0, lo.1, lo.2], [hi.0, hi.1, hi.2]),
                cam_origin: [cam_origin.0, cam_origin.1, cam_origin.2],
                cam_look_at: [cam_look_at.0, cam_look_at.1, cam_look_at.2],
                cam_up: [cam_up.0, cam_up.1, cam_up.2],
                fov_y_deg,
                sun_azimuth_deg,
                sun_elevation_deg,
                sun_intensity,
                sun_color: [sun_color.0, sun_color.1, sun_color.2],
                width,
                height,
                spp_frames: frames,
                seed,
            },
        )
    })?;
    let (az, el) = (sun_azimuth_deg.to_radians(), sun_elevation_deg.to_radians());
    let toward_sun = [az.cos() * el.cos(), el.sin(), az.sin() * el.cos()];
    let pixels = out.depth.len();
    let classes: Vec<u8> = (0..pixels)
        .map(|i| match out.class(i) {
            ReceiverClass::Miss => 0,
            ReceiverClass::Terrain => 3,
            ReceiverClass::Splat => 4,
            ReceiverClass::Lidar => 5,
        })
        .collect();
    let shadow: Vec<i8> = out
        .shadow_mask(toward_sun, min_sun_cosine)
        .into_iter()
        .map(|mask| match mask {
            None => -1,
            Some(false) => 0,
            Some(true) => 1,
        })
        .collect();
    let (h, w) = (height as usize, width as usize);
    let d = PyDict::new_bound(py);
    for (name, values) in [
        ("radiance", out.radiance),
        ("albedo", out.albedo),
        ("normal", out.normal),
    ] {
        d.set_item(
            name,
            PyArray1::from_vec_bound(py, values).reshape([h, w, 3])?,
        )?;
    }
    d.set_item(
        "depth",
        PyArray1::from_vec_bound(py, out.depth).reshape([h, w])?,
    )?;
    d.set_item(
        "receiver_class",
        PyArray1::from_vec_bound(py, classes).reshape([h, w])?,
    )?;
    d.set_item(
        "shadow",
        PyArray1::from_vec_bound(py, shadow).reshape([h, w])?,
    )?;
    d.set_item("terrain_triangles", out.terrain_triangles)?;
    d.set_item("splat_proxies", out.splat_proxies)?;
    d.set_item("lidar_proxies", out.lidar_proxies)?;
    // fused_reference.path_trace is recorded inside the reference tracer.
    certificate_capture.finish();
    crate::core::certificate::emit_certificate_for_kwarg(py, certificate.as_ref())?;
    Ok(d.into())
}
