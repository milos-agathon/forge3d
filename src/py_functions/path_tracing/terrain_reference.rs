// src/py_functions/path_tracing/terrain_reference.rs
// PROMETHEUS Python seam: GPU-backed converged terrain path-traced reference
// rooted in HybridPathTracer::render_terrain_reference. This is the honest
// GPU entry point — the legacy `hybrid_render` stays a CPU/SDF compatibility
// wrapper and never claimed to reach the GPU path.
// RELEVANT FILES: src/path_tracing/hybrid_compute/render_terrain.rs,
//                 python/forge3d/path_tracing.py

use super::super::super::*;

/// Trust boundary for `sun_color`: exactly three finite, non-negative numbers.
/// (0,0,0) is intentionally valid (disables the sun via colour). The value is
/// received as an untyped object so EVERY malformed shape — scalar, string,
/// numeric strings, byte-oriented sequences, wrong arity — is rejected HERE
/// with ValueError; a typed `(f32, f32, f32)` parameter would fail PyO3
/// extraction first and cross as TypeError instead.
#[cfg(feature = "extension-module")]
fn extract_sun_color(obj: &Bound<'_, PyAny>) -> PyResult<[f32; 3]> {
    use pyo3::types::{PyByteArray, PyBytes, PyMemoryView, PyString};
    let reject =
        || PyValueError::new_err("sun_color must be exactly three finite, non-negative numbers");
    // Byte-oriented containers iterate as small ints and would otherwise pass.
    if obj.is_instance_of::<PyString>()
        || obj.is_instance_of::<PyBytes>()
        || obj.is_instance_of::<PyByteArray>()
        || obj.is_instance_of::<PyMemoryView>()
    {
        return Err(reject());
    }
    let mut out = [0.0f32; 3];
    let mut count = 0usize;
    for item in obj.iter().map_err(|_| reject())? {
        let item = item.map_err(|_| reject())?;
        if count == 3 {
            return Err(reject());
        }
        if item.is_instance_of::<PyString>() || item.is_instance_of::<PyBytes>() {
            return Err(reject());
        }
        out[count] = item.extract::<f32>().map_err(|_| reject())?;
        count += 1;
    }
    if count != 3 {
        return Err(reject());
    }
    if out.iter().any(|c| !c.is_finite() || *c < 0.0) {
        return Err(reject());
    }
    Ok(out)
}

/// Render a converged path-traced reference of a real DEM under sun + IBL,
/// optionally mixed with mesh geometry (terrain stays a first-class primitive
/// of the shared hybrid traversal).
///
/// Returns a dict:
///   rgba (H,W,4) u8, albedo (H,W,3) f32, normal (H,W,3) f32, depth (H,W) f32
///   frames: int, variance: float (max per-pixel variance of the running-mean
///   luminance across the last convergence window), converged: bool (always
///   True — non-convergence raises), and peak_host_visible_bytes /
///   minmax_pyramid_bytes / gpu_resource_bytes memory diagnostics.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    heightmap,
    width,
    height,
    cam,
    spacing = (1.0, 1.0),
    exaggeration = 1.0,
    albedo = (0.6, 0.6, 0.6),
    sun_azimuth_deg = 315.0,
    sun_elevation_deg = 45.0,
    sun_intensity = 2.5,
    env_map = None,
    env_intensity = 0.35,
    mesh_vertices = None,
    mesh_indices = None,
    spp = 1u32,
    max_frames = 512,
    min_frames = 32,
    variance_threshold = 1e-3,
    seed = 7u32,
    certificate = None,
    // Appended (not inserted) to keep positional compatibility of the publicly
    // re-exported native signature: an old positional call must not rebind
    // env_map (or any later arg) to sun_color. None means the legacy warm
    // white (1.0, 0.97, 0.92); received untyped so malformed values raise
    // ValueError (see extract_sun_color), not PyO3's extraction TypeError.
    sun_color = None,
))]
pub(crate) fn hybrid_render_terrain_reference(
    py: Python<'_>,
    heightmap: numpy::PyReadonlyArray2<'_, f32>,
    width: u32,
    height: u32,
    cam: &Bound<'_, pyo3::types::PyDict>,
    spacing: (f32, f32),
    exaggeration: f32,
    albedo: (f32, f32, f32),
    sun_azimuth_deg: f32,
    sun_elevation_deg: f32,
    sun_intensity: f32,
    env_map: Option<numpy::PyReadonlyArray3<'_, f32>>,
    env_intensity: f32,
    mesh_vertices: Option<numpy::PyReadonlyArray2<'_, f32>>,
    mesh_indices: Option<numpy::PyReadonlyArray2<'_, u32>>,
    spp: u32,
    max_frames: u32,
    min_frames: u32,
    variance_threshold: f32,
    seed: u32,
    certificate: Option<Bound<'_, PyAny>>,
    sun_color: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    use crate::path_tracing::hybrid_compute::{HybridPathTracer, TerrainReferenceDesc};
    use numpy::PyArray1;

    // Trust boundary: raised at the PyO3 layer as ValueError — validate_desc
    // returns RenderError, which would cross as PyRuntimeError. Validated
    // BEFORE the first GPU touch so a missing adapter cannot pre-empt it with
    // a device RuntimeError.
    let sun_color: [f32; 3] = match sun_color.as_ref() {
        None => [1.0, 0.97, 0.92],
        Some(obj) => extract_sun_color(obj)?,
    };

    let certificate_capture =
        crate::core::certificate::begin_render_capture("hybrid_render_terrain_reference");
    // Fallible first GPU touch: later ctx() calls cannot fail once this succeeds.
    crate::core::gpu::try_ctx()?;

    let dem = heightmap.as_array();
    let (dem_h, dem_w) = (dem.shape()[0] as u32, dem.shape()[1] as u32);
    let heights: Vec<f32> = dem.iter().copied().collect();

    let get_vec3 = |key: &str, default: [f32; 3]| -> PyResult<[f32; 3]> {
        match cam.get_item(key)? {
            Some(v) => {
                let t: (f32, f32, f32) = v.extract()?;
                Ok([t.0, t.1, t.2])
            }
            None => Ok(default),
        }
    };
    let cam_origin = get_vec3("origin", [0.0, 50.0, 120.0])?;
    let cam_look_at = get_vec3("look_at", [0.0, 0.0, 0.0])?;
    let cam_up = get_vec3("up", [0.0, 1.0, 0.0])?;
    let fov_y_deg: f32 = match cam.get_item("fov_y")? {
        Some(v) => v.extract()?,
        None => 45.0,
    };
    let exposure: f32 = match cam.get_item("exposure")? {
        Some(v) => v.extract()?,
        None => 1.0,
    };

    let env = match &env_map {
        Some(arr) => {
            let a = arr.as_array();
            if a.shape()[2] != 3 {
                return Err(PyValueError::new_err("env_map must have shape (H, W, 3)"));
            }
            Some((
                a.iter().copied().collect::<Vec<f32>>(),
                a.shape()[1] as u32,
                a.shape()[0] as u32,
            ))
        }
        None => None,
    };

    let mesh = match (&mesh_vertices, &mesh_indices) {
        (Some(v), Some(i)) => {
            let v = v.as_array();
            let i = i.as_array();
            if v.shape()[1] != 3 {
                return Err(PyValueError::new_err(
                    "mesh_vertices must have shape (N, 3)",
                ));
            }
            if i.shape()[1] != 3 {
                return Err(PyValueError::new_err("mesh_indices must have shape (M, 3)"));
            }
            Some((
                v.iter().copied().collect::<Vec<f32>>(),
                i.iter().copied().collect::<Vec<u32>>(),
            ))
        }
        (None, None) => None,
        _ => {
            return Err(PyValueError::new_err(
                "mesh_vertices and mesh_indices must be provided together",
            ))
        }
    };

    let desc = TerrainReferenceDesc {
        heights,
        dem_width: dem_w,
        dem_height: dem_h,
        spacing,
        exaggeration,
        albedo: [albedo.0, albedo.1, albedo.2],
        cam_origin,
        cam_look_at,
        cam_up,
        fov_y_deg,
        exposure,
        sun_azimuth_deg,
        sun_elevation_deg,
        sun_intensity,
        sun_color,
        env_map: env,
        env_intensity,
        mesh,
        width,
        height,
        seed,
        spp,
        max_frames,
        min_frames,
        variance_threshold,
    };

    let tracer = HybridPathTracer::new()?;
    let out = tracer.render_terrain_reference(&desc)?;

    let d = pyo3::types::PyDict::new_bound(py);
    let rgba = PyArray1::<u8>::from_vec_bound(py, out.rgba).reshape([
        height as usize,
        width as usize,
        4,
    ])?;
    let albedo_arr = PyArray1::<f32>::from_vec_bound(py, out.albedo).reshape([
        height as usize,
        width as usize,
        3,
    ])?;
    let normal_arr = PyArray1::<f32>::from_vec_bound(py, out.normal).reshape([
        height as usize,
        width as usize,
        3,
    ])?;
    let depth_arr = PyArray1::<f32>::from_vec_bound(py, out.depth)
        .reshape([height as usize, width as usize])?;
    d.set_item("rgba", rgba)?;
    d.set_item("albedo", albedo_arr)?;
    d.set_item("normal", normal_arr)?;
    d.set_item("depth", depth_arr)?;
    d.set_item("frames", out.frames)?;
    d.set_item("variance", out.variance)?;
    d.set_item("converged", out.converged)?;
    d.set_item("peak_host_visible_bytes", out.peak_host_visible_bytes)?;
    d.set_item("minmax_pyramid_bytes", out.minmax_pyramid_bytes)?;
    d.set_item("gpu_resource_bytes", out.gpu_resource_bytes)?;
    // The hybrid_pt.* passes (live gpu_ms when timestamps are granted) are
    // recorded inside HybridPathTracer::render_terrain_reference.
    certificate_capture.finish();
    crate::core::certificate::emit_certificate_for_kwarg(py, certificate.as_ref())?;
    Ok(d.into_py(py))
}
