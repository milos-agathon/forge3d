// src/py_functions/inverse.rs
// DIFFERENTIA Python seam: `inverse_solve` runs the GPU reverse-mode solve
// (per-texel albedo + sun direction/intensity + turbidity) and
// `inverse_render_primal` renders the differentiable primal at explicit
// parameters — the synthetic-target synthesis path used by the recovery test.
// Both are honest GPU entry points: no device -> a raised error, never a
// CPU fallback.
// RELEVANT FILES: src/path_tracing/inverse/mod.rs, python/forge3d/inverse.py

use super::super::*;

#[cfg(all(feature = "extension-module", feature = "enable-inverse-pt"))]
use crate::path_tracing::inverse::{render_primal, solve, InverseParams, InverseSolveDesc};

/// Shared cam dict decode (origin/look_at/up/fov_y/exposure), matching
/// hybrid_render_terrain_reference's key conventions.
#[cfg(all(feature = "extension-module", feature = "enable-inverse-pt"))]
fn extract_cam(cam: &Bound<'_, PyDict>) -> PyResult<([f32; 3], [f32; 3], [f32; 3], f32, f32)> {
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
    Ok((cam_origin, cam_look_at, cam_up, fov_y_deg, exposure))
}

/// Azimuth/elevation degrees -> unit direction toward the sun, the
/// TerrainReferenceDesc convention: [cos(az)cos(el), sin(el), sin(az)cos(el)].
#[cfg(all(feature = "extension-module", feature = "enable-inverse-pt"))]
fn sun_dir_from_deg(azimuth_deg: f32, elevation_deg: f32) -> [f32; 3] {
    let az = azimuth_deg.to_radians();
    let el = elevation_deg.to_radians();
    [az.cos() * el.cos(), el.sin(), az.sin() * el.cos()]
}

/// Unit direction -> (azimuth_deg, elevation_deg), inverse of the above.
#[cfg(all(feature = "extension-module", feature = "enable-inverse-pt"))]
fn sun_deg_from_dir(dir: [f32; 3]) -> (f32, f32) {
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2])
        .sqrt()
        .max(1e-8);
    let (x, y, z) = (dir[0] / len, dir[1] / len, dir[2] / len);
    (z.atan2(x).to_degrees(), y.asin().to_degrees())
}

/// Per-texel albedo argument: (H,W,3) f32 array, or an (r,g,b) triple
/// broadcast over the whole DEM.
#[cfg(all(feature = "extension-module", feature = "enable-inverse-pt"))]
fn extract_albedo(obj: &Bound<'_, PyAny>, texels: usize) -> PyResult<Vec<f32>> {
    if let Ok(arr) = obj.extract::<numpy::PyReadonlyArray3<'_, f32>>() {
        let a = arr.as_array();
        if a.shape()[2] != 3 || a.shape()[0] * a.shape()[1] != texels {
            return Err(PyValueError::new_err(format!(
                "albedo array must have shape ({texels} texels, 3); got {:?}",
                a.shape()
            )));
        }
        return Ok(a.iter().copied().collect());
    }
    if let Ok(triple) = obj.extract::<(f32, f32, f32)>() {
        let mut v = Vec::with_capacity(texels * 3);
        for _ in 0..texels {
            v.extend_from_slice(&[triple.0, triple.1, triple.2]);
        }
        return Ok(v);
    }
    Err(PyValueError::new_err(
        "albedo must be an (H, W, 3) float32 array or an (r, g, b) triple",
    ))
}

/// Solve for terrain albedo, sun direction/intensity, and atmospheric
/// turbidity from a single observed image (reverse-mode gradient
/// descent through the terrain ReSTIR/hybrid primal). The observation is
/// the forward beauty image — Reinhard-space u8, no sRGB encode.
///
/// Returns a dict: albedo (demH,demW,3) f32 linear reflectance,
/// sun_dir (3,) toward-sun unit vector, sun_azimuth_deg, sun_elevation_deg,
/// sun_intensity, turbidity, loss_history (list of per-iteration mean
/// losses), iterations_run, peak_host_visible_bytes (render-scoped
/// ledger peak — the solve's own host-visible allocation), and
/// rgba (H,W,4) u8 — the recovered beauty. The AEQUITAS recovery score
/// (`albedo_delta_e2000_median`) is computed by the Python wrapper from
/// the canonical tests/_deltae.py — never reimplemented natively.
#[cfg(all(feature = "extension-module", feature = "enable-inverse-pt"))]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    target_rgba,
    heightmap,
    width,
    height,
    cam,
    spacing = (1.0, 1.0),
    exaggeration = 1.0,
    init_albedo = None,
    sun_azimuth_deg = 315.0,
    sun_elevation_deg = 45.0,
    sun_intensity = 2.5,
    turbidity = 2.5,
    sun_color = (1.0, 0.97, 0.92),
    env_map = None,
    env_intensity = 0.35,
    iters = 32,
    spp = 4,
    frames = 2,
    tile_size = 32,
    seed = 7u32,
    lr_albedo = 0.05,
    lr_sun = 0.02,
    lr_turbidity = 0.02,
    early_stop_tol = 1e-4,
    early_stop_patience = 8,
    spatial_reuse = true,
    edge_term = true,
    score_correction = true,
))]
pub(crate) fn inverse_solve(
    py: Python<'_>,
    target_rgba: numpy::PyReadonlyArray3<'_, u8>,
    heightmap: numpy::PyReadonlyArray2<'_, f32>,
    width: u32,
    height: u32,
    cam: &Bound<'_, PyDict>,
    spacing: (f32, f32),
    exaggeration: f32,
    init_albedo: Option<Bound<'_, PyAny>>,
    sun_azimuth_deg: f32,
    sun_elevation_deg: f32,
    sun_intensity: f32,
    turbidity: f32,
    sun_color: (f32, f32, f32),
    env_map: Option<numpy::PyReadonlyArray3<'_, f32>>,
    env_intensity: f32,
    iters: u32,
    spp: u32,
    frames: u32,
    tile_size: u32,
    seed: u32,
    lr_albedo: f32,
    lr_sun: f32,
    lr_turbidity: f32,
    early_stop_tol: f32,
    early_stop_patience: u32,
    spatial_reuse: bool,
    edge_term: bool,
    score_correction: bool,
) -> PyResult<Py<PyAny>> {
    use numpy::PyArray1;

    let g = crate::core::gpu::try_ctx()?;
    let (cam_origin, cam_look_at, cam_up, fov_y_deg, exposure) = extract_cam(cam)?;

    let tgt = target_rgba.as_array();
    if tgt.shape()[2] != 4 {
        return Err(PyValueError::new_err(
            "target_rgba must have shape (H, W, 4)",
        ));
    }
    let target: Vec<u8> = tgt.iter().copied().collect();

    let dem = heightmap.as_array();
    let (dem_h, dem_w) = (dem.shape()[0] as u32, dem.shape()[1] as u32);
    let heights: Vec<f32> = dem.iter().copied().collect();
    let texels = (dem_w as usize) * (dem_h as usize);

    let init_alb = match &init_albedo {
        Some(obj) => extract_albedo(obj, texels)?,
        None => vec![0.5f32; texels * 3],
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

    let desc = InverseSolveDesc {
        heights,
        dem_width: dem_w,
        dem_height: dem_h,
        spacing,
        exaggeration,
        target_rgba: target,
        width,
        height,
        cam_origin,
        cam_look_at,
        cam_up,
        fov_y_deg,
        exposure,
        init_albedo: init_alb,
        init: InverseParams {
            sun_dir: sun_dir_from_deg(sun_azimuth_deg, sun_elevation_deg),
            sun_intensity,
            turbidity,
        },
        sun_color: [sun_color.0, sun_color.1, sun_color.2],
        env_map: env,
        env_intensity,
        iters,
        spp,
        frames,
        tile_size,
        seed,
        lr_albedo,
        lr_sun,
        lr_turbidity,
        early_stop_tol,
        early_stop_patience,
        spatial_reuse,
        edge_term,
        score_correction,
    };
    let out = solve(&g.device, &g.queue, &desc)
        .map_err(|e| PyRuntimeError::new_err(format!("inverse_solve failed: {e}")))?;

    let (az_deg, el_deg) = sun_deg_from_dir(out.sun_dir);
    let d = PyDict::new_bound(py);
    d.set_item(
        "albedo",
        PyArray1::<f32>::from_vec_bound(py, out.albedo).reshape([
            dem_h as usize,
            dem_w as usize,
            3,
        ])?,
    )?;
    d.set_item(
        "sun_dir",
        PyArray1::<f32>::from_vec_bound(py, out.sun_dir.to_vec()),
    )?;
    d.set_item("sun_azimuth_deg", az_deg)?;
    d.set_item("sun_elevation_deg", el_deg)?;
    d.set_item("sun_intensity", out.sun_intensity)?;
    d.set_item("turbidity", out.turbidity)?;
    d.set_item("loss_history", out.loss_history)?;
    d.set_item("iterations_run", out.iterations_run)?;
    d.set_item("peak_host_visible_bytes", out.peak_host_visible_bytes)?;
    d.set_item(
        "rgba",
        PyArray1::<u8>::from_vec_bound(py, out.rgba).reshape([
            height as usize,
            width as usize,
            4,
        ])?,
    )?;
    Ok(d.into_py(py))
}

/// Render the differentiable primal at explicit scene parameters —
/// dispatches the same forward chain the solver dual-dispatches (identical
/// to `hybrid_render_terrain_reference` at matched frames/seed/spp).
/// Returns a dict: rgba (H,W,4) u8 forward beauty, mean_rgb.
#[cfg(all(feature = "extension-module", feature = "enable-inverse-pt"))]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    heightmap,
    width,
    height,
    cam,
    albedo,
    spacing = (1.0, 1.0),
    exaggeration = 1.0,
    sun_azimuth_deg = 315.0,
    sun_elevation_deg = 45.0,
    sun_intensity = 2.5,
    turbidity = 2.5,
    sun_color = (1.0, 0.97, 0.92),
    env_map = None,
    env_intensity = 0.35,
    spp = 8,
    frames = 2,
    seed = 7u32,
))]
pub(crate) fn inverse_render_primal(
    py: Python<'_>,
    heightmap: numpy::PyReadonlyArray2<'_, f32>,
    width: u32,
    height: u32,
    cam: &Bound<'_, PyDict>,
    albedo: &Bound<'_, PyAny>,
    spacing: (f32, f32),
    exaggeration: f32,
    sun_azimuth_deg: f32,
    sun_elevation_deg: f32,
    sun_intensity: f32,
    turbidity: f32,
    sun_color: (f32, f32, f32),
    env_map: Option<numpy::PyReadonlyArray3<'_, f32>>,
    env_intensity: f32,
    spp: u32,
    frames: u32,
    seed: u32,
) -> PyResult<Py<PyAny>> {
    use numpy::PyArray1;

    let g = crate::core::gpu::try_ctx()?;
    let (cam_origin, cam_look_at, cam_up, fov_y_deg, exposure) = extract_cam(cam)?;

    let dem = heightmap.as_array();
    let (dem_h, dem_w) = (dem.shape()[0] as u32, dem.shape()[1] as u32);
    let heights: Vec<f32> = dem.iter().copied().collect();
    let texels = (dem_w as usize) * (dem_h as usize);
    let alb = extract_albedo(albedo, texels)?;

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

    // The primal path ignores the observed image; a zeroed target satisfies
    // the descriptor contract without affecting the render.
    let desc = InverseSolveDesc {
        heights,
        dem_width: dem_w,
        dem_height: dem_h,
        spacing,
        exaggeration,
        target_rgba: vec![0u8; (width as usize) * (height as usize) * 4],
        width,
        height,
        cam_origin,
        cam_look_at,
        cam_up,
        fov_y_deg,
        exposure,
        init_albedo: alb.clone(),
        init: InverseParams {
            sun_dir: sun_dir_from_deg(sun_azimuth_deg, sun_elevation_deg),
            sun_intensity,
            turbidity,
        },
        sun_color: [sun_color.0, sun_color.1, sun_color.2],
        env_map: env,
        env_intensity,
        iters: 1,
        spp,
        frames,
        tile_size: 64,
        seed,
        lr_albedo: 0.0,
        lr_sun: 0.0,
        lr_turbidity: 0.0,
        early_stop_tol: 0.0,
        early_stop_patience: 1,
        spatial_reuse: true,
        edge_term: false,
        score_correction: false,
    };
    let params = desc.init;
    let (rgba, mean) = render_primal(&g.device, &g.queue, &desc, &params, &alb)
        .map_err(|e| PyRuntimeError::new_err(format!("inverse_render_primal failed: {e}")))?;

    let d = PyDict::new_bound(py);
    d.set_item(
        "rgba",
        PyArray1::<u8>::from_vec_bound(py, rgba).reshape([height as usize, width as usize, 4])?,
    )?;
    d.set_item("mean_rgb", mean[0])?;
    Ok(d.into_py(py))
}
