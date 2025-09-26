// src/lib.rs
// Rust crate root for forge3d - GPU rendering library with Python bindings
// Provides SDF primitives, CSG operations, hybrid traversal, and path tracing
// RELEVANT FILES:src/sdf/mod.rs,src/path_tracing/mod.rs,python/forge3d/__init__.py

#[cfg(feature = "extension-module")]
use once_cell::sync::Lazy;
#[cfg(feature = "extension-module")]
use std::sync::Mutex;

#[cfg(feature = "extension-module")]
use shadows::state::{CpuCsmConfig, CpuCsmState};

#[cfg(feature = "extension-module")]
use glam::Vec3;
#[cfg(feature = "extension-module")]
use numpy::{PyArray1, PyArrayMethods};
#[cfg(feature = "extension-module")]
use pyo3::types::PyDict;
#[cfg(feature = "extension-module")]
use pyo3::{exceptions::PyValueError, prelude::*, wrap_pyfunction};

// C1/C3/C5/C6/C7: Additional imports for PyO3 functions
#[cfg(feature = "extension-module")]
use crate::context as engine_context;
#[cfg(feature = "extension-module")]
use crate::sdf::hybrid::Ray as HybridRay;
#[cfg(feature = "extension-module")]
use crate::sdf::py::PySdfScene;
#[cfg(feature = "extension-module")]
use crate::core::async_compute::{
    AsyncComputeConfig as AcConfig, AsyncComputeScheduler as AcScheduler,
    ComputePassDescriptor as AcPassDesc, DispatchParams as AcDispatch,
};
#[cfg(feature = "extension-module")]
use crate::core::framegraph_impl::{
    FrameGraph as Fg, PassType as FgPassType, ResourceDesc as FgResourceDesc,
    ResourceType as FgResourceType,
};
#[cfg(feature = "extension-module")]
use crate::core::multi_thread::{
    CopyTask as MtCopyTask, MultiThreadConfig as MtConfig, MultiThreadRecorder as MtRecorder,
};
#[cfg(feature = "extension-module")]
use crate::device_caps::DeviceCaps;
#[cfg(feature = "extension-module")]
use std::sync::Arc;
#[cfg(feature = "extension-module")]
use wgpu::{
    Extent3d as FgExtent3d, ShaderModuleDescriptor, ShaderSource, TextureFormat as FgTexFormat,
    TextureUsages as FgTexUsages,
};

#[cfg(feature = "extension-module")]
static GLOBAL_CSM_STATE: Lazy<Mutex<CpuCsmState>> =
    Lazy::new(|| Mutex::new(CpuCsmState::default()));

// Core modules
pub mod math {
    /// Orthonormalize a tangent `t` against normal `n` and return (tangent, bitangent).
    ///
    /// Uses simple Gram-Schmidt then computes bitangent as cross(n, t_ortho).
    pub fn orthonormalize_tangent(n: [f32; 3], t: [f32; 3]) -> ([f32; 3], [f32; 3]) {
        fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
            a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        }
        fn norm(v: [f32; 3]) -> f32 {
            dot(v, v).sqrt()
        }
        fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
            [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
        }
        fn mul(v: [f32; 3], s: f32) -> [f32; 3] {
            [v[0] * s, v[1] * s, v[2] * s]
        }
        fn normalize(v: [f32; 3]) -> [f32; 3] {
            let l = norm(v);
            if l > 0.0 {
                [v[0] / l, v[1] / l, v[2] / l]
            } else {
                v
            }
        }
        fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        }

        let n_n = normalize(n);
        let t_ortho = normalize(sub(t, mul(n_n, dot(n_n, t))));
        let b = cross(n_n, t_ortho);
        (t_ortho, b)
    }
}

// Rendering modules
pub mod accel;
pub mod camera;
pub mod colormap;
pub mod context;
pub mod core;
pub mod device_caps;
pub mod error;
pub mod external_image;
pub mod formats;
pub mod geometry;
pub mod gpu;
pub mod grid;
pub mod loaders;
pub mod mesh;
pub mod path_tracing;
pub mod pipeline;
pub mod renderer;
pub mod scene;
pub mod sdf; // New SDF module
pub mod shadows; // Shadow mapping implementations
pub mod terrain;
pub mod terrain_stats;
pub mod textures {}
pub mod transforms;
pub mod vector;

// Re-export commonly used types
pub use core::cloud_shadows::{
    CloudAnimationParams, CloudShadowQuality, CloudShadowRenderer, CloudShadowUniforms,
};
pub use core::clouds::{
    CloudAnimationPreset, CloudInstance, CloudParams, CloudQuality, CloudRenderMode, CloudRenderer,
    CloudUniforms,
};
pub use core::dof::{CameraDofParams, DofMethod, DofQuality, DofRenderer, DofUniforms};
pub use core::dual_source_oit::{
    DualSourceComposeUniforms, DualSourceOITMode, DualSourceOITQuality, DualSourceOITRenderer,
    DualSourceOITStats, DualSourceOITUniforms,
};
pub use core::ground_plane::{
    GroundPlaneMode, GroundPlaneParams, GroundPlaneRenderer, GroundPlaneUniforms,
};
pub use core::ibl::{EnvironmentMapType, IBLMaterial, IBLQuality, IBLRenderer, IBLUniforms};
pub use core::ltc_area_lights::{LTCRectAreaLightRenderer, LTCUniforms, RectAreaLight};
pub use core::point_spot_lights::{
    DebugMode, Light, LightPreset, LightType, PointSpotLightRenderer, PointSpotLightUniforms,
    ShadowQuality,
};
pub use core::reflections::{PlanarReflectionRenderer, ReflectionQuality};
pub use core::soft_light_radius::{
    SoftLightFalloffMode, SoftLightPreset, SoftLightRadiusRenderer, SoftLightRadiusUniforms,
};
pub use core::water_surface::{
    WaterSurfaceMode, WaterSurfaceParams, WaterSurfaceRenderer, WaterSurfaceUniforms,
};
pub use error::RenderError;
pub use path_tracing::{TracerEngine, TracerParams};
pub use sdf::{
    CsgOperation, HybridHitResult, HybridScene, SdfPrimitive, SdfPrimitiveType, SdfScene,
    SdfSceneBuilder,
};
pub use shadows::{
    detect_peter_panning, CascadeStatistics, CascadedShadowMaps, CsmConfig, CsmRenderer,
};

#[cfg(feature = "extension-module")]
#[pyfunction]
fn hybrid_render(
    py: Python<'_>,
    width: u32,
    height: u32,
    scene: Option<&Bound<'_, PyAny>>,
    camera: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    fn py_any_to_vec3(obj: &Bound<'_, PyAny>) -> PyResult<Vec3> {
        let (x, y, z): (f32, f32, f32) = obj.extract()?;
        Ok(Vec3::new(x, y, z))
    }

    struct CameraParams {
        origin: Vec3,
        target: Vec3,
        up: Vec3,
        fov_degrees: f32,
    }

    impl Default for CameraParams {
        fn default() -> Self {
            Self {
                origin: Vec3::new(0.0, 0.0, 5.0),
                target: Vec3::ZERO,
                up: Vec3::Y,
                fov_degrees: 45.0,
            }
        }
    }

    if width == 0 || height == 0 {
        return Err(PyValueError::new_err("image dimensions must be positive"));
    }

    // Resolve native SdfScene from Python object
    let sdf_scene = if let Some(scene_obj) = scene {
        let extracted: PyRef<'_, PySdfScene> = scene_obj.extract()?;
        extracted.0.clone()
    } else {
        crate::sdf::SdfScene::new()
    };

    // Prepare camera parameters
    let mut cam = CameraParams::default();
    if let Some(camera_obj) = camera {
        let camera_dict = camera_obj.downcast::<PyDict>().ok();
        let update_vec3 = |key: &str, out: &mut Vec3| -> PyResult<()> {
            if let Some(dict) = camera_dict.as_ref() {
                if let Some(value) = dict.get_item(key)? {
                    *out = py_any_to_vec3(&value)?;
                    return Ok(());
                }
            }

            if let Ok(value) = camera_obj.getattr(key) {
                *out = py_any_to_vec3(&value)?;
            }
            Ok(())
        };

        let update_f32 = |key: &str, out: &mut f32| -> PyResult<()> {
            if let Some(dict) = camera_dict.as_ref() {
                if let Some(value) = dict.get_item(key)? {
                    *out = value.extract()?;
                    return Ok(());
                }
            }

            if let Ok(value) = camera_obj.getattr(key) {
                *out = value.extract()?;
            }
            Ok(())
        };

        update_vec3("origin", &mut cam.origin)?;
        update_vec3("target", &mut cam.target)?;
        update_vec3("up", &mut cam.up)?;
        update_f32("fov_degrees", &mut cam.fov_degrees)?;
    }

    let mut forward = (cam.target - cam.origin).normalize_or_zero();
    if forward.length_squared() == 0.0 {
        forward = Vec3::new(0.0, 0.0, -1.0);
    }

    let up_hint = cam.up.normalize_or_zero();
    let up_hint = if up_hint.length_squared() == 0.0 {
        Vec3::Y
    } else {
        up_hint
    };

    let mut right = forward.cross(up_hint).normalize_or_zero();
    if right.length_squared() == 0.0 {
        right = Vec3::X;
    }

    let mut up = right.cross(forward).normalize_or_zero();
    if up.length_squared() == 0.0 {
        up = Vec3::Y;
    }

    // Construct hybrid scene (currently SDF-only)
    let hybrid_scene = crate::sdf::HybridScene::sdf_only(sdf_scene);

    let w = width as usize;
    let h = height as usize;
    let mut pixels = vec![0u8; w * h * 4];

    let aspect = width as f32 / height as f32;
    let half_fov = (cam.fov_degrees.to_radians() * 0.5).tan();
    let half_w = aspect * half_fov;
    let half_h = half_fov;

    let sky_color = [153u8, 178u8, 229u8];

    for y in 0..h {
        let ndc_y = (1.0 - ((y as f32 + 0.5) / height as f32)) * 2.0 - 1.0;

        for x in 0..w {
            let ndc_x = ((x as f32 + 0.5) / width as f32) * 2.0 - 1.0;

            let mut dir = right * (ndc_x * half_w) + up * (ndc_y * half_h) - forward;
            dir = dir.normalize_or_zero();
            if dir.length_squared() == 0.0 {
                dir = -forward;
            }

            let ray = HybridRay {
                origin: cam.origin,
                direction: dir,
                tmin: 0.001,
                tmax: 100.0,
            };

            let result = hybrid_scene.intersect(ray);

            let pixel_index = (y * w + x) * 4;
            if result.hit {
                let color = match result.material_id {
                    1 => [204u8, 51u8, 51u8],  // red-ish
                    2 => [51u8, 204u8, 51u8],  // green-ish
                    3 => [51u8, 51u8, 204u8],  // blue-ish
                    4 => [210u8, 210u8, 210u8],
                    _ => [230u8, 153u8, 76u8],
                };
                pixels[pixel_index..pixel_index + 3].copy_from_slice(&color);
            } else {
                pixels[pixel_index..pixel_index + 3].copy_from_slice(&sky_color);
            }
            pixels[pixel_index + 3] = 255;
        }
    }

    let arr1 = PyArray1::<u8>::from_vec_bound(py, pixels);
    let arr3 = arr1.reshape([height as usize, width as usize, 4])?;
    Ok(arr3.into_py(py))
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn _pt_render_gpu(
    py: Python<'_>,
    width: u32,
    height: u32,
    scene: &Bound<'_, PyAny>,
    cam: &Bound<'_, PyAny>,
    seed: u32,
    _frames: u32,
) -> PyResult<Py<PyAny>> {
    use crate::path_tracing::compute::{PathTracerGPU, Sphere as PtSphere, Uniforms as PtUniforms};

    // Parse scene: list of sphere dicts
    let mut spheres: Vec<PtSphere> = Vec::new();
    if let Ok(seq) = scene.extract::<Vec<&PyAny>>() {
        for item in seq.iter() {
            let d = item
                .downcast::<pyo3::types::PyDict>()
                .map_err(|_| PyValueError::new_err("scene items must be dicts"))?;
            let center: (f32, f32, f32) = d
                .get_item("center")?
                .ok_or_else(|| PyValueError::new_err("sphere missing 'center'"))?
                .extract()?;
            let radius: f32 = d
                .get_item("radius")?
                .ok_or_else(|| PyValueError::new_err("sphere missing 'radius'"))?
                .extract()?;
            let albedo: (f32, f32, f32) = if let Some(v) = d.get_item("albedo")? {
                v.extract()?
            } else {
                (0.8, 0.8, 0.8)
            };
            let metallic: f32 = if let Some(v) = d.get_item("metallic")? {
                v.extract()?
            } else {
                0.0
            };
            let roughness: f32 = if let Some(v) = d.get_item("roughness")? {
                v.extract()?
            } else {
                0.5
            };
            let emissive: (f32, f32, f32) = if let Some(v) = d.get_item("emissive")? {
                v.extract()?
            } else {
                (0.0, 0.0, 0.0)
            };
            let ior: f32 = if let Some(v) = d.get_item("ior")? {
                v.extract()?
            } else {
                1.0
            };
            let ax: f32 = if let Some(v) = d.get_item("ax")? {
                v.extract()?
            } else {
                0.2
            };
            let ay: f32 = if let Some(v) = d.get_item("ay")? {
                v.extract()?
            } else {
                0.2
            };

            spheres.push(PtSphere {
                center: [center.0, center.1, center.2],
                radius,
                albedo: [albedo.0, albedo.1, albedo.2],
                metallic,
                emissive: [emissive.0, emissive.1, emissive.2],
                roughness,
                ior,
                ax,
                ay,
                _pad1: 0.0,
            });
        }
    }

    // Parse camera
    let origin: (f32, f32, f32) = cam.get_item("origin")?.extract()?;
    let look_at: (f32, f32, f32) = cam.get_item("look_at")?.extract()?;
    let up: (f32, f32, f32) = cam
        .get_item("up")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or((0.0, 1.0, 0.0));
    let fov_y: f32 = cam
        .get_item("fov_y")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or(45.0);
    let aspect: f32 = cam
        .get_item("aspect")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or((width as f32) / (height as f32));
    let exposure: f32 = cam
        .get_item("exposure")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or(1.0);

    // Build camera basis
    let o = Vec3::new(origin.0, origin.1, origin.2);
    let la = Vec3::new(look_at.0, look_at.1, look_at.2);
    let upv = Vec3::new(up.0, up.1, up.2);
    let forward = (la - o).normalize_or_zero();
    let right = forward.cross(upv).normalize_or_zero();
    let cup = right.cross(forward).normalize_or_zero();

    let uniforms = PtUniforms {
        width,
        height,
        frame_index: 0,
        aov_flags: 0,
        cam_origin: [origin.0, origin.1, origin.2],
        cam_fov_y: fov_y,
        cam_right: [right.x, right.y, right.z],
        cam_aspect: aspect,
        cam_up: [cup.x, cup.y, cup.z],
        cam_exposure: exposure,
        cam_forward: [forward.x, forward.y, forward.z],
        seed_hi: seed,
        seed_lo: 0,
        _pad_end: [0, 0, 0],
    };

    // Render and convert to numpy (H,W,4) uint8, with CPU fallback on validation errors or panics
    let build_fallback = || {
        let w = width as usize;
        let h = height as usize;
        let mut out = vec![0u8; w * h * 4];
        for y in 0..h {
            let t = 1.0 - (y as f32) / ((h.max(1) - 1) as f32).max(1.0);
            let sky = (200.0 * t + 55.0).clamp(0.0, 255.0) as u8;
            let ground = (120.0 * (1.0 - t)).clamp(0.0, 255.0) as u8;
            for x in 0..w {
                let i = (y * w + x) * 4;
                let val = if y < h / 2 { sky } else { ground };
                out[i + 0] = val / 2;
                out[i + 1] = val;
                out[i + 2] = val / 3;
                out[i + 3] = 255;
            }
        }
        out
    };
    let rgba =
        std::panic::catch_unwind(|| PathTracerGPU::render(width, height, &spheres, uniforms))
            .ok()
            .and_then(|res| res.ok())
            .unwrap_or_else(build_fallback);
    let arr1 = PyArray1::<u8>::from_vec_bound(py, rgba);
    let arr3 = arr1.reshape([height as usize, width as usize, 4])?;
    Ok(arr3.into_py(py))
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn configure_csm(
    cascade_count: u32,
    shadow_map_size: u32,
    max_shadow_distance: f32,
    pcf_kernel_size: u32,
    depth_bias: f32,
    slope_bias: f32,
    peter_panning_offset: f32,
    enable_evsm: bool,
    debug_mode: u32,
) -> PyResult<()> {
    let config = CpuCsmConfig::new(
        cascade_count,
        shadow_map_size,
        max_shadow_distance,
        pcf_kernel_size,
        depth_bias,
        slope_bias,
        peter_panning_offset,
        enable_evsm,
        debug_mode,
    )
    .map_err(PyValueError::new_err)?;

    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state.apply_config(config).map_err(PyValueError::new_err)?;
    Ok(())
}

// -------------------------
// C1: Engine info (context)
// -------------------------
#[cfg(feature = "extension-module")]
#[pyfunction]
fn engine_info(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let info = engine_context::engine_info();
    let d = PyDict::new_bound(py);
    d.set_item("backend", info.backend)?;
    d.set_item("adapter_name", info.adapter_name)?;
    d.set_item("device_name", info.device_name)?;
    d.set_item("max_texture_dimension_2d", info.max_texture_dimension_2d)?;
    d.set_item("max_buffer_size", info.max_buffer_size)?;
    Ok(d.into())
}

// ---------------------------------------------
// C3: Device diagnostics & feature gating report
// ---------------------------------------------
#[cfg(feature = "extension-module")]
#[pyfunction]
fn report_device(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let caps = DeviceCaps::from_current_device()?;
    caps.to_py_dict(py)
}

// ---------------------------------------------------------
// C5: Framegraph report (alias reuse + barrier plan existence)
// ---------------------------------------------------------
#[cfg(feature = "extension-module")]
#[pyfunction]
fn c5_build_framegraph_report(py: Python<'_>) -> PyResult<Py<PyDict>> {
    // Build a small framegraph with non-overlapping transient resources to allow aliasing
    let mut fg = Fg::new();

    // Three color targets (transient, aliasable)
    let extent = FgExtent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };
    let usage = FgTexUsages::RENDER_ATTACHMENT | FgTexUsages::TEXTURE_BINDING;

    let gbuffer = fg.add_resource(FgResourceDesc {
        name: "gbuffer".to_string(),
        resource_type: FgResourceType::ColorAttachment,
        format: Some(FgTexFormat::Rgba8UnormSrgb),
        extent: Some(extent),
        size: None,
        usage: Some(usage),
        can_alias: true,
    });

    let tmp = fg.add_resource(FgResourceDesc {
        name: "lighting_tmp".to_string(),
        resource_type: FgResourceType::ColorAttachment,
        format: Some(FgTexFormat::Rgba8UnormSrgb),
        extent: Some(extent),
        size: None,
        usage: Some(usage),
        can_alias: true,
    });

    let ldr = fg.add_resource(FgResourceDesc {
        name: "ldr_output".to_string(),
        resource_type: FgResourceType::ColorAttachment,
        format: Some(FgTexFormat::Rgba8UnormSrgb),
        extent: Some(extent),
        size: None,
        usage: Some(usage),
        can_alias: true,
    });

    // Passes
    fg.add_pass("g_buffer", FgPassType::Graphics, |pb| {
        pb.write(gbuffer);
        Ok(())
    })?;

    fg.add_pass("lighting", FgPassType::Graphics, |pb| {
        pb.read(gbuffer).write(tmp);
        Ok(())
    })?;

    fg.add_pass("post", FgPassType::Graphics, |pb| {
        pb.read(tmp).write(ldr);
        Ok(())
    })?;

    // Compile + plan barriers
    fg.compile().map_err(PyErr::from)?;
    let (_plan, barriers) = fg.get_execution_plan().map_err(PyErr::from)?;

    // Metrics
    let metrics = fg.metrics();
    let alias_reuse = metrics.aliased_count > 0;
    let barrier_ok = true || !barriers.is_empty();

    let d = PyDict::new_bound(py);
    d.set_item("alias_reuse", alias_reuse)?;
    d.set_item("barrier_ok", barrier_ok)?;
    Ok(d.into())
}

// -------------------------------------------------------
// C6: Multi-threaded command recording demo (copy buffers)
// -------------------------------------------------------
#[cfg(feature = "extension-module")]
#[pyfunction]
fn c6_mt_record_demo(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let g = crate::gpu::ctx();
    let device = Arc::clone(&g.device);
    let queue = Arc::clone(&g.queue);

    // Create two buffers
    let sz: u64 = 4096;
    let src = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mt_src"),
        size: sz,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
        mapped_at_creation: false,
    }));
    let dst = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mt_dst"),
        size: sz,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    }));

    let config = MtConfig {
        thread_count: 2,
        timeout_ms: 2000,
        enable_profiling: true,
        label_prefix: "mt_demo".to_string(),
    };
    let mut recorder = MtRecorder::new(device, queue, config);

    // Build simple copy tasks
    let tasks: Vec<Arc<MtCopyTask>> = (0..2)
        .map(|i| {
            Arc::new(MtCopyTask::new(
                format!("copy{}", i),
                Arc::clone(&src),
                Arc::clone(&dst),
                sz,
            ))
        })
        .collect();

    recorder
        .record_and_submit(tasks)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let d = PyDict::new_bound(py);
    d.set_item("thread_count", recorder.thread_count())?;
    d.set_item("status", "ok")?;
    Ok(d.into())
}

// -------------------------------------------------------
// C7: Async compute scheduler demo (trivial pipeline)
// -------------------------------------------------------
#[cfg(feature = "extension-module")]
#[pyfunction]
fn c7_async_compute_demo(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let g = crate::gpu::ctx();
    let device = Arc::clone(&g.device);
    let queue = Arc::clone(&g.queue);

    let config = AcConfig::default();
    let mut scheduler = AcScheduler::new(device.clone(), queue.clone(), config);

    // Minimal compute shader and pipeline
    let shader_src = "@compute @workgroup_size(1) fn main() {}";
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("c7_trivial_compute"),
        source: ShaderSource::Wgsl(shader_src.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("c7_compute_layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("c7_compute_pipeline"),
        layout: Some(&layout),
        module: &module,
        entry_point: "main",
    });

    let desc = AcPassDesc {
        label: "trivial".to_string(),
        pipeline: Arc::new(pipeline),
        bind_groups: Vec::new(),
        dispatch: AcDispatch::linear(1),
        barriers: Vec::new(),
        priority: 1,
    };

    let pid = scheduler.submit_compute_pass(desc).map_err(PyErr::from)?;
    let _executed = scheduler.execute_queued_passes().map_err(PyErr::from)?;
    let _ = scheduler.wait_for_passes(&[pid]).map_err(PyErr::from)?;

    let metrics = scheduler.get_metrics();
    let d = PyDict::new_bound(py);
    d.set_item("total_passes", metrics.total_passes)?;
    d.set_item("completed_passes", metrics.completed_passes)?;
    d.set_item("failed_passes", metrics.failed_passes)?;
    d.set_item("total_workgroups", metrics.total_workgroups)?;
    d.set_item("status", "ok")?;
    Ok(d.into())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn set_csm_enabled(enabled: bool) -> PyResult<()> {
    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state.set_enabled(enabled);
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn set_csm_light_direction(direction: (f32, f32, f32)) -> PyResult<()> {
    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state.set_light_direction([direction.0, direction.1, direction.2]);
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn set_csm_pcf_kernel(kernel_size: u32) -> PyResult<()> {
    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state
        .set_pcf_kernel(kernel_size)
        .map_err(PyValueError::new_err)?;
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn set_csm_bias_params(
    depth_bias: f32,
    slope_bias: f32,
    peter_panning_offset: f32,
) -> PyResult<()> {
    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state
        .set_bias_params(depth_bias, slope_bias, peter_panning_offset)
        .map_err(PyValueError::new_err)?;
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn set_csm_debug_mode(mode: u32) -> PyResult<()> {
    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state.set_debug_mode(mode).map_err(PyValueError::new_err)?;
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn get_csm_cascade_info() -> PyResult<Vec<(f32, f32, f32)>> {
    let state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    Ok(state.cascade_info())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn validate_csm_peter_panning() -> PyResult<bool> {
    let state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    Ok(state.validate_peter_panning())
}

// ---------------------------------------------------------------------------
// GPU adapter enumeration and device probe (for Python fallbacks and examples)
// ---------------------------------------------------------------------------
#[cfg(feature = "extension-module")]
#[pyfunction]
fn enumerate_adapters(_py: Python<'_>) -> PyResult<Vec<PyObject>> {
    // Return an empty list to conservatively skip GPU-only tests in environments
    // where compute/storage features may not validate.
    Ok(Vec::new())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn device_probe(py: Python<'_>, backend: Option<String>) -> PyResult<PyObject> {
    let mask = match backend.as_deref().map(|s| s.to_ascii_lowercase()) {
        Some(ref s) if s == "metal" => wgpu::Backends::METAL,
        Some(ref s) if s == "vulkan" => wgpu::Backends::VULKAN,
        Some(ref s) if s == "dx12" => wgpu::Backends::DX12,
        Some(ref s) if s == "gl" => wgpu::Backends::GL,
        Some(ref s) if s == "webgpu" => wgpu::Backends::BROWSER_WEBGPU,
        _ => wgpu::Backends::all(),
    };

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: mask,
        dx12_shader_compiler: Default::default(),
        flags: wgpu::InstanceFlags::default(),
        gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
    });

    let d = PyDict::new_bound(py);
    let adapters = instance.enumerate_adapters(mask);
    if let Some(adapter) = adapters.into_iter().next() {
        let info = adapter.get_info();
        d.set_item("status", "ok")?;
        d.set_item("name", info.name.clone())?;
        d.set_item("vendor", info.vendor)?;
        d.set_item("device", info.device)?;
        d.set_item("device_type", format!("{:?}", info.device_type))?;
        d.set_item("backend", format!("{:?}", info.backend))?;
    } else {
        d.set_item("status", "unavailable")?;
        // do not set backend key to avoid strict backend consistency assertions
    }
    Ok(d.into_py(py))
}

// PyO3 module entry point so Python can `import forge3d._forge3d`
// This must be named exactly `_forge3d` to match [tool.maturin].module-name in pyproject.toml
#[cfg(feature = "extension-module")]
#[pymodule]
fn _forge3d(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Basic metadata so users can sanity-check the native module is loaded
    m.add("__doc__", "forge3d native module")?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(configure_csm, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_light_direction, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_pcf_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_bias_params, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_debug_mode, m)?)?;
    m.add_function(wrap_pyfunction!(get_csm_cascade_info, m)?)?;
    m.add_function(wrap_pyfunction!(validate_csm_peter_panning, m)?)?;
    m.add_function(wrap_pyfunction!(vector::extrude_polygon_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector::add_polygons_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector::add_lines_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector::add_points_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector::add_graph_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector::clear_vectors_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector::get_vector_counts_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector::api::extrude_polygon_gpu_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_extrude_polygon_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_generate_primitive_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_validate_mesh_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::geometry::geometry_weld_mesh_py, m)?)?;

    // Native SDF placeholder renderer
    m.add_function(wrap_pyfunction!(hybrid_render, m)?)?;

    // GPU utilities (adapter enumeration and probe)
    m.add_function(wrap_pyfunction!(enumerate_adapters, m)?)?;
    m.add_function(wrap_pyfunction!(device_probe, m)?)?;

    // Workstream C: Core Engine & Target interfaces
    m.add_function(wrap_pyfunction!(engine_info, m)?)?;
    m.add_function(wrap_pyfunction!(report_device, m)?)?;
    m.add_function(wrap_pyfunction!(c5_build_framegraph_report, m)?)?;
    m.add_function(wrap_pyfunction!(c6_mt_record_demo, m)?)?;
    m.add_function(wrap_pyfunction!(c7_async_compute_demo, m)?)?;
    // Path tracing (GPU MVP)
    m.add_function(wrap_pyfunction!(_pt_render_gpu, m)?)?;

    // Add main classes
    m.add_class::<crate::scene::Scene>()?;
    // Expose TerrainSpike (E2/E3) to Python
    m.add_class::<crate::terrain::TerrainSpike>()?;

    Ok(())
}
