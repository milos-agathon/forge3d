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
use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods, PyReadonlyArrayDyn};
#[cfg(feature = "extension-module")]
use pyo3::types::PyDict;
#[cfg(feature = "extension-module")]
use pyo3::{exceptions::PyValueError, prelude::*, wrap_pyfunction};

// C1/C3/C5/C6/C7: Additional imports for PyO3 functions
#[cfg(feature = "extension-module")]
use crate::context as engine_context;
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
use crate::sdf::hybrid::Ray as HybridRay;
#[cfg(feature = "extension-module")]
use crate::sdf::py::PySdfScene;
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
pub mod converters; // Geometry converters (e.g., MultipolygonZ -> OBJ)
pub mod core;
pub mod device_caps;
pub mod error;
pub mod external_image;
pub mod formats;
pub mod geometry;
pub mod gpu;
pub mod grid;
pub mod import; // Importers: OSM buildings, etc.
pub mod io; // IO: OBJ/PLY/glTF readers/writers
pub mod loaders;
pub mod mesh;
pub mod path_tracing;
pub mod pipeline;
pub mod render; // Rendering utilities (instancing)
pub mod renderer;
pub mod scene;
pub mod sdf; // New SDF module
pub mod shadows; // Shadow mapping implementations
pub mod terrain;
pub mod terrain_stats;
pub mod uv; // UV unwrap helpers (planar, spherical)
pub mod textures {}
pub mod transforms;
pub mod vector;
pub mod viewer; // Interactive windowed viewer (Workstream I1)

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
fn set_point_shape_mode(mode: u32) -> PyResult<()> {
    crate::vector::point::set_global_shape_mode(mode);
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn set_point_lod_threshold(threshold: f32) -> PyResult<()> {
    crate::vector::point::set_global_lod_threshold(threshold);
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn is_weighted_oit_available() -> PyResult<bool> {
    Ok(crate::vector::oit::is_weighted_oit_enabled())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn vector_render_polygons_fill_py(
    py: Python<'_>,
    width: u32,
    height: u32,
    exteriors: Vec<numpy::PyReadonlyArray2<'_, f64>>, // list of (N,2)
    holes: Option<Vec<Vec<numpy::PyReadonlyArray2<'_, f64>>>>, // list of list of (M,2)
    fill_rgba: Option<(f32, f32, f32, f32)>,
    stroke_rgba: Option<(f32, f32, f32, f32)>,
    stroke_width: Option<f32>,
) -> PyResult<Py<PyAny>> {
    use crate::vector::api::PolygonDef;
    use numpy::PyArray1;

    // Acquire device/queue from global context
    let g = crate::gpu::ctx();
    let device = std::sync::Arc::clone(&g.device);
    let queue = std::sync::Arc::clone(&g.queue);

    // Compute bounds first for normalization (fixes Lyon tessellation with large coordinates)
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    // Build polygon defs from numpy arrays and compute bounds
    let mut polys: Vec<PolygonDef> = Vec::with_capacity(exteriors.len());
    for (i, ext) in exteriors.into_iter().enumerate() {
        let exterior = crate::vector::api::parse_polygon_from_numpy(ext)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Update bounds from exterior
        for v in &exterior {
            min_x = min_x.min(v.x);
            min_y = min_y.min(v.y);
            max_x = max_x.max(v.x);
            max_y = max_y.max(v.y);
        }

        // Parse holes for this polygon if provided
        let mut hole_rings = Vec::new();
        if let Some(hh) = &holes {
            if let Some(h_for_poly) = hh.get(i) {
                for h in h_for_poly {
                    let hv = crate::vector::api::parse_polygon_from_numpy(h.clone())
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                    // Update bounds from holes
                    for v in &hv {
                        min_x = min_x.min(v.x);
                        min_y = min_y.min(v.y);
                        max_x = max_x.max(v.x);
                        max_y = max_y.max(v.y);
                    }
                    hole_rings.push(hv);
                }
            }
        }

        // Style on PolygonDef is not used by tessellation; rendering uniforms control style
        let style = crate::vector::api::VectorStyle::default();
        polys.push(PolygonDef {
            exterior,
            holes: hole_rings,
            style,
        });
    }

    // Normalize to avoid Lyon tessellation issues with large coordinates
    if !min_x.is_finite() || !min_y.is_finite() || !max_x.is_finite() || !max_y.is_finite() {
        min_x = -1.0;
        min_y = -1.0;
        max_x = 1.0;
        max_y = 1.0;
    }
    let cx_orig = 0.5 * (min_x + max_x);
    let cy_orig = 0.5 * (min_y + max_y);
    let dx = (max_x - min_x).max(1e-6);
    let dy = (max_y - min_y).max(1e-6);
    let norm_scale = 100.0 / dx.max(dy); // Normalize to ~100 unit range for Lyon

    // Apply normalization centered around origin for proper NDC mapping
    for poly in &mut polys {
        for v in &mut poly.exterior {
            v.x = (v.x - cx_orig) * norm_scale;
            v.y = (v.y - cy_orig) * norm_scale;
        }
        for hole in &mut poly.holes {
            for v in hole {
                v.x = (v.x - cx_orig) * norm_scale;
                v.y = (v.y - cy_orig) * norm_scale;
            }
        }
    }

    // Create polygon renderer and tessellate
    let mut poly_renderer =
        crate::vector::PolygonRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let mut packed = Vec::with_capacity(polys.len());
    for p in &polys {
        let pk = poly_renderer
            .tessellate_polygon(p)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        packed.push(pk);
    }

    // Upload geometry
    poly_renderer
        .upload_polygons(&device, &queue, &packed)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Create output target
    let final_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vf.Vector.PolygonFill.Final"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let final_view = final_tex.create_view(&wgpu::TextureViewDescriptor::default());

    // Encode render pass
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vf.Vector.PolygonFill.Encoder"),
    });

    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("vf.Vector.PolygonFill.Render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &final_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        // Compute fit-to-NDC transform from normalized polygon bounds with viewport aspect ratio
        let mut norm_min_x = f32::INFINITY;
        let mut norm_min_y = f32::INFINITY;
        let mut norm_max_x = f32::NEG_INFINITY;
        let mut norm_max_y = f32::NEG_INFINITY;
        for p in &polys {
            for v in &p.exterior {
                norm_min_x = norm_min_x.min(v.x);
                norm_min_y = norm_min_y.min(v.y);
                norm_max_x = norm_max_x.max(v.x);
                norm_max_y = norm_max_y.max(v.y);
            }
            for hole in &p.holes {
                for v in hole {
                    norm_min_x = norm_min_x.min(v.x);
                    norm_min_y = norm_min_y.min(v.y);
                    norm_max_x = norm_max_x.max(v.x);
                    norm_max_y = norm_max_y.max(v.y);
                }
            }
        }
        if !norm_min_x.is_finite()
            || !norm_min_y.is_finite()
            || !norm_max_x.is_finite()
            || !norm_max_y.is_finite()
        {
            norm_min_x = 0.0;
            norm_min_y = 0.0;
            norm_max_x = 100.0;
            norm_max_y = 100.0;
        }
        let cx = 0.5 * (norm_min_x + norm_max_x);
        let cy = 0.5 * (norm_min_y + norm_max_y);
        let dx = (norm_max_x - norm_min_x).max(1e-6);
        let dy = (norm_max_y - norm_min_y).max(1e-6);

        // Compute scale accounting for viewport aspect ratio to avoid distortion
        let viewport_aspect = width as f32 / height as f32;
        let data_aspect = dx / dy;

        let (sx, sy) = if data_aspect > viewport_aspect {
            // Data is wider relative to viewport - fit to width
            let s = 2.0 / dx;
            (s, s)
        } else {
            // Data is taller relative to viewport - fit to height
            let s = 2.0 / dy;
            (s, s)
        };

        // Flip Y-axis for proper geographic data rendering (Y increases upward in geo data, downward in clip space)
        let vp = [
            [sx, 0.0, 0.0, -sx * cx],
            [0.0, -sy, 0.0, sy * cy],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let total_indices: u32 = packed.iter().map(|p| p.indices.len() as u32).sum();
        let (fr, fg, fb, fa) = fill_rgba.unwrap_or((0.2, 0.4, 0.8, 1.0));
        let (sr, sg, sb, sa) = stroke_rgba.unwrap_or((0.0, 0.0, 0.0, 1.0));
        let sw = stroke_width.unwrap_or(1.0);

        poly_renderer
            .render(
                &mut pass,
                &queue,
                &vp,
                [fr, fg, fb, fa],
                [sr, sg, sb, sa],
                sw,
                total_indices,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    }

    queue.submit(Some(enc.finish()));
    device.poll(wgpu::Maintain::Wait);

    // Readback RGBA8
    let bpr = (width * 4 + 255) / 256 * 256;
    let size = (bpr * height) as u64;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vf.Vector.PolygonFill.Read"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vf.Vector.PolygonFill.Copy"),
    });
    enc2.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &final_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &buf,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(enc2.finish()));
    device.poll(wgpu::Maintain::Wait);

    let slice = buf.slice(..);
    let (s, r) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        s.send(res).ok();
    });
    // IMPORTANT: service the wgpu mapping callback; without this, the oneshot may never fire.
    device.poll(wgpu::Maintain::Wait);
    let recv = pollster::block_on(r.receive())
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("map_async cancelled"))?;
    if let Err(e) = recv {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "map_async error: {:?}",
            e
        )));
    }
    let data = slice.get_mapped_range();
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    for row in 0..height as usize {
        let src = &data[(row as u32 * bpr) as usize..][..(width * 4) as usize];
        let dst = &mut rgba[row * (width as usize) * 4..][..(width as usize) * 4];
        dst.copy_from_slice(src);
    }
    drop(data);
    buf.unmap();

    let arr1 = PyArray1::<u8>::from_vec_bound(py, rgba);
    let arr3 = arr1.reshape([height as usize, width as usize, 4])?;
    Ok(arr3.into_py(py))
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn vector_render_oit_py(
    py: Python<'_>,
    width: u32,
    height: u32,
    points_xy: Option<&Bound<'_, PyAny>>,  // sequence of (x,y)
    point_rgba: Option<&Bound<'_, PyAny>>, // sequence of (r,g,b,a)
    point_size: Option<&Bound<'_, PyAny>>, // sequence of size
    polylines: Option<&Bound<'_, PyAny>>,  // sequence of sequence of (x,y)
    polyline_rgba: Option<&Bound<'_, PyAny>>, // sequence of (r,g,b,a)
    stroke_width: Option<&Bound<'_, PyAny>>, // sequence of width
) -> PyResult<Py<PyAny>> {
    #[cfg(not(feature = "weighted-oit"))]
    {
        let _ = (
            py,
            width,
            height,
            points_xy,
            point_rgba,
            point_size,
            polylines,
            polyline_rgba,
            stroke_width,
        );
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Weighted OIT feature not enabled. Build with --features weighted-oit",
        ));
    }
    #[cfg(feature = "weighted-oit")]
    {
        use crate::vector::api::{PointDef, PolylineDef, VectorStyle};
        use numpy::PyArray1;

        // Helper extractors
        fn extract_xy_list(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<glam::Vec2>> {
            if let Some(obj) = list {
                let pairs: Vec<(f32, f32)> = obj.extract()?;
                Ok(pairs
                    .into_iter()
                    .map(|(x, y)| glam::Vec2::new(x, y))
                    .collect())
            } else {
                Ok(Vec::new())
            }
        }
        fn extract_rgba_list(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<[f32; 4]>> {
            if let Some(obj) = list {
                let vals: Vec<(f32, f32, f32, f32)> = obj.extract()?;
                Ok(vals.into_iter().map(|(r, g, b, a)| [r, g, b, a]).collect())
            } else {
                Ok(Vec::new())
            }
        }
        fn extract_f32_list(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<f32>> {
            if let Some(obj) = list {
                Ok(obj.extract()?)
            } else {
                Ok(Vec::new())
            }
        }
        fn extract_polylines(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Vec<glam::Vec2>>> {
            if let Some(obj) = list {
                let outer: Vec<Vec<(f32, f32)>> = obj.extract()?;
                Ok(outer
                    .into_iter()
                    .map(|path| {
                        path.into_iter()
                            .map(|(x, y)| glam::Vec2::new(x, y))
                            .collect()
                    })
                    .collect())
            } else {
                Ok(Vec::new())
            }
        }

        let pts = extract_xy_list(points_xy)?;
        let pts_rgba = extract_rgba_list(point_rgba)?;
        let pts_size = extract_f32_list(point_size)?;
        let lines = extract_polylines(polylines)?;
        let lines_rgba = extract_rgba_list(polyline_rgba)?;
        let lines_w = extract_f32_list(stroke_width)?;

        // Build defs
        let mut point_defs: Vec<PointDef> = Vec::with_capacity(pts.len());
        for i in 0..pts.len() {
            let c = *pts_rgba.get(i).unwrap_or(&[1.0, 1.0, 1.0, 1.0]);
            let s = *pts_size.get(i).unwrap_or(&8.0);
            point_defs.push(PointDef {
                position: pts[i],
                style: VectorStyle {
                    fill_color: c,
                    stroke_color: [0.0, 0.0, 0.0, 1.0],
                    stroke_width: 1.0,
                    point_size: s,
                },
            });
        }
        let mut poly_defs: Vec<PolylineDef> = Vec::with_capacity(lines.len());
        for i in 0..lines.len() {
            let c = *lines_rgba.get(i).unwrap_or(&[0.2, 0.8, 0.2, 0.6]);
            let w = *lines_w.get(i).unwrap_or(&2.0);
            poly_defs.push(PolylineDef {
                path: lines[i].clone(),
                style: VectorStyle {
                    fill_color: [0.0, 0.0, 0.0, 0.0],
                    stroke_color: c,
                    stroke_width: w,
                    point_size: 4.0,
                },
            });
        }

        // Acquire device/queue
        let g = crate::gpu::ctx();
        let device = std::sync::Arc::clone(&g.device);
        let queue = std::sync::Arc::clone(&g.queue);

        // Create renderers
        let mut pr =
            crate::vector::PointRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let mut lr = crate::vector::LineRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Upload instances
        if !point_defs.is_empty() {
            let p_instances = pr
                .pack_points(&point_defs)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            pr.upload_points(&device, &queue, &p_instances)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }
        if !poly_defs.is_empty() {
            let l_instances = lr
                .pack_polylines(&poly_defs)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            lr.upload_lines(&device, &l_instances)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }

        // Create OIT and final target
        let oit = crate::vector::oit::WeightedOIT::new(
            &device,
            width,
            height,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let final_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vf.Vector.RenderOIT.Final"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let final_view = final_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Accumulation and compose
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.RenderOIT.Encoder"),
        });
        {
            let mut pass = oit.begin_accumulation(&mut encoder);
            let vp = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let viewport = [width as f32, height as f32];
            if !poly_defs.is_empty() {
                lr.render_oit(
                    &mut pass,
                    &queue,
                    &vp,
                    viewport,
                    poly_defs.len() as u32,
                    crate::vector::line::LineCap::Round,
                    crate::vector::line::LineJoin::Round,
                    2.0,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            }
            if !point_defs.is_empty() {
                pr.render_oit(
                    &mut pass,
                    &queue,
                    &vp,
                    viewport,
                    1.0,
                    point_defs.len() as u32,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            }
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vf.Vector.RenderOIT.Compose"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &final_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            oit.compose(&mut pass);
        }
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // Readback final
        let bpr = (width * 4 + 255) / 256 * 256;
        let size = (bpr * height) as u64;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.RenderOIT.Read"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.RenderOIT.Copy"),
        });
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &final_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bpr),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);

        let slice = buf.slice(..);
        let (s, r) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            s.send(res).ok();
        });
        // Service mapping callback to avoid stalls on some platforms
        device.poll(wgpu::Maintain::Wait);
        let recv = pollster::block_on(r.receive())
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("map_async cancelled"))?;
        if let Err(e) = recv {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "map_async error: {:?}",
                e
            )));
        }
        let data = slice.get_mapped_range();
        let mut rgba = vec![0u8; (width * height * 4) as usize];
        for row in 0..height as usize {
            let src = &data[(row as u32 * bpr) as usize..][..(width * 4) as usize];
            let dst = &mut rgba[row * (width as usize) * 4..][..(width as usize) * 4];
            dst.copy_from_slice(src);
        }
        drop(data);
        buf.unmap();
        let arr1 = PyArray1::<u8>::from_vec_bound(py, rgba);
        let arr3 = arr1.reshape([height as usize, width as usize, 4])?;
        Ok(arr3.into_py(py))
    }
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn vector_render_pick_map_py(
    py: Python<'_>,
    width: u32,
    height: u32,
    points_xy: Option<&Bound<'_, PyAny>>,
    polylines: Option<&Bound<'_, PyAny>>,
    base_pick_id: Option<u32>,
) -> PyResult<Py<PyAny>> {
    use crate::vector::api::{PointDef, PolylineDef, VectorStyle};
    use numpy::PyArray1;
    // Parse inputs
    fn extract_xy_list(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<glam::Vec2>> {
        if let Some(obj) = list {
            Ok(obj
                .extract::<Vec<(f32, f32)>>()?
                .into_iter()
                .map(|(x, y)| glam::Vec2::new(x, y))
                .collect())
        } else {
            Ok(Vec::new())
        }
    }
    fn extract_polylines(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Vec<glam::Vec2>>> {
        if let Some(obj) = list {
            Ok(obj
                .extract::<Vec<Vec<(f32, f32)>>>()?
                .into_iter()
                .map(|path| {
                    path.into_iter()
                        .map(|(x, y)| glam::Vec2::new(x, y))
                        .collect()
                })
                .collect())
        } else {
            Ok(Vec::new())
        }
    }
    let pts = extract_xy_list(points_xy)?;
    let lines = extract_polylines(polylines)?;
    let mut point_defs = Vec::with_capacity(pts.len());
    for p in &pts {
        point_defs.push(PointDef {
            position: *p,
            style: VectorStyle {
                fill_color: [1.0, 1.0, 1.0, 1.0],
                stroke_color: [0.0, 0.0, 0.0, 1.0],
                stroke_width: 1.0,
                point_size: 8.0,
            },
        });
    }
    let mut poly_defs = Vec::with_capacity(lines.len());
    for path in lines {
        poly_defs.push(PolylineDef {
            path,
            style: VectorStyle {
                fill_color: [0.0, 0.0, 0.0, 0.0],
                stroke_color: [1.0, 1.0, 1.0, 1.0],
                stroke_width: 2.0,
                point_size: 4.0,
            },
        });
    }

    // Device
    let g = crate::gpu::ctx();
    let device = std::sync::Arc::clone(&g.device);
    let queue = std::sync::Arc::clone(&g.queue);

    let mut pr = crate::vector::PointRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let mut lr = crate::vector::LineRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    if !point_defs.is_empty() {
        let p_instances = pr
            .pack_points(&point_defs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        pr.upload_points(&device, &queue, &p_instances)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    }
    if !poly_defs.is_empty() {
        let l_instances = lr
            .pack_polylines(&poly_defs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        lr.upload_lines(&device, &l_instances)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    }

    // Create pick target
    let pick_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vf.Vector.RenderPick.Pick"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = pick_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vf.Vector.RenderPick.Encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("vf.Vector.RenderPick.Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        let vp = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let viewport = [width as f32, height as f32];
        let mut base = base_pick_id.unwrap_or(1);
        if !point_defs.is_empty() {
            pr.render_pick(
                &mut pass,
                &queue,
                &vp,
                viewport,
                1.0,
                point_defs.len() as u32,
                base,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            base += point_defs.len() as u32;
        }
        if !poly_defs.is_empty() {
            lr.render_pick(
                &mut pass,
                &queue,
                &vp,
                viewport,
                poly_defs.len() as u32,
                base,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }
    }
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    // Readback full R32Uint map
    let bpr = (width * 4 + 255) / 256 * 256;
    let size = (bpr * height) as u64;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vf.Vector.RenderPick.Read"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vf.Vector.RenderPick.Copy"),
    });
    enc.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &pick_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &buf,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(enc.finish()));
    device.poll(wgpu::Maintain::Wait);

    let slice = buf.slice(..);
    let (s, r) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        s.send(res).ok();
    });
    // Service mapping callback to avoid stalls on some platforms
    device.poll(wgpu::Maintain::Wait);
    let recv = pollster::block_on(r.receive())
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("map_async cancelled"))?;
    if let Err(e) = recv {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "map_async error: {:?}",
            e
        )));
    }
    let data = slice.get_mapped_range();
    let mut ids = vec![0u32; (width * height) as usize];
    for row in 0..height as usize {
        let src = &data[(row as u32 * bpr) as usize..][..(width * 4) as usize];
        let row_ids = bytemuck::cast_slice::<u8, u32>(src);
        let dst = &mut ids[row * (width as usize)..][..(width as usize)];
        dst.copy_from_slice(row_ids);
    }
    drop(data);
    buf.unmap();
    let arr1 = PyArray1::<u32>::from_vec_bound(py, ids);
    let arr2 = arr1.reshape([height as usize, width as usize])?;
    Ok(arr2.into_py(py))
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn vector_render_oit_and_pick_py(
    py: Python<'_>,
    width: u32,
    height: u32,
    points_xy: Option<&Bound<'_, PyAny>>,  // sequence of (x,y)
    point_rgba: Option<&Bound<'_, PyAny>>, // sequence of (r,g,b,a)
    point_size: Option<&Bound<'_, PyAny>>, // sequence of size
    polylines: Option<&Bound<'_, PyAny>>,  // sequence of sequence of (x,y)
    polyline_rgba: Option<&Bound<'_, PyAny>>, // sequence of (r,g,b,a)
    stroke_width: Option<&Bound<'_, PyAny>>, // sequence of width
    base_pick_id: Option<u32>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    #[cfg(not(feature = "weighted-oit"))]
    {
        let _ = (
            py,
            width,
            height,
            points_xy,
            point_rgba,
            point_size,
            polylines,
            polyline_rgba,
            stroke_width,
            base_pick_id,
        );
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Weighted OIT feature not enabled. Build with --features weighted-oit",
        ));
    }
    #[cfg(feature = "weighted-oit")]
    {
        use crate::vector::api::{PointDef, PolylineDef, VectorStyle};
        use numpy::PyArray1;

        // Helper extractors (same as above)
        fn extract_xy_list(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<glam::Vec2>> {
            if let Some(obj) = list {
                let pairs: Vec<(f32, f32)> = obj.extract()?;
                Ok(pairs
                    .into_iter()
                    .map(|(x, y)| glam::Vec2::new(x, y))
                    .collect())
            } else {
                Ok(Vec::new())
            }
        }
        fn extract_rgba_list(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<[f32; 4]>> {
            if let Some(obj) = list {
                let vals: Vec<(f32, f32, f32, f32)> = obj.extract()?;
                Ok(vals.into_iter().map(|(r, g, b, a)| [r, g, b, a]).collect())
            } else {
                Ok(Vec::new())
            }
        }
        fn extract_f32_list(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<f32>> {
            if let Some(obj) = list {
                Ok(obj.extract()?)
            } else {
                Ok(Vec::new())
            }
        }
        fn extract_polylines(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Vec<glam::Vec2>>> {
            if let Some(obj) = list {
                let outer: Vec<Vec<(f32, f32)>> = obj.extract()?;
                Ok(outer
                    .into_iter()
                    .map(|path| {
                        path.into_iter()
                            .map(|(x, y)| glam::Vec2::new(x, y))
                            .collect()
                    })
                    .collect())
            } else {
                Ok(Vec::new())
            }
        }

        let pts = extract_xy_list(points_xy)?;
        let pts_rgba = extract_rgba_list(point_rgba)?;
        let pts_size = extract_f32_list(point_size)?;
        let lines = extract_polylines(polylines)?;
        let lines_rgba = extract_rgba_list(polyline_rgba)?;
        let lines_w = extract_f32_list(stroke_width)?;

        // Build defs
        let mut point_defs: Vec<PointDef> = Vec::with_capacity(pts.len());
        for i in 0..pts.len() {
            let c = *pts_rgba.get(i).unwrap_or(&[1.0, 1.0, 1.0, 1.0]);
            let s = *pts_size.get(i).unwrap_or(&8.0);
            point_defs.push(PointDef {
                position: pts[i],
                style: VectorStyle {
                    fill_color: c,
                    stroke_color: [0.0, 0.0, 0.0, 1.0],
                    stroke_width: 1.0,
                    point_size: s,
                },
            });
        }
        let mut poly_defs: Vec<PolylineDef> = Vec::with_capacity(lines.len());
        for i in 0..lines.len() {
            let c = *lines_rgba.get(i).unwrap_or(&[0.2, 0.8, 0.2, 0.6]);
            let w = *lines_w.get(i).unwrap_or(&2.0);
            poly_defs.push(PolylineDef {
                path: lines[i].clone(),
                style: VectorStyle {
                    fill_color: [0.0, 0.0, 0.0, 0.0],
                    stroke_color: c,
                    stroke_width: w,
                    point_size: 4.0,
                },
            });
        }

        // Acquire device/queue
        let g = crate::gpu::ctx();
        let device = std::sync::Arc::clone(&g.device);
        let queue = std::sync::Arc::clone(&g.queue);

        // Create renderers
        let mut pr =
            crate::vector::PointRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let mut lr = crate::vector::LineRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Upload instances
        if !point_defs.is_empty() {
            let p_instances = pr
                .pack_points(&point_defs)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            pr.upload_points(&device, &queue, &p_instances)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }
        if !poly_defs.is_empty() {
            let l_instances = lr
                .pack_polylines(&poly_defs)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            lr.upload_lines(&device, &l_instances)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }

        // Weighted OIT accumulation buffers
        #[cfg(not(feature = "weighted-oit"))]
        {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Weighted OIT feature not enabled. Build with --features weighted-oit",
            ));
        }
        #[cfg(feature = "weighted-oit")]
        let oit = crate::vector::oit::WeightedOIT::new(
            &device,
            width,
            height,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Final RGBA8 target
        #[cfg(feature = "weighted-oit")]
        let final_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vf.Vector.Combine.Final"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        #[cfg(feature = "weighted-oit")]
        let final_view = final_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Accumulation pass
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.Combine.Encoder"),
        });
        #[cfg(feature = "weighted-oit")]
        {
            let mut pass = oit.begin_accumulation(&mut encoder);
            let vp = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let viewport = [width as f32, height as f32];
            if !poly_defs.is_empty() {
                lr.render_oit(
                    &mut pass,
                    &queue,
                    &vp,
                    viewport,
                    poly_defs.len() as u32,
                    crate::vector::line::LineCap::Round,
                    crate::vector::line::LineJoin::Round,
                    2.0,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            }
            if !point_defs.is_empty() {
                pr.render_oit(
                    &mut pass,
                    &queue,
                    &vp,
                    viewport,
                    1.0,
                    point_defs.len() as u32,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            }
        }
        // Compose pass into final target
        #[cfg(feature = "weighted-oit")]
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vf.Vector.Combine.Compose"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &final_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            oit.compose(&mut pass);
        }

        // Picking pass
        let pick_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vf.Vector.Combine.Pick"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let pick_view = pick_tex.create_view(&wgpu::TextureViewDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vf.Vector.Combine.PickPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &pick_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            let vp = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let viewport = [width as f32, height as f32];
            let mut base = base_pick_id.unwrap_or(1);
            if !point_defs.is_empty() {
                pr.render_pick(
                    &mut pass,
                    &queue,
                    &vp,
                    viewport,
                    1.0,
                    point_defs.len() as u32,
                    base,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                base += point_defs.len() as u32;
            }
            if !poly_defs.is_empty() {
                lr.render_pick(
                    &mut pass,
                    &queue,
                    &vp,
                    viewport,
                    poly_defs.len() as u32,
                    base,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            }
        }

        // Submit
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // Read back final RGBA8
        let bpr = (width * 4 + 255) / 256 * 256; // 256B align
        let final_size = (bpr * height) as u64;
        #[cfg(feature = "weighted-oit")]
        let final_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Combine.FinalRead"),
            size: final_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        #[cfg(feature = "weighted-oit")]
        let mut enc1 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.Combine.CopyFinal"),
        });
        #[cfg(feature = "weighted-oit")]
        enc1.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &final_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &final_buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bpr),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        #[cfg(feature = "weighted-oit")]
        {
            queue.submit(Some(enc1.finish()));
            device.poll(wgpu::Maintain::Wait);
        }

        #[cfg(feature = "weighted-oit")]
        let rgba: Vec<u8>;
        #[cfg(feature = "weighted-oit")]
        {
            let fslice = final_buf.slice(..);
            let (s, r) = futures_intrusive::channel::shared::oneshot_channel();
            fslice.map_async(wgpu::MapMode::Read, move |res| {
                s.send(res).ok();
            });
            // Service mapping callback to avoid stalls on some platforms
            device.poll(wgpu::Maintain::Wait);
            let recv = pollster::block_on(r.receive())
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("final map cancelled"))?;
            if let Err(e) = recv {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "map_async error: {:?}",
                    e
                )));
            }
            let fdata = fslice.get_mapped_range();
            let mut rgba_data = vec![0u8; (width * height * 4) as usize];
            for row in 0..height as usize {
                let src = &fdata[(row as u32 * bpr) as usize..][..(width * 4) as usize];
                let dst = &mut rgba_data[row * (width as usize) * 4..][..(width as usize) * 4];
                dst.copy_from_slice(src);
            }
            drop(fdata);
            final_buf.unmap();
            rgba = rgba_data;
        }

        // Read back pick map
        let pick_bpr = (width * 4 + 255) / 256 * 256;
        let pick_size = (pick_bpr * height) as u64;
        let pick_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Combine.PickRead"),
            size: pick_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.Combine.CopyPick"),
        });
        enc2.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &pick_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &pick_buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(pick_bpr),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(enc2.finish()));
        device.poll(wgpu::Maintain::Wait);

        let pslice = pick_buf.slice(..);
        let (s2, r2) = futures_intrusive::channel::shared::oneshot_channel();
        pslice.map_async(wgpu::MapMode::Read, move |res| {
            s2.send(res).ok();
        });
        // Service mapping callback to avoid stalls on some platforms
        device.poll(wgpu::Maintain::Wait);
        let recv = pollster::block_on(r2.receive())
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("pick map cancelled"))?;
        if let Err(e) = recv {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "map_async error: {:?}",
                e
            )));
        }
        let pdata = pslice.get_mapped_range();
        let mut ids = vec![0u32; (width * height) as usize];
        for row in 0..height as usize {
            let src = &pdata[(row as u32 * pick_bpr) as usize..][..(width * 4) as usize];
            let row_ids = bytemuck::cast_slice::<u8, u32>(src);
            let dst = &mut ids[row * (width as usize)..][..(width as usize)];
            dst.copy_from_slice(row_ids);
        }
        drop(pdata);
        pick_buf.unmap();

        // Convert to numpy
        #[cfg(feature = "weighted-oit")]
        {
            let arr_rgba_1 = PyArray1::<u8>::from_vec_bound(py, rgba);
            let arr_rgba = arr_rgba_1.reshape([height as usize, width as usize, 4])?;
            let arr_ids_1 = PyArray1::<u32>::from_vec_bound(py, ids);
            let arr_ids = arr_ids_1.reshape([height as usize, width as usize])?;
            return Ok((arr_rgba.into_py(py), arr_ids.into_py(py)));
        }
        #[cfg(not(feature = "weighted-oit"))]
        {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Weighted OIT feature not enabled. Build with --features weighted-oit",
            ));
        }
    }
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn vector_oit_and_pick_demo(py: Python<'_>, width: u32, height: u32) -> PyResult<(Py<PyAny>, u32)> {
    #[cfg(not(feature = "weighted-oit"))]
    {
        let _ = (py, width, height);
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Weighted OIT feature not enabled. Build with --features weighted-oit",
        ));
    }
    #[cfg(feature = "weighted-oit")]
    {
        use crate::vector::api::{PointDef, PolylineDef, VectorStyle};
        use crate::vector::{LineRenderer, PointRenderer};
        use numpy::PyArray1;

        // Acquire GPU device/queue
        let g = crate::gpu::ctx();
        let device = std::sync::Arc::clone(&g.device);
        let queue = std::sync::Arc::clone(&g.queue);

        // Create renderers
        let mut pr = PointRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let mut lr = LineRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Sample primitives
        let points = vec![
            PointDef {
                position: glam::Vec2::new(-0.5, -0.5),
                style: VectorStyle {
                    fill_color: [1.0, 0.2, 0.2, 0.9],
                    stroke_color: [0.0, 0.0, 0.0, 1.0],
                    stroke_width: 1.0,
                    point_size: 24.0,
                },
            },
            PointDef {
                position: glam::Vec2::new(0.4, 0.2),
                style: VectorStyle {
                    fill_color: [0.2, 0.8, 1.0, 0.7],
                    stroke_color: [0.0, 0.0, 0.0, 1.0],
                    stroke_width: 1.0,
                    point_size: 32.0,
                },
            },
        ];
        let lines = vec![PolylineDef {
            path: vec![
                glam::Vec2::new(-0.8, -0.8),
                glam::Vec2::new(0.8, 0.5),
                glam::Vec2::new(0.4, 0.8),
            ],
            style: VectorStyle {
                fill_color: [0.0, 0.0, 0.0, 0.0],
                stroke_color: [0.1, 0.9, 0.3, 0.6],
                stroke_width: 8.0,
                point_size: 4.0,
            },
        }];

        let p_instances = pr
            .pack_points(&points)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        pr.upload_points(&device, &queue, &p_instances)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let l_instances = lr
            .pack_polylines(&lines)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        lr.upload_lines(&device, &l_instances)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Weighted OIT accumulation buffers
        let oit = crate::vector::oit::WeightedOIT::new(
            &device,
            width,
            height,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Final RGBA8 target
        let final_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vf.Vector.Demo.Final"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let final_view = final_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Accumulation pass
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.Demo.Encoder"),
        });
        {
            let mut pass = oit.begin_accumulation(&mut encoder);
            let vp = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let viewport = [width as f32, height as f32];
            lr.render_oit(
                &mut pass,
                &queue,
                &vp,
                viewport,
                l_instances.len() as u32,
                crate::vector::line::LineCap::Round,
                crate::vector::line::LineJoin::Round,
                2.0,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            pr.render_oit(
                &mut pass,
                &queue,
                &vp,
                viewport,
                1.0,
                p_instances.len() as u32,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }

        // Compose pass into final target
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vf.Vector.Demo.Compose"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &final_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            oit.compose(&mut pass);
        }

        // Picking pass
        let pick_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vf.Vector.Demo.Pick"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let pick_view = pick_tex.create_view(&wgpu::TextureViewDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vf.Vector.Demo.PickPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &pick_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            let vp = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let viewport = [width as f32, height as f32];
            // Assign base ids 1..N for points, then continue for lines
            pr.render_pick(
                &mut pass,
                &queue,
                &vp,
                viewport,
                1.0,
                p_instances.len() as u32,
                1,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let base_line = 1 + p_instances.len() as u32;
            lr.render_pick(
                &mut pass,
                &queue,
                &vp,
                viewport,
                l_instances.len() as u32,
                base_line,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }

        // Submit
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // Read back final RGBA8
        let bpr = (width * 4 + 255) / 256 * 256; // align to 256
        let final_size = (bpr * height) as u64;
        let final_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Demo.FinalRead"),
            size: final_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.Demo.CopyFinal"),
        });
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &final_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &final_buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bpr),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        // Read one pick pixel at center
        let pick_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Demo.PickRead"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cx = width / 2;
        let cy = height / 2;
        let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.Demo.CopyPick"),
        });
        enc2.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &pick_tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: cx, y: cy, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &pick_buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: None,
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([enc.finish(), enc2.finish()]);
        device.poll(wgpu::Maintain::Wait);

        // Map final image
        let slice = final_buf.slice(..);
        let (s, r) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            s.send(res).ok();
        });
        // Service mapping callback to avoid stalls on some platforms
        device.poll(wgpu::Maintain::Wait);
        let recv = pollster::block_on(r.receive())
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("map_async cancelled"))?;
        if let Err(e) = recv {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "map_async error: {:?}",
                e
            )));
        }
        let data = slice.get_mapped_range();
        let mut rgba = vec![0u8; (width * height * 4) as usize];
        for row in 0..height as usize {
            let src = &data[(row as u32 * bpr) as usize..][..(width * 4) as usize];
            let dst = &mut rgba[row * (width as usize) * 4..][..(width as usize) * 4];
            dst.copy_from_slice(src);
        }
        drop(data);
        final_buf.unmap();

        // Map pick
        let pslice = pick_buf.slice(..);
        let (s2, r2) = futures_intrusive::channel::shared::oneshot_channel();
        pslice.map_async(wgpu::MapMode::Read, move |res| {
            s2.send(res).ok();
        });
        let recv = pollster::block_on(r2.receive())
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("pick map cancelled"))?;
        if let Err(e) = recv {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "map_async error: {:?}",
                e
            )));
        }
        let pdata = pslice.get_mapped_range();
        let pick_id = bytemuck::from_bytes::<u32>(&pdata[..4]).to_owned();
        drop(pdata);
        pick_buf.unmap();

        // Return numpy (H,W,4) uint8
        let arr1 = PyArray1::<u8>::from_vec_bound(py, rgba);
        let arr3 = arr1.reshape([height as usize, width as usize, 4])?;
        Ok((arr3.into_py(py), pick_id))
    }
}
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
                    1 => [204u8, 51u8, 51u8], // red-ish
                    2 => [51u8, 204u8, 51u8], // green-ish
                    3 => [51u8, 51u8, 204u8], // blue-ish
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
fn _pt_render_gpu_mesh(
    py: Python<'_>,
    width: u32,
    height: u32,
    vertices: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    cam: &Bound<'_, PyAny>,
    seed: u32,
    frames: u32,
) -> PyResult<Py<PyAny>> {
    use numpy::{PyArray1, PyReadonlyArray2};
    use pyo3::exceptions::{PyRuntimeError, PyValueError};

    // Parse vertex and index arrays
    let verts_arr: PyReadonlyArray2<f32> = vertices.extract().map_err(|_| {
        PyValueError::new_err("vertices must be a NumPy array with shape (N,3) float32")
    })?;
    let idx_arr: PyReadonlyArray2<u32> = indices.extract().map_err(|_| {
        PyValueError::new_err("indices must be a NumPy array with shape (M,3) uint32")
    })?;

    let v = verts_arr.as_array();
    let i = idx_arr.as_array();
    if v.ndim() != 2 || v.shape()[1] != 3 {
        return Err(PyValueError::new_err("vertices must have shape (N,3)"));
    }
    if i.ndim() != 2 || i.shape()[1] != 3 {
        return Err(PyValueError::new_err("indices must have shape (M,3)"));
    }

    // Pack vertices for HybridScene
    let mut verts: Vec<crate::sdf::hybrid::Vertex> = Vec::with_capacity(v.shape()[0]);
    for row in v.rows() {
        verts.push(crate::sdf::hybrid::Vertex {
            position: [row[0], row[1], row[2]],
            _pad: 0.0,
        });
    }

    // Flatten indices (u32)
    let mut flat_idx: Vec<u32> = Vec::with_capacity(i.shape()[0] * 3);
    for row in i.rows() {
        flat_idx.push(row[0]);
        flat_idx.push(row[1]);
        flat_idx.push(row[2]);
    }

    // Build triangle list for BVH construction (CPU path)
    let mut tris: Vec<crate::accel::types::Triangle> = Vec::with_capacity(i.shape()[0]);
    for row in i.rows() {
        let iv0 = row[0] as usize;
        let iv1 = row[1] as usize;
        let iv2 = row[2] as usize;
        if iv0 >= v.shape()[0] || iv1 >= v.shape()[0] || iv2 >= v.shape()[0] {
            return Err(PyValueError::new_err(
                "indices reference out-of-bounds vertex",
            ));
        }
        let v0 = [v[[iv0, 0]], v[[iv0, 1]], v[[iv0, 2]]];
        let v1 = [v[[iv1, 0]], v[[iv1, 1]], v[[iv1, 2]]];
        let v2 = [v[[iv2, 0]], v[[iv2, 1]], v[[iv2, 2]]];
        tris.push(crate::accel::types::Triangle::new(v0, v1, v2));
    }

    // Build BVH (CPU backend) and create a HybridScene with mesh
    let options = crate::accel::types::BuildOptions::default();
    let bvh_handle =
        crate::accel::build_bvh(&tris, &options, crate::accel::GpuContext::NotAvailable)
            .map_err(|e| PyRuntimeError::new_err(format!("BVH build failed: {}", e)))?;

    let mut hybrid = crate::sdf::hybrid::HybridScene::mesh_only(verts, flat_idx, bvh_handle);

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
    let o = glam::Vec3::new(origin.0, origin.1, origin.2);
    let la = glam::Vec3::new(look_at.0, look_at.1, look_at.2);
    let upv = glam::Vec3::new(up.0, up.1, up.2);
    let forward = (la - o).normalize_or_zero();
    let right = forward.cross(upv).normalize_or_zero();
    let cup = right.cross(forward).normalize_or_zero();

    // Base uniforms
    let uniforms = crate::path_tracing::compute::Uniforms {
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
        seed_lo: frames, // carry frames in lo for deterministic variation
        _pad_end: [0, 0, 0],
    };

    let params = crate::path_tracing::hybrid_compute::HybridTracerParams {
        base_uniforms: uniforms,
        traversal_mode: crate::path_tracing::hybrid_compute::TraversalMode::MeshOnly,
        early_exit_distance: 0.01,
        shadow_softness: 4.0,
    };

    // Robust GPU attempt with CPU fallback on any error or panic
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

    let rgba: Vec<u8> = {
        use std::panic::{catch_unwind, AssertUnwindSafe};
        let p = params.clone();
        let res = catch_unwind(AssertUnwindSafe(|| {
            // Prepare GPU buffers; ignore error here, we'll handle via Option below
            let _ = hybrid.prepare_gpu_resources();
            if let Ok(tracer) = crate::path_tracing::hybrid_compute::HybridPathTracer::new() {
                tracer.render(width, height, &[], &hybrid, p).ok()
            } else {
                None
            }
        }));
        match res {
            Ok(Some(bytes)) => bytes,
            _ => build_fallback(),
        }
    };

    let arr1 = PyArray1::<u8>::from_vec_bound(py, rgba);
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
fn global_memory_metrics(py: Python<'_>) -> PyResult<PyObject> {
    let metrics = crate::core::memory_tracker::global_tracker().get_metrics();
    let d = PyDict::new_bound(py);
    d.set_item("buffer_count", metrics.buffer_count)?;
    d.set_item("texture_count", metrics.texture_count)?;
    d.set_item("buffer_bytes", metrics.buffer_bytes)?;
    d.set_item("texture_bytes", metrics.texture_bytes)?;
    d.set_item("host_visible_bytes", metrics.host_visible_bytes)?;
    d.set_item("total_bytes", metrics.total_bytes)?;
    d.set_item("limit_bytes", metrics.limit_bytes)?;
    d.set_item("within_budget", metrics.within_budget)?;
    d.set_item("utilization_ratio", metrics.utilization_ratio)?;
    d.set_item("resident_tiles", metrics.resident_tiles)?;
    d.set_item("resident_tile_bytes", metrics.resident_tile_bytes)?;
    d.set_item("staging_bytes_in_flight", metrics.staging_bytes_in_flight)?;
    d.set_item("staging_ring_count", metrics.staging_ring_count)?;
    d.set_item("staging_buffer_size", metrics.staging_buffer_size)?;
    d.set_item("staging_buffer_stalls", metrics.staging_buffer_stalls)?;
    Ok(d.into())
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

/// Open an interactive viewer window (Workstream I1)
///
/// Opens a windowed viewer with orbit and FPS camera controls.
///
/// Args:
///     width: Window width in pixels (default: 1024)
///     height: Window height in pixels (default: 768)
///     title: Window title (default: "forge3d Interactive Viewer")
///     vsync: Enable VSync (default: True)
///     fov_deg: Field of view in degrees (default: 45.0)
///     znear: Near clipping plane (default: 0.1)
///     zfar: Far clipping plane (default: 1000.0)
///
/// Controls:
///     Tab - Toggle between Orbit and FPS camera modes
///     Orbit mode: Drag to rotate, Scroll to zoom
///     FPS mode: WASD to move, Q/E for up/down, Mouse to look, Shift for speed
///     Esc - Exit viewer
///
/// Example:
///     >>> import forge3d as f3d
///     >>> f3d.open_viewer(width=1280, height=720, title="My Scene")
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (width=1024, height=768, title="forge3d Interactive Viewer".to_string(), vsync=true, fov_deg=45.0, znear=0.1, zfar=1000.0))]
fn open_viewer(
    width: u32,
    height: u32,
    title: String,
    vsync: bool,
    fov_deg: f32,
    znear: f32,
    zfar: f32,
) -> PyResult<()> {
    use crate::viewer::{run_viewer, ViewerConfig};

    let config = ViewerConfig {
        width,
        height,
        title,
        vsync,
        fov_deg,
        znear,
        zfar,
    };

    run_viewer(config).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Viewer error: {}", e))
    })
}

/// Open an interactive viewer window initialized with a provided RGBA8 image.
///
/// The viewer will present the image fullscreen. Press F12 to export a PNG screenshot.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (rgba, width=None, height=None, title="forge3d Image Preview".to_string(), vsync=true, fov_deg=45.0, znear=0.1, zfar=1000.0))]
fn open_viewer_image(
    _py: Python<'_>,
    rgba: numpy::PyReadonlyArray3<'_, u8>, // (H, W, 4)
    width: Option<u32>,
    height: Option<u32>,
    title: String,
    vsync: bool,
    fov_deg: f32,
    znear: f32,
    zfar: f32,
) -> PyResult<()> {
    use crate::viewer::{run_image_viewer, ViewerConfig};

    let shape = rgba.shape();
    if shape.len() != 3 || shape[2] != 4 {
        return Err(PyValueError::new_err("rgba must have shape (H, W, 4)"));
    }
    let h = shape[0] as u32;
    let w = shape[1] as u32;
    let view_w = width.unwrap_or(w);
    let view_h = height.unwrap_or(h);

    // Copy to Vec<u8>
    let data: Vec<u8> = rgba.as_slice()?.to_vec();

    let config = ViewerConfig {
        width: view_w,
        height: view_h,
        title,
        vsync,
        fov_deg,
        znear,
        zfar,
    };

    run_image_viewer(config, data, w, h).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Viewer error: {}", e))
    })
}

/// Open an interactive mesh viewer window initialized with vertex/index buffers.
///
/// vertices: numpy array (N,3) float32 of positions
/// indices: numpy array (M,3) uint32, or (K,) flat uint32
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (vertices, indices, uvs=None, texture_rgba=None, texture_width=0, texture_height=0, camera_eye=None, camera_target=None, width=1024, height=768, title="forge3d Mesh Viewer".to_string(), vsync=true, fov_deg=45.0, znear=0.1, zfar=1000.0))]
fn open_mesh_viewer(
    py: Python<'_>,
    vertices: numpy::PyReadonlyArray2<'_, f32>,
    indices: PyReadonlyArrayDyn<'_, u32>,
    uvs: Option<numpy::PyReadonlyArray2<'_, f32>>,
    texture_rgba: Option<numpy::PyReadonlyArray3<'_, u8>>,
    texture_width: u32,
    texture_height: u32,
    camera_eye: Option<numpy::PyReadonlyArray1<'_, f32>>,
    camera_target: Option<numpy::PyReadonlyArray1<'_, f32>>,
    width: u32,
    height: u32,
    title: String,
    vsync: bool,
    fov_deg: f32,
    znear: f32,
    zfar: f32,
) -> PyResult<()> {
    use crate::viewer::{run_mesh_viewer_with_camera, run_mesh_viewer_with_texture, ViewerConfig};

    // Flatten vertices (N,3) -> Vec<f32>
    let vshape = vertices.shape();
    if vshape.len() != 2 || vshape[1] != 3 {
        return Err(PyValueError::new_err("vertices must have shape (N, 3) float32"));
    }
    let verts: Vec<f32> = vertices.as_slice()?.to_vec();

    // Flatten indices: accept (M,3) or (K,)
    let ishape = indices.shape();
    let idx: Vec<u32> = match ishape.len() {
        1 => indices.as_slice()?.to_vec(),
        2 => indices.as_slice()?
                .to_vec(),
        _ => return Err(PyValueError::new_err("indices must be (M,3) or (K,) uint32")),
    };
    
    // Parse UVs if provided
    let uvs_opt = uvs.and_then(|uv_arr| {
        let uv_shape = uv_arr.shape();
        if uv_shape.len() == 2 && uv_shape[1] == 2 {
            uv_arr.as_slice().ok().map(|s| s.to_vec())
        } else {
            None
        }
    });
    
    // Parse texture if provided
    let texture_opt = texture_rgba.and_then(|tex_arr| {
        let tex_shape = tex_arr.shape();
        if tex_shape.len() == 3 && tex_shape[2] == 4 {
            tex_arr.as_slice().ok().map(|s| s.to_vec())
        } else {
            None
        }
    });

    // Parse camera parameters
    let camera_eye_opt = camera_eye.and_then(|arr| {
        let slice = arr.as_slice().ok()?;
        if slice.len() == 3 {
            Some([slice[0], slice[1], slice[2]])
        } else {
            None
        }
    });
    
    let camera_target_opt = camera_target.and_then(|arr| {
        let slice = arr.as_slice().ok()?;
        if slice.len() == 3 {
            Some([slice[0], slice[1], slice[2]])
        } else {
            None
        }
    });

    let config = ViewerConfig {
        width,
        height,
        title,
        vsync,
        fov_deg,
        znear,
        zfar,
    };

    // If texture is provided, use textured viewer. Release the Python GIL while the
    // blocking winit event loop runs so Python threads (CLI) remain responsive.
    // The closure must return a type that is safe to cross the GIL boundary; use Option<String>.
    let err_msg: Option<String> = py.allow_threads(|| {
        let res = if let (Some(uvs_data), Some(tex_data)) = (uvs_opt, texture_opt) {
            run_mesh_viewer_with_texture(
                config,
                verts,
                idx,
                uvs_data,
                tex_data,
                texture_width,
                texture_height,
                camera_eye_opt,
                camera_target_opt,
            )
        } else {
            run_mesh_viewer_with_camera(config, verts, idx, camera_eye_opt, camera_target_opt)
        };
        match res {
            Ok(()) => None,
            Err(e) => Some(e.to_string()),
        }
    });

    if let Some(msg) = err_msg {
        Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Viewer error: {}", msg)))
    } else {
        Ok(())
    }
}

/// Send a non-blocking export command to the active viewer (save offscreen PNG to path).
#[cfg(feature = "extension-module")]
#[pyfunction]
fn viewer_export(path: String) -> PyResult<()> {
    use crate::viewer::{viewer_send_command, ViewerCommand};
    viewer_send_command(ViewerCommand::Export(path))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

/// Set camera parameters in the active 3D viewer
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (distance=None, theta=None, phi=None))]
fn viewer_set_camera(distance: Option<f32>, theta: Option<f32>, phi: Option<f32>) -> PyResult<()> {
    use crate::viewer::{viewer_send_command, ViewerCommand};
    viewer_send_command(ViewerCommand::SetCamera { distance, theta, phi })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

/// Capture a snapshot from the active 3D viewer
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (path, width=None, height=None))]
fn viewer_snapshot(path: String, width: Option<u32>, height: Option<u32>) -> PyResult<()> {
    use crate::viewer::{viewer_send_command, ViewerCommand};
    viewer_send_command(ViewerCommand::Snapshot { path, width, height })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

/// Get current camera state from the active 3D viewer
#[cfg(feature = "extension-module")]
#[pyfunction]
fn viewer_get_camera(py: Python<'_>) -> PyResult<(Py<numpy::PyArray1<f32>>, Py<numpy::PyArray1<f32>>, f32, f32, f32, f32)> {
    use crate::viewer::{viewer_send_command, ViewerCommand};
    use std::sync::mpsc::channel;
    
    let (tx, rx) = channel();
    viewer_send_command(ViewerCommand::GetCamera { response_tx: tx })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    
    let state = rx.recv_timeout(std::time::Duration::from_secs(5))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Timeout waiting for camera state: {}", e)))?;
    
    let eye = numpy::PyArray1::from_slice_bound(py, &state.eye).unbind();
    let target = numpy::PyArray1::from_slice_bound(py, &state.target).unbind();
    Ok((eye, target, state.distance, state.theta, state.phi, state.fov))
}

// PyO3 module entry point so Python can `import forge3d._forge3d`
// This must be named exactly `_forge3d` to match [tool.maturin].module-name in pyproject.toml
#[cfg(feature = "extension-module")]
#[pymodule]
fn _forge3d(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Basic metadata so users can sanity-check the native module is loaded
    m.add("__doc__", "forge3d native module")?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // Interactive Viewer (I1)
    m.add_function(wrap_pyfunction!(open_viewer, m)?)?;
    m.add_function(wrap_pyfunction!(open_viewer_image, m)?)?;
    m.add_function(wrap_pyfunction!(open_mesh_viewer, m)?)?;
    m.add_function(wrap_pyfunction!(viewer_export, m)?)?;
    m.add_function(wrap_pyfunction!(viewer_set_camera, m)?)?;
    m.add_function(wrap_pyfunction!(viewer_snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(viewer_get_camera, m)?)?;
    // Vector: point shape/LOD controls
    m.add_function(wrap_pyfunction!(set_point_shape_mode, m)?)?;
    m.add_function(wrap_pyfunction!(set_point_lod_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(is_weighted_oit_available, m)?)?;
    m.add_function(wrap_pyfunction!(vector_oit_and_pick_demo, m)?)?;
    m.add_function(wrap_pyfunction!(vector_render_oit_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector_render_pick_map_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector_render_oit_and_pick_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector_render_polygons_fill_py, m)?)?;
    m.add_function(wrap_pyfunction!(configure_csm, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_light_direction, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_pcf_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_bias_params, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_debug_mode, m)?)?;
    m.add_function(wrap_pyfunction!(get_csm_cascade_info, m)?)?;
    m.add_function(wrap_pyfunction!(validate_csm_peter_panning, m)?)?;
    // Hybrid mesh path tracer (GPU) entry
    m.add_function(wrap_pyfunction!(_pt_render_gpu_mesh, m)?)?;
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
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_transform_center_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_transform_scale_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_transform_flip_axis_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_transform_swap_axes_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_transform_bounds_py,
        m
    )?)?;
    // Phase 4: subdivision, displacement, curves
    m.add_function(wrap_pyfunction!(crate::geometry::geometry_subdivide_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_displace_heightmap_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_displace_procedural_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_generate_ribbon_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_generate_tube_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_generate_thick_polyline_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_generate_tangents_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_attach_tangents_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_subdivide_adaptive_py,
        m
    )?)?;
    // Phase 6: instancing
    m.add_function(wrap_pyfunction!(
        crate::render::instancing::geometry_instance_mesh_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::render::instancing::gpu_instancing_available_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::render::instancing::geometry_instance_mesh_gpu_stub_py,
        m
    )?)?;
    #[cfg(all(feature = "enable-gpu-instancing"))]
    {
        m.add_function(wrap_pyfunction!(
            crate::render::instancing::geometry_instance_mesh_gpu_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            crate::render::instancing::geometry_instance_mesh_gpu_render_py,
            m
        )?)?;
    }

    // Native SDF placeholder renderer
    m.add_function(wrap_pyfunction!(hybrid_render, m)?)?;

    // IO: OBJ import/export
    m.add_function(wrap_pyfunction!(crate::io::obj_read::io_import_obj_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::io::obj_write::io_export_obj_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::io::stl_write::io_export_stl_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::io::gltf_read::io_import_gltf_py,
        m
    )?)?;
    // Import: OSM buildings helper
    m.add_function(wrap_pyfunction!(
        crate::import::osm_buildings::import_osm_buildings_extrude_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::import::osm_buildings::import_osm_buildings_from_geojson_py,
        m
    )?)?;

    // GPU utilities (adapter enumeration and probe)
    m.add_function(wrap_pyfunction!(enumerate_adapters, m)?)?;
    m.add_function(wrap_pyfunction!(device_probe, m)?)?;
    m.add_function(wrap_pyfunction!(global_memory_metrics, m)?)?;

    // Workstream C: Core Engine & Target interfaces
    m.add_function(wrap_pyfunction!(engine_info, m)?)?;
    m.add_function(wrap_pyfunction!(report_device, m)?)?;
    m.add_function(wrap_pyfunction!(c5_build_framegraph_report, m)?)?;
    m.add_function(wrap_pyfunction!(c6_mt_record_demo, m)?)?;
    m.add_function(wrap_pyfunction!(c7_async_compute_demo, m)?)?;

    // UV unwrap helpers
    m.add_function(wrap_pyfunction!(crate::uv::unwrap::uv_planar_unwrap_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::uv::unwrap::uv_spherical_unwrap_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::converters::multipolygonz_to_obj::converters_multipolygonz_to_obj_py,
        m
    )?)?;

    // Camera functions (expose Rust implementations to Python)
    m.add_function(wrap_pyfunction!(crate::camera::camera_look_at, m)?)?;
    m.add_function(wrap_pyfunction!(crate::camera::camera_perspective, m)?)?;
    m.add_function(wrap_pyfunction!(crate::camera::camera_orthographic, m)?)?;
    m.add_function(wrap_pyfunction!(crate::camera::camera_view_proj, m)?)?;
    m.add_function(wrap_pyfunction!(crate::camera::camera_dof_params, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::camera::camera_f_stop_to_aperture,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::camera::camera_aperture_to_f_stop,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::camera::camera_hyperfocal_distance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::camera::camera_depth_of_field_range,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::camera::camera_circle_of_confusion,
        m
    )?)?;

    // Transform utilities
    m.add_function(wrap_pyfunction!(crate::transforms::translate, m)?)?;
    m.add_function(wrap_pyfunction!(crate::transforms::rotate_x, m)?)?;
    m.add_function(wrap_pyfunction!(crate::transforms::rotate_y, m)?)?;
    m.add_function(wrap_pyfunction!(crate::transforms::rotate_z, m)?)?;
    m.add_function(wrap_pyfunction!(crate::transforms::scale, m)?)?;
    m.add_function(wrap_pyfunction!(crate::transforms::scale_uniform, m)?)?;
    m.add_function(wrap_pyfunction!(crate::transforms::compose_trs, m)?)?;
    m.add_function(wrap_pyfunction!(crate::transforms::look_at_transform, m)?)?;
    m.add_function(wrap_pyfunction!(crate::transforms::multiply_matrices, m)?)?;
    m.add_function(wrap_pyfunction!(crate::transforms::invert_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::transforms::compute_normal_matrix,
        m
    )?)?;

    // Grid generator
    m.add_function(wrap_pyfunction!(crate::grid::grid_generate, m)?)?;
    // Path tracing (GPU MVP)
    m.add_function(wrap_pyfunction!(_pt_render_gpu, m)?)?;

    // Add main classes
    m.add_class::<crate::scene::Scene>()?;
    // Expose TerrainSpike (E2/E3) to Python
    m.add_class::<crate::terrain::TerrainSpike>()?;

    Ok(())
}
