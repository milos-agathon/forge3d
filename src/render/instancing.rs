// src/render/instancing.rs
// Minimal CPU instancing utility: duplicates a base mesh for each transform.
// This serves as a portable fallback in environments where GPU instancing may not be available.

use glam::{Mat3, Mat4, Vec3};

use crate::geometry::MeshBuffers;

#[cfg(feature = "extension-module")]
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
#[cfg(all(feature = "extension-module", feature = "enable-gpu-instancing"))]
use numpy::{PyArray1, PyArrayMethods};
#[cfg(feature = "extension-module")]
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

fn apply_transform_position(m: &Mat4, p: [f32; 3]) -> [f32; 3] {
    let v = m.transform_point3(Vec3::new(p[0], p[1], p[2]));
    [v.x, v.y, v.z]
}

fn apply_transform_normal(m: &Mat3, n: [f32; 3]) -> [f32; 3] {
    let v = (*m * Vec3::new(n[0], n[1], n[2])).normalize_or_zero();
    [v.x, v.y, v.z]
}

/// Instance a base mesh by a list of 4x4 row-major transforms.
#[cfg(feature = "extension-module")]
#[pyfunction]
pub fn geometry_instance_mesh_py(
    py: Python<'_>,
    mesh: &Bound<'_, PyDict>,
    transforms: PyReadonlyArray2<'_, f32>, // (N,16) or (N,4x4)
) -> PyResult<PyObject> {
    if transforms.ndim() != 2 {
        return Err(PyValueError::new_err("transforms must be 2D (N,16)"));
    }
    let shape = transforms.shape();
    let n = shape[0];
    let cols = shape[1];
    if cols != 16 {
        return Err(PyValueError::new_err("transforms must have shape (N,16) row-major 4x4"));
    }

    let base = crate::geometry::mesh_from_python(mesh)?;
    let vcount = base.positions.len();
    let icount = base.indices.len();

    let mut out = MeshBuffers::with_capacity(vcount * n, icount * n);

    // Precompute normal transform per instance (inverse-transpose of upper-left 3x3)
    let arr = transforms.as_array();
    for i in 0..n {
        let r = arr.row(i);
        let m = Mat4::from_cols_array(&[
            r[0], r[4], r[8], r[12],
            r[1], r[5], r[9], r[13],
            r[2], r[6], r[10], r[14],
            r[3], r[7], r[11], r[15],
        ]);
        let normal_m = Mat3::from_mat4(m).inverse().transpose();
        let base_index_offset = out.positions.len() as u32;

        // Positions / normals / uvs / tangents
        for vi in 0..vcount {
            out.positions.push(apply_transform_position(&m, base.positions[vi]));
            if base.normals.len() == vcount {
                out.normals.push(apply_transform_normal(&normal_m, base.normals[vi]));
            }
            if base.uvs.len() == vcount {
                out.uvs.push(base.uvs[vi]);
            }
            if base.tangents.len() == vcount {
                out.tangents.push(base.tangents[vi]);
            }
        }

        // Indices
        for &idx in &base.indices {
            out.indices.push(idx + base_index_offset);
        }
    }

    crate::geometry::mesh_to_python(py, &out)
}

/// Return whether GPU instancing path is available in this build.
#[cfg(feature = "extension-module")]
#[pyfunction]
pub fn gpu_instancing_available_py() -> PyResult<bool> {
    // Report availability based on the feature flag.
    Ok(cfg!(feature = "enable-gpu-instancing"))
}

/// Stub for GPU instancing (draw indirect) path.
/// Returns False to indicate CPU fallback should be used.
#[cfg(feature = "extension-module")]
#[pyfunction]
pub fn geometry_instance_mesh_gpu_stub_py(
    _py: Python<'_>,
    _mesh: &Bound<'_, PyDict>,
    _transforms: PyReadonlyArray2<'_, f32>,
) -> PyResult<bool> {
    Ok(false)
}

/// Experimental GPU instancing entry point (feature-gated).
/// Currently delegates to CPU instancing until the hardware path is implemented.
#[cfg(all(feature = "extension-module", feature = "enable-gpu-instancing"))]
#[pyfunction]
pub fn geometry_instance_mesh_gpu_py(
    py: Python<'_>,
    mesh: &Bound<'_, PyDict>,
    transforms: PyReadonlyArray2<'_, f32>,
) -> PyResult<PyObject> {
    geometry_instance_mesh_py(py, mesh, transforms)
}

/// Render an instanced mesh to RGBA8 using GPU instancing (feature-gated).
#[cfg(all(feature = "extension-module", feature = "enable-gpu-instancing"))]
#[pyfunction]
pub fn geometry_instance_mesh_gpu_render_py(
    py: Python<'_>,
    width: u32,
    height: u32,
    mesh: &Bound<'_, PyDict>,
    transforms: PyReadonlyArray2<'_, f32>, // (N,16) row-major
) -> PyResult<Py<PyAny>> {
    if width == 0 || height == 0 {
        return Err(PyValueError::new_err("image dimensions must be positive"));
    }
    if transforms.ndim() != 2 || transforms.shape()[1] != 16 {
        return Err(PyValueError::new_err(
            "transforms must be 2D of shape (N,16) row-major 4x4",
        ));
    }

    // Base mesh
    let base = crate::geometry::mesh_from_python(mesh)?;
    let vcount = base.positions.len();
    if vcount == 0 || base.indices.is_empty() {
        return Err(PyValueError::new_err("base mesh is empty"));
    }
    // Build PN vertices (default normal if missing)
    #[cfg(feature = "enable-gpu-instancing")]
    use crate::render::mesh_instanced::VertexPN as Vpn;
    let vertices: Vec<Vpn> = (0..vcount)
        .map(|i| Vpn {
            position: base.positions[i],
            normal: if base.normals.len() == vcount {
                base.normals[i]
            } else {
                [0.0, 0.0, 1.0]
            },
        })
        .collect();
    let indices: Vec<u32> = base.indices.clone();

    // GPU context
    let g = crate::gpu::ctx();
    let color_format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let depth_format = Some(wgpu::TextureFormat::Depth32Float);

    // Renderer
    #[cfg(feature = "enable-gpu-instancing")]
    let mut renderer = crate::render::mesh_instanced::MeshInstancedRenderer::new(
        &g.device,
        color_format,
        depth_format,
    );
    renderer.set_mesh(&g.device, &g.queue, &vertices, &indices);

    // Camera
    let view = glam::Mat4::look_at_rh(glam::Vec3::new(3.0, 2.0, 3.0), glam::Vec3::ZERO, glam::Vec3::Y);
    let proj = crate::camera::perspective_wgpu(
        45.0f32.to_radians(),
        (width as f32 / height as f32).max(1e-3),
        0.1,
        100.0,
    );
    renderer.set_view_proj(view, proj);
    renderer.set_color([0.85, 0.85, 0.9, 1.0]);
    renderer.set_light([0.3, 0.7, 0.2], 1.2);

    // Upload transforms (row-major)
    let arr = transforms.as_array();
    let n = arr.shape()[0];
    let mut rows: Vec<[f32; 16]> = Vec::with_capacity(n);
    for i in 0..n {
        let r = arr.row(i);
        rows.push([
            r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13],
            r[14], r[15],
        ]);
    }
    renderer.upload_instances_from_rowmajor(&g.device, &g.queue, &rows);

    // Offscreen targets
    let color = g.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("instanced_color"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: color_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
    let depth = g.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("instanced_depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

    // Encode render pass
    let mut encoder = g
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("instanced_encoder"),
        });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("instanced_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        renderer.render(&mut pass, &g.queue, n as u32);
    }

    // Readback
    let row_bytes = (width * 4) as u32;
    let padded_bpr = crate::gpu::align_copy_bpr(row_bytes);
    let output_size = (padded_bpr * height) as u64;
    let readback = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("instanced_readback"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &color,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &readback,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(std::num::NonZeroU32::new(padded_bpr).unwrap().into()),
                rows_per_image: Some(std::num::NonZeroU32::new(height).unwrap().into()),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    g.queue.submit(Some(encoder.finish()));

    // Map and pack rows without padding
    g.device.poll(wgpu::Maintain::Wait);
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    g.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().map_err(|e| PyValueError::new_err(format!(
        "Failed to map readback buffer: {:?}",
        e
    )))?;
    let data = slice.get_mapped_range();
    let mut rgba = vec![0u8; (width as usize) * (height as usize) * 4];
    let row_bytes_usize = (row_bytes as usize).max(1);
    let padded_bpr_usize = (padded_bpr as usize).max(row_bytes_usize);
    for y in 0..(height as usize) {
        let src_offset = y * padded_bpr_usize;
        let dst_offset = y * (width as usize) * 4;
        rgba[dst_offset..dst_offset + row_bytes_usize]
            .copy_from_slice(&data[src_offset..src_offset + row_bytes_usize]);
    }
    drop(data);
    readback.unmap();

    // Return numpy array (H,W,4)
    let arr1 = PyArray1::<u8>::from_vec_bound(py, rgba);
    let out = arr1.reshape([height as usize, width as usize, 4])?;
    Ok(out.into_py(py))
}
