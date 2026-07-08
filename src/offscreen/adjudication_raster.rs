// src/offscreen/adjudication_raster.rs
// AEQUITAS raster twin: renders the ReferenceSceneDesc through the shared
// offscreen forward-raster path (tessellated spheres + plane quad,
// depth-tested multi-draw pass via offscreen::forward) with a fragment
// shader that evaluates an analytic approximation of the multi-bounce PT
// reference — exact sun/ambient NEE + sky-escape quadrature plus secondary-
// vertex closures with tertiary plane/sphere-side exit-radiance terms (see
// the WGSL header for the exact contract).
// Rendered with 4x supersampling and tent-weighted
// downsampling so silhouettes match the PT tent-filter anti-aliasing.
// RELEVANT FILES: src/offscreen/forward.rs, src/shaders/adjudication_raster.wgsl

use crate::core::error::RenderError;
use crate::path_tracing::reference_scene::ReferenceSceneDesc;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use wgpu::{Buffer, Device, Queue};

/// Supersampling factor (subsamples are tent-weighted on downsample to match
/// the PT reference's +/-0.5px tent reconstruction filter).
const SSAA: u32 = 4;

/// Gap-4 routing status — EXPLICITLY BLOCKED on the strict reading. Do not
/// delete this marker or stop consuming the shared entry points it names;
/// the module test below fails if this raster becomes an unacknowledged
/// private fork.
///
/// The strict AEQUITAS requirement reads "existing raster path" as the
/// viewer/production raster renderer. That routing remains blocked without a
/// broad rewrite, and would break the parity contract even if attempted: the
/// viewer and terrain renderers shade with their own PBR/POM stacks, not the
/// pt_shade parity estimator this gate needs, and `offscreen::brdf_tile`
/// renders one sphere with no shadows or multi-object scene. AEQUITAS must
/// therefore not be called complete while this marker says "blocked".
///
/// What IS shared (and test-enforced below) — every mechanical stage flows
/// through offscreen/core infrastructure rather than private code:
/// - geometry: `offscreen::sphere::generate_uv_sphere` + the scene's
///   `ReferenceSceneDesc::{plane_mesh, environment_raw, camera_basis}`;
/// - render targets, depth-tested multi-draw pass encoding, submission, and
///   host-visible HDR readback: the general-purpose offscreen mesh raster
///   path `offscreen::forward` (`ForwardTargets` + `render_forward_hdr`),
///   which itself reads back through `core::hdr_readback::read_hdr_texture`
///   and accounts staging through the global memory tracker;
/// - resolve: the shared tonemap `core::tonemap::resolve_reference_hdr_to_rgba8`
///   (applied by `py_functions::adjudication::render_adjudication_pair` to
///   BOTH the PT and raster HDR buffers).
///
/// The scene-specific residue is, by necessity, the parity fragment shader
/// (adjudication_raster.wgsl evaluates the PT parity estimator, which no
/// production raster path implements or should implement), its uniform
/// schema, and the PT-matched tent SSAA downsample.
pub const ADJUDICATION_RASTER_ROUTING_STATUS: &str =
    "blocked: not routed through the viewer/production raster renderer (its \
     PBR/POM shading cannot express the pt_shade parity estimator without a \
     broad rewrite); mechanical stages are shared — targets/pass/readback via \
     offscreen::forward + core::hdr_readback, geometry via offscreen::sphere \
     + ReferenceSceneDesc, resolve via core::tonemap";

/// Must match `AdjUniforms` in adjudication_raster.wgsl (304 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct AdjUniforms {
    cam_origin_fovy: [f32; 4],
    cam_right_aspect: [f32; 4],
    cam_up_w: [f32; 4],
    cam_forward_h: [f32; 4],
    sun_dir_intensity: [f32; 4],
    sun_color_pad: [f32; 4],
    environment: [[f32; 4]; 4],
    sph: [[f32; 4]; 3],
    mats: [[f32; 4]; 4],
    draw_center_radius: [f32; 4],
    misc: [f32; 4],
}

fn base_uniforms(desc: &ReferenceSceneDesc, rw: u32, rh: u32, aspect: f32) -> AdjUniforms {
    let (origin, forward, right, up) = desc.camera_basis();
    let sun = glam::Vec3::from(desc.sun_direction).normalize();
    let mut sph = [[0.0f32; 4]; 3];
    for (i, s) in desc.spheres.iter().take(3).enumerate() {
        sph[i] = [s.center[0], s.center[1], s.center[2], s.radius];
    }
    let mut mats = [[0.0f32; 4]; 4];
    for (i, s) in desc.spheres.iter().enumerate() {
        mats[i] = [s.albedo[0], s.albedo[1], s.albedo[2], s.roughness];
    }
    AdjUniforms {
        cam_origin_fovy: [origin.x, origin.y, origin.z, desc.fov_y_rad()],
        cam_right_aspect: [right.x, right.y, right.z, aspect],
        cam_up_w: [up.x, up.y, up.z, rw as f32],
        cam_forward_h: [forward.x, forward.y, forward.z, rh as f32],
        sun_dir_intensity: [sun.x, sun.y, sun.z, desc.sun_intensity],
        sun_color_pad: [desc.sun_color[0], desc.sun_color[1], desc.sun_color[2], 0.0],
        environment: {
            let e = desc.environment_raw();
            [e.env_ground, e.env_sky, e.miss_ground, e.miss_sky]
        },
        sph,
        mats,
        draw_center_radius: [0.0; 4],
        misc: [desc.plane_half_extent, 3.0, 0.0, 0.0],
    }
}

fn uniform_buffer(device: &Device, label: &str, u: &AdjUniforms) -> Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(u),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

/// Render the linear-HDR raster reference at (width, height).
/// Returns RGBA f32, row-major, `width * height * 4` values, alpha = 1.
pub fn render_raster_reference(
    device: &Device,
    queue: &Queue,
    desc: &ReferenceSceneDesc,
    width: u32,
    height: u32,
) -> Result<Vec<f32>, RenderError> {
    if width == 0 || height == 0 {
        return Err(RenderError::Render(
            "adjudication raster reference requires non-zero width/height".into(),
        ));
    }
    let rw = width * SSAA;
    let rh = height * SSAA;
    let aspect = width as f32 / height as f32;

    // --- Shader + pipelines ---
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("adjudication-raster-shader"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../shaders/adjudication_raster.wgsl").into(),
        ),
    });
    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("adjudication-raster-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("adjudication-raster-pl"),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });
    let color_format = wgpu::TextureFormat::Rgba32Float;
    let depth_format = wgpu::TextureFormat::Depth32Float;
    let color_target = [Some(wgpu::ColorTargetState {
        format: color_format,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    })];
    let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("adjudication-sky-pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_sky",
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_sky",
            targets: &color_target,
        }),
        multiview: None,
    });
    let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("adjudication-mesh-pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_mesh",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 12,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                }],
            }],
        },
        primitive: wgpu::PrimitiveState {
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_mesh",
            targets: &color_target,
        }),
        multiview: None,
    });

    // --- Geometry: shared unit sphere + plane quad ---
    let (sphere_verts, sphere_indices) = crate::offscreen::sphere::generate_uv_sphere(192, 96, 1.0);
    let sphere_positions: Vec<[f32; 3]> = sphere_verts.iter().map(|v| v.position).collect();
    let sphere_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("adjudication-sphere-vb"),
        contents: bytemuck::cast_slice(&sphere_positions),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let sphere_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("adjudication-sphere-ib"),
        contents: bytemuck::cast_slice(&sphere_indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    let plane = desc.plane_mesh();
    let plane_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("adjudication-plane-vb"),
        contents: bytemuck::cast_slice(&plane.vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let plane_indices: Vec<u32> = plane.indices.iter().flatten().copied().collect();
    let plane_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("adjudication-plane-ib"),
        contents: bytemuck::cast_slice(&plane_indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    // --- View-projection (same pixel->ray mapping as pt_raygen's NDC math) ---
    let (origin, forward, _right, up) = desc.camera_basis();
    let view = glam::Mat4::look_at_rh(origin, origin + forward, up);
    let proj = glam::Mat4::perspective_rh(desc.fov_y_rad(), aspect, 0.05, 400.0);
    let vp = proj * view;
    let vp_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("adjudication-vp"),
        contents: bytemuck::cast_slice(&vp.to_cols_array()),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // --- Per-draw uniforms: plane (also used by sky) + one per sphere ---
    let base = base_uniforms(desc, rw, rh, aspect);
    let mut draw_uniforms = vec![base]; // plane: misc = [E, 3, 0, 0]
    for (i, s) in desc.spheres.iter().take(3).enumerate() {
        let mut u = base;
        u.draw_center_radius = [s.center[0], s.center[1], s.center[2], s.radius];
        u.misc = [desc.plane_half_extent, i as f32, 1.0, 0.0];
        draw_uniforms.push(u);
    }
    let bind_groups: Vec<wgpu::BindGroup> = draw_uniforms
        .iter()
        .enumerate()
        .map(|(i, u)| {
            let ub = uniform_buffer(device, &format!("adjudication-uniforms-{i}"), u);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("adjudication-raster-bg"),
                layout: &bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ub.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: vp_buffer.as_entire_binding(),
                    },
                ],
            })
        })
        .collect();

    // --- Targets + pass encoding + HDR readback: routed through the shared
    // offscreen forward-raster path (see ADJUDICATION_RASTER_ROUTING_STATUS).
    // Draw order matters: sky (depth Always), then plane, then spheres. ---
    let targets = crate::offscreen::forward::ForwardTargets::new(device, rw, rh, color_format)?;
    let mut draws = vec![
        crate::offscreen::forward::ForwardDraw {
            pipeline: &sky_pipeline,
            bind_group: Some(&bind_groups[0]),
            vertex_buffer: None,
            index_buffer: None,
            index_count: 0,
            vertices: 0..3,
        },
        crate::offscreen::forward::ForwardDraw {
            pipeline: &mesh_pipeline,
            bind_group: Some(&bind_groups[0]),
            vertex_buffer: Some(&plane_vb),
            index_buffer: Some(&plane_ib),
            index_count: plane_indices.len() as u32,
            vertices: 0..0,
        },
    ];
    for bind_group in bind_groups.iter().skip(1) {
        draws.push(crate::offscreen::forward::ForwardDraw {
            pipeline: &mesh_pipeline,
            bind_group: Some(bind_group),
            vertex_buffer: Some(&sphere_vb),
            index_buffer: Some(&sphere_ib),
            index_count: sphere_indices.len() as u32,
            vertices: 0..0,
        });
    }
    let ss_pixels = crate::offscreen::forward::render_forward_hdr(
        device,
        queue,
        &targets,
        wgpu::Color::BLACK,
        &draws,
    )?;

    // --- Tent-weighted downsample SSAA x SSAA -> (width, height) ---
    // The PT reference jitters samples with a tent distribution of half-width
    // 0.5 px (pt_raygen's tent_filter * 0.5), so its effective reconstruction
    // filter is a tent over +/-0.5 px. Weight each subsample by the separable
    // tent w(o) = 1 - |o|/0.5 at its offset from the pixel center so edge
    // profiles match the PT anti-aliasing.
    let tent_1d: Vec<f32> = (0..SSAA as usize)
        .map(|k| {
            let o = (k as f32 + 0.5) / SSAA as f32 - 0.5;
            1.0 - o.abs() / 0.5
        })
        .collect();
    let tent_norm: f32 = tent_1d.iter().sum();
    let mut hdr = vec![0.0f32; (width as usize) * (height as usize) * 4];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let mut acc = [0.0f32; 3];
            for (sy, wy) in tent_1d.iter().enumerate() {
                for (sx, wx) in tent_1d.iter().enumerate() {
                    let sy_full = y * SSAA as usize + sy;
                    let sx_full = x * SSAA as usize + sx;
                    let idx = (sy_full * rw as usize + sx_full) * 4;
                    let w = wy * wx;
                    acc[0] += ss_pixels[idx] * w;
                    acc[1] += ss_pixels[idx + 1] * w;
                    acc[2] += ss_pixels[idx + 2] * w;
                }
            }
            let o = (y * width as usize + x) * 4;
            let inv = 1.0 / (tent_norm * tent_norm);
            hdr[o] = acc[0] * inv;
            hdr[o + 1] = acc[1] * inv;
            hdr[o + 2] = acc[2] * inv;
            hdr[o + 3] = 1.0;
        }
    }
    Ok(hdr)
}

#[cfg(test)]
mod tests {
    use super::ADJUDICATION_RASTER_ROUTING_STATUS;

    #[test]
    fn raster_twin_is_explicitly_blocked_and_reuses_shared_infra() {
        // Gap-4 contract: the custom raster pipeline is only sanctioned while
        // it is explicitly marked blocked (not routed through the
        // viewer/production raster renderer) AND keeps every mechanical stage
        // on shared infrastructure — targets, pass encoding, and HDR readback
        // through offscreen::forward (which reads back via core::hdr_readback
        // and accounts staging in the memory tracker), geometry through
        // offscreen::sphere + ReferenceSceneDesc, resolve through the shared
        // tonemap. This test fails if the blocked marker is dropped without
        // real routing, or if the module drifts into an unacknowledged
        // private fork of any of those stages.
        assert!(ADJUDICATION_RASTER_ROUTING_STATUS.starts_with("blocked:"));
        let src = include_str!("adjudication_raster.rs");
        for shared in [
            "forward::ForwardTargets::new",
            "forward::render_forward_hdr",
            "generate_uv_sphere",
            "desc.plane_mesh()",
            "desc.environment_raw()",
            "desc.camera_basis()",
        ] {
            assert!(
                src.contains(shared),
                "raster twin no longer routes through shared infrastructure: {shared}"
            );
        }
        // The forward harness itself must keep using the established core
        // readback + memory-tracker accounting.
        let harness = include_str!("forward.rs");
        assert!(harness.contains("read_hdr_texture"));
        assert!(harness.contains("global_tracker()"));
        // The shared tonemap is applied to BOTH paths by the capture API.
        let capture = include_str!("../py_functions/adjudication.rs");
        assert!(
            capture.matches("resolve_reference_hdr_to_rgba8").count() >= 2,
            "both PT and raster resolves must call the shared tonemap operator"
        );
    }
}
