use wgpu::util::DeviceExt;

use super::math::{create_look_at_matrix, create_perspective_matrix, identity_matrix};
use super::request::PreparedBrdfTileRequest;

mod render_pass;
mod timestamps;

pub(super) use render_pass::encode_render_pass;
pub(super) use timestamps::TimestampResources;

const CAMERA_POS: [f32; 3] = [0.0, 0.0, 2.0];
const LOOK_AT: [f32; 3] = [0.0, 0.0, 0.0];
const UP: [f32; 3] = [0.0, 1.0, 0.0];
const PARAMS_MIN_SIZE: usize = 256;

pub(super) struct RenderTargets {
    pub(super) render_target: wgpu::Texture,
    render_view: wgpu::TextureView,
    _depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

pub(super) struct MeshBuffers {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
}

pub(super) struct UniformResources {
    pub(super) bind_group: wgpu::BindGroup,
    pub(super) debug_buffer: wgpu::Buffer,
}

impl RenderTargets {
    pub(super) fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let render_target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen.brdf_tile.render_target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen.brdf_tile.depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        Self {
            render_view: render_target.create_view(&wgpu::TextureViewDescriptor::default()),
            depth_view: depth_texture.create_view(&wgpu::TextureViewDescriptor::default()),
            render_target,
            _depth_texture: depth_texture,
        }
    }
}

impl MeshBuffers {
    pub(super) fn new(device: &wgpu::Device, sphere_sectors: u32, sphere_stacks: u32) -> Self {
        let (vertices, indices) =
            crate::offscreen::sphere::generate_uv_sphere(sphere_sectors, sphere_stacks, 1.0);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("offscreen.brdf_tile.vertex_buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("offscreen.brdf_tile.index_buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
        }
    }
}

impl UniformResources {
    pub(super) fn new(
        device: &wgpu::Device,
        pipeline: &crate::offscreen::pipeline::BrdfTilePipeline,
        request: &PreparedBrdfTileRequest,
    ) -> Self {
        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("offscreen.brdf_tile.uniforms"),
            contents: bytemuck::cast_slice(&[camera_uniforms(request.width, request.height)]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("offscreen.brdf_tile.params"),
            contents: &padded_params_bytes(request),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let shading_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("offscreen.brdf_tile.shading"),
            contents: bytemuck::cast_slice(&[shading_params(request)]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let debug_push_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("offscreen.brdf_tile.debug_push"),
            contents: bytemuck::bytes_of(&debug_push(request)),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let debug_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("offscreen.brdf_tile.debug_buffer"),
            contents: bytemuck::cast_slice(&[u32::MAX, 0, u32::MAX, 0]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let bind_group = pipeline.create_bind_group(
            device,
            &uniforms_buffer,
            &params_buffer,
            &shading_buffer,
            &debug_buffer,
            &debug_push_buffer,
        );

        Self {
            bind_group,
            debug_buffer,
        }
    }
}

fn camera_uniforms(width: u32, height: u32) -> crate::offscreen::pipeline::Uniforms {
    let view_matrix = create_look_at_matrix(CAMERA_POS, LOOK_AT, UP);
    let aspect = width as f32 / height as f32;
    let projection_matrix = create_perspective_matrix(60.0_f32.to_radians(), aspect, 0.1, 100.0);

    crate::offscreen::pipeline::Uniforms {
        model_matrix: identity_matrix(),
        view_matrix,
        projection_matrix,
    }
}

fn padded_params_bytes(request: &PreparedBrdfTileRequest) -> Vec<u8> {
    let params = brdf_tile_params(request);
    let params_bytes = bytemuck::bytes_of(&params);
    let target_len = params_bytes.len().max(PARAMS_MIN_SIZE);
    let mut padded = vec![0u8; target_len];
    padded[..params_bytes.len()].copy_from_slice(params_bytes);
    padded
}

fn brdf_tile_params(
    request: &PreparedBrdfTileRequest,
) -> crate::offscreen::pipeline::BrdfTileParams {
    let f0 = compute_f0(request.base_color, request.metallic);

    crate::offscreen::pipeline::BrdfTileParams {
        light_dir: request.light_dir,
        _pad0: 0.0,
        light_color: [1.0, 1.0, 1.0],
        light_intensity: request.light_intensity,
        camera_pos: CAMERA_POS,
        _pad1: 0.0,
        base_color: request.base_color,
        metallic: request.metallic,
        roughness: request.roughness,
        ndf_only: flag(request.ndf_only),
        g_only: flag(request.g_only),
        dfg_only: flag(request.dfg_only),
        spec_only: flag(request.spec_only),
        roughness_visualize: flag(request.roughness_visualize),
        f0,
        _pad_f0: 0.0,
        clearcoat: request.clearcoat.clamp(0.0, 1.0),
        clearcoat_roughness: request.clearcoat_roughness.clamp(0.0, 1.0),
        sheen: request.sheen.clamp(0.0, 1.0),
        sheen_tint: request.sheen_tint.clamp(0.0, 1.0),
        specular_tint: request.specular_tint.clamp(0.0, 1.0),
        debug_lambert_only: flag(request.debug_lambert_only),
        debug_diffuse_only: flag(request.debug_diffuse_only),
        debug_energy: flag(request.debug_energy),
        debug_d: flag(request.debug_d),
        debug_g_dbg: 0,
        debug_spec_no_nl: flag(request.debug_spec_no_nl),
        debug_angle_sweep: flag(request.debug_angle_sweep),
        debug_angle_component: request.debug_angle_component,
        debug_no_srgb: flag(request.debug_no_srgb),
        debug_kind: request.debug_kind,
        _pad_debug_kind: [0, 0, 0],
        _pad2: 0,
        _pad3: 0,
        _pad4: 0,
        _pad5: 0,
        _pad6: 0,
        _pad7: 0,
    }
}

fn shading_params(
    request: &PreparedBrdfTileRequest,
) -> crate::offscreen::pipeline::ShadingParamsGPU {
    crate::offscreen::pipeline::ShadingParamsGPU {
        brdf: request.model_u32,
        metallic: request.metallic,
        roughness: request.roughness,
        sheen: 0.0,
        clearcoat: 0.0,
        subsurface: 0.0,
        anisotropy: 0.0,
        exposure: request.exposure,
        output_mode: request.output_mode,
        _pad_out0: 0,
        _pad_out1: 0,
        _pad_out2: 0,
    }
}

fn debug_push(request: &PreparedBrdfTileRequest) -> crate::offscreen::pipeline::DebugPush {
    crate::offscreen::pipeline::DebugPush {
        mode: request.wi3_mode,
        roughness: request.wi3_roughness,
        _pad: [0.0, 0.0],
    }
}

fn compute_f0(base_color: [f32; 3], metallic: f32) -> [f32; 3] {
    let dielectric_f0 = [0.04f32, 0.04, 0.04];
    [
        dielectric_f0[0] * (1.0 - metallic) + base_color[0] * metallic,
        dielectric_f0[1] * (1.0 - metallic) + base_color[1] * metallic,
        dielectric_f0[2] * (1.0 - metallic) + base_color[2] * metallic,
    ]
}

fn flag(value: bool) -> u32 {
    if value {
        1
    } else {
        0
    }
}
