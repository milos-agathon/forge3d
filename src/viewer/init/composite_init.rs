// src/viewer/init/composite_init.rs
// Composite pipeline initialization for the Viewer

use std::sync::Arc;
use wgpu::{BindGroupLayout, Device, RenderPipeline};

/// Composite shader source
pub const COMPOSITE_SHADER: &str = r#"
struct CompParams {
    far_plane : f32,
    _pad0 : f32,
    _pad1 : f32,
    _pad2 : f32,
};

@group(0) @binding(0) var sky_tex : texture_2d<f32>;
@group(0) @binding(1) var depth_tex : texture_2d<f32>;
@group(0) @binding(2) var fog_tex : texture_2d<f32>;
@group(0) @binding(3) var<uniform> params : CompParams;
@group(0) @binding(4) var color_tex : texture_2d<f32>;

struct VSOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
    var out: VSOut;
    let x = f32(vi & 1u) * 2.0;
    let y = f32((vi >> 1u) & 1u) * 2.0;
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(inp: VSOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(color_tex));
    let px = vec2<i32>(inp.uv * dims);
    let depth = textureLoad(depth_tex, px, 0).r;
    let sky = textureLoad(sky_tex, px, 0).rgb;
    let fog = textureLoad(fog_tex, px, 0);
    let color = textureLoad(color_tex, px, 0).rgb;

    let is_sky = depth >= params.far_plane * 0.999;
    var base = select(color, sky, is_sky);
    base = base * (1.0 - fog.a) + fog.rgb;
    return vec4<f32>(base, 1.0);
}
"#;

/// Create composite bind group layout
pub fn create_composite_bind_group_layout(device: &Arc<Device>) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewer.comp.bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

/// Create composite render pipeline
pub fn create_composite_pipeline(
    device: &Arc<Device>,
    comp_bgl: &BindGroupLayout,
    surface_format: wgpu::TextureFormat,
) -> RenderPipeline {
    let comp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("viewer.comp.shader"),
        source: wgpu::ShaderSource::Wgsl(COMPOSITE_SHADER.into()),
    });

    let comp_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewer.comp.pl"),
        bind_group_layouts: &[comp_bgl],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("viewer.comp.pipeline"),
        layout: Some(&comp_pl_layout),
        vertex: wgpu::VertexState {
            module: &comp_shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &comp_shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
}
