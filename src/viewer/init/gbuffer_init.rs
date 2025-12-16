// src/viewer/init/gbuffer_init.rs
// GBuffer pipeline initialization for the Viewer

use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupLayout, Buffer, Device, RenderPipeline, Sampler, Texture, TextureView,
};

use crate::core::screen_space_effects::ScreenSpaceEffectsManager;

/// Resources created during GBuffer initialization
pub struct GBufferResources {
    pub geom_bind_group_layout: Option<BindGroupLayout>,
    pub geom_pipeline: Option<RenderPipeline>,
    pub geom_camera_buffer: Option<Buffer>,
    pub geom_bind_group: Option<BindGroup>,
    pub geom_vb: Option<Buffer>,
    pub z_texture: Option<Texture>,
    pub z_view: Option<TextureView>,
    pub albedo_texture: Option<Texture>,
    pub albedo_view: Option<TextureView>,
    pub albedo_sampler: Option<Sampler>,
    pub comp_bind_group_layout: Option<BindGroupLayout>,
    pub comp_pipeline: Option<RenderPipeline>,
}

impl Default for GBufferResources {
    fn default() -> Self {
        Self {
            geom_bind_group_layout: None,
            geom_pipeline: None,
            geom_camera_buffer: None,
            geom_bind_group: None,
            geom_vb: None,
            z_texture: None,
            z_view: None,
            albedo_texture: None,
            albedo_view: None,
            albedo_sampler: None,
            comp_bind_group_layout: None,
            comp_pipeline: None,
        }
    }
}

/// GBuffer geometry shader source
pub const GBUFFER_GEOM_SHADER: &str = r#"
struct Camera {
    view : mat4x4<f32>,
    proj : mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> uCam : Camera;
@group(0) @binding(1) var tAlbedo : texture_2d<f32>;
@group(0) @binding(2) var sAlbedo : sampler;

struct VSIn {
    @location(0) pos : vec3<f32>,
    @location(1) nrm : vec3<f32>,
    @location(2) uv  : vec2<f32>,
    @location(3) rough_metal : vec2<f32>,
};
struct VSOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) v_nrm_vs : vec3<f32>,
    @location(1) v_depth_vs : f32,
    @location(2) v_uv : vec2<f32>,
    @location(3) v_rough_metal : vec2<f32>,
};

@vertex
fn vs_main(inp: VSIn) -> VSOut {
    var out: VSOut;
    let pos_ws = vec4<f32>(inp.pos, 1.0);
    let pos_vs = uCam.view * pos_ws;
    out.pos = uCam.proj * pos_vs;
    let nrm_vs = (uCam.view * vec4<f32>(inp.nrm, 0.0)).xyz;
    out.v_nrm_vs = normalize(nrm_vs);
    out.v_depth_vs = -pos_vs.z;
    out.v_uv = inp.uv;
    out.v_rough_metal = inp.rough_metal;
    return out;
}

struct FSOut {
    @location(0) normal_rgba : vec4<f32>,
    @location(1) albedo_rgba : vec4<f32>,
    @location(2) depth_r : f32,
};

@fragment
fn fs_main(inp: VSOut) -> FSOut {
    var out: FSOut;
    let n = normalize(inp.v_nrm_vs);
    let enc = n * 0.5 + vec3<f32>(0.5);
    out.normal_rgba = vec4<f32>(enc, clamp(inp.v_rough_metal.x, 0.0, 1.0));
    let color = textureSample(tAlbedo, sAlbedo, inp.v_uv);
    out.albedo_rgba = vec4<f32>(color.rgb, clamp(inp.v_rough_metal.y, 0.0, 1.0));
    out.depth_r = inp.v_depth_vs;
    return out;
}
"#;

/// Create GBuffer resources if GI manager is available
pub fn create_gbuffer_resources(
    device: &Arc<Device>,
    gi: Option<&ScreenSpaceEffectsManager>,
    width: u32,
    height: u32,
    surface_format: wgpu::TextureFormat,
) -> GBufferResources {
    let gi_ref = match gi {
        Some(g) => g,
        None => return GBufferResources::default(),
    };

    // Z-buffer for rasterization
    let z_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viewer.gbuf.z"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let z_view = z_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Camera uniform
    let geom_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viewer.gbuf.cam"),
        size: (std::mem::size_of::<[[f32; 4]; 4]>() * 2) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Bind group layout: camera uniform + albedo texture + sampler
    let geom_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewer.gbuf.geom.bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    // Shader for geometry GBuffer write
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("viewer.gbuf.geom.shader"),
        source: wgpu::ShaderSource::Wgsl(GBUFFER_GEOM_SHADER.into()),
    });

    // Pipeline layout
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewer.gbuf.geom.pl"),
        bind_group_layouts: &[&geom_bgl],
        push_constant_ranges: &[],
    });

    let gb = gi_ref.gbuffer();
    let gb_cfg = gb.config();
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("viewer.gbuf.geom.pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 40,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 12,
                        shader_location: 1,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 24,
                        shader_location: 2,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 32,
                        shader_location: 3,
                    },
                ],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: gb_cfg.normal_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: gb_cfg.material_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: gb_cfg.depth_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    // Composite bind group layout and pipeline - delegated to composite_init
    let comp_bgl = super::composite_init::create_composite_bind_group_layout(device);
    let comp_pipeline =
        super::composite_init::create_composite_pipeline(device, &comp_bgl, surface_format);

    GBufferResources {
        geom_bind_group_layout: Some(geom_bgl),
        geom_pipeline: Some(pipeline),
        geom_camera_buffer: Some(geom_camera_buffer),
        geom_bind_group: None,
        geom_vb: None,
        z_texture: Some(z_texture),
        z_view: Some(z_view),
        albedo_texture: None,
        albedo_view: None,
        albedo_sampler: None,
        comp_bind_group_layout: Some(comp_bgl),
        comp_pipeline: Some(comp_pipeline),
    }
}
